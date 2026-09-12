"""Frozen TRAIN-only input-gain calibration; no fitting or runtime mutation.

Physical windows are recorded body/sensory backgrounds, cold-started for 700 ms.
They are not a faithful replay of the whole body's neural history. The matched
hunger sweep changes only artificial semantic_hunger; native hunger/taste inputs
remain recorded, which isolates that artificial channel, not organism physiology.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fly_arena.brain import Brain, BrainConfig
from fly_semantic.mapping import digest, file_digest, load_mapping
from fly_semantic.runtime import load_calibration, make_features
from semantic_tools.physical_collection import load_plan


GAINS = (4.0, 4.5, 5.0, 6.0)
SCENES = ("hungry_away", "sated_away", "meal_fast", "limited_meal", "sated_food")
MATCHED_LEVELS = (.45, .6, .75, .85)
LEGACY_LEVELS = (0., .2, .85, 1.)
PROBE_MS, ANALYSIS_MS = 700, 300


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_train_windows(source, study, group=1201):
    if group not in study["seed_groups"]["train"]:
        raise ValueError("Gain calibration can open declared TRAIN groups only")
    windows = {}
    for scene in SCENES:
        candidates = [e for e in study["episodes"] if e["split"] == "train"
                      and e["seed_group"] == group and e["scene"] == scene]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one declared TRAIN episode per scene")
        ep = candidates[0]
        path = Path(source) / "episodes" / "connected" / (ep["episode_id"] + ".npz")
        sha = file_digest(path)
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"].item()))
            if (meta["episode"] != ep or meta["plan_digest"] != study["digest"]
                    or meta["condition"] != "connected" or meta["body_physics"] is not True
                    or meta["source_hashes"] != study["source_hashes"]
                    or meta["feature_digest"] != study["feature_digest"]
                    or meta["calibration_digest"] != study["calibration_digest"]):
                raise ValueError(f"TRAIN source provenance mismatch: {path}")
            specs = meta["stimulus_channels"]
            names = [s["name"] for s in specs]
            if len(names) != len(set(names)) or names.count("semantic_hunger") != 1:
                raise ValueError("Recorded stimulus channels must be unique and include hunger")
            ticks = ep["duration_ms"] // 10
            lum = a["replay_luminance"]
            if lum.shape[0] != ticks or not np.isfinite(lum).all():
                raise ValueError("Incomplete/nonfinite TRAIN retinal record")
            values = []
            for i, spec in enumerate(specs):
                value = a[f"replay_values_{i}"]
                if value.shape != (ticks, len(spec["indices"])) or not np.isfinite(value).all():
                    raise ValueError("Incomplete/nonfinite TRAIN current record")
                values.append(value[-PROBE_MS // 10:].copy())
            windows[scene] = {"episode": ep, "path": path.as_posix(), "sha256": sha,
                "metadata": meta, "specs": specs, "values": values,
                "luminance": lum[-PROBE_MS // 10:].copy()}
    return windows


def pair_difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    delta = right - left
    # Per-cell normalization; silent pairs receive zero, 1 Hz denominator floor.
    normalized = np.abs(delta) / np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.)
    active = (left >= 1.) | (right >= 1.)
    return {"cells": int(left.size), "changed_at_least_1hz": int((np.abs(delta) >= 1.).sum()),
        "rms_difference_hz": float(np.sqrt(np.mean(delta * delta))),
        "maximum_absolute_difference_hz": float(np.max(np.abs(delta))),
        "active_union_cells": int(active.sum()),
        "mean_normalized_abs_difference_all_cells": float(normalized.mean()),
        "mean_normalized_abs_difference_active_union": float(normalized[active].mean()) if active.any() else 0.,
        "normalization": "abs(right-left)/max(abs(left),abs(right),1Hz), separately per cell"}


def replay_probe(brain, initial, features, input_indices, luminance, specs, values,
                 gain, old_gain, override=None):
    brain.set_state(initial)
    features.reset()
    counts_sum = np.zeros(len(brain.ids), np.int64)
    finite = True
    direct_tick_rates = []
    hunger_inputs = []
    for tick, light in enumerate(luminance):
        channels = {}
        for spec, value in zip(specs, values):
            tick_value = value[tick]
            if spec["name"] == "semantic_hunger":
                # Recover only the recorded artificial drive; no teacher labels.
                tick_value = tick_value * np.float32(gain / old_gain) if override is None else np.full_like(tick_value, gain * override)
                hunger_inputs.extend((np.asarray(tick_value) / gain).tolist())
            channels[spec["name"]] = (np.asarray(spec["indices"], np.int32), tick_value)
        _, counts = brain.step(light, sensory_currents=channels)
        features.update(counts, 10)
        finite = bool(finite and all(np.isfinite(getattr(brain, name)).all()
                      for name in ("voltage", "current", "queue")))
        direct_tick_rates.append(counts[input_indices].astype(float) * 100.)
        if tick >= (PROBE_MS - ANALYSIS_MS) // 10:
            counts_sum += counts
    hz = counts_sum / (ANALYSIS_MS / 1000)
    input_hz, feature_hz = hz[input_indices], hz[features.indices]
    result = {"finite_every_10ms": finite, "port_max_hz": float(input_hz.max()),
        "port_mean_hz": float(input_hz.mean()), "port_active_cells_at_least_1hz": int((input_hz >= 1).sum()),
        "port_saturated_fraction": float((input_hz >= 300).mean()),
        "global_max_hz": float(hz.max()), "global_mean_hz": float(hz.mean()),
        "global_saturated_fraction": float((hz >= 300).mean()),
        "feature_mean_hz": float(feature_hz.mean()),
        "feature_active_cells_at_least_1hz": int((feature_hz >= 1).sum()),
        "input_hunger_min": float(min(hunger_inputs)), "input_hunger_max": float(max(hunger_inputs))}
    result["legacy_rate_guard_pass"] = bool(finite and result["port_max_hz"] < 200
                                            and result["global_saturated_fraction"] < .01)
    result["v3_rate_guard_pass"] = bool(finite and result["port_saturated_fraction"] < .01
                                       and result["global_saturated_fraction"] < .01)
    return result, {"input_hz": input_hz, "feature_hz": feature_hz,
                    "final_traces_hz": features.values(), "input_tick_hz": np.asarray(direct_tick_rates)}


def run(source, output, graph=Path("data/graph")):
    source, output, graph = Path(source), Path(output), Path(graph)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "gain-probe-plan.json").exists():
        raise FileExistsError("Gain probe already frozen; choose a fresh output directory")
    study = load_plan(source)
    windows = load_train_windows(source, study)
    cal_dir = Path(study["calibration_dir"])
    mapping = load_mapping(cal_dir / "mapping.json", graph)
    calibration = load_calibration(cal_dir / "calibration.json", mapping)
    if (calibration["digest"] != study["calibration_digest"]
            or mapping["digest"] != study["mapping_digest"]):
        raise ValueError("Historical calibration/mapping changed")
    features = make_features(mapping, calibration)
    input_indices = np.array([v["index"] for v in mapping["ports"]["hunger"]], np.int32)
    for window in windows.values():
        spec = next(s for s in window["specs"] if s["name"] == "semantic_hunger")
        if not np.array_equal(input_indices, spec["indices"]):
            raise ValueError("Recorded hunger addresses differ from frozen mapping")
    protected = {str(source / "plan.json"): file_digest(source / "plan.json"),
        **{w["path"]: w["sha256"] for w in windows.values()},
        **{str(p): file_digest(p) for p in (cal_dir / "mapping.json", cal_dir / "calibration.json")}}
    sources = {str(p): file_digest(p) for p in (Path(__file__), Path("fly_arena/brain.py"),
        Path("fly_semantic/features.py"), Path("fly_semantic/mapping.py"), Path("fly_semantic/runtime.py"),
        Path("semantic_tools/physical_collection.py"))}
    plan = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "TRAIN-only descriptive calibration, no model fit, no selection or acceptance of task quality",
        "parent_plan_digest": study["digest"], "gains_mv": list(GAINS), "group": 1201,
        "scenes": list(SCENES), "probe_ms": PROBE_MS, "analysis_last_ms": ANALYSIS_MS,
        "physical_window": "last 700 ms of recorded inputs, neural state cold-started with episode seed",
        "matched_scene": "limited_meal", "matched_hunger_levels": list(MATCHED_LEVELS),
        "legacy_hunger_levels": list(LEGACY_LEVELS), "legacy_luminance": .4, "legacy_seed": 314159,
        "historical_gate": {"downstream_changed_cells_min": 8, "difference_threshold_hz": 1.,
            "compare_hunger": [.2, .85], "port_max_hz_exclusive": 200.,
            "global_fraction_above_300hz_exclusive": .01, "finite_required": True},
        "v3_rate_guard": {"finite_required": True, "port_fraction_above_300hz_exclusive": .01,
            "global_fraction_above_300hz_exclusive": .01,
            "declared_before_results": True,
            "reason": "Original 200Hz input cap was a v1 engineering choice, not a biological or semantic-spec constraint; v3 guards proximity to the existing 333Hz refractory ceiling. Old result retained diagnostically."},
        "normalized_cell_difference": "abs(right-left)/max(abs(left),abs(right),1Hz)",
        "fitted_parameters": [], "selected_gain_mv": None, "validation_or_test_opened": False,
        "brain_config": study["brain_config"], "brain_units": "additive LIF drive in mV relative to rest, not pA",
        "feature_digest": features.feature_digest, "mapping_digest": mapping["digest"],
        "calibration_digest": calibration["digest"], "source_hashes": sources,
        "protected_sha256": protected, "total_simulated_seconds": 4 * (5 + 4 + 4) * .7,
        "limits": ["Same recorded body inputs do not imply the alternative gain would produce the same free-body trajectory.",
            "700ms cold-start windows are calibration samples, not full-history physical predictions.",
            "Matched overrides change only artificial hunger; native body-dependent current remains recorded.",
            "Rate differences indicate distinguishability of these probes only, not learned NEED_FOOD accuracy.",
            "Historical rate/propagation gate computed unchanged as diagnostic compatibility, not automatic v3 exclusion.",
            "New v3 finite/global-and-input saturation guards were explicitly declared before results; broader physical task validation still required."]}
    plan["digest"] = digest(plan)
    write_new(output / "gain-probe-plan.json", plan)
    started = time.perf_counter()
    brain = Brain(graph, BrainConfig(**study["brain_config"]), seed=314159)
    nominal_sha = hashlib.sha256(memoryview(brain.weights)).hexdigest()
    generic_initial = brain.get_state()
    rows, arrays = [], {}
    probes = [("recorded", scene, None) for scene in SCENES]
    probes += [("matched", "limited_meal", h) for h in MATCHED_LEVELS]
    probes += [("legacy", "uniform", h) for h in LEGACY_LEVELS]
    for gain in GAINS:
        for kind, scene, level in probes:
            initial = dict(generic_initial)
            if kind == "legacy":
                luminance = np.full((PROBE_MS // 10, len(brain.retina)), .4, np.float32)
                specs = [{"name": "semantic_hunger", "indices": input_indices.tolist()}]
                values = [np.zeros((PROBE_MS // 10, len(input_indices)), np.float32)]
            else:
                window = windows[scene]
                luminance, specs, values = window["luminance"], window["specs"], window["values"]
                initial["voltage"] = np.random.default_rng(window["episode"]["seed"]).uniform(0, 2, len(brain.ids)).astype(np.float32)
            key = f"gain_{gain:g}_{kind}_{scene}" + (f"_h_{level:g}" if level is not None else "")
            row, raw = replay_probe(brain, initial, features, input_indices, luminance, specs, values,
                                    gain, calibration["hunger_gain_mv"], level)
            row.update(key=key, gain_mv=gain, kind=kind, scene=scene, hunger_override=level)
            rows.append(row)
            for name, value in raw.items():
                arrays[key + "__" + name] = value
            print(json.dumps(row, allow_nan=False), flush=True)
    comparisons, gates = [], []
    for gain in GAINS:
        matched = [r for r in rows if r["gain_mv"] == gain and r["kind"] == "matched"]
        for i, left in enumerate(matched):
            for right in matched[i+1:]:
                comparisons.append({"gain_mv": gain, "same_background": "train-1201-limited_meal",
                    "hunger_pair": [left["hunger_override"], right["hunger_override"]],
                    **{name: pair_difference(arrays[left["key"] + "__" + name], arrays[right["key"] + "__" + name])
                       for name in ("input_hz", "feature_hz")}})
        legacy = [r for r in rows if r["gain_mv"] == gain and r["kind"] == "legacy"]
        lower = next(r for r in legacy if r["hunger_override"] == .2)
        upper = next(r for r in legacy if r["hunger_override"] == .85)
        difference = pair_difference(arrays[lower["key"] + "__feature_hz"], arrays[upper["key"] + "__feature_hz"])
        gates.append({"gain_mv": gain, "historical_downstream_difference": difference,
            "historical_gate_pass": all(r["legacy_rate_guard_pass"] for r in legacy) and difference["changed_at_least_1hz"] >= 8,
            "v3_all_probe_rate_guards_pass": all(r["v3_rate_guard_pass"] for r in rows if r["gain_mv"] == gain),
            "all_recorded_and_matched_rate_guards_pass": all(r["legacy_rate_guard_pass"] for r in rows if r["gain_mv"] == gain and r["kind"] != "legacy")})
    changed = [p for p, sha in protected.items() if file_digest(p) != sha]
    weights_unchanged = hashlib.sha256(memoryview(brain.weights)).hexdigest() == nominal_sha
    if changed or not weights_unchanged or any(file_digest(p) != sha for p, sha in sources.items()):
        raise ValueError("Protected source/graph or recorded inputs changed during calibration")
    arrays.update(input_indices=input_indices, feature_indices=features.indices,
                  input_body_ids=np.asarray(brain.ids[input_indices]))
    raw_path = output / "gain-probe-rates.npz"
    with raw_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result = {"schema_version": 1, "status": "CALIBRATION_ONLY_NO_GAIN_SELECTED", "plan_digest": plan["digest"],
        "rows": rows, "matched_comparisons": comparisons, "gain_gates": gates,
        "selected_gain_mv": None, "validation_or_test_opened": False,
        "weights_unchanged": weights_unchanged, "nominal_weights_sha256": nominal_sha,
        "changed_protected_files": changed, "raw_path": raw_path.as_posix(), "raw_sha256": file_digest(raw_path),
        "wall_seconds": time.perf_counter() - started, "total_simulated_seconds": plan["total_simulated_seconds"],
        "limits": plan["limits"]}
    result["digest"] = digest(result)
    write_new(output / "gain-probe-results.json", result)
    lines = ["# TRAIN-only artificial hunger gain probe", "", "Calibration only; no gain selected and no head trained.",
        "", "Five TRAIN 1201 recorded body backgrounds, last 700 ms cold-started, last 300 ms measured. "
        "Matched hunger overrides use exactly the same limited-meal background and initial neural state. "
        "They change only the artificial channel. Rates are Hz; injected drive is mV relative to rest.", "",
        "| Gain mV | Historical gate | V3 all-probe rate guards | .45 vs .75 downstream changed ≥1 Hz | .60 vs .75 changed |", "|---|---|---|---|---|"]
    for gate in gates:
        pairs = {tuple(p["hunger_pair"]): p for p in comparisons if p["gain_mv"] == gate["gain_mv"]}
        lines.append(f"| {gate['gain_mv']:g} | {gate['historical_gate_pass']} | {gate['v3_all_probe_rate_guards_pass']} | "
                     f"{pairs[(.45,.75)]['feature_hz']['changed_at_least_1hz']} | {pairs[(.6,.75)]['feature_hz']['changed_at_least_1hz']} |")
    lines += ["", "Historical gate: at least 8 downstream cells differ ≥1 Hz for hunger .2 vs .85; "
        "every 0/.2/.85/1 probe must be finite, all input cells <200 Hz, and <1% of global cells ≥300 Hz. "
        "The exact original uniform-retina .4 protocol and seed 314159 are retained.", "",
        "V3 rate guard, declared before results: finite states, <1% of input cells and <1% of all cells ≥300 Hz. "
        "The earlier 200 Hz cutoff remains a reported v1 diagnostic, not an automatic v3 exclusion.", "",
        "Detailed per-cell input/downstream rates, normalized pair differences and traces are in the JSON/NPZ. "
        "The 512 primary downstream cells exclude direct recipients. No biological hunger circuit is claimed.", "",
        *["- " + item for item in plan["limits"]], "", f"Simulated {plan['total_simulated_seconds']:.1f} s; wall {result['wall_seconds']:.1f} s. "
        "Frozen v1/v2 input artifacts and core weights verified unchanged."]
    with (output / "GAIN_PROBE.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/semantic-v2/physical-study"))
    parser.add_argument("--output", type=Path, default=Path("runs/semantic-v3/calibration"))
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    args = parser.parse_args()
    value = run(args.source, args.output, args.graph)
    print(json.dumps({k: value[k] for k in ("status", "gain_gates", "wall_seconds")}), flush=True)
