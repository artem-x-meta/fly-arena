"""Short input-only calibration in the unmodified full LIF backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from fly_arena.brain import Brain
from fly_semantic.mapping import build_mapping, digest, load_mapping


def run(graph, output, amplitudes=(8., 12., 20., 28.), selection_pool="cb_intrinsic"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "calibration.json").exists():
        raise FileExistsError("Use a new output directory; calibration is frozen")
    mapping_path = output / "mapping.json"
    if mapping_path.exists():
        mapping = load_mapping(mapping_path, graph)
        if mapping.get("selection_pool", "cb_intrinsic") != selection_pool:
            raise ValueError("Existing mapping uses a different artificial input pool")
    else:
        mapping = build_mapping(graph, selection_pool=selection_pool)
        mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    amplitudes = tuple(float(a) for a in amplitudes)
    if not amplitudes or any(not np.isfinite(a) or a <= 0 for a in amplitudes):
        raise ValueError("Amplitudes must be positive finite drive values")
    plan = {"schema_version": 1, "amplitudes_mv": list(amplitudes), "selection_pool": selection_pool,
        "probe_ms": 700, "analysis_last_ms": 300, "luminance": .4,
        "hunger_levels": [0., .2, .85, 1.], "mapping_digest": mapping["digest"],
        "choice_rule": "first amplitude with >=8 downstream cells changed by >=1Hz between hunger .2/.85, port rate <200Hz, <1% global cells >=300Hz, all finite; otherwise TARGET_NOT_REACHED",
        "meaning": "engineering calibration only; no held-out labels or task performance used"}
    (output / "calibration-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    brain = Brain(Path(graph), seed=314159)
    initial = brain.get_state()
    features = np.array([r["index"] for r in mapping["features"]])
    ports = {k: np.array([r["index"] for r in v], np.int32) for k, v in mapping["ports"].items()}
    rows = []
    selected = None
    started = time.perf_counter()
    for amplitude in plan["amplitudes_mv"]:
        rates = {}
        probe_rows = []
        for level in plan["hunger_levels"]:
            brain.set_state(initial)
            counts = np.zeros(len(brain.ids), np.int64)
            for tick in range(70):
                _, spikes = brain.step(np.full(len(brain.retina), .4, np.float32),
                    sensory_currents={"semantic_hunger": (ports["hunger"], amplitude * level)})
                if tick >= 40:
                    counts += spikes
            hz = counts / .3
            rates[level] = hz[features]
            probe_rows.append({"level": level, "changed_from_zero": None,
                "global_mean_hz": float(hz.mean()), "global_saturated_fraction": float(np.mean(hz >= 300)),
                "port_max_hz": float(hz[ports["hunger"]].max()),
                "port_mean_hz": float(hz[ports["hunger"]].mean()),
                "finite": bool(np.isfinite(brain.voltage).all() and np.isfinite(brain.current).all())})
        for probe in probe_rows:
            probe["changed_from_zero"] = int(np.count_nonzero(np.abs(rates[probe["level"]] - rates[0.]) >= 1.))
        changed = int(np.count_nonzero(np.abs(rates[.85] - rates[.2]) >= 1.))
        acceptable = changed >= 8 and all(r["finite"] and r["global_saturated_fraction"] < .01 and r["port_max_hz"] < 200 for r in probe_rows)
        row = {"amplitude_mv": amplitude, "changed_downstream_cells": changed,
               "accepted": acceptable, "probes": probe_rows}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if acceptable:
            selected = amplitude
            break
    # Same amplitude/count for all input IDs. This is propagation, not behavior.
    cue_rows = []
    if selected is not None:
        for key in ("101", "102", "103"):
            brain.set_state(initial)
            counts = np.zeros(len(brain.ids), np.int64)
            for tick in range(50):
                channels = {"semantic_cue": (ports[key], selected)} if 10 <= tick < 35 else {}
                _, spikes = brain.step(np.full(len(brain.retina), .4, np.float32), sensory_currents=channels)
                if 10 <= tick < 35:
                    counts += spikes
            cue_rows.append({"concept_id": int(key), "feature_mean_hz": (counts[features] / .25).tolist(),
                "global_mean_hz": float((counts / .25).mean()),
                "global_saturated_fraction": float(np.mean(counts / .25 >= 300)),
                "port_max_hz": float((counts[ports[key]] / .25).max()),
                "port_mean_hz": float((counts[ports[key]] / .25).mean()),
                "finite": bool(np.isfinite(brain.voltage).all() and np.isfinite(brain.current).all())})
    pairwise = []
    for a in range(len(cue_rows)):
        for b in range(a + 1, len(cue_rows)):
            left, right = np.array(cue_rows[a]["feature_mean_hz"]), np.array(cue_rows[b]["feature_mean_hz"])
            pairwise.append({"concept_ids": [cue_rows[a]["concept_id"], cue_rows[b]["concept_id"]],
                "changed_downstream_cells_at_least_1hz": int(np.count_nonzero(np.abs(left - right) >= 1.)),
                "l2_distance_hz": float(np.linalg.norm(left - right))})
    result = {"schema_version": 1, "status": "CALIBRATED" if selected is not None else "TARGET_NOT_REACHED",
        "hunger_status": "CALIBRATED" if selected is not None else "TARGET_NOT_REACHED",
        "cue_status": "UNVALIDATED", "cue_status_reason": "single-seed single-background propagation probes only; no robust cue calibration or behavior evaluated",
        "mapping_digest": mapping["digest"], "plan_digest": digest(plan), "brain_config": asdict(brain.config),
        "hunger_gain_mv": selected, "cue_gain_mv": None, "cue_probe_gain_mv": selected, "pulse_ms": 250,
        "brain_units": "additive LIF drive in mV relative to rest, not pA",
        "rows": rows, "cue_probes": cue_rows, "cue_pairwise_probe_differences": pairwise,
        "wall_seconds": time.perf_counter() - started}
    result["digest"] = digest(result)
    (output / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--output", type=Path, default=Path("runs/semantic-v1/calibration"))
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[8., 12., 20., 28.])
    parser.add_argument("--pool", choices=["cb_intrinsic", "cb_sensory"], default="cb_intrinsic")
    args = parser.parse_args()
    run(args.graph, args.output, args.amplitudes, selection_pool=args.pool)
