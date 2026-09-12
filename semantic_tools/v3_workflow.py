"""Gain-development replay and prospective physical validation for NEED_FOOD v3.

Development reuses ONLY old train/validation body input recordings. A changed
gain replays those inputs; final test runs a new physical body at the new gain.
Old core code, calibration artifacts, heads and studies are not edited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fly_arena.brain import Brain, BrainConfig
from fly_arena.config import load_config
from fly_semantic.mapping import digest, file_digest, load_mapping
from fly_semantic.runtime import SemanticSimulation, load_calibration, make_features
from semantic_tools.physical_collection import NeuralRecorder, SCENES, source_hashes as legacy_collection_sources


ROOT = Path("runs/semantic-v3")
SOURCE = Path("runs/semantic-v2/physical-study")
OLD_CALIBRATION = Path("runs/semantic-v1/calibration-selected")
PROBE = ROOT / "calibration/gain-probe-results.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def workflow_sources():
    return {"semantic_tools/v3_workflow.py": file_digest(Path(__file__)), **legacy_collection_sources()}


def test_episodes():
    result = []
    for group, portion in zip((4201, 4202, 4203), (.6, 1.2, 1.6)):
        for index, (scene, duration) in enumerate(SCENES.items()):
            seed = group * 10 + index
            rng = np.random.default_rng(seed + 510000)
            cfg = load_config(Path("configs/ethology-unscaled.toml"), "feeding-contact")
            cfg.setdefault("motor", {})["adaptive_proboscis"] = True
            cfg["initial"]["energy"] = float(rng.uniform(75, 85) if scene.startswith("sated") else rng.uniform(10, 20))
            cfg["initial"]["sleep_pressure"] = float(rng.uniform(.02, .15))
            cfg["environment"]["light"] = float(rng.uniform(.65, 1.))
            cfg["organism"]["intake_rate"] = float(rng.uniform(1.25, 1.5) if scene == "meal_slow" else rng.uniform(2.6, 3.))
            if scene.endswith("away"):
                cfg["food"] = []
            else:
                cfg["food"][0].update(x=float(rng.uniform(1.52, 1.68)), y=float(rng.uniform(-.06, .06)),
                    radius=float(rng.uniform(1.05, 1.17)), amount=portion if scene == "limited_meal" else 5.,
                    taste=float(rng.uniform(.75, 1.)), odor=float(rng.uniform(.6, 1.)))
            result.append({"episode_id": f"test-{group}-{scene}", "split": "test", "seed_group": group,
                "seed": seed, "scene": scene, "duration_ms": duration,
                "expected_release": scene in ("meal_fast", "meal_slow"), "config": cfg})
    return result


def prepare():
    from semantic_tools.v3_learning import protocol_fields
    if (ROOT / "protocol.json").exists():
        raise FileExistsError("V3 protocol already frozen")
    probe = json.loads(PROBE.read_text())
    parent = json.loads((SOURCE / "plan.json").read_text())
    old = json.loads((OLD_CALIBRATION / "calibration.json").read_text())
    mapping = load_mapping(OLD_CALIBRATION / "mapping.json", Path("data/graph"))
    fields = protocol_fields()
    gains = [4.5, 5., 6.]
    gates = {float(row["gain_mv"]): row for row in probe["gain_gates"]}
    if any(not gates[g]["v3_all_probe_rate_guards_pass"] for g in gains):
        raise ValueError("A predeclared development gain did not pass v3 calibration guards")
    training = [ep for ep in parent["episodes"] if ep["split"] != "test"]
    source_records = {ep["episode_id"]: {"path": (SOURCE / "episodes/connected" / (ep["episode_id"] + ".npz")).as_posix(),
        "sha256": file_digest(SOURCE / "episodes/connected" / (ep["episode_id"] + ".npz"))} for ep in training}
    protocol = {"schema_version": 3, **fields, "gains_mv": gains, "source_plan_sha256": file_digest(SOURCE / "plan.json"),
        "source_plan_digest": parent["digest"], "source_records": source_records, "probe_sha256": file_digest(PROBE),
        "workflow_sources": workflow_sources(), "test_episodes": test_episodes(),
        "scope": "18 old physical TRAIN and6 validation input traces replayed at candidate gains; 18 new free-body test episodes after selection",
        "safety_rule": "predeclared v3 finite states, <1% global and addressed inputs >=300Hz; historical <200Hz input ceiling separately reported",
        "input_precision": "Development changes the old float32 semantic current by gain/4; this preserves encoded need up to float32 rounding. Fresh physics uses original organism float64 need."}
    protocol["digest"] = digest(protocol)
    write_json(ROOT / "protocol.json", protocol)
    for gain in gains:
        folder = ROOT / "development" / f"gain-{gain:g}"
        calibration_dir = folder / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        (calibration_dir / "mapping.json").write_bytes((OLD_CALIBRATION / "mapping.json").read_bytes())
        calibrated = {"schema_version": 1, "status": "CALIBRATED", "mapping_digest": mapping["digest"],
            "brain_config": old["brain_config"], "hunger_gain_mv": gain, "hunger_status": "CALIBRATED",
            "cue_status": "UNVALIDATED", "cue_gain_mv": None, "pulse_ms": 250,
            "scope": "Artificial hunger drive only; gain passed TRAIN-only v3 probe, not a proven behavioral result",
            "brain_units": "additive drive mV relative to rest", "probe_sha256": file_digest(PROBE),
            "v3_rate_gate": gates[gain], "parent_calibration_sha256": file_digest(OLD_CALIBRATION / "calibration.json")}
        calibrated["digest"] = digest(calibrated)
        write_json(calibration_dir / "calibration.json", calibrated)
        study = {k: copy.deepcopy(v) for k, v in parent.items() if k != "digest"}
        study.update(fields)
        study.update(schema_version=2, v3_protocol_digest=protocol["digest"], hunger_gain_mv=gain,
            scope=protocol["scope"], calibration_dir=calibration_dir.as_posix(), calibration_digest=calibrated["digest"],
            feature_digest=make_features(mapping, calibrated).feature_digest,
            source_records=source_records, development_workflow_sources=protocol["workflow_sources"],
            replay_workflow_sources=protocol["workflow_sources"],
            episodes=copy.deepcopy(training) + copy.deepcopy(protocol["test_episodes"]),
            seed_groups={"train": [1201, 1202, 1203], "validation": [2201], "test": [4201, 4202, 4203]},
            collection_conditions=["connected", "homeostasis_off", "transmission_off"],
            baseline_model_path="runs/semantic-v2/physical-study/state-readout.npz",
            baseline_model_sha256=file_digest("runs/semantic-v2/physical-study/state-readout.npz"),
            nominal_weights_sha256=json.loads((SOURCE / "selection.json").read_text(encoding="utf-8"))["nominal_weights_sha256"],
            include_synthetic_train=False)
        study["digest"] = digest(study)
        write_json(folder / "plan.json", study)
    return protocol


def load_study(folder):
    study = json.loads((Path(folder) / "plan.json").read_text())
    plain = dict(study)
    if plain.pop("digest", None) != digest(plain):
        raise ValueError("V3 study digest mismatch")
    if study["development_workflow_sources"] != workflow_sources():
        raise ValueError("V3 workflow sources changed after protocol freeze")
    return study


def _validate_operation(folder, study, ep):
    if study != load_study(folder):
        raise ValueError("Operation study differs from the frozen plan in its output directory")
    planned = {item["episode_id"]: item for item in study["episodes"]}
    if planned.get(ep["episode_id"]) != ep:
        raise ValueError("Operation episode differs from the frozen plan")


def replay(source, folder, study, ep, condition="connected", development=False):
    source, folder = Path(source), Path(folder)
    _validate_operation(folder, study, ep)
    if development:
        if condition != "connected" or ep["split"] not in ("train", "validation"):
            raise ValueError("Development source must be frozen training/validation only, connected condition")
        record = study["source_records"].get(ep["episode_id"])
        if (record is None or source.resolve() != Path(record["path"]).resolve()
                or file_digest(source) != record["sha256"]):
            raise ValueError("Development input path or bytes differ from frozen source records")
    else:
        if condition not in ("homeostasis_off", "transmission_off", "replay_check") or ep["split"] != "test":
            raise ValueError("Fresh test replay requires a declared test control condition")
        if source.resolve() != (folder / "episodes/connected" / (ep["episode_id"] + ".npz")).resolve():
            raise ValueError("Fresh test replay requires its matched connected physical source")
        from semantic_tools.v3_learning import verify_test_binding
        verify_test_binding(folder)
    target = folder / "episodes" / condition / (ep["episode_id"] + ".npz")
    if target.exists():
        raise FileExistsError(target)
    cal_dir = Path(study["calibration_dir"])
    mapping = load_mapping(cal_dir / "mapping.json", Path("data/graph"))
    calibration = load_calibration(cal_dir / "calibration.json", mapping)
    if calibration["digest"] != study["calibration_digest"]:
        raise ValueError("V3 calibration changed")
    brain = Brain(Path("data/graph"), BrainConfig(**study["brain_config"]), seed=ep["seed"])
    nominal = hashlib.sha256(memoryview(brain.weights)).hexdigest()
    if nominal != study["nominal_weights_sha256"]:
        raise ValueError("Current nominal graph weights differ from the frozen v3 plan")
    if condition == "transmission_off":
        brain.weights = np.zeros_like(brain.weights)
    features = make_features(mapping, calibration)
    started = time.perf_counter()
    with np.load(source, allow_pickle=False) as a:
        meta = json.loads(str(a["metadata"]))
        if meta["episode"] != ep or meta["nominal_weights_sha256"] != nominal:
            raise ValueError("Replay body/graph identity mismatch")
        if development:
            # Frozen bytes establish the full old calibration/source lineage.
            parent = json.loads((SOURCE / "plan.json").read_text())
            if meta["plan_digest"] != parent["digest"] or meta["source_hashes"] != parent["source_hashes"]:
                raise ValueError("Development source does not retain its parent plan/source identity")
        elif (meta.get("body_physics") is not True or meta["plan_digest"] != study["digest"]
                or meta["calibration_digest"] != study["calibration_digest"]
                or meta["feature_digest"] != study["feature_digest"]
                or meta.get("physical_workflow_sources") != study["development_workflow_sources"]):
            raise ValueError("Fresh replay source differs from the frozen physical plan/calibration/features")
        original_gain = 4. if development else study["hunger_gain_mv"]
        specs = meta["stimulus_channels"]
        stim = [a[f"replay_values_{i}"] for i in range(len(specs))]
        luminance = a["replay_luminance"]
        exact_need = a["replay_need_float64"] if "replay_need_float64" in a else None
        ticks = ep["duration_ms"] // 10
        if (ep["duration_ms"] % 10 or study["sample_ms"] % 10
                or ep["duration_ms"] % study["sample_ms"] or luminance.ndim < 2
                or len(luminance) != ticks or not np.isfinite(luminance).all()):
            raise ValueError("Replay visual input shape, timing or finite-value check failed")
        names = [spec["name"] for spec in specs]
        if len(set(names)) != len(names) or names.count("semantic_hunger") != 1:
            raise ValueError("Replay requires unique channels and exactly one hunger input")
        for spec, values in zip(specs, stim):
            indices = np.asarray(spec["indices"])
            if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                    or values.shape != (ticks, len(indices)) or not np.isfinite(values).all()):
                raise ValueError("Replay current addresses or finite array shape differ from recorded ticks")
        if not development and exact_need is None:
            raise ValueError("Fresh physical replay requires recorded float64 need")
        if exact_need is not None and (exact_need.dtype != np.float64 or exact_need.shape != (ticks,)
                or not np.isfinite(exact_need).all() or np.any((exact_need < 0) | (exact_need > 1))):
            raise ValueError("Replay need must be finite float64 with one value per tick in [0,1]")
        sample_ticks = study["sample_ms"] // 10
        expected_times = np.arange(study["sample_ms"], ep["duration_ms"] + 1, study["sample_ms"])
        if not np.array_equal(a["time_ms"], expected_times):
            raise ValueError("Replay sample timestamps differ from the frozen sampling grid")
        if any(a[key].shape[0] != len(expected_times) for key in ("y", "hunger", "intake", "taste", "action")):
            raise ValueError("Replay body labels differ from the frozen sampling grid")
        if exact_need is not None:
            active = bool(study.get("teacher_initial_active", False))
            labels = []
            for index, need in enumerate(exact_need):
                active = bool(need >= study["teacher_on"] or (active and need > study["teacher_off"]))
                if (index + 1) % sample_ticks == 0:
                    labels.append(active)
            if (not np.array_equal(a["hunger"], exact_need[sample_ticks - 1::sample_ticks])
                    or not np.array_equal(a["y"], labels)):
                raise ValueError("Fresh physical replay need and teacher labels are not coupled to the same ticks")
        X, input_rates = [], []
        for tick in range(len(luminance)):
            channels = {}
            for spec, values in zip(specs, stim):
                indices = np.array(spec["indices"], np.int32)
                current = values[tick]
                if spec["name"] == "semantic_hunger":
                    if condition == "homeostasis_off":
                        continue
                    current = (study["hunger_gain_mv"] * exact_need[tick] if exact_need is not None
                               else current * np.float32(study["hunger_gain_mv"] / original_gain))
                channels[spec["name"]] = (indices, current)
            _, counts = brain.step(luminance[tick], sensory_currents=channels)
            z = features.update(counts, 10)
            input_rates.append(counts[[r["index"] for r in mapping["ports"]["hunger"]]] * 100.)
            if (tick + 1) * 10 % study["sample_ms"] == 0:
                X.append(z)
        X = np.array(X)
        if condition == "replay_check":
            np.testing.assert_array_equal(X, a["X"])
        metadata = {**meta, "plan_digest": study["digest"], "condition": condition,
            "feature_digest": features.feature_digest, "calibration_digest": calibration["digest"],
            "body_physics": False, "matched_original_hunger": True, "source_input_path": source.as_posix(),
            "source_input_sha256": file_digest(source), "replay_gain_mv": study["hunger_gain_mv"],
            "replay_original_gain_mv": original_gain, "replay_workflow_sources": workflow_sources(),
            "replay_scope": "Recorded body inputs held fixed; no free-body intervention result",
            "old_plan_digest": meta["plan_digest"], "source_hashes": study["source_hashes"],
            "source_hashes_role": "Inherited parent collector identity for old physical loader; replay_workflow_sources identifies the actual v3 execution code",
            "wall_seconds": time.perf_counter() - started}
        arrays = {key: a[key].copy() for key in ("y", "time_ms", "hunger", "intake", "taste", "action")}
        arrays.update(X=X, diagnostic_input_rates=np.array(input_rates), metadata=np.array(json.dumps(metadata, allow_nan=False)))
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
    print(json.dumps({"gain": study["hunger_gain_mv"], "episode": ep["episode_id"], "condition": condition,
                      "seconds": round(metadata["wall_seconds"], 2)}), flush=True)


class NeedRecorder(NeuralRecorder):
    def __init__(self, *args):
        super().__init__(*args)
        self.exact_need = []

    def step(self, *args, **kwargs):
        self.exact_need.append(float(self.organism.hunger))
        return super().step(*args, **kwargs)


def collect_test(folder, study, ep):
    folder = Path(folder)
    _validate_operation(folder, study, ep)
    target = folder / "episodes/connected" / (ep["episode_id"] + ".npz")
    if target.exists() or ep["split"] != "test":
        raise ValueError("New physical collection requires an unrecorded test episode")
    from semantic_tools.v3_learning import verify_test_binding
    verify_test_binding(folder)
    cal_dir = Path(study["calibration_dir"])
    sim = SemanticSimulation(ep["config"], graph=Path("data/graph"), seed=ep["seed"], enabled=True,
        mapping_path=cal_dir / "mapping.json", calibration_path=cal_dir / "calibration.json",
        readout_path=folder / "state-readout.npz", episode_id=ep["episode_id"])
    recorder = NeedRecorder(sim.legacy.brain, sim.channel, sim.organism)
    sim.legacy.brain = recorder
    started = time.perf_counter()
    X, y, times, needs, intake, taste, actions, scores = [], [], [], [], [], [], [], []
    active, transitions = False, []
    events = []
    try:
        if (hashlib.sha256(memoryview(sim.brain.weights)).hexdigest() != study["nominal_weights_sha256"]
                or sim.channel.features.feature_digest != study["feature_digest"]
                or sim.channel.calibration["digest"] != study["calibration_digest"]):
            raise ValueError("Fresh physical runtime graph/calibration/features differ from frozen plan")
        for tick in range(ep["duration_ms"] // 10):
            need = sim.organism.hunger
            previous = active
            active = bool(need >= study["teacher_on"] or (active and need > study["teacher_off"]))
            if active != previous:
                transitions.append({"time_ms": sim.brain.clock, "active": active, "hunger": need})
            sim.step()
            events.extend(e.to_dict() for e in sim.channel.last_events)
            if (tick + 1) * 10 % study["sample_ms"] == 0:
                X.append(sim.channel.features.values()); y.append(active); times.append(sim.brain.clock); needs.append(need)
                intake.append(sim.organism.state.ingested_total)
                taste.append(max([sim.last_frame.mouth_taste, *sim.last_frame.tarsal_taste.values()]))
                actions.append(sim.last_decision.action); scores.append(sim.channel.last_scores[1])
        meta = {"episode": ep, "condition": "connected", "plan_digest": study["digest"],
            "feature_digest": sim.channel.features.feature_digest, "calibration_digest": sim.channel.calibration["digest"],
            "nominal_weights_sha256": hashlib.sha256(memoryview(sim.brain.weights)).hexdigest(),
            "target_transitions": transitions, "source_hashes": study["source_hashes"],
            "physical_workflow_sources": workflow_sources(), "body_physics": True,
            "source_hashes_role": "Inherited parent collector identity for old physical loader; physical_workflow_sources identifies the actual v3 execution code",
            "resource_balance": sim.resource_balances(), "minimum_upright": sim.minimum_upright,
            "ingested_total": sim.organism.state.ingested_total, "final_hunger": sim.organism.hunger,
            "stimulus_channels": recorder.channel_specs, "live_events": events,
            "wall_seconds": time.perf_counter() - started}
        arrays = {"X": np.array(X), "y": np.array(y, np.int8), "time_ms": np.array(times, np.int64),
            "hunger": np.array(needs), "intake": np.array(intake), "taste": np.array(taste), "action": np.array(actions),
            "live_scores": np.array(scores), "replay_luminance": np.array(recorder.luminance),
            "replay_need_float64": np.array(recorder.exact_need, np.float64),
            "diagnostic_input_rates": np.array(recorder.input_rates), "metadata": np.array(json.dumps(meta, allow_nan=False))}
        arrays.update({f"replay_values_{i}": np.array(values) for i, values in enumerate(recorder.channel_values)})
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        print(json.dumps({"episode": ep["episode_id"], "gain": study["hunger_gain_mv"], "intake": meta["ingested_total"],
            "hunger": meta["final_hunger"], "seconds": round(meta["wall_seconds"], 2)}), flush=True)
    finally:
        sim.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "develop", "collect", "replay"])
    p.add_argument("--folder", type=Path, default=ROOT / "selected")
    p.add_argument("--gain", type=float)
    p.add_argument("--group", type=int)
    p.add_argument("--episode")
    p.add_argument("--condition", choices=["homeostasis_off", "transmission_off", "replay_check"], default="replay_check")
    a = p.parse_args()
    if a.command == "prepare":
        prepare()
        return
    folder = ROOT / "development" / f"gain-{a.gain:g}" if a.command == "develop" else a.folder
    study = load_study(folder)
    for ep in study["episodes"]:
        if (a.command == "develop") == (ep["split"] == "test"):
            continue
        if a.group is not None and ep["seed_group"] != a.group or a.episode is not None and ep["episode_id"] != a.episode:
            continue
        if a.command == "develop":
            replay(study["source_records"][ep["episode_id"]]["path"], folder, study, ep, development=True)
        elif a.command == "collect":
            collect_test(folder, study, ep)
        else:
            replay(folder / "episodes/connected" / (ep["episode_id"] + ".npz"), folder, study, ep, a.condition)


if __name__ == "__main__":
    main()
