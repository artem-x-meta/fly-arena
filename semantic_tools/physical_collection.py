"""Prospective physical NEED_FOOD collection with exact neural-input replay.

The head never controls the body, so competing heads can be evaluated on the
same physical trajectory. Replay controls hold the recorded body's inputs fixed;
they are not counterfactual free-body trajectories.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fly_arena.brain import Brain, BrainConfig
from fly_arena.config import load_config
from fly_semantic.mapping import digest, file_digest, load_mapping
from fly_semantic.runtime import SemanticSimulation, load_calibration, make_features


CALIBRATION = Path("runs/semantic-v1/calibration-selected")
BASELINE_MODEL = Path("runs/semantic-v1/state-selected/state-readout.npz")
OUTPUT = Path("runs/semantic-v2/physical-study")
SCENES = {"hungry_away": 2000, "sated_away": 2000, "meal_fast": 5500,
          "meal_slow": 7500, "limited_meal": 5000, "sated_food": 3000}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def source_hashes():
    paths = [Path("semantic_tools/physical_collection.py"), *Path("fly_semantic").glob("*.py"), *Path("fly_arena").glob("*.py")]
    return {p.as_posix(): file_digest(p) for p in sorted(paths)}


def freeze_plan(output=OUTPUT, calibration=CALIBRATION):
    output, calibration = Path(output), Path(calibration)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise FileExistsError("Plan already frozen; collect it or choose a fresh directory")
    mapping = load_mapping(calibration / "mapping.json", Path("data/graph"))
    calibrated = load_calibration(calibration / "calibration.json", mapping)
    features = make_features(mapping, calibrated)
    episodes = []
    groups = {"train": [1201, 1202, 1203], "validation": [2201], "test": [3201, 3202]}
    for split, seed_groups in groups.items():
        for group in seed_groups:
            for index, (scene, duration) in enumerate(SCENES.items()):
                seed = group * 10 + index
                rng = np.random.default_rng(seed + 400000)
                cfg = load_config(Path("configs/ethology-unscaled.toml"), "feeding-contact")
                cfg.setdefault("motor", {})["adaptive_proboscis"] = True
                cfg["initial"]["energy"] = float(rng.uniform(75, 85) if scene.startswith("sated") else rng.uniform(10, 20))
                cfg["initial"]["sleep_pressure"] = float(rng.uniform(.02, .15))
                cfg["environment"]["light"] = float(rng.uniform(.65, 1.))
                rate = rng.uniform(1.25, 1.5) if scene == "meal_slow" else rng.uniform(2.6, 3.)
                cfg["organism"]["intake_rate"] = float(rate)
                if scene.endswith("away"):
                    cfg["food"] = []
                else:
                    patch = cfg["food"][0]
                    patch.update(x=float(rng.uniform(1.52, 1.68)), y=float(rng.uniform(-.06, .06)),
                        radius=float(rng.uniform(1.05, 1.17)), amount=.8 if scene == "limited_meal" else 5.,
                        taste=float(rng.uniform(.75, 1.)), odor=float(rng.uniform(.6, 1.)))
                episodes.append({"episode_id": f"{split}-{group}-{scene}", "split": split,
                    "seed_group": group, "seed": seed, "scene": scene, "duration_ms": duration,
                    "expected_release": scene in ("meal_fast", "meal_slow"), "config": cfg})
    study = {"schema_version": 2, "scope": "preliminary physical-only state head training; frozen v1 graph/input/features",
        "calibration_dir": calibration.as_posix(), "mapping_digest": mapping["digest"],
        "calibration_digest": calibrated["digest"], "feature_digest": features.feature_digest,
        "brain_config": calibrated["brain_config"], "baseline_model_path": BASELINE_MODEL.as_posix(),
        "baseline_model_sha256": file_digest(BASELINE_MODEL), "seed_groups": groups, "episodes": episodes,
        "sample_ms": 100, "label_reference_offset_ms": 10, "teacher_on": .7, "teacher_off": .55,
        "teacher_initial_active": False, "indicator_on": .7, "indicator_off": .4, "confirm_samples": 2,
        "score_threshold": .5, "l2_grid": [.001, .01, .1], "max_iter": 1000,
        "target_f1": .8, "indicator_target_f1": .8, "release_required_fraction": .75,
        "release_max_latency_ms": 1000, "release_stable_ms": 500, "maximum_false_active_fraction_on_sated": .1,
        "baseline_improvement_min": .1, "conditions": ["connected", "homeostasis_off", "transmission_off"],
        "controls_scope": "same recorded physical sensory inputs replayed through full Brain; no counterfactual body resimulation",
        "model_selection": "highest validation indicator F1, then raw F1, then stronger L2; test remains unseen",
        "uncertainty": "two held-out seed groups; descriptive pilot, not final 100-episode-per-seed study",
        "source_hashes": source_hashes()}
    study["collection_conditions"] = list(study["conditions"])
    study["acceptance"] = {key: study[key] for key in ("target_f1", "indicator_target_f1", "release_required_fraction",
        "release_max_latency_ms", "release_stable_ms", "maximum_false_active_fraction_on_sated", "baseline_improvement_min")}
    study["digest"] = digest(study)
    write_json(output / "plan.json", study)
    return study


def load_plan(output):
    study = json.loads((Path(output) / "plan.json").read_text())
    copy = dict(study)
    if copy.pop("digest", None) != digest(copy) or study.get("schema_version") != 2:
        raise ValueError("Physical study plan digest/schema mismatch")
    ids = [ep["episode_id"] for ep in study["episodes"]]
    seeds = [ep["seed"] for ep in study["episodes"]]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError("Physical episodes or RNG seeds overlap")
    group_sets = [set(study["seed_groups"][split]) for split in ("train", "validation", "test")]
    if any(a & b for i, a in enumerate(group_sets) for b in group_sets[:i]):
        raise ValueError("Physical seed groups overlap")
    if any(ep["seed_group"] not in study["seed_groups"][ep["split"]] for ep in study["episodes"]):
        raise ValueError("Physical episode split mismatch")
    return study


class NeuralRecorder:
    """Transparent recording around the existing instance proxy, with no policy."""
    def __init__(self, original, channel, organism):
        self.original, self.channel, self.organism = original, channel, organism
        self.luminance, self.channel_specs, self.channel_values = [], None, []
        self.input_rates, self.feature_rates = [], []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def step(self, luminance, milliseconds=10, **kwargs):
        if milliseconds != 10 or kwargs.get("blind", False) or kwargs.get("disabled_channels") or len(kwargs.get("blocked_outputs", ())):
            raise ValueError("Collector expects standard 10ms unablated physical input")
        channels = dict(kwargs.get("sensory_currents") or {})
        channels["semantic_hunger"] = (self.channel.ports["hunger"], self.channel.calibration["hunger_gain_mv"] * self.organism.hunger)
        channels.update(kwargs.get("modulation") or {})
        specs = [{"name": name, "indices": np.asarray(idx, np.int32).tolist()} for name, (idx, _) in channels.items()]
        if self.channel_specs is None:
            self.channel_specs = specs
            self.channel_values = [[] for _ in specs]
        if specs != self.channel_specs:
            raise ValueError("Changing neural input addresses requires another recording schema")
        self.luminance.append(np.asarray(luminance, np.float32).copy())
        for values, (indices, value) in zip(self.channel_values, channels.values()):
            values.append(np.broadcast_to(np.asarray(value, np.float32), np.asarray(indices).shape).copy())
        result = self.original.step(luminance, milliseconds, **kwargs)
        counts = result[1]
        self.input_rates.append(counts[self.channel.ports["hunger"]].astype(np.float64) * 100.)
        self.feature_rates.append([float(counts[self.channel.features.indices].mean() * 100),
                                   int(np.count_nonzero(counts[self.channel.features.indices]))])
        return result


def _paths(output, condition, episode):
    folder = Path(output) / "episodes" / condition
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (episode["episode_id"] + ".npz")


def collect_episode(graph, output, study, ep):
    path = _paths(output, "connected", ep)
    if path.exists():
        raise FileExistsError(path)
    if source_hashes() != study["source_hashes"]:
        raise ValueError("Collection source changed after plan freeze")
    started = time.perf_counter()
    calibration = Path(study["calibration_dir"])
    sim = SemanticSimulation(ep["config"], graph=Path(graph), seed=ep["seed"], enabled=True,
        mapping_path=calibration / "mapping.json", calibration_path=calibration / "calibration.json", episode_id=ep["episode_id"])
    recorder = NeuralRecorder(sim.legacy.brain, sim.channel, sim.organism)
    sim.legacy.brain = recorder
    nominal_sha = hashlib.sha256(memoryview(sim.brain.weights)).hexdigest()
    X, y, times, needs, intake, taste, actions = [], [], [], [], [], [], []
    active = study["teacher_initial_active"]
    transitions = []
    try:
        for tick in range(ep["duration_ms"] // 10):
            need = sim.organism.hunger
            previous = active
            active = bool(need >= study["teacher_on"] or (active and need > study["teacher_off"]))
            if previous != active:
                transitions.append({"time_ms": sim.brain.clock, "active": active, "hunger": need})
            sim.step()
            if (tick + 1) * 10 % study["sample_ms"] == 0:
                X.append(sim.channel.features.values())
                y.append(active); times.append(sim.brain.clock); needs.append(need)
                intake.append(sim.organism.state.ingested_total)
                taste.append(max([sim.last_frame.mouth_taste, *sim.last_frame.tarsal_taste.values()]))
                actions.append(sim.last_decision.action)
        meta = {"episode": ep, "plan_digest": study["digest"], "condition": "connected",
            "feature_digest": sim.channel.features.feature_digest, "calibration_digest": sim.channel.calibration["digest"],
            "nominal_weights_sha256": nominal_sha, "target_transitions": transitions,
            "source_hashes": study["source_hashes"], "body_physics": True,
            "minimum_upright": sim.minimum_upright, "resource_balance": sim.resource_balances(),
            "final_hunger": sim.organism.hunger, "ingested_total": sim.organism.state.ingested_total,
            "wall_seconds": time.perf_counter() - started, "stimulus_channels": recorder.channel_specs}
        arrays = {"X": np.array(X), "y": np.array(y, np.int8), "time_ms": np.array(times, np.int64),
            "hunger": np.array(needs), "intake": np.array(intake), "taste": np.array(taste), "action": np.array(actions),
            "replay_luminance": np.array(recorder.luminance), "diagnostic_input_rates": np.array(recorder.input_rates),
            "diagnostic_feature_rates": np.array(recorder.feature_rates),
            "metadata": np.array(json.dumps(meta, allow_nan=False))}
        arrays.update({f"replay_values_{i}": np.array(values) for i, values in enumerate(recorder.channel_values)})
        np.savez_compressed(path, **arrays)
        print(json.dumps({"episode": ep["episode_id"], "intake": meta["ingested_total"], "hunger": meta["final_hunger"],
            "target_transitions": transitions, "seconds": round(meta["wall_seconds"], 2)}), flush=True)
    finally:
        sim.close()


def replay_episode(graph, output, study, ep, condition):
    if condition not in ("replay_check", "homeostasis_off", "transmission_off"):
        raise ValueError("Unknown input replay condition")
    source = _paths(output, "connected", ep)
    path = _paths(output, condition, ep)
    if path.exists():
        raise FileExistsError(path)
    calibration_dir = Path(study["calibration_dir"])
    mapping = load_mapping(calibration_dir / "mapping.json", graph)
    calibration = load_calibration(calibration_dir / "calibration.json", mapping)
    brain = Brain(Path(graph), BrainConfig(**study["brain_config"]), seed=ep["seed"])
    if condition == "transmission_off":
        brain.weights = np.zeros_like(brain.weights)
    features = make_features(mapping, calibration)
    started = time.perf_counter()
    with np.load(source, allow_pickle=False) as archive:
        original_meta = json.loads(str(archive["metadata"]))
        if original_meta["plan_digest"] != study["digest"] or original_meta["episode"] != ep:
            raise ValueError("Replay source provenance mismatch")
        specs = original_meta["stimulus_channels"]
        luminance = archive["replay_luminance"]
        values = [archive[f"replay_values_{i}"] for i in range(len(specs))]
        X = []
        diag_inputs, diag_features = [], []
        for tick in range(len(luminance)):
            channels = {spec["name"]: (np.array(spec["indices"], np.int32), value[tick])
                for spec, value in zip(specs, values) if not (condition == "homeostasis_off" and spec["name"] == "semantic_hunger")}
            _, counts = brain.step(luminance[tick], sensory_currents=channels)
            feature = features.update(counts, 10)
            diag_inputs.append(counts[[r["index"] for r in mapping["ports"]["hunger"]]] * 100.)
            diag_features.append([float(counts[features.indices].mean() * 100), int(np.count_nonzero(counts[features.indices]))])
            if (tick + 1) * 10 % study["sample_ms"] == 0:
                X.append(feature)
        X = np.array(X)
        if condition == "replay_check":
            np.testing.assert_array_equal(X, archive["X"], err_msg="Connected replay must reproduce every physical neural feature exactly")
        meta = {**original_meta, "condition": condition, "body_physics": False,
            "replay_of_sha256": file_digest(source), "replay_scope": study["controls_scope"],
            "exact_original_features": bool(np.array_equal(X, archive["X"])), "wall_seconds": time.perf_counter() - started}
        arrays = {k: archive[k].copy() for k in ("y", "time_ms", "hunger", "intake", "taste", "action")}
        arrays.update(X=X, metadata=np.array(json.dumps(meta, allow_nan=False)),
            diagnostic_input_rates=np.array(diag_inputs), diagnostic_feature_rates=np.array(diag_features))
        np.savez_compressed(path, **arrays)
    print(json.dumps({"episode": ep["episode_id"], "condition": condition,
        "exact_original_features": meta["exact_original_features"], "seconds": round(meta["wall_seconds"], 2)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "collect", "replay"])
    p.add_argument("--graph", type=Path, default=Path("data/graph"))
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--calibration", type=Path, default=CALIBRATION)
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--group", type=int)
    p.add_argument("--episode")
    p.add_argument("--condition", default="replay_check", choices=["replay_check", "homeostasis_off", "transmission_off"])
    a = p.parse_args()
    if a.command == "plan":
        freeze_plan(a.output, a.calibration)
        return
    study = load_plan(a.output)
    for ep in study["episodes"]:
        if ep["split"] != a.split or (a.group is not None and ep["seed_group"] != a.group) or (a.episode is not None and ep["episode_id"] != a.episode):
            continue
        if a.command == "collect":
            collect_episode(a.graph, a.output, study, ep)
        else:
            replay_episode(a.graph, a.output, study, ep, a.condition)


if __name__ == "__main__":
    main()
