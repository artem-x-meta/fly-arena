"""Frozen-core NEED_FOOD assay; body-only labels never reach the readout."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from fly_arena.brain import Brain
from fly_arena.neural_ports import NeuralAdapter
from fly_arena.organism import Organism
from fly_arena.sensors import SensorFrame
from fly_semantic.mapping import digest, file_digest, load_mapping
from fly_semantic.runtime import load_calibration, make_features
from fly_semantic.readout import LinearReadout
from fly_semantic.protocol import OutputStateMachine


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_plan(output):
    study = json.loads((Path(output) / "plan.json").read_text())
    value = dict(study)
    if value.pop("digest", None) != digest(value) or value.get("schema_version") != 1:
        raise ValueError("Experiment plan digest/schema mismatch")
    if (type(study["duration_ms"]) is not int or type(study["sample_ms"]) is not int
            or study["sample_ms"] <= 0 or study["sample_ms"] % 10
            or study["duration_ms"] <= 0 or study["duration_ms"] % study["sample_ms"]):
        raise ValueError("Plan duration/sampling must align with 10ms integration")
    if (not 0 <= study["teacher_off"] < study["teacher_on"] <= 1
            or not 0 <= study["indicator_off"] < study["indicator_on"] <= 1
            or not 0 <= study["score_threshold"] <= 1
            or type(study["teacher_initial_active"]) is not bool
            or type(study["confirm_samples"]) is not int or study["confirm_samples"] < 1):
        raise ValueError("Invalid frozen teacher/readout thresholds")
    groups = study["seed_groups"]
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Plan must declare train/validation/test seed groups")
    seeds = [seed for entries in groups.values() for seed in entries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed groups overlap across splits")
    episodes = study["episodes"]
    for name in ("episode_id", "neural_seed", "background_seed"):
        values = [ep[name] for ep in episodes]
        if len(set(values)) != len(values):
            raise ValueError(f"Repeated episode identity/stream: {name}")
    if any(ep["split"] not in groups or ep["seed_group"] not in groups[ep["split"]] for ep in episodes):
        raise ValueError("Episode assigned to a mismatched split/seed group")
    if any(not any(ep["split"] == split for ep in episodes) for split in groups):
        raise ValueError("Empty experiment split")
    return study


def source_audit():
    """Current files only; no retrospective claim about completed collection."""
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "fly_arena" / "brain.py",
             root / "fly_arena" / "organism.py", root / "fly_arena" / "neural_ports.py"]
    paths.extend((root / "fly_semantic").glob("*.py"))
    return {"files": {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(paths)},
        "scope": "Files hashed at this operation; earlier pilot collection may predate the provenance checks. These hashes do not retroactively attest its loaded source.",
        "recipe_correction": "Read thresholds/sampling from frozen plan; original pilot numeric settings unchanged."}


def stage_b_calibration(source, destination):
    """Annotate stored calibration evidence honestly without rewriting its source."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "calibration.json").read_text())
    result = dict(original)
    result.pop("digest")
    vectors = [np.array(r["feature_mean_hz"]) for r in result["cue_probes"]]
    distinct = len(vectors) == 3 and all(not np.array_equal(vectors[i], vectors[j]) for i in range(3) for j in range(i))
    result.update(hunger_status="CALIBRATED", cue_status="UNVALIDATED" if distinct else "TARGET_NOT_REACHED",
        cue_gain_mv=None, derived_from_sha256=file_digest(source / "calibration.json"),
        scope="Stage B hunger transfer only; cue gain disabled until separate adequate calibration")
    result["digest"] = digest(result)
    write_json(destination / "calibration.json", result)
    (destination / "mapping.json").write_bytes((source / "mapping.json").read_bytes())


def plan(output, calibration):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists():
        raise FileExistsError("Study already frozen; reuse collect/train/evaluate commands")
    episodes = []
    groups = {"train": list(range(110, 114)), "validation": [210, 211], "test": list(range(310, 315))}
    for split, seeds in groups.items():
        for seed in seeds:
            for index in range(8 if split == "train" else 4):
                # Independent child streams keep background independent of label side.
                rng = np.random.default_rng(np.random.SeedSequence([seed, index, 901]))
                high = index % 2 == 1
                need = float(rng.uniform(.72, .94) if high else rng.uniform(.08, .64))
                gut_fraction = float(rng.uniform(0, min(.18, .98 - need)))
                energy = 100 * (1 - need / (1 - gut_fraction))
                initial = {"energy": energy, "sleep_pressure": float(rng.uniform(0, .95)),
                    "dust_by_region": {r: float(rng.uniform(0, .8)) for r in
                        ("head", "antenna_left", "antenna_right", "front_left", "front_right")},
                    "gut_batches": [{"food_id": "initial_store", "amount": 10 * gut_fraction, "energy_density": 12.}]}
                episodes.append({"episode_id": f"{split}-{seed}-{index}", "split": split,
                    "seed_group": seed, "neural_seed": seed * 100 + index,
                    "background_seed": seed * 1000 + index, "initial": initial})
    result = {"schema_version": 1, "scope": "preliminary Stage B; full LIF + existing organism, synthetic local sensory backgrounds; no physical body in training assay",
        "calibration_digest": calibration["digest"], "duration_ms": 1500, "sample_ms": 100,
        "teacher_on": .7, "teacher_off": .55, "teacher_initial_active": False,
        "l2": .001, "max_iter": 500, "score_threshold": .5,
        "indicator_on": .7, "indicator_off": .4, "confirm_samples": 2,
        "target_f1": .8, "control_gain_min": .15,
        "organism_config": {"life_time_scale": 60.},
        "background": "independent scalar luminance .05..95 every100ms, fixed per-eye contrast; independent sleep/dust; zero taste to test need away from food",
        "conditions": ["connected", "homeostasis_off", "transmission_off", "cross_episode_shuffle", "train_constant"],
        "episodes": episodes, "seed_groups": groups,
        "limitations": "60 short episodes total, 5 test groups x4 episodes; finite pilot, no final100/seed claim; physical transfer measured separately"}
    result["digest"] = digest(result)
    write_json(path, result)
    return result


def collect(graph, calibration_dir, output, split, condition="connected"):
    output = Path(output)
    study = load_plan(output)
    mapping = load_mapping(Path(calibration_dir) / "mapping.json", graph)
    calibration = load_calibration(Path(calibration_dir) / "calibration.json", mapping)
    if calibration["digest"] != study["calibration_digest"]:
        raise ValueError("Calibration changed after freezing experiment")
    dest = output / "episodes" / condition
    dest.mkdir(parents=True, exist_ok=True)
    brain = Brain(Path(graph))
    frozen_weights_sha = __import__("hashlib").sha256(memoryview(brain.weights)).hexdigest()
    initial_brain = brain.get_state()
    if condition == "transmission_off":
        # Explicit local ablation, not an edit to graph arrays/normal model weights.
        brain.weights = np.zeros_like(brain.weights)
    elif condition not in ("connected", "homeostasis_off"):
        raise ValueError("Unknown collection condition")
    registry = json.loads((Path(graph) / "behavior_ports.json").read_text())
    hunger_indices = np.array([r["index"] for r in mapping["ports"]["hunger"]], np.int32)
    for ep in [e for e in study["episodes"] if e["split"] == split]:
        target = dest / (ep["episode_id"] + ".npz")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite episode {target}")
        started = time.perf_counter()
        brain.set_state(initial_brain)
        brain.voltage[:] = np.random.default_rng(ep["neural_seed"]).uniform(0, 2, len(brain.ids)).astype(np.float32)
        adapter = NeuralAdapter(brain, registry)
        features = make_features(mapping, calibration)
        body = Organism(study["organism_config"], ep["initial"])
        rng = np.random.default_rng(ep["background_seed"])
        active = study["teacher_initial_active"]
        X, y, hunger, times = [], [], [], []
        frame = SensorFrame(dust_afferents=ep["initial"]["dust_by_region"], tarsal_taste={})
        eye_contrast = rng.uniform(-.04, .04, 2)
        for tick in range(study["duration_ms"] // 10):
            if tick % 10 == 0:
                luminance = np.clip(rng.uniform(.05, .95) + eye_contrast[brain.eyes], 0, 1).astype(np.float32)
                movement = float(rng.uniform(0, 1.))
            need = body.hunger
            active = bool(need >= study["teacher_on"] or (active and need > study["teacher_off"]))
            sensory, modulation = adapter.currents(frame, body)
            if condition != "homeostasis_off":
                sensory["semantic_hunger"] = (hunger_indices, calibration["hunger_gain_mv"] * need)
            _, counts = brain.step(luminance, sensory_currents=sensory, modulation=modulation)
            values = features.update(counts, 10)
            body.advance(.01, "WALK", movement_proxy=movement)
            if (tick + 1) % (study["sample_ms"] // 10) == 0:
                X.append(values)
                y.append(active)
                hunger.append(need)
                times.append(brain.clock)
        metadata = {"plan_digest": study["digest"], "feature_digest": features.feature_digest,
            "calibration_digest": calibration["digest"], "episode": ep, "condition": condition,
            "nominal_weights_sha256": frozen_weights_sha, "wall_seconds": time.perf_counter() - started,
            "unit": "real full LIF and actual Organism dynamics, synthetic sensory background, no MuJoCo body"}
        np.savez_compressed(target, X=np.array(X), y=np.array(y, np.int8), hunger=np.array(hunger),
            time_ms=np.array(times), metadata=np.array(json.dumps(metadata, allow_nan=False)))
        print(json.dumps({"episode": ep["episode_id"], "condition": condition, "positive": int(sum(y)),
            "seconds": round(metadata["wall_seconds"], 3)}), flush=True)


def load_episodes(output, split, condition="connected"):
    output = Path(output)
    study = load_plan(output)
    if split not in study["seed_groups"] or condition not in ("connected", "homeostasis_off", "transmission_off"):
        raise ValueError("Unknown episode split/condition")
    rows = []
    for ep in study["episodes"]:
        if ep["split"] != split:
            continue
        path = output / "episodes" / condition / (ep["episode_id"] + ".npz")
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
            if (meta["plan_digest"] != study["digest"] or meta["episode"] != ep or meta["condition"] != condition
                    or meta.get("calibration_digest") != study["calibration_digest"]
                    or not isinstance(meta.get("feature_digest"), str) or not meta["feature_digest"]
                    or not isinstance(meta.get("nominal_weights_sha256"), str) or not meta["nominal_weights_sha256"]):
                raise ValueError("Episode provenance mismatch")
            row = {k: a[k].copy() for k in ("X", "y", "hunger", "time_ms")} | {"metadata": meta}
            times = np.arange(study["sample_ms"], study["duration_ms"] + 1, study["sample_ms"])
            if (row["X"].ndim != 2 or row["X"].shape[0] != len(times) or not 1 <= row["X"].shape[1] <= 1024
                    or not np.isfinite(row["X"]).all() or row["y"].shape != times.shape
                    or not np.isin(row["y"], (0, 1)).all() or row["hunger"].shape != times.shape
                    or not np.isfinite(row["hunger"]).all() or np.any((row["hunger"] < 0) | (row["hunger"] > 1))
                    or not np.array_equal(row["time_ms"], times)):
                raise ValueError("Episode feature/label/time alignment mismatch")
            rows.append(row)
    if len({(r["metadata"]["feature_digest"], r["metadata"]["nominal_weights_sha256"], r["X"].shape[1]) for r in rows}) != 1:
        raise ValueError("Mixed episode feature/weight identities")
    return study, rows


def metrics(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if (truth.ndim != 1 or truth.size == 0 or predicted.shape != truth.shape
            or not np.isin(truth, (0, 1)).all() or not np.isin(predicted, (0, 1)).all()):
        raise ValueError("Metrics require aligned nonempty binary vectors")
    truth, predicted = truth.astype(bool), predicted.astype(bool)
    tp = int(np.sum(truth & predicted)); fp = int(np.sum(~truth & predicted)); fn = int(np.sum(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"precision": precision, "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        "accuracy": float(np.mean(truth == predicted)), "silence_fraction": float(np.mean(~predicted)),
        "tp": tp, "fp": fp, "fn": fn, "samples": len(truth)}


def score_rows(model, rows, study=None):
    settings = {"score_threshold": .5, "indicator_on": .7, "indicator_off": .4,
                "confirm_samples": 2, "sample_ms": 100} | (study or {})
    truth, pred, indicated, delays = [], [], [], []
    groups = {}
    for row in rows:
        if row["metadata"]["feature_digest"] != model.feature_digest:
            raise ValueError("Scored episode and model feature identities differ")
        scores = model.predict_scores(row["X"])[:, 0]
        machine = OutputStateMachine(row["metadata"]["episode"]["episode_id"], supported_concepts=(1,),
            on_threshold=settings["indicator_on"], off_threshold=settings["indicator_off"],
            confirm_samples=settings["confirm_samples"], sample_ms=settings["sample_ms"])
        outputs = []
        ontime = None
        for s, t in zip(scores, row["time_ms"]):
            events = machine.update({1: float(s)}, int(t))
            cached = machine.poll_state(int(t))
            outputs.append(cached[1]["active"])
            if any(e.active for e in events) and ontime is None:
                ontime = int(t)
        if row["y"][0]:
            delays.append(ontime)
        truth.extend(row["y"]); pred.extend(scores >= settings["score_threshold"]); indicated.extend(outputs)
        seed = str(row["metadata"]["episode"]["seed_group"])
        entry = groups.setdefault(seed, {"y": [], "pred": [], "indicator": []})
        entry["y"].extend(row["y"].tolist()); entry["pred"].extend((scores >= settings["score_threshold"]).tolist()); entry["indicator"].extend(outputs)
    return {"raw": metrics(truth, pred), "indicator": metrics(truth, indicated),
        "initial_high_detection_ms": delays, "initial_high_misses": sum(t is None for t in delays),
        "by_seed": {k: {"raw": metrics(v["y"], v["pred"]), "indicator": metrics(v["y"], v["indicator"])} for k, v in groups.items()}}


def train(output):
    output = Path(output)
    study, rows = load_episodes(output, "train")
    if (output / "state-readout.npz").exists():
        raise FileExistsError("Model frozen; use a new study directory to retrain")
    features = {r["metadata"]["feature_digest"] for r in rows}
    if len(features) != 1:
        raise ValueError("Mixed feature manifests")
    _, validation = load_episodes(output, "validation")
    if (validation[0]["metadata"]["feature_digest"] not in features
            or validation[0]["metadata"]["nominal_weights_sha256"] != rows[0]["metadata"]["nominal_weights_sha256"]):
        raise ValueError("Training and validation identities differ")
    X = np.concatenate([r["X"] for r in rows]); y = np.concatenate([r["y"] for r in rows])
    model = LinearReadout.fit(X, y, feature_digest=features.pop(), l2=study["l2"], max_iter=study["max_iter"])
    model.save(output / "state-readout.npz")
    result = {"plan_digest": study["digest"], "model_sha256": file_digest(output / "state-readout.npz"),
        "diagnostics": model.diagnostics, "train": score_rows(model, rows, study), "validation": score_rows(model, validation, study),
        "test_seen": False, "trained_parameters": model.weights.size + model.bias.size,
        "train_positive_fraction": float(y.mean()), "source_audit": source_audit()}
    write_json(output / "training.json", result)
    print(json.dumps(result), flush=True)


def evaluate(output):
    output = Path(output)
    study, rows = load_episodes(output, "test")
    model = LinearReadout.load(output / "state-readout.npz", expected_feature_digest=rows[0]["metadata"]["feature_digest"])
    training = json.loads((output / "training.json").read_text())
    if (training["plan_digest"] != study["digest"]
            or training["model_sha256"] != file_digest(output / "state-readout.npz")):
        raise ValueError("Model or plan changed since training")
    results = {"connected": score_rows(model, rows, study)}
    for condition in ("homeostasis_off", "transmission_off"):
        _, control = load_episodes(output, "test", condition)
        if control[0]["metadata"]["nominal_weights_sha256"] != rows[0]["metadata"]["nominal_weights_sha256"]:
            raise ValueError("Control nominal graph weights differ")
        results[condition] = score_rows(model, control, study)
    # Uniform independent permutation of whole episodes, preserving within-episode trace order.
    permutation = np.random.default_rng(81817).permutation(len(rows))
    shuffled = [dict(row, X=rows[int(i)]["X"]) for row, i in zip(rows, permutation)]
    results["cross_episode_shuffle"] = score_rows(model, shuffled, study)
    constant = training["train_positive_fraction"] >= .5
    truth = np.concatenate([r["y"] for r in rows])
    results["train_constant"] = metrics(truth, np.full(len(truth), constant))
    differences = {}
    for condition in ("homeostasis_off", "transmission_off", "cross_episode_shuffle"):
        a, b = results["connected"]["by_seed"], results[condition]["by_seed"]
        delta = np.array([a[k]["raw"]["f1"] - b[k]["raw"]["f1"] for k in a])
        rng = np.random.default_rng(914)
        bootstrap = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
        differences[condition] = {"mean_seed_f1_difference": float(delta.mean()),
            "paired_seed_bootstrap_95pct": np.quantile(bootstrap, [.025, .975]).tolist(), "seed_groups": len(delta)}
    quality = results["connected"]["raw"]["f1"] >= study["target_f1"]
    indicator_quality = results["connected"]["indicator"]["f1"] >= study["target_f1"]
    gains = all(results["connected"]["raw"]["f1"] - results[c]["raw"]["f1"] >= study["control_gain_min"]
                for c in ("homeostasis_off", "cross_episode_shuffle"))
    final = {"schema_version": 1, "scope": study["scope"], "status": "PRELIMINARY_TARGET_REACHED" if quality and gains else "TARGET_NOT_REACHED",
        "quality_target_reached": quality, "control_gain_target_reached": gains,
        "indicator_quality_target_reached": indicator_quality,
        "live_output_ready": bool(quality and gains and indicator_quality),
        "acceptance_scope": "Frozen primary target uses raw scores at the plan threshold; indicator/live readiness is separately reported and cannot upgrade a failed primary target. Physical transfer remains a separate check.",
        "plan_digest": study["digest"], "model_sha256": file_digest(output / "state-readout.npz"),
        "conditions": results, "paired_differences": differences, "shuffle_permutation": permutation.tolist(),
        "limitations": study["limitations"], "test_tuned": False, "source_audit": source_audit()}
    write_json(output / "results.json", final)
    print(json.dumps(final), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["derive-calibration", "plan", "collect", "train", "evaluate"])
    p.add_argument("--graph", type=Path, default=Path("data/graph"))
    p.add_argument("--calibration", type=Path, default=Path("runs/semantic-v1/calibration-stage-b"))
    p.add_argument("--output", type=Path, default=Path("runs/semantic-v1/state-study"))
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--condition", choices=["connected", "homeostasis_off", "transmission_off"], default="connected")
    args = p.parse_args()
    if args.command == "derive-calibration":
        stage_b_calibration("runs/semantic-v1/calibration-low", args.calibration)
    elif args.command == "plan":
        mapping = load_mapping(args.calibration / "mapping.json", args.graph)
        plan(args.output, load_calibration(args.calibration / "calibration.json", mapping))
    elif args.command == "collect":
        collect(args.graph, args.calibration, args.output, args.split, args.condition)
    elif args.command == "train":
        train(args.output)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
