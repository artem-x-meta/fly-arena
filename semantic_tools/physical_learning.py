"""Train a neural-only NEED_FOOD head on frozen whole physical episodes.

This offline helper never changes a simulator or consumes its random stream.
Train/validation selection is persisted before any test features are loaded.
Failed, incomplete, missing or identity-mismatched trials fail the operation;
there is deliberately no option to drop a trial and report a passing subset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from fly_semantic.mapping import digest, file_digest
from fly_semantic.protocol import OutputStateMachine
from fly_semantic.readout import LinearReadout
from semantic_tools.state_experiment import metrics
from semantic_tools.transition_metrics import match_need_release


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_plan(study):
    if study.get("digest") != digest({k: v for k, v in study.items() if k != "digest"}):
        raise ValueError("Physical study plan digest mismatch")
    if type(study["sample_ms"]) is not int or study["sample_ms"] <= 0:
        raise ValueError("Invalid physical sampling interval")
    episodes = study["episodes"]
    if not episodes or len({ep["episode_id"] for ep in episodes}) != len(episodes):
        raise ValueError("Duplicate or empty episode identities")
    groups = defaultdict(set)
    for ep in episodes:
        if ep["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown episode split")
        groups[ep["split"]].add(ep["seed_group"])
        duration = ep.get("duration_ms", study.get("duration_ms"))
        if type(duration) is not int or duration <= 0 or duration % study["sample_ms"]:
            raise ValueError("Episode duration must align with sampling")
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Empty physical experiment split")
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Seed groups overlap across whole-episode splits")
    grid = study["l2_grid"]
    if not grid or len(set(grid)) != len(grid) or any(not np.isfinite(v) or v < 0 for v in grid):
        raise ValueError("Invalid predeclared regularization grid")
    if study.get("selection_rule", "indicator_f1_then_raw_f1_then_larger_l2") != "indicator_f1_then_raw_f1_then_larger_l2":
        raise ValueError("Unsupported model selection rule")


def _identity(row):
    meta = row["metadata"]
    return meta["feature_digest"], meta["calibration_digest"], meta["nominal_weights_sha256"], row["X"].shape[1]


def _consistent(rows):
    if not rows or len({_identity(row) for row in rows}) != 1:
        raise ValueError("Mixed or empty feature/calibration/weight identities")


def _teacher_at(transitions, times):
    # match_need_release validates the sequence, including its initial false state.
    output = np.zeros(len(times), dtype=np.int8)
    for item in transitions:
        output[times >= item["time_ms"]] = int(item["active"])
    return output


def load_rows(output, study, split, condition="connected"):
    """Load every planned episode in a split; validate identity and alignment."""
    validate_plan(study)
    if split not in ("train", "validation", "test") or condition not in study["collection_conditions"]:
        raise ValueError("Unknown physical split/condition")
    rows = []
    for ep in study["episodes"]:
        if ep["split"] != split:
            continue
        path = Path(output) / "episodes" / condition / (ep["episode_id"] + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"].item()))
            row = {key: archive[key].copy() for key in ("X", "y", "time_ms", "hunger", "intake")}
        if (meta.get("episode") != ep or meta.get("condition") != condition
                or meta.get("plan_digest") != study["digest"]
                or meta.get("calibration_digest") != study["calibration_digest"]
                or any(not isinstance(meta.get(key), str) or not meta[key] for key in ("feature_digest", "nominal_weights_sha256"))):
            raise ValueError(f"Episode provenance mismatch: {path}")
        expected_sources = study.get("collection_source_hashes", study.get("source_hashes"))
        if not expected_sources or meta.get("source_hashes") != expected_sources:
            raise ValueError(f"Episode source identity mismatch: {path}")
        if study.get("feature_digest") and meta["feature_digest"] != study["feature_digest"]:
            raise ValueError("Episode feature manifest differs from frozen plan")
        if study.get("nominal_weights_sha256") and meta["nominal_weights_sha256"] != study["nominal_weights_sha256"]:
            raise ValueError("Episode nominal graph differs from frozen plan")
        if meta.get("status", "completed") != "completed":
            raise ValueError(f"Failed episode retained as a study failure: {path}")
        duration = ep.get("duration_ms", study.get("duration_ms"))
        times = np.arange(study["sample_ms"], duration + 1, study["sample_ms"], dtype=np.int64)
        n = len(times)
        if (row["X"].ndim != 2 or row["X"].shape[0] != n or not 1 <= row["X"].shape[1] <= 1024
                or not np.isfinite(row["X"]).all()
                or any(row[key].shape != (n,) for key in ("y", "time_ms", "hunger", "intake"))
                or not np.array_equal(row["time_ms"], times) or row["time_ms"].dtype.kind not in "iu"
                or not np.isin(row["y"], (0, 1)).all()
                or not np.isfinite(row["hunger"]).all() or np.any((row["hunger"] < 0) | (row["hunger"] > 1))
                or not np.isfinite(row["intake"]).all() or np.any(row["intake"] < 0) or np.any(np.diff(row["intake"]) < -1e-12)):
            raise ValueError(f"Incomplete or malformed physical episode: {path}")
        transitions = meta["target_transitions"]
        match_need_release(transitions, [], duration_ms=duration)
        if not np.array_equal(row["y"], _teacher_at(transitions, times - study.get("label_reference_offset_ms", 10))):
            raise ValueError(f"Teacher transition/sample alignment mismatch: {path}")
        row.update(metadata=meta, path=str(path), sha256=file_digest(path))
        rows.append(row)
    _consistent(rows)
    return rows


def _machine(study, episode_id):
    return OutputStateMachine(episode_id, supported_concepts=(1,),
        on_threshold=study["indicator_on"], off_threshold=study["indicator_off"],
        confirm_samples=study["confirm_samples"], sample_ms=study["sample_ms"],
        heartbeat_ms=study.get("heartbeat_ms", 1000), event_ttl_ms=study.get("event_ttl_ms", 1500))


def _stable_releases(matches, events, stable_ms):
    """Keep the original first-off match and separately require sustained release."""
    transitions, previous = [], False
    for event in events:
        if event["active"] != previous:
            transitions.append((event["sim_time_ms"], event["active"]))
            previous = event["active"]
    for match in matches:
        found = None
        if match["indicator_active_before_target_off"]:
            for index, (when, active) in enumerate(transitions):
                end = match["inactive_interval_end_ms"]
                if active or when < match["target_off_ms"] or when + stable_ms > end:
                    continue
                next_on = next((time for time, state in transitions[index + 1:] if state), None)
                if next_on is None or next_on >= when + stable_ms:
                    found = when
                    break
        match.update(stable_neural_off_ms=found, stable_latency_ms=None if found is None else found - match["target_off_ms"],
                     stable_release_supported=found is not None, required_stable_ms=stable_ms)
    return matches


def _score_episode(model, row, study):
    if model.feature_digest != row["metadata"]["feature_digest"]:
        raise ValueError("Readout and episode feature identities differ")
    scores = np.asarray(model.predict_scores(row["X"]))
    if scores.shape != (len(row["y"]), 1) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Readout must emit one finite NEED_FOOD score per row")
    scores = scores[:, 0]
    ep = row["metadata"]["episode"]
    machine = _machine(study, ep["episode_id"])
    events, indicated = [], []
    for score, when in zip(scores, row["time_ms"]):
        events.extend(event.to_dict() for event in machine.update({1: float(score)}, int(when)))
        indicated.append(machine.poll_state(int(when))[1]["active"])
    transitions = row["metadata"]["target_transitions"]
    duration = ep.get("duration_ms", study.get("duration_ms"))
    releases = _stable_releases(match_need_release(transitions, events, duration_ms=duration), events, study.get("release_stable_ms", 0))
    # Initial hunger must be detected while that first true interval still lasts.
    starts_high = bool(transitions and transitions[0]["active"] and transitions[0]["time_ms"] == 0)
    first_off = next((item["time_ms"] for item in transitions if not item["active"]), duration + 1)
    first_on = next((event["sim_time_ms"] for event in events if event["active"] and event["sim_time_ms"] < first_off), None)
    false_activations, previous = 0, False
    for event in events:
        if event["active"] and not previous and not _teacher_at(transitions, np.array([event["sim_time_ms"]]))[0]:
            false_activations += 1
        previous = event["active"]
    raw = scores >= study["score_threshold"]
    actual_intake = float(row["intake"][-1])
    feeding = ep.get("expected_release", ep.get("scene") == "physical_feeding")
    sated = ep.get("scene") in ("sated_away", "sated_food")
    released = bool(actual_intake > 0 and releases and all(item["stable_release_supported"] for item in releases))
    return {"episode_id": ep["episode_id"], "seed_group": ep["seed_group"], "scene": ep.get("scene"),
        "condition": row["metadata"]["condition"], "raw": metrics(row["y"], raw),
        "indicator": metrics(row["y"], indicated), "scores": scores.tolist(), "indicator_samples": indicated,
        "targets": row["y"].astype(int).tolist(), "time_ms": row["time_ms"].astype(int).tolist(), "events": events,
        "initial_high": starts_high, "initial_high_detection_ms": first_on if starts_high else None,
        "initial_high_missed": bool(starts_high and first_on is None), "false_activation_events": false_activations,
        "feeding_case": bool(feeding), "sated_case": sated, "intake": actual_intake, "release_matches": releases,
        "false_active_fraction": float(np.mean(np.asarray(indicated)[row["y"] == 0])) if np.any(row["y"] == 0) else None,
        "release_supported": released, "target_transitions": transitions}


def score_rows(model, rows, study):
    """Only X enters the model; body state and event metadata are evaluator data."""
    if not rows:
        raise ValueError("Cannot score an empty planned split")
    cases = [_score_episode(model, row, study) for row in rows]
    groups = defaultdict(list)
    for case in cases:
        groups[str(case["seed_group"])].append(case)

    def summarize(selected):
        truth = np.concatenate([case["targets"] for case in selected])
        raw = np.concatenate([np.array(case["scores"]) >= study["score_threshold"] for case in selected])
        indicator = np.concatenate([case["indicator_samples"] for case in selected])
        feeding = [case for case in selected if case["feeding_case"]]
        delays = [item["stable_latency_ms"] for case in feeding for item in case["release_matches"] if item["stable_release_supported"]]
        high = [case for case in selected if case["initial_high"]]
        sated_false = [case["false_active_fraction"] for case in selected if case["sated_case"] and case["false_active_fraction"] is not None]
        return {"raw": metrics(truth, raw), "indicator": metrics(truth, indicator), "episodes": len(selected),
            "initial_high_cases": len(high), "initial_high_misses": sum(case["initial_high_missed"] for case in high),
            "initial_high_detection_ms": [case["initial_high_detection_ms"] for case in high],
            "false_activation_events": sum(case["false_activation_events"] for case in selected),
            "maximum_false_active_fraction_on_sated": max(sated_false) if sated_false else None,
            "feeding_cases": len(feeding), "feeding_cases_with_intake": sum(case["intake"] > 0 for case in feeding),
            "feeding_cases_with_target_off": sum(bool(case["release_matches"]) for case in feeding),
            "feeding_cases_with_release": sum(case["release_supported"] for case in feeding),
            "release_fraction_all_feeding_cases": sum(case["release_supported"] for case in feeding) / len(feeding) if feeding else None,
            "supported_release_latency_ms": delays,
            "maximum_supported_release_latency_ms": max(delays) if delays else None}

    return summarize(cases) | {"by_seed": {key: summarize(values) for key, values in groups.items()}, "cases": cases}


def paired_seed_bootstrap(a, b, *, seed=914, samples=10000):
    if set(a["by_seed"]) != set(b["by_seed"]) or not a["by_seed"]:
        raise ValueError("Paired comparison requires identical nonempty seed groups")
    keys = sorted(a["by_seed"])
    result = {}
    for metric in ("raw", "indicator"):
        differences = np.array([a["by_seed"][key][metric]["f1"] - b["by_seed"][key][metric]["f1"] for key in keys])
        rng = np.random.default_rng(seed)
        bootstrap = differences[rng.integers(0, len(keys), size=(samples, len(keys)))].mean(axis=1)
        result[metric] = {"mean_seed_f1_difference": float(differences.mean()),
            "paired_seed_bootstrap_95pct": np.quantile(bootstrap, [.025, .975]).tolist(),
            "by_seed_difference": dict(zip(keys, differences.tolist()))}
    return {"seed_groups": len(keys), "bootstrap_samples": samples, "bootstrap_seed": seed, **result}


def _data_manifest(rows):
    return [{"episode_id": row["metadata"]["episode"]["episode_id"], "split": row["metadata"]["episode"]["split"],
             "path": row["path"], "sha256": row["sha256"]} for row in rows]


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: file_digest(root / name) for name in ("semantic_tools/physical_learning.py", "semantic_tools/transition_metrics.py",
        "semantic_tools/state_experiment.py", "fly_semantic/readout.py", "fly_semantic/protocol.py")}


def train(output, study, synthetic_rows=()):
    """Fit a finite predeclared L2 grid; select on validation, never load test."""
    output = Path(output)
    validate_plan(study)
    if (output / "state-readout.npz").exists() or (output / "selection.json").exists():
        raise FileExistsError("Readout already frozen; use a new study for retraining")
    train_rows = load_rows(output, study, "train")
    validation = load_rows(output, study, "validation")
    synthetic_rows = list(synthetic_rows)
    if bool(synthetic_rows) != bool(study.get("include_synthetic_train", False)):
        raise ValueError("Synthetic training inclusion must be predeclared")
    if any(row["metadata"]["episode"]["split"] != "train" for row in synthetic_rows):
        raise ValueError("Synthetic validation/test leakage into training")
    all_train = train_rows + synthetic_rows
    _consistent(all_train + validation)
    X = np.concatenate([row["X"] for row in all_train])
    y = np.concatenate([row["y"] for row in all_train])
    candidates = []
    for l2 in study["l2_grid"]:
        model = LinearReadout.fit(X, y, feature_digest=_identity(train_rows[0])[0], l2=l2,
            max_iter=study["max_iter"], seed=study.get("training_seed", 42))
        name = f"l2-{l2:g}"
        model_path = output / "candidates" / (name + ".npz")
        if model_path.exists():
            raise FileExistsError("Candidate already exists; incomplete training must use a fresh output directory")
        model.save(model_path)
        scored = score_rows(model, validation, study)
        result = {"l2": float(l2), "model_path": str(model_path), "model_sha256": file_digest(model_path),
            "diagnostics": model.diagnostics, "validation": scored}
        write_json(output / "candidates" / (name + "-validation.json"), result)
        candidates.append(result)
    selected = max(candidates, key=lambda row: (row["validation"]["indicator"]["f1"], row["validation"]["raw"]["f1"], row["l2"]))
    destination = output / "state-readout.npz"
    with destination.open("xb") as stream:
        stream.write(Path(selected["model_path"]).read_bytes())
    model = LinearReadout.load(destination, expected_feature_digest=_identity(train_rows[0])[0])
    manifest = {"schema_version": 1, "plan_digest": study["digest"], "model_sha256": file_digest(destination),
        "feature_digest": model.feature_digest, "calibration_digest": _identity(train_rows[0])[1],
        "nominal_weights_sha256": _identity(train_rows[0])[2], "selected_l2": selected["l2"],
        "selection_rule": "indicator_f1_then_raw_f1_then_larger_l2", "test_seen": False,
        "train_data": _data_manifest(all_train), "validation_data": _data_manifest(validation),
        "source_hashes": _source_hashes(), "candidates": [{k: v for k, v in c.items() if k != "validation"} for c in candidates],
        "train_positive_fraction": float(y.mean()), "trained_parameters": int(model.weights.size + model.bias.size),
        "diagnostics": model.diagnostics, "validation": selected["validation"], "train": score_rows(model, train_rows, study)}
    manifest["digest"] = digest(manifest)
    write_json(output / "selection.json", manifest)
    write_json(output / "training.json", {"plan_digest": study["digest"], "selection_digest": manifest["digest"],
        "model_sha256": manifest["model_sha256"], "test_seen": False, "selected_l2": selected["l2"],
        "diagnostics": model.diagnostics, "trained_parameters": manifest["trained_parameters"],
        "train": manifest["train"], "validation": manifest["validation"], "source_hashes": manifest["source_hashes"]})
    return manifest


def _verify_selection(output, study):
    path = Path(output) / "selection.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("digest") != digest({k: v for k, v in value.items() if k != "digest"})
            or value["plan_digest"] != study["digest"] or value["test_seen"] is not False
            or value["source_hashes"] != _source_hashes()
            or value["model_sha256"] != file_digest(Path(output) / "state-readout.npz")):
        raise ValueError("Frozen selection, source, plan or model changed before test")
    for entry in value["train_data"] + value["validation_data"]:
        if file_digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Training/validation data changed after selection")
    return value


def _shuffle_rows(rows, seed):
    # Diagnostic null: one entire donor per target; align normalized episode time.
    # Resampling is monotone nearest-neighbor, not a new physical neural replay.
    if len(rows) < 2:
        raise ValueError("Cannot shuffle a single episode")
    rng = np.random.default_rng(seed)
    permutation = np.empty(len(rows), dtype=int)
    ordered = rng.permutation(len(rows))
    permutation[ordered] = np.roll(ordered, 1)
    shuffled = []
    for row, donor in zip(rows, permutation):
        source = rows[int(donor)]["X"]
        indices = np.rint(np.linspace(0, len(source) - 1, len(row["X"]))).astype(int)
        shuffled.append(dict(row, X=source[indices]))
    return shuffled, permutation.tolist()


class _ConstantHead:
    def __init__(self, feature_digest, score):
        self.feature_digest, self.score = feature_digest, score

    def predict_scores(self, X):
        return np.full((len(X), 1), self.score)


def assess_acceptance(connected, controls, study, baseline_connected=None):
    """Apply only thresholds explicitly frozen in the study, never hidden defaults."""
    limits = study["acceptance"]
    allowed = {"raw_f1_min", "indicator_f1_min", "target_f1", "indicator_target_f1", "release_required_fraction", "release_max_latency_ms", "release_stable_ms",
               "initial_high_misses_max", "false_activation_events_max", "control_gain_min",
               "maximum_false_active_fraction_on_sated", "baseline_improvement_min"}
    if set(limits) - allowed:
        raise ValueError("Unknown frozen acceptance criterion")
    checks = {}
    for key, mode in (("raw_f1_min", "raw"), ("indicator_f1_min", "indicator"), ("target_f1", "raw"), ("indicator_target_f1", "indicator")):
        if key in limits:
            checks[key] = connected[mode]["f1"] >= limits[key]
    if "release_required_fraction" in limits:
        value = connected["release_fraction_all_feeding_cases"]
        checks["release_required_fraction"] = value is not None and value >= limits["release_required_fraction"]
    if "release_stable_ms" in limits and limits["release_stable_ms"] != study.get("release_stable_ms", 0):
        raise ValueError("Release scoring duration differs from frozen acceptance")
    if "release_max_latency_ms" in limits:
        value = connected["maximum_supported_release_latency_ms"]
        checks["release_max_latency_ms"] = value is not None and value <= limits["release_max_latency_ms"]
    if "maximum_false_active_fraction_on_sated" in limits:
        value = connected["maximum_false_active_fraction_on_sated"]
        checks["maximum_false_active_fraction_on_sated"] = value is not None and value <= limits["maximum_false_active_fraction_on_sated"]
    if "baseline_improvement_min" in limits:
        if baseline_connected is None:
            raise ValueError("Frozen baseline improvement criterion requires baseline evaluation")
        checks["baseline_improvement_min"] = connected["indicator"]["f1"] - baseline_connected["indicator"]["f1"] >= limits["baseline_improvement_min"]
    for key, metric in (("initial_high_misses_max", "initial_high_misses"), ("false_activation_events_max", "false_activation_events")):
        if key in limits:
            checks[key] = connected[metric] <= limits[key]
    if "control_gain_min" in limits:
        for name, requirement in limits["control_gain_min"].items():
            checks["control_gain_min:" + name] = connected["raw"]["f1"] - controls[name]["raw"]["f1"] >= requirement
    return {"checks": checks, "status": "PRELIMINARY_TARGET_REACHED" if checks and all(checks.values()) else "TARGET_NOT_REACHED"}


def evaluate(output, study, baseline_model_path):
    output = Path(output)
    validate_plan(study)
    if (output / "results.json").exists():
        raise FileExistsError("Held-out evaluation already recorded; do not overwrite")
    selected = _verify_selection(output, study)  # This precedes any test loading.
    if file_digest(baseline_model_path) != study["baseline_model_sha256"]:
        raise ValueError("Historical paired baseline model changed")
    test = load_rows(output, study, "test")
    model = LinearReadout.load(output / "state-readout.npz", expected_feature_digest=selected["feature_digest"])
    baseline = LinearReadout.load(baseline_model_path, expected_feature_digest=selected["feature_digest"])
    recorded = {"connected": test}
    for condition in study["collection_conditions"]:
        if condition != "connected":
            recorded[condition] = load_rows(output, study, "test", condition)
    _consistent([row for rows in recorded.values() for row in rows])
    if _identity(test[0])[:3] != (selected["feature_digest"], selected["calibration_digest"], selected["nominal_weights_sha256"]):
        raise ValueError("Held-out feature/graph identities differ from training")
    shuffled, permutation = _shuffle_rows(test, study.get("shuffle_seed", 81817))
    conditions = recorded | {"cross_episode_shuffle": shuffled}
    candidate_scores = {key: score_rows(model, rows, study) for key, rows in conditions.items()}
    baseline_scores = {key: score_rows(baseline, rows, study) for key, rows in conditions.items()}
    constant = _ConstantHead(model.feature_digest, float(selected["train_positive_fraction"] >= .5))
    candidate_scores["train_constant"] = score_rows(constant, test, study)
    acceptance = assess_acceptance(candidate_scores["connected"], candidate_scores, study, baseline_scores["connected"])
    result = {"schema_version": 1, **acceptance, "plan_digest": study["digest"], "selection_digest": selected["digest"],
        "model_sha256": selected["model_sha256"], "baseline_model_sha256": file_digest(baseline_model_path),
        "conditions": candidate_scores, "baseline_conditions": baseline_scores,
        "paired_candidate_minus_baseline": {key: paired_seed_bootstrap(candidate_scores[key], baseline_scores[key]) for key in conditions},
        "paired_connected_minus_control": {key: paired_seed_bootstrap(candidate_scores["connected"], value)
            for key, value in candidate_scores.items() if key != "connected"},
        "shuffle_permutation": permutation, "test_data": _data_manifest([row for rows in recorded.values() for row in rows]),
        "shuffle_method": "Deranged whole-episode donors with monotone nearest-neighbor alignment of normalized episode time; diagnostic, not a physical replay or an acceptance criterion",
        "test_tuned": False, "all_planned_episodes_included": True, "source_hashes": _source_hashes(),
        "scope": study.get("scope", "Preliminary fixed physical episodes; state head has no motor authority"),
        "release_denominator": "All planned expected_release cases, including no-intake, missing true release, and never-active indicators; unmatched latency remains null"}
    write_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path)
    args = parser.parse_args()
    study = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
    if args.command == "train":
        result = train(args.output, study)
    else:
        if args.baseline_model is None:
            parser.error("evaluate requires --baseline-model")
        result = evaluate(args.output, study, args.baseline_model)
    print(json.dumps({key: result[key] for key in ("plan_digest", "model_sha256", "status", "selected_l2") if key in result}), flush=True)


if __name__ == "__main__":
    main()
