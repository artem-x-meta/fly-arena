"""Read-only TRAIN diagnostics for the physical semantic v2 collection.

This helper cannot open validation/test episodes or fit/change a readout. Body
labels, hunger and direct input rates are evaluator diagnostics only. The frozen
v1 head receives X alone. Requested groups must be complete; incomplete or
identity-mismatched files fail, and reports are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from fly_semantic.mapping import digest, file_digest
from fly_semantic.protocol import OutputStateMachine
from fly_semantic.readout import LinearReadout
from semantic_tools.physical_collection import OUTPUT, load_plan
from semantic_tools.state_experiment import metrics


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return None if not values.size else {
        "count": int(values.size), "mean": float(values.mean()),
        "minimum": float(values.min()), "maximum": float(values.max())}


def _metric_pair(targets, raw, indicator, mask):
    if not np.any(mask):
        return {"samples": 0, "raw": None, "indicator": None}
    return {"samples": int(np.count_nonzero(mask)),
        "raw": metrics(targets[mask], raw[mask]),
        "indicator": metrics(targets[mask], indicator[mask])}


def _load_episode(path, ep, study):
    with np.load(path, allow_pickle=False) as a:
        row = {key: a[key].copy() for key in ("X", "y", "time_ms", "hunger", "intake", "taste", "action",
            "diagnostic_input_rates", "diagnostic_feature_rates")}
        meta = json.loads(str(a["metadata"].item()))
    if (meta.get("episode") != ep or ep["split"] != "train" or meta.get("condition") != "connected"
            or meta.get("plan_digest") != study["digest"] or meta.get("source_hashes") != study["source_hashes"]
            or meta.get("feature_digest") != study["feature_digest"]
            or meta.get("calibration_digest") != study["calibration_digest"] or meta.get("body_physics") is not True):
        raise ValueError(f"TRAIN episode provenance mismatch: {path}")
    n, ticks = ep["duration_ms"] // study["sample_ms"], ep["duration_ms"] // 10
    if (row["X"].shape != (n, 1024) or row["diagnostic_input_rates"].shape != (ticks, 32)
            or row["diagnostic_feature_rates"].shape != (ticks, 2)
            or any(row[k].shape != (n,) for k in ("y", "time_ms", "hunger", "intake", "taste", "action"))
            or not np.array_equal(row["time_ms"], np.arange(study["sample_ms"], ep["duration_ms"] + 1, study["sample_ms"]))
            or not np.isin(row["y"], (0, 1)).all()):
        raise ValueError(f"Incomplete or malformed TRAIN episode: {path}")
    for key in ("X", "hunger", "intake", "taste", "diagnostic_input_rates", "diagnostic_feature_rates"):
        if not np.isfinite(row[key]).all() or np.any(row[key] < 0):
            raise ValueError(f"Invalid numeric TRAIN diagnostics: {path}: {key}")
    if np.any(row["hunger"] > 1) or np.any(row["diagnostic_feature_rates"][:, 1] > 512):
        raise ValueError(f"Out-of-range TRAIN diagnostics: {path}")
    teacher = np.zeros(n, dtype=np.int8)
    for item in meta["target_transitions"]:
        teacher[row["time_ms"] - study["label_reference_offset_ms"] >= item["time_ms"]] = int(item["active"])
    if not np.array_equal(teacher, row["y"]):
        raise ValueError(f"TRAIN teacher timing mismatch: {path}")
    return row | {"metadata": meta, "path": path.as_posix(), "sha256": file_digest(path)}


def _window(row, scores, indicator, *, start_ms, stop_ms):
    sample_mask = (row["time_ms"] > start_ms) & (row["time_ms"] <= stop_ms)
    tick_times = np.arange(1, len(row["diagnostic_input_rates"]) + 1) * 10
    tick_mask = (tick_times > start_ms) & (tick_times <= stop_ms)
    direct = row["diagnostic_input_rates"][tick_mask]
    selected = row["diagnostic_feature_rates"][tick_mask]
    return {"start_exclusive_ms": start_ms, "end_inclusive_ms": stop_ms,
        "sample_count": int(sample_mask.sum()), "neural_bin_count": int(tick_mask.sum()),
        "hunger": _stats(row["hunger"][sample_mask]), "baseline_score": _stats(scores[sample_mask]),
        "baseline_active_fraction": float(indicator[sample_mask].mean()) if sample_mask.any() else None,
        "input_mean_hz": float(direct.mean()) if direct.size else None,
        "input_active_fraction_per_10ms_bin": float((direct > 0).mean()) if direct.size else None,
        "selected_mean_hz": float(selected[:, 0].mean()) if len(selected) else None,
        "selected_active_fraction_per_10ms_bin": float(selected[:, 1].mean() / 512) if len(selected) else None,
        "trace_mean_hz": float(row["X"][sample_mask].mean()) if sample_mask.any() else None,
        "taste": _stats(row["taste"][sample_mask]),
        "actions": dict(Counter(row["action"][sample_mask].tolist()))}


def analyze_episode(row, study, model):
    # This is the only prediction call. No label, time, state or input rate enters.
    scores = model.predict_scores(row["X"])[:, 0]
    machine = OutputStateMachine(row["metadata"]["episode"]["episode_id"], supported_concepts=(1,),
        on_threshold=study["indicator_on"], off_threshold=study["indicator_off"],
        confirm_samples=study["confirm_samples"], sample_ms=study["sample_ms"])
    events, indicator = [], []
    for score, when in zip(scores, row["time_ms"]):
        events.extend(e.to_dict() for e in machine.update({1: float(score)}, int(when)))
        indicator.append(machine.poll_state(int(when))[1]["active"])
    indicator = np.asarray(indicator, bool)
    raw = scores >= study["score_threshold"]
    need = row["hunger"]
    masks = {"low_at_or_below_teacher_off": need <= study["teacher_off"],
        "history_dependent_band": (need > study["teacher_off"]) & (need < study["teacher_on"]),
        "high_at_or_above_teacher_on": need >= study["teacher_on"]}
    ep = row["metadata"]["episode"]
    windows = {"initial_500ms": _window(row, scores, indicator, start_ms=0, stop_ms=500),
        "last_500ms": _window(row, scores, indicator, start_ms=ep["duration_ms"] - 500, stop_ms=ep["duration_ms"])}
    for index, item in enumerate(t for t in row["metadata"]["target_transitions"] if not t["active"]):
        when = item["time_ms"]
        for name, start, stop in (("before_release_500ms", max(0, when - 500), when),
                                  ("after_release_500ms", when, min(ep["duration_ms"], when + 500)),
                                  ("settled_after_release", when + 500, min(ep["duration_ms"], when + 1500))):
            windows[f"release_{index}_{name}"] = _window(row, scores, indicator, start_ms=start, stop_ms=stop)
    contributions = ((row["X"] - model.mean) / model.scale) * model.weights[:, 0]
    mean_contributions = contributions.mean(axis=0)
    top_indices = np.argsort(-np.abs(mean_contributions), kind="stable")[:10]
    return {"episode_id": ep["episode_id"], "seed_group": ep["seed_group"], "scene": ep["scene"],
        "path": row["path"], "sha256": row["sha256"], "duration_ms": ep["duration_ms"],
        "actual_intake": row["metadata"]["ingested_total"], "final_hunger": row["metadata"]["final_hunger"],
        "target_transitions": row["metadata"]["target_transitions"], "baseline_events": events,
        "baseline_raw": metrics(row["y"], raw), "baseline_indicator": metrics(row["y"], indicator),
        "baseline_score": _stats(scores), "baseline_active_samples": int(indicator.sum()),
        "baseline_mean_logit": float((contributions.sum(axis=1) + model.bias[0]).mean()),
        "baseline_bias": float(model.bias[0]),
        "largest_mean_logit_contributions": [{"feature_column": int(i), "tau_ms": 50 if i < 512 else 200,
            "mapping_feature_row": int(i % 512), "mean_contribution": float(mean_contributions[i])} for i in top_indices],
        "bands": {name: _metric_pair(row["y"], raw, indicator, mask) for name, mask in masks.items()},
        "windows": windows,
        "samples": {"time_ms": row["time_ms"].tolist(), "target": row["y"].tolist(),
            "hunger": need.tolist(), "baseline_score": scores.tolist(), "baseline_indicator": indicator.tolist()}}


def run(output, report, groups=None):
    output, report = Path(output), Path(report)
    if report.exists():
        raise FileExistsError(f"Use a fresh diagnostics report path: {report}")
    study = load_plan(output)
    selected_groups = set(study["seed_groups"]["train"] if groups is None else groups)
    if not selected_groups or not selected_groups <= set(study["seed_groups"]["train"]):
        raise ValueError("Diagnostics can read declared TRAIN groups only")
    planned = [ep for ep in study["episodes"] if ep["split"] == "train" and ep["seed_group"] in selected_groups]
    model_path = Path(study["baseline_model_path"])
    if file_digest(model_path) != study["baseline_model_sha256"]:
        raise ValueError("Frozen historical readout changed")
    model = LinearReadout.load(model_path, expected_feature_digest=study["feature_digest"])
    rows = [_load_episode(output / "episodes" / "connected" / (ep["episode_id"] + ".npz"), ep, study) for ep in planned]
    if len({r["metadata"]["nominal_weights_sha256"] for r in rows}) != 1:
        raise ValueError("Mixed nominal graph weights in TRAIN collection")
    cases = [analyze_episode(row, study, model) for row in rows]
    targets = np.concatenate([row["y"] for row in rows])
    scores = np.concatenate([case["samples"]["baseline_score"] for case in cases])
    indicated = np.concatenate([case["samples"]["baseline_indicator"] for case in cases])
    value = {"schema_version": 1, "scope": "TRAIN-only descriptive diagnosis of frozen v1 readout on new physical X; no model fitting or acceptance claim",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "plan_digest": study["digest"],
        "baseline_model_path": model_path.as_posix(), "baseline_model_sha256": file_digest(model_path),
        "feature_digest": model.feature_digest, "train_groups": sorted(selected_groups), "episodes": len(cases),
        "all_requested_training_episodes_included": True,
        "all_training_episodes_included": len(cases) == sum(ep["split"] == "train" for ep in study["episodes"]),
        "validation_or_test_opened": False, "raw": metrics(targets, scores >= study["score_threshold"]),
        "indicator": metrics(targets, indicated), "cases": cases,
        "source_hashes": {p.as_posix(): file_digest(p) for p in [Path(__file__), Path("fly_semantic/readout.py"),
            Path("fly_semantic/protocol.py"), Path("semantic_tools/physical_collection.py")]},
        "limits": ["Group means can hide individually informative or saturated cells; no separability claim follows from similar means.",
            "Input rates, hunger, taste, action and time are diagnostic only; X alone enters the old head.",
            "Window ranges use neural bin/sample end times; labels describe hunger 10 ms before sample end.",
            "Histories in the teacher hysteresis band may differ; band error metrics do not resolve causal memory capacity.",
            "This is training-set feedback; final claims require the separately frozen new test."]}
    value["digest"] = digest(value)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--group", type=int, action="append")
    args = parser.parse_args()
    result = run(args.output, args.report, args.group)
    print(json.dumps({key: result[key] for key in ("episodes", "train_groups", "all_training_episodes_included", "raw", "indicator")}, allow_nan=False))


if __name__ == "__main__":
    main()
