"""Bounded v3 selection on neural replays, then one frozen fresh physical test.

Only neural X enters the linear head. Training masks are exported as exactly zero
coefficients in the existing 1024-feature format, requiring no runtime shortcut.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from fly_semantic.mapping import digest, file_digest
from fly_semantic.readout import LinearReadout
from semantic_tools import physical_learning as physical


GAINS = (4.5, 5.0, 6.0)
MASKS = ("all", "low_saturation")
L2_GRID = (.001, .01, .1)
SELECTION_RULE = "all_validation_gates_then_min_limited_recall_then_indicator_f1_then_raw_f1_then_larger_l2_then_declared_order"
ACCEPTANCE = {
    "target_f1": .8, "indicator_target_f1": .8,
    "minimum_limited_case_recall": .8, "minimum_hungry_away_case_recall": .8,
    "maximum_false_active_fraction_on_sated": .1,
    "release_required_fraction": .75, "release_max_latency_ms": 1000,
    "release_stable_ms": 500,
}


def protocol_fields():
    """Merge into each plan before computing its content digest."""
    return {
        "v3_gain_grid": list(GAINS), "v3_mask_grid": list(MASKS),
        "l2_grid": list(L2_GRID), "v3_selection_rule": SELECTION_RULE,
        # physical.load_rows retains its historical plan validation.
        "selection_rule": "indicator_f1_then_raw_f1_then_larger_l2",
        "v3_mask_definition": {"rate_hz": 300., "maximum_fraction_exclusive": .25,
            "source": "TRAIN rows only; maximum of both traces per cell; mask both columns"},
        "acceptance": deepcopy(ACCEPTANCE), "release_stable_ms": 500,
        "sample_ms": 100, "teacher_on": .7, "teacher_off": .55,
        "label_reference_offset_ms": 10, "indicator_on": .7,
        "indicator_off": .4, "confirm_samples": 2, "score_threshold": .5,
    }


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("semantic_tools/v3_learning.py", "semantic_tools/physical_learning.py",
        "semantic_tools/transition_metrics.py", "semantic_tools/state_experiment.py",
        "fly_semantic/readout.py", "fly_semantic/protocol.py", "fly_semantic/features.py")
    return {name: file_digest(root / name) for name in names}


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _sealed(value):
    return value | {"digest": digest(value)}


def _read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("digest") != digest({k: v for k, v in value.items() if k != "digest"}):
        raise ValueError(f"Artifact digest mismatch: {path}")
    return value


def validate_plan(study):
    physical.validate_plan(study)
    for key, expected in protocol_fields().items():
        if study.get(key) != expected:
            raise ValueError(f"Frozen v3 protocol differs: {key}")
    gain = study.get("hunger_gain_mv")
    if type(gain) not in (int, float) or gain not in GAINS:
        raise ValueError("Plan gain is outside the frozen candidate grid")
    if not study.get("feature_digest") or not study.get("nominal_weights_sha256"):
        raise ValueError("Plan must bind feature and nominal graph identities")


def training_mask(X, kind):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 1024 or not len(X) or not np.isfinite(X).all() or np.any(X < 0):
        raise ValueError("Mask requires finite nonnegative TRAIN X with 1024 columns")
    if kind not in MASKS:
        raise ValueError("Unknown frozen feature mask")
    fraction = (np.maximum(X[:, :512], X[:, 512:]) >= 300.).mean(axis=0)
    cells = np.ones(512, dtype=bool) if kind == "all" else fraction < .25
    if not cells.any():
        raise ValueError("No cells survive the predeclared mask")
    return np.tile(cells, 2), {
        "kind": kind, "fit_split": "train", "fit_rows": len(X),
        "saturation_fraction_per_cell": fraction.tolist(),
        "kept_cells": np.flatnonzero(cells).tolist(),
        "kept_columns": np.flatnonzero(np.tile(cells, 2)).tolist(),
    }


def assess_acceptance(scored, study):
    if study.get("acceptance") != ACCEPTANCE or study.get("release_stable_ms") != 500:
        raise ValueError("Frozen v3 acceptance changed")
    limited = [c for c in scored["cases"] if c["scene"] == "limited_meal"]
    hungry = [c for c in scored["cases"] if c["scene"] == "hungry_away"]
    minimum_limited = min((c["indicator"]["recall"] for c in limited), default=None)
    minimum_hungry = min((c["indicator"]["recall"] for c in hungry), default=None)
    release = scored["release_fraction_all_feeding_cases"]
    latency = scored["maximum_supported_release_latency_ms"]
    false = scored["maximum_false_active_fraction_on_sated"]
    checks = {
        "target_f1": scored["raw"]["f1"] >= .8,
        "indicator_target_f1": scored["indicator"]["f1"] >= .8,
        "minimum_limited_case_recall": minimum_limited is not None and minimum_limited >= .8,
        "minimum_hungry_away_case_recall": minimum_hungry is not None and minimum_hungry >= .8,
        "maximum_false_active_fraction_on_sated": false is not None and false <= .1,
        "release_required_fraction": release is not None and release >= .75,
        "release_max_latency_ms": latency is not None and latency <= 1000,
    }
    return {"checks": checks,
        "status": "PRELIMINARY_TARGET_REACHED" if all(checks.values()) else "TARGET_NOT_REACHED",
        "minimum_limited_case_recall": minimum_limited,
        "minimum_hungry_away_case_recall": minimum_hungry,
        "limited_cases": len(limited), "hungry_away_cases": len(hungry)}


def selection_key(candidate):
    assessment = candidate["assessment"]
    minimum = assessment["minimum_limited_case_recall"]
    return (all(assessment["checks"].values()), -1. if minimum is None else minimum,
        candidate["validation"]["indicator"]["f1"], candidate["validation"]["raw"]["f1"],
        candidate["l2"], -candidate["candidate_index"])


def _verify_sources(hashes):
    root = Path(__file__).resolve().parents[1]
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        if file_digest(path) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def _verify_data(manifest):
    for item in manifest:
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen development data changed: {item['path']}")


def _calibration_files(study):
    """Bind actual calibration bytes as well as declared feature identities."""
    if not study.get("calibration_dir"):
        return []  # Tiny standalone learning fixtures have no neural backend.
    folder = Path(study["calibration_dir"])
    calibration = _read_sealed(folder / "calibration.json")
    mapping = _read_sealed(folder / "mapping.json")
    if (calibration["digest"] != study["calibration_digest"]
            or calibration["mapping_digest"] != mapping["digest"]
            or mapping["digest"] != study["mapping_digest"]
            or calibration["hunger_gain_mv"] != study["hunger_gain_mv"]):
        raise ValueError("Actual calibration/mapping differs from frozen study")
    return [{"path": str(folder / name), "sha256": file_digest(folder / name)}
        for name in ("mapping.json", "calibration.json")]


def _validate_replay(rows, study):
    workflow = study.get("development_workflow_sources")
    if workflow:
        _verify_sources(workflow)
    inputs = []
    for row in rows:
        meta = row["metadata"]
        if meta.get("body_physics") is not False or meta.get("matched_original_hunger") is not True:
            raise ValueError("Development requires declared neural replay with original hunger")
        if workflow and meta.get("replay_workflow_sources") != workflow:
            raise ValueError("Development replay workflow differs from frozen plan")
        path, expected = meta.get("source_input_path"), meta.get("source_input_sha256")
        if not path or not isinstance(expected, str) or len(expected) != 64 or file_digest(path) != expected:
            raise ValueError("Recorded native input provenance mismatch")
        inputs.append({"path": str(path), "sha256": expected})
    return inputs


def _assert_test_absent(output, study):
    paths = [Path(output) / "episodes" / condition / (ep["episode_id"] + ".npz")
        for ep in study["episodes"] if ep["split"] == "test"
        for condition in study["collection_conditions"]]
    if any(p.exists() for p in paths):
        raise ValueError("Future test outputs already exist before binding")
    return [p.as_posix() for p in paths]


def train(root, gain_dirs):
    """Fit exactly the frozen 3 gains x 2 masks x 3 L2 grid; never open test."""
    root = Path(root)
    selected_dir = root / "selected"
    candidates_dir = root / "candidates"
    if selected_dir.exists() or candidates_dir.exists():
        raise FileExistsError("Use a new v3 root; selection/candidates already exist")
    prepared = []
    for output in map(Path, gain_dirs):
        study = _read_sealed(output / "plan.json")
        validate_plan(study)
        _verify_sources(study.get("collection_source_hashes", study["source_hashes"]))
        _assert_test_absent(output, study)
        rows = physical.load_rows(output, study, "train")
        validation = physical.load_rows(output, study, "validation")
        physical._consistent(rows + validation)
        inputs = _validate_replay(rows + validation, study)
        inputs.extend(_calibration_files(study))
        prepared.append((float(study["hunger_gain_mv"]), output, study, rows, validation, inputs))
    prepared.sort(key=lambda item: item[0])
    if [item[0] for item in prepared] != list(GAINS):
        raise ValueError("Training requires exactly one development directory per frozen gain")
    # Paired native inputs, labels, and episode definitions must match across gains.
    reference = prepared[0]
    for item in prepared[1:]:
        if (item[2]["nominal_weights_sha256"] != reference[2]["nominal_weights_sha256"]
                or item[2].get("mapping_digest") != reference[2].get("mapping_digest")):
            raise ValueError("Gain candidates change nominal graph or port mapping")
        if item[2]["episodes"] != reference[2]["episodes"]:
            raise ValueError("Gain development episode plans are not paired")
        for a, b in zip(reference[3] + reference[4], item[3] + item[4]):
            if (a["metadata"]["source_input_sha256"] != b["metadata"]["source_input_sha256"]
                    or any(not np.array_equal(a[k], b[k]) for k in ("y", "hunger", "intake", "time_ms"))):
                raise ValueError("Gain development inputs or body labels are not paired")
    candidates_dir.mkdir(parents=True, exist_ok=False)
    candidates, manifests, development, input_manifest = [], [], [], []
    for gain, output, study, rows, validation, inputs in prepared:
        X = np.concatenate([r["X"] for r in rows])
        y = np.concatenate([r["y"] for r in rows])
        manifest = physical._data_manifest(rows + validation)
        manifests.extend(manifest)
        input_manifest.extend(inputs)
        development.append({"gain": gain, "path": str(output), "plan_digest": study["digest"],
            "plan_path": str(output / "plan.json"), "plan_sha256": file_digest(output / "plan.json"),
            "source_hashes": study.get("collection_source_hashes", study["source_hashes"]),
            "development_workflow_sources": study.get("development_workflow_sources", {}),
            "feature_digest": study["feature_digest"], "calibration_digest": study["calibration_digest"]})
        for mask_kind in MASKS:
            keep, mask_info = training_mask(X, mask_kind)
            masked = X.copy()
            masked[:, ~keep] = 0.
            for l2 in L2_GRID:
                index = len(candidates)
                name = f"{index:02d}-gain-{gain:g}-{mask_kind}-l2-{l2:g}"
                destination = candidates_dir / name
                destination.mkdir()
                model = LinearReadout.fit(masked, y, feature_digest=study["feature_digest"],
                    l2=l2, max_iter=study.get("max_iter", 500), seed=study.get("training_seed", 42))
                if not model.diagnostics["converged"]:
                    raise RuntimeError(f"Candidate optimizer failed: {name}")
                if np.any(model.weights[~keep] != 0) or np.any(model.mean[~keep] != 0):
                    raise RuntimeError("Masked columns retain an inference contribution")
                model_path = destination / "state-readout.npz"
                model.save(model_path)
                scored = physical.score_rows(model, validation, study)
                candidate = {"candidate_index": index, "name": name, "gain": gain,
                    "mask": mask_info, "l2": l2, "model_path": str(model_path),
                    "model_sha256": file_digest(model_path), "feature_digest": study["feature_digest"],
                    "calibration_digest": study["calibration_digest"],
                    "nominal_weights_sha256": rows[0]["metadata"]["nominal_weights_sha256"],
                    "development_dir": str(output), "development_plan_digest": study["digest"],
                    "train_positive_fraction": float(y.mean()), "train_rows": len(y),
                    "nonzero_weight_columns": np.flatnonzero(np.any(model.weights != 0, axis=1)).tolist(),
                    "optimizer": model.diagnostics, "validation": scored,
                    "assessment": assess_acceptance(scored, study), "test_seen": False}
                candidate = _sealed(candidate)
                _write_new(destination / "validation.json", candidate)
                candidates.append(candidate)
    chosen = max(candidates, key=selection_key)
    selected_dir.mkdir(parents=True, exist_ok=False)
    destination = selected_dir / "state-readout.npz"
    with destination.open("xb") as stream:
        stream.write(Path(chosen["model_path"]).read_bytes())
    selection = _sealed({"schema_version": 3, "selection_rule": SELECTION_RULE,
        "candidate_count": len(candidates), "candidate_index": chosen["candidate_index"],
        "candidate_name": chosen["name"], "gain": chosen["gain"], "mask": chosen["mask"],
        "selected_l2": chosen["l2"], "model_sha256": chosen["model_sha256"],
        "feature_digest": chosen["feature_digest"], "calibration_digest": chosen["calibration_digest"],
        "nominal_weights_sha256": chosen["nominal_weights_sha256"],
        "development_dir": chosen["development_dir"], "train_positive_fraction": chosen["train_positive_fraction"],
        "nonzero_weight_columns": chosen["nonzero_weight_columns"],
        "assessment": chosen["assessment"], "validation": chosen["validation"],
        "development": development, "development_data": manifests, "recorded_inputs": input_manifest,
        "candidate_artifacts": [{"path": str(candidates_dir / c["name"] / "validation.json"),
            "sha256": file_digest(candidates_dir / c["name"] / "validation.json"),
            "model_path": c["model_path"], "model_sha256": c["model_sha256"]} for c in candidates],
        "source_hashes": source_hashes(), "test_seen": False, "all_candidates_reported": True,
        "scope": "Model selection on matched native-input neural replay of v2 train/validation; not fresh physical acceptance"})
    _write_new(selected_dir / "selection.json", selection)
    _write_new(selected_dir / "training.json", _sealed({"schema_version": 3,
        "selection_digest": selection["digest"], "protocol": protocol_fields(),
        "candidates": candidates, "source_hashes": source_hashes()}))
    return selection


def _verify_selection(selected_dir):
    selected_dir = Path(selected_dir)
    selection = _read_sealed(selected_dir / "selection.json")
    if selection["source_hashes"] != source_hashes():
        raise ValueError("Learning sources changed after selection")
    if selection["selection_rule"] != SELECTION_RULE or selection["candidate_count"] != 18 or selection["test_seen"] is not False:
        raise ValueError("Selection protocol mismatch")
    if file_digest(selected_dir / "state-readout.npz") != selection["model_sha256"]:
        raise ValueError("Selected model changed")
    _verify_data(selection["development_data"] + selection["recorded_inputs"])
    for item in selection["development"]:
        if file_digest(item["plan_path"]) != item["plan_sha256"]:
            raise ValueError("Development plan changed")
        _verify_sources(item["source_hashes"])
        _verify_sources(item["development_workflow_sources"])
    for item in selection["candidate_artifacts"]:
        if file_digest(item["path"]) != item["sha256"] or file_digest(item["model_path"]) != item["model_sha256"]:
            raise ValueError("Candidate artifact changed")
    return selection


def bind_test(selected_dir):
    """Freeze fresh physical plan after selection and before any test output."""
    output = Path(selected_dir)
    if (output / "test-binding.json").exists():
        raise FileExistsError("Test already bound")
    selected = _verify_selection(output)
    study = _read_sealed(output / "plan.json")
    validate_plan(study)
    _verify_sources(study.get("collection_source_hashes", study["source_hashes"]))
    for key in ("feature_digest", "calibration_digest", "nominal_weights_sha256"):
        if study[key] != selected[key]:
            raise ValueError("Fresh test identity differs from selected candidate")
    if study["hunger_gain_mv"] != selected["gain"]:
        raise ValueError("Fresh test gain differs from selected candidate")
    test_groups = {ep["seed_group"] for ep in study["episodes"] if ep["split"] == "test"}
    if test_groups != {4201, 4202, 4203} or len([ep for ep in study["episodes"] if ep["split"] == "test"]) != 18:
        raise ValueError("Fresh physical test requires the frozen 3 x 6 episodes")
    absent = _assert_test_absent(output, study)
    binding = _sealed({"schema_version": 3, "plan_digest": study["digest"],
        "plan_sha256": file_digest(output / "plan.json"), "selection_digest": selected["digest"],
        "selection_sha256": file_digest(output / "selection.json"),
        "model_sha256": selected["model_sha256"], "source_hashes": source_hashes(),
        "collection_source_hashes": study.get("collection_source_hashes", study["source_hashes"]),
        "development_workflow_sources": study.get("development_workflow_sources", {}),
        "calibration_files": _calibration_files(study),
        "future_test_outputs_absent": absent, "test_seen": False})
    _write_new(output / "test-binding.json", binding)
    return binding


def verify_test_binding(selecteddir):
    """Verify frozen artifacts without requiring test outputs to remain absent."""
    output = Path(selecteddir)
    selected = _verify_selection(output)
    study = _read_sealed(output / "plan.json")
    validate_plan(study)
    binding = _read_sealed(output / "test-binding.json")
    if (binding["plan_sha256"] != file_digest(output / "plan.json")
            or binding["selection_sha256"] != file_digest(output / "selection.json")
            or binding["model_sha256"] != selected["model_sha256"]
            or binding["source_hashes"] != source_hashes() or binding["test_seen"] is not False):
        raise ValueError("Frozen test binding changed")
    _verify_sources(binding["collection_source_hashes"])
    _verify_sources(binding["development_workflow_sources"])
    _verify_data(binding["calibration_files"])
    return binding


def evaluate(selecteddir):
    """Open every planned test only after verifying the pre-collection binding."""
    output = Path(selecteddir)
    if (output / "results.json").exists():
        raise FileExistsError("Held-out result already recorded")
    binding = verify_test_binding(output)
    selected = _read_sealed(output / "selection.json")
    study = _read_sealed(output / "plan.json")
    if set(study["collection_conditions"]) != {"connected", "homeostasis_off", "transmission_off"}:
        raise ValueError("Fresh test must retain all declared controls")
    recorded = {condition: physical.load_rows(output, study, "test", condition)
        for condition in study["collection_conditions"]}
    physical._consistent([r for rows in recorded.values() for r in rows])
    if physical._identity(recorded["connected"][0])[:3] != (
            selected["feature_digest"], selected["calibration_digest"], selected["nominal_weights_sha256"]):
        raise ValueError("Test feature/graph identities differ from selection")
    if any(row["metadata"].get("body_physics") is not True for row in recorded["connected"]):
        raise ValueError("Connected acceptance requires fresh physical episodes")
    workflow = study.get("development_workflow_sources")
    if workflow:
        for rows in recorded.values():
            for row in rows:
                meta = row["metadata"]
                key = "physical_workflow_sources" if meta.get("body_physics") else "replay_workflow_sources"
                if meta.get(key) != workflow:
                    raise ValueError("Test physical/replay workflow differs from frozen plan")
    model = LinearReadout.load(output / "state-readout.npz", expected_feature_digest=selected["feature_digest"])
    keep = np.zeros(1024, dtype=bool)
    keep[selected["mask"]["kept_columns"]] = True
    if np.any(model.weights[~keep] != 0):
        raise ValueError("Selected feature mask was bypassed")
    shuffled, permutation = physical._shuffle_rows(recorded["connected"], study.get("shuffle_seed", 81817))
    conditions = recorded | {"cross_episode_shuffle": shuffled}
    scored = {key: physical.score_rows(model, rows, study) for key, rows in conditions.items()}
    constant = physical._ConstantHead(model.feature_digest, float(selected["train_positive_fraction"] >= .5))
    scored["train_constant"] = physical.score_rows(constant, recorded["connected"], study)
    result = _sealed({"schema_version": 3, **assess_acceptance(scored["connected"], study),
        "plan_digest": study["digest"], "selection_digest": selected["digest"],
        "test_binding_digest": binding["digest"], "model_sha256": selected["model_sha256"],
        "gain": selected["gain"], "mask": selected["mask"]["kind"], "selected_l2": selected["selected_l2"],
        "conditions": scored, "shuffle_permutation": permutation,
        "shuffle_method": "Deranged whole-episode donors; monotone normalized-time nearest-neighbor alignment; diagnostic only",
        "paired_connected_minus_control": {key: physical.paired_seed_bootstrap(scored["connected"], value)
            for key, value in scored.items() if key != "connected"},
        "test_data": physical._data_manifest([r for rows in recorded.values() for r in rows]),
        "all_planned_episodes_included": True, "test_tuned": False,
        "source_hashes": source_hashes(), "historical_v2_is_paired_baseline": False,
        "scope": "Three-seed preliminary fresh physical NEED_FOOD test; fixed head and artificial input; controls may be neural replay"})
    _write_new(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "bind-test", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("runs/semantic-v3"))
    args = parser.parse_args()
    if args.command == "train":
        result = train(args.root, [args.root / "development" / f"gain-{gain:g}" for gain in GAINS])
    elif args.command == "bind-test":
        result = bind_test(args.root / "selected")
    else:
        result = evaluate(args.root / "selected")
    print(json.dumps({k: result[k] for k in ("digest", "status", "gain", "mask", "selected_l2", "candidate_count") if k in result}, allow_nan=False))


if __name__ == "__main__":
    main()
