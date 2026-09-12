"""v3 masks, bounded selection, retained failures, and pre-test binding."""
from copy import deepcopy
import json

import numpy as np
import pytest

from fly_semantic.mapping import digest, file_digest
from fly_semantic.readout import LinearReadout
from semantic_tools import physical_learning as physical
from semantic_tools import v3_learning as learning


def sealed(value):
    return value | {"digest": digest(value)}


def score_fixture():
    return {"raw": {"f1": .9}, "indicator": {"f1": .9},
        "cases": [{"scene": "limited_meal", "indicator": {"recall": .9}},
                  {"scene": "hungry_away", "indicator": {"recall": .9}}],
        "release_fraction_all_feeding_cases": 1.,
        "maximum_supported_release_latency_ms": 300,
        "maximum_false_active_fraction_on_sated": 0.}


def test_mask_uses_both_causal_traces_and_exclusive_threshold():
    X = np.zeros((4, 1024))
    X[0, 7] = 300.
    X[0, 512 + 9] = 300.
    X[:, 11] = 299.999
    keep, info = learning.training_mask(X, "low_saturation")
    assert not keep[7] and not keep[519]
    assert not keep[9] and not keep[521]
    assert keep[11] and keep[523]
    assert info["fit_rows"] == 4
    assert info["saturation_fraction_per_cell"][7] == .25
    assert learning.training_mask(X, "all")[0].all()


@pytest.mark.parametrize("change", ["nan", "negative", "wrong_dimension", "empty", "all_saturated", "unknown"])
def test_invalid_or_empty_mask_fails(change):
    X = np.ones((4, 1024))
    kind = "low_saturation"
    if change == "nan": X[0, 0] = np.nan
    elif change == "negative": X[0, 0] = -1
    elif change == "wrong_dimension": X = X[:, :-1]
    elif change == "empty": X = X[:0]
    elif change == "all_saturated": X[:] = 300
    elif change == "unknown": kind = "chosen_after_test"
    with pytest.raises(ValueError):
        learning.training_mask(X, kind)


def test_mask_is_zero_weight_in_original_runtime_without_extra_features():
    X = np.zeros((30, 1024))
    y = np.tile([0, 1], 15)
    X[:, 0] = y * 30
    X[:, 1] = 320 + y
    keep, _ = learning.training_mask(X, "low_saturation")
    train = X.copy()
    train[:, ~keep] = 0
    model = LinearReadout.fit(train, y, feature_digest="fixture", l2=.01)
    assert np.all(model.weights[~keep] == 0)
    original = model.predict_scores(X)
    X[:, ~keep] = 1e10
    assert np.array_equal(original, model.predict_scores(X))


@pytest.mark.parametrize("change", ["limited", "hungry", "sated", "release", "latency", "raw", "indicator", "missing_limited"])
def test_each_requirement_can_independently_fail(change):
    scored = score_fixture()
    if change == "limited": scored["cases"][0]["indicator"]["recall"] = .799
    elif change == "hungry": scored["cases"][1]["indicator"]["recall"] = .799
    elif change == "sated": scored["maximum_false_active_fraction_on_sated"] = .101
    elif change == "release": scored["release_fraction_all_feeding_cases"] = .749
    elif change == "latency": scored["maximum_supported_release_latency_ms"] = 1001
    elif change == "raw": scored["raw"]["f1"] = .799
    elif change == "indicator": scored["indicator"]["f1"] = .799
    else: scored["cases"] = scored["cases"][1:]
    result = learning.assess_acceptance(scored, learning.protocol_fields())
    assert result["status"] == "TARGET_NOT_REACHED"
    assert not all(result["checks"].values())


def test_selection_prioritizes_all_gates_then_worst_limited_not_pooled_f1():
    good = score_fixture()
    bad = deepcopy(good)
    bad["indicator"]["f1"] = .999
    bad["cases"][0]["indicator"]["recall"] = .79
    def candidate(index, scored):
        return {"candidate_index": index, "l2": .01, "validation": scored,
                "assessment": learning.assess_acceptance(scored, learning.protocol_fields())}
    assert max([candidate(0, bad), candidate(1, good)], key=learning.selection_key)["candidate_index"] == 1
    a = candidate(0, bad)
    b = deepcopy(a)
    b["candidate_index"] = 1
    b["assessment"]["minimum_limited_case_recall"] = .795
    b["validation"]["indicator"]["f1"] = .6
    assert learning.selection_key(b) > learning.selection_key(a)


SCENES = ("hungry_away", "sated_away", "meal_fast", "meal_slow", "limited_meal", "sated_food")


def plan_fixture(gain, source):
    episodes = [{"episode_id": f"{split}-{group}-{scene}", "split": split,
        "seed_group": group, "scene": scene, "duration_ms": 2000,
        "expected_release": scene in ("meal_fast", "meal_slow"), "config": {}}
        for split, group in (("train", 1), ("validation", 2), ("test", 4201), ("test", 4202), ("test", 4203))
        for scene in SCENES]
    return sealed({**learning.protocol_fields(), "episodes": episodes, "hunger_gain_mv": gain,
        "feature_digest": f"features-{gain}", "calibration_digest": f"calibration-{gain}",
        "nominal_weights_sha256": "fixture-weights", "mapping_digest": "fixture-mapping",
        "source_hashes": {str(source): file_digest(source)}, "max_iter": 500,
        "collection_conditions": ["connected", "homeostasis_off", "transmission_off"]})


def row_fixture(study, ep, input_path, condition="connected", *, physical_body=False):
    time = np.arange(100, ep["duration_ms"] + 1, 100, dtype=np.int64)
    meal = ep["expected_release"]
    high = ep["scene"] in ("hungry_away", "limited_meal") or meal
    transitions = ([{"time_ms": 0, "active": True}] if high else [])
    if meal: transitions.append({"time_ms": 790, "active": False})
    y = physical._teacher_at(transitions, time - 10)
    X = np.zeros((len(time), 1024))
    X[:, 0] = 30 * y
    X[:, 1] = 320
    if condition != "connected": X[:, 0] = 0
    return {"X": X, "y": y, "time_ms": time, "hunger": np.where(y, .8, .3),
        "intake": np.linspace(0, 4 if meal else .8 if ep["scene"] == "limited_meal" else 0, len(time)),
        "metadata": {"episode": ep, "plan_digest": study["digest"], "condition": condition,
            "feature_digest": study["feature_digest"], "calibration_digest": study["calibration_digest"],
            "nominal_weights_sha256": study["nominal_weights_sha256"], "source_hashes": study["source_hashes"],
            "body_physics": physical_body, "matched_original_hunger": True,
            "source_input_path": str(input_path), "source_input_sha256": file_digest(input_path),
            "target_transitions": transitions}}


def save_row(output, row):
    meta = row["metadata"]
    path = output / "episodes" / meta["condition"] / (meta["episode"]["episode_id"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in row.items() if k != "metadata"},
                        metadata=np.asarray(json.dumps(meta)))
    return path


def development_fixture(tmp_path):
    root = tmp_path / "study"
    source = tmp_path / "collection.py"
    source.write_text("fixture-source", encoding="utf-8")
    inputs = tmp_path / "inputs.npz"
    inputs.write_bytes(b"fixture-recorded-inputs")
    dirs = []
    for gain in learning.GAINS:
        folder = root / "development" / f"gain-{gain:g}"
        study = plan_fixture(gain, source)
        learning._write_new(folder / "plan.json", study)
        for ep in study["episodes"]:
            if ep["split"] in ("train", "validation"):
                save_row(folder, row_fixture(study, ep, inputs))
        dirs.append(folder)
    return root, dirs, inputs, source


def test_bounded_training_binding_and_full_evaluation(tmp_path):
    root, dirs, inputs, source = development_fixture(tmp_path)
    selected = learning.train(root, dirs)
    assert selected["candidate_count"] == 18 and selected["test_seen"] is False
    assert len(selected["candidate_artifacts"]) == 18
    assert len(selected["development_data"]) == 36
    output = root / "selected"
    study = plan_fixture(selected["gain"], source)
    learning._write_new(output / "plan.json", study)
    binding = learning.bind_test(output)
    assert len(binding["future_test_outputs_absent"]) == 54
    for ep in study["episodes"]:
        if ep["split"] == "test":
            for condition in study["collection_conditions"]:
                save_row(output, row_fixture(study, ep, inputs, condition, physical_body=condition == "connected"))
    assert learning.verify_test_binding(output)["digest"] == binding["digest"]
    result = learning.evaluate(output)
    assert result["status"] == "PRELIMINARY_TARGET_REACHED"
    assert result["limited_cases"] == 3
    assert result["conditions"]["connected"]["episodes"] == 18
    assert result["conditions"]["transmission_off"]["indicator"]["f1"] == 0
    assert result["test_tuned"] is False
    with pytest.raises(FileExistsError): learning.evaluate(output)
    with pytest.raises(FileExistsError): learning.train(root, dirs)


@pytest.mark.parametrize("change", ["plan", "model", "development_data", "workflow"])
def test_changed_bound_artifacts_fail_before_any_test_open(tmp_path, monkeypatch, change):
    root, dirs, inputs, source = development_fixture(tmp_path)
    selected = learning.train(root, dirs)
    output = root / "selected"
    study = plan_fixture(selected["gain"], source)
    learning._write_new(output / "plan.json", study)
    learning.bind_test(output)
    if change == "plan":
        study["extra_after_binding"] = True
        study = sealed({k: v for k, v in study.items() if k != "digest"})
        (output / "plan.json").write_text(json.dumps(study), encoding="utf-8")
    elif change == "model":
        with (output / "state-readout.npz").open("ab") as stream: stream.write(b"changed")
    elif change == "development_data":
        with open(selected["development_data"][0]["path"], "ab") as stream: stream.write(b"changed")
    else: source.write_text("changed source", encoding="utf-8")
    def no_test_loading(*args, **kwargs):
        raise AssertionError("No NPZ may be opened before binding validation")
    monkeypatch.setattr(learning.np, "load", no_test_loading)
    with pytest.raises(ValueError): learning.evaluate(output)


def test_existing_test_prevents_training_without_loading_or_writing_candidates(tmp_path):
    root, dirs, inputs, _ = development_fixture(tmp_path)
    study = json.loads((dirs[0] / "plan.json").read_text())
    ep = next(e for e in study["episodes"] if e["split"] == "test")
    save_row(dirs[0], row_fixture(study, ep, inputs))
    with pytest.raises(ValueError, match="already exist"):
        learning.train(root, dirs)
    assert not (root / "candidates").exists()


@pytest.mark.parametrize("change", ["hunger", "inputs", "body", "graph", "mapping"])
def test_unpaired_or_misrepresented_gain_data_rejected(tmp_path, change):
    root, dirs, inputs, _ = development_fixture(tmp_path)
    study = json.loads((dirs[1] / "plan.json").read_text())
    ep = study["episodes"][0]
    row = row_fixture(study, ep, inputs)
    if change == "hunger": row["hunger"][:] = .9
    elif change == "inputs": row["metadata"]["source_input_sha256"] = "0" * 64
    elif change == "body": row["metadata"]["body_physics"] = True
    else:
        study["nominal_weights_sha256" if change == "graph" else "mapping_digest"] = "changed"
        study = sealed({k: v for k, v in study.items() if k != "digest"})
        (dirs[1] / "plan.json").write_text(json.dumps(study), encoding="utf-8")
        for item in study["episodes"]:
            if item["split"] in ("train", "validation"):
                save_row(dirs[1], row_fixture(study, item, inputs))
        row = row_fixture(study, ep, inputs)
    save_row(dirs[1], row)
    with pytest.raises(ValueError): learning.train(root, dirs)
    assert not (root / "candidates").exists()


def test_acceptance_threshold_change_is_rejected_not_silently_used():
    study = learning.protocol_fields()
    study["acceptance"]["minimum_limited_case_recall"] = .2
    with pytest.raises(ValueError, match="acceptance changed"):
        learning.assess_acceptance(score_fixture(), study)
