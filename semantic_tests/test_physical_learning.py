"""Physical study provenance, held-out selection, and meaningful meal release."""
import copy
import json

import numpy as np
import pytest

from fly_semantic.mapping import digest, file_digest
from fly_semantic.readout import LinearReadout
from semantic_tools import physical_learning as learning


class ScoresFromFeatures:
    feature_digest = "fixture-features"

    def predict_scores(self, X):
        return np.asarray(X)[:, :1]


def seal(study):
    study["digest"] = digest({key: value for key, value in study.items() if key != "digest"})
    return study


def study_fixture():
    episodes = [{"episode_id": f"{split}-{index}", "split": split, "seed_group": group,
                 "scene": "meal_fast" if index == 0 else "sated_away", "expected_release": index == 0,
                 "duration_ms": 1000, "config": {"initial": {"energy": 10 if index == 0 else 80}}}
                for split, group in (("train", 1), ("validation", 2), ("test", 3)) for index in (0, 1)]
    return seal({"episodes": episodes, "sample_ms": 100, "teacher_on": .7, "teacher_off": .55,
        "label_reference_offset_ms": 10, "indicator_on": .7, "indicator_off": .4,
        "confirm_samples": 2, "score_threshold": .5, "release_stable_ms": 200,
        "l2_grid": [.001, .01, .1], "max_iter": 100, "calibration_digest": "fixture-calibration",
        "feature_digest": ScoresFromFeatures.feature_digest, "source_hashes": {"fixture.py": "fixture-sha"},
        "collection_conditions": ["connected", "homeostasis_off"], "baseline_model_sha256": "fixture-baseline",
        "acceptance": {"target_f1": .8, "indicator_target_f1": .8, "release_required_fraction": 1.,
                       "release_max_latency_ms": 500, "release_stable_ms": 200,
                       "maximum_false_active_fraction_on_sated": .1, "baseline_improvement_min": .1}})


def episode_row(study, ep, scores=None, condition="connected"):
    duration = ep["duration_ms"]
    time = np.arange(100, duration + 1, 100, dtype=np.int64)
    transitions = [{"time_ms": 0, "active": True}, {"time_ms": 490, "active": False}] if ep["expected_release"] else []
    y = learning._teacher_at(transitions, time - 10)
    features = np.array(scores if scores is not None else np.where(y, .95, .05), dtype=float)
    return {"X": features[:, None], "y": y, "time_ms": time, "hunger": np.where(y, .8, .3),
            "intake": np.linspace(0, 4, len(time)) if ep["expected_release"] else np.zeros(len(time)),
            "metadata": {"episode": ep, "plan_digest": study["digest"], "condition": condition,
                "feature_digest": study["feature_digest"], "calibration_digest": study["calibration_digest"],
                "nominal_weights_sha256": "fixture-weights", "source_hashes": study["source_hashes"],
                "target_transitions": transitions}}


def save_row(output, row):
    meta = row["metadata"]
    path = output / "episodes" / meta["condition"] / (meta["episode"]["episode_id"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: value for key, value in row.items() if key != "metadata"}, metadata=np.array(json.dumps(meta)))
    return path


def populate(output, study, splits=("train", "validation", "test")):
    for ep in study["episodes"]:
        if ep["split"] in splits:
            for condition in study["collection_conditions"]:
                row = episode_row(study, ep, condition=condition)
                if condition != "connected":
                    row["X"][:] = .05
                save_row(output, row)


def test_whole_episode_identity_and_exact_teacher_alignment(tmp_path):
    study = study_fixture()
    populate(tmp_path, study)
    rows = learning.load_rows(tmp_path, study, "train")
    assert len(rows) == 2
    assert rows[0]["y"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert rows[0]["sha256"] == file_digest(rows[0]["path"])


@pytest.mark.parametrize("change", ["source", "plan", "episode", "calibration", "features", "weights", "failed", "partial", "labels", "time", "nan", "negative_intake"])
def test_corrupt_or_failed_trial_cannot_be_dropped(tmp_path, change):
    study = study_fixture()
    populate(tmp_path, study)
    row = episode_row(study, study["episodes"][0])
    meta = row["metadata"]
    if change == "source":
        meta["source_hashes"] = {"different.py": "changed"}
    elif change == "plan":
        meta["plan_digest"] = "changed"
    elif change == "episode":
        meta["episode"] = copy.deepcopy(meta["episode"])
        meta["episode"]["config"]["initial"]["energy"] = 99
    elif change in ("calibration", "features"):
        meta["calibration_digest" if change == "calibration" else "feature_digest"] = "changed"
    elif change == "weights":
        meta["nominal_weights_sha256"] = "changed"
    elif change == "failed":
        meta["status"] = "error"
    elif change == "partial":
        row["X"] = row["X"][:-1]
    elif change == "labels":
        row["y"][4] = 1
    elif change == "time":
        row["time_ms"][4] += 1
    elif change == "nan":
        row["X"][0, 0] = np.nan
    else:
        row["intake"][1] = -1
    save_row(tmp_path, row)
    with pytest.raises(ValueError):
        learning.load_rows(tmp_path, study, "train")


def test_missing_trial_fails_instead_of_shortening_denominator(tmp_path):
    study = study_fixture()
    save_row(tmp_path, episode_row(study, study["episodes"][0]))
    with pytest.raises(FileNotFoundError):
        learning.load_rows(tmp_path, study, "train")


def test_split_overlap_rejected_even_with_a_valid_plan_digest():
    study = study_fixture()
    study["episodes"][-1]["seed_group"] = 1
    seal(study)
    with pytest.raises(ValueError, match="overlap"):
        learning.validate_plan(study)


def test_missed_initial_hunger_cannot_be_repaired_by_a_later_false_on():
    study = study_fixture()
    row = episode_row(study, study["episodes"][0], scores=[.1] * 5 + [.9] * 3 + [.1] * 2)
    case = learning.score_rows(ScoresFromFeatures(), [row], study)["cases"][0]
    assert case["initial_high_missed"]
    assert case["initial_high_detection_ms"] is None
    assert case["false_activation_events"] == 1
    assert not case["release_supported"]


def test_real_release_requires_prior_on_food_and_stable_off():
    study = study_fixture()
    row = episode_row(study, study["episodes"][0])
    result = learning.score_rows(ScoresFromFeatures(), [row], study)
    assert result["feeding_cases_with_release"] == 1
    match = result["cases"][0]["release_matches"][0]
    assert match["stable_neural_off_ms"] == 600
    assert match["stable_latency_ms"] == 110
    row["intake"][:] = 0
    result = learning.score_rows(ScoresFromFeatures(), [row], study)
    assert result["release_fraction_all_feeding_cases"] == 0


def test_early_off_and_silence_do_not_satisfy_release():
    study = study_fixture()
    for scores in ([.9, .9] + [.1] * 8, [.1] * 10):
        row = episode_row(study, study["episodes"][0], scores=scores)
        result = learning.score_rows(ScoresFromFeatures(), [row], study)
        assert result["release_fraction_all_feeding_cases"] == 0
        assert result["supported_release_latency_ms"] == []


def test_unstable_off_waits_for_first_later_sustained_off():
    matches = [{"target_off_ms": 300, "inactive_interval_end_ms": 1500, "indicator_active_before_target_off": True}]
    events = [{"sim_time_ms": t, "active": state} for t, state in ((100, True), (400, False), (500, True), (800, False))]
    result = learning._stable_releases(matches, events, 500)
    assert result[0]["stable_neural_off_ms"] == 800
    assert result[0]["stable_latency_ms"] == 500


def test_expected_meal_without_target_off_stays_in_release_denominator():
    study = study_fixture()
    first = episode_row(study, study["episodes"][0])
    second = copy.deepcopy(first)
    second["metadata"]["episode"]["episode_id"] = "failed-meal"
    second["metadata"]["target_transitions"] = [{"time_ms": 0, "active": True}]
    second["y"][:] = 1
    second["X"][:] = .9
    result = learning.score_rows(ScoresFromFeatures(), [first, second], study)
    assert result["feeding_cases"] == 2
    assert result["feeding_cases_with_target_off"] == 1
    assert result["release_fraction_all_feeding_cases"] == .5


def test_privileged_state_does_not_change_scores_or_indicator():
    study = study_fixture()
    original = episode_row(study, study["episodes"][0])
    changed = copy.deepcopy(original)
    changed["hunger"][:] = 0
    changed["intake"][:] = 0
    changed["metadata"]["episode"].update(config={"initial": {"energy": 100}}, cue_id=102, hidden_food=[50, 50], seed_group=900)
    a = learning.score_rows(ScoresFromFeatures(), [original], study)["cases"][0]
    b = learning.score_rows(ScoresFromFeatures(), [changed], study)["cases"][0]
    assert a["scores"] == b["scores"]
    assert a["indicator_samples"] == b["indicator_samples"]
    assert a["raw"] == b["raw"]
    assert a["indicator"] == b["indicator"]


def test_train_only_normalizer_and_selection_before_any_test_load(tmp_path, monkeypatch):
    study = study_fixture()
    populate(tmp_path, study, splits=("train", "validation"))  # No test files exist.
    validation = episode_row(study, study["episodes"][2])
    validation["X"] += 1000
    save_row(tmp_path, validation)
    original_loader = learning.load_rows
    calls = []

    def recorded_loader(output, plan, split, condition="connected"):
        calls.append(split)
        assert split != "test"
        return original_loader(output, plan, split, condition)

    monkeypatch.setattr(learning, "load_rows", recorded_loader)
    selected = learning.train(tmp_path, study)
    model = LinearReadout.load(tmp_path / "state-readout.npz", expected_feature_digest=study["feature_digest"])
    train_X = np.concatenate([row["X"] for row in original_loader(tmp_path, study, "train")])
    np.testing.assert_array_equal(model.mean, train_X.mean(axis=0))
    np.testing.assert_array_equal(model.scale, train_X.std(axis=0))
    assert calls == ["train", "validation"]
    assert selected["test_seen"] is False
    assert len(selected["candidates"]) == 3
    assert (tmp_path / "training.json").exists()
    with pytest.raises(FileExistsError):
        learning.train(tmp_path, study)


def test_normalized_time_shuffle_preserves_whole_donor_and_target_shapes():
    rows = [{"X": np.array([[1.], [2.]]), "y": np.array([0, 0])},
            {"X": np.array([[10.], [20.], [30.], [40.]]), "y": np.array([1, 1, 1, 1])}]
    shuffled, permutation = learning._shuffle_rows(rows, 1)
    assert permutation == [1, 0]
    assert shuffled[0]["X"].ravel().tolist() == [10, 40]
    assert shuffled[1]["X"].ravel().tolist() == [1, 1, 2, 2]
    for original, changed in zip(rows, shuffled):
        np.testing.assert_array_equal(original["y"], changed["y"])


def test_seed_bootstrap_pairs_groups_not_frames():
    a = {"by_seed": {"10": {"raw": {"f1": .9}, "indicator": {"f1": .8}},
                     "20": {"raw": {"f1": .6}, "indicator": {"f1": .4}}}}
    b = {"by_seed": {"20": {"raw": {"f1": .4}, "indicator": {"f1": .2}},
                     "10": {"raw": {"f1": .7}, "indicator": {"f1": .6}}}}
    result = learning.paired_seed_bootstrap(a, b, samples=200)
    assert result["seed_groups"] == 2
    assert result["raw"]["mean_seed_f1_difference"] == pytest.approx(.2)
    assert result["indicator"]["paired_seed_bootstrap_95pct"] == pytest.approx([.2, .2])
    del b["by_seed"]["10"]
    with pytest.raises(ValueError):
        learning.paired_seed_bootstrap(a, b)


def test_acceptance_cannot_hide_failed_release_behind_high_f1():
    study = study_fixture()
    connected = {"raw": {"f1": .99}, "indicator": {"f1": .99}, "release_fraction_all_feeding_cases": .5,
                 "maximum_supported_release_latency_ms": 100, "maximum_false_active_fraction_on_sated": 0}
    baseline = {"indicator": {"f1": .5}}
    result = learning.assess_acceptance(connected, {}, study, baseline)
    assert result["status"] == "TARGET_NOT_REACHED"
    assert not result["checks"]["release_required_fraction"]


def test_complete_evaluation_pairs_same_features_and_freezes_results(tmp_path):
    study = study_fixture()
    baseline = LinearReadout(feature_digest=study["feature_digest"], concept_ids=(1,), mean=[0], scale=[1], weights=[[0]], bias=[-4])
    baseline_path = tmp_path / "baseline.npz"
    baseline.save(baseline_path)
    study["baseline_model_sha256"] = file_digest(baseline_path)
    seal(study)
    populate(tmp_path, study)
    learning.train(tmp_path, study)
    result = learning.evaluate(tmp_path, study, baseline_path)
    assert result["all_planned_episodes_included"]
    assert result["test_tuned"] is False
    assert len(result["test_data"]) == 4
    assert result["conditions"]["connected"]["episodes"] == 2
    assert result["baseline_conditions"]["connected"]["indicator"]["silence_fraction"] == 1
    assert result["conditions"]["connected"]["feeding_cases_with_release"] == 1
    with pytest.raises(FileExistsError):
        learning.evaluate(tmp_path, study, baseline_path)


def test_changed_training_data_is_rejected_before_test_loading(tmp_path, monkeypatch):
    study = study_fixture()
    populate(tmp_path, study, splits=("train", "validation"))
    learning.train(tmp_path, study)
    row = episode_row(study, study["episodes"][0])
    row["X"] += .01
    save_row(tmp_path, row)
    monkeypatch.setattr(learning, "load_rows", lambda *args: pytest.fail("Test loaded before selection integrity check"))
    with pytest.raises(ValueError, match="data changed"):
        learning.evaluate(tmp_path, study, tmp_path / "unused-baseline.npz")
