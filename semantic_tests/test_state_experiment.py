"""Study boundaries and reported UI behavior, without a full-graph benchmark."""
import copy
import json

import numpy as np
import pytest

from fly_arena.organism import Organism
from fly_semantic.mapping import digest
from fly_semantic.readout import LinearReadout
from semantic_tools import state_experiment as experiment


class ScoresFromFeatures:
    feature_digest = "fixture-neural-features"

    def predict_scores(self, X):
        return np.asarray(X, dtype=float)


def row(scores, labels, *, episode="fixture-1", seed=1):
    scores = np.asarray(scores, dtype=float)
    return {"X": scores[:, None], "y": np.array(labels, np.int8),
            "hunger": np.full(len(scores), .85), "time_ms": np.arange(1, len(scores) + 1) * 100,
            "metadata": {"feature_digest": ScoresFromFeatures.feature_digest,
                         "nominal_weights_sha256": "fixture-original-weights",
                         "episode": {"episode_id": episode, "seed_group": seed}}}


def test_plan_keeps_whole_episodes_and_rng_streams_disjoint(tmp_path):
    study = experiment.plan(tmp_path, {"digest": "fixture-calibration"})
    assert experiment.load_plan(tmp_path) == study
    assert len(study["episodes"]) == 60
    assert {split: sum(ep["split"] == split for ep in study["episodes"])
            for split in study["seed_groups"]} == {"train": 32, "validation": 8, "test": 20}
    for name in ("episode_id", "neural_seed", "background_seed"):
        assert len({ep[name] for ep in study["episodes"]}) == 60
    for split in study["seed_groups"]:
        episodes = [ep for ep in study["episodes"] if ep["split"] == split]
        needs = [Organism(study["organism_config"], ep["initial"]).hunger for ep in episodes]
        assert sum(need >= study["teacher_on"] for need in needs) == len(needs) // 2
        assert all(.08 <= need <= .94 for need in needs)
    with pytest.raises(FileExistsError):
        experiment.plan(tmp_path, {"digest": "do-not-overwrite"})


@pytest.mark.parametrize("reseal", [False, True])
def test_modified_plan_or_overlapping_split_is_rejected(tmp_path, reseal):
    study = experiment.plan(tmp_path, {"digest": "fixture-calibration"})
    study["seed_groups"]["test"][0] = study["seed_groups"]["train"][0]
    if reseal:
        study["digest"] = digest({k: v for k, v in study.items() if k != "digest"})
    (tmp_path / "plan.json").write_text(json.dumps(study))
    with pytest.raises(ValueError, match="digest|overlap"):
        experiment.load_plan(tmp_path)


def test_indicator_hysteresis_delay_and_false_positives_are_reported():
    result = experiment.score_rows(ScoresFromFeatures(), [row([.9, .9, .5, .3, .3], [1, 1, 0, 0, 0])])
    assert result["raw"]["f1"] == .8
    assert result["indicator"]["f1"] == .4
    assert result["indicator"]["fn"] == 1
    assert result["indicator"]["fp"] == 2
    assert result["initial_high_detection_ms"] == [200]
    assert result["initial_high_misses"] == 0


def test_good_raw_f1_does_not_disguise_a_silent_live_indicator():
    result = experiment.score_rows(ScoresFromFeatures(), [row([.65] * 5, [1] * 5)])
    assert result["raw"]["f1"] == 1
    assert result["indicator"]["f1"] == 0
    assert result["indicator"]["silence_fraction"] == 1
    assert result["initial_high_misses"] == 1


def test_score_threshold_and_indicator_settings_come_from_plan():
    result = experiment.score_rows(ScoresFromFeatures(), [row([.65, .65], [1, 1])],
        {"score_threshold": .7, "indicator_on": .6, "indicator_off": .3, "confirm_samples": 1})
    assert result["raw"]["f1"] == 0
    assert result["indicator"]["f1"] == 1
    assert result["initial_high_detection_ms"] == [100]


def test_privileged_metadata_and_body_values_cannot_change_neural_prediction():
    original = row([.9, .9, .5, .3, .3], [1, 1, 0, 0, 0])
    changed = copy.deepcopy(original)
    changed["hunger"][:] = 0
    changed["metadata"]["episode"].update(episode_id="different-human-event", initial={"energy": 100},
        neural_seed=0, background_seed=1234, hidden_food_coordinates=[900, 200], cue_id=103)
    a = experiment.score_rows(ScoresFromFeatures(), [original])
    b = experiment.score_rows(ScoresFromFeatures(), [changed])
    assert a == b


@pytest.mark.parametrize("truth,prediction", [([], []), ([1], [1, 0]), ([1, np.nan], [1, 1]), ([[1]], [[1]])])
def test_malformed_metric_inputs_cannot_silently_broadcast(truth, prediction):
    with pytest.raises(ValueError):
        experiment.metrics(truth, prediction)


def test_train_only_scaler_and_no_test_loading(tmp_path, monkeypatch):
    train_row = row([-2, -1, 1, 2], [0, 0, 1, 1], episode="train-fixture")
    validation_row = row([1000, 2000], [1, 0], episode="validation-fixture", seed=2)
    calls = []
    study = {"digest": "fixture-plan", "l2": .01, "max_iter": 100}
    def load(directory, split, condition="connected"):
        assert str(directory) == str(tmp_path)
        calls.append(split)
        assert split in {"train", "validation"}, "Training must never load held-out test"
        return study, [train_row if split == "train" else validation_row]
    monkeypatch.setattr(experiment, "load_episodes", load)
    experiment.train(tmp_path)
    model = LinearReadout.load(tmp_path / "state-readout.npz", expected_feature_digest=ScoresFromFeatures.feature_digest)
    np.testing.assert_array_equal(model.mean, train_row["X"].mean(axis=0))
    np.testing.assert_array_equal(model.scale, train_row["X"].std(axis=0))
    assert calls == ["train", "validation"]
    report = json.loads((tmp_path / "training.json").read_text())
    assert report["test_seen"] is False
    assert report["diagnostics"]["train_rows"] == 4


def test_all_scored_episode_feature_identities_must_match_model():
    a, b = row([.9], [1]), row([.9], [1], episode="second-episode")
    b["metadata"]["feature_digest"] = "another-calibration"
    with pytest.raises(ValueError, match="identities differ"):
        experiment.score_rows(ScoresFromFeatures(), [a, b])
