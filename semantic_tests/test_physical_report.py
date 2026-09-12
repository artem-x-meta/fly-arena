"""A failed or unbound physical pilot must not become a ready-demo report."""
import copy
import json

import numpy as np
import pytest

from fly_semantic.mapping import file_digest
from semantic_tools import physical_learning as learning
from semantic_tools import physical_report as report
from semantic_tests.test_physical_learning import ScoresFromFeatures, episode_row, study_fixture


def evidence_fixture():
    plan = study_fixture()
    plan.update(indicator_target_f1=.8, mapping_digest="mapping-sha", release_max_latency_ms=1000)
    rows = [episode_row(plan, ep) for ep in plan["episodes"] if ep["split"] == "test"]
    score = learning.score_rows(ScoresFromFeatures(), rows, plan)
    conditions = {key: copy.deepcopy(score) for key in ("connected", "homeostasis_off", "transmission_off", "cross_episode_shuffle")}
    results = {"conditions": conditions, "baseline_conditions": copy.deepcopy(conditions),
        "status": "TARGET_NOT_REACHED", "model_sha256": "model-v2-sha", "baseline_model_sha256": "model-v1-sha",
        "checks": {"indicator_target_f1": False},
        "paired_candidate_minus_baseline": {"connected": {"indicator": {"mean_seed_f1_difference": 0,
            "paired_seed_bootstrap_95pct": [0, 0]}}}}
    return {"plan": plan, "selection": {"selected_l2": .01, "trained_parameters": 2,
        "model_sha256": "model-v2-sha", "nominal_weights_sha256": "weight-sha", "validation": score},
        "results": results, "recorded": {"connected": rows},
        "binding_guard": {"preservation": {"checked_files": 254}},
        "study_dir": "fixture-study", "binding_path": "fixture-binding.json", "validation_history": []}


def test_failed_candidate_guard_stops_before_any_test_loading(tmp_path, monkeypatch):
    study = study_fixture()
    (tmp_path / "plan.json").write_text(json.dumps(study))
    binding = tmp_path / "binding.json"
    binding.write_text("{}")
    monkeypatch.setattr(report.integrity, "check_candidate", lambda path: {"pass": False})
    monkeypatch.setattr(learning, "load_rows", lambda *args: pytest.fail("Do not read tests after a failed guard"))
    with pytest.raises(ValueError, match="254 protected"):
        report.verify_evidence(tmp_path, binding)


def test_partial_preservation_check_does_not_authorize_positive_report(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps(study_fixture()))
    binding = tmp_path / "binding.json"
    binding.write_text("{}")
    monkeypatch.setattr(report.integrity, "check_candidate", lambda path: {"pass": True, "preservation": {"checked_files": 145}})
    with pytest.raises(ValueError, match="254 protected"):
        report.verify_evidence(tmp_path, binding)


def test_default_promotion_remains_pending_even_without_an_error():
    value = report.promotion_status(None, evidence_fixture())
    assert value["status"] == "PENDING"


def test_a_failed_pilot_cannot_be_promoted(tmp_path):
    path = tmp_path / "promotion.json"
    path.write_text(json.dumps({"status": "PROMOTED", "model_sha256": "model-v2-sha"}))
    with pytest.raises(ValueError, match="failed pilot"):
        report.promotion_status(path, evidence_fixture())


def test_pending_promotion_still_has_to_reference_the_evaluated_model(tmp_path):
    path = tmp_path / "promotion.json"
    path.write_text(json.dumps({"status": "PENDING", "model_sha256": "another-model"}))
    with pytest.raises(ValueError, match="different model"):
        report.promotion_status(path, evidence_fixture())


def test_report_separates_failed_experiment_from_a_demo_launch(tmp_path):
    evidence = evidence_fixture()
    text = report.render_report(evidence, {"status": "NOT_PROMOTED", "reason": "Критерии не выполнены."}, tmp_path)
    assert "TARGET_NOT_REACHED" in text
    assert "модель остаётся экспериментальной" in text
    assert "NOT_PROMOTED" in text
    assert "fly_semantic demo" not in text
    assert "а не новая свободная жизнь" in text
    assert "не означало готовности" in text


def test_plot_artifact_preserves_exact_old_new_scores_and_food_amounts(tmp_path):
    evidence = evidence_fixture()
    report._plot(evidence, tmp_path)
    for name in ("physical-head-comparison.png", "physical-head-comparison.svg", "physical-head-comparison.npz"):
        assert (tmp_path / name).stat().st_size > 100
    with np.load(tmp_path / "physical-head-comparison.npz", allow_pickle=False) as arrays:
        case = evidence["results"]["conditions"]["connected"]["cases"][0]
        np.testing.assert_array_equal(arrays["meal_0_new_scores"], case["scores"])
        np.testing.assert_array_equal(arrays["meal_0_actual_intake"], evidence["recorded"]["connected"][0]["intake"])
        assert json.loads(str(arrays["metadata"]))["model_sha256"] == "model-v2-sha"


def test_changed_recomputed_metrics_are_rejected():
    with pytest.raises(ValueError, match="changed"):
        report._same({"f1": .5}, {"f1": .9}, "changed")
