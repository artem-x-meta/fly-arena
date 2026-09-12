import copy
import json

import pytest

from semantic_tools.report import physical_cases, resource_evidence, validate_bindings
from semantic_tools.state_experiment import metrics


def binding_fixture():
    return {"training": {"model_sha256": "model", "plan_digest": "plan"},
            "results": {"model_sha256": "model", "plan_digest": "plan"},
            "physical": {"model_sha256": "model"},
            "study": {"digest": "plan", "calibration_digest": "calibration"},
            "physical_plan": {"model_sha256": "model", "calibration_sha256": "cal-file"},
            "mapping": {"digest": "mapping"},
            "calibration": {"digest": "calibration", "mapping_digest": "mapping"},
            "verification": {"status": "PASS", "semantic_code_before": "current-code",
                             "semantic_code_after": "current-code", "mapping_digest": "mapping",
                             "checks": {"resume": "PASS", "disabled": "PASS"}},
            "model_sha": "model", "calibration_sha": "cal-file", "current_code": "current-code"}


def test_report_accepts_matching_current_artifacts():
    args = binding_fixture()
    before = copy.deepcopy(args)
    validate_bindings(**args)
    assert args == before


@pytest.mark.parametrize("section,key", [
    ("training", "model_sha256"), ("results", "model_sha256"), ("physical", "model_sha256"),
    ("physical_plan", "model_sha256"), ("training", "plan_digest"), ("results", "plan_digest"),
    ("study", "calibration_digest"), ("physical_plan", "calibration_sha256"),
    ("calibration", "mapping_digest"), ("verification", "semantic_code_after"),
    ("verification", "mapping_digest"), ("verification", "status"),
])
def test_stale_or_mixed_artifacts_cannot_be_reported_as_current_success(section, key):
    args = binding_fixture()
    args[section][key] = "another-artifact"
    with pytest.raises(ValueError):
        validate_bindings(**args)


def test_failed_individual_check_blocks_verified_claim():
    args = binding_fixture()
    args["verification"]["checks"]["resume"] = "FAIL"
    with pytest.raises(ValueError, match="passing runtime"):
        validate_bindings(**args)


def test_physical_summary_must_match_all_planned_episode_evidence(tmp_path):
    sample = {"target": True, "score": .9, "active": True}
    row = {"episode": "1-hungry-connected", "seed": 1, "scene": "hungry", "condition": "connected",
           "duration_s": .1, "samples": [sample]}
    (tmp_path / "1-hungry-connected.json").write_text(json.dumps(row))
    plan = {"seeds": [1], "scenes": {"hungry": .1}, "conditions": ["connected"], "score_threshold": .5}
    correct = metrics([True], [True])
    summary = {"episodes": 1, "conditions": {"connected": {"raw": correct, "indicator": correct.copy()}}}
    assert physical_cases(tmp_path, plan, summary) == [row]
    summary["conditions"]["connected"]["indicator"]["f1"] = 0
    with pytest.raises(ValueError, match="pooled metrics"):
        physical_cases(tmp_path, plan, summary)


def test_missing_resource_profiles_are_unmeasured_not_zero(tmp_path):
    result, text = resource_evidence(tmp_path)
    assert result["status"] == "NOT_MEASURED"
    assert result["vram_peak_bytes"] is None
    assert "не измерены" in text


def test_resource_comparison_uses_same_duration_and_measured_values(tmp_path):
    base = {"simulation_seconds": 1, "loop_wall_seconds": 5, "ram_peak_bytes": 100 * 1024**2,
            "enabled": False}
    enabled = base | {"enabled": True, "loop_wall_seconds": 5.5, "ram_peak_bytes": 120 * 1024**2}
    (tmp_path / "profile-disabled.json").write_text(json.dumps(base))
    (tmp_path / "profile-enabled.json").write_text(json.dumps(enabled))
    result, _ = resource_evidence(tmp_path)
    assert result["loop_overhead_fraction"] == pytest.approx(.1)
    assert result["additional_peak_working_set_bytes"] == 20 * 1024**2
    assert result["vram_peak_bytes"] is None
    enabled["simulation_seconds"] = 2
    (tmp_path / "profile-enabled.json").write_text(json.dumps(enabled))
    with pytest.raises(ValueError, match="matched disabled/enabled"):
        resource_evidence(tmp_path)
