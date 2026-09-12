"""Candidate and preservation guards must reject changed scientific artifacts."""
import json

import pytest

from semantic_tools.v2_integrity import bind_candidate, check, check_candidate, semantic_digest, sha256


@pytest.fixture
def study(tmp_path):
    runtime = tmp_path / "fly_semantic/runtime.py"
    runtime.parent.mkdir()
    runtime.write_text("# frozen runtime\n", encoding="utf-8")
    protected = tmp_path / "old-head.npz"
    protected.write_bytes(b"preserved v1")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "schema_version": 1, "kind": "semantic_v2_preservation_baseline",
        "semantic_code_digest": semantic_digest(tmp_path),
        "categories": {"old_model": {"old-head.npz": {"sha256": sha256(protected), "bytes": protected.stat().st_size}}},
    }), encoding="utf-8")
    roles = ("model", "mapping", "calibration", "study_plan", "training_records", "validation_records", "source:collector")
    artifacts = {}
    for index, role in enumerate(roles):
        path = tmp_path / f"artifact-{index}.json"
        path.write_text(json.dumps({"role": role}), encoding="utf-8")
        artifacts[role] = path
    return tmp_path, baseline, artifacts


def bind(study):
    root, baseline, artifacts = study
    destination = root / "candidate.json"
    bind_candidate(destination, artifacts, [root / "test/results.json"], manifest=baseline, root=root)
    return destination


def test_binding_tracks_exact_head_and_does_not_require_test_absence_forever(study):
    root, _, artifacts = study
    destination = bind(study)
    assert check_candidate(destination, root=root)["pass"]
    result = root / "test/results.json"
    result.parent.mkdir()
    result.write_text("{}", encoding="utf-8")
    assert check_candidate(destination, root=root)["pass"]
    artifacts["model"].write_text("changed head", encoding="utf-8")
    actual = check_candidate(destination, root=root)
    assert not actual["pass"]
    assert [item["role"] for item in actual["changed_or_missing_artifacts"]] == ["model"]


def test_existing_test_results_cannot_receive_prospective_binding(study):
    root, baseline, artifacts = study
    results = root / "results.json"
    results.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="absent before binding"):
        bind_candidate(root / "candidate.json", artifacts, [results], manifest=baseline, root=root)
    assert not (root / "candidate.json").exists()


def test_binding_cannot_be_replaced_after_evaluation(study):
    destination = bind(study)
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        bind(study)
    assert destination.read_bytes() == before


def test_new_runtime_file_breaks_old_checkpoint_digest_but_new_tool_does_not(study):
    root, baseline, _ = study
    tool = root / "semantic_tools/new_tool.py"
    tool.parent.mkdir()
    tool.write_text("# allowed addition", encoding="utf-8")
    assert check(baseline, root=root)["pass"]
    (root / "fly_semantic/new_module.py").write_text("# incompatible addition", encoding="utf-8")
    actual = check(baseline, root=root)
    assert not actual["pass"]
    assert not actual["semantic_code_matches"]


def test_legacy_file_change_prevents_candidate_binding(study):
    root, baseline, artifacts = study
    (root / "old-head.npz").write_bytes(b"edited v1")
    actual = check(baseline, root=root)
    assert not actual["pass"]
    assert actual["categories"]["old_model"]["changed_or_missing"][0]["path"] == "old-head.npz"
    with pytest.raises(ValueError, match="Protected v1 files changed"):
        bind_candidate(root / "candidate.json", artifacts, [root / "test/results.json"], manifest=baseline, root=root)


def test_replaced_baseline_cannot_hide_changed_artifacts(study):
    root, baseline, _ = study
    destination = bind(study)
    document = json.loads(baseline.read_text(encoding="utf-8"))
    document["categories"] = {}
    baseline.write_text(json.dumps(document), encoding="utf-8")
    actual = check_candidate(destination, root=root)
    assert not actual["pass"]
    assert not actual["baseline_manifest_matches"]


def test_binding_rejects_paths_outside_workspace(study):
    root, baseline, artifacts = study
    with pytest.raises(ValueError, match="within the workspace"):
        bind_candidate(root / "candidate.json", artifacts, [root.parent / "outside.json"], manifest=baseline, root=root)


def test_complete_artifact_roles_and_future_outputs_are_required(study):
    root, baseline, artifacts = study
    with pytest.raises(ValueError, match="outputs are required"):
        bind_candidate(root / "candidate.json", artifacts, [], manifest=baseline, root=root)
    artifacts.pop("training_records")
    with pytest.raises(ValueError, match="training and validation"):
        bind_candidate(root / "candidate.json", artifacts, [root / "test/results.json"], manifest=baseline, root=root)
