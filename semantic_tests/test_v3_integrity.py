"""Preserve completed studies and reject changed prospective v3 artifacts."""
import json

import pytest

from semantic_tools.v3_integrity import (
    bind_candidate, check, check_candidate, core_digest, semantic_digest, sha256,
)


@pytest.fixture
def study(tmp_path):
    for folder in ("fly_semantic", "fly_arena"):
        source = tmp_path / folder / "runtime.py"
        source.parent.mkdir()
        source.write_text("# completed v2\n", encoding="utf-8")
    protected = tmp_path / "old-episode.npz"
    protected.write_bytes(b"completed v2 episode")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "schema_version": 1, "kind": "semantic_v3_preservation_baseline",
        "semantic_code_digest": semantic_digest(tmp_path),
        "legacy_core_code_digest": core_digest(tmp_path),
        "categories": {"old_v2_data": {"old-episode.npz": {"sha256": sha256(protected), "bytes": protected.stat().st_size}}},
    }), encoding="utf-8")
    roles = ("model", "mapping", "calibration", "study_plan", "training_records",
             "validation_records", "source:collector", "data:train-4101")
    artifacts = {}
    for index, role in enumerate(roles):
        artifact = tmp_path / f"artifact-{index}.json"
        artifact.write_text(json.dumps({"role": role}), encoding="utf-8")
        artifacts[role] = artifact
    return tmp_path, baseline, artifacts


def bind(study):
    root, baseline, artifacts = study
    destination = root / "candidate.json"
    bind_candidate(destination, artifacts, [root / "test/results.json"], manifest=baseline, root=root)
    return destination


@pytest.mark.parametrize("role", ["model", "calibration", "data:train-4101", "source:collector"])
def test_binding_rejects_mutated_candidate_components(study, role):
    root, _, artifacts = study
    binding = bind(study)
    assert check_candidate(binding, root=root)["pass"]
    artifacts[role].write_text("changed after binding", encoding="utf-8")
    result = check_candidate(binding, root=root)
    assert not result["pass"]
    assert [item["role"] for item in result["changed_or_missing_artifacts"]] == [role]


def test_finished_outputs_are_allowed_without_rebinding(study):
    root, _, _ = study
    binding = bind(study)
    output = root / "test/results.json"
    output.parent.mkdir()
    output.write_text("{}", encoding="utf-8")
    assert check_candidate(binding, root=root)["pass"]
    before = binding.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        bind(study)
    assert binding.read_bytes() == before


def test_existing_test_cannot_be_labeled_prospective(study):
    root, baseline, artifacts = study
    output = root / "result.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="absent before binding"):
        bind_candidate(root / "candidate.json", artifacts, [output], manifest=baseline, root=root)
    assert not (root / "candidate.json").exists()


@pytest.mark.parametrize("folder", ["fly_arena", "fly_semantic"])
def test_python_addition_in_frozen_package_breaks_digest(study, folder):
    root, baseline, _ = study
    (root / folder / "new_module.py").write_text("# changes checkpoint identity", encoding="utf-8")
    assert not check(baseline, root=root)["pass"]


def test_new_sibling_module_preserves_old_runtime(study):
    root, baseline, _ = study
    new_source = root / "semantic_tools/v3_gain.py"
    new_source.parent.mkdir()
    new_source.write_text("# permitted sibling", encoding="utf-8")
    assert check(baseline, root=root)["pass"]


def test_missing_completed_v2_episode_prevents_binding(study):
    root, baseline, artifacts = study
    (root / "old-episode.npz").unlink()
    assert not check(baseline, root=root)["pass"]
    with pytest.raises(ValueError, match="Protected v1/v2"):
        bind_candidate(root / "candidate.json", artifacts, [root / "result.json"], manifest=baseline, root=root)


def test_replacing_manifest_cannot_redefine_old_baseline(study):
    root, baseline, _ = study
    binding = bind(study)
    document = json.loads(baseline.read_text(encoding="utf-8"))
    document["categories"] = {}
    baseline.write_text(json.dumps(document), encoding="utf-8")
    result = check_candidate(binding, root=root)
    assert not result["pass"]
    assert not result["baseline_manifest_matches"]


@pytest.mark.parametrize("malformation", ["missing_role", "no_source", "no_outputs", "duplicate_output", "binding_output", "escape_workspace"])
def test_incomplete_or_invalid_binding_never_publishes(study, malformation):
    root, baseline, artifacts = study
    destination = root / "candidate.json"
    outputs = [root / "result.json"]
    if malformation == "missing_role":
        artifacts.pop("training_records")
    elif malformation == "no_source":
        artifacts.pop("source:collector")
    elif malformation == "no_outputs":
        outputs = []
    elif malformation == "duplicate_output":
        outputs *= 2
    elif malformation == "binding_output":
        outputs = [destination]
    else:
        outputs = [root.parent / "outside.json"]
    with pytest.raises(ValueError):
        bind_candidate(destination, artifacts, outputs, manifest=baseline, root=root)
    assert not destination.exists()
