"""Append-only preservation and prospective candidate binding for semantic v3.

Reuse the v2 byte/hash utilities without changing the candidate-bound v2 sources.
Only this module's explicit destination reports are written; no simulator starts.
The baseline extends all 254 protected v1 records with the completed v2 study,
including every recorded physical/control/derived episode and old checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from semantic_tools import v2_integrity as prior
from semantic_tools.v2_integrity import (
    ROOT, _path, _read, _relative, _write_new, core_digest, semantic_digest,
    sha256, timestamp,
)


DEFAULT_MANIFEST = ROOT / "reports/semantic_channel_v3/baseline-manifest.json"
PREVIOUS_MANIFEST = "reports/semantic_channel_v2/baseline-manifest.json"
PREVIOUS_BINDING = "runs/semantic-v2/physical-study/candidate-binding.json"
EXPECTED_SEMANTIC_DIGEST = "4ed2cbfba1d03562da863e0b8ec725faa8e4783d46a0765a52529175caafb57b"
EXPECTED_CORE_DIGEST = "04a519569f0a52b75d6b6666f5f3fe6523b03236cfd5fcd8daacc3a831c789ff"
V2_TOOLS = (
    "physical_collection.py", "physical_learning.py", "physical_mixed.py",
    "physical_report.py", "v2_derivation_audit.py", "v2_diagnostics.py", "v2_integrity.py",
)
V2_TESTS = (
    "test_physical_collection.py", "test_physical_learning.py",
    "test_physical_report.py", "test_v2_integrity.py",
)


def selected_categories(root=ROOT):
    """Explicit v2 source names exclude concurrently created v3 sibling files."""
    root = Path(root).resolve()
    old = _read(root / PREVIOUS_MANIFEST)
    categories = {name: [root / path for path in records]
                  for name, records in old["categories"].items()}
    categories["semantic_tools_v2"] = [root / "semantic_tools" / name for name in V2_TOOLS]
    categories["semantic_tests_v2"] = [root / "semantic_tests" / name for name in V2_TESTS]
    categories["semantic_docs_v2"] = [root / "docs/SEMANTIC_CHANNEL_V2.md"]
    categories["semantic_reports_v2"] = sorted(
        p for p in (root / "reports/semantic_channel_v2").glob("*") if p.is_file())
    runs = root / "runs/semantic-v2"
    categories["semantic_study_and_checkpoint_artifacts_v2"] = sorted(
        p for p in runs.rglob("*") if p.is_file() and "episodes" not in p.relative_to(runs).parts)
    categories["semantic_physical_episodes_v2"] = sorted((runs / "physical-study/episodes/connected").glob("*.npz"))
    categories["semantic_replay_controls_v2"] = sorted(
        p for condition in ("homeostasis_off", "transmission_off", "replay_check")
        for p in (runs / "physical-study/episodes" / condition).glob("*.npz"))
    categories["semantic_mixed_copies_v2"] = sorted((runs / "mixed-study/episodes/connected").glob("*.npz"))
    return categories


def freeze(manifest=DEFAULT_MANIFEST, *, root=ROOT):
    root = Path(root).resolve()
    manifest = _path(manifest, root)
    if manifest.exists():
        raise FileExistsError("Baseline already frozen; use check")
    previous = prior.check(root / PREVIOUS_MANIFEST, root=root)
    candidate = prior.check_candidate(root / PREVIOUS_BINDING, root=root)
    if not previous["pass"] or not candidate["pass"]:
        raise ValueError("Previous protected baseline or frozen v2 candidate already differs")
    if semantic_digest(root) != EXPECTED_SEMANTIC_DIGEST or core_digest(root) != EXPECTED_CORE_DIGEST:
        raise ValueError("Current runtime digests differ from completed v2")
    # The final audit also binds files written after v2's prospective candidate.
    audit = _read(root / "reports/semantic_channel_v2/final-audit.json")
    changed = [name for name, expected in audit["artifacts"].items()
               if not _path(name, root).is_file() or sha256(_path(name, root)) != expected]
    if changed:
        raise ValueError(f"Completed v2 audit artifacts already differ: {changed}")
    records = {
        category: {_relative(p, root): {"sha256": sha256(p), "bytes": p.stat().st_size}
                   for p in sorted(set(paths))}
        for category, paths in selected_categories(root).items()
    }
    all_names = [name for items in records.values() for name in items]
    if len(all_names) != len(set(all_names)):
        raise ValueError("Preservation categories unexpectedly overlap")
    result = {
        "schema_version": 1, "kind": "semantic_v3_preservation_baseline", "created_utc": timestamp(),
        "semantic_code_digest": EXPECTED_SEMANTIC_DIGEST, "legacy_core_code_digest": EXPECTED_CORE_DIGEST,
        "previous_manifest": PREVIOUS_MANIFEST, "previous_manifest_sha256": sha256(root / PREVIOUS_MANIFEST),
        "previous_candidate": PREVIOUS_BINDING, "previous_candidate_sha256": sha256(root / PREVIOUS_BINDING),
        "previous_protected_count": previous["checked_files"], "categories": records,
        "file_count": len(all_names),
        "selection": "All previous 254 preservation records plus explicit v2 tools/tests/docs, every v2 report, every v2 study/head/checkpoint artifact, all 36 connected physical episodes, 24 replay controls, 2 exact replay records, and 24 mixed-study copies. v1 calibration files remain covered by the previous records. No v3 paths are selected.",
        "new_files_policy": "New sibling tools/configs/reports/runs are permitted. Any change or Python addition in fly_arena or fly_semantic fails the runtime digest check.",
    }
    _write_new(manifest, result)
    return result


def check(manifest=DEFAULT_MANIFEST, *, root=ROOT):
    root = Path(root).resolve()
    manifest = _path(manifest, root)
    frozen = _read(manifest)
    if frozen.get("kind") != "semantic_v3_preservation_baseline" or frozen.get("schema_version") != 1:
        raise ValueError("Unrecognized preservation manifest")
    categories = {}
    for category, records in frozen["categories"].items():
        changes = []
        for name, record in records.items():
            path = _path(name, root)
            actual = sha256(path) if path.is_file() else None
            if actual != record["sha256"]:
                changes.append({"path": name, "expected_sha256": record["sha256"], "actual_sha256": actual})
        categories[category] = {"checked_files": len(records), "changed_or_missing": changes, "pass": not changes}
    actual_semantic, actual_core = semantic_digest(root), core_digest(root)
    semantic_matches = actual_semantic == frozen["semantic_code_digest"]
    core_matches = actual_core == frozen["legacy_core_code_digest"]
    return {
        "checked_utc": timestamp(), "manifest": _relative(manifest, root), "manifest_sha256": sha256(manifest),
        "checked_files": sum(item["checked_files"] for item in categories.values()), "categories": categories,
        "semantic_code_digest": actual_semantic, "semantic_code_matches": semantic_matches,
        "legacy_core_code_digest": actual_core, "legacy_core_code_matches": core_matches,
        "pass": semantic_matches and core_matches and all(item["pass"] for item in categories.values()),
    }


def bind_candidate(destination, artifacts, evaluation_outputs, *, manifest=DEFAULT_MANIFEST, root=ROOT):
    """Bind exact v3 sources/model/calibration/data before every listed test output.

    As in v2, required roles cover model, mapping, calibration, study plan,
    training records, validation records and source:<name>. Include every new
    collector/trainer/evaluator plus array files as explicit additional roles.
    Origin is the v3 preservation manifest, whose SHA is embedded in the binding.
    This does not prove that data have never been inspected elsewhere.
    """
    root = Path(root).resolve()
    required = {"model", "mapping", "calibration", "study_plan", "training_records", "validation_records"}
    if not required.issubset(artifacts) or not any(role.startswith("source:") for role in artifacts):
        raise ValueError("Candidate must bind model/mapping/calibration/study plan, training and validation records, and source files")
    if not evaluation_outputs:
        raise ValueError("Explicit future evaluation outputs are required")
    destination = _path(destination, root)
    if destination.exists():
        raise FileExistsError("Candidate binding is immutable; use a fresh study")
    preservation = check(manifest, root=root)
    if not preservation["pass"]:
        raise ValueError("Protected v1/v2 files changed; refusing candidate binding")
    future = [_path(path, root) for path in evaluation_outputs]
    if len(set(future)) != len(future) or any(path.exists() for path in future):
        raise ValueError("All expected test outputs must be unique and absent before binding")
    if destination in future:
        raise ValueError("Candidate binding cannot also be a future evaluation output")
    records = {role: {"path": _relative(path, root), "sha256": sha256(_path(path, root))}
               for role, path in sorted(artifacts.items())}
    result = {
        "schema_version": 1, "kind": "semantic_v3_prospective_candidate", "bound_utc": timestamp(),
        "baseline_manifest": preservation["manifest"], "baseline_manifest_sha256": preservation["manifest_sha256"],
        "semantic_code_digest": preservation["semantic_code_digest"],
        "legacy_core_code_digest": preservation["legacy_core_code_digest"], "artifacts": records,
        "evaluation_outputs_absent_at_binding": [_relative(path, root) for path in future],
        "policy": "Use this exact candidate for every held-out condition. Failures are retained; tuning after evaluation requires a fresh test set. Previously observed v2 test episodes are development data for v3 and cannot serve as its prospective test.",
    }
    _write_new(destination, result)
    return result


def check_candidate(binding, *, root=ROOT):
    root = Path(root).resolve()
    binding = _path(binding, root)
    frozen = _read(binding)
    if frozen.get("kind") != "semantic_v3_prospective_candidate" or frozen.get("schema_version") != 1:
        raise ValueError("Unrecognized candidate binding")
    differences = []
    for role, record in frozen["artifacts"].items():
        path = _path(record["path"], root)
        actual = sha256(path) if path.is_file() else None
        if actual != record["sha256"]:
            differences.append({"role": role, "path": record["path"], "expected_sha256": record["sha256"], "actual_sha256": actual})
    baseline = _path(frozen["baseline_manifest"], root)
    baseline_matches = baseline.is_file() and sha256(baseline) == frozen["baseline_manifest_sha256"]
    preservation = check(baseline, root=root) if baseline_matches else None
    semantic_matches = semantic_digest(root) == frozen["semantic_code_digest"]
    core_matches = core_digest(root) == frozen["legacy_core_code_digest"]
    return {
        "checked_utc": timestamp(), "binding_sha256": sha256(binding), "changed_or_missing_artifacts": differences,
        "baseline_manifest_matches": baseline_matches, "semantic_code_matches": semantic_matches,
        "legacy_core_code_matches": core_matches, "preservation": preservation,
        "pass": bool(not differences and baseline_matches and semantic_matches and core_matches and preservation["pass"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "check", "bind-candidate", "check-candidate"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--evaluation-output", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.manifest)
    elif args.action == "check":
        result = check(args.manifest)
    elif args.action == "bind-candidate":
        if args.binding is None or args.artifacts is None:
            parser.error("bind-candidate requires --binding and --artifacts")
        result = bind_candidate(args.binding, _read(args.artifacts), args.evaluation_output, manifest=args.manifest)
    else:
        if args.binding is None:
            parser.error("check-candidate requires --binding")
        result = check_candidate(args.binding)
    if args.output is not None:
        _write_new(_path(args.output), result)
    summary = {key: value for key, value in result.items() if key not in ("categories", "artifacts", "preservation")}
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if result.get("pass") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
