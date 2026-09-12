"""Preserve v1 artifacts and bind a v2 candidate before held-out evaluation.

This module reads simulator files and writes only explicitly requested JSON
reports. It never constructs an arena, edits a model, or launches an experiment.
The baseline is append-only: new v2 files are permitted, existing v1 files are
checked byte for byte. Candidate binding also records the expected test output
paths and requires that none exists when the binding is first published.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports/semantic_channel_v2/baseline-manifest.json"
V1_MANIFEST = "reports/semantic_channel/protected-manifest.json"
V1_TOOLS = (
    "audit.py", "calibrate.py", "physical_transfer.py", "profile.py", "report.py",
    "state_experiment.py", "transition_metrics.py", "verify_runtime.py",
)
V1_TESTS = (
    "test_cli.py", "test_features.py", "test_mapping.py", "test_protocol.py",
    "test_readout.py", "test_runtime.py", "test_semantic_report.py",
    "test_state_experiment.py", "test_transition_metrics.py",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _path(value, root=ROOT):
    root = Path(root).resolve()
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact must remain within the workspace: {value}")
    return path


def _relative(value, root=ROOT):
    return _path(value, root).relative_to(Path(root).resolve()).as_posix()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def semantic_digest(root=ROOT):
    """Same digest algorithm as runtime, without importing the simulator."""
    files = {path.name: sha256(path) for path in sorted((Path(root) / "fly_semantic").glob("*.py"))}
    return hashlib.sha256(json.dumps(files, sort_keys=True, allow_nan=False).encode()).hexdigest()


def core_digest(root=ROOT):
    result = hashlib.sha256()
    for path in sorted((Path(root) / "fly_arena").glob("*.py")):
        result.update(path.name.encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def selected_categories(root=ROOT):
    root = Path(root)
    old = _read(root / V1_MANIFEST)
    categories = {"legacy_protected": [root / name for name in old["files"]]}
    categories["semantic_runtime"] = list((root / "fly_semantic").glob("*.py"))
    categories["semantic_tools_v1"] = [root / "semantic_tools" / name for name in V1_TOOLS]
    categories["semantic_tests_v1"] = [root / "semantic_tests" / name for name in V1_TESTS]
    categories["semantic_configs_v1"] = [root / "semantic_configs/default.toml", root / "semantic_configs/need-food.toml"]
    categories["semantic_launcher_v1"] = [root / "launchers" / "18_run_semantic_channel.cmd"]
    categories["semantic_docs_v1"] = [root / "docs/SEMANTIC_CHANNEL.md"]
    categories["semantic_reports_v1"] = [p for p in (root / "reports/semantic_channel").glob("*") if p.is_file()]
    runs = root / "runs/semantic-v1"
    categories["semantic_calibrations_v1"] = sorted(runs.glob("calibration*/*.json"))
    categories["semantic_heads_and_plans_v1"] = sorted(
        p for folder in ("state-study", "state-selected") for p in (runs / folder).glob("*")
        if p.is_file() and p.suffix in (".json", ".npz"))
    categories["semantic_physical_results_v1"] = sorted((runs / "physical-selected").glob("*.json"))
    categories["semantic_checkpoints_v1"] = sorted(
        p for folder in ("smoke", "smoke-final", "smoke-resume-final", "verification")
        for p in (runs / folder).rglob("*")
        if p.is_file() and p.suffix in (".json", ".npz", ".xml", ".jsonl"))
    return categories


def freeze(manifest=DEFAULT_MANIFEST, *, root=ROOT):
    root = Path(root).resolve()
    manifest = _path(manifest, root)
    if manifest.exists():
        raise FileExistsError("Baseline already frozen; use check")
    old = _read(root / V1_MANIFEST)
    changed = [name for name, expected in old["files"].items()
               if not (root / name).is_file() or sha256(root / name) != expected]
    if changed:
        raise ValueError(f"Previous protected baseline already differs: {changed}")
    records = {}
    for category, paths in selected_categories(root).items():
        records[category] = {_relative(p, root): {"sha256": sha256(p), "bytes": p.stat().st_size}
                             for p in sorted(set(paths))}
    result = {
        "schema_version": 1, "kind": "semantic_v2_preservation_baseline", "created_utc": timestamp(),
        "semantic_code_digest": semantic_digest(root), "legacy_core_code_digest": old["core_code_digest"],
        "previous_manifest": V1_MANIFEST, "previous_manifest_sha256": sha256(root / V1_MANIFEST),
        "previous_protected_count": len(old["files"]), "categories": records,
        "file_count": sum(len(items) for items in records.values()),
        "selection": "Existing legacy protected files plus semantic v1 runtime, tools, tests, configs, launcher, documentation, reports, calibrations, heads, plans, physical results, and verification/smoke checkpoints. Large neural episode NPZ collections are intentionally outside this preservation manifest.",
        "new_files_policy": "New sibling tools/configs/reports may be added. Any fly_semantic Python addition changes the runtime digest and fails compatibility.",
    }
    _write_new(manifest, result)
    return result


def check(manifest=DEFAULT_MANIFEST, *, root=ROOT):
    root = Path(root).resolve()
    manifest = _path(manifest, root)
    frozen = _read(manifest)
    if frozen.get("kind") != "semantic_v2_preservation_baseline" or frozen.get("schema_version") != 1:
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
    actual_semantic = semantic_digest(root)
    actual_core = core_digest(root)
    core_matches = ("legacy_core_code_digest" not in frozen or actual_core == frozen["legacy_core_code_digest"])
    return {
        "checked_utc": timestamp(), "manifest": _relative(manifest, root), "manifest_sha256": sha256(manifest),
        "checked_files": sum(item["checked_files"] for item in categories.values()), "categories": categories,
        "semantic_code_digest": actual_semantic,
        "semantic_code_matches": actual_semantic == frozen["semantic_code_digest"],
        "legacy_core_code_digest": actual_core, "legacy_core_code_matches": core_matches,
        "pass": core_matches and actual_semantic == frozen["semantic_code_digest"] and all(item["pass"] for item in categories.values()),
    }


def validate_collection_artifacts(study, graph, *, episodes=(), manifest=DEFAULT_MANIFEST, root=ROOT):
    """External guard for the already frozen physical collector.

    Run before each collection/replay batch and after collection. This validates
    the sources and all v1 artifacts without modifying the prospectively frozen
    collector. Reconstructing nominal weights calls the unchanged pure graph
    conversion only; no brain steps, organism or physics are constructed.
    Optional episode NPZ paths verify metadata against these same identities.
    """
    from dataclasses import asdict
    import numpy as np
    from fly_arena.brain import BrainConfig, signed_weights
    from fly_semantic.mapping import digest, load_mapping
    from fly_semantic.runtime import load_calibration, make_features

    root = Path(root).resolve()
    graph = _path(graph, root)
    document = dict(study)
    if document.pop("digest", None) != digest(document) or study.get("schema_version") != 2:
        raise ValueError("Collection plan digest/schema mismatch")
    preservation = check(manifest, root=root)
    if not preservation["pass"]:
        raise ValueError("Protected v1 artifacts changed before collection/replay")
    source_differences = [name for name, expected in study["source_hashes"].items()
                          if not _path(name, root).is_file() or sha256(_path(name, root)) != expected]
    if source_differences:
        raise ValueError(f"Frozen collector/runtime source changed: {source_differences}")
    calibration_dir = _path(study["calibration_dir"], root)
    mapping = load_mapping(calibration_dir / "mapping.json", graph)
    calibration = load_calibration(calibration_dir / "calibration.json", mapping)
    features = make_features(mapping, calibration)
    actual_identities = {"mapping_digest": mapping["digest"], "calibration_digest": calibration["digest"],
                         "feature_digest": features.feature_digest}
    if any(study[name] != value for name, value in actual_identities.items()):
        raise ValueError("Current mapping/calibration/features differ from frozen collection plan")
    config = asdict(BrainConfig())
    if config != study["brain_config"] or config != calibration["brain_config"]:
        raise ValueError("Physical collector default BrainConfig differs from calibrated/frozen config")
    baseline_model = _path(study["baseline_model_path"], root)
    if sha256(baseline_model) != study["baseline_model_sha256"]:
        raise ValueError("Frozen baseline comparison head changed")
    ptr = np.load(graph / "ptr.npy", mmap_mode="r")
    counts = np.load(graph / "counts.npy", mmap_mode="r")
    signs = np.load(graph / "signs.npy", mmap_mode="r")
    weights = signed_weights(ptr, counts, signs, config["synapse_gain"])
    nominal_sha = hashlib.sha256(memoryview(weights)).hexdigest()
    del weights
    prior_nominal = _read(root / "reports/semantic_channel/baseline-performance.json")["runtime_weights_sha256"]
    if nominal_sha != prior_nominal:
        raise ValueError("Reconstructed nominal graph weights differ from protected v1 runtime weights")
    episode_records = []
    planned = {ep["episode_id"]: ep for ep in study["episodes"]}
    for value in episodes:
        path = _path(value, root)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        ep = metadata["episode"]
        if (metadata["plan_digest"] != study["digest"] or planned.get(ep["episode_id"]) != ep
                or metadata["feature_digest"] != study["feature_digest"]
                or metadata["calibration_digest"] != study["calibration_digest"]
                or metadata["nominal_weights_sha256"] != nominal_sha
                or metadata["source_hashes"] != study["source_hashes"]):
            raise ValueError(f"Recorded episode provenance differs from current frozen plan: {path}")
        episode_records.append({"path": _relative(path, root), "sha256": sha256(path), "condition": metadata["condition"]})
    return {
        "checked_utc": timestamp(), "pass": True, "plan_digest": study["digest"],
        "protected_files_checked": preservation["checked_files"],
        "baseline_manifest_sha256": preservation["manifest_sha256"],
        **actual_identities, "brain_config": config, "nominal_weights_sha256": nominal_sha,
        "nominal_weights_match_protected_v1": True, "source_files_checked": len(study["source_hashes"]),
        "episode_artifacts_checked": episode_records,
        "method": "External pre/post batch guard; original collector remains byte-identical to its frozen plan. Nominal weights reconstructed by existing signed_weights without simulation.",
    }


def bind_candidate(destination, artifacts, evaluation_outputs, *, manifest=DEFAULT_MANIFEST, root=ROOT):
    """Freeze named files before test; callers must list every planned output.

    Required roles: model, mapping, calibration, study_plan, training_records,
    validation_records. Add all collector/trainer/evaluator source files as
    source:<name> roles. Record manifests may themselves bind episode arrays;
    callers should include the arrays as extra roles for direct byte checks.
    This is an artifact-integrity guard, not proof that hidden test data have
    never been observed elsewhere. No test data may be used to select a head.
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
        raise ValueError("Protected v1 files changed; refusing candidate binding")
    future = [_path(path, root) for path in evaluation_outputs]
    if len(set(future)) != len(future) or any(path.exists() for path in future):
        raise ValueError("All expected test outputs must be unique and absent before binding")
    records = {role: {"path": _relative(path, root), "sha256": sha256(_path(path, root))}
               for role, path in sorted(artifacts.items())}
    if _relative(destination, root) in [record["path"] for record in records.values()]:
        raise ValueError("Candidate cannot include itself")
    result = {
        "schema_version": 1, "kind": "semantic_v2_prospective_candidate", "bound_utc": timestamp(),
        "baseline_manifest": preservation["manifest"], "baseline_manifest_sha256": preservation["manifest_sha256"],
        "semantic_code_digest": preservation["semantic_code_digest"], "artifacts": records,
        "evaluation_outputs_absent_at_binding": [_relative(path, root) for path in future],
        "policy": "Use this exact candidate for every held-out condition. Failures are retained; tuning after evaluation requires a fresh test set.",
    }
    _write_new(destination, result)
    return result


def check_candidate(binding, *, root=ROOT):
    root = Path(root).resolve()
    binding = _path(binding, root)
    frozen = _read(binding)
    if frozen.get("kind") != "semantic_v2_prospective_candidate" or frozen.get("schema_version") != 1:
        raise ValueError("Unrecognized candidate binding")
    differences = []
    for role, item in frozen["artifacts"].items():
        path = _path(item["path"], root)
        actual = sha256(path) if path.is_file() else None
        if actual != item["sha256"]:
            differences.append({"role": role, "path": item["path"], "expected_sha256": item["sha256"], "actual_sha256": actual})
    baseline = _path(frozen["baseline_manifest"], root)
    baseline_matches = baseline.is_file() and sha256(baseline) == frozen["baseline_manifest_sha256"]
    preservation = check(baseline, root=root) if baseline_matches else None
    runtime_matches = semantic_digest(root) == frozen["semantic_code_digest"]
    return {"checked_utc": timestamp(), "binding_sha256": sha256(binding), "changed_or_missing_artifacts": differences,
            "baseline_manifest_matches": baseline_matches, "semantic_code_matches": runtime_matches,
            "preservation": preservation,
            "pass": not differences and baseline_matches and runtime_matches and preservation["pass"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "check", "bind-candidate", "check-candidate"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--artifacts", type=Path, help="JSON mapping from artifact roles to workspace paths")
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
