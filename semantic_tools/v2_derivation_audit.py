"""Independent read-only audit of physical-to-mixed training derivation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fly_semantic.mapping import digest
from semantic_tools.v2_integrity import sha256, timestamp


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(source=Path("runs/semantic-v2/physical-study"), derived=Path("runs/semantic-v2/mixed-study"),
          snapshot=Path("reports/semantic_channel_v2/derivation-source-snapshot.json")):
    source, derived = Path(source), Path(derived)
    before = read(snapshot)
    parent, study = read(source / "plan.json"), read(derived / "plan.json")
    differences = []
    for kind, plan in (("parent", parent), ("derived", study)):
        if plan["digest"] != digest({k: v for k, v in plan.items() if k != "digest"}):
            differences.append(f"{kind} plan digest invalid")
    if sha256(source / "plan.json") != before["physical_plan_sha256"]:
        differences.append("original physical plan changed after independent source snapshot")
    for category in ("physical_train_validation", "neural_train_only"):
        for path, expected in before[category].items():
            if sha256(path) != expected:
                differences.append(f"original source changed: {path}")
    for key, value in parent.items():
        if key not in ("digest", "scope") and study.get(key) != value:
            differences.append(f"original protocol/criteria changed: {key}")
    permitted_additions = {"include_synthetic_train", "supplemental_train", "supplemental_plan_path",
                          "supplemental_plan_sha256", "derived_from", "derivation_source_sha256"}
    if set(study) - set(parent) != permitted_additions:
        differences.append("unexpected added/removed derived-plan fields")
    if study.get("include_synthetic_train") is not True:
        differences.append("supplemental training not declared")
    if study["derived_from"]["plan_digest"] != parent["digest"] or study["derived_from"]["plan_sha256"] != sha256(source / "plan.json"):
        differences.append("parent plan lineage mismatch")
    if sha256("semantic_tools/physical_mixed.py") != study["derivation_source_sha256"]:
        differences.append("derivation helper changed")
    training = read(source / "training.json")
    if study["derived_from"]["parent_model_sha256"] != training["model_sha256"]:
        differences.append("rejected parent head SHA differs")
    if study["derived_from"]["parent_validation"] != training["validation"]:
        differences.append("parent validation results differ")
    physical = []
    array_count = 0
    for ep in parent["episodes"]:
        if ep["split"] not in ("train", "validation"):
            continue
        src = source / "episodes/connected" / (ep["episode_id"] + ".npz")
        dst = derived / "episodes/connected" / src.name
        local_differences = []
        with np.load(src, allow_pickle=False) as original, np.load(dst, allow_pickle=False) as copied:
            if set(original.files) != set(copied.files):
                local_differences.append("archive members differ")
            for name in sorted(set(original.files) & set(copied.files) - {"metadata"}):
                left, right = original[name], copied[name]
                array_count += 1
                if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
                    local_differences.append(f"array dtype/shape/bytes differ: {name}")
            metadata = read_metadata(original)
            actual = read_metadata(copied)
            expected = dict(metadata, plan_digest=study["digest"], original_plan_digest=parent["digest"],
                derived_from_path=src.as_posix(), derived_from_sha256=sha256(src),
                derivation="Existing physical measurements copied unchanged; only plan lineage metadata updated for added training source")
            if actual != expected:
                local_differences.append("metadata differs beyond explicit new plan lineage")
        physical.append({"episode_id": ep["episode_id"], "source": src.as_posix(), "source_sha256": sha256(src),
                         "derived": dst.as_posix(), "derived_sha256": sha256(dst), "differences": local_differences})
        differences.extend(f"{ep['episode_id']}: {item}" for item in local_differences)
    supplemental_plan_path = Path(study["supplemental_plan_path"])
    supplemental_plan = read(supplemental_plan_path)
    expected_supplemental = {ep["episode_id"] for ep in supplemental_plan["episodes"] if ep["split"] == "train"}
    records = study["supplemental_train"]
    if len(records) != 32 or len({item["episode_id"] for item in records}) != 32 or {item["episode_id"] for item in records} != expected_supplemental:
        differences.append("supplemental set is not the exact 32 historical TRAIN episodes")
    if sha256(supplemental_plan_path) != before["neural_plan_sha256"] or sha256(supplemental_plan_path) != study["supplemental_plan_sha256"]:
        differences.append("historical source plan changed")
    for record in records:
        expected_path = supplemental_plan_path.parent / "episodes/connected" / (record["episode_id"] + ".npz")
        if (record["split"] != "train" or Path(record["path"]).resolve() != expected_path.resolve()
                or record["sha256"] != sha256(expected_path)):
            differences.append(f"supplemental training path/split/SHA mismatch: {record['episode_id']}")
    existing_test = [str(folder / "episodes/connected" / (ep["episode_id"] + ".npz"))
                     for folder in (source, derived) for ep in parent["episodes"] if ep["split"] == "test"
                     and (folder / "episodes/connected" / (ep["episode_id"] + ".npz")).exists()]
    return {"checked_utc": timestamp(), "pass": not differences, "differences": differences,
        "parent_plan_digest": parent["digest"], "derived_plan_digest": study["digest"],
        "original_sources_unchanged": 56 if not any("original source changed" in item for item in differences) else None,
        "physical_episodes_checked": len(physical), "physical_arrays_checked_bytewise": array_count,
        "supplemental_train_episodes": len(records), "physical_episode_audits": physical,
        "existing_test_paths_at_audit": existing_test, "test_contents_opened": False,
        "original_test_paths_at_pre_derivation_snapshot": before["existing_expected_original_test_files"],
        "method": "Independent pre-derivation SHA snapshot of 56 original NPZ files; compare every physical array by exact dtype, shape and payload bytes; compare full metadata excluding only explicitly reconstructed lineage fields; verify all original protocol/acceptance fields and exact historical train-only set.",
        "audit_source_sha256": sha256(Path(__file__))}


def read_metadata(archive):
    return json.loads(str(archive["metadata"].item()))


if __name__ == "__main__":
    result = audit()
    output = Path("reports/semantic_channel_v2/mixed-derivation-audit.json")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "physical_episode_audits"}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)
