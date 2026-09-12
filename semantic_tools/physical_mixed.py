"""Explicit derived study: physical episodes plus historical training-only states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fly_semantic.mapping import digest, file_digest
from .physical_collection import load_plan, write_json
from .physical_learning import load_rows, train
from .state_experiment import load_episodes


HISTORICAL = Path("runs/semantic-v1/state-selected")


def prepare(source, output):
    source, output = Path(source), Path(output)
    if (output / "plan.json").exists():
        raise FileExistsError("Derived study already exists")
    parent = load_plan(source)
    if any((source / "episodes" / "connected").glob("test-*.npz")):
        raise ValueError("This derivation requires an unseen physical test")
    training = json.loads((source / "training.json").read_text(encoding="utf-8"))
    records = load_rows(source, parent, "train") + load_rows(source, parent, "validation")
    old_plan, old_rows = load_episodes(HISTORICAL, "train")
    supplemental = []
    for row in old_rows:
        ep = row["metadata"]["episode"]
        path = HISTORICAL / "episodes" / "connected" / (ep["episode_id"] + ".npz")
        if ep["split"] != "train" or row["metadata"]["feature_digest"] != parent["feature_digest"]:
            raise ValueError("Historical data must be training-only with identical features")
        supplemental.append({"path": path.as_posix(), "sha256": file_digest(path),
                             "episode_id": ep["episode_id"], "split": "train"})
    study = {k: v for k, v in parent.items() if k != "digest"}
    study.update(scope="preliminary mixed training: 18 real physical episodes plus32 historical synthetic-background training episodes; all validation/test physical",
        include_synthetic_train=True, supplemental_train=supplemental,
        supplemental_plan_path=(HISTORICAL / "plan.json").as_posix(),
        supplemental_plan_sha256=file_digest(HISTORICAL / "plan.json"),
        derived_from={"plan_path": (source / "plan.json").as_posix(), "plan_sha256": file_digest(source / "plan.json"),
            "plan_digest": parent["digest"], "reason": "Physical-only validation releases after meals but false activations remain; add previously available training states without seeing test",
            "parent_model_sha256": training["model_sha256"], "parent_validation": training["validation"],
            "physical_data": [{"path": r["path"], "sha256": r["sha256"]} for r in records]},
        derivation_source_sha256=file_digest(Path(__file__)))
    study["digest"] = digest(study)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", study)
    for row in records:
        src = Path(row["path"])
        target = output / "episodes" / "connected" / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        with np.load(src, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        meta = json.loads(str(arrays["metadata"]))
        meta.update(plan_digest=study["digest"], original_plan_digest=parent["digest"],
                    derived_from_path=src.as_posix(), derived_from_sha256=row["sha256"],
                    derivation="Existing physical measurements copied unchanged; only plan lineage metadata updated for added training source")
        arrays["metadata"] = np.array(json.dumps(meta, allow_nan=False))
        np.savez_compressed(target, **arrays)
        with np.load(src, allow_pickle=False) as before, np.load(target, allow_pickle=False) as after:
            for name in before.files:
                if name != "metadata":
                    np.testing.assert_array_equal(before[name], after[name])
    write_json(output / "derivation.json", {"status": "PASS", "parent_plan_digest": parent["digest"],
        "new_plan_digest": study["digest"], "physical_episodes_identical_arrays": len(records),
        "supplemental_train_episodes": len(supplemental), "heldout_test_seen": False})
    return study


def train_mixed(output):
    output = Path(output)
    study = load_plan(output)
    if not study.get("include_synthetic_train"):
        raise ValueError("Not a declared mixed study")
    if file_digest(study["supplemental_plan_path"]) != study["supplemental_plan_sha256"]:
        raise ValueError("Historical training plan changed")
    _, rows = load_episodes(Path(study["supplemental_plan_path"]).parent, "train")
    by_id = {r["metadata"]["episode"]["episode_id"]: r for r in rows}
    if set(by_id) != {r["episode_id"] for r in study["supplemental_train"]}:
        raise ValueError("Unexpected supplemental episodes")
    for record in study["supplemental_train"]:
        if file_digest(record["path"]) != record["sha256"] or record["split"] != "train":
            raise ValueError("Supplemental training data changed or leaked")
        by_id[record["episode_id"]].update(path=record["path"], sha256=record["sha256"])
    return train(output, study, list(by_id.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "train"])
    p.add_argument("--source", type=Path, default=Path("runs/semantic-v2/physical-study"))
    p.add_argument("--output", type=Path, default=Path("runs/semantic-v2/mixed-study"))
    a = p.parse_args()
    result = prepare(a.source, a.output) if a.command == "prepare" else train_mixed(a.output)
    print(json.dumps({k: result[k] for k in ("digest", "model_sha256", "selected_l2") if k in result}))
