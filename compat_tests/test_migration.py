import copy
from pathlib import Path
import shutil

import numpy as np
import pytest

from fly_arena.checkpoint import load_checkpoint, save_checkpoint
from fly_compat.migration import (ARCHIVED_CODE, ROOT, assert_same_state, check_fingerprints,
                                  copy_checkpoint_metadata, migrate, prove_factory_delta)


def test_known_source_delta_proves_archived_digest_and_default_call():
    proof = prove_factory_delta()
    assert proof["canonicalized_code"] == ARCHIVED_CODE
    assert proof["current_code"] != ARCHIVED_CODE
    assert len(proof["in_memory_replacements"]) == 2
    assert proof["default_single_fly_calls"]
    assert all(call["arguments"] == [] for call in proof["default_single_fly_calls"])


def test_unrelated_source_edit_is_rejected_even_when_factory_delta_is_present(tmp_path):
    for path in (ROOT / "fly_arena").glob("*.py"):
        shutil.copyfile(path, tmp_path / path.name)
    path = tmp_path / "organism.py"
    path.write_bytes(path.read_bytes() + b"\n# An unrelated source change must not receive approval.\n")
    with pytest.raises(ValueError, match="beyond"):
        prove_factory_delta(tmp_path)


def test_every_noncode_fingerprint_remains_strict():
    expected = {"code": ARCHIVED_CODE, "config": "cfg", "model": "body", "graph": {"id": 1},
                "ports": "portset", "versions": {"mujoco": "pinned"}, "mode": "ethology-hybrid"}
    actual = {**expected, "code": "current"}
    assert len(check_fingerprints(expected, actual, {"current_code": "current"})) == 6
    for key in set(expected) - {"code"}:
        changed = {**actual, key: "unexpected"}
        with pytest.raises(ValueError, match=key):
            check_fingerprints(expected, changed, {"current_code": "current"})
    with pytest.raises(ValueError, match="unknown"):
        check_fingerprints(expected, {**actual, "unknown": 1}, {"current_code": "current"})


def test_migration_refuses_input_aliases_and_existing_destination_before_runtime(tmp_path):
    source = tmp_path / "old.npz"
    source.write_bytes(b"original checkpoint must not be modified")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="separate copy"):
        migrate(source, source)
    alias = tmp_path / "hardlink.npz"
    alias.hardlink_to(source)
    with pytest.raises(ValueError, match="separate copy"):
        migrate(source, alias)
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"existing progress")
    with pytest.raises(FileExistsError):
        migrate(source, destination)
    assert source.read_bytes() == original
    assert destination.read_bytes() == b"existing progress"


def test_metadata_copy_preserves_all_state_arrays_exactly(tmp_path):
    source, destination = tmp_path / "old.npz", tmp_path / "new.npz"
    expected = {"code": ARCHIVED_CODE, "model": "unchanged"}
    actual = {**expected, "code": "verified-current"}
    state = {"native": np.arange(257, dtype=np.uint8), "brain": np.arange(20, dtype=np.float32).reshape(4, 5),
             "nested": [{"clock": 60.0, "signed_zero": np.array([-0.0, 0.0])}], "none": None}
    save_checkpoint(source, state, expected)
    original = source.read_bytes()
    members = copy_checkpoint_metadata(source, destination, expected, actual)
    restored, compatible = load_checkpoint(destination, expected=actual)
    assert_same_state(state, restored)
    assert compatible == actual
    assert len(members) == 3
    assert source.read_bytes() == original
    changed = copy.deepcopy(restored)
    changed["nested"][0]["signed_zero"][0] = 0.0
    with pytest.raises(ValueError, match="signed_zero"):
        assert_same_state(state, changed)
