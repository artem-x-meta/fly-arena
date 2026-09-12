import numpy as np
import pytest

from fly_arena.checkpoint import load_checkpoint, save_checkpoint


def test_atomic_checkpoint_roundtrip_and_incompatible_state_rejected(tmp_path):
    path = tmp_path / "path with spaces" / "latest.npz"
    save_checkpoint(path, {"delays": np.arange(18, dtype=np.float32).reshape(3, 6),
                           "nested": [True, {"count": np.int64(9)}]}, {"graph": "abc", "code": "123"})
    state, compatibility = load_checkpoint(path, {"graph": "abc", "code": "123"})
    np.testing.assert_array_equal(state["delays"], np.arange(18).reshape(3, 6))
    assert state["nested"][1]["count"] == 9
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_checkpoint(path, {"bad": float("nan")}, compatibility)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="graph"):
        load_checkpoint(path, {"graph": "different", "code": "123"})
    assert list(path.parent.glob("*.tmp")) == []
