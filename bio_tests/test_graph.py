"""Independent small-graph checks for signed, incoming-normalized transmission."""
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from fly_bio.graph import Connectome


def make_graph(path):
    path.mkdir()
    arrays = {
        "ids": np.array([100, 200, 300, 400, 500], np.int64),
        "ptr": np.array([0, 3, 4, 6, 7, 7], np.int64),
        # Weak self-edge, duplicate 0->1 edge, recurrence, and an isolated node.
        "posts": np.array([0, 1, 1, 2, 0, 1, 1], np.int32),
        "counts": np.array([1, 2, 3, 4, 5, 6, 7], np.uint32),
        "signs": np.array([-1, 1, -1, 1, 1], np.int8),
    }
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value)
    (path / "manifest.json").write_text(json.dumps({"format_version": 1, "neurons": 5, "edges": 7}))
    (path / "ports.json").write_text(json.dumps({"retina": [0], "uv": [[.5, .5]], "eye": [0]}))
    # Deliberately shuffled: annotations must align by body ID, not file order.
    feather.write_feather(pa.table({"bodyId": [500, 300, 100, 400, 200],
                                   "type": [None, "L1", "R1-R6", "L2", "Mi1"]}), path / "neurons.feather")
    return arrays


def test_signed_incoming_normalization_and_retention(tmp_path):
    path = tmp_path / "graph"
    make_graph(path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    graph = Connectome(path)
    assert graph.n == 5 and graph.edge_count == graph.incoming.nnz == 7
    assert graph.incoming.dtype == np.float32
    np.testing.assert_array_equal(graph.incoming_totals, [6, 18, 4, 0, 0])
    expected = np.zeros((5, 5))
    expected[0, 0], expected[0, 2] = -1 / 6, -5 / 6
    expected[1, 0], expected[1, 2], expected[1, 3] = -5 / 18, -6 / 18, 7 / 18
    expected[2, 1] = 1
    drive = np.array([.3, -.5, .7, .2, 999], np.float32)
    np.testing.assert_allclose(graph.incoming @ drive, expected @ drive, atol=3e-8)
    # SciPy abs() may coalesce duplicate entries in place; audit a writable
    # copy while the actual operator retains its original seven edge entries.
    np.testing.assert_allclose(np.asarray(abs(graph.incoming.copy()).sum(axis=1)).ravel(), [1, 1, 1, 0, 0])
    assert graph.incoming.nnz == 7
    assert graph.metadata["synaptic_contacts"] == 28
    assert graph.metadata["array_hashes_live_verified"] is False
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    assert before == after
    with pytest.raises(ValueError):
        graph.incoming.data[0] = 0


def test_types_match_ids_and_missing_population_is_empty(tmp_path):
    path = tmp_path / "graph"
    make_graph(path)
    graph = Connectome(path)
    np.testing.assert_array_equal(graph.types, ["R1-R6", "Mi1", "L1", "L2", ""])
    np.testing.assert_array_equal(graph.indices("L1"), [2])
    np.testing.assert_array_equal(graph.indices(("R1-R6", "L2")), [0, 3])
    assert graph.indices("DNp01").size == 0
    assert graph.indices([]).dtype == np.int32


def test_malformed_target_is_rejected(tmp_path):
    path = tmp_path / "graph"
    arrays = make_graph(path)
    arrays["posts"][0] = 6
    np.save(path / "posts.npy", arrays["posts"])
    with pytest.raises(ValueError, match="Invalid edge"):
        Connectome(path)
