"""Ports must remain tied to real graph identity and nonmotor anatomy."""
import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from fly_semantic.mapping import digest, file_digest, load_mapping, validate_mapping


FILES = ("ids.npy", "ptr.npy", "posts.npy", "counts.npy", "signs.npy",
         "ports.json", "neurons.feather", "behavior_ports.json")


def seal(document):
    document["digest"] = digest({k: v for k, v in document.items() if k != "digest"})
    return document


@pytest.fixture
def graph_mapping(tmp_path):
    """Small synthetic file format fixture; never presented as a real fly."""
    n = 90
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    np.save(tmp_path / "ids.npy", ids)
    np.save(tmp_path / "ptr.npy", np.zeros(n + 1, np.int64))
    np.save(tmp_path / "posts.npy", np.zeros(0, np.int32))
    np.save(tmp_path / "counts.npy", np.zeros(0, np.uint32))
    np.save(tmp_path / "signs.npy", np.ones(n, np.int8))
    ports = {"retina": [0], "lamina": [1], "DNa02": {"L": [2], "R": []},
             "DNp09": {"L": [], "R": []}, "MDN": {"L": [], "R": []}}
    (tmp_path / "ports.json").write_text(json.dumps(ports))
    (tmp_path / "behavior_ports.json").write_text(json.dumps({"groups": {}}))
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset": "test-only", "neurons": n}))
    feather.write_feather(pa.table({"bodyId": ids, "superclass": ["cb_intrinsic"] * n}),
                          tmp_path / "neurons.feather")
    def rows(indices):
        return [{"index": int(i), "body_id": int(ids[i]), "superclass": "cb_intrinsic"} for i in indices]
    doc = {"schema_version": 1, "dataset": "test-only", "neuron_count": n, "cells_per_port": 16,
           "graph_files": {name: file_digest(tmp_path / name) for name in FILES},
           "ports": {key: rows(range(4 + k * 16, 20 + k * 16))
                     for k, key in enumerate(("hunger", "101", "102", "103"))},
           "features": rows(range(68, 76)), "excluded_indices": list(range(68))}
    return tmp_path, seal(doc)


def test_valid_mapping_loads_without_mutation(graph_mapping):
    graph, doc = graph_mapping
    before = copy.deepcopy(doc)
    path = graph / "mapping.json"
    path.write_text(json.dumps(doc))
    assert load_mapping(path, graph) == doc
    assert validate_mapping(doc, graph) is doc
    assert doc == before


def test_sensory_pool_requires_explicit_manifest_and_authoritative_anatomy(graph_mapping):
    graph, doc = graph_mapping
    records = feather.read_table(graph / "neurons.feather").to_pydict()
    for rows in doc["ports"].values():
        for row in rows:
            records["superclass"][row["index"]] = "cb_sensory"
    feather.write_feather(pa.table(records), graph / "neurons.feather")
    doc["graph_files"]["neurons.feather"] = file_digest(graph / "neurons.feather")
    # Absence of the additive field continues to mean the old intrinsic pool.
    with pytest.raises(ValueError, match="central intrinsic"):
        validate_mapping(seal(doc), graph)
    doc["selection_pool"] = "cb_sensory"
    assert validate_mapping(seal(doc), graph) is doc
    row = doc["ports"]["hunger"][0]
    records["superclass"][row["index"]] = "motor"
    feather.write_feather(pa.table(records), graph / "neurons.feather")
    doc["graph_files"]["neurons.feather"] = file_digest(graph / "neurons.feather")
    with pytest.raises(ValueError, match="central sensory"):
        validate_mapping(seal(doc), graph)


def test_unknown_selection_pool_cannot_broaden_allowed_anatomy(graph_mapping):
    graph, doc = graph_mapping
    doc["selection_pool"] = "descending"
    with pytest.raises(ValueError, match="selection pool"):
        validate_mapping(seal(doc), graph)


@pytest.mark.parametrize("files", [{}, {"ids.npy": "0" * 64}, {**dict.fromkeys(FILES, "0" * 64), "extra.npy": "0" * 64}])
def test_cannot_omit_or_extend_graph_fingerprints(graph_mapping, files):
    graph, doc = graph_mapping
    doc["graph_files"] = files
    with pytest.raises(ValueError, match="eight graph fingerprints"):
        validate_mapping(seal(doc), graph)


def test_changed_graph_rejected_even_with_valid_mapping_digest(graph_mapping):
    graph, doc = graph_mapping
    np.save(graph / "counts.npy", np.ones(1, np.uint32))
    with pytest.raises(ValueError, match="graph mismatch: counts.npy"):
        validate_mapping(doc, graph)


@pytest.mark.parametrize("count", [0, 15, 65, True, 16.0])
def test_port_size_budget_is_enforced(graph_mapping, count):
    graph, doc = graph_mapping
    doc["cells_per_port"] = count
    with pytest.raises(ValueError, match="port budget"):
        validate_mapping(seal(doc), graph)


def test_digest_cannot_authorize_injection_into_native_motor(graph_mapping):
    graph, doc = graph_mapping
    doc["ports"]["101"][0].update(index=2, body_id=1002)
    with pytest.raises(ValueError, match="native sensory and motor"):
        validate_mapping(seal(doc), graph)


@pytest.mark.parametrize("target,superclass", [("input", "descending"), ("feature", "motor")])
def test_real_anatomy_overrides_forged_mapping_label(graph_mapping, target, superclass):
    graph, doc = graph_mapping
    records = feather.read_table(graph / "neurons.feather").to_pydict()
    index = doc["ports"]["101"][0]["index"] if target == "input" else doc["features"][0]["index"]
    records["superclass"][index] = superclass
    feather.write_feather(pa.table(records), graph / "neurons.feather")
    doc["graph_files"]["neurons.feather"] = file_digest(graph / "neurons.feather")
    # The mapping still falsely claims cb_intrinsic; authoritative graph wins.
    with pytest.raises(ValueError, match="central intrinsic|internal cells"):
        validate_mapping(seal(doc), graph)


def test_actual_inhibitory_sign_rejected(graph_mapping):
    graph, doc = graph_mapping
    signs = np.load(graph / "signs.npy")
    signs[doc["ports"]["hunger"][0]["index"]] = -1
    np.save(graph / "signs.npy", signs)
    doc["graph_files"]["signs.npy"] = file_digest(graph / "signs.npy")
    with pytest.raises(ValueError, match="positive-sign"):
        validate_mapping(seal(doc), graph)


def test_dataset_must_match_actual_manifest(graph_mapping):
    graph, doc = graph_mapping
    doc["dataset"] = "another-connectome"
    with pytest.raises(ValueError, match="dataset mismatch"):
        validate_mapping(seal(doc), graph)


def test_body_id_must_exactly_match_index(graph_mapping):
    graph, doc = graph_mapping
    doc["features"][0]["body_id"] += 1
    with pytest.raises(ValueError, match="body ID mismatch"):
        validate_mapping(seal(doc), graph)


@pytest.mark.parametrize("native", [False, True])
def test_features_cannot_read_direct_stimulation(graph_mapping, native):
    graph, doc = graph_mapping
    index = 0 if native else doc["ports"]["hunger"][0]["index"]
    doc["features"][0].update(index=index, body_id=1000 + index)
    if native:
        # Removing the exclusion cannot hide a native port either.
        doc["excluded_indices"].remove(index)
    with pytest.raises(ValueError, match="overlapping|must be excluded"):
        validate_mapping(seal(doc), graph)
