"""Frozen, anatomy-selected artificial ports. No semantic biology is implied."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def native_exclusions(graph):
    graph = Path(graph)
    ids = np.load(graph / "ids.npy", mmap_mode="r")
    ports = json.loads((graph / "ports.json").read_text())
    registry = json.loads((graph / "behavior_ports.json").read_text())
    excluded = list(ports["retina"]) + list(ports["lamina"])
    for name in ("DNp09", "DNa02", "MDN"):
        for side in ("L", "R"):
            excluded.extend(ports[name][side])
    for group in registry["groups"].values():
        # Exclude known sensory AND behavioral output ports conservatively.
        selected = np.searchsorted(ids, group["body_ids"])
        if np.any(selected >= len(ids)) or not np.array_equal(ids[selected], group["body_ids"]):
            raise ValueError("Native port body IDs do not match graph")
        excluded.extend(selected.tolist())
    return np.unique(excluded).astype(np.int32)


def build_mapping(graph, seed=42, cells_per_port=32, feature_count=512, selection_pool="cb_intrinsic"):
    graph = Path(graph)
    if not 16 <= cells_per_port <= 64 or not 1 <= feature_count <= 512:
        raise ValueError("Port/feature budget outside v0.1 bounds")
    if selection_pool not in ("cb_intrinsic", "cb_sensory"):
        raise ValueError("Artificial input pool must be cb_intrinsic or cb_sensory")
    ids = np.load(graph / "ids.npy", mmap_mode="r")
    ptr = np.load(graph / "ptr.npy", mmap_mode="r")
    posts = np.load(graph / "posts.npy", mmap_mode="r")
    counts = np.load(graph / "counts.npy", mmap_mode="r")
    signs = np.load(graph / "signs.npy", mmap_mode="r")
    records = feather.read_table(graph / "neurons.feather", columns=["bodyId", "superclass", "type"]).to_pylist()
    by_id = {r["bodyId"]: r for r in records}
    superclasses = np.array([by_id[int(i)]["superclass"] for i in ids])
    excluded = native_exclusions(graph)
    allowed = (superclasses == selection_pool) & (signs > 0)
    allowed[excluded] = False
    degree = np.diff(ptr)
    # Structural selection only, before any activity, labels or learned scores.
    if not np.any(allowed):
        raise ValueError("No positive non-native cells in the requested input pool")
    cutoff = float(np.quantile(degree[allowed], .75))
    candidates = np.flatnonzero(allowed & (degree >= cutoff))
    if len(candidates) < 4 * cells_per_port:
        raise ValueError("Insufficient structural input candidates for disjoint ports")
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=4 * cells_per_port, replace=False)
    groups = {name: np.sort(part) for name, part in zip(
        ("hunger", "101", "102", "103"), np.split(selected, 4))}
    excluded = np.union1d(excluded, selected).astype(np.int32)
    internal = np.isin(superclasses, ["cb_intrinsic", "vnc_intrinsic", "ol_intrinsic"])
    internal[excluded] = False
    # Balanced structural sampling of downstream cells, never response/label selection.
    rankings = []
    for sources in groups.values():
        strength = np.zeros(len(ids), np.float64)
        for pre in sources:
            start, stop = ptr[pre:pre + 2]
            np.add.at(strength, posts[start:stop], counts[start:stop])
        eligible = np.flatnonzero(internal & (strength > 0))
        rankings.append(eligible[np.lexsort((ids[eligible], -strength[eligible]))].tolist())
    features = []
    seen = set()
    for rank in range(max(map(len, rankings))):
        for ranking in rankings:
            if rank < len(ranking) and ranking[rank] not in seen:
                features.append(ranking[rank])
                seen.add(ranking[rank])
                if len(features) == feature_count:
                    break
        if len(features) == feature_count:
            break
    if len(features) != feature_count:
        raise ValueError("Insufficient non-input downstream cells")
    def rows(indices):
        return [{"index": int(i), "body_id": int(ids[i]), "type": by_id[int(ids[i])]["type"],
                 "superclass": str(superclasses[i])} for i in indices]
    document = {"schema_version": 1, "seed": seed, "dataset": "male-cns:v1.0",
        "graph_files": {name: file_digest(graph / name) for name in
            ("ids.npy", "ptr.npy", "posts.npy", "counts.npy", "signs.npy", "ports.json", "neurons.feather", "behavior_ports.json")},
        "neuron_count": len(ids), "cells_per_port": cells_per_port,
        "selection": f"seeded without replacement, positive model sign, {selection_pool}, top outgoing-degree quartile; not a biological homeostasis/word port",
        "degree_cutoff": cutoff, "feature_selection": "round-robin strongest summed existing contacts from each port, internal cells only; no activity/labels",
        "ports": {name: rows(part) for name, part in groups.items()},
        "features": rows(features), "excluded_indices": excluded.tolist()}
    # Keep the original default construction byte-for-byte reproducible. Old
    # manifests without this additive field always mean cb_intrinsic.
    if selection_pool != "cb_intrinsic":
        document["selection_pool"] = selection_pool
    document["digest"] = digest(document)
    return document


def validate_mapping(document, graph=None):
    if not isinstance(document, dict):
        raise ValueError("Semantic mapping must be an object")
    value = dict(document)
    expected = value.pop("digest", None)
    if (expected != digest(value) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1):
        raise ValueError("Invalid semantic mapping digest/schema")
    required_graph_files = {"ids.npy", "ptr.npy", "posts.npy", "counts.npy", "signs.npy",
                            "ports.json", "neurons.feather", "behavior_ports.json"}
    graph_files = value.get("graph_files")
    if not isinstance(graph_files, dict) or set(graph_files) != required_graph_files:
        raise ValueError("Semantic mapping requires all eight graph fingerprints")
    if any(not isinstance(sha, str) or len(sha) != 64
           or any(char not in "0123456789abcdef" for char in sha) for sha in graph_files.values()):
        raise ValueError("Invalid semantic mapping graph fingerprint")
    if not isinstance(value.get("dataset"), str) or not value["dataset"]:
        raise ValueError("Semantic mapping requires a dataset")
    n, count = value.get("neuron_count"), value.get("cells_per_port")
    if type(n) is not int or n < 1 or type(count) is not int or not 16 <= count <= 64:
        raise ValueError("Invalid semantic neuron count or port budget")
    selection_pool = value.get("selection_pool", "cb_intrinsic")
    if selection_pool not in ("cb_intrinsic", "cb_sensory"):
        raise ValueError("Invalid artificial input selection pool")
    all_inputs = []
    if not isinstance(value.get("ports"), dict) or set(value["ports"]) != {"hunger", "101", "102", "103"}:
        raise ValueError("Semantic mapping requires all four artificial ports")
    groups = list(value["ports"].values()) + [value.get("features")]
    if any(not isinstance(entries, list) or any(not isinstance(row, dict)
               or type(row.get("index")) is not int or type(row.get("body_id")) is not int
               for row in entries) for entries in groups):
        raise ValueError("Semantic mapping rows require integer indices and body IDs")
    for entries in value["ports"].values():
        if len(entries) != count:
            raise ValueError("Unequal input port sizes")
        all_inputs.extend(row["index"] for row in entries)
    features = [row["index"] for row in value["features"]]
    if (not isinstance(value.get("excluded_indices"), list)
            or any(type(i) is not int for i in value["excluded_indices"])):
        raise ValueError("Semantic exclusions require integer indices")
    excluded = set(value["excluded_indices"])
    if (len(set(all_inputs)) != len(all_inputs) or len(set(features)) != len(features)
            or not 1 <= len(features) <= 512 or not set(all_inputs) <= excluded
            or set(features) & excluded or any(type(i) is not int or not 0 <= i < n for i in all_inputs + features + list(excluded))):
        raise ValueError("Invalid/overlapping semantic mapping indices")
    if graph is not None:
        graph = Path(graph)
        manifest = json.loads((graph / "manifest.json").read_text())
        if value["dataset"] != manifest.get("dataset"):
            raise ValueError("Semantic mapping dataset mismatch")
        for name, sha in graph_files.items():
            if file_digest(graph / name) != sha:
                raise ValueError(f"Semantic mapping graph mismatch: {name}")
        ids = np.load(graph / "ids.npy", mmap_mode="r")
        if (ids.ndim != 1 or len(ids) != n or manifest.get("neurons") != n
                or np.any(ids[1:] <= ids[:-1])):
            raise ValueError("Semantic mapping neuron IDs/count do not match graph")
        native = set(native_exclusions(graph))
        if not native <= excluded:
            raise ValueError("Native directly driven ports must be excluded")
        if native & set(all_inputs):
            raise ValueError("Artificial inputs must exclude native sensory and motor ports")
        for entries in groups:
            if any(int(ids[row["index"]]) != row["body_id"] for row in entries):
                raise ValueError("Semantic mapping body ID mismatch")
        annotations = feather.read_table(graph / "neurons.feather", columns=["bodyId", "superclass"]).to_pylist()
        by_id = {row["bodyId"]: row["superclass"] for row in annotations}
        if len(by_id) != len(annotations) or any(int(ids[i]) not in by_id for i in all_inputs + features):
            raise ValueError("Semantic mapping annotations have missing/duplicate body IDs")
        signs = np.load(graph / "signs.npy", mmap_mode="r")
        if signs.shape != (n,):
            raise ValueError("Semantic mapping transmitter signs do not match graph")
        if any(by_id[int(ids[i])] != selection_pool or signs[i] <= 0 for i in all_inputs):
            pool_label = "central intrinsic" if selection_pool == "cb_intrinsic" else "central sensory"
            raise ValueError(f"Artificial inputs must be positive-sign {pool_label} ({selection_pool}) cells")
        if any(by_id[int(ids[i])] not in {"cb_intrinsic", "vnc_intrinsic", "ol_intrinsic"} for i in features):
            raise ValueError("Semantic features must be internal cells")
    return document


def load_mapping(path, graph=None):
    return validate_mapping(json.loads(Path(path).read_text(encoding="utf-8")), graph)
