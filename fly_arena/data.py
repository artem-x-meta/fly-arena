"""Download MaleCNS directly from Janelia and build a sparse outgoing graph.

The release includes many unannotated segmentation fragments. We simulate all
non-glial entries with an assigned superclass. Every released connection between
those entries is retained, including weak connections and self-connections.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from numba import njit

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SOURCES = {
    "annotations": ("body-annotations-male-cns-v1.0-minconf-0.5.feather", 14483314,
                    "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"),
    "transmitters": ("body-neurotransmitters-male-cns-v1.0.feather", 43282834,
                     "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"),
    "connections": ("connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1051241946,
                    "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, size, digest) in SOURCES.items():
        path = raw_dir / name
        paths[key] = path
        if path.exists() and path.stat().st_size == size and sha256(path) == digest:
            print(f"Verified: {name}", flush=True)
            continue
        part = path.with_suffix(".part")
        print(f"Downloading {name} ({size / 1e6:.1f} MB)", flush=True)
        for attempt in range(4):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset >= size:
                    part.unlink()
                    offset = 0
                request = Request(BASE_URL + name, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urlopen(request, timeout=90) as response:
                    if offset and response.status != 206:
                        offset = 0
                    if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Invalid Content-Range while resuming download")
                    with part.open("ab" if offset else "wb") as target:
                        last = time.monotonic()
                        while chunk := response.read(4 << 20):
                            target.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last > 5:
                                print(f"  {100 * offset / size:.0f}%", flush=True)
                                last = time.monotonic()
                if part.stat().st_size != size or sha256(part) != digest:
                    part.unlink()
                    raise RuntimeError(f"Size or SHA-256 mismatch: {name}")
                os.replace(part, path)
                break
            except (OSError, RuntimeError) as error:
                if attempt == 3:
                    raise RuntimeError(f"Download failed; rerun prepare to resume: {name}") from error
                print(f"Retry {attempt + 1}: {error}", flush=True)
    return paths


def map_ids(ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(ids, values)
    safe = np.minimum(indices, len(ids) - 1)
    return safe, (indices < len(ids)) & (ids[safe] == values)


@njit(cache=True)
def fill_csr(rows, cursor, posts, counts):
    for row in rows:
        source, target, count = row
        i = cursor[source]
        posts[i] = target
        counts[i] = count
        cursor[source] += 1


def make_ports(nodes, ptr, posts, counts) -> dict:
    """Infer a retinal sampling grid from annotated optic-lobe hex coordinates.

    This is a geometrical proxy, not a calibrated registration of this specimen's
    eyes to NeuroMechFly. R1-R6 lack hex annotations, so the strongest annotated
    L1/L2/L3 target on the same side supplies their column.
    """
    types = np.asarray(nodes["type"])
    sides = np.asarray(nodes["somaSide"])
    roots = np.asarray(nodes["rootSide"])
    hexes = np.c_[nodes["assignedOlHex1"], nodes["assignedOlHex2"]].astype(float)
    lamina = np.isin(types, ["L1", "L2", "L3"])
    valid_hex = np.isfinite(hexes).all(axis=1)
    retina, coords, eyes = [], [], []
    for i in np.flatnonzero(types == "R1-R6"):
        side = roots[i] or sides[i]
        outgoing = posts[ptr[i]:ptr[i + 1]]
        values = counts[ptr[i]:ptr[i + 1]]
        candidates = np.flatnonzero(lamina[outgoing] & valid_hex[outgoing] & (sides[outgoing] == side))
        if side not in ("L", "R") or not len(candidates):
            continue
        target = outgoing[candidates[np.argmax(values[candidates])]]
        retina.append(int(i))
        coords.append(hexes[target])
        eyes.append(0 if side == "L" else 1)
    if not retina:
        raise ValueError("No retinal ports found in this dataset")
    coords = np.asarray(coords)
    eyes = np.asarray(eyes)
    # Axial hex coordinates -> cartesian, then normalized per eye.
    xy = np.c_[coords[:, 0] - .5 * coords[:, 1], np.sqrt(3) * .5 * coords[:, 1]]
    uv = np.empty_like(xy)
    for eye in (0, 1):
        mask = eyes == eye
        if not mask.any():
            raise ValueError("Missing an eye's retinal mapping")
        low, high = xy[mask].min(axis=0), xy[mask].max(axis=0)
        uv[mask] = .08 + .84 * (xy[mask] - low) / np.maximum(high - low, 1)
    uv[:, 1] = 1 - uv[:, 1]
    uv[eyes == 1, 0] = 1 - uv[eyes == 1, 0]
    ports = {"retina": retina, "eye": eyes.tolist(), "uv": uv.tolist(),
             "lamina": np.flatnonzero(np.isin(types, ["L1", "L2", "L3", "L5"])).tolist()}
    for kind in ("DNa02", "DNp09", "MDN"):
        ports[kind] = {side: np.flatnonzero((types == kind) & (sides == side)).tolist()
                       for side in ("L", "R")}
    for kind in ("DNa02", "DNp09"):
        if not all(ports[kind].values()):
            raise ValueError(f"Missing required {kind} readout neurons")
    ports["notes"] = {
        "retina": "R1-R6 strongest same-side annotated L1/L2/L3 partner; approximate image registration",
        "motor": "DNa02 L/R rate difference -> gait asymmetry; DNp09 mean rate -> gait magnitude",
        "not_connected": "No direct neuron-to-muscle registration; no olfaction, taste or internal metabolism",
    }
    return ports


def prepare(data_dir: Path, raw_dir: Path | None = None) -> dict:
    raw_dir = raw_dir or data_dir / "raw"
    paths = download(raw_dir)
    graph = data_dir / "graph"
    graph.mkdir(parents=True, exist_ok=True)
    # Commit marker removed first, so an interrupted build cannot look complete.
    (graph / "manifest.json").unlink(missing_ok=True)
    table = feather.read_table(paths["annotations"])
    keep = np.array([bool(x) for x in table["superclass"].to_pylist()])
    keep &= np.array([x != "Glia" for x in table["status"].to_pylist()])
    table = table.filter(pa.array(keep)).sort_by("bodyId")
    columns = ["bodyId", "type", "superclass", "somaSide", "rootSide", "assignedOlHex1", "assignedOlHex2"]
    nodes = table.select(columns).to_pydict()
    ids = np.asarray(nodes["bodyId"], dtype=np.int64)
    n = len(ids)
    if len(np.unique(ids)) != n:
        raise ValueError("Duplicate annotation IDs")
    np.save(graph / "ids.npy", ids)
    feather.write_feather(table, graph / "neurons.feather")
    nt_rows = feather.read_table(paths["transmitters"], columns=["body", "consensus_nt"]).to_pydict()
    nt_map = dict(zip(nt_rows["body"], nt_rows["consensus_nt"]))
    neurotransmitters = [nt_map.get(int(i)) or "unknown" for i in ids]
    # Receptor-specific signs and neuromodulation are unavailable in this model.
    # Treat GABA/Glu/histamine as inhibitory, all others as excitatory proxies.
    signs = np.array([-1 if nt in ("gaba", "glutamate", "histamine") else 1
                      for nt in neurotransmitters], dtype=np.int8)
    np.save(graph / "signs.npy", signs)
    outdegree = np.zeros(n, dtype=np.int64)
    edge_count = raw_rows = contacts = 0
    temp = graph / "edges.tmp"
    print(f"Building sparse graph for {n:,} annotated non-glial entries...", flush=True)
    with pa.memory_map(str(paths["connections"]), "r") as source, temp.open("wb") as stream:
        reader = ipc.open_file(source)
        for batch_i in range(reader.num_record_batches):
            batch = reader.get_batch(batch_i)
            pre, pre_ok = map_ids(ids, batch.column("body_pre").to_numpy())
            post, post_ok = map_ids(ids, batch.column("body_post").to_numpy())
            count = batch.column("weight").to_numpy()
            mask = pre_ok & post_ok
            selected = np.c_[pre[mask], post[mask], count[mask]]
            if len(selected) and (selected[:, 2].min() <= 0 or selected[:, 2].max() > np.iinfo(np.int32).max):
                raise ValueError("Unexpected connection strength")
            selected.astype(np.int32).tofile(stream)
            outdegree += np.bincount(pre[mask], minlength=n)
            edge_count += int(mask.sum())
            raw_rows += len(batch)
            contacts += int(count[mask].sum())
            if batch_i % 300 == 0:
                print(f"  {100 * batch_i / reader.num_record_batches:.0f}%: {edge_count:,} retained edges", flush=True)
    ptr = np.r_[0, np.cumsum(outdegree)]
    np.save(graph / "ptr.npy", ptr)
    posts = np.lib.format.open_memmap(graph / "posts.npy", "w+", dtype=np.int32, shape=(edge_count,))
    counts = np.lib.format.open_memmap(graph / "counts.npy", "w+", dtype=np.uint32, shape=(edge_count,))
    compact = np.memmap(temp, "int32", "r", shape=(edge_count, 3))
    cursor = ptr[:-1].copy()
    for start in range(0, edge_count, 1_000_000):
        fill_csr(compact[start:start + 1_000_000], cursor, posts, counts)
    if not np.array_equal(cursor, ptr[1:]):
        raise ValueError("CSR edge preservation check failed")
    posts.flush()
    counts.flush()
    ports = make_ports(nodes, ptr, posts, counts)
    (graph / "ports.json").write_text(json.dumps(ports), encoding="utf-8")
    del compact
    temp.unlink()
    manifest = {
        "format_version": 1, "dataset": "male-cns:v1.0", "source": "https://male-cns.janelia.org/download/",
        "license": "CC BY 4.0", "neurons": n, "edges": edge_count, "synaptic_contacts": contacts,
        "raw_connection_rows": raw_rows, "retinal_inputs": len(ports["retina"]),
        "selection": "assigned superclass, status != Glia; every edge between retained entries",
        "transmitter_counts": dict(Counter(neurotransmitters)),
        "sources": {k: {"url": BASE_URL + v[0], "bytes": v[1], "sha256": v[2]} for k, v in SOURCES.items()},
        "arrays": {name: sha256(graph / name) for name in ("ids.npy", "ptr.npy", "posts.npy", "counts.npy", "signs.npy", "ports.json")},
    }
    (graph / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Ready: {n:,} neurons, {edge_count:,} edges, {len(ports['retina']):,} visual inputs.", flush=True)
    return manifest

