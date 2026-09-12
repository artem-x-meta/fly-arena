"""Read-only MaleCNS connectivity audit around the published circuit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


def audit(graph: Path, output: Path):
    graph, output = Path(graph), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((graph / "manifest.json").read_text())
    ids = np.load(graph / "ids.npy", mmap_mode="r")
    ptr = np.load(graph / "ptr.npy", mmap_mode="r")
    posts = np.load(graph / "posts.npy", mmap_mode="r")
    counts = np.load(graph / "counts.npy", mmap_mode="r")
    signs = np.load(graph / "signs.npy", mmap_mode="r")
    rows = feather.read_table(graph / "neurons.feather", columns=["bodyId", "type", "somaSide", "rootSide", "hemibrainType", "synonyms"]).to_pylist()
    by_id = {row["bodyId"]: row for row in rows}
    ordered = [by_id[int(body)] for body in ids]
    types = np.array([row["type"] for row in ordered])
    selected = {kind: np.flatnonzero(types == kind) for kind in ("LC4", "LPLC2", "DNp01")}
    edges = []
    for kind in ("LC4", "LPLC2"):
        for pre in selected[kind]:
            for post in selected["DNp01"]:
                for offset in np.flatnonzero(posts[ptr[pre]:ptr[pre + 1]] == post):
                    edge = int(ptr[pre] + offset)
                    edges.append({"pre_body_id": int(ids[pre]), "pre_type": kind,
                                  "pre_side": ordered[pre]["somaSide"] or ordered[pre]["rootSide"],
                                  "post_body_id": int(ids[post]), "post_side": ordered[post]["somaSide"],
                                  "synapses": int(counts[edge]), "legacy_sign_proxy": int(signs[pre])})
    with (output / "direct_edges.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(edges[0]) if edges else ["pre_body_id", "post_body_id", "synapses"])
        writer.writeheader()
        writer.writerows(edges)
    summary = []
    for kind in ("LC4", "LPLC2"):
        for post in selected["DNp01"]:
            group = [edge for edge in edges if edge["pre_type"] == kind and edge["post_body_id"] == int(ids[post])]
            summary.append({"pre_type": kind, "post_body_id": int(ids[post]), "post_side": ordered[post]["somaSide"],
                            "edge_rows": len(group), "presynaptic_cells": len({e['pre_body_id'] for e in group}),
                            "synapses": sum(e["synapses"] for e in group)})
    report = {"dataset": manifest["dataset"], "graph_arrays": manifest["arrays"],
              "query": "type IN ('LC4','LPLC2','DNp01'); direct outgoing CSR edges to DNp01; no weight threshold",
              "groups": {kind: [ordered[i] for i in indices] for kind, indices in selected.items()},
              "direct_connections": summary,
              "interpretation": "Direct anatomical chemical connections are present. Synapse counts are not fitted conductances or proof of dynamic signal propagation. Paper recordings used other specimens; no FlyWire body ID is transplanted."}
    (output / "anatomy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return report
