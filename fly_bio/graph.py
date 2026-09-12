"""Read-only, complete MaleCNS graph for experimental continuous dynamics.

The stored anatomical counts are dimensionless.  This loader divides each
signed count by the total number of incoming contacts on its postsynaptic
cell.  That is an engineering normalization, not a fitted synaptic conductance.
No population, weak edge, recurrent edge or repeated edge is removed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy.sparse import csr_matrix


class Connectome:
    """Retain every stored neuron/edge; ``incoming @ x`` maps pre to post.

    ``incoming`` has postsynaptic rows and presynaptic columns.  For every
    nonempty row, the sum of absolute *stored edge* weights is one (up to
    float32 rounding).  Thus multiplying this operator by a gain below one
    gives a contraction for a 1-Lipschitz pointwise release function.  The gain,
    cell time constants, resting state and sensory drive belong to the model.

    The source-only signs come from the legacy graph.  In particular all
    glutamatergic neurons have negative signs there; receptor-specific effects
    and modulatory transmitter mechanisms are not reconstructed by this class.
    """

    def __init__(self, graph_path: str | Path):
        self.path = Path(graph_path)
        manifest_bytes = (self.path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("format_version") != 1:
            raise ValueError("Unsupported connectome format")
        self.ids = np.load(self.path / "ids.npy", mmap_mode="r")
        ptr = np.load(self.path / "ptr.npy", mmap_mode="r")
        posts = np.load(self.path / "posts.npy", mmap_mode="r")
        counts = np.load(self.path / "counts.npy", mmap_mode="r")
        self.signs = np.load(self.path / "signs.npy", mmap_mode="r")
        self.n = len(self.ids)
        self.edge_count = len(posts)
        if (self.ids.ndim != 1 or self.signs.shape != (self.n,)
                or ptr.shape != (self.n + 1,) or counts.shape != posts.shape
                or posts.ndim != 1 or ptr[0] != 0 or ptr[-1] != self.edge_count
                or np.any(np.diff(ptr) < 0) or np.any(np.diff(self.ids) <= 0)
                or not np.isin(self.signs, [-1, 1]).all()):
            raise ValueError("Invalid retained graph arrays")
        if (manifest.get("neurons", self.n) != self.n
                or manifest.get("edges", self.edge_count) != self.edge_count):
            raise ValueError("Retained graph size disagrees with its manifest")

        rows = feather.read_table(self.path / "neurons.feather",
                                  columns=["bodyId", "type"]).to_pydict()
        by_id = dict(zip(rows["bodyId"], rows["type"]))
        if len(by_id) != len(rows["bodyId"]) or any(int(i) not in by_id for i in self.ids):
            raise ValueError("Missing or duplicate retained neuron annotations")
        self.types = np.asarray([by_id[int(i)] or "" for i in self.ids])
        self.ports = json.loads((self.path / "ports.json").read_text(encoding="utf-8"))

        # Chunking bounds temporary arrays instead of allocating a float64
        # contact vector or a repeated source-index vector over all 25M edges.
        chunk = 1_000_000
        self.incoming_totals = np.zeros(self.n, np.float64)
        for begin in range(0, self.edge_count, chunk):
            end = min(begin + chunk, self.edge_count)
            targets, numbers = posts[begin:end], counts[begin:end]
            if (np.any(targets < 0) or np.any(targets >= self.n)
                    or np.any(numbers <= 0) or not np.isfinite(numbers).all()):
                raise ValueError("Invalid edge target or anatomical contact count")
            self.incoming_totals += np.bincount(targets, weights=numbers, minlength=self.n)
        weights = np.empty(self.edge_count, np.float32)
        for begin in range(0, self.edge_count, chunk):
            end = min(begin + chunk, self.edge_count)
            edge_positions = np.arange(begin, end, dtype=ptr.dtype)
            source = np.searchsorted(ptr, edge_positions, side="right") - 1
            weights[begin:end] = (counts[begin:end] / self.incoming_totals[posts[begin:end]]) * self.signs[source]

        # Building from CSR arrays and transposing retains duplicate rows in
        # the anatomical export. A coordinate constructor can coalesce them.
        outgoing = csr_matrix((weights, posts, ptr), shape=(self.n, self.n), copy=False)
        self.incoming = outgoing.transpose().tocsr(copy=True)
        if self.incoming.nnz != self.edge_count:
            raise ValueError("Sparse conversion changed the retained edge count")
        for value in (self.types, self.incoming_totals, self.incoming.data,
                      self.incoming.indices, self.incoming.indptr):
            value.flags.writeable = False
        self.metadata = {
            "dataset": manifest.get("dataset"),
            "neurons": self.n,
            "edges": self.edge_count,
            "synaptic_contacts": int(self.incoming_totals.sum()),
            "graph_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "declared_array_sha256": dict(manifest.get("arrays", {})),
            "declared_sources": dict(manifest.get("sources", {})),
            "normalization": "W[post,pre] = sign[pre] * contacts[pre,post] / total_incoming_contacts[post]; dimensionless engineering choice",
            "signs": "Stored source-only sign proxy: GABA/glutamate/histamine negative; other transmitters positive; not receptor-specific",
            "retention": "Every stored node and edge retained, including weak, recurrent, self and duplicate edges",
            "matrix_dtype": str(self.incoming.dtype),
            "matrix_orientation": "postsynaptic rows, presynaptic columns",
            "array_hashes_live_verified": False,
        }

    def indices(self, type_names: str | list[str] | tuple[str, ...]) -> np.ndarray:
        """CSR neuron indices of exact annotation types, in retained ID order."""
        names = [type_names] if isinstance(type_names, str) else list(type_names)
        return np.flatnonzero(np.isin(self.types, names)).astype(np.int32)
