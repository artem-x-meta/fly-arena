"""Small causal neural observations; no organism or event metadata is accepted.

These exponential traces are external decoder memory, not connectome memory.
The caller must supply the union of *all* directly stimulated neuron indices,
including native vision, sensory and modulation inputs, as exclusions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _indices(values: Any, name: str, neuron_count: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.size and (np.any(raw < 0) or np.any(raw >= neuron_count)):
        raise ValueError(f"{name} contains an index outside the graph")
    return raw.astype(np.int64, copy=True)


class CausalFeatures:
    """Two rate traces per selected cell, flattened in tau-major order.

    Each update regards counts/dt as the mean firing rate of the completed
    interval, then applies the exact exponential EMA for that constant rate.
    This uses past and current counts only and does not claim sub-bin timing.
    Rates and returned features have units Hz; clocks and taus use milliseconds.
    """

    STATE_FORMAT = "fly_semantic.causal_features.v1"

    def __init__(
        self,
        indices: Any,
        *,
        neuron_count: int,
        excluded_indices: Any,
        tau_ms: Any = (50.0, 200.0),
        manifest_digest: str = "",
    ):
        if isinstance(neuron_count, (bool, np.bool_)) or not isinstance(neuron_count, (int, np.integer)) or neuron_count < 1:
            raise ValueError("neuron_count must be a positive integer")
        self.neuron_count = int(neuron_count)
        self.indices = _indices(indices, "indices", self.neuron_count)
        self.excluded_indices = np.unique(_indices(excluded_indices, "excluded_indices", self.neuron_count))
        if not 1 <= self.indices.size <= 512 or np.unique(self.indices).size != self.indices.size:
            raise ValueError("features require 1..512 unique neuron indices")
        if np.intersect1d(self.indices, self.excluded_indices).size:
            raise ValueError("primary features must exclude all directly stimulated cells")
        self.tau_ms = np.asarray(tau_ms, dtype=np.float64).copy()
        if self.tau_ms.shape != (2,) or not np.all(np.isfinite(self.tau_ms)) or np.any(self.tau_ms <= 0) or self.tau_ms[0] >= self.tau_ms[1]:
            raise ValueError("tau_ms must contain two increasing positive finite time constants")
        if not isinstance(manifest_digest, str):
            raise ValueError("manifest_digest must be a string")
        self.manifest_digest = manifest_digest
        self.n_features = int(self.indices.size * self.tau_ms.size)
        self.feature_dim = self.n_features
        identity = {
            "format": self.STATE_FORMAT,
            "neuron_count": self.neuron_count,
            "indices": self.indices.tolist(),
            "excluded_indices": self.excluded_indices.tolist(),
            "tau_ms": self.tau_ms.tolist(),
            "manifest_digest": self.manifest_digest,
            "units": "Hz",
            "order": "tau_major",
            "update": "constant_interval_rate_exact_ema",
        }
        self.identity_digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.feature_digest = self.identity_digest
        for value in (self.indices, self.excluded_indices, self.tau_ms):
            value.flags.writeable = False
        self.reset()

    def reset(self) -> None:
        self._traces = np.zeros((self.tau_ms.size, self.indices.size), dtype=np.float64)
        self.elapsed_ms = 0.0
        self.update_count = 0

    def update(self, spike_counts: Any, dt_ms: float) -> np.ndarray:
        """Observe one completed neural interval and return a detached vector."""
        counts = np.asarray(spike_counts)
        if counts.shape != (self.neuron_count,) or counts.dtype.kind not in "iuf" or not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("spike_counts must be a finite nonnegative full-graph vector")
        if isinstance(dt_ms, (bool, np.bool_)) or not np.isscalar(dt_ms) or not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be positive and finite")
        dt = float(dt_ms)
        next_time = self.elapsed_ms + dt
        if not np.isfinite(next_time) or next_time <= self.elapsed_ms:
            raise ValueError("causal clock must advance finitely")
        rates = counts[self.indices].astype(np.float64) * (1000.0 / dt)
        alpha = -np.expm1(-dt / self.tau_ms)[:, None]
        next_traces = (1.0 - alpha) * self._traces + alpha * rates[None, :]
        if not np.all(np.isfinite(next_traces)):
            raise ValueError("feature update overflowed")
        self._traces = next_traces
        self.elapsed_ms = next_time
        self.update_count += 1
        return self.values()

    def values(self) -> np.ndarray:
        return self._traces.reshape(-1).copy()

    def get_state(self) -> dict:
        return {
            "format": self.STATE_FORMAT,
            "identity_digest": self.identity_digest,
            "elapsed_ms": self.elapsed_ms,
            "update_count": self.update_count,
            "traces_hz": self._traces.copy(),
        }

    def set_state(self, state: dict) -> None:
        """Restore atomically, rejecting a different port/feature/clock layout."""
        expected = {"format", "identity_digest", "elapsed_ms", "update_count", "traces_hz"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("malformed feature checkpoint")
        if state["format"] != self.STATE_FORMAT or state["identity_digest"] != self.identity_digest:
            raise ValueError("feature checkpoint identity mismatch")
        elapsed = state["elapsed_ms"]
        updates = state["update_count"]
        if isinstance(elapsed, (bool, np.bool_)) or not np.isscalar(elapsed) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid feature clock")
        if isinstance(updates, (bool, np.bool_)) or not isinstance(updates, (int, np.integer)) or updates < 0 or ((updates == 0) != (elapsed == 0)):
            raise ValueError("feature clock and update count disagree")
        traces = np.asarray(state["traces_hz"], dtype=np.float64)
        if traces.shape != self._traces.shape or not np.all(np.isfinite(traces)) or np.any(traces < 0):
            raise ValueError("invalid feature traces")
        if updates == 0 and np.any(traces != 0):
            raise ValueError("unobserved feature state must be empty")
        self._traces = traces.copy()
        self.elapsed_ms = float(elapsed)
        self.update_count = int(updates)
