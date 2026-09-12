"""A stable graded-release model on every retained MaleCNS cell and edge.

The dynamical family follows continuous voltage / rectified release models,
not a fitted physiological reconstruction. Voltages and release are arbitrary
units, never Hz or measured mV. Incoming normalization, gain and balanced tonic
drive are explicit engineering assumptions; no image-derived feature reaches
any cell except the declared retinal ports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class GradedConfig:
    gain: float = .8
    tau_s: float = .05
    dt_s: float = .01
    input_gain: float = 1.
    resting_voltage: float = .5
    reference_luminance: float = .5

    def validate(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("All model parameters must be finite")
        if not 0 <= self.gain < 1:
            raise ValueError("Normalized recurrent gain must lie in [0,1) for contraction")
        if not 0 < self.dt_s <= self.tau_s or self.input_gain < 0 or self.resting_voltage <= 0:
            raise ValueError("Require 0 < dt <= tau, nonnegative input gain and positive rest")
        if not 0 <= self.reference_luminance <= 1:
            raise ValueError("Reference luminance must be in [0,1]")
        return self


class GradedNetwork:
    """tau*dV/dt = -V + bias + gain*P@relu(V) + retinal_input.

    P's absolute incoming row sums are <=1. Rectification is 1-Lipschitz,
    so gain<1 makes the stationary map a contraction. Leak is integrated
    exponentially with recurrent input held during one step. No reset/spikes,
    artificial LC4/GF injection, behavioural policy or fitted looming formula.
    """
    def __init__(self, graph, config: GradedConfig | None = None):
        self.graph = graph
        self.config = (config or GradedConfig()).validate()
        self.n = graph.incoming.shape[0]
        self.retina = np.asarray(graph.ports["retina"], np.int32)
        if self.retina.size != np.unique(self.retina).size:
            raise ValueError("Retinal ports must be unique cell indices")
        if not hasattr(graph, "_graded_matrix_fingerprint"):
            digest = hashlib.sha256()
            for array in (graph.incoming.data, graph.incoming.indices, graph.incoming.indptr):
                digest.update(str(array.dtype).encode())
                digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
            if hasattr(graph, "ids"):
                digest.update(memoryview(np.ascontiguousarray(graph.ids)).cast("B"))
            graph._graded_matrix_fingerprint = digest.hexdigest()
        self.graph_fingerprint = hashlib.sha256(
            graph._graded_matrix_fingerprint.encode() + self.retina.tobytes()).hexdigest()
        self.reference_release = np.full(self.n, self.config.resting_voltage, np.float32)
        # Written as baseline-relative release below, algebraically equivalent
        # to this explicit balance. It prevents arbitrary static runaway while
        # preserving nonzero tonic release that can be reduced by inhibition.
        self.bias = self.reference_release - self.config.gain * (graph.incoming @ self.reference_release)
        self.voltage = self.reference_release.copy()
        self.elapsed_steps = 0
        self.last_input = np.full(len(self.retina), self.config.reference_luminance, np.float32)
        self.last_external_drive = np.zeros(self.n, np.float32)
        self.last_recurrent_drive = np.zeros(self.n, np.float32)

    def _drive(self, luminance, input_off):
        values = np.asarray(luminance, np.float32)
        if values.shape != self.retina.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Retinal luminance must be a finite [0,1] vector in port order")
        drive = np.zeros(self.n, np.float32)
        if not input_off:
            drive[self.retina] = self.config.input_gain * (values - self.config.reference_luminance)
        return values, drive

    def step(self, luminance, *, input_off=False, freeze_indices=(), freeze_voltage=None):
        values, drive = self._drive(luminance, input_off)
        indices = np.asarray(freeze_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.n):
            raise ValueError("Frozen cells must be valid indices")
        frozen = None
        if indices.size:
            if freeze_voltage is None:
                raise ValueError("Freezing modulation requires explicit pre-stimulus voltage")
            frozen = np.broadcast_to(np.asarray(freeze_voltage, np.float32), (self.n,))
            if not np.isfinite(frozen).all():
                raise ValueError("Frozen voltages must be finite")
            self.voltage[indices] = frozen[indices]
        delta_release = np.maximum(self.voltage, 0.) - self.reference_release
        recurrent = self.config.gain * (self.graph.incoming @ delta_release)
        equilibrium = self.reference_release + recurrent + drive
        alpha = np.float32(-math.expm1(-self.config.dt_s / self.config.tau_s))
        self.voltage += alpha * (equilibrium - self.voltage)
        if indices.size:
            self.voltage[indices] = frozen[indices]
        if not np.isfinite(self.voltage).all():
            raise ArithmeticError("Non-finite graded network state")
        self.elapsed_steps += 1
        self.last_input[:] = values
        self.last_external_drive[:] = drive
        self.last_recurrent_drive[:] = recurrent
        return self.voltage

    def equilibrate(self, luminance, *, tolerance=2e-7, max_iterations=2000, input_off=False):
        """Numerical resting state for the FIRST image only, without lookahead.

        This initializes a stationary stimulus; it does not advance the neural
        clock. Moving stimuli are then compared with a matched static movie
        from the exact same initial voltage, including any residual error.
        """
        if not math.isfinite(tolerance) or tolerance <= 0 or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("Equilibrium tolerance and iteration limit must be positive")
        values, drive = self._drive(luminance, input_off)
        residual = math.inf
        for iteration in range(max_iterations):
            recurrent = self.config.gain * (self.graph.incoming @ (
                np.maximum(self.voltage, 0.) - self.reference_release))
            updated = self.reference_release + recurrent + drive
            residual = float(np.max(np.abs(updated - self.voltage)))
            self.voltage[:] = updated
            if residual <= tolerance:
                self.last_input[:] = values
                self.last_external_drive[:] = drive
                self.last_recurrent_drive[:] = recurrent
                return {"iterations": iteration + 1, "residual_au": residual,
                        "clock_advanced": False}
        raise ArithmeticError(f"Equilibrium not reached; residual={residual:g}")

    def get_state(self):
        return {"format": 1, "config": asdict(self.config), "steps": self.elapsed_steps,
                "graph_fingerprint": self.graph_fingerprint,
                "voltage": self.voltage.copy(), "last_input": self.last_input.copy(),
                "external_drive": self.last_external_drive.copy(),
                "recurrent_drive": self.last_recurrent_drive.copy()}

    def set_state(self, state):
        if (state.get("format") != 1 or state.get("config") != asdict(self.config)
                or state.get("graph_fingerprint") != self.graph_fingerprint):
            raise ValueError("Incompatible graded model state")
        pairs = (("voltage", self.n), ("last_input", len(self.retina)),
                 ("external_drive", self.n), ("recurrent_drive", self.n))
        arrays = {name: np.asarray(state[name], np.float32) for name, _ in pairs}
        if any(arrays[name].shape != (size,) or not np.isfinite(arrays[name]).all() for name, size in pairs):
            raise ValueError("Invalid graded model arrays")
        if not isinstance(state["steps"], int) or state["steps"] < 0:
            raise ValueError("Invalid neural clock")
        self.voltage[:] = arrays["voltage"]
        self.last_input[:] = arrays["last_input"]
        self.last_external_drive[:] = arrays["external_drive"]
        self.last_recurrent_drive[:] = arrays["recurrent_drive"]
        self.elapsed_steps = state["steps"]
