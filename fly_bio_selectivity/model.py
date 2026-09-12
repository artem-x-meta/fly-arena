"""An explicit delay sensitivity experiment on the unchanged graded graph.

The 18/13 ms defaults proxy differences between temporal-filter peak times in
Behnia et al. (2014), Fig. 3. They are NOT measured axonal/synaptic delays or
membrane time constants, and do not reproduce the paper's complete filters.
Every original node, edge, sign, normalization and membrane equation is kept.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from fly_bio.model import GradedConfig, GradedNetwork


BEHNIA_DELAY_BY_TYPE_S = MappingProxyType({"Mi1": .018, "Tm1": .013})
DELAYED_STATE_FORMAT = "fly_bio_selectivity.delayed.v1"


class DelayedGradedNetwork(GradedNetwork):
    """Delay selected presynaptic RELEASE, with causal fractional interpolation.

    At step start, voltage is V(t). Its rectified release is stored at the
    current history slot. A delay d=(k+f)*dt reads (1-f)*r(t-k*dt) plus
    f*r(t-(k+1)*dt). This delayed release replaces only selected entries in
    the full release vector before the unchanged sparse matrix product.

    ``equilibrate`` fills the entire prehistory with release under the first
    static image. Its fixed point is unchanged by a constant delay. No trial
    stimulus name, angle, motion direction or future image enters this model.
    """

    def __init__(self, graph, config: GradedConfig | None = None, *,
                 delay_by_type_s: Mapping[str, float] | None = None):
        selected = BEHNIA_DELAY_BY_TYPE_S if delay_by_type_s is None else delay_by_type_s
        if not isinstance(selected, Mapping):
            raise ValueError("Delay configuration must map cell type names to seconds")
        delays = {}
        for name, value in selected.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Delayed cell types must be nonempty strings")
            try:
                seconds = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Delays must be finite nonnegative seconds") from exc
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Delays must be finite nonnegative seconds")
            delays[name] = seconds
        self.delay_by_type_s = MappingProxyType(dict(sorted(delays.items())))
        super().__init__(graph, config)

        positive = {name: seconds for name, seconds in delays.items() if seconds > 0}
        types = np.asarray(getattr(graph, "types", []))
        if positive and types.shape != (self.n,):
            raise ValueError("Positive delays require one cell type annotation per node")
        if any(not np.any(types == name) for name in positive):
            raise ValueError("Delayed type is absent from graph annotations")
        self.delayed_indices = np.flatnonzero(np.isin(types, list(positive))).astype(np.int32)
        self.delayed_indices.flags.writeable = False
        self.delay_steps = np.asarray(
            [positive[str(types[i])] / self.config.dt_s for i in self.delayed_indices],
            dtype=np.float64)
        self.delay_steps.flags.writeable = False
        self._whole_steps = np.floor(self.delay_steps).astype(np.int64)
        self._fraction = (self.delay_steps - self._whole_steps).astype(np.float32)
        self._columns = np.arange(len(self.delayed_indices))
        slots = int(math.ceil(float(self.delay_steps.max()))) + 1 if self.delay_steps.size else 1
        self.history = np.empty((slots, len(self.delayed_indices)), np.float32)
        self.history_cursor = 0  # Slot for the next step's current release.
        self._fill_history_from_voltage()

    def _fill_history_from_voltage(self):
        self.history[:] = np.maximum(self.voltage[self.delayed_indices], 0.)
        self.history_cursor = self.elapsed_steps % len(self.history)

    def equilibrate(self, luminance, *, tolerance=2e-7, max_iterations=2000, input_off=False):
        result = super().equilibrate(luminance, tolerance=tolerance,
                                     max_iterations=max_iterations, input_off=input_off)
        self._fill_history_from_voltage()
        return result

    def step(self, luminance, *, input_off=False, freeze_indices=(), freeze_voltage=None):
        # The zero-delay path delegates the complete arithmetic, validation and
        # telemetry update, ensuring bit-identical behavior with the parent.
        if not self.delayed_indices.size:
            return super().step(luminance, input_off=input_off,
                                freeze_indices=freeze_indices, freeze_voltage=freeze_voltage)
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

        release = np.maximum(self.voltage, 0.)
        cursor, slots = self.history_cursor, len(self.history)
        self.history[cursor] = release[self.delayed_indices]
        newer = self.history[(cursor - self._whole_steps) % slots, self._columns]
        older = self.history[(cursor - self._whole_steps - 1) % slots, self._columns]
        release[self.delayed_indices] = newer + self._fraction * (older - newer)
        delta_release = release - self.reference_release
        recurrent = self.config.gain * (self.graph.incoming @ delta_release)
        equilibrium = self.reference_release + recurrent + drive
        alpha = np.float32(-math.expm1(-self.config.dt_s / self.config.tau_s))
        self.voltage += alpha * (equilibrium - self.voltage)
        if indices.size:
            self.voltage[indices] = frozen[indices]
        if not np.isfinite(self.voltage).all():
            raise ArithmeticError("Non-finite graded network state")
        self.elapsed_steps += 1
        self.history_cursor = (cursor + 1) % slots
        self.last_input[:] = values
        self.last_external_drive[:] = drive
        self.last_recurrent_drive[:] = recurrent
        return self.voltage

    def get_state(self):
        # A distinct outer format prevents the old GradedNetwork.set_state
        # from silently loading V while discarding the causally required past.
        return {"format": DELAYED_STATE_FORMAT, "base_state": super().get_state(),
                "delay_by_type_s": dict(self.delay_by_type_s),
                "delayed_indices": self.delayed_indices.copy(),
                "history": self.history.copy(), "history_cursor": self.history_cursor}

    def set_state(self, state):
        if (state.get("format") != DELAYED_STATE_FORMAT
                or state.get("delay_by_type_s") != dict(self.delay_by_type_s)):
            raise ValueError("Incompatible delayed graded model state")
        try:
            indices = np.asarray(state["delayed_indices"])
            history = np.asarray(state["history"], np.float32)
            cursor = state["history_cursor"]
            base = state["base_state"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid delayed graded model state") from exc
        if (indices.shape != self.delayed_indices.shape
                or not np.issubdtype(indices.dtype, np.integer)
                or not np.array_equal(indices, self.delayed_indices)):
            raise ValueError("Incompatible delayed cell indices")
        if (history.shape != self.history.shape or not np.isfinite(history).all()
                or np.any(history < 0)):
            raise ValueError("Invalid delayed release history")
        if (not isinstance(base, dict) or not isinstance(base.get("steps"), int)
                or not isinstance(cursor, int) or isinstance(cursor, bool)
                or not 0 <= cursor < len(self.history)
                or cursor != base["steps"] % len(self.history)):
            raise ValueError("Invalid delay history clock")
        super().set_state(base)
        self.history[:] = history
        self.history_cursor = cursor
