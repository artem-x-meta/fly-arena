"""A deliberately simple CPU LIF model, NOT dynamics supplied by Janelia.

All cells are modeled as spiking (even naturally graded photoreceptors).
Voltages are relative to rest, in mV; dt is 1 ms. Synaptic current decays with
tau=5 ms and membrane voltage with tau=20 ms. Spikes arrive after 2 ms.
No training, reward, procedural wandering, or fallback motor policy occurs here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from numba import njit


@dataclass
class BrainConfig:
    synapse_gain: float = 0.075
    background: float = 4.0
    lamina_bias: float = 10.0
    arousal: float = 12.0
    turn_bias: float = 6.0
    visual_gain: float = 28.0
    speed_gain: float = 0.015
    turn_gain: float = 0.015


@njit(cache=True)
def advance(ptr, posts, weights, voltage, current, refractory, queue, clock, drive, steps, silenced=None):
    n = len(voltage)
    counts = np.zeros(n, dtype=np.int32)
    membrane_decay = np.exp(-1.0 / 20.0)
    current_decay = np.exp(-1.0 / 5.0)
    for tick in range(steps):
        slot = (clock + tick) % 3
        delivery = (clock + tick + 2) % 3
        # Process deliveries and all cells before propagating current spikes.
        for i in range(n):
            current[i] = current[i] * current_decay + queue[slot, i]
            queue[slot, i] = 0.0
            if silenced is not None and silenced[i]:
                voltage[i] = 0.0
                current[i] = 0.0
                refractory[i] = 0
                continue
            if refractory[i] > 0:
                refractory[i] -= 1
                voltage[i] = 0.0
                continue
            voltage[i] = max(-30.0, membrane_decay * voltage[i] + (1 - membrane_decay) * (drive[i] + current[i]))
            if voltage[i] >= 7.0:
                voltage[i] = 0.0
                refractory[i] = 2
                counts[i] += 1
                for edge in range(ptr[i], ptr[i + 1]):
                    queue[delivery, posts[edge]] += weights[edge]
    return counts


@njit(cache=True)
def signed_weights(ptr, counts, signs, gain):
    weights = np.empty(len(counts), dtype=np.float32)
    for source in range(len(signs)):
        for edge in range(ptr[source], ptr[source + 1]):
            weights[edge] = float(counts[edge]) * signs[source] * gain
    return weights


class MotorDecoder:
    """Engineering adapter: named descending-neuron rates -> two gait amplitudes."""
    def __init__(self, ports: dict, config: BrainConfig):
        self.ports, self.config = ports, config
        self.rates = {key: np.zeros(2) for key in ("DNa02", "DNp09", "MDN")}

    def update(self, counts: np.ndarray, seconds: float) -> np.ndarray:
        alpha = -np.expm1(-seconds / .1)
        for kind in self.rates:
            for side_i, side in enumerate(("L", "R")):
                selected = self.ports[kind][side]
                rate = float(counts[selected].mean() / seconds) if selected else 0.0
                self.rates[kind][side_i] += alpha * (rate - self.rates[kind][side_i])
        speed = np.clip(self.config.speed_gain * (self.rates["DNp09"].mean() - self.rates["MDN"].mean()), 0, 1)
        turn = np.clip(self.config.turn_gain * (self.rates["DNa02"][0] - self.rates["DNa02"][1]), -.65, .65)
        # Leftward command reduces left-side stride magnitude.
        return np.clip([speed - turn, speed + turn], 0, 1.2)


class Brain:
    def __init__(self, graph: Path, config: BrainConfig | None = None, seed: int = 1):
        if not (graph / "manifest.json").exists():
            raise FileNotFoundError(f"No prepared connectome at {graph}. Run: python -m fly_arena prepare")
        self.manifest = json.loads((graph / "manifest.json").read_text())
        if self.manifest["format_version"] != 1:
            raise ValueError("Unsupported graph format")
        self.config = config or BrainConfig()
        self.ids = np.load(graph / "ids.npy", mmap_mode="r")
        self.ptr = np.load(graph / "ptr.npy", mmap_mode="r")
        self.posts = np.load(graph / "posts.npy", mmap_mode="r")
        counts = np.load(graph / "counts.npy", mmap_mode="r")
        signs = np.load(graph / "signs.npy")
        self.weights = signed_weights(self.ptr, counts, signs, self.config.synapse_gain)
        n = len(signs)
        self.voltage = np.random.default_rng(seed).uniform(0, 2, n).astype(np.float32)
        self.current = np.zeros(n, np.float32)
        self.refractory = np.zeros(n, np.int16)
        self.queue = np.zeros((3, n), np.float32)
        self.clock = 0
        self.last_command = np.zeros(2)
        self.last_blocked_outputs = np.empty(0, np.int32)
        self.channel_contributions = {}
        self.ports = json.loads((graph / "ports.json").read_text())
        self.retina = np.array(self.ports["retina"], dtype=np.int32)
        self.eyes = np.array(self.ports["eye"], dtype=np.int32)
        self.uv = np.array(self.ports["uv"], dtype=np.float32)
        self.filtered_luminance = np.zeros(len(self.retina), dtype=np.float32)
        self.decoder = MotorDecoder(self.ports, self.config)
        self.base_drive = np.full(n, self.config.background, np.float32)
        self.base_drive[self.ports["lamina"]] += self.config.lamina_bias
        for side in ("L", "R"):
            self.base_drive[self.ports["DNp09"][side]] += self.config.arousal
            self.base_drive[self.ports["DNa02"][side]] += self.config.turn_bias

    def sample_eyes(self, frames: np.ndarray) -> np.ndarray:
        height, width = frames.shape[1:3]
        x = np.rint(self.uv[:, 0] * (width - 1)).astype(int)
        y = np.rint(self.uv[:, 1] * (height - 1)).astype(int)
        rgb = frames[self.eyes, y, x].astype(np.float32) / 255
        return rgb @ np.array([.2126, .7152, .0722], np.float32)

    def step(self, luminance: np.ndarray, milliseconds: int = 10, *, blind=False,
             sensory_currents: dict | None = None, modulation: dict | None = None,
             disabled_channels=(), blocked_outputs=()):
        """Advance all channels together; sparse currents are (indices, mV).

        ``blocked_outputs`` contains CSR indices resolved by NeuralAdapter. It
        silences those cells inside LIF, leaving all stored graph weights intact.
        ``blind`` affects vision alone. The original return contract is retained.
        """
        if milliseconds < 1 or int(milliseconds) != milliseconds:
            raise ValueError("Brain time must be a positive integer number of milliseconds")
        luminance = np.asarray(luminance, np.float32)
        if luminance.shape != self.filtered_luminance.shape or not np.isfinite(luminance).all():
            raise ValueError("Invalid retinal luminance")
        self.filtered_luminance += (1 - np.exp(-milliseconds / 10)) * (np.clip(luminance, 0, 1) - self.filtered_luminance)
        drive = self.base_drive.copy()
        disabled = set(disabled_channels)
        channels = {"vision": (self.retina, self.config.visual_gain * self.filtered_luminance / (.1 + self.filtered_luminance))}
        if blind:
            disabled.add("vision")
        for additions in (sensory_currents or {}, modulation or {}):
            if set(channels) & set(additions):
                raise ValueError("Neural channel names must be unique across sensory/modulation inputs")
            channels.update(additions)
        contributions = {"base": {"enabled": True, "recipients": len(drive), "sum_mv": float(np.sum(drive, dtype=np.float64)),
                                   "max_abs_mv": float(np.max(np.abs(drive)))}}
        for name, (indices, values) in channels.items():
            indices = np.asarray(indices)
            if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
                raise ValueError(f"Invalid indices for neural channel {name}")
            if np.any(indices < 0) or np.any(indices >= len(drive)):
                raise ValueError(f"Neural channel {name} indexes outside the graph")
            indices = indices.astype(np.int32)
            values = np.broadcast_to(np.asarray(values, np.float32), indices.shape)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite current for neural channel {name}")
            enabled = name not in disabled
            if enabled:
                np.add.at(drive, indices, values)
            contributions[name] = {"enabled": enabled, "recipients": len(indices),
                "sum_mv": float(np.sum(values, dtype=np.float64)) if enabled else 0.,
                "max_abs_mv": float(np.max(np.abs(values))) if enabled and len(values) else 0.}
        blocked = np.asarray(blocked_outputs)
        if blocked.ndim != 1 or (blocked.size and not np.issubdtype(blocked.dtype, np.integer)):
            raise ValueError("Blocked output indices must be an integer vector")
        if np.any(blocked < 0) or np.any(blocked >= len(drive)):
            raise ValueError("Blocked output index outside graph")
        silenced = None
        if blocked.size:
            silenced = np.zeros(len(drive), np.bool_)
            silenced[blocked.astype(np.int32)] = True
            self.queue[:, silenced] = 0.
        self.last_blocked_outputs = blocked.astype(np.int32).copy()
        counts = advance(self.ptr, self.posts, self.weights, self.voltage, self.current,
                         self.refractory, self.queue, self.clock, drive, milliseconds, silenced)
        self.clock += milliseconds
        command = self.decoder.update(counts, milliseconds / 1000)
        self.last_command = command.copy()
        self.channel_contributions = contributions
        return command, counts

    def get_state(self) -> dict:
        """All mutable neural/filter state for the full runtime checkpoint."""
        return {"voltage": self.voltage.copy(), "current": self.current.copy(),
                "refractory": self.refractory.copy(), "queue": self.queue.copy(),
                "clock": self.clock, "filtered_luminance": self.filtered_luminance.copy(),
                "decoder_rates": {name: value.copy() for name, value in self.decoder.rates.items()},
                "last_command": self.last_command.copy(),
                "last_blocked_outputs": self.last_blocked_outputs.copy(),
                "channel_contributions": json.loads(json.dumps(self.channel_contributions))}

    def set_state(self, state: dict):
        arrays = ("voltage", "current", "refractory", "queue", "filtered_luminance", "last_command")
        restored = {}
        for name in arrays:
            value = np.asarray(state[name])
            original = getattr(self, name)
            if value.shape != original.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint neural array {name}")
            restored[name] = value.astype(original.dtype)
        clock = state["clock"]
        if int(clock) != clock or clock < 0:
            raise ValueError("Invalid checkpoint neural clock")
        decoder_rates = state["decoder_rates"]
        if set(decoder_rates) != set(self.decoder.rates):
            raise ValueError("Invalid checkpoint decoder groups")
        for name, value in decoder_rates.items():
            value = np.asarray(value)
            if value.shape != (2,) or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f"Invalid checkpoint rate {name}")
        last_blocked = np.asarray(state.get("last_blocked_outputs", []))
        if last_blocked.ndim != 1 or (last_blocked.size and not np.issubdtype(last_blocked.dtype, np.integer)) or np.any(last_blocked < 0) or np.any(last_blocked >= len(self.voltage)):
            raise ValueError("Invalid checkpoint blocked neural outputs")
        for name, value in restored.items():
            getattr(self, name)[:] = value
        self.clock = int(clock)
        self.last_blocked_outputs = last_blocked.astype(np.int32).copy()
        for name, value in decoder_rates.items():
            self.decoder.rates[name][:] = value
        self.channel_contributions = state.get("channel_contributions", {})

    def save(self, path: Path):
        """Save the neural state only; this is not a full MuJoCo resume checkpoint."""
        np.savez_compressed(path, voltage=self.voltage, current=self.current, refractory=self.refractory,
                            queue=self.queue, clock=self.clock, luminance=self.filtered_luminance,
                            rates=np.array(list(self.decoder.rates.values())),
                            config=json.dumps(asdict(self.config)),
                            graph_manifest=json.dumps(self.manifest))
