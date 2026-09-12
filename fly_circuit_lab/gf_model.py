"""Independent implementation of Ache et al. (2019), GF model equations 3-7.

Angles: full diameter in degrees. Speeds: degrees/s. Output: mV, not Hz.
Two fitted excitatory contributions and two fitted inhibitory contributions;
this is a phenomenological circuit model, not a replacement for the full LIF.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PARAMETER_PATH = Path(__file__).parent / "parameters" / "ache2019.json"


def parameters():
    return json.loads(PARAMETER_PATH.read_text(encoding="utf-8"))


def _components(theta_lplc2, omega_lc4, theta_i1, theta_i2, p):
    theta = np.asarray(theta_lplc2, dtype=float)
    lc4 = p["c1_mv_per_deg_s"] * np.asarray(omega_lc4, dtype=float)
    # The logarithmic Gaussian has a zero limit at theta=0.
    log_ratio = np.log(np.maximum(theta, np.finfo(float).tiny) / p["c3_deg"])
    lplc2 = p["c2_mv"] * np.exp(-.5 * (log_ratio / p["c4_log_width"])**2)
    lplc2 = np.where(theta > 0, lplc2, 0.)
    exponent = -(np.asarray(theta_i1) - p["c7_deg"]) / p["c8_deg"]
    i1 = p["c5_mv"] + p["c6_mv"] / (1 + np.exp(np.clip(exponent, -700, 700)))
    i2 = p["c9_mv"] * np.exp(-.5 * ((np.asarray(theta_i2) - p["c10_deg"]) / p["c11_deg"])**2)
    return {"lc4_mv": lc4, "lplc2_mv": lplc2, "i1_mv": i1, "i2_mv": i2}


def combine(components, p=None, block=()):
    p = p or parameters()
    blocked = set(block)
    unknown = blocked - {"lc4", "lplc2", "i1", "i2", "output", "input"}
    if unknown:
        raise ValueError(f"Unknown paper circuit intervention: {sorted(unknown)}")
    result = {key: np.asarray(value).copy() for key, value in components.items()}
    # A dependent inhibitory pathway disappears with its LC4 input.
    if "lc4" in blocked:
        blocked.add("i2")
    for name in ("lc4", "lplc2", "i1", "i2"):
        if name in blocked:
            result[f"{name}_mv"] = np.zeros_like(result[f"{name}_mv"])
    total = sum(p[f"weight_{name}"] * result[f"{name}_mv"] for name in ("lc4", "lplc2", "i1", "i2"))
    result["gf_raw_mv"] = total
    rest = _components(0., 0., 0., 0., p)
    baseline = sum(p[f"weight_{name}"] * float(rest[f"{name}_mv"]) for name in ("lc4", "lplc2", "i1", "i2") if name not in blocked)
    result["gf_delta_mv"] = np.zeros_like(total) if "output" in blocked else total - baseline
    if "output" in blocked:
        result["gf_raw_mv"] = np.zeros_like(total)
    return result


def evaluate(time_s, angle_deg, angular_speed_deg_s=None, *, block=(), p=None):
    """Delayed equations on a sampled trace, with a zero-stimulus prehistory.

    Angle/speed samples may be irregular; delays use linear interpolation.
    If speed is omitted, finite differences are an explicit input approximation.
    No smoothing is hidden in this function.
    """
    p = p or parameters()
    t, angle = np.asarray(time_s, float), np.asarray(angle_deg, float)
    if t.ndim != 1 or len(t) < 2 or angle.shape != t.shape or not np.isfinite(t).all() or not np.isfinite(angle).all():
        raise ValueError("Time and full-angle inputs must be finite equally-sized vectors")
    if np.any(np.diff(t) <= 0) or np.any((angle < 0) | (angle > 180)):
        raise ValueError("Time must increase and full angles must be in [0,180]")
    speed = np.gradient(angle, t) if angular_speed_deg_s is None else np.asarray(angular_speed_deg_s, float)
    if speed.shape != t.shape or not np.isfinite(speed).all():
        raise ValueError("Angular speed must be a finite vector matching time")
    if "input" in block:
        angle, speed = np.zeros_like(angle), np.zeros_like(speed)
    def delayed(values, delay):
        return np.interp(t - delay, t, values, left=0., right=float(values[-1]))
    result = combine(_components(delayed(angle, p["delay_lplc2_s"]), delayed(speed, p["delay_lc4_s"]),
                                  delayed(angle, p["delay_i1_s"]), delayed(angle, p["delay_i2_s"]), p), p, block)
    return {"time_s": t.copy(), "angle_deg": angle.copy(), "angular_speed_deg_s": speed.copy(), **result}


def centered_smooth(values, samples=5):
    """Centered moving average with odd shrinking end windows (MATLAB smooth style)."""
    values = np.asarray(values, float)
    if samples < 1 or samples % 2 != 1:
        raise ValueError("Centered smoothing window must be positive and odd")
    result = np.empty_like(values)
    half = samples // 2
    for i in range(len(values)):
        radius = min(half, i, len(values) - i - 1)
        result[i] = values[i-radius:i+radius+1].mean(axis=0)
    return result


class GFStream:
    """Causal 0.5-ms realization for online eye inputs, with serializable delays.

    A trailing 2.5-ms average adds 1 ms of delay compared with the centered
    five-sample paper plotting average. Existing LIF clocks are not modified.
    """
    timestep = .0005

    def __init__(self, channels=2, block=()):
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("Circuit channel count must be a positive integer")
        self.p = parameters()
        self.block = tuple(block)
        combine(_components(np.zeros(channels), np.zeros(channels), np.zeros(channels), np.zeros(channels), self.p), self.p, self.block)
        self.channels = channels
        self.angle = np.zeros((128, channels))
        self.speed = np.zeros((128, channels))
        self.output_history = np.zeros((5, channels))
        self.clock = 0
        self.last = {}

    def step(self, angle_deg, angular_speed_deg_s, seconds=.01):
        angle, speed = np.asarray(angle_deg, float), np.asarray(angular_speed_deg_s, float)
        if angle.shape != (self.channels,) or speed.shape != angle.shape or not np.isfinite(angle).all() or not np.isfinite(speed).all():
            raise ValueError("Invalid online circuit input")
        if np.any((angle < 0) | (angle > 180)):
            raise ValueError("Full angles must lie in [0,180]")
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Circuit time must be finite and positive")
        ticks = round(seconds / self.timestep)
        if ticks < 1 or not np.isclose(ticks * self.timestep, seconds, atol=1e-12, rtol=0):
            raise ValueError("Circuit time must be a positive multiple of 0.5 ms")
        if "input" in self.block:
            angle, speed = np.zeros_like(angle), np.zeros_like(speed)
        for _ in range(ticks):
            slot = self.clock % 128
            self.angle[slot], self.speed[slot] = angle, speed
            def past(array, delay):
                return array[(self.clock - round(delay / self.timestep)) % 128]
            self.last = combine(_components(past(self.angle, self.p["delay_lplc2_s"]), past(self.speed, self.p["delay_lc4_s"]),
                                              past(self.angle, self.p["delay_i1_s"]), past(self.angle, self.p["delay_i2_s"]), self.p), self.p, self.block)
            self.output_history[self.clock % 5] = self.last["gf_delta_mv"]
            self.clock += 1
        self.last["gf_online_mv"] = self.output_history.mean(axis=0)
        return {key: value.copy() for key, value in self.last.items()}

    def get_state(self):
        return {"version": 1, "parameters": self.p, "channels": self.channels, "block": list(self.block),
                "angle": self.angle.copy(), "speed": self.speed.copy(), "output_history": self.output_history.copy(),
                "clock": self.clock, "last": {k: v.copy() for k, v in self.last.items()}}

    def set_state(self, saved):
        if saved["version"] != 1 or saved["parameters"] != self.p or saved["channels"] != self.channels or saved["block"] != list(self.block):
            raise ValueError("Incompatible paper-circuit checkpoint")
        for key in ("angle", "speed", "output_history"):
            value = np.asarray(saved[key])
            if value.shape != getattr(self, key).shape or not np.isfinite(value).all():
                raise ValueError("Invalid circuit delay state")
            getattr(self, key)[:] = value
        if int(saved["clock"]) != saved["clock"] or saved["clock"] < 0:
            raise ValueError("Invalid circuit clock")
        self.clock = int(saved["clock"])
        self.last = {k: np.array(v) for k, v in saved["last"].items()}
