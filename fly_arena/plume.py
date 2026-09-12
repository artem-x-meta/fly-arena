"""Bounded stochastic odor packets, advected at the environment exchange rate.

Concentrations and wind are engineering signals, not calibrated fluid dynamics.
The field owns world positions. Only two local samples leave this module.
"""
from __future__ import annotations

import copy
import math

import numpy as np


PLUME_DEFAULTS = {
    "wind_speed_mm_s": 4.0, "wind_direction_deg": 180.0,
    "wind_swing_deg": 8.0, "wind_period_s": 12.0,
    "emission_rate_hz": 12.0, "on_mean_s": .55, "off_mean_s": .3,
    "axial_sigma_mm": .22, "lateral_sigma_mm": .35,
    "axial_diffusion_mm2_s": .015, "lateral_diffusion_mm2_s": .15,
    "lifetime_s": 12.0, "max_packets": 512, "prefill_s": 8.0,
}


class OdorPlume:
    def __init__(self, food, config=None, seed=1):
        self.config = dict(PLUME_DEFAULTS)
        unknown = set(config or {}) - self.config.keys()
        if unknown:
            raise ValueError(f"Unknown plume settings: {sorted(unknown)}")
        self.config.update(config or {})
        c = self.config
        if any(not math.isfinite(v) for v in c.values()):
            raise ValueError("Plume settings must be finite")
        if any(c[key] <= 0 for key in ("wind_period_s", "on_mean_s", "off_mean_s", "axial_sigma_mm", "lateral_sigma_mm", "lifetime_s", "max_packets")):
            raise ValueError("Plume time constants, widths and capacity must be positive")
        if any(c[key] < 0 for key in ("wind_speed_mm_s", "emission_rate_hz", "axial_diffusion_mm2_s", "lateral_diffusion_mm2_s", "prefill_s")):
            raise ValueError("Plume rates must be nonnegative")
        if int(c["max_packets"]) != c["max_packets"]:
            raise ValueError("max_packets must be an integer")
        self.rng = np.random.default_rng(seed)
        self.ids = [patch.id for patch in food]
        self.emitting = np.ones(len(food), dtype=bool)
        # Columns: x_mm, y_mm, age_physics_s, strength; bounded oldest-first queue.
        self.packets = np.empty((0, 4), dtype=float)
        self.time_s = -float(c["prefill_s"])
        self.dropped_packets = 0
        self.steps = 0
        # A documented pre-existing field at t=0; no organism time passes here.
        while self.time_s < -1e-10:
            self.advance(food, min(.01, -self.time_s))
        self.time_s = 0.0
        self.steps = 0

    @property
    def wind(self):
        c = self.config
        angle = math.radians(c["wind_direction_deg"] + c["wind_swing_deg"] * math.sin(2 * math.pi * self.time_s / c["wind_period_s"]))
        return c["wind_speed_mm_s"] * np.array([math.cos(angle), math.sin(angle)])

    def advance(self, food, dt):
        if not 0 < dt <= .010000001:
            raise ValueError("Plume advances on intervals no longer than 10 ms")
        if [p.id for p in food] != self.ids:
            raise ValueError("Plume source identity changed")
        wind = self.wind
        if len(self.packets):
            self.packets[:, :2] += wind * dt
            self.packets[:, 2] += dt
            self.packets = self.packets[self.packets[:, 2] <= self.config["lifetime_s"]]
        born = []
        for index, patch in enumerate(food):
            tau = self.config["on_mean_s"] if self.emitting[index] else self.config["off_mean_s"]
            if self.rng.random() < -math.expm1(-dt / tau):
                self.emitting[index] = not self.emitting[index]
            if self.emitting[index] and patch.odor > 0 and (patch.amount > 0 or patch.odor_when_empty):
                count = self.rng.poisson(self.config["emission_rate_hz"] * dt)
                for _ in range(count):
                    jitter = self.rng.normal(0, .1, 2)
                    born.append([patch.x + jitter[0], patch.y + jitter[1], 0., patch.odor])
        if born:
            self.packets = np.concatenate((self.packets, np.asarray(born)))
        limit = int(self.config["max_packets"])
        if len(self.packets) > limit:
            self.dropped_packets += len(self.packets) - limit
            self.packets = self.packets[-limit:]
        self.time_s += dt
        self.steps += 1

    def sample(self, point):
        if not len(self.packets):
            return 0.
        c = self.config
        wind = self.wind
        norm = np.linalg.norm(wind)
        axis = wind / norm if norm > 1e-10 else np.array([1., 0.])
        cross = np.array([-axis[1], axis[0]])
        delta = np.asarray(point)[:2] - self.packets[:, :2]
        age = self.packets[:, 2]
        axial_variance = c["axial_sigma_mm"]**2 + 2 * c["axial_diffusion_mm2_s"] * age
        lateral_variance = c["lateral_sigma_mm"]**2 + 2 * c["lateral_diffusion_mm2_s"] * age
        power = -.5 * ((delta @ axis)**2 / axial_variance + (delta @ cross)**2 / lateral_variance)
        return float(np.sum(self.packets[:, 3] * np.exp(power - age / c["lifetime_s"]) / (1 + .25 * age)))

    def get_state(self):
        return {"version": 1, "config": dict(self.config), "ids": list(self.ids),
                "packets": self.packets.copy(), "emitting": self.emitting.copy(),
                "rng": copy.deepcopy(self.rng.bit_generator.state), "time_s": self.time_s,
                "steps": self.steps, "dropped_packets": self.dropped_packets}

    def set_state(self, saved):
        if saved["version"] != 1 or saved["config"] != self.config or saved["ids"] != self.ids:
            raise ValueError("Incompatible plume checkpoint")
        packets = np.asarray(saved["packets"])
        if packets.ndim != 2 or packets.shape[1] != 4 or len(packets) > self.config["max_packets"] or not np.isfinite(packets).all():
            raise ValueError("Invalid saved odor packets")
        self.packets = packets.copy()
        self.emitting = np.asarray(saved["emitting"], dtype=bool).copy()
        self.rng.bit_generator.state = copy.deepcopy(saved["rng"])
        self.time_s, self.steps = float(saved["time_s"]), int(saved["steps"])
        self.dropped_packets = int(saved["dropped_packets"])
