"""Physical looming trajectories and an explicit image-based hybrid detector.

The hybrid assay uses a magenta object isolated by colour in the eye images.
It is an engineered visual task, not a model of LC4/LPLC2 computation.
Neural experiments use the unmodified retinal input separately.
"""
from __future__ import annotations

import copy
import math

import numpy as np


def validate_looming(objects):
    for obj in objects:
        for name in ("start", "end"):
            point = np.asarray(obj[name], dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(f"Looming {name} must be a finite 3D position")
        if obj.get("radius", 0) <= 0 or obj.get("duration_s", 0) <= 0 or obj.get("time_s", 0) < 0:
            raise ValueError("Looming requires radius>0, duration_s>0 and time_s>=0")
        occupied = obj["duration_s"] + obj.get("hold_s", .1)
        if obj.get("repeat_s", occupied + 1) <= occupied:
            raise ValueError("Repeated looming trajectories must not overlap themselves")


def looming_position(obj, time_s):
    elapsed = time_s - obj.get("time_s", 0.)
    if elapsed < 0:
        return np.array([0., 0., -20.]), -1
    cycle = int(elapsed / obj["repeat_s"]) if obj.get("repeat_s") else 0
    phase = elapsed - cycle * obj.get("repeat_s", 0.)
    if phase > obj["duration_s"] + obj.get("hold_s", .1):
        return np.array([0., 0., -20.]), -1
    fraction = min(1., phase / obj["duration_s"])
    return np.asarray(obj["start"]) * (1 - fraction) + np.asarray(obj["end"]) * fraction, cycle


class LoomingEvents:
    def __init__(self, objects):
        validate_looming(objects)
        self.objects = copy.deepcopy(objects)
        self.last_cycles = [-1] * len(objects)

    def poll(self, time_s):
        events = []
        for i, obj in enumerate(self.objects):
            _, cycle = looming_position(obj, time_s)
            if cycle >= 0 and cycle > self.last_cycles[i]:
                speed = float(np.linalg.norm(np.asarray(obj["end"]) - obj["start"]) / obj["duration_s"])
                events.append({"kind": "looming", "id": obj.get("id", str(i)), "cycle": cycle,
                               "time_s": time_s, "radius_mm": obj["radius"], "speed_mm_s": speed,
                               "radius_over_speed_s": obj["radius"] / speed if speed > 0 else None})
                self.last_cycles[i] = cycle
        return events

    def get_state(self):
        return {"objects": copy.deepcopy(self.objects), "last_cycles": list(self.last_cycles)}

    def set_state(self, saved):
        if saved["objects"] != self.objects or len(saved["last_cycles"]) != len(self.objects):
            raise ValueError("Incompatible looming event state")
        self.last_cycles = list(saved["last_cycles"])


class VisualLoomingDetector:
    def __init__(self, config=None):
        self.config = {"min_area_fraction": .004, "growth_fraction_s": .025, "filter_tau_s": .04}
        self.config.update(config or {})
        if any(not math.isfinite(v) or v <= 0 for v in self.config.values()):
            raise ValueError("Visual detector constants must be finite and positive")
        self.previous_area = np.zeros(2)
        self.growth = np.zeros(2)
        self.scores = np.zeros(2)
        self.initialized = False
        self.last_area = np.zeros(2)

    def step(self, eyes, dt=.01, blind=False):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Visual time must be positive")
        frames = np.asarray(eyes)
        if frames.shape != (2, 96, 96, 3):
            raise ValueError("Looming detector expects the two existing 96x96 RGB eyes")
        if blind:
            self.previous_area[:] = 0
            self.growth[:] = 0
            self.scores[:] = 0
            self.last_area[:] = 0
            self.initialized = False
            return self.scores.copy()
        rgb = frames.astype(float)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = (red > 1.5 * green + 20) & (blue > 1.5 * green + 20) & (red + blue > 150)
        area = mask.mean(axis=(1, 2))
        derivative = (area - self.previous_area) / dt if self.initialized else np.zeros(2)
        alpha = -math.expm1(-dt / self.config["filter_tau_s"])
        self.growth += alpha * (derivative - self.growth)
        self.scores = np.where(area >= self.config["min_area_fraction"],
                               np.clip(self.growth / self.config["growth_fraction_s"] *
                                       np.sqrt(area / self.config["min_area_fraction"]), 0, 4), 0.)
        self.previous_area, self.last_area = area.copy(), area.copy()
        self.initialized = True
        return self.scores.copy()

    def get_state(self):
        return {"config": dict(self.config), "previous_area": self.previous_area.copy(),
                "growth": self.growth.copy(), "scores": self.scores.copy(),
                "last_area": self.last_area.copy(), "initialized": self.initialized}

    def set_state(self, saved):
        if saved["config"] != self.config:
            raise ValueError("Incompatible visual detector configuration")
        for name in ("previous_area", "growth", "scores", "last_area"):
            value = np.asarray(saved[name])
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("Invalid visual detector checkpoint")
            setattr(self, name, value.copy())
        self.initialized = bool(saved["initialized"])
