"""Optical input adapter; it sees eye pixels and camera calibration only."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from .gf_model import GFStream


class PaperVision:
    def __init__(self, fovy_degrees, *, block=(), threshold_mv=1.0):
        self.fovy = np.asarray(fovy_degrees, dtype=float)
        if self.fovy.shape != (2,) or np.any((self.fovy <= 0) | (self.fovy >= 180)):
            raise ValueError("Two calibrated camera vertical fields of view are required")
        if not np.isfinite(threshold_mv) or threshold_mv <= 0:
            raise ValueError("Engineering motor threshold must be positive")
        self.threshold_mv = float(threshold_mv)
        self.stream = GFStream(2, block)
        self.previous_angle = np.zeros(2)
        self.angle_deg = np.zeros(2)
        self.omega_deg_s = np.zeros(2)
        self.last = {"gf_online_mv": np.zeros(2)}
        y, x = np.mgrid[0:96, 0:96]
        self.rays = []
        for fov in self.fovy:
            focal = 48 / np.tan(np.deg2rad(fov / 2))
            rays = np.stack(((x - 47.5) / focal, (47.5 - y) / focal, np.ones_like(x)), axis=-1)
            self.rays.append(rays / np.linalg.norm(rays, axis=-1, keepdims=True))

    def measure_angle(self, rgb, eye):
        red, green, blue = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
        mask = (red > 1.5 * green + 20) & (blue > 1.5 * green + 20) & (red + blue > 150)
        components, count = label(mask)
        if count == 0:
            return 0.
        areas = np.bincount(components.ravel())
        areas[0] = 0
        winner = int(np.argmax(areas))
        if areas[winner] < 3:
            return 0.
        rays = self.rays[eye][components == winner]
        center = rays.mean(axis=0)
        center /= np.linalg.norm(center)
        # A sphere projects to a cone. The largest visible angular radius gives
        # a colour-segmented cone estimate, not hidden simulator geometry.
        diameter = 2 * np.rad2deg(np.arccos(np.clip(np.min(rays @ center), -1, 1)))
        return float(np.clip(diameter, 0, 180))

    def step(self, eyes, seconds=.01, blind=False):
        if np.asarray(eyes).shape != (2, 96, 96, 3):
            raise ValueError("Paper adapter uses the existing two 96x96 eye images")
        self.angle_deg = np.zeros(2) if blind else np.array([self.measure_angle(eyes[i], i) for i in range(2)])
        self.omega_deg_s = (self.angle_deg - self.previous_angle) / seconds
        if blind:
            self.omega_deg_s[:] = 0
        self.previous_angle = self.angle_deg.copy()
        self.last = self.stream.step(self.angle_deg, self.omega_deg_s, seconds)
        return np.clip(self.last["gf_online_mv"] / self.threshold_mv, 0, 4)

    def get_state(self):
        return {"fovy": self.fovy.copy(), "threshold_mv": self.threshold_mv,
                "previous_angle": self.previous_angle.copy(), "angle_deg": self.angle_deg.copy(),
                "omega_deg_s": self.omega_deg_s.copy(), "last": {k: v.copy() for k, v in self.last.items()},
                "stream": self.stream.get_state()}

    def set_state(self, saved):
        if not np.array_equal(saved["fovy"], self.fovy) or saved["threshold_mv"] != self.threshold_mv:
            raise ValueError("Incompatible paper optical adapter")
        for name in ("previous_angle", "angle_deg", "omega_deg_s"):
            value = np.asarray(saved[name], dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("Invalid optical adapter checkpoint")
            setattr(self, name, value.copy())
        self.last = {k: np.array(v) for k, v in saved["last"].items()}
        self.stream.set_state(saved["stream"])


class CircuitVisionAdapter:
    def __init__(self, probe, original, control):
        if control not in ("observe", "paper"):
            raise ValueError("Choose observe or paper control explicitly")
        self.probe, self.original, self.control = probe, original, control
        self.legacy_scores = np.zeros(2)

    def step(self, eyes, dt=.01, blind=False):
        scores = self.probe.step(eyes, dt, blind)
        self.legacy_scores = self.original.step(eyes, dt, blind) if self.original is not None else np.zeros(2)
        return self.legacy_scores.copy() if self.control == "observe" else scores

    def get_state(self):
        return {"version": 1, "control": self.control, "probe": self.probe.get_state(),
                "original": self.original.get_state() if self.original is not None else None,
                "legacy_scores": self.legacy_scores.copy()}

    def set_state(self, saved):
        if saved["version"] != 1 or saved["control"] != self.control:
            raise ValueError("Incompatible circuit vision wrapper")
        self.probe.set_state(saved["probe"])
        if self.original is not None:
            self.original.set_state(saved["original"])
        elif saved["original"] is not None:
            raise ValueError("Unexpected legacy detector state")
        self.legacy_scores = np.asarray(saved["legacy_scores"]).copy()
