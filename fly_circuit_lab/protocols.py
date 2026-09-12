"""Published optical stimuli; full angles, positive r/v, time zero at collision."""
from __future__ import annotations

import numpy as np


def looming(rv_ms, *, dt=.0005, start_deg=5., end_deg=90., pre_s=.1, hold_s=.15):
    if rv_ms <= 0 or dt <= 0 or not 0 < start_deg < end_deg < 180:
        raise ValueError("Invalid looming protocol")
    ratio = rv_ms / 1000.
    start_time = -ratio / np.tan(np.deg2rad(start_deg / 2))
    end_time = -ratio / np.tan(np.deg2rad(end_deg / 2))
    t = np.arange(np.floor((start_time - pre_s) / dt) * dt, end_time + hold_s + dt / 2, dt)
    active = (t >= start_time) & (t < end_time)
    angle = np.zeros_like(t)
    speed = np.zeros_like(t)
    angle[active] = np.rad2deg(2 * np.arctan(ratio / -t[active]))
    speed[active] = np.rad2deg(2 * ratio / (t[active]**2 + ratio**2))
    angle[t >= end_time] = end_deg
    return t, angle, speed, {"rv_ms": rv_ms, "start_deg": start_deg, "end_deg": end_deg,
                              "dt_s": dt, "loom_start_s": start_time, "loom_end_s": end_time,
                              "time_zero": "hypothetical collision", "angle": "full angular diameter"}


def linear_expansion(speed_deg_s=100., *, dt=.0005, start_deg=5., end_deg=90.):
    if speed_deg_s <= 0:
        raise ValueError("Expansion speed must be positive")
    end = (end_deg - start_deg) / speed_deg_s
    t = np.arange(-.1, end + .15, dt)
    angle = np.where(t < 0, 0., np.minimum(end_deg, start_deg + speed_deg_s * t))
    speed = np.where((t >= 0) & (t < end), speed_deg_s, 0.)
    return t, angle, speed
