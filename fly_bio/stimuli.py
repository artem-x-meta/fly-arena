"""Reproducible display protocols; the neural model receives pixels only.

This is an engineered visual display, not a reconstruction of the compound eye.
Angles and condition names belong to the display and report, never to a neural
input beyond the retinal image sampler. Retinal registration deliberately uses
the unchanged nearest-UV mapping in ``fly_arena.brain.Brain.sample_eyes``.

Each condition starts with its first image already present. In particular a
shrinking disk is large throughout the preperiod: it must not be introduced as
an onset flash at motion onset. Compare a trial with the same initial image held
static, and warm up the neural state on that image before starting the trial.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping

import numpy as np


KINDS = ("uniform", "dark_loom", "shrinking", "translating", "matched_flash",
         "bright_loom", "input_off", "static")
EYES = ("both", "left", "right")
LUMINANCE_WEIGHTS = np.array([.2126, .7152, .0722], np.float32)


@dataclass(frozen=True)
class FrameProtocol:
    """Parameters of a square pinhole display; all angles are full diameters.

    ``contrast`` is a luminance excursion relative to the available distance
    from ``background`` to black (dark conditions) or white (bright loom).
    With the defaults both excursions have magnitude .4: disk .1 or .9 on .5.
    The moving disk is held at its final position/size throughout the postperiod.
    """

    kind: str = "dark_loom"
    size: int = 96
    fov_deg: float = 90.0
    contrast: float = .8
    background: float = .5
    pre_s: float = .3
    stimulus_s: float = .8
    post_s: float = .4
    min_diameter_deg: float = 6.0
    max_diameter_deg: float = 70.0
    translating_diameter_deg: float = 10.0
    eye: str = "both"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown stimulus {self.kind!r}; choose from {KINDS}")
        if self.eye not in EYES:
            raise ValueError(f"Unknown eye {self.eye!r}; choose from {EYES}")
        if not isinstance(self.size, (int, np.integer)) or self.size < 8:
            raise ValueError("size must be an integer >= 8")
        numbers = [self.fov_deg, self.contrast, self.background, self.pre_s,
                   self.stimulus_s, self.post_s, self.min_diameter_deg,
                   self.max_diameter_deg, self.translating_diameter_deg]
        if not np.isfinite(numbers).all():
            raise ValueError("Protocol parameters must be finite")
        if not 0 < self.fov_deg < 180:
            raise ValueError("fov_deg must lie between 0 and 180")
        if not 0 <= self.contrast <= 1 or not 0 <= self.background <= 1:
            raise ValueError("contrast and background must lie in [0, 1]")
        if self.pre_s < 0 or self.post_s < 0 or self.stimulus_s <= 0:
            raise ValueError("Pre/post durations must be nonnegative; stimulus_s positive")
        if not 0 < self.min_diameter_deg < self.max_diameter_deg < self.fov_deg:
            raise ValueError("Need 0 < min diameter < max diameter < fov")
        if not 0 < self.translating_diameter_deg < self.fov_deg:
            raise ValueError("Translating diameter must lie between 0 and fov")

    @property
    def duration_s(self) -> float:
        return self.pre_s + self.stimulus_s + self.post_s

    def times(self, dt_s: float = .01) -> np.ndarray:
        """Return frame-start times for a half-open trial [0, duration)."""
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        count = int(np.ceil(self.duration_s / dt_s - 1e-12))
        return np.arange(count, dtype=np.float64) * dt_s

    def frames(self, times_s) -> np.ndarray:
        return _frames(self, times_s)

    def baseline_frame(self) -> np.ndarray:
        """One (2,H,W,3) frame for warmup and a condition-matched static trial."""
        return self.frames([0.0])[0]

    @property
    def protocol_config(self) -> dict:
        """JSON-safe display recipe, baseline fingerprint, and explicit scope."""
        baseline = self.baseline_frame()
        return {
            **asdict(self),
            "generator": "fly_bio.stimuli.FrameProtocol:v1",
            "angle_convention": "full angular diameter, degrees",
            "projection": "square pinhole, pixel-center unit rays, no antialiasing",
            "trajectory": "linear cotangent of half diameter; shrinking reverses loom",
            "postperiod": "hold final image; no offset flash",
            "baseline_image": {
                "encoding": "regenerate using this configuration at time_s=0",
                "sha256_float32": hashlib.sha256(baseline.tobytes()).hexdigest(),
                "shape": list(baseline.shape),
                "mean_luminance_per_eye": baseline.mean(axis=(1, 2, 3)).tolist(),
                "comparison": "same initial image held for entire reference trial",
                "warmup": "establish initial image before trial; no onset flash",
            },
            "matched_flash": "uniform per-eye image matches dark-loom frame mean exactly",
            "neural_input": "sampled image luminance at existing retinal ports only",
            "labels_are_model_inputs": False,
            "scope": "engineered display protocol, not a calibrated biological retina",
        }


def _unit_rays(size: int, fov_deg: float) -> np.ndarray:
    half_span = np.tan(np.deg2rad(fov_deg) / 2)
    coordinate = ((np.arange(size) + .5) / size * 2 - 1) * half_span
    x, y = np.meshgrid(coordinate, coordinate)
    rays = np.stack((x, y, np.ones_like(x)), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _loom_diameter(protocol: FrameProtocol, progress: float) -> float:
    near_cot = 1 / np.tan(np.deg2rad(protocol.min_diameter_deg) / 2)
    far_cot = 1 / np.tan(np.deg2rad(protocol.max_diameter_deg) / 2)
    cot = (1 - progress) * near_cot + progress * far_cot
    return float(np.rad2deg(2 * np.arctan(1 / cot)))


def _frames(protocol: FrameProtocol, times_s) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("times_s must be a finite, nonnegative 1-D array")
    if len(times) > 1 and (np.diff(times) < 0).any():
        raise ValueError("times_s must be nondecreasing")
    frames = np.full((len(times), 2, protocol.size, protocol.size, 3),
                     protocol.background, dtype=np.float32)
    if protocol.kind in ("uniform", "input_off") or not len(times):
        return frames
    rays = _unit_rays(protocol.size, protocol.fov_deg)
    selected_eyes = (0, 1) if protocol.eye == "both" else ((0,) if protocol.eye == "left" else (1,))
    disk_level = protocol.background * (1 - protocol.contrast)
    if protocol.kind == "bright_loom":
        disk_level = protocol.background + protocol.contrast * (1 - protocol.background)
    progress = np.clip((times - protocol.pre_s) / protocol.stimulus_s, 0, 1)
    for index, fraction in enumerate(progress):
        center = np.array([0., 0., 1.])
        if protocol.kind == "shrinking":
            diameter = _loom_diameter(protocol, 1 - fraction)
        elif protocol.kind == "translating":
            diameter = protocol.translating_diameter_deg
            # Keep the full disk inside the horizontal view, with 1 degree margin.
            extent = max(0., (protocol.fov_deg - diameter) / 2 - 1.)
            azimuth = np.deg2rad((2 * fraction - 1) * extent)
            center = np.array([np.sin(azimuth), 0., np.cos(azimuth)])
        else:
            diameter = _loom_diameter(protocol, 0 if protocol.kind == "static" else fraction)
        mask = (rays @ center) >= np.cos(np.deg2rad(diameter) / 2)
        if protocol.kind == "matched_flash":
            # Float32 gray makes equality precise to image averaging roundoff.
            gray = np.full((protocol.size, protocol.size), protocol.background, np.float32)
            gray[mask] = disk_level
            level = gray.mean(dtype=np.float64)
            for eye in selected_eyes:
                frames[index, eye] = level
        else:
            for eye in selected_eyes:
                frames[index, eye][mask] = disk_level
    return frames


def stimulus_frames(kind: str, times_s, **kwargs) -> np.ndarray:
    """Generate float32 (T,2,H,W,3) images in [0,1] with ``FrameProtocol`` options."""
    return FrameProtocol(kind=kind, **kwargs).frames(times_s)


def baseline_frames(protocol: FrameProtocol, count: int = 1) -> np.ndarray:
    """Repeat the condition's initial image, e.g. for neural warmup/static control."""
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    return np.repeat(protocol.baseline_frame()[None], count, axis=0)


def retinal_movie(frames: np.ndarray, ports: Mapping) -> np.ndarray:
    """Sample RGB movie into (T,nretina) luminance in stored retinal-port order.

    The exact old nearest-pixel/BT.709 rule applies. ``uint8`` means [0,255];
    floating point means [0,1], avoiding data-dependent brightness rescaling.
    No smoothing, adaptation, motion detection, or condition metadata is applied.
    """
    frames = np.asarray(frames)
    if frames.ndim != 5 or frames.shape[1] != 2 or frames.shape[-1] != 3:
        raise ValueError("Expected frames shape (T,2,H,W,3)")
    if min(frames.shape[2:4]) < 1:
        raise ValueError("Images must have nonzero height and width")
    if frames.dtype != np.uint8 and not np.issubdtype(frames.dtype, np.floating):
        raise ValueError("Frames must be uint8 [0,255] or floating point [0,1]")
    if not np.isfinite(frames).all():
        raise ValueError("Frames must be finite")
    if np.issubdtype(frames.dtype, np.floating) and frames.size and (frames.min() < 0 or frames.max() > 1):
        raise ValueError("Floating-point frames must be normalized to [0,1]")
    retina = np.asarray(ports["retina"])
    eye_values = np.asarray(ports["eye"])
    uv = np.asarray(ports["uv"], np.float32)
    if retina.ndim != 1 or eye_values.shape != retina.shape or uv.shape != (len(retina), 2):
        raise ValueError("Retina, eye, and UV port shapes disagree")
    if not np.isin(eye_values, [0, 1]).all() or not np.isfinite(uv).all() or (uv < 0).any() or (uv > 1).any():
        raise ValueError("Eye indices must be 0/1 and UV coordinates in [0,1]")
    eyes = eye_values.astype(np.int64)
    height, width = frames.shape[2:4]
    x = np.rint(uv[:, 0] * (width - 1)).astype(int)
    y = np.rint(uv[:, 1] * (height - 1)).astype(int)
    sampled = frames[:, eyes, y, x].astype(np.float32)
    if frames.dtype == np.uint8:
        sampled /= 255
    return sampled @ LUMINANCE_WEIGHTS


def matched_retinal_flash(frames: np.ndarray, ports: Mapping) -> np.ndarray:
    """Uniform images matching the movie's per-eye mean *retinal* luminance.

    Retinal UV samples are not uniformly distributed over display pixels. This
    control therefore matches total sampled retinal drive more closely than
    ``matched_flash``, which matches display-pixel means. The original movie's
    spatial arrangement is discarded, preserving each eye's temporal luminance
    mean within floating-point precision. There is no neural-response feedback.
    """
    sampled = retinal_movie(frames, ports)
    eyes = np.asarray(ports["eye"])
    result = np.empty(np.asarray(frames).shape, dtype=np.float32)
    for eye in (0, 1):
        mask = eyes == eye
        if not mask.any():
            raise ValueError("Retinal-mean flash requires sampled ports for both eyes")
        means = sampled[:, mask].mean(axis=1, dtype=np.float64)
        result[:, eye] = means[:, None, None, None]
    return result
