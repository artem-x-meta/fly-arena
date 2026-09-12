"""Controlled motion movies delivered to a model exclusively as RGB pixels.

Opposite directions share their entire preperiod image. Motion is periodic and
the measurement window discards complete onset cycles, then covers an integer
number of cycles. A radial grating supplies the same control for expansion and
contraction; it is an engineered motion assay, not an approaching solid object.
The original dark-disk assay and original nearest-UV retinal sampler remain
available unchanged. Neither direction nor any angle is passed to neurons.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from typing import Mapping

import numpy as np

from fly_bio.stimuli import (
    FrameProtocol,
    matched_retinal_flash,
    retinal_movie,
)


DIRECTIONS = ("right", "left", "up", "down")
RADIAL_DIRECTIONS = ("expanding", "contracting")
EYES = ("both", "left", "right")


def _times_array(times_s) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("times_s must be a finite, nonnegative 1-D array")
    if len(times) > 1 and (np.diff(times) < 0).any():
        raise ValueError("times_s must be nondecreasing")
    return times


@dataclass(frozen=True)
class MotionProtocol:
    """A square display grating with defined temporal and spatial frequency.

    ``spatial_cycles`` means cycles per display width, also for radial distance.
    Thus two radial cycles per width give one cycle from center to a side edge.
    This is display geometry, not anatomical retinotopy or motion preprocessing.
    ``contrast`` is the luminance amplitude divided by the available symmetric
    range around background; at background .5 this is Michelson contrast.
    ``polarity=-1`` reverses bright/dark relative to the same background.

    The default stimulus lasts three cycles. The default analysis discards one
    complete cycle and measures the remaining two. Pre/post periods are static
    holds; there is no image-onset or image-offset flash at the motion boundary.
    """

    kind: str = "grating"
    direction: str = "right"
    size: int = 96
    contrast: float = .8
    background: float = .5
    spatial_cycles: float = 2.0
    temporal_hz: float = 2.0
    phase_rad: float = 0.0
    polarity: int = 1
    eye: str = "both"
    pre_s: float = .5
    stimulus_s: float = 1.5
    post_s: float = .25

    def __post_init__(self):
        choices = {"grating": DIRECTIONS, "rings": RADIAL_DIRECTIONS}
        if self.kind not in choices:
            raise ValueError("kind must be 'grating' or 'rings'")
        if self.direction not in choices[self.kind]:
            raise ValueError(f"Invalid direction {self.direction!r} for {self.kind}")
        if self.eye not in EYES:
            raise ValueError(f"eye must be one of {EYES}")
        if isinstance(self.size, bool) or not isinstance(self.size, (int, np.integer)) or self.size < 8:
            raise ValueError("size must be an integer >= 8")
        numbers = [self.contrast, self.background, self.spatial_cycles,
                   self.temporal_hz, self.phase_rad, self.pre_s,
                   self.stimulus_s, self.post_s]
        if not np.isfinite(numbers).all():
            raise ValueError("Protocol parameters must be finite")
        if not 0 <= self.contrast <= 1 or not 0 <= self.background <= 1:
            raise ValueError("contrast and background must lie in [0, 1]")
        if self.spatial_cycles <= 0 or self.temporal_hz <= 0:
            raise ValueError("Spatial and temporal frequency must be positive")
        if isinstance(self.polarity, bool) or self.polarity not in (-1, 1):
            raise ValueError("polarity must be +1 or -1")
        if self.pre_s < 0 or self.post_s < 0 or self.stimulus_s <= 0:
            raise ValueError("Pre/post durations must be nonnegative; stimulus_s positive")
        cycles = self.stimulus_s * self.temporal_hz
        if cycles < 1 or not np.isclose(cycles, round(cycles), rtol=0, atol=1e-9):
            raise ValueError("stimulus_s must cover a positive integer number of temporal cycles")

    @property
    def duration_s(self) -> float:
        return self.pre_s + self.stimulus_s + self.post_s

    @property
    def stimulus_cycles(self) -> int:
        return int(round(self.stimulus_s * self.temporal_hz))

    def times(self, dt_s: float = .01) -> np.ndarray:
        """Return frame-start times for a half-open trial [0, duration)."""
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        count = int(np.ceil(self.duration_s / dt_s - 1e-12))
        return np.arange(count, dtype=np.float64) * dt_s

    def analysis_window(self, discard_cycles: int = 1) -> tuple[float, float]:
        """The predetermined half-open window, excluding onset and static holds."""
        if (isinstance(discard_cycles, bool)
                or not isinstance(discard_cycles, (int, np.integer))
                or not 0 <= discard_cycles < self.stimulus_cycles):
            raise ValueError("discard_cycles must leave at least one complete stimulus cycle")
        return (self.pre_s + discard_cycles / self.temporal_hz,
                self.pre_s + self.stimulus_s)

    def analysis_mask(self, times_s, discard_cycles: int = 1) -> np.ndarray:
        """Select the fixed window; use a frame dt that divides the cycle period."""
        times = _times_array(times_s)
        start, end = self.analysis_window(discard_cycles)
        # Align decimal frame boundaries robustly without including the endpoint.
        return (times >= start - 1e-12) & (times < end - 1e-12)

    def frames(self, times_s) -> np.ndarray:
        times = _times_array(times_s)
        result = np.full((len(times), 2, self.size, self.size, 3),
                         self.background, dtype=np.float32)
        if not len(times):
            return result
        coordinate = (np.arange(self.size, dtype=np.float64) + .5) / self.size - .5
        x, y = np.meshgrid(coordinate, coordinate)
        if self.kind == "rings":
            distance = np.hypot(x, y)
        else:
            distance = x if self.direction in ("right", "left") else y
        sign = 1 if self.direction in ("right", "down", "expanding") else -1
        elapsed = np.clip(times - self.pre_s, 0, self.stimulus_s)
        spatial_phase = 2 * np.pi * self.spatial_cycles * distance + self.phase_rad
        # Reduce periodic phase so the integer-cycle postperiod is exactly the
        # initial image. Opposite directions also have bitwise-identical starts.
        cycles = np.remainder(self.temporal_hz * elapsed, 1.0)
        phase = spatial_phase[None] - sign * 2 * np.pi * cycles[:, None, None]
        amplitude = self.polarity * self.contrast * min(self.background, 1 - self.background)
        gray = (self.background + amplitude * np.cos(phase)).astype(np.float32)
        selected = (0, 1) if self.eye == "both" else ((0,) if self.eye == "left" else (1,))
        for eye in selected:
            result[:, eye] = gray[..., None]
        return result

    def baseline_frame(self) -> np.ndarray:
        """Initial (2,H,W,3) image for warmup and a condition-matched static run."""
        return self.frames([0.0])[0]

    @property
    def protocol_config(self) -> dict:
        baseline = self.baseline_frame()
        return {
            **asdict(self),
            "generator": "fly_bio_selectivity.stimuli.MotionProtocol:v1",
            "coordinates": "pixel centers; x increases right, y increases down",
            "spatial_frequency_units": "cycles per display width",
            "radial_control_scope": "periodic radial motion; not a solid looming object",
            "stimulus_cycles": self.stimulus_cycles,
            "phase_matched_controls": "opposite directions have exactly the same preperiod image",
            "analysis": "discard onset cycles, then measure a fixed integer number of cycles",
            "postperiod": "hold final image (identical to initial at integer cycle count)",
            "baseline_image": {
                "encoding": "regenerate at time_s=0 with this configuration",
                "sha256_float32": hashlib.sha256(baseline.tobytes()).hexdigest(),
                "mean_luminance_per_eye": baseline.mean(axis=(1, 2, 3)).tolist(),
                "warmup": "establish this image before motion; compare with it held static",
            },
            "matched_flash": "uniform image matches each eye's sampled retinal mean, not pixel mean",
            "neural_input": "original nearest-UV retinal sampling of RGB images only",
            "labels_are_model_inputs": False,
            "scope": "engineered display, not a calibrated compound eye or anatomical retinotopy",
        }


def direction_protocols(*, phases=(0.0, np.pi), **kwargs) -> tuple[MotionProtocol, ...]:
    """Four cardinal directions at each phase, with common explicit parameters."""
    return tuple(MotionProtocol(kind="grating", direction=direction,
                                phase_rad=float(phase), **kwargs)
                 for phase in phases for direction in DIRECTIONS)


def radial_protocols(*, phases=(0.0, np.pi), **kwargs) -> tuple[MotionProtocol, ...]:
    """Expansion/contraction pairs at each phase, sharing the starting image."""
    return tuple(MotionProtocol(kind="rings", direction=direction,
                                phase_rad=float(phase), **kwargs)
                 for phase in phases for direction in RADIAL_DIRECTIONS)


def retinal_mean_control(protocol: MotionProtocol | FrameProtocol, times_s,
                         ports: Mapping) -> np.ndarray:
    """Return RGB uniform flashes matched to actual sampled dose in each eye.

    This delegates to the original sampler/control; no neural output is used to
    construct the display. Warm up each such control on its own first image.
    """
    return matched_retinal_flash(protocol.frames(times_s), ports)


def opposite_protocol(protocol: MotionProtocol) -> MotionProtocol:
    opposite = {"right": "left", "left": "right", "up": "down", "down": "up",
                "expanding": "contracting", "contracting": "expanding"}
    return replace(protocol, direction=opposite[protocol.direction])
