"""Contracts at the environment/organism/brain/motor boundaries.

Only the environment sees food coordinates. SensorFrame carries local readings.
Contact durations use physical seconds; tangential travel uses millimetres.
"""
from dataclasses import dataclass, field

import numpy as np


DUST_REGIONS = ("head", "antenna_left", "antenna_right", "front_left", "front_right")


@dataclass
class SensorFrame:
    time_s: float = 0.0
    vision: np.ndarray | None = None
    odor_left: float = 0.0
    odor_right: float = 0.0
    tarsal_taste: dict[str, float] = field(default_factory=dict)
    mouth_taste: float = 0.0
    tactile_by_region: dict[str, float] = field(default_factory=dict)
    dust_afferents: dict[str, float] = field(default_factory=dict)
    upright: float = 1.0
    ground_support: int = 6
    movement: float = 0.0
    local_wake_stimulus: float = 0.0
    mouth_contact: bool = False
    # Local airflow measured at the antennae, expressed in the fly's frame.
    wind_body_x: float = 0.0
    wind_body_y: float = 0.0
    looming_left: float = 0.0
    looming_right: float = 0.0


@dataclass
class NeuralReadout:
    walk_left: float = 0.0
    walk_right: float = 0.0
    feed_drive: float = 0.0
    groom_drives: dict[str, float] = field(default_factory=dict)
    sleep_drive: float = 0.0
    supported_ports: dict[str, object] = field(default_factory=dict)
    raw_rates: dict[str, float] = field(default_factory=dict)
    escape_drive: float = 0.0
    escape_turn: float = 0.0


@dataclass
class PhysicalEvents:
    # The motor integrates permission AND mouth overlap at every physics step.
    contact_s_by_food: dict[str, float] = field(default_factory=dict)
    mouth_contact_s_by_food: dict[str, float] = field(default_factory=dict)
    grooming_sliding_by_pair: dict[str, float] = field(default_factory=dict)
    grooming_contact_s_by_pair: dict[str, float] = field(default_factory=dict)
    local_wake_stimulus: float = 0.0
