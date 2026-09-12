"""Versioned TOML configuration and initial-condition/intervention scenarios."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tomllib


DEFAULT_CONFIG = {
    "schema_version": 1,
    "profile": "demo-fast",
    "organism": {"life_time_scale": 60.0},
    "initial": {"energy": 25.0, "sleep_pressure": .3, "dust_by_region": {
        "head": .8, "antenna_left": .3, "antenna_right": .3, "front_left": .2, "front_right": .2}},
    "environment": {},
    "behavior": {},
    "neural": {},
    "required_neural_pathways": ["feeding", "grooming"],
    "navigation": {"neural_weight": .4, "odor_turn_gain": 3.0, "search_speed": .55},
    "logging": {"telemetry_hz": 10.0, "rotate_rows": 36000, "checkpoint_wall_s": 120.0},
    "food": [{"id": "sugar_a", "x": 2.0, "y": 0.0, "radius": 1.0, "amount": 10.0,
              "energy_density": 12.0, "taste": 1.0, "odor": 1.0},
             {"id": "sugar_b", "x": -7.0, "y": 4.0, "radius": 1.0, "amount": 10.0,
              "energy_density": 12.0, "taste": 1.0, "odor": 1.0}],
    "events": [],
}


def merge_config(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path: Path | None = None, scenario: str | None = None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("rb") as stream:
            cfg = merge_config(cfg, tomllib.load(stream))
    if cfg["profile"] == "unscaled":
        cfg["organism"]["life_time_scale"] = 1.0
    if scenario:
        scenario_path = Path(scenario)
        if not scenario_path.is_file():
            scenario_path = Path(__file__).resolve().parent.parent / "scenarios" / f"{scenario}.toml"
        with scenario_path.open("rb") as stream:
            cfg = merge_config(cfg, tomllib.load(stream))
        cfg["scenario"] = scenario_path.stem
    if cfg["schema_version"] != 1 or cfg["profile"] not in ("demo-fast", "unscaled"):
        raise ValueError("Unsupported configuration schema or time profile")
    if cfg["profile"] == "unscaled" and cfg["organism"]["life_time_scale"] != 1:
        raise ValueError("unscaled profile requires life_time_scale=1")
    if not 0 <= cfg["navigation"]["neural_weight"] <= 1:
        raise ValueError("navigation.neural_weight must lie in [0,1]")
    if not 0 < cfg["logging"]["telemetry_hz"] <= 100:
        raise ValueError("telemetry_hz must be in (0, 100]")
    if cfg["logging"]["rotate_rows"] < 1 or cfg["logging"]["checkpoint_wall_s"] < 0:
        raise ValueError("Invalid log rotation or checkpoint frequency")
    config_digest(cfg)  # Also rejects non-finite JSON values.
    return cfg
