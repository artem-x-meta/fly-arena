"""Engineered surge/cast control driven solely by local odor and airflow."""
from __future__ import annotations

import copy
import math

import numpy as np


SEARCH_DEFAULTS = {
    "odor_on": .08, "odor_off": .035, "lost_after_s": .2,
    "surge_speed": .65, "cast_speed": .5, "turn_gain": .65,
    "cast_first_s": .45, "cast_expansion_s": .3, "cast_max_s": 2.,
    "cast_timeout_s": 10., "neural_weight": .25,
}


class SurgeCastNavigator:
    def __init__(self, config, seed=1):
        self.config = dict(SEARCH_DEFAULTS)
        self.config.update({k: v for k, v in config.items() if k in SEARCH_DEFAULTS})
        if not 0 <= self.config["odor_off"] < self.config["odor_on"]:
            raise ValueError("Odor hysteresis requires 0 <= off < on")
        if any(not math.isfinite(v) or v < 0 for v in self.config.values()):
            raise ValueError("Invalid search configuration")
        self.rng = np.random.default_rng(seed)
        self.phase = "CAST"
        self.cast_sign = int(self.rng.choice([-1, 1]))
        self.cast_leg = 0
        self.leg_elapsed = 0.
        self.phase_elapsed = 0.
        self.since_odor = 1e6
        self.in_odor = False
        self.turn_noise = 0.
        self.last_engineered = np.zeros(2)
        self.last_neural = np.zeros(2)
        self.heading_error = 0.
        self.transitions = 0

    def _phase(self, name):
        if self.phase != name:
            self.phase, self.phase_elapsed = name, 0.
            self.transitions += 1
            if name == "CAST":
                self.cast_leg, self.leg_elapsed = 0, 0.

    def step(self, frame, neural, mode, dt=.01, *, active=True):
        self.last_neural = np.array([neural.walk_left, neural.walk_right])
        if mode == "ethology-neural":
            self._phase("NEURAL_ONLY")
            self.last_engineered[:] = 0.
            return self.last_neural.copy()
        c = self.config
        odor = max(frame.odor_left, frame.odor_right)
        self.in_odor = odor >= (c["odor_off"] if self.in_odor else c["odor_on"])
        self.since_odor = 0. if self.in_odor else self.since_odor + dt
        if not active:
            self.last_engineered[:] = 0.
            return self.last_neural.copy()
        self.phase_elapsed += dt
        wind_speed = math.hypot(frame.wind_body_x, frame.wind_body_y)
        self.turn_noise += -.8 * self.turn_noise * dt + .3 * math.sqrt(dt) * self.rng.normal()
        if wind_speed < 1e-5:
            self._phase("NO_WIND")
            difference = (frame.odor_left - frame.odor_right) / max(.1, odor)
            error, speed = float(np.clip(2 * difference + self.turn_noise, -1, 1)), c["cast_speed"]
        else:
            upwind = math.atan2(-frame.wind_body_y, -frame.wind_body_x)
            if self.in_odor or (self.phase == "SURGE" and self.since_odor < c["lost_after_s"]):
                self._phase("SURGE")
                error, speed = upwind, c["surge_speed"]
            else:
                self._phase("CAST")
                duration = min(c["cast_max_s"], c["cast_first_s"] + self.cast_leg * c["cast_expansion_s"])
                if self.leg_elapsed >= duration:
                    self.cast_sign *= -1
                    self.cast_leg += 1
                    self.leg_elapsed = 0.
                if self.phase_elapsed >= c["cast_timeout_s"]:
                    # Restart a bounded local crosswind sweep, with no source hint.
                    self.cast_leg, self.phase_elapsed = 0, 0.
                    self.cast_sign = int(self.rng.choice([-1, 1]))
                error = upwind + self.cast_sign * math.pi / 2
                error = math.atan2(math.sin(error), math.cos(error))
                # Count the sweep after aligning crosswind. Counting the slow
                # physical turn itself made direction switches pre-empt turns.
                if abs(error) < .45:
                    self.leg_elapsed += dt
                speed = c["cast_speed"]
        # During a failed feeding retry, approach the contact already under the
        # front tarsi. No odor estimate can override this short local approach.
        if max([frame.mouth_taste, *(v for k, v in frame.tarsal_taste.items() if k.upper() in ("LF", "RF"))]) > .15:
            front = {key.upper(): value for key, value in frame.tarsal_taste.items()}
            error = .5 * (front.get("LF", 0.) - front.get("RF", 0.))
            speed = .25
        self.heading_error = float(error)
        turn = np.clip(c["turn_gain"] * error, -.65, .65)
        speed *= max(.45, 1 - .2 * abs(error))
        self.last_engineered = np.clip([speed - turn, speed + turn], 0, 1.2)
        return np.clip((1 - c["neural_weight"]) * self.last_engineered + c["neural_weight"] * self.last_neural, 0, 1.2)

    def get_state(self):
        return {"version": 1, "config": dict(self.config), "rng": copy.deepcopy(self.rng.bit_generator.state),
                **{key: copy.deepcopy(getattr(self, key)) for key in (
                    "phase", "cast_sign", "cast_leg", "leg_elapsed", "phase_elapsed", "since_odor",
                    "in_odor", "turn_noise", "last_engineered", "last_neural", "heading_error", "transitions")}}

    def set_state(self, saved):
        if saved["version"] != 1 or saved["config"] != self.config:
            raise ValueError("Incompatible search checkpoint")
        self.rng.bit_generator.state = copy.deepcopy(saved["rng"])
        for key in saved.keys() - {"version", "config", "rng"}:
            setattr(self, key, copy.deepcopy(saved[key]))
