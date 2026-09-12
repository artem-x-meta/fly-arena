"""Engineered resource defence using bounded local perception and own needs.

This is an explicit policy, not a neural or biological aggression model. No
world, food coordinates, rival energy, winner identity or external body force
is available here. PUSH is a forward walking request into physical contact.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from .resource_motivation import ResourceMotivation
from .combat_motor import FENCE_DURATION_S


@dataclass(frozen=True)
class SocialCommand:
    state: str
    gait: tuple[float, float]
    reason: str


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class SocialController:
    def __init__(self, *, seed=1):
        self.rng = np.random.default_rng(seed)
        self.motivation = ResourceMotivation()
        self.fence_cooldown_s = 0.
        self.assessed_intake_total = 0.
        self.state = "NONE"
        self.elapsed_s = 0.
        self.food_memory_s = 0.
        self.fatigue = 0.
        self.cooldown_s = 0.
        self.contact_s = 0.
        self.retreat_heading = 0.
        self.retreat_run_s = 0.
        self.retreat_position = np.zeros(2)
        self.max_retreat_mm = 0.
        self.resource_position = None
        self.resource_heading = 0.
        self.last_reason = "initial"
        self.counters = dict(encounters=0, pushes=0, fences=0, retreats=0, returns=0, recoveries=0)
        self.durations = {}
        # Draw from the same distribution for each identity, never assign a
        # winner. Independent per-bout persistence breaks exact symmetry.
        self.persistence = float(self.rng.uniform(.85, 1.25))

    def _enter(self, state, reason):
        if self.state != state:
            self.state, self.elapsed_s = state, 0.
            key = {"ORIENT": "encounters", "PUSH": "pushes", "FENCE": "fences", "RETREAT": "retreats",
                   "RETURN": "returns", "RECOVER": "recoveries"}.get(state)
            if key:
                self.counters[key] += 1
        self.last_reason = reason

    def _steer(self, bearing, speed):
        # Same turn convention as SurgeCastNavigator / MotorArbiter.
        turn = float(np.clip(.85 * bearing, -.65, .65))
        return tuple(float(x) for x in np.clip([speed - turn, speed + turn], 0., 1.2))

    def _retreat(self, sense, reason):
        # Back away along our own axis; a moving opponent must not turn a
        # withdrawal into another pursuit around the patch.
        self.retreat_heading = sense.own_heading
        self.retreat_position = np.asarray(getattr(sense, "own_position_mm", (0., 0., 0.)))[:2].copy()
        self.retreat_run_s = 0.
        self._enter("RETREAT", reason)

    def step(self, sense, local, organism, *, dt=.01, enabled=True, base_action="WALK"):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Social interval must be positive and finite")
        self.elapsed_s += dt
        wants_conflict = self.motivation.step(sense, local, organism, dt=dt, base_action=base_action)
        self.fence_cooldown_s = max(0., self.fence_cooldown_s - dt)
        self.food_memory_s = max(0., self.food_memory_s - dt)
        self.cooldown_s = max(0., self.cooldown_s - dt)
        taste = max([local.mouth_taste, *local.tarsal_taste.values()])
        if taste >= .15:
            self.food_memory_s = 5.
            # Remember our own pose at a real taste event. The world never
            # supplies a food target. This is short-term engineered odometry.
            if self.state in {"NONE", "RECOVER"}:
                self.resource_position = np.asarray(getattr(sense, "own_position_mm", (0., 0., 0.)))[:2].copy()
                self.resource_heading = sense.own_heading
        if self.state in {"PUSH", "ORIENT", "FENCE"}:
            self.fatigue += dt * (.20 + (.70 if sense.body_contact else 0.))
            self.contact_s = self.contact_s + dt if sense.body_contact else max(0., self.contact_s - dt)
        else:
            self.fatigue = max(0., self.fatigue - dt * .24)
            self.contact_s = max(0., self.contact_s - dt)

        # Disabling the channel removes even a remembered engagement. Ordinary
        # sleep/escape/exhaustion and loss of upright posture retain priority.
        protected = base_action in {"SLEEP", "SLEEP_ENTRY", "WAKE", "EXHAUSTED", "ESCAPE"}
        if not enabled or organism.exhausted or protected or local.upright < .8:
            if local.upright < .8:
                self.cooldown_s = max(self.cooldown_s, 1.)
            self._enter("NONE", "disabled_or_physical_guard")
            return None

        if local.ground_support < 3:
            self.last_reason = "ground_support_guard"
            return None

        # Briefly retry a just-interrupted meal at the place it was tasted.
        # Otherwise the search gait carried the animals several body lengths
        # away during the access-assessment delay. This pause is not aggression
        # and cannot repeat until another real meal has been swallowed.
        if self.state in {"NONE", "ASSESS"}:
            own = np.asarray(getattr(sense, "own_position_mm", (0., 0., 0.)))[:2]
            near_meal = self.resource_position is not None and np.linalg.norm(own - self.resource_position) <= 1.5
            if (self.state == "NONE" and base_action == "WALK" and sense.detected and near_meal
                    and organism.hunger >= .25 and
                    organism.state.ingested_total > self.assessed_intake_total + .01):
                self.assessed_intake_total = organism.state.ingested_total
                self._enter("ASSESS", "brief_retry_of_interrupted_local_meal")
            if self.state == "ASSESS":
                if (base_action == "FEED" or not sense.detected or not near_meal
                        or organism.hunger < .25 or self.elapsed_s >= 1.5 or wants_conflict):
                    self._enter("NONE", "local_access_assessment_complete")
                else:
                    return self._result((0., 0.), dt)

        if self.state == "FENCE":
            if not sense.detected or not wants_conflict or local.upright < .85:
                self.fence_cooldown_s = 1.
                self._enter("NONE", "fencing_stopped_by_local_guard")
                return None
            if self.elapsed_s < FENCE_DURATION_S:
                return self._result((0., 0.), dt)
            self.fence_cooldown_s = 1.
            self._enter("PUSH", "foreleg_stroke_complete")

        if self.state == "RETREAT":
            own = np.asarray(getattr(sense, "own_position_mm", (0., 0., 0.)))[:2]
            axis = np.array([math.cos(self.retreat_heading), math.sin(self.retreat_heading)])
            displacement = float(-(own - self.retreat_position) @ axis)
            self.max_retreat_mm = max(self.max_retreat_mm, displacement)
            if displacement >= .5 or self.elapsed_s >= 2.:
                self._enter("RETURN", "withdrawal_complete_return_to_last_taste")
            else:
                # Equal reverse drives have been measured to produce a stable
                # backstep. Opposed signs were NOT a reliable in-place turn.
                return self._result((-.25, -.25), dt)

        if self.state == "RETURN":
            own = np.asarray(getattr(sense, "own_position_mm", (0., 0., 0.)))[:2]
            delta = self.resource_position - own if self.resource_position is not None else np.zeros(2)
            distance = float(np.linalg.norm(delta))
            target = math.atan2(delta[1], delta[0]) if distance > .35 else self.resource_heading
            error = wrap(target - sense.own_heading)
            front_taste = max([local.mouth_taste, *(value for key, value in local.tarsal_taste.items()
                                                  if key.upper() in {"LF", "RF"})])
            found_food = front_taste >= .15
            if found_food or (distance <= .35 and abs(error) < .35) or self.elapsed_s >= 4.:
                self.cooldown_s = float(self.rng.uniform(2.5, 4.))
                self._enter("RECOVER", "local_food_reacquired" if found_food else
                            "return_complete" if distance <= .35 else "return_timeout")
            else:
                speed = .20 if distance > .35 and abs(error) < .35 else 0.
                return self._result(self._steer(error, speed), dt)

        if self.state == "RECOVER":
            # Let the original local search/feeding policy resume immediately;
            # only social engagement waits for the refractory interval.
            if self.cooldown_s > 0:
                self.durations["RECOVER"] = self.durations.get("RECOVER", 0.) + dt
                return None
            self._enter("NONE", "recovered")

        motivated = wants_conflict
        if self.state in {"ORIENT", "PUSH"}:
            if not sense.detected or not motivated:
                self.cooldown_s = .5
                self._enter("NONE", "rival_lost_or_resource_need_receded")
                return None
            if self.fatigue >= self.persistence or self.elapsed_s > 3.:
                self._retreat(sense, "own_effort_limit")
                return self._result((-.25, -.25), dt)
        elif motivated and sense.detected and self.cooldown_s <= 0:
            self.persistence = float(self.rng.uniform(.85, 1.25))
            self._enter("ORIENT", "own_feeding_access_failed_near_rival")
        else:
            return None

        # Never continue a blind charge after the rival leaves the local range.
        bearing = sense.bearing
        if self.state == "ORIENT" and abs(bearing) < .3:
            self._enter("PUSH", "facing_local_rival")
        elif self.state == "PUSH" and abs(bearing) > .65:
            self._enter("ORIENT", "rival_moved_sideways")
        if (self.state == "PUSH" and self.fence_cooldown_s <= 0 and
                sense.distance_mm is not None and sense.distance_mm <= 2.3 and
                abs(bearing) < .35 and local.ground_support >= 4 and local.upright > .90):
            self._enter("FENCE", "local_foreleg_fencing_bout")
            return self._result((0., 0.), dt)
        # Small stride amplitude maintains a supported shove at this body
        # scale. Faster walking slid both spherical proxies past each other.
        speed = .1125 if self.state == "PUSH" and abs(bearing) < .3 else .055
        return self._result(self._steer(bearing, speed), dt)

    def _result(self, gait, dt):
        self.durations[self.state] = self.durations.get(self.state, 0.) + dt
        return SocialCommand(self.state, gait, self.last_reason)

    def summary(self):
        return {"state": self.state, "fatigue": self.fatigue,
                "food_memory_s": self.food_memory_s,
                "max_retreat_mm": self.max_retreat_mm,
                "resource_motivation": self.motivation.summary(),
                "counters": dict(self.counters),
                "durations_s": {k: round(v, 4) for k, v in self.durations.items()}}
