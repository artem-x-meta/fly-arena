"""A bounded foreleg fencing stroke, using only existing joint actuators.

This is an engineered foreleg gesture inspired by fencing in male ethograms.
It is not the hindleg-supported body rise and downward strike of a lunge.
All inverse kinematics run in scratch data; neither positions nor forces are
written into the live integration state. Ordinary motor actions are inherited.
"""
from __future__ import annotations

import numpy as np
from flygym.compose import ActuatorType

from .namespaced_motor import NamespacedMotorArbiter


FENCE_DURATION_S = .60


class CombatMotorArbiter(NamespacedMotorArbiter):
    """One additional actuator owner, with the original life primitives intact."""

    fence_duration_s = FENCE_DURATION_S

    def __init__(self, arena):
        super().__init__(arena)
        self._prepare_fence()

    def _prepare_fence(self):
        self._reset_scratch()
        origin = self.scratch.xpos[self.arena.thorax_id].copy()
        rotation = self.scratch.xmat[self.arena.thorax_id].reshape(3, 3).copy()
        tips = {leg: rotation.T @ (self.scratch.site_xpos[
            self.arena.site_ids[f"{leg}_tip"]] - origin) for leg in ("lf", "rf")}
        # Timed raise, anterior/inward reach, short downward tap and recovery.
        # Millimetres in the animal's own thorax frame; no opponent coordinates.
        knots = np.array([0., .12, .24, .34, .41, .54, FENCE_DURATION_S])
        offsets = np.array([[0., 0., 0.], [0., 0., 0.],
                            [.03, .06, .30], [.50, .30, .30],
                            [.50, .30, .15], [0., 0., 0.], [0., 0., 0.]])
        self._fence_times = np.linspace(0., FENCE_DURATION_S, 61)
        table, errors = [], []
        for time in self._fence_times:
            offset = np.array([np.interp(time, knots, offsets[:, axis]) for axis in range(3)])
            for leg, sign in (("lf", 1.), ("rf", -1.)):
                local = tips[leg] + offset * [1., -sign, 1.]
                errors.append(self._solve_site(leg, origin + rotation @ local))
            table.append(self.scratch.qpos[self.qpos_ids[self.leg_indices]].copy())
        self._fence_table = np.asarray(table)
        # End with the exact default pose, avoiding accumulated IK regularisation.
        self._fence_table[self._fence_times <= .12] = self.default
        self._fence_table[self._fence_times >= .54] = self.default
        self.ik_errors["FENCE"] = max(errors)

    def can_fence(self):
        """Entry gate; four grounded legs and an upright body before lifting."""
        return not self.motor_off and self.arena.upright > .90 and self.arena.ground_support >= 4

    def select(self, decision, *, motor_off=False):
        action = decision if isinstance(decision, str) else getattr(decision, "action", "IDLE")
        action = str(getattr(action, "value", action)).upper()
        if action != "FENCE":
            return super().select(decision, motor_off=motor_off)
        self.motor_off = bool(motor_off)
        if self.motor_off or self.arena.upright < .85 or self.arena.ground_support < 3:
            return super().select("IDLE", motor_off=motor_off)
        if self.owner != "FENCE":
            if not self.can_fence():
                return super().select("IDLE", motor_off=motor_off)
            super().select("WALK", motor_off=motor_off)
            self.owner, self.elapsed = "FENCE", 0.

    def update(self, gait):
        if self.owner != "FENCE":
            return super().update(gait)
        # A lost stance ends the gesture through the same owner, without a
        # teleport, an external wrench, or a second writer to the actuators.
        if self.motor_off or self.arena.upright < .85 or self.arena.ground_support < 3:
            super().select("IDLE", motor_off=self.motor_off)
            return super().update(gait)
        time = min(self.elapsed, FENCE_DURATION_S)
        phase = min(time / FENCE_DURATION_S * (len(self._fence_table) - 1),
                    len(self._fence_table) - 1)
        left = min(int(phase), len(self._fence_table) - 2)
        weight = phase - left
        desired = np.zeros_like(self.targets)
        desired[self.leg_indices] = ((1. - weight) * self._fence_table[left]
                                    + weight * self._fence_table[left + 1])
        self.adhesion[:] = 1.
        if .12 <= time < .54:
            self.adhesion[[0, 3]] = 0.
        self.primitive_phase = ("fence_settle" if time < .12 else
                                "fence_raise" if time < .24 else
                                "fence_reach" if time < .34 else
                                "fence_tap" if time < .41 else
                                "fence_recover" if time < .54 else "fence_hold")
        dt = self.arena.sim.timestep * self.arena.controller_stride
        self.targets += np.clip(desired - self.targets, -12. * dt, 12. * dt)
        self.arena.sim.set_actuator_inputs(self.arena.fly.name, ActuatorType.POSITION, self.targets)
        self.arena.sim.set_leg_adhesion_states(self.arena.fly.name, self.adhesion)
