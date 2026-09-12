"""The existing motor primitives with each animal's own joint addresses.

The single-fly constructor resolves literal ``fly/`` joint names. Its
initialization is mirrored here to resolve the requested namespace *before*
building IK tables; all selection, stepping and checkpoint methods are inherited.
"""
from __future__ import annotations

import mujoco as mj
import numpy as np

from fly_arena.motor import MotorArbiter


class NamespacedMotorArbiter(MotorArbiter):
    def __init__(self, arena):
        self.arena = arena
        self.order = arena.fly.get_actuated_jointdofs_order("position")
        self.leg_indices = np.array([i for i, d in enumerate(self.order) if d.child.is_leg()])
        self.mouth_indices = np.array([i for i, d in enumerate(self.order) if not d.child.is_leg()])
        self.default = arena.steps.default_pose_by_dof_order(arena.dof_order)
        self.targets = np.zeros(len(self.order))
        self.targets[self.leg_indices] = self.default
        self.adhesion = np.ones(6)
        self.owner = "IDLE"
        self.elapsed = 0.
        self.motor_off = False
        self.primitive_phase = "hold"
        self.feed_mouth_targets = np.array([-1.52, 2.80])
        model = arena.sim.mj_model
        self.joint_ids = np.array([model.joint(f"{arena.fly.name}/{d.name}").id for d in self.order])
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = model.jnt_dofadr[self.joint_ids]
        self.scratch = mj.MjData(model)
        self.crouch = self.default.copy()
        self._groom_tables = {}
        self.adaptive_proboscis = False
        self.escape_turn = 0.0
        self.escape_target_heading = 0.0
        self.escape_turn_done = False
        self.ik_errors = {}
        self._prepare_primitives()
