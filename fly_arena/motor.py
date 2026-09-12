"""Single actuator owner, engineered motor primitives and scratch-data IK."""
from __future__ import annotations

import copy
import numpy as np
import mujoco as mj
from flygym.compose import ActuatorType
from flygym_demo.complex_terrain import HybridControllerObservation


class MotorArbiter:
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
        self.joint_ids = np.array([model.joint(f"fly/{d.name}").id for d in self.order])
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

    def _reset_scratch(self):
        """A deterministic kinematic workspace, never the live integration data."""
        model = self.arena.sim.mj_model
        mj.mj_resetDataKeyframe(model, self.scratch, model.key("neutral").id)
        self.scratch.qpos[self.qpos_ids[self.leg_indices]] = self.default
        self.scratch.qpos[self.qpos_ids[self.mouth_indices]] = 0.
        mj.mj_forward(model, self.scratch)

    def _solve_site(self, leg, target, iterations=90):
        model, data = self.arena.sim.mj_model, self.scratch
        indices = np.array([i for i, d in enumerate(self.order) if d.child.pos == leg])
        qids, vids = self.qpos_ids[indices], self.qvel_ids[indices]
        site = self.arena.site_ids[f"{leg}_tip"]
        jacobian = np.zeros((3, model.nv))
        reference = data.qpos[qids].copy()
        for _ in range(iterations):
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            error = np.asarray(target) - data.site_xpos[site]
            if np.linalg.norm(error) < .0008:
                break
            mj.mj_jacSite(model, data, jacobian, None, site)
            jac = jacobian[:, vids]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(3) * .002, error)
            delta += .002 * (reference - data.qpos[qids])
            data.qpos[qids] += np.clip(delta, -.12, .12)
            data.qpos[qids] = np.clip(data.qpos[qids], -3.4, 3.4)
        mj.mj_kinematics(model, data)
        return float(np.linalg.norm(np.asarray(target) - data.site_xpos[site]))

    def _prepare_primitives(self):
        """Manual task-space curves, solved once, replayed through position drives."""
        self._reset_scratch()
        for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
            site = self.arena.site_ids[f"{leg}_tip"]
            target = self.scratch.site_xpos[site].copy() + [0., 0., .16]
            self.ik_errors[f"crouch_{leg}"] = self._solve_site(leg, target)
        self.crouch = self.scratch.qpos[self.qpos_ids[self.leg_indices]].copy()
        for name in ("GROOM_FRONT", "GROOM_HEAD"):
            self._reset_scratch()
            thorax_position = self.scratch.xpos[self.arena.thorax_id].copy()
            thorax_rotation = self.scratch.xmat[self.arena.thorax_id].reshape(3, 3).copy()
            table = []
            errors = []
            for phase in np.linspace(0., 2. * np.pi, 32, endpoint=False):
                for leg, sign in (("lf", 1.), ("rf", -1.)):
                    if name == "GROOM_FRONT":
                        local = [.55 + sign * .025 * np.sin(phase), sign * .036,
                                 -.36 + sign * .045 * np.cos(phase)]
                    else:
                        local = [.43 + .09 * np.sin(phase), sign * .105,
                                 .02 + .13 * np.cos(phase)]
                    target = thorax_position + thorax_rotation @ np.array(local)
                    errors.append(self._solve_site(leg, target))
                table.append(self.scratch.qpos[self.qpos_ids[self.leg_indices]].copy())
            self._groom_tables[name] = np.asarray(table)
            self.ik_errors[name] = max(errors)

    @property
    def ingestion_enabled(self):
        return self.owner == "FEED" and self.elapsed >= .6 and not self.motor_off and self.arena.upright > .85

    def select(self, decision, *, motor_off=False):
        action = decision if isinstance(decision, str) else getattr(decision, "action", "IDLE")
        action = getattr(action, "value", action)
        action = str(action).upper()
        if action not in {"WALK", "IDLE", "FEED", "GROOM_FRONT", "GROOM_HEAD", "SLEEP_ENTRY", "SLEEP", "WAKE", "EXHAUSTED", "ESCAPE"}:
            raise ValueError(f"Unsupported motor action: {action}")
        self.motor_off = bool(motor_off)
        if self.motor_off:
            action = "IDLE"
        if self.arena.upright < .7:
            action = "IDLE"
        if action != self.owner:
            self.owner, self.elapsed = action, 0.
            self.escape_turn = float(getattr(decision, "escape_turn", 0.0))
            if action == "ESCAPE":
                self.escape_target_heading = self.arena.heading + self.escape_turn * 2.4
                self.escape_turn_done = False
                for name in ("retraction_correction", "stumbling_correction", "retraction_persistence_counter"):
                    getattr(self.arena.controller, name)[:] = 0.
                self.arena.controller.cpg_network.curr_magnitudes[:] = 0.
        if self.owner == "FEED":
            self._fit_mouth_to_local_floor()

    def _fit_mouth_to_local_floor(self):
        """One-axis analytic IK from own head pose to the known local floor plane.

        No food position/availability is read. The flat arena's feeding surface
        is 0.01 mm above its floor; physical overlap still decides ingestion.
        Fixing the haustellum target leaves the rostrum free to compensate for
        stance-dependent thorax height after walking or grooming.
        """
        data = self.arena.sim.mj_data
        head = self.arena.site_ids["head_frame"]
        rotation = data.site_xmat[head].reshape(3, 3)
        origin_z = data.site_xpos[head, 2] + rotation[2] @ np.array([.43, 0., -.274])
        if getattr(self, "adaptive_proboscis", False):
            # Search profile: preserve an anterior mouth endpoint across low
            # stances by using both joints. The target is in the head frame;
            # neither a food position nor its presence is available here.
            solutions = []
            for haustellum in np.linspace(.2, 3.1, 61):
                cosine, sine = np.cos(haustellum), np.sin(haustellum)
                vx = -.371 + .30 * cosine - .18 * sine
                vz = -.0196 - .30 * sine - .18 * cosine
                a = rotation[2, 0] * vx + rotation[2, 2] * vz
                b = rotation[2, 0] * vz - rotation[2, 2] * vx
                magnitude = np.hypot(a, b)
                ratio = (.01 - origin_z) / max(magnitude, 1e-12)
                if abs(ratio) > 1:
                    continue
                phase = np.arctan2(b, a)
                offset = np.arccos(ratio)
                for sign in (-1, 1):
                    q = np.arctan2(np.sin(phase + sign * offset), np.cos(phase + sign * offset))
                    if -1.78 <= q <= .28:
                        forward = .43 + np.cos(q) * vx + np.sin(q) * vz
                        score = (forward - .85)**2 + .005 * (q + 1.52)**2
                        solutions.append((score, q, haustellum))
            if solutions:
                _, rostrum, haustellum = min(solutions)
                self.feed_mouth_targets[:] = [rostrum, haustellum]
                return
        haustellum = 2.80
        cosine, sine = np.cos(haustellum), np.sin(haustellum)
        vector = np.array([-.371 + .30 * cosine - .18 * sine,
                           0., -.0196 - .30 * sine - .18 * cosine])
        a = rotation[2, 0] * vector[0] + rotation[2, 2] * vector[2]
        b = rotation[2, 0] * vector[2] - rotation[2, 2] * vector[0]
        magnitude = np.hypot(a, b)
        phase = np.arctan2(b, a)
        offset = np.arccos(np.clip((.01 - origin_z) / max(magnitude, 1e-12), -1., 1.))
        candidates = [phase + sign * offset + turns * 2. * np.pi
                      for sign in (-1., 1.) for turns in (-1., 0., 1.)]
        candidates = [q for q in candidates if -1.8 <= q <= .3]
        if candidates:
            rostrum = min(candidates, key=lambda q: abs(q + 1.52))
        else:
            # Unreachable floor: choose the closest admissible endpoint, never
            # move the body or enlarge the contact region to claim a meal.
            candidates = [-1.8, .3, np.clip(phase + np.pi, -1.8, .3)]
            rostrum = min(candidates, key=lambda q: abs(origin_z + a * np.cos(q) + b * np.sin(q) - .01))
        self.feed_mouth_targets[:] = [rostrum, haustellum]

    def update(self, gait):
        arena = self.arena
        desired = np.zeros_like(self.targets)
        desired[self.leg_indices] = self.default
        self.adhesion[:] = 1.
        if self.owner == "ESCAPE" and self.elapsed < .1:
            self.primitive_phase = "escape_support"
        elif self.owner in ("WALK", "ESCAPE"):
            if self.owner == "ESCAPE":
                error = np.arctan2(np.sin(self.escape_target_heading - arena.heading),
                                   np.cos(self.escape_target_heading - arena.heading))
                if abs(error) < .25:
                    self.escape_turn_done = True
                speed = .8 if self.escape_turn_done else .25
                turn = 0. if self.escape_turn_done else np.clip(.8 * error, -.6, .6)
                gait = np.clip([speed - turn, speed + turn], 0, 1.2)
            obs = HybridControllerObservation.from_sim(arena.sim, arena.fly.name)
            action = arena.controller.step(gait, obs)
            desired[self.leg_indices] = action.joint_angles
            self.adhesion[:] = action.adhesion_onoff
            self.primitive_phase = "walk" if self.owner == "WALK" else "escape_run" if self.escape_turn_done else "escape_turn"
        elif self.owner == "FEED":
            desired[self.leg_indices] = self.crouch
            if self.elapsed >= .25:
                desired[self.mouth_indices] = self.feed_mouth_targets
            self.primitive_phase = "settle" if self.elapsed < .25 else "extend" if self.elapsed < .6 else "contact"
        elif self.owner in ("GROOM_FRONT", "GROOM_HEAD"):
            self.primitive_phase = "settle" if self.elapsed < .25 else "stroke"
            if self.elapsed >= .25 and self.owner in self._groom_tables:
                table = self._groom_tables[self.owner]
                phase = ((self.elapsed - .25) * 3.) % 1. * len(table)
                left = int(phase)
                desired[self.leg_indices] = table[left] * (1. - phase + left) + table[(left + 1) % len(table)] * (phase - left)
                self.adhesion[[0, 3]] = 0.
        else:
            self.primitive_phase = "hold"
        dt = arena.sim.timestep * arena.controller_stride
        # Rate-limited ownership handoff: no joint target jumps or competing writers.
        if self.owner in ("WALK", "ESCAPE") and self.elapsed > .15:
            self.targets[:] = desired
        else:
            self.targets += np.clip(desired - self.targets, -12. * dt, 12. * dt)
        arena.sim.set_actuator_inputs(arena.fly.name, ActuatorType.POSITION, self.targets)
        arena.sim.set_leg_adhesion_states(arena.fly.name, self.adhesion)

    def get_state(self):
        controller = self.arena.controller
        cpg = controller.cpg_network
        rng = cpg.random_state.get_state()
        return {"owner": self.owner, "elapsed": self.elapsed, "motor_off": self.motor_off,
                "adaptive_proboscis": self.adaptive_proboscis,
                "escape_turn": self.escape_turn,
                "escape_target_heading": self.escape_target_heading,
                "escape_turn_done": self.escape_turn_done,
                "primitive_phase": self.primitive_phase, "targets": self.targets.copy(), "adhesion": self.adhesion.copy(),
                "feed_mouth_targets": self.feed_mouth_targets.copy(),
                "cpg": {k: getattr(cpg, k).copy() for k in ("curr_phases", "curr_magnitudes", "intrinsic_freqs", "intrinsic_amps")},
                "rng": [rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])],
                "controller_last_info": copy.deepcopy(controller.last_info),
                "corrections": {k: getattr(controller, k).copy() for k in
                                ("retraction_correction", "stumbling_correction", "retraction_persistence_counter")}}

    def set_state(self, state):
        for key in ("owner", "elapsed", "motor_off", "primitive_phase", "adaptive_proboscis", "escape_turn", "escape_target_heading", "escape_turn_done"):
            setattr(self, key, state[key])
        for key in ("targets", "adhesion", "feed_mouth_targets"):
            setattr(self, key, np.asarray(state[key]).copy())
        cpg = self.arena.controller.cpg_network
        for key, value in state["cpg"].items():
            setattr(cpg, key, np.asarray(value).copy())
        rng = state["rng"]
        cpg.random_state.set_state((rng[0], np.asarray(rng[1], np.uint32), rng[2], rng[3], rng[4]))
        for key, value in state["corrections"].items():
            getattr(self.arena.controller, key)[:] = value
        self.arena.controller.last_info = copy.deepcopy(state["controller_last_info"])
