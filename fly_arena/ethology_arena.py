"""Actuated proboscis and contact instrumentation for the explicit hybrid model.

Lengths are mm. Contact patches are small engineering substitutes for soft
cuticle and fluid surfaces; they are not reconstructions of labellar anatomy.
"""
from __future__ import annotations

import copy
import numpy as np
import mujoco as mj
from flygym.anatomy import (ActuatedDOFPreset, AnatomicalJoint, AxesSet, AxisOrder,
                           BodySegment, JointPreset, Skeleton, PASSIVE_TARSAL_LINKS)
from flygym.compose import NeuroMechFly, KinematicPosePreset, ActuatorType

from .arena import Arena
from .sensors import PhysicalEvents


def make_ethology_fly(name="fly"):
    neutral = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    anatomical = JointPreset.LEGS_ONLY.to_joint_list()
    for parent, child, axes in (("c_thorax", "c_head", []),
                               ("c_head", "c_rostrum", ["pitch"]),
                               ("c_rostrum", "c_haustellum", ["pitch"])):
        anatomical.append(AnatomicalJoint(BodySegment(parent), BodySegment(child), AxesSet(axes)))
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=anatomical)
    fly = NeuroMechFly(name=name)
    joints = fly.add_joints(skeleton, neutral_pose=neutral, stiffness=.05, damping=.06)
    for dof, joint in joints.items():
        if dof.child.link in PASSIVE_TARSAL_LINKS:
            joint.stiffness[0], joint.damping[0] = 7.5, .01
        elif not dof.child.is_leg():
            joint.stiffness[0], joint.damping[0] = .01, .005
            joint.limited = True
            joint.range = [-1.8, .3] if dof.child.link == "rostrum" else [-.3, 3.3]
    legs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(legs, ActuatorType.POSITION, neutral_input=neutral,
                      kp=45., forcerange=(-65., 65.))
    mouth = [d for d in joints if not d.child.is_leg()]
    fly.add_actuators(mouth, ActuatorType.POSITION, kp=2., forcerange=(-3., 3.))
    fly.add_leg_adhesion(gain=40.)
    fly.colorize()
    fly.bodyseg_to_mjcfbody[BodySegment("c_head")].add_site(
        name="head_frame", pos=[0, 0, 0], size=[.01]*3, rgba=[0, 0, 0, 0])
    fly.bodyseg_to_mjcfbody[BodySegment("c_haustellum")].add_site(
        name="mouth", pos=[.30, 0, -.18], size=[.025, .025, .025], rgba=[1., .3, .1, 1.])
    for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
        fly.bodyseg_to_mjcfbody[BodySegment(f"{leg}_tarsus1")].add_site(
            name=f"{leg}_tip", pos=[0, 0, -.20], size=[.012]*3, rgba=[0, 0, 0, 0])
        fly.bodyseg_to_mjcfbody[BodySegment(f"{leg}_tarsus5")].add_site(
            name=f"{leg}_taste", pos=[0, 0, -.075], size=[.01]*3, rgba=[0, 0, 0, 0])
    for side in ("l", "r"):
        fly.bodyseg_to_mjcfbody[BodySegment(f"{side}_funiculus")].add_site(
            name=f"{side}_odor", pos=[0, 0, -.10], size=[.01]*3, rgba=[0, 0, 0, 0])
    # Only these added patches have self-contact pairs, never all body meshes.
    for leg in ("lf", "rf"):
        fly.bodyseg_to_mjcfbody[BodySegment(f"{leg}_tarsus1")].add_geom(
            name=f"{leg}_rub", type=mj.mjtGeom.mjGEOM_SPHERE,
            pos=[0, 0, -.20], size=[.045, 0, 0], mass=1e-9,
            contype=0, conaffinity=0, rgba=[.8, .5, .2, .15], group=1)
    for region, segment, pos, radius in (
        ("head", "c_head", [.30, 0, .10], .12),
        ("antenna_left", "l_funiculus", [0, 0, -.10], .06),
        ("antenna_right", "r_funiculus", [0, 0, -.10], .06)):
        body = fly.bodyseg_to_mjcfbody[BodySegment(segment)]
        body.add_geom(name=f"clean_{region}", type=mj.mjtGeom.mjGEOM_SPHERE,
                      pos=pos, size=[radius, 0, 0], mass=1e-9,
                      contype=0, conaffinity=0, rgba=[.7, .5, .2, .12], group=1)
    return fly


class EthologyArena(Arena):
    """Extended body; old Arena and its two modes retain their factory defaults."""
    mouth_contact_thickness = .04

    def __init__(self, seed=1, food_patches=None, warmup=True, *, eye_size=96,
                 mouth_contact_enabled=True, grooming_contacts_enabled=True, looming_objects=None):
        self.food_patches = [dict(p) for p in (food_patches or [])]
        self.looming_objects = copy.deepcopy(looming_objects or [])
        from .escape import validate_looming
        validate_looming(self.looming_objects)
        self.mouth_contact_enabled = bool(mouth_contact_enabled)
        self._grooming_contacts_enabled = bool(grooming_contacts_enabled)
        super().__init__(seed=seed, eye_size=eye_size, warmup=warmup)
        self.looming_mocap_ids = [int(self.sim.mj_model.body(f"looming_body_{i}").mocapid[0])
                                  for i in range(len(self.looming_objects))]
        self.site_ids = {name: self.sim.mj_model.site(f"fly/{name}").id for name in
                         ("mouth", "head_frame", "l_odor", "r_odor", *[f"{l}_{s}" for l in
                          ("lf", "lm", "lh", "rf", "rm", "rh") for s in ("tip", "taste")])}
        self.region_geom_ids = {region: self.sim.mj_model.geom(f"fly/clean_{region}").id
                                for region in ("head", "antenna_left", "antenna_right")}
        self.rub_geom_ids = {leg: self.sim.mj_model.geom(f"fly/{leg}_rub").id for leg in ("lf", "rf")}
        self._groom_pair_ids = np.array([i for i in range(self.sim.mj_model.npair)
                                        if self.sim.mj_model.pair(i).name.startswith("groom_")])
        self._contact_jacobian = np.zeros((3, self.sim.mj_model.nv))
        self._contact_wrench = np.zeros(6)
        self.ground_geom_ids = set(map(int, self.sim._internal_ground_geom_ids))
        self.leg_by_geom_id = {}
        for segment, geoms in self.fly.bodyseg_to_mjcfgeom.items():
            if segment.is_leg():
                for geom in geoms:
                    self.leg_by_geom_id[self.sim.mj_model.geom(geom.name).id] = segment.pos
        self.max_groom_contact_force = 0.
        self.contact_pair_labels = {}
        self.contact_pair_labels[frozenset(self.rub_geom_ids.values())] = "front_left|front_right"
        for leg, gid in self.rub_geom_ids.items():
            for region, rid in self.region_geom_ids.items():
                self.contact_pair_labels[frozenset((gid, rid))] = f"{region}|front_{'left' if leg == 'lf' else 'right'}"
        from .motor import MotorArbiter
        self.motor = MotorArbiter(self)

    def _make_fly(self):
        return make_ethology_fly()

    def _configure_world(self):
        for i, food in enumerate(self.food_patches):
            food.setdefault("id", str(i))
            self.world.mjcf_root.worldbody.add_geom(
                name=f"food_{i}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                pos=[food.get("x", 0.), food.get("y", 0.), food.get("surface_z", .01) - .005],
                size=[food.get("radius", 1.), .005, 0], rgba=[.8, .36, .12, 1.],
                contype=0, conaffinity=0)
        for i, obj in enumerate(self.looming_objects):
            body = self.world.mjcf_root.worldbody.add_body(name=f"looming_body_{i}", mocap=True, pos=[0, 0, -20])
            material = self.world.mjcf_root.add_material(name=f"looming_material_{i}", rgba=[.8, .03, .85, 1], emission=.5)
            geom = body.add_geom(name=f"looming_geom_{i}", type=mj.mjtGeom.mjGEOM_SPHERE,
                                 size=[obj["radius"], 0, 0], material=material.name,
                                 contype=0, conaffinity=0)
            self.world.ground_geoms.append(geom)

    def update_looming(self, time_s, *, refresh=False):
        if not self.looming_objects:
            return
        from .escape import looming_position
        for obj, mocap_id in zip(self.looming_objects, self.looming_mocap_ids):
            self.sim.mj_data.mocap_pos[mocap_id] = looming_position(obj, time_s)[0]
        if refresh:
            warmstart = self.sim.mj_data.qacc_warmstart.copy()
            mj.mj_forward(self.sim.mj_model, self.sim.mj_data)
            self.sim.mj_data.qacc_warmstart[:] = warmstart

    def _configure_contacts(self):
        pairs = [("lf_rub", "rf_rub")]
        pairs += [(f"{leg}_rub", f"clean_{region}") for leg in ("lf", "rf")
                  for region in ("head", "antenna_left", "antenna_right")]
        for a, b in pairs:
            self.world.mjcf_root.add_pair(name=f"groom_{a}_{b}",
                geomname1=f"fly/{a}", geomname2=f"fly/{b}",
                friction=(.4, .4, .002, .0001, .0001), solref=(.002, 1.),
                solimp=(.9, .95, .001, .5, 2.),
                margin=.001 if self.grooming_contacts_enabled else -1.)

    @property
    def grooming_contacts_enabled(self):
        return self._grooming_contacts_enabled

    @grooming_contacts_enabled.setter
    def grooming_contacts_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled != self._grooming_contacts_enabled:
            self._grooming_contacts_enabled = enabled
            # Every grooming geom is a sphere; the largest radius sum is .165 mm.
            # A -1 mm inclusion margin therefore excludes every such constraint.
            # Candidate contacts may remain in data, with exclude=1 and no force.
            self.sim.mj_model.pair_margin[self._groom_pair_ids] = .001 if enabled else -1.

    def _apply_leg_action(self, action):
        # Warmup uses the same leg targets as the old body, with a closed mouth.
        order = self.fly.get_actuated_jointdofs_order("position")
        targets = np.zeros(len(order))
        targets[[i for i, d in enumerate(order) if d.child.is_leg()]] = action.joint_angles
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, targets)
        self.sim.set_leg_adhesion_states(self.fly.name, action.adhesion_onoff)

    @property
    def mouth_position(self):
        return self.sim.mj_data.site_xpos[self.site_ids["mouth"]].copy()

    @property
    def tarsi_positions(self):
        return np.array([self.sim.mj_data.site_xpos[self.site_ids[f"{l}_taste"]]
                         for l in ("lf", "lm", "lh", "rf", "rm", "rh")])

    @property
    def tarsal_positions(self):
        return self.tarsi_positions

    @property
    def antenna_positions(self):
        return np.array([self.sim.mj_data.site_xpos[self.site_ids[f"{s}_odor"]] for s in ("l", "r")])

    @property
    def stable(self):
        return self.upright > .85 and self.movement < 2. and self.ground_support >= 3

    @property
    def movement(self):
        """Free-body linear speed in mm per physical second."""
        return float(np.linalg.norm(self.sim.mj_data.qvel[:3]))

    @property
    def ground_support(self):
        supported = set()
        for c in self.sim.mj_data.contact:
            if c.exclude or c.dist > .001:
                continue
            for leg_geom, ground_geom in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if int(ground_geom) in self.ground_geom_ids and int(leg_geom) in self.leg_by_geom_id:
                    supported.add(self.leg_by_geom_id[int(leg_geom)])
        return len(supported)

    def step(self, command, milliseconds=10):
        return self.step_behavior("WALK", command, milliseconds=milliseconds)

    def step_behavior(self, decision, gait, milliseconds=10, motor_off=False):
        gait = np.asarray(gait, float)
        if gait.shape != (2,) or not np.isfinite(gait).all():
            raise ValueError("Invalid gait command")
        if milliseconds <= 0 or not np.isfinite(milliseconds):
            raise ValueError("Physical interval must be positive and finite")
        self.control = np.clip(gait, 0, 1.2)
        self.motor.select(decision, motor_off=motor_off)
        events = PhysicalEvents()
        self.max_groom_contact_force = 0.
        for _ in range(round(milliseconds / 1000 / self.sim.timestep)):
            self.update_looming(self.physics_steps * self.sim.timestep)
            if self.physics_steps % self.controller_stride == 0:
                self.motor.update(self.control)
            self.sim.step()
            self.physics_steps += 1
            self.motor.elapsed += self.sim.timestep
            self._accumulate_events(events)
        if not np.isfinite(self.sim.mj_data.qpos).all():
            raise RuntimeError("Non-finite physics state")
        return events

    def _accumulate_events(self, events):
        dt = self.sim.timestep
        if self.mouth_contact_enabled:
            mouth = self.sim.mj_data.site_xpos[self.site_ids["mouth"]]
            for food in self.food_patches:
                if (abs(mouth[2] - food.get("surface_z", .01)) <= self.mouth_contact_thickness
                        and np.linalg.norm(mouth[:2] - [food.get("x", 0.), food.get("y", 0.)]) <= food.get("radius", 1.)):
                    key = str(food["id"])
                    events.mouth_contact_s_by_food[key] = events.mouth_contact_s_by_food.get(key, 0.) + dt
                    if self.motor.ingestion_enabled:
                        events.contact_s_by_food[key] = events.contact_s_by_food.get(key, 0.) + dt
                    # One physical mouth samples one mixture contributor per step.
                    # Stable patch ordering resolves overlapping food surfaces.
                    break
        if not self.grooming_contacts_enabled:
            return
        data, model = self.sim.mj_data, self.sim.mj_model
        by_pair = {}
        for index, contact in enumerate(data.contact):
            if contact.exclude or contact.dist > .001:
                continue
            label = self.contact_pair_labels.get(frozenset((int(contact.geom1), int(contact.geom2))))
            if label is None:
                continue
            velocities = []
            for gid in (contact.geom1, contact.geom2):
                bodyid = int(model.geom_bodyid[gid])
                jac = self._contact_jacobian
                jac.fill(0.)
                mj.mj_jac(model, data, jac, None, contact.pos, bodyid)
                velocities.append(jac @ data.qvel)
            relative = velocities[0] - velocities[1]
            normal = contact.frame[:3]
            sliding = float(np.linalg.norm(relative - np.dot(relative, normal) * normal)) * dt
            by_pair[label] = max(by_pair.get(label, 0.), sliding)
            wrench = self._contact_wrench
            wrench.fill(0.)
            mj.mj_contactForce(model, data, index, wrench)
            self.max_groom_contact_force = max(self.max_groom_contact_force, float(np.linalg.norm(wrench[:3])))
        for label, sliding in by_pair.items():
            events.grooming_sliding_by_pair[label] = events.grooming_sliding_by_pair.get(label, 0.) + sliding
            events.grooming_contact_s_by_pair[label] = events.grooming_contact_s_by_pair.get(label, 0.) + dt

    def get_state(self):
        kind = mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mj.mj_stateSize(self.sim.mj_model, kind))
        mj.mj_getState(self.sim.mj_model, self.sim.mj_data, state, kind)
        # Wall-clock profiler durations are not simulation state. Canonicalize
        # those 15 counters only; keep every dynamical/derived/constraint field.
        durations = self.sim.mj_data.timer.duration.copy()
        try:
            self.sim.mj_data.timer.duration[:] = 0.
            native = np.frombuffer(self.sim.mj_data.__getstate__(), dtype=np.uint8).copy()
        finally:
            self.sim.mj_data.timer.duration[:] = durations
        return {"integration": state, "physics_steps": self.physics_steps,
                # MuJoCo's native data serializer also retains derived poses,
                # constraint caches and contacts from the last substep. Keeping
                # them avoids a one-substep sensory shift on mid-walk resume.
                # This is opaque native data, stored as uint8, never Python pickle.
                "native_mj_data": native,
                "native_mujoco_version": mj.__version__,
                "control": self.control.copy(), "motor": self.motor.get_state(),
                "mouth_contact_enabled": self.mouth_contact_enabled,
                "grooming_contacts_enabled": self.grooming_contacts_enabled,
                "mouth_contact_thickness": self.mouth_contact_thickness,
                "pair_margin": self.sim.mj_model.pair_margin.copy(),
                "max_groom_contact_force": self.max_groom_contact_force,
                "food_patches": copy.deepcopy(self.food_patches)}

    def set_state(self, state):
        self.mouth_contact_enabled = bool(state["mouth_contact_enabled"])
        self.grooming_contacts_enabled = bool(state["grooming_contacts_enabled"])
        self.mouth_contact_thickness = float(state["mouth_contact_thickness"])
        self.sim.mj_model.pair_margin[:] = state["pair_margin"]
        self.max_groom_contact_force = float(state["max_groom_contact_force"])
        self.food_patches = copy.deepcopy(state["food_patches"])
        mj.mj_setState(self.sim.mj_model, self.sim.mj_data,
                      np.asarray(state["integration"]), mj.mjtState.mjSTATE_INTEGRATION)
        self.physics_steps = int(state["physics_steps"])
        self.control = np.asarray(state["control"]).copy()
        self.motor.set_state(state["motor"])
        if "native_mj_data" in state:
            if state["native_mujoco_version"] != mj.__version__:
                raise ValueError("Incompatible native MuJoCo snapshot version")
            native = np.asarray(state["native_mj_data"])
            if native.dtype != np.uint8 or native.ndim != 1:
                raise ValueError("Invalid native MuJoCo state")
            restored = mj.MjData.__new__(mj.MjData)
            restored.__setstate__(native.tobytes())
            if restored.qpos.shape != self.sim.mj_data.qpos.shape or restored.ctrl.shape != self.sim.mj_data.ctrl.shape:
                raise ValueError("Native MuJoCo snapshot has different dimensions")
            mj.mj_copyData(self.sim.mj_data, self.sim.mj_model, restored)
            return
        # mj_forward refreshes derived quantities (site poses, contacts) but its
        # solver also overwrites the restored warm-start guess, which makes the
        # next step diverge from the uninterrupted run in the last digits.
        warmstart = self.sim.mj_data.qacc_warmstart.copy()
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.sim.mj_data.qacc_warmstart[:] = warmstart
