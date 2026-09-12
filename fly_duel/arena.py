"""Two flies in one arena, each with its own body, controller and senses.

This is the physical substrate shared by passive competition and the optional
engineered social policy. Proximity and physical touch can be observed locally;
the arena itself never chooses an opponent-directed action or applies a push.

The single-fly package is untouched. Each fly is presented to the existing
motor primitives and to `Environment.sense` through `FlyUnit`, which exposes the
same small interface a single-fly arena did. The duel-specific motor constructor
resolves each animal's own joints before preparing the inherited primitives.
"""
from __future__ import annotations

import numpy as np
import mujoco as mj
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.compose import ActuatorType
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation, HybridTurningController, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action)

from fly_arena.ethology_arena import make_ethology_fly
from fly_arena.sensors import PhysicalEvents

from .namespaced_motor import NamespacedMotorArbiter
from .social_sensors import SocialFrame, observe_social_frame
from .collisions import configure_mesh_collisions
from .combat_motor import CombatMotorArbiter

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
# One sphere per thorax, so two bodies exclude each other without turning every
# body mesh into a collider. Radius is about the half-width of a real thorax.
BODY_PROXY_RADIUS = .38
# Measured head mesh bounds in its body frame are x[-.015,.495], y[+/- .376],
# z[-.276,.401] mm. This small engineering proxy covers its anterior surface;
# it is an optional stable collision approximation, not a cuticle reconstruction.
HEAD_PROXY_RADIUS = .35
HEAD_PROXY_POSITION = (.24, 0., .06)


class FlyUnit:
    """One fly's body, controller and sensors, shaped like a single-fly arena."""

    def __init__(self, duel, name, fly):
        self.duel, self.name, self.fly = duel, name, fly
        self.control = np.zeros(2)
        self.steps = PreprogrammedSteps()
        self.dof_order = [d for d in fly.get_actuated_jointdofs_order("position") if d.child.is_leg()]
        self.controller_stride = 5
        self.controller = None            # built once the simulation exists
        self.motor = None
        self.max_groom_contact_force = 0.

    # --- resolved after the model is compiled -------------------------------
    def bind(self, sim, seed):
        self.sim = sim
        model = sim.mj_model
        self.thorax_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{self.name}/c_thorax")
        self.site_ids = {key: model.site(f"{self.name}/{key}").id for key in
                         ("mouth", "head_frame", "l_odor", "r_odor",
                          *[f"{leg}_{kind}" for leg in LEGS for kind in ("tip", "taste")])}
        self.region_geom_ids = {region: model.geom(f"{self.name}/clean_{region}").id
                                for region in ("head", "antenna_left", "antenna_right")}
        self.rub_geom_ids = {leg: model.geom(f"{self.name}/{leg}_rub").id for leg in ("lf", "rf")}
        self.proxy_geom_id = model.geom(f"{self.name}/body_proxy").id
        self.head_proxy_geom_id = (model.geom(f"{self.name}/head_proxy").id
                                   if self.duel.head_contact else None)
        self.leg_by_geom_id = {}
        for segment, geoms in self.fly.bodyseg_to_mjcfgeom.items():
            if segment.is_leg():
                for geom in geoms:
                    self.leg_by_geom_id[model.geom(geom.name).id] = segment.pos
        self.contact_pair_labels = {frozenset(self.rub_geom_ids.values()): "front_left|front_right"}
        for leg, gid in self.rub_geom_ids.items():
            for region, rid in self.region_geom_ids.items():
                side = "left" if leg == "lf" else "right"
                self.contact_pair_labels[frozenset((gid, rid))] = f"{region}|front_{side}"
        self.eye_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, f"{self.name}/{side}_eye_cam_camera")
                        for side in ("l", "r")]
        self.controller = HybridTurningController(timestep=sim.timestep * self.controller_stride,
                                                  preprogrammed_steps=self.steps,
                                                  output_dof_order=self.dof_order)
        self.controller.reset(seed=seed)
        self.motor = CombatMotorArbiter(self)

    # --- the interface MotorArbiter and Environment.sense expect ------------
    @property
    def position(self):
        return self.sim.mj_data.xpos[self.thorax_id].copy()

    @property
    def heading(self):
        mat = self.sim.mj_data.xmat[self.thorax_id].reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    @property
    def upright(self):
        return float(self.sim.mj_data.xmat[self.thorax_id].reshape(3, 3)[2, 2])

    @property
    def movement(self):
        start = self.sim.mj_model.body_dofadr[self.thorax_id]
        return float(np.linalg.norm(self.sim.mj_data.qvel[start:start + 3]))

    @property
    def mouth_position(self):
        return self.sim.mj_data.site_xpos[self.site_ids["mouth"]].copy()

    @property
    def antenna_positions(self):
        return np.array([self.sim.mj_data.site_xpos[self.site_ids[f"{s}_odor"]] for s in ("l", "r")])

    @property
    def tarsal_positions(self):
        return np.array([self.sim.mj_data.site_xpos[self.site_ids[f"{leg}_taste"]] for leg in LEGS])

    @property
    def ground_support(self):
        supported = set()
        for contact in self.sim.mj_data.contact:
            if contact.exclude or contact.dist > .001:
                continue
            for leg_geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if int(other) in self.duel.ground_geom_ids and int(leg_geom) in self.leg_by_geom_id:
                    supported.add(self.leg_by_geom_id[int(leg_geom)])
        return len(supported)

    @property
    def stable(self):
        return self.upright > .85 and self.movement < 2. and self.ground_support >= 3

    def eyes(self):
        frames = []
        for camera in self.eye_ids:
            self.duel.eye_renderer.update_scene(self.sim.mj_data, camera, scene_option=self.duel.eye_options)
            frames.append(self.duel.eye_renderer.render())
        return np.stack(frames)

    def apply_leg_action(self, action):
        # The ethology body has 44 position actuators; the gait controller only
        # produces the 42 leg angles, so the proboscis pair is left at zero here
        # and driven by the motor arbiter instead.
        order = self.fly.get_actuated_jointdofs_order("position")
        targets = np.zeros(len(order))
        targets[[i for i, dof in enumerate(order) if dof.child.is_leg()]] = action.joint_angles
        self.sim.set_actuator_inputs(self.name, ActuatorType.POSITION, targets)
        self.sim.set_leg_adhesion_states(self.name, action.adhesion_onoff)

    def observation(self):
        return HybridControllerObservation.from_sim(self.sim, self.name)


class DuelArena:
    """One world, two flies, shared food."""

    mouth_contact_thickness = .04

    def __init__(self, seed=1, *, food_patches=None, eye_size=96, start=None,
                 body_contact=True, warmup=True, names=("fly", "rival"),
                 start_headings=None, head_contact=False, collision_geometry="legacy"):
        self.food_patches = [dict(patch) for patch in (food_patches or [])]
        self.body_contact = bool(body_contact)
        self.head_contact = bool(head_contact)
        if collision_geometry not in {"legacy", "mesh"}:
            raise ValueError("collision_geometry must be legacy or mesh")
        self.collision_geometry = collision_geometry
        self._mesh_names = []
        # Measured: the grooming half of the per-substep contact loop costs about
        # 15% of a model second. It can only fire when a fly carries dust, so the
        # owner switches it off when none does.
        self.track_grooming = True
        self.names = tuple(names)
        if len(self.names) != 2 or len(set(self.names)) != 2:
            raise ValueError("DuelArena requires two distinct fly names")
        if start is None:
            start = [(-3.5, 0., .8), (3.5, 0., .8)]
        headings = np.zeros(2) if start_headings is None else np.asarray(start_headings, dtype=float)
        if headings.shape != (2,) or not np.isfinite(headings).all():
            raise ValueError("start_headings must contain two finite angles in radians")
        self.world = FlatGroundWorld(name="duel_arena", half_size=22)
        spec = self.world.mjcf_root
        self.world.ground_geom.rgba = [.24, .31, .33, 1]
        spec.material("grid").texrepeat = [14, 14]
        spec.texture("checker").rgb1 = [.22, .28, .29]
        spec.texture("checker").rgb2 = [.25, .31, .32]
        for i, (pos, size) in enumerate([
                ((-20.5, 0, 3), (.5, 21, 3)), ((20.5, 0, 3), (.5, 21, 3)),
                ((0, -20.5, 3), (20, .5, 3)), ((0, 20.5, 3), (20, .5, 3))]):
            geom = spec.worldbody.add_geom(name=f"wall_{i}", type=mj.mjtGeom.mjGEOM_BOX, pos=pos,
                                           size=size, rgba=[.58, .65, .62, 1], contype=0, conaffinity=0)
            self.world.ground_geoms.append(geom)
        for i, patch in enumerate(self.food_patches):
            patch.setdefault("id", str(i))
            spec.worldbody.add_geom(
                name=f"food_{i}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                pos=[patch.get("x", 0.), patch.get("y", 0.), patch.get("surface_z", .01) - .005],
                size=[patch.get("radius", 1.), .005, 0], rgba=[.8, .36, .12, 1.],
                contype=0, conaffinity=0)

        tint = ([.95, .62, .18, 1.], [.55, .78, .95, 1.])
        self.units = []
        for index, name in enumerate(self.names):
            fly = make_ethology_fly(name)
            fly.add_vision()
            if self.collision_geometry == "mesh":
                self._mesh_names.append(configure_mesh_collisions(fly, index, enabled=self.body_contact))
            thorax = fly.bodyseg_to_mjcfbody[BodySegment("c_thorax")]
            thorax.add_geom(name="body_proxy", type=mj.mjtGeom.mjGEOM_SPHERE, pos=[.1, 0, 0],
                            size=[BODY_PROXY_RADIUS, 0, 0], mass=1e-9,
                            contype=0, conaffinity=0, rgba=tint[index % 2], group=1)
            if self.head_contact:
                head = fly.bodyseg_to_mjcfbody[BodySegment("c_head")]
                head.add_geom(name="head_proxy", type=mj.mjtGeom.mjGEOM_SPHERE,
                              pos=HEAD_PROXY_POSITION, size=[HEAD_PROXY_RADIUS, 0, 0],
                              mass=1e-9, contype=0, conaffinity=0,
                              rgba=[0., 0., 0., 0.], group=1)
            angle = float(headings[index])
            rotation = Rotation3D("quat", [np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
            self.world.add_fly(fly, list(start[index]), rotation,
                               bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
                               add_ground_contact_sensors=False)
            self.units.append(FlyUnit(self, name, fly))
        self._configure_contacts()
        self.sim = Simulation(self.world, timestep=.0001)
        self.sim.mj_model.vis.global_.offwidth = 1024
        self.sim.mj_model.vis.global_.offheight = 768
        self.eye_renderer = mj.Renderer(self.sim.mj_model, eye_size, eye_size)
        self.eye_options = mj.MjvOption()
        self.eye_options.geomgroup[1:3] = 0
        self.body_renderer = None
        self.ground_geom_ids = set(map(int, self.sim._internal_ground_geom_ids))
        for index, unit in enumerate(self.units):
            unit.bind(self.sim, seed + index)
            unit.apply_leg_action(LocomotionAction(
                unit.steps.default_pose_by_dof_order(unit.dof_order), np.ones(6, bool)))
        self._jacobian = np.zeros((3, self.sim.mj_model.nv))
        self._wrench = np.zeros(6)
        self._social_proxy_ids = frozenset(unit.proxy_geom_id for unit in self.units)
        self.mesh_owner = np.full(self.sim.mj_model.ngeom, -1, dtype=np.int8)
        self.core_mesh = np.zeros(self.sim.mj_model.ngeom, dtype=bool)
        self.foreleg_mesh = np.zeros(self.sim.mj_model.ngeom, dtype=bool)
        if self.collision_geometry == "mesh":
            for index, mesh_names in enumerate(self._mesh_names):
                for name in mesh_names:
                    gid = self.sim.mj_model.geom(name).id
                    self.mesh_owner[gid] = index
                    part = name.rsplit("/", 1)[-1]
                    self.core_mesh[gid] = part.startswith("c_abdomen") or part in {
                        "c_thorax", "c_head", "l_eye", "r_eye", "l_wing", "r_wing"}
                    self.foreleg_mesh[gid] = part.startswith(("lf_", "rf_"))
        self._social_contact_pairs = {
            frozenset((self.sim.mj_model.geom(a).id, self.sim.mj_model.geom(b).id))
            for _, a, b in self._duel_pair_specs
        }
        self._social_touch = False
        self._social_impulse = 0.0
        self._social_contact_position = None
        self.max_interfly_penetration_mm = 0.
        self.max_core_penetration_mm = 0.
        self.worst_contact_pair = None
        self.foreleg_contact_s = np.zeros(2)
        self.duel_pair_id = mj.mj_name2id(self.sim.mj_model, mj.mjtObj.mjOBJ_PAIR, "duel_bodies")
        if warmup:
            self.sim.warmup(.05)
        else:
            # Simulation resets the neutral qpos without refreshing xmat/xpos.
            mj.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.sim.mj_data.time = 0
        self.physics_steps = 0

    def _configure_contacts(self):
        for name in self.names:
            pairs = [("lf_rub", "rf_rub")]
            pairs += [(f"{leg}_rub", f"clean_{region}") for leg in ("lf", "rf")
                      for region in ("head", "antenna_left", "antenna_right")]
            for a, b in pairs:
                self.world.mjcf_root.add_pair(
                    name=f"groom_{name}_{a}_{b}", geomname1=f"{name}/{a}", geomname2=f"{name}/{b}",
                    friction=(.4, .4, .002, .0001, .0001), solref=(.002, 1.),
                    solimp=(.9, .95, .001, .5, 2.), margin=.001)
        # Passive geometry retains its one thorax pair. Close encounter scenes
        # can add three head pairs to prevent face-on visual interpenetration.
        # There are never all-mesh or opponent-leg contact pairs.
        first, second = self.names
        if self.collision_geometry == "mesh":
            # Cross-fly mesh masks perform broadphase filtering; no duplicate
            # proxy constraints and no 69x69 list of explicit pair entries.
            self._duel_pair_specs = []
            return
        self._duel_pair_specs = [("duel_bodies", f"{first}/body_proxy", f"{second}/body_proxy")]
        if self.head_contact:
            self._duel_pair_specs.extend([
                ("duel_heads", f"{first}/head_proxy", f"{second}/head_proxy"),
                ("duel_first_head", f"{first}/head_proxy", f"{second}/body_proxy"),
                ("duel_second_head", f"{first}/body_proxy", f"{second}/head_proxy"),
            ])
        for name, a, b in self._duel_pair_specs:
            self.world.mjcf_root.add_pair(
                name=name, geomname1=a, geomname2=b,
                friction=(.6, .6, .005, .0001, .0001), solref=(.004, 1.),
                solimp=(.9, .95, .002, .5, 2.), margin=.001 if self.body_contact else -1.)

    def step(self, decisions, gaits, milliseconds=10, motor_off=False):
        """Advance both flies together; one physics stream, two controllers."""
        events = [PhysicalEvents() for _ in self.units]
        self._social_touch = False
        self._social_impulse = 0.0
        self._social_contact_position = None
        self.foreleg_contact_s[:] = 0.
        for unit in self.units:
            unit.max_groom_contact_force = 0.
        for index, unit in enumerate(self.units):
            unit.control = np.clip(np.asarray(gaits[index], float), -1.2, 1.2)
            unit.motor.select(decisions[index], motor_off=motor_off)
        for _ in range(round(milliseconds / 1000 / self.sim.timestep)):
            for unit in self.units:
                if self.physics_steps % unit.controller_stride == 0:
                    unit.motor.update(unit.control)
            self.sim.step()
            self.physics_steps += 1
            for unit in self.units:
                unit.motor.elapsed += self.sim.timestep
            self._accumulate(events)
        if not np.isfinite(self.sim.mj_data.qpos).all():
            raise RuntimeError("Non-finite physics state")
        return events

    def _accumulate(self, events):
        dt = self.sim.timestep
        data = self.sim.mj_data
        for index, unit in enumerate(self.units):
            mouth = data.site_xpos[unit.site_ids["mouth"]]
            for patch in self.food_patches:
                if (abs(mouth[2] - patch.get("surface_z", .01)) <= self.mouth_contact_thickness
                        and np.linalg.norm(mouth[:2] - [patch.get("x", 0.), patch.get("y", 0.)])
                        <= patch.get("radius", 1.)):
                    key = str(patch["id"])
                    bucket = events[index]
                    bucket.mouth_contact_s_by_food[key] = bucket.mouth_contact_s_by_food.get(key, 0.) + dt
                    if unit.motor.ingestion_enabled:
                        bucket.contact_s_by_food[key] = bucket.contact_s_by_food.get(key, 0.) + dt
                    break
        model = self.sim.mj_model
        by_pair = [{} for _ in self.units] if self.track_grooming else None
        forelegs_touching = set()
        for row, contact in enumerate(data.contact):
            if contact.exclude or contact.dist > .001:
                continue
            geoms = frozenset((int(contact.geom1), int(contact.geom2)))
            if self.is_interfly_contact(int(contact.geom1), int(contact.geom2)):
                self._social_touch = True
                self._social_contact_position = np.asarray(contact.pos).copy()
                self._wrench.fill(0.0)
                mj.mj_contactForce(model, data, row, self._wrench)
                self._social_impulse += float(np.linalg.norm(self._wrench[:3])) * dt
                penetration = max(0., -float(contact.dist))
                if self.core_mesh[contact.geom1] and self.core_mesh[contact.geom2]:
                    self.max_core_penetration_mm = max(self.max_core_penetration_mm, penetration)
                for gid in (contact.geom1, contact.geom2):
                    if self.foreleg_mesh[gid]:
                        forelegs_touching.add(int(self.mesh_owner[gid]))
                if penetration > self.max_interfly_penetration_mm:
                    self.max_interfly_penetration_mm = penetration
                    self.worst_contact_pair = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
            if not self.track_grooming:
                continue
            for index, unit in enumerate(self.units):
                label = unit.contact_pair_labels.get(geoms)
                if label is None:
                    continue
                velocities = []
                for gid in (contact.geom1, contact.geom2):
                    jac = self._jacobian
                    jac.fill(0.)
                    mj.mj_jac(model, data, jac, None, contact.pos, int(model.geom_bodyid[gid]))
                    velocities.append(jac @ data.qvel)
                relative = velocities[0] - velocities[1]
                normal = np.asarray(contact.frame[:3])
                sliding = float(np.linalg.norm(relative - np.dot(relative, normal) * normal)) * dt
                by_pair[index][label] = max(by_pair[index].get(label, 0.), sliding)
                wrench = self._wrench
                wrench.fill(0.)
                mj.mj_contactForce(model, data, row, wrench)
                unit.max_groom_contact_force = max(unit.max_groom_contact_force,
                                                   float(np.linalg.norm(wrench[:3])))
        for index in forelegs_touching:
            self.foreleg_contact_s[index] += dt
        for index, pairs in enumerate(by_pair or []):
            bucket = events[index]
            for label, sliding in pairs.items():
                bucket.grooming_sliding_by_pair[label] = bucket.grooming_sliding_by_pair.get(label, 0.) + sliding
                bucket.grooming_contact_s_by_pair[label] = bucket.grooming_contact_s_by_pair.get(label, 0.) + dt

    def social_frames(self, range_mm=3.0, enabled=True) -> list[SocialFrame]:
        """Read local opponent observations for the preceding physics exchange.

        Calls do not consume the contact record or change physics. The one
        physical contact set has equal summed impulse magnitude for both participants;
        bearing is evaluated in each animal's own frame.
        """
        positions = [unit.position for unit in self.units]
        return [observe_social_frame(
            own_position=positions[index], own_heading=unit.heading,
            rival_position=positions[1 - index], range_mm=range_mm, enabled=enabled,
            body_contact=self._social_touch, contact_impulse=self._social_impulse,
            contact_position=self._social_contact_position,
        ) for index, unit in enumerate(self.units)]

    def contacts_between_flies(self):
        """Number of active contacts among the declared opponent proxy pairs."""
        if not self.body_contact:
            return 0
        return sum(1 for contact in self.sim.mj_data.contact
                   if not contact.exclude and contact.dist <= .001
                   and self.is_interfly_contact(int(contact.geom1), int(contact.geom2)))

    def is_interfly_contact(self, first, second):
        if not self.body_contact:
            return False
        if self.collision_geometry == "mesh":
            a, b = self.mesh_owner[first], self.mesh_owner[second]
            return bool(a >= 0 and b >= 0 and a != b)
        return frozenset((first, second)) in self._social_contact_pairs

    def render(self, *, distance=14., azimuth=135., elevation=-40., lookat=None, size=(640, 480)):
        if self.body_renderer is None:
            self.body_renderer = mj.Renderer(self.sim.mj_model, size[1], size[0])
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        centre = np.mean([unit.position for unit in self.units], axis=0) if lookat is None else np.asarray(lookat)
        camera.lookat[:] = centre
        self.body_renderer.update_scene(self.sim.mj_data, camera)
        return self.body_renderer.render()

    def close(self):
        self.eye_renderer.close()
        if self.body_renderer is not None:
            self.body_renderer.close()
