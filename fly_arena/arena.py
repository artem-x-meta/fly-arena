"""A physical 40 x 40 mm arena and a standard NeuroMechFly walking body."""
from __future__ import annotations

import mujoco as mj
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridTurningController, HybridControllerObservation, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)


class Arena:
    def __init__(self, seed=1, eye_size=96, *, warmup=True):
        self.fly = self._make_fly()
        self.fly.add_vision()
        self.world = FlatGroundWorld(name="male_cns_arena", half_size=22)
        spec = self.world.mjcf_root
        self.world.ground_geom.rgba = [.24, .31, .33, 1]
        spec.material("grid").texrepeat = [14, 14]
        spec.texture("checker").rgb1 = [.22, .28, .29]
        spec.texture("checker").rgb2 = [.25, .31, .32]
        self.obstacles = []
        for i, (pos, size) in enumerate([
            ((-20.5, 0, 3), (.5, 21, 3)), ((20.5, 0, 3), (.5, 21, 3)),
            ((0, -20.5, 3), (20, .5, 3)), ((0, 20.5, 3), (20, .5, 3)),
        ]):
            geom = spec.worldbody.add_geom(name=f"wall_{i}", type=mj.mjtGeom.mjGEOM_BOX,
                                          pos=pos, size=size, rgba=[.58, .65, .62, 1],
                                          contype=0, conaffinity=0)
            self.world.ground_geoms.append(geom)
        lid = spec.worldbody.add_geom(name="invisible_lid", type=mj.mjtGeom.mjGEOM_BOX,
                                      pos=[0, 0, 6.5], size=[21, 21, .5],
                                      rgba=[0, 0, 0, 0], contype=0, conaffinity=0)
        self.world.ground_geoms.append(lid)
        # All physical obstacles are registered before the fly's contact pairs.
        for i, (x, y, radius, height) in enumerate([(-9, 7, 1.8, .3), (8, -5, 2.2, .4), (8, 9, 1.4, .2)]):
            geom = spec.worldbody.add_geom(name=f"stone_{i}", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                                          pos=[x, y, 0], size=[radius, radius * .75, height],
                                          rgba=[.39, .44, .43, 1], contype=0, conaffinity=0)
            self.world.ground_geoms.append(geom)
            self.obstacles.append((x, y, radius))
        # Contrasting panels create visual structure. They carry no task reward.
        for i, x in enumerate(np.linspace(-18, 18, 13)):
            for y, color in ((19.98, [.10, .14, .17, 1]), (-19.98, [.77, .72, .45, 1])):
                spec.worldbody.add_geom(name=f"stripe_{i}_{'n' if y > 0 else 's'}",
                                        type=mj.mjtGeom.mjGEOM_BOX, pos=[x, y, 3], size=[.7, .01, 2.8],
                                        rgba=color, contype=0, conaffinity=0)
        for name, pos, color in (("amber", [19.96, 7, 2.5], [1, .64, .18, 1]),
                                 ("blue", [-19.96, -7, 2.5], [.12, .55, .77, 1])):
            mat = spec.add_material(name=f"beacon_{name}", rgba=color, emission=.6)
            spec.worldbody.add_geom(name=f"beacon_{name}", type=mj.mjtGeom.mjGEOM_BOX,
                                    pos=pos, size=[.02, 2, 2], material=mat.name,
                                    contype=0, conaffinity=0)
        self._configure_world()
        self.world.add_fly(self.fly, [0, 0, .8], Rotation3D("quat", [1, 0, 0, 0]),
                           bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
                           add_ground_contact_sensors=False)
        self._configure_contacts()
        self.sim = Simulation(self.world, timestep=.0001)
        self.sim.mj_model.vis.global_.offwidth = 1024
        self.sim.mj_model.vis.global_.offheight = 768
        self.thorax_id = mj.mj_name2id(self.sim.mj_model, mj.mjtObj.mjOBJ_BODY, "fly/c_thorax")
        if self.thorax_id < 0:
            raise RuntimeError("Thorax body not found")
        self.steps = PreprogrammedSteps()
        self.dof_order = [d for d in self.fly.get_actuated_jointdofs_order("position") if d.child.is_leg()]
        # Update the gait/reflex adapter at 2 kHz; retain 10 kHz MuJoCo physics.
        self.controller_stride = 5
        self.controller = HybridTurningController(timestep=self.sim.timestep * self.controller_stride,
                                                  preprogrammed_steps=self.steps,
                                                  output_dof_order=self.dof_order)
        self.controller.reset(seed=seed)
        initial = LocomotionAction(self.steps.default_pose_by_dof_order(self.dof_order), np.ones(6, bool))
        self._apply_leg_action(initial)
        if warmup:
            self.sim.warmup(.05)
        self.sim.mj_data.time = 0
        self.eye_renderer = mj.Renderer(self.sim.mj_model, eye_size, eye_size)
        self.eye_options = mj.MjvOption()
        self.eye_options.geomgroup[1:3] = 0
        self.eye_ids = [mj.mj_name2id(self.sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, f"fly/{side}_eye_cam_camera")
                        for side in ("l", "r")]
        if min(self.eye_ids) < 0:
            raise RuntimeError("Eye camera not found")
        self.body_renderer = None
        self.control = np.zeros(2)
        self.physics_steps = 0

    def _make_fly(self):
        return make_locomotion_fly(name="fly", add_adhesion=True, colorize=True)

    def _configure_world(self):
        pass

    def _configure_contacts(self):
        pass

    def _apply_leg_action(self, action):
        apply_locomotion_action(self.sim, self.fly.name, action)

    @property
    def position(self):
        return self.sim.mj_data.xpos[self.thorax_id].copy()

    @property
    def heading(self):
        mat = self.sim.mj_data.xmat[self.thorax_id].reshape(3, 3)
        return np.arctan2(mat[1, 0], mat[0, 0])

    @property
    def upright(self):
        return float(self.sim.mj_data.xmat[self.thorax_id].reshape(3, 3)[2, 2])

    def eyes(self):
        frames = []
        for camera in self.eye_ids:
            self.eye_renderer.update_scene(self.sim.mj_data, camera, scene_option=self.eye_options)
            frames.append(self.eye_renderer.render())
        return np.stack(frames)

    def step(self, command, milliseconds=10):
        command = np.asarray(command, float)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError("Invalid gait command")
        self.control = np.clip(command, 0, 1.2)
        for _ in range(round(milliseconds / 1000 / self.sim.timestep)):
            if self.physics_steps % self.controller_stride == 0:
                obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
                action = self.controller.step(self.control, obs)
                apply_locomotion_action(self.sim, self.fly.name, action)
            self.sim.step()
            self.physics_steps += 1
        if not np.isfinite(self.sim.mj_data.qpos).all():
            raise RuntimeError("Non-finite physics state")

    def render(self, *, closeup=False):
        if self.body_renderer is None:
            self.body_renderer = mj.Renderer(self.sim.mj_model, 480, 640)
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.lookat = self.position if closeup else [0, 0, .1]
        camera.distance = 7.0 if closeup else 49
        camera.azimuth = 125
        camera.elevation = -35 if closeup else -70
        self.body_renderer.update_scene(self.sim.mj_data, camera)
        return self.body_renderer.render()

    def close(self):
        self.eye_renderer.close()
        if self.body_renderer is not None:
            self.body_renderer.close()


class BodyDemo:
    """Explicit scripted diagnostic, not connectome-derived behavior.

    Wandering and obstacle avoidance use world pose. This controller is never
    constructed in connectome mode. It makes installation/physics easy to check
    before downloading the graph.
    """
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.turn = 0.0

    def step(self, arena: Arena, seconds=.01):
        self.turn += -.5 * self.turn * seconds + .5 * np.sqrt(seconds) * self.rng.normal()
        x, y, _ = arena.position
        force = np.zeros(2)
        if max(abs(x), abs(y)) > 14:
            force -= np.array([x, y]) / 10
        for ox, oy, radius in arena.obstacles:
            delta = np.array([x - ox, y - oy])
            distance = np.linalg.norm(delta)
            if distance < radius + 4:
                force += delta / max(distance, .1) * 3
        if np.linalg.norm(force) > .1:
            target = np.arctan2(force[1], force[0])
            error = np.arctan2(np.sin(target - arena.heading), np.cos(target - arena.heading))
            turn = np.clip(error * .5, -.55, .55)
        else:
            turn = np.clip(self.turn, -.35, .35)
        return np.clip([.85 - turn, .85 + turn], .2, 1.2)
