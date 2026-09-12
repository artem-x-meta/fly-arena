"""Collision regressions for visible bodies passing through an opponent.

The old small thorax/head spheres miss the abdomen, wings and limbs. Counted
contacts alone therefore are insufficient: first check that every visible mesh
participates, and then test actual contact constraints away from the old spheres.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_PHYSICS_TEST") != "1",
                                reason="Requires a working MuJoCo OpenGL context")


def _visible_meshes(model, name):
    import mujoco as mj

    return [index for index in range(model.ngeom)
            if model.geom_type[index] == mj.mjtGeom.mjGEOM_MESH
            and model.geom(index).name.startswith(name + "/")]


def _channels_allow(model, first, second):
    return bool((model.geom_contype[first] & model.geom_conaffinity[second])
                or (model.geom_contype[second] & model.geom_conaffinity[first]))


def _active_opponent_contacts(arena):
    model = arena.sim.mj_model
    first, second = [set(_visible_meshes(model, unit.name)) for unit in arena.units]
    return [(index, contact) for index, contact in enumerate(arena.sim.mj_data.contact)
            if not contact.exclude and contact.dist <= .001
            and ((int(contact.geom1) in first and int(contact.geom2) in second)
                 or (int(contact.geom2) in first and int(contact.geom1) in second))]


@pytest.mark.parametrize("enabled", [True, False])
def test_every_visible_mesh_has_opponent_coverage_without_new_self_contacts(enabled):
    from fly_duel.arena import DuelArena

    arena = DuelArena(collision_geometry="mesh", body_contact=enabled, warmup=False,
                      names=("amber", "cyan"))
    try:
        model = arena.sim.mj_model
        groups = [_visible_meshes(model, name) for name in arena.names]
        # Assert named body regions, rather than accepting a nonempty subset
        # which could accidentally contain only the same old thorax/head parts.
        for name, mesh_ids in zip(arena.names, groups):
            labels = {model.geom(index).name.split("/", 1)[1] for index in mesh_ids}
            assert {"c_thorax", "c_head", "c_abdomen12", "c_abdomen3",
                    "c_abdomen4", "c_abdomen5", "c_abdomen6",
                    "l_wing", "r_wing", "l_eye", "r_eye"} <= labels
            for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
                assert {leg + "_coxa", leg + "_trochanterfemur", leg + "_tibia",
                        leg + "_tarsus1", leg + "_tarsus5"} <= labels
            for first in mesh_ids:
                for second in mesh_ids:
                    assert not _channels_allow(model, first, second), (
                        model.geom(first).name, model.geom(second).name)
                # Existing explicit floor/groom pairs remain responsible for
                # those contacts: the new masks must not add duplicate paths.
                for ground in arena.ground_geom_ids:
                    assert not _channels_allow(model, first, ground)
        for first in groups[0]:
            for second in groups[1]:
                assert _channels_allow(model, first, second) is enabled, (
                    model.geom(first).name, model.geom(second).name)
    finally:
        arena.close()


@pytest.mark.parametrize("enabled", [True, False])
def test_rear_body_contact_is_a_physical_constraint_and_global_ablation_removes_it(enabled):
    import mujoco as mj
    from fly_duel.arena import DuelArena

    # Two same-facing flies: the first head meets the second abdomen/wing while
    # their thorax sphere centres are 2.6 mm apart (sphere diameter was .76 mm).
    # This is a deliberately overlapping initial contact fixture, not a claim that
    # teleporting flies demonstrates normal fighting or robust locomotion.
    arena = DuelArena(collision_geometry="mesh", start=[(0., 0., .8), (2.6, 0., .8)],
                      start_headings=[0., 0.], body_contact=enabled, warmup=False)
    try:
        arena.track_grooming = False
        arena.step(["IDLE", "IDLE"], [[0., 0.], [0., 0.]], milliseconds=.1)
        model, data = arena.sim.mj_model, arena.sim.mj_data
        contacts = _active_opponent_contacts(arena)
        assert bool(contacts) is enabled
        assert (arena.contacts_between_flies() > 0) is enabled
        impulse = 0.
        wrench = np.zeros(6)
        for index, contact in contacts:
            mj.mj_contactForce(model, data, index, wrench)
            impulse += np.linalg.norm(wrench[:3]) * arena.sim.timestep
        assert bool(impulse > 0.) is enabled
        for frame in arena.social_frames(range_mm=0.):
            assert frame.body_contact is enabled
            assert frame.contact_impulse == pytest.approx(impulse)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    finally:
        arena.close()


@pytest.mark.parametrize("label,starts,headings", [
    ("rear", [(0., 0., .8), (3.6, 0., .8)], [0., 0.]),
    ("side", [(0., -1.5, .8), (-.8, 1.5, .8)], [np.pi / 2, 0.]),
    ("oblique", [(0., 0., .8), (3.4, 1.1, .8)], [np.pi / 6, np.pi]),
])
def test_normal_gait_approaches_block_visible_body_overlap(label, starts, headings, tmp_path):
    import json
    import mujoco as mj
    from fly_duel.arena import DuelArena

    results = {}
    for enabled in (False, True):
        arena = DuelArena(collision_geometry="mesh", body_contact=enabled,
                          start=starts, start_headings=headings)
        try:
            arena.track_grooming = False
            model, data = arena.sim.mj_model, arena.sim.mj_data
            core = [[model.geom(unit.name + "/" + part).id for part in (
                "c_head", "c_thorax", "c_abdomen12", "c_abdomen3",
                "c_abdomen4", "c_abdomen5", "c_abdomen6")]
                    for unit in arena.units]
            cuticle = [ids + [model.geom(unit.name + "/" + part).id
                             for part in ("l_wing", "r_wing", "l_eye", "r_eye")]
                       for ids, unit in zip(core, arena.units)]

            def surface_distance(groups):
                # Distance queries ignore collision masks, so the disabled
                # control cannot hide an overlap by generating no contacts.
                return min(mj.mj_geomDistance(model, data, first, second, .1, None)
                           for first in groups[0] for second in groups[1])

            assert surface_distance(core) > 0., "Bodies must start apart"
            deepest_core, deepest_cuticle, minimum_upright, impulse = 0., 0., 1., 0.
            contact_steps = 0
            for _ in range(50):
                # Ordinary walking actuators advance the challenger. Initial
                # placement above is the only position assignment in this test.
                arena.step(["WALK", "IDLE"], [[.4, .4], [0., 0.]])
                deepest_core = max(deepest_core, -surface_distance(core))
                deepest_cuticle = max(deepest_cuticle, -surface_distance(cuticle))
                minimum_upright = min(minimum_upright, *(unit.upright for unit in arena.units))
                social = arena.social_frames(range_mm=0.)[0]
                contact_steps += int(social.body_contact)
                impulse += social.contact_impulse
            results[str(enabled)] = {
                "core_penetration_mm": deepest_core,
                "body_wing_eye_penetration_mm": deepest_cuticle,
                "all_mesh_penetration_mm": arena.max_interfly_penetration_mm,
                "minimum_upright": minimum_upright,
                "contact_steps": contact_steps,
                "contact_impulse": impulse,
            }
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        finally:
            arena.close()

    (tmp_path / (label + ".json")).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(label, json.dumps(results))
    enabled, disabled = results["True"], results["False"]
    assert disabled["core_penetration_mm"] > .15, "Control must reproduce visible ghosting"
    assert disabled["contact_steps"] == 0 and disabled["contact_impulse"] == 0.
    assert enabled["contact_steps"] > 0 and enabled["contact_impulse"] > 0.
    assert enabled["core_penetration_mm"] < .04
    assert enabled["core_penetration_mm"] < disabled["core_penetration_mm"] * .1
    assert enabled["body_wing_eye_penetration_mm"] < .04
    assert enabled["minimum_upright"] > .90
