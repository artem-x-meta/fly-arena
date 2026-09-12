"""Local social sensing and the two bodies' independent actuator addresses."""
import os

import numpy as np
import pytest

from fly_duel.social_sensors import observe_social_frame


def test_proximity_is_bounded_and_bearing_uses_own_heading():
    visible = observe_social_frame(own_position=[1, 1, 0], own_heading=np.pi / 2,
                                   rival_position=[1, 3, 0], range_mm=2.0)
    assert visible.detected and visible.distance_mm == 2.0
    assert abs(visible.bearing) < 1e-12
    hidden = observe_social_frame(own_position=[1, 1, 0], own_heading=np.pi / 2,
                                  rival_position=[1, 3.001, 0], range_mm=2.0)
    assert not hidden.detected and hidden.distance_mm is None
    assert hidden.bearing == 0.0


def test_touch_has_local_bearing_without_unbounded_distance():
    touched = observe_social_frame(
        own_position=[0, 0, 0], own_heading=np.pi / 2,
        rival_position=[8, 0, 0], range_mm=0.0,
        body_contact=True, contact_impulse=0.125, contact_position=[0, -0.2, 0],
    )
    assert touched.detected and touched.body_contact
    assert touched.distance_mm is None
    assert touched.contact_impulse == 0.125
    assert abs(abs(touched.bearing) - np.pi) < 1e-12


def test_sensor_ablation_suppresses_touch_as_well_as_proximity():
    frame = observe_social_frame(
        own_position=[0, 0, 0], own_heading=0.7, rival_position=[0.5, 0, 0],
        enabled=False, body_contact=True, contact_impulse=12.0,
    )
    assert not frame.detected and not frame.body_contact
    assert frame.contact_impulse == 0.0 and frame.distance_mm is None
    assert frame.own_heading == 0.7
    assert frame.own_position_mm == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("radius", [-1.0, np.nan, np.inf])
def test_sensor_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="range"):
        observe_social_frame(own_position=[0, 0, 0], own_heading=0,
                             rival_position=[0, 0, 0], range_mm=radius)


physics = pytest.mark.skipif(os.environ.get("RUN_PHYSICS_TEST") != "1",
                             reason="Requires a working MuJoCo OpenGL context")


@physics
def test_each_motor_resolves_own_joints_and_rotated_ik():
    import mujoco as mj
    from fly_duel.arena import DuelArena

    arena = DuelArena(names=("amber", "blue"), start_headings=[0, np.pi], warmup=False)
    try:
        first, second = arena.units
        assert not set(first.motor.qpos_ids) & set(second.motor.qpos_ids)
        assert not set(first.motor.qvel_ids) & set(second.motor.qvel_ids)
        model = arena.sim.mj_model
        for unit in arena.units:
            for joint in unit.motor.joint_ids:
                assert mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(joint)).startswith(unit.name + "/")
            assert max(unit.motor.ik_errors.values()) < 0.03
        assert abs(first.heading) < 1e-12
        assert abs(abs(second.heading) - np.pi) < 1e-12
        # The task-space targets rotate with the body, leaving joint-space
        # primitives invariant under this flat-floor half-turn and translation.
        np.testing.assert_allclose(first.motor.crouch, second.motor.crouch, atol=1e-8)
        for name in first.motor._groom_tables:
            np.testing.assert_allclose(first.motor._groom_tables[name], second.motor._groom_tables[name], atol=1e-7)
    finally:
        arena.close()


@physics
@pytest.mark.parametrize("enabled", [True, False])
def test_body_impulse_comes_from_actual_constraints_even_without_grooming(enabled):
    import mujoco as mj
    from fly_duel.arena import DuelArena

    arena = DuelArena(start=[(0, 0, .8), (.65, 0, .8)], body_contact=enabled, warmup=False)
    try:
        arena.track_grooming = False
        # One exact physics step: force * dt is independently recoverable from
        # its constraint, so this catches endpoint-only and force/impulse errors.
        arena.step(["IDLE", "IDLE"], [[0, 0], [0, 0]], milliseconds=0.1)
        data, model = arena.sim.mj_data, arena.sim.mj_model
        expected = 0.0
        wrench = np.zeros(6)
        for index, contact in enumerate(data.contact):
            if (not contact.exclude and contact.dist <= .001
                    and frozenset((int(contact.geom1), int(contact.geom2))) == arena._social_proxy_ids):
                mj.mj_contactForce(model, data, index, wrench)
                expected += np.linalg.norm(wrench[:3]) * arena.sim.timestep
        frames = arena.social_frames(range_mm=0.0)
        for frame in frames:
            assert frame.body_contact is enabled
            assert frame.contact_impulse == pytest.approx(expected)
        assert bool(expected > 0.0) is enabled
        before = data.qpos.copy()
        assert arena.social_frames(range_mm=0.0) == frames
        np.testing.assert_array_equal(before, data.qpos)
        assert not arena.social_frames(enabled=False)[0].detected
    finally:
        arena.close()


@physics
@pytest.mark.parametrize("head_enabled,body_enabled", [(False, True), (True, True), (True, False)])
def test_optional_head_contact_is_sensed_and_global_contact_ablation_disables_it(head_enabled, body_enabled):
    import mujoco as mj
    from fly_duel.arena import DuelArena

    # Head proxies touch here while the original thorax proxies remain apart.
    arena = DuelArena(start=[(-1.10, 0., .8), (1.10, 0., .8)], start_headings=[0., np.pi],
                      head_contact=head_enabled, body_contact=body_enabled, warmup=False)
    try:
        arena.track_grooming = False
        assert len(arena._social_contact_pairs) == (4 if head_enabled else 1)
        arena.step(["IDLE", "IDLE"], [[0, 0], [0, 0]], milliseconds=.1)
        touching = head_enabled and body_enabled
        assert (arena.contacts_between_flies() > 0) is touching
        frames = arena.social_frames(range_mm=0.)
        for frame in frames:
            assert frame.body_contact is touching
            assert (frame.contact_impulse > 0.) is touching
        if touching:
            model, data = arena.sim.mj_model, arena.sim.mj_data
            head_pair = frozenset(unit.head_proxy_geom_id for unit in arena.units)
            active_pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in data.contact
                            if not c.exclude and c.dist <= .001}
            assert head_pair in active_pairs
            assert arena._social_proxy_ids not in active_pairs
            wrench = np.zeros(6)
            expected = 0.
            for index, contact in enumerate(data.contact):
                if (not contact.exclude and contact.dist <= .001
                        and frozenset((int(contact.geom1), int(contact.geom2))) in arena._social_contact_pairs):
                    mj.mj_contactForce(model, data, index, wrench)
                    expected += np.linalg.norm(wrench[:3]) * arena.sim.timestep
            assert frames[0].contact_impulse == pytest.approx(expected)
    finally:
        arena.close()
