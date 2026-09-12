"""Measured foreleg motion and unchanged ordinary actuator ownership."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_PHYSICS_TEST") != "1",
                                reason="Requires a working MuJoCo OpenGL context")


def _tips(arena, unit):
    data = arena.sim.mj_data
    rotation = data.xmat[unit.thorax_id].reshape(3, 3)
    return np.asarray([rotation.T @ (data.site_xpos[unit.site_ids[f"{side}_tip"]]
                                    - unit.position) for side in ("lf", "rf")])


def _equal_state(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _equal_state(first[key], second[key])
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _equal_state(a, b)
    else:
        assert first == second


def test_normal_actions_preserve_existing_outputs_and_controller_state():
    from fly_duel.arena import DuelArena
    from fly_duel.combat_motor import CombatMotorArbiter
    from fly_duel.namespaced_motor import NamespacedMotorArbiter

    arena = DuelArena(names=("amber", "blue"), start_headings=[0., np.pi])
    try:
        unit = arena.units[1]
        original = NamespacedMotorArbiter(unit)
        before = arena.sim.mj_data.qpos.copy()
        combat = CombatMotorArbiter(unit)
        np.testing.assert_array_equal(before, arena.sim.mj_data.qpos)
        for action in ("WALK", "IDLE", "FEED", "GROOM_FRONT", "GROOM_HEAD",
                       "SLEEP_ENTRY", "SLEEP", "WAKE", "EXHAUSTED", "ESCAPE"):
            for elapsed in (0., .30, .70):
                state = original.get_state()
                original.select(action)
                original.elapsed = elapsed
                original.update(np.array([.6, .4]))
                expected = original.get_state()
                expected_ctrl = arena.sim.mj_data.ctrl.copy()
                combat.set_state(state)
                combat.select(action)
                combat.elapsed = elapsed
                combat.update(np.array([.6, .4]))
                _equal_state(expected, combat.get_state())
                np.testing.assert_array_equal(expected_ctrl, arena.sim.mj_data.ctrl)
    finally:
        arena.close()


def test_fence_lifts_reaches_taps_and_recovers_with_four_leg_support():
    from fly_duel.arena import DuelArena
    from fly_duel.combat_motor import CombatMotorArbiter, FENCE_DURATION_S

    arena = DuelArena(names=("amber", "blue"), start_headings=[0., np.pi],
                      collision_geometry="mesh")
    try:
        arena.track_grooming = False
        for unit in arena.units:
            unit.motor = CombatMotorArbiter(unit)
            assert unit.motor.ik_errors["FENCE"] < .002
        np.testing.assert_allclose(arena.units[0].motor._fence_table,
                                   arena.units[1].motor._fence_table, atol=1e-7)
        for _ in range(15):
            arena.step(["IDLE", "IDLE"], [[0., 0.], [0., 0.]])
        baseline = np.array([_tips(arena, unit) for unit in arena.units])
        samples, supports, uprights = [], [], []
        phases = set()
        for _ in range(65):
            arena.step(["FENCE", "FENCE"], [[0., 0.], [0., 0.]])
            samples.append(np.array([_tips(arena, unit) for unit in arena.units]) - baseline)
            for unit in arena.units:
                assert unit.motor.owner == "FENCE"
                assert not unit.motor.ingestion_enabled
                assert np.all(unit.motor.adhesion[[1, 2, 4, 5]] == 1.)
                supports.append(unit.ground_support)
                uprights.append(unit.upright)
                phases.add(unit.motor.primitive_phase)
        samples = np.asarray(samples)
        assert min(supports) >= 4
        assert min(uprights) > .98
        assert np.all(samples[..., 2].max(axis=0) > .25)
        assert np.all(samples[..., 0].max(axis=0) > .45)
        # Both forelegs descend by at least .10 mm before returning to stance.
        assert np.all(samples[29, ..., 2] - samples[40, ..., 2] > .10)
        assert np.all(np.abs(samples[-1, ..., 2]) < .06)
        assert phases == {"fence_settle", "fence_raise", "fence_reach", "fence_tap",
                          "fence_recover", "fence_hold"}
        for unit in arena.units:
            assert unit.motor.elapsed > FENCE_DURATION_S  # repeated select did not reset
            np.testing.assert_array_equal(unit.motor.adhesion, np.ones(6))
            np.testing.assert_array_equal(unit.motor.targets[unit.motor.leg_indices], unit.motor.default)
    finally:
        arena.close()


def test_fence_makes_real_foreleg_contact_from_separated_bodies():
    import mujoco as mj
    from fly_duel.arena import DuelArena
    from fly_duel.combat_motor import CombatMotorArbiter

    arena = DuelArena(collision_geometry="mesh",
                      start=[(-1.125, 0., .8), (1.125, 0., .8)],
                      start_headings=[0., np.pi])
    try:
        arena.track_grooming = False
        for unit in arena.units:
            unit.motor = CombatMotorArbiter(unit)
        for _ in range(15):
            arena.step(["IDLE", "IDLE"], [[0., 0.], [0., 0.]])
            assert arena.contacts_between_flies() == 0
        assert arena.max_interfly_penetration_mm == 0.
        touching = set()
        peak_force = 0.
        wrench = np.zeros(6)
        for _ in range(65):
            arena.step(["FENCE", "FENCE"], [[0., 0.], [0., 0.]])
            assert all(unit.upright > .98 and unit.ground_support >= 4 for unit in arena.units)
            for index, contact in enumerate(arena.sim.mj_data.contact):
                if (contact.exclude or contact.dist > .001 or
                        not arena.is_interfly_contact(int(contact.geom1), int(contact.geom2))):
                    continue
                names = [arena.sim.mj_model.geom(gid).name
                         for gid in (contact.geom1, contact.geom2)]
                if all("/lf_" in name or "/rf_" in name for name in names):
                    touching.add(tuple(names))
                    mj.mj_contactForce(arena.sim.mj_model, arena.sim.mj_data, index, wrench)
                    peak_force = max(peak_force, np.linalg.norm(wrench[:3]))
        assert touching, "The raised forelegs must actually meet the opponent's forelegs"
        assert peak_force > 0., "Geometric proximity alone is not a physical tap"
        assert arena.max_interfly_penetration_mm < .005
    finally:
        arena.close()


def test_fence_motor_ablation_support_guard_and_restored_phase(monkeypatch):
    from fly_duel.arena import DuelArena, FlyUnit
    from fly_duel.combat_motor import CombatMotorArbiter

    arena = DuelArena()
    try:
        unit = arena.units[0]
        motor = unit.motor = CombatMotorArbiter(unit)
        for _ in range(15):
            arena.step(["IDLE", "IDLE"], [[0., 0.], [0., 0.]])
        motor.select("FENCE", motor_off=True)
        assert motor.owner == "IDLE" and not motor.can_fence()
        motor.select("FENCE", motor_off=False)
        assert motor.owner == "FENCE"
        motor.elapsed = .315
        snapshot = motor.get_state()
        motor.update(np.zeros(2))
        expected = motor.get_state()
        ctrl = arena.sim.mj_data.ctrl.copy()
        motor.select("IDLE")
        motor.set_state(snapshot)
        motor.update(np.zeros(2))
        _equal_state(expected, motor.get_state())
        np.testing.assert_array_equal(ctrl, arena.sim.mj_data.ctrl)
        monkeypatch.setattr(FlyUnit, "ground_support", property(lambda self: 2))
        assert not motor.can_fence()
        motor.update(np.zeros(2))
        assert motor.owner == "IDLE" and not motor.ingestion_enabled
        motor.select("FENCE")
        assert motor.owner == "IDLE"
    finally:
        arena.close()
