"""Real MuJoCo interventions; opt in with RUN_PHYSICS_TEST=1.

Thresholds target this deterministic motor-validation scene, not physiology.
"""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_PHYSICS_TEST") != "1",
                                reason="Requires a working MuJoCo OpenGL context")


def food():
    return dict(id="test", x=.9, y=0., radius=2., surface_z=.01,
                amount=5., energy_density=10., taste=1., odor=1.)


def test_extended_walk_has_one_owner_and_exact_same_instance_resume():
    from fly_arena.ethology_arena import EthologyArena
    arena = EthologyArena()
    try:
        start = arena.position.copy()
        for _ in range(30):
            arena.step([.8, .8])
        assert len(arena.motor.order) == 44
        assert len(arena.motor.leg_indices) == 42
        assert np.linalg.norm(arena.position[:2] - start[:2]) > .1
        assert arena.upright > .85
        checkpoint = arena.get_state()
        for _ in range(5):
            arena.step([.8, .8])
        expected = arena.get_state()["integration"]
        arena.set_state(checkpoint)
        for _ in range(5):
            arena.step([.8, .8])
        np.testing.assert_array_equal(arena.get_state()["integration"], expected)
        for action in ("FEED", "GROOM_FRONT", "GROOM_HEAD", "SLEEP", "WALK"):
            for _ in range(65):
                arena.step_behavior(action, [.6, .6])
                assert arena.upright > .80
                assert arena.motor.owner == action
        assert np.isfinite(arena.sim.mj_data.qpos).all()
    finally:
        arena.close()


@pytest.mark.parametrize("blocked", [False, True])
def test_food_transfer_requires_actuated_mouth_contact(blocked):
    from fly_arena.ethology_arena import EthologyArena
    from fly_arena.environment import Environment
    from fly_arena.organism import Organism
    arena = EthologyArena(food_patches=[food()], mouth_contact_enabled=not blocked)
    organism, environment = Organism(initial={"energy": 20.}), Environment([food()])
    try:
        for _ in range(110):
            events = arena.step_behavior("FEED", [0., 0.])
            environment.apply_physical_events(events, organism, .01)
        assert arena.upright > .9
        assert arena.motor.ingestion_enabled
        assert abs(arena.mouth_position[2] - .01) < arena.mouth_contact_thickness
        if blocked:
            assert organism.state.gut_amount == 0.
            assert environment.food[0].amount == 5.
        else:
            assert organism.state.gut_amount > .5
            assert environment.food[0].amount < 4.5
        assert abs(environment.food[0].amount + organism.state.gut_amount - 5.) < 1e-10
    finally:
        arena.close()


@pytest.mark.parametrize("action", ["GROOM_FRONT", "GROOM_HEAD"])
@pytest.mark.parametrize("blocked", [False, True])
def test_grooming_removes_dust_only_with_real_contact_and_sliding(action, blocked):
    from fly_arena.ethology_arena import EthologyArena
    from fly_arena.environment import Environment
    from fly_arena.organism import Organism
    from fly_arena.sensors import DUST_REGIONS
    arena = EthologyArena(grooming_contacts_enabled=not blocked)
    dust = dict.fromkeys(DUST_REGIONS, 0.)
    if action == "GROOM_FRONT":
        dust["front_left"] = dust["front_right"] = 1.
    else:
        dust["head"] = dust["antenna_left"] = dust["antenna_right"] = 1.
    organism = Organism(initial={"dust_by_region": dust})
    environment = Environment()
    travel = 0.
    try:
        for _ in range(130):
            events = arena.step_behavior(action, [0., 0.])
            travel += sum(events.grooming_sliding_by_pair.values())
            environment.apply_physical_events(events, organism, .01)
            assert arena.upright > .9
        if blocked:
            assert travel == 0.
            assert organism.state.dust_by_region == dust
            assert environment.removed_dust == 0.
        else:
            assert travel > .05
            assert environment.removed_dust > .01
            if action == "GROOM_HEAD":
                assert organism.state.dust_by_region["head"] < .95
                assert organism.state.dust_by_region["antenna_left"] < .95
                assert organism.state.dust_by_region["antenna_right"] < .95
                assert organism.state.dust_by_region["front_left"] > .01
        assert abs(sum(organism.state.dust_by_region.values()) + environment.removed_dust - sum(dust.values())) < 1e-9
    finally:
        arena.close()


@pytest.mark.parametrize("action", ["FEED", "GROOM_FRONT", "GROOM_HEAD"])
def test_motor_resume_in_fresh_model_without_warmup(action):
    from fly_arena.ethology_arena import EthologyArena
    original = EthologyArena(food_patches=[food()])
    restored = None
    try:
        for _ in range(75):
            original.step_behavior(action, [0., 0.])
        checkpoint = original.get_state()
        events_original = [original.step_behavior(action, [0., 0.]) for _ in range(10)]
        restored = EthologyArena(food_patches=[food()], warmup=False)
        restored.set_state(checkpoint)
        events_restored = [restored.step_behavior(action, [0., 0.]) for _ in range(10)]
        # MuJoCo serializes no derived data: the uninterrupted run reads site
        # poses one 0.1 ms substep stale, a restored run reads them refreshed by
        # mj_forward, and the inverse-kinematic primitives feed that difference
        # back into the targets. Physical events stay exact; the continuous
        # state carries the declared resume tolerance, whose absolute part
        # scales with the vector because the same array holds near-zero solver
        # warm-start terms next to physical coordinates.
        expected = original.get_state()["integration"]
        scale = max(1., float(np.max(np.abs(expected))))
        np.testing.assert_allclose(restored.get_state()["integration"], expected,
                                   atol=1e-7 * scale, rtol=1e-6)
        assert events_original == events_restored
    finally:
        original.close()
        if restored is not None:
            restored.close()


def test_runtime_contact_ablation_has_zero_constraint_force_and_resumes():
    import mujoco as mj
    from fly_arena.ethology_arena import EthologyArena
    arena = EthologyArena(grooming_contacts_enabled=False)
    try:
        assert len(arena._groom_pair_ids) == 7
        for _ in range(75):
            events = arena.step_behavior("GROOM_HEAD", [0., 0.])
            assert not events.grooming_contact_s_by_pair
            for index, contact in enumerate(arena.sim.mj_data.contact):
                if frozenset((int(contact.geom1), int(contact.geom2))) in arena.contact_pair_labels:
                    force = np.zeros(6)
                    mj.mj_contactForce(arena.sim.mj_model, arena.sim.mj_data, index, force)
                    assert contact.exclude == 1
                    assert contact.efc_address == -1
                    np.testing.assert_array_equal(force, np.zeros(6))
        arena.grooming_contacts_enabled = True
        contact_s = 0.
        for _ in range(40):
            events = arena.step_behavior("GROOM_HEAD", [0., 0.])
            contact_s += sum(events.grooming_contact_s_by_pair.values())
        assert contact_s > .1
        assert arena.max_groom_contact_force > 0.
        checkpoint = arena.get_state()
        expected_events = [arena.step_behavior("GROOM_HEAD", [0., 0.]) for _ in range(5)]
        expected = arena.get_state()
        arena.grooming_contacts_enabled = False
        arena.mouth_contact_enabled = False
        arena.mouth_contact_thickness = .001
        arena.step_behavior("GROOM_HEAD", [0., 0.])
        arena.set_state(checkpoint)
        assert arena.grooming_contacts_enabled
        assert arena.mouth_contact_enabled
        assert arena.mouth_contact_thickness == .04
        assert arena.max_groom_contact_force == checkpoint["max_groom_contact_force"]
        actual_events = [arena.step_behavior("GROOM_HEAD", [0., 0.]) for _ in range(5)]
        np.testing.assert_array_equal(arena.get_state()["integration"], expected["integration"])
        np.testing.assert_array_equal(arena.get_state()["pair_margin"], expected["pair_margin"])
        assert actual_events == expected_events
    finally:
        arena.close()
