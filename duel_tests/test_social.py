from types import SimpleNamespace

import numpy as np
import pytest

from fly_arena.organism import Organism
from fly_arena.sensors import SensorFrame
from fly_duel.combat_motor import FENCE_DURATION_S
from fly_duel.social import SocialController


def senses(*, food=True, seen=True, contact=False, bearing=0., heading=0., distance=1.):
    return (SimpleNamespace(detected=seen, bearing=bearing, distance_mm=distance if seen else None,
                            body_contact=contact, contact_impulse=.1 if contact else 0., own_heading=heading),
            SensorFrame(mouth_taste=1. if food else 0., upright=1., ground_support=6))


def failed_feeding_until_engagement(policy, sense, local, organism):
    """Local failure history: a FEED request alone has swallowed nothing."""
    for _ in range(350):
        command = policy.step(sense, local, organism, base_action="FEED")
        if command is not None:
            return command
    pytest.fail("Hungry fly with a local rival must react to sustained failed access")


def test_rival_and_local_resource_are_both_required():
    for food, rival in ((False, True), (True, False), (False, False)):
        policy = SocialController()
        sense, local = senses(food=food, seen=rival)
        organism = Organism(initial={"energy": 14.})
        for _ in range(400):
            assert policy.step(sense, local, organism) is None
        assert policy.counters["pushes"] == policy.counters["fences"] == 0


def test_sated_fly_does_not_defend_food():
    sense, local = senses()
    policy, organism = SocialController(), Organism(initial={"energy": 100.})
    for _ in range(400):
        assert policy.step(sense, local, organism) is None


def test_turn_then_push_and_channel_off_cancels():
    organism, policy = Organism(initial={"energy": 14.}), SocialController()
    # Outside foreleg range, this exercises orientation and forward motion.
    sense, local = senses(bearing=1., distance=3.)
    command = failed_feeding_until_engagement(policy, sense, local, organism)
    assert command.state == "ORIENT" and command.gait[1] > command.gait[0]
    sense.bearing = 0.
    assert policy.step(sense, local, organism).state == "PUSH"
    assert policy.step(sense, local, organism, enabled=False) is None
    assert policy.state == "NONE"


def test_contact_effort_causes_retreat_then_returns_control():
    organism, policy = Organism(initial={"energy": 14.}), SocialController(seed=101)
    sense, local = senses(contact=True)
    failed_feeding_until_engagement(policy, sense, local, organism)
    for _ in range(250):
        command = policy.step(sense, local, organism)
        if command and command.state == "RETREAT":
            break
    assert policy.state == "RETREAT"
    sense.own_heading = policy.retreat_heading
    sense.detected, sense.body_contact = False, False
    for _ in range(205):
        command = policy.step(sense, local, organism)
    assert policy.state in {"RETURN", "RECOVER"}
    sense.own_heading = policy.resource_heading
    command = policy.step(sense, local, organism)
    assert policy.state == "RECOVER" and command is None
    assert policy.counters["retreats"] == 1


def test_lost_rival_stops_pursuit_and_food_memory_expires():
    organism, policy = Organism(initial={"energy": 14.}), SocialController()
    sense, local = senses()
    failed_feeding_until_engagement(policy, sense, local, organism)
    sense.detected, local.mouth_taste = False, 0.
    assert policy.step(sense, local, organism) is None
    for _ in range(810):
        policy.step(sense, local, organism)
    sense.detected = True
    assert policy.step(sense, local, organism) is None
    assert policy.motivation.recent_resource_s == 0.


def test_physical_and_life_priorities_win():
    organism = Organism(initial={"energy": 14.})
    for action in ("SLEEP", "SLEEP_ENTRY", "WAKE", "EXHAUSTED", "ESCAPE"):
        sense, local = senses()
        policy = SocialController()
        failed_feeding_until_engagement(policy, sense, local, organism)
        assert policy.step(sense, local, organism, base_action=action) is None
    sense, local = senses()
    policy = SocialController()
    failed_feeding_until_engagement(policy, sense, local, organism)
    local.upright = .5
    assert policy.step(sense, local, organism) is None
    local.upright, local.ground_support = 1., 2
    assert policy.step(sense, local, organism) is None


def test_same_seed_replays_social_decisions():
    organism = Organism(initial={"energy": 14.})
    a, b = SocialController(seed=9), SocialController(seed=9)
    sense, local = senses(contact=True)
    for _ in range(1000):
        assert a.step(sense, local, organism) == b.step(sense, local, organism)
    assert a.summary() == b.summary()


def test_food_pose_is_not_dragged_along_by_the_fight():
    organism, policy = Organism(initial={"energy": 14.}), SocialController()
    sense, local = senses()
    sense.own_position_mm = (-.9, 0., 1.)
    failed_feeding_until_engagement(policy, sense, local, organism)
    assert policy.state == "FENCE"
    sense.own_position_mm, sense.own_heading = (1., 1., 1.), .7
    policy.step(sense, local, organism)
    np.testing.assert_array_equal(policy.resource_position, [-.9, 0.])
    assert policy.resource_heading == 0.


def test_fresh_front_taste_ends_return_before_a_stale_waypoint():
    organism, policy = Organism(initial={"energy": 14.}), SocialController()
    sense, local = senses()
    policy.state = "RETURN"
    policy.resource_position = np.array([10., 10.])
    policy.resource_heading = np.pi
    assert policy.step(sense, local, organism) is None
    assert policy.state == "RECOVER"
    assert policy.last_reason == "local_food_reacquired"


def test_feeding_opportunity_and_successful_own_intake_prevent_attack():
    organism, policy = Organism(initial={"energy": 14.}), SocialController()
    sense, local = senses()
    for _ in range(150):
        assert policy.step(sense, local, organism, base_action="FEED") is None
    for _ in range(400):
        organism.consume("own-meal", .01, 1.)
        assert policy.step(sense, local, organism, base_action="FEED") is None
    assert organism.hunger > .25
    assert policy.counters["encounters"] == 0
    assert policy.summary()["resource_motivation"]["frustration"] == 0.


def test_near_rival_fence_is_one_bounded_bout_then_push_with_refractory():
    organism, policy = Organism(initial={"energy": 14.}), SocialController()
    sense, local = senses()
    command = failed_feeding_until_engagement(policy, sense, local, organism)
    assert command.state == "FENCE" and command.gait == (0., 0.)
    states = []
    for _ in range(round(FENCE_DURATION_S / .01) + 25):
        states.append(policy.step(sense, local, organism).state)
    assert "FENCE" in states and states[-1] == "PUSH"
    assert policy.counters["fences"] == 1
    assert policy.durations["FENCE"] == pytest.approx(FENCE_DURATION_S, abs=.01)


def test_fencing_entry_requires_four_grounded_legs():
    organism, policy = Organism(initial={"energy": 14.}), SocialController()
    sense, local = senses()
    local.ground_support = 3
    assert failed_feeding_until_engagement(policy, sense, local, organism).state == "PUSH"
    assert policy.counters["fences"] == 0
    local.ground_support = 4
    assert policy.step(sense, local, organism).state == "FENCE"
    local.ground_support = 2
    assert policy.step(sense, local, organism) is None


@pytest.mark.parametrize("bearing,distance,state", [(1., 3., "ORIENT"),
                                                      (0., 3., "PUSH"), (0., 1., "FENCE")])
def test_real_intake_cancels_each_active_engagement(bearing, distance, state):
    organism, policy = Organism(initial={"energy": 14.}), SocialController()
    sense, local = senses(bearing=bearing, distance=distance)
    assert failed_feeding_until_engagement(policy, sense, local, organism).state == state
    organism.consume("own-meal", .02, 1.)
    assert policy.step(sense, local, organism, base_action="FEED") is None
    assert policy.state == "NONE"
