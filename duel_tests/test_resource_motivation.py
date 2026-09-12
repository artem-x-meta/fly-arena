from types import SimpleNamespace

import pytest

from fly_arena.organism import Organism
from fly_arena.sensors import SensorFrame
from fly_duel.resource_motivation import ResourceMotivation


def fixture(*, energy=14., food=True, rival=True):
    return (ResourceMotivation(), SimpleNamespace(detected=rival),
            SensorFrame(mouth_taste=1. if food else 0.),
            Organism(initial={"energy": energy}))


def observe(model, sense, local, organism, seconds, *, dt=.01, action="FEED", intake=0.):
    result = []
    for _ in range(round(seconds / dt)):
        if intake:
            organism.consume("own-meal", min(intake * dt, organism.gut_free_capacity), 1.)
        result.append(model.step(sense, local, organism, dt=dt, base_action=action))
    return result


def test_initial_feeding_opportunity_prevents_self_created_conflict():
    model, sense, local, organism = fixture()
    assert not any(observe(model, sense, local, organism, 1.5))
    assert model.frustration == pytest.approx(0., abs=1e-12)
    assert not any(observe(model, sense, local, organism, .70))
    assert any(observe(model, sense, local, organism, .06))


def test_abundant_steady_intake_remains_peaceful_even_while_hungry():
    model, sense, local, organism = fixture()
    model.step(sense, local, organism)
    assert not any(observe(model, sense, local, organism, 4., intake=1.))
    assert organism.hunger > .25
    assert model.own_intake_ema == pytest.approx(1., abs=.001)
    assert model.frustration == 0.


def test_real_intermittent_meal_does_not_turn_every_missed_contact_into_attack():
    model, sense, local, organism = fixture()
    model.step(sense, local, organism)
    for _ in range(10):
        assert not any(observe(model, sense, local, organism, .1, intake=2.))
        assert not any(observe(model, sense, local, organism, .3))
    assert model.frustration == 0.


def test_interrupted_hungry_meal_can_trigger_conflict_after_a_delay():
    model, sense, local, organism = fixture()
    model.step(sense, local, organism)
    assert not any(observe(model, sense, local, organism, 2., intake=1.))
    local.mouth_taste = 0.  # depleted patch no longer tastes of available food
    assert not any(observe(model, sense, local, organism, .5))
    assert any(observe(model, sense, local, organism, 1.5))
    assert model.reason == "own_access_failed_near_rival"
    organism.consume("own-meal", .02, 1.)
    assert not model.step(sense, local, organism, base_action="FEED")
    assert model.frustration == 0.


@pytest.mark.parametrize("options", [dict(food=False), dict(rival=False), dict(energy=100.)])
def test_resource_rival_and_own_hunger_are_all_required(options):
    args = fixture(**options)
    assert not any(observe(*args, 10.))
    assert args[0].frustration == 0.


def test_full_gut_suppresses_aggression_before_energy_is_assimilated():
    model, sense, local, organism = fixture()
    organism.consume("previous-meal", organism.gut_free_capacity, 1.)
    assert not any(observe(model, sense, local, organism, 4.))
    assert model.reason == "own_need_satisfied"


def test_no_new_evidence_means_resource_memory_and_frustration_expire():
    model, sense, local, organism = fixture()
    assert any(observe(model, sense, local, organism, 3.))
    local.mouth_taste = 0.
    observe(model, sense, local, organism, 8.1)
    assert model.recent_resource_s == 0.
    assert model.frustration == 0.
    assert not model.can_engage


def test_lost_rival_stops_attack_and_recovers_without_reading_its_state():
    model, sense, local, organism = fixture()
    assert any(observe(model, sense, local, organism, 3.))
    sense.detected = False
    assert not any(observe(model, sense, local, organism, 1.1))
    assert model.frustration == 0.


@pytest.mark.parametrize("action", ["GROOM_HEAD", "IDLE", "SLEEP", "ESCAPE"])
def test_other_priorities_do_not_exhaust_feeding_opportunity(action):
    model, sense, local, organism = fixture()
    assert not any(observe(model, sense, local, organism, 4., action=action))
    assert model.opportunity_remaining_s == 1.5
    assert not any(observe(model, sense, local, organism, 1.5, action="FEED"))


def test_equal_local_histories_ignore_hidden_resource_and_rival_budgets():
    a, sa, la, oa = fixture()
    b, sb, lb, ob = fixture()
    # Deliberately contradictory irrelevant fields: the policy must not read
    # either these global facts or the rival's needs to predict a winner.
    sa.food_remaining, sa.rival_hunger = 40., 0.
    sb.food_remaining, sb.rival_hunger = 4., 1.
    la.food_remaining, lb.food_remaining = 100., 0.
    oa.opponent_energy, ob.opponent_energy = 100., 0.
    for index in range(700):
        intake = .01 if 80 <= index < 250 else 0.
        if intake:
            oa.consume("own", intake, 1.)
            ob.consume("own", intake, 1.)
        la.mouth_taste = lb.mouth_taste = float(index < 250)
        assert a.step(sa, la, oa, base_action="FEED") == b.step(sb, lb, ob, base_action="FEED")
        assert a.summary() == b.summary()


def test_same_patch_with_later_depletion_changes_decisions_only_after_own_intake_fails():
    abundant, sa, la, oa = fixture()
    scarce, ss, ls, os = fixture()
    abundant.step(sa, la, oa)
    scarce.step(ss, ls, os)
    for _ in range(200):
        oa.consume("own", .01, 1.)
        os.consume("own", .01, 1.)
        assert abundant.step(sa, la, oa) == scarce.step(ss, ls, os)
    ls.mouth_taste = 0.
    scarcity_responses = []
    for _ in range(200):
        oa.consume("own", .01, 1.)
        assert not abundant.step(sa, la, oa)
        scarcity_responses.append(scarce.step(ss, ls, os))
    assert any(scarcity_responses)


def test_historical_intake_is_not_mistaken_for_a_current_meal():
    model, sense, local, organism = fixture()
    organism.state.ingested_total = 100.
    model.step(sense, local, organism)
    assert model.own_intake_ema == 0.
    assert model.intake_hold_s == 0.


def test_physical_interval_changes_resolution_not_opportunity_duration():
    first_engagements = []
    for dt in (.005, .01, .025):
        model, sense, local, organism = fixture()
        for index in range(round(3. / dt)):
            if model.step(sense, local, organism, dt=dt, base_action="FEED"):
                first_engagements.append((index + 1) * dt)
                break
    assert first_engagements == pytest.approx([2.25] * 3, abs=.025)


@pytest.mark.parametrize("dt", [0., -1., float("nan"), float("inf")])
def test_invalid_dt_rejected_before_changing_state(dt):
    model, sense, local, organism = fixture()
    before = model.summary()
    with pytest.raises(ValueError, match="positive and finite"):
        model.step(sense, local, organism, dt=dt)
    assert model.summary() == before
