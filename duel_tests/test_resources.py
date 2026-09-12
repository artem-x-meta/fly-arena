import copy
import math

import pytest

from fly_arena.environment import Environment
from fly_arena.organism import Organism
from fly_arena.sensors import PhysicalEvents
from fly_duel.resources import apply_shared_events


def world(amount=.03, **config):
    return Environment([dict(id="food", x=0, y=0, amount=amount, energy_density=12)], config)


def contact(duration=.01):
    return PhysicalEvents(contact_s_by_food={"food": duration})


def test_symmetric_last_drop_and_conserved_energy_composition():
    env, flies = world(), [Organism(), Organism()]
    results = apply_shared_events(env, [contact(), contact()], flies, .01)
    assert [row["intake_by_food"]["food"] for row in results] == [.015, .015]
    assert [fly.state.gut_amount for fly in flies] == [.015, .015]
    assert [fly.state.gut_food_energy for fly in flies] == pytest.approx([.18, .18])
    assert env.food[0].amount == 0
    assert env.ingested_food == .03
    assert env.last_transfers["intake_by_food"] == {"food": .03}


def test_reversing_contestants_keeps_individual_allocations_identical():
    def trial(reverse):
        env = world(amount=.0237)
        flies = [Organism({"intake_rate": 1.3}), Organism({"intake_rate": 2.7})]
        events = [contact(.007), contact(.009)]
        apply_shared_events(env, events[::-1] if reverse else events,
                            flies[::-1] if reverse else flies, .01)
        return [fly.get_state() for fly in flies], env.get_state()
    assert trial(False) == trial(True)


def test_shortage_is_shared_in_proportion_to_actual_permitted_contact():
    env, flies = world(amount=.024), [Organism(), Organism()]
    apply_shared_events(env, [contact(.01), contact(.005)], flies, .01)
    assert [fly.state.ingested_total for fly in flies] == pytest.approx([.016, .008])
    assert env.food[0].amount >= 0
    assert math.fsum(fly.state.ingested_total for fly in flies) == pytest.approx(env.initial_food)


def test_gut_capacity_bounds_demand_before_proportional_allocation():
    env = world(amount=.015)
    flies = [Organism({"gut_capacity": 1}, {"gut_amount": .995}), Organism()]
    apply_shared_events(env, [contact(), contact()], flies, .01)
    assert [fly.state.ingested_total for fly in flies] == pytest.approx([.003, .012])
    assert flies[0].state.gut_amount <= 1


def test_multiple_patches_follow_environment_order_with_remaining_gut_capacity():
    env = Environment([dict(id=key, x=0, y=0, amount=.04) for key in ("a", "b")])
    flies = [Organism({"gut_capacity": .03}), Organism({"gut_capacity": .03})]
    events = [PhysicalEvents(contact_s_by_food={"b": .01, "a": .01}) for _ in flies]
    results = apply_shared_events(env, events, flies, .02)
    for result in results:
        assert list(result["intake_by_food"]) == ["a", "b"]
        assert result["intake_by_food"] == pytest.approx({"a": .02, "b": .01})
    assert [patch.amount for patch in env.food] == pytest.approx([0, .02])


@pytest.mark.parametrize("case", ["disabled", "raw_contact_only", "empty", "absent", "unknown"])
def test_no_intake_without_resource_and_permitted_mouth_contact(case):
    env, flies = world(), [Organism(), Organism()]
    events = [contact(), contact()]
    if case == "disabled":
        env.config["mouth_contact_enabled"] = False
    elif case == "raw_contact_only":
        events = [PhysicalEvents(mouth_contact_s_by_food={"food": .01}) for _ in flies]
    elif case == "empty":
        env = world(amount=0)
    elif case == "absent":
        env = Environment()
    else:
        events = [PhysicalEvents(contact_s_by_food={"missing": .01}) for _ in flies]
    initial = env.initial_food
    results = apply_shared_events(env, events, flies, .01)
    assert all(result["intake_by_food"] == {} for result in results)
    assert all(fly.state.ingested_total == 0 for fly in flies)
    assert env.ingested_food == 0
    assert sum(patch.amount for patch in env.food) == initial


@pytest.mark.parametrize("bad", [
    PhysicalEvents(contact_s_by_food={"food": -.001}),
    PhysicalEvents(contact_s_by_food={"food": float("nan")}),
    PhysicalEvents(contact_s_by_food={"food": float("inf")}),
    PhysicalEvents(contact_s_by_food={"food": .011}),
    PhysicalEvents(contact_s_by_food={"food": .006, "other": .006}),
    PhysicalEvents(grooming_sliding_by_pair={"front_left|head": -.1}),
    PhysicalEvents(grooming_sliding_by_pair={"front_left|head": .1},
                   grooming_contact_s_by_pair={"front_left|head": .02}),
    PhysicalEvents(grooming_contact_s_by_pair={"front_left|head": float("nan")}),
    PhysicalEvents(mouth_contact_s_by_food={"food": float("inf")}),
    PhysicalEvents(grooming_sliding_by_pair={"front_left|unknown": .1}),
])
def test_invalid_second_event_rejects_complete_batch_before_any_mutation(bad):
    env, flies = world(), [Organism(), Organism()]
    before = copy.deepcopy((env.get_state(), [fly.get_state() for fly in flies]))
    with pytest.raises(ValueError):
        apply_shared_events(env, [contact(), bad], flies, .01)
    assert (env.get_state(), [fly.get_state() for fly in flies]) == before


@pytest.mark.parametrize("dt", [0, -.01, float("nan"), float("inf")])
def test_invalid_interval_does_not_initialize_or_mutate_ledgers(dt):
    env, flies = world(), [Organism(), Organism()]
    before = env.get_state()
    with pytest.raises(ValueError):
        apply_shared_events(env, [contact(), contact()], flies, dt)
    assert env.get_state() == before


def test_grooming_and_deposition_use_existing_rules_with_pooled_dust_ledger():
    env = world(dust_deposition_per_life_s=.1)
    flies = [Organism(initial={"dust_by_region": {"head": dust}}) for dust in (1., 2.)]
    event = PhysicalEvents(contact_s_by_food={"food": .01},
                           grooming_sliding_by_pair={"front_left|head": .01},
                           grooming_contact_s_by_pair={"front_left|head": .01})
    for _ in range(3):
        results = apply_shared_events(env, [event, event], flies, .01)
        assert [result["cleaned_by_pair"]["front_left|head"] for result in results] == [.008, .008]
    assert env.initial_dust == 3.
    remaining = math.fsum(value for fly in flies for value in fly.state.dust_by_region.values())
    assert remaining + env.removed_dust == pytest.approx(3. + env.added_dust)
    assert env.added_dust == pytest.approx(.006)
    assert env.ingested_food == pytest.approx(.03)


def test_scheduled_dust_respects_caller_initialized_pooled_baseline():
    env = Environment(events=[dict(time_s=0, kind="dust", region="head", amount=.4)])
    flies = [Organism(initial={"dust_by_region": {"head": dust}}) for dust in (1., 2.)]
    env.initial_dust = 3.
    env.apply_scheduled_events(0, flies[0])
    apply_shared_events(env, [PhysicalEvents(), PhysicalEvents()], flies, .01)
    assert env.initial_dust == 3.
    assert env.added_dust == .4
    assert sum(fly.state.dust_by_region["head"] for fly in flies) == 3.4


def test_mismatched_and_duplicate_contestants_reject_without_mutation():
    env, fly = world(), Organism()
    before = env.get_state()
    with pytest.raises(ValueError):
        apply_shared_events(env, [], [fly], .01)
    with pytest.raises(ValueError):
        apply_shared_events(env, [contact(), contact()], [fly, fly], .01)
    assert env.get_state() == before
    assert fly.state.ingested_total == 0
