import numpy as np
import pytest

from fly_arena.environment import Environment
from fly_arena.organism import Organism
from fly_arena.sensors import PhysicalEvents


def test_recurring_refill_restarts_physical_food_flow_and_resume_does_not_replay():
    fly = Organism(initial={"energy": 10})
    env = Environment([dict(id="a", x=0, y=0, amount=.01)], config={"recurring_events": [
        dict(kind="refill", time_s=.02, repeat_s=.02, amount=.03, capacity=.1, food_id="a")]})
    contact = PhysicalEvents(contact_s_by_food={"a": .01})
    env.apply_physical_events(contact, fly, .01)
    assert env.food[0].amount == 0
    env.apply_physical_events(contact, fly, .01)
    assert fly.state.ingested_total == .01
    event = env.apply_scheduled_events(.02, fly)
    assert event[0]["amount_added"] == .03
    env.apply_physical_events(contact, fly, .01)
    assert fly.state.ingested_total == pytest.approx(.03)
    restored = Environment()
    restored.set_state(env.get_state())
    assert restored.apply_scheduled_events(.02, fly) == []
    assert len(restored.apply_scheduled_events(.04, fly)) == 1
    assert len(restored.recurring_events) == 1
    assert restored.check_balances(fly)["food_residual"] == pytest.approx(0)


def test_field_event_schedule_and_rng_survive_environment_checkpoint():
    food = [dict(id="a", x=0, y=0)]
    config = {"odor_model": "filaments", "recurring_events": [
        dict(kind="dust", time_s=.2, repeat_s=.2, region="head", amount=.1)]}
    env = Environment(food, config, seed=101)
    fly = Organism()
    for i in range(25):
        env.apply_scheduled_events(i * .01, fly)
        env.advance_field(.01)
    restored = Environment(food, config, seed=2)
    restored.set_state(env.get_state())
    other = Organism()
    other.set_state(fly.get_state())
    for i in range(25, 75):
        assert env.apply_scheduled_events(i * .01, fly) == restored.apply_scheduled_events(i * .01, other)
        env.advance_field(.01)
        restored.advance_field(.01)
        assert env.odor_at([-2, .3, 1]) == restored.odor_at([-2, .3, 1])
        np.testing.assert_array_equal(env.plume.packets, restored.plume.packets)
    assert fly.state.dust_by_region == other.state.dust_by_region
