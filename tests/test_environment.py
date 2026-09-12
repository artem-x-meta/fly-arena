import pytest

from fly_arena.environment import Environment
from fly_arena.organism import Organism
from fly_arena.sensors import PhysicalEvents


def organism(**kwargs):
    return Organism({"life_time_scale": 60, "digestion_rate": .02,
                     "basal_metabolic_rate": 0, "movement_metabolic_rate": 0},
                    {"energy": 10, **kwargs})


def test_physical_intake_is_single_flow_and_blocked_mouth_cannot_ingest():
    for enabled in (True, False):
        fly = organism()
        env = Environment([dict(id="a", x=0, y=0, amount=.07, energy_density=10)],
                          {"mouth_contact_enabled": enabled})
        for _ in range(10):
            env.apply_physical_events(PhysicalEvents(contact_s_by_food={"a": .01}), fly, .01)
            fly.advance(.01, "FEED")
        expected = .07 if enabled else 0
        assert fly.state.ingested_total == pytest.approx(expected)
        assert env.food[0].amount == pytest.approx(.07 - expected)
        assert fly.state.gut_amount + fly.state.digested_total == pytest.approx(expected)
        assert fly.state.energy == pytest.approx(10 + .8 * 10 * expected)


def test_life_scale_does_not_multiply_intake_and_zero_calorie_has_no_energy():
    fly = organism()
    env = Environment([dict(id="a", x=0, y=0, amount=1, energy_density=0)])
    env.apply_physical_events(PhysicalEvents(contact_s_by_food={"a": .01}), fly, .01)
    fly.advance(.01, "FEED")
    assert fly.state.ingested_total == pytest.approx(.02)
    assert fly.state.energy == 10


def test_taste_and_smell_are_not_ingestion_events():
    fly = organism()
    env = Environment([dict(id="a", x=0, y=0, amount=0, odor=2, odor_when_empty=True)])
    assert env.odor_at([0, 0, .01]) == 2
    assert env.taste_at([0, 0, .01]) == 0
    env.apply_physical_events(PhysicalEvents(), fly, .01)
    assert fly.state.gut_amount == 0
    # Physical contact while not feeding also cannot fill the gut.
    env.apply_physical_events(PhysicalEvents(mouth_contact_s_by_food={"a": .01}), fly, .01)
    assert fly.state.gut_amount == 0


def test_dust_requires_sliding_and_contact_and_conserves_transfers():
    for enabled in (True, False):
        fly = organism(dust_by_region={"head": 1})
        env = Environment(config={"cleaning_enabled": enabled})
        env.apply_physical_events(PhysicalEvents(grooming_sliding_by_pair={"front_left|head": 1}), fly, .01)
        assert fly.state.dust_by_region["head"] == 1
        event = PhysicalEvents(grooming_sliding_by_pair={"front_left|head": .01},
                               grooming_contact_s_by_pair={"front_left|head": .01})
        env.apply_physical_events(event, fly, .01)
        if enabled:
            assert fly.state.dust_by_region["head"] < 1
            assert fly.state.dust_by_region["front_left"] > 0
            assert env.removed_dust > 0
        else:
            assert fly.state.dust_by_region["head"] == 1
        assert sum(fly.state.dust_by_region.values()) + env.removed_dust == pytest.approx(1)
        env.apply_physical_events(PhysicalEvents(grooming_sliding_by_pair={"front_left|front_right": .1},
                                                grooming_contact_s_by_pair={"front_left|front_right": .01}), fly, .01)
        assert sum(fly.state.dust_by_region.values()) + env.removed_dust == pytest.approx(1)


def test_events_resume_without_replaying_refill_or_dust():
    fly = organism()
    env = Environment([dict(id="a", x=0, y=0, amount=0)], events=[
        dict(time_s=0, kind="refill", food_id="a", amount=2),
        dict(time_s=0, kind="dust", region="head", amount=.4)])
    assert len(env.apply_scheduled_events(0, fly)) == 2
    saved = env.get_state()
    restored = Environment()
    restored.set_state(saved)
    assert restored.apply_scheduled_events(0, fly) == []
    assert restored.food[0].amount == 2
    assert restored.check_balances(fly) == {"food_residual": 0, "dust_residual": 0}


def test_overlapping_food_cannot_create_extra_contact_time():
    env = Environment()
    with pytest.raises(ValueError, match="exceeds"):
        env.apply_physical_events(PhysicalEvents(contact_s_by_food={"a": .01, "b": .01}), organism(), .01)
