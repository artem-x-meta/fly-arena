import json

import pytest

from fly_arena.organism import Organism, OrganismConfig, SimulationClocks


def test_food_composition_and_mass_energy_ledgers():
    fly = Organism(OrganismConfig(digestion_rate=1, basal_metabolic_rate=.1,
                                 assimilation_efficiency=.75), {"energy": 12})
    fly.consume("nutritious", 2, 8)
    fly.consume("noncaloric", 3, 0)
    first = fly.advance(1, "IDLE")
    assert first.assimilated_energy == 6
    fly.advance(4, "IDLE")
    assert fly.state.gut_amount == 0
    assert fly.state.ingested_total == fly.state.digested_total == 5
    assert fly.state.digested_food_energy_total == 16
    assert fly.state.assimilated_energy_total == 12
    assert fly.state.assimilation_loss_total == 4
    assert 12 + fly.state.assimilated_energy_total == pytest.approx(
        fly.state.energy + fly.state.metabolic_spent_total + fly.state.energy_overflow_total)


def test_full_gut_reduces_hunger_before_digestion_and_overfill_is_atomic():
    fly = Organism(initial={"energy": 10})
    hunger = fly.hunger
    fly.consume("food", 9, 10)
    assert fly.hunger == pytest.approx(hunger / 10)
    before = fly.get_state()
    with pytest.raises(ValueError, match="capacity"):
        fly.advance(.1, "FEED", [("a", .5, 10), ("b", 1, 0)])
    assert fly.get_state() == before


def test_sleep_does_not_refill_energy_but_digestion_can():
    empty = Organism(initial={"energy": 20, "sleep_pressure": .9})
    full = Organism(initial={"energy": 20, "sleep_pressure": .9,
                             "gut_amount": 2, "gut_energy_density": 20})
    empty.advance(10, "SLEEP")
    full.advance(10, "SLEEP")
    assert empty.state.energy < 20 < full.state.energy
    assert empty.state.sleep_pressure == pytest.approx(full.state.sleep_pressure)
    assert empty.state.sleep_pressure < .9


def test_nonnutritive_food_starvation_and_overflow_are_accounted():
    fly = Organism(OrganismConfig(digestion_rate=10, basal_metabolic_rate=1), {"energy": .1})
    fly.advance(1, "IDLE", [("sweet_zero", 3, 0)])
    assert fly.exhausted and fly.state.energy == 0
    assert fly.state.metabolic_spent_total == .1
    assert fly.state.metabolic_unmet_total == .9
    fly.advance(100, "SLEEP")
    assert fly.state.energy == 0
    fly.advance(1, "FEED", [("caloric", 2, 100)])
    assert fly.state.energy == fly.config.energy_capacity
    assert fly.state.energy_overflow_total > 0
    assert .1 + fly.state.assimilated_energy_total == pytest.approx(
        fly.state.energy + fly.state.metabolic_spent_total + fly.state.energy_overflow_total)


def test_life_scaling_does_not_double_scale_ingestion_or_neural_time():
    fly = Organism(OrganismConfig(life_time_scale=30, digestion_rate=0), {"energy": 50})
    fly.advance(.1, "FEED", [("food", .2, 5)])
    assert fly.state.gut_amount == .2
    assert fly.clocks.physics_time_s == fly.clocks.neural_time_s == .1
    assert fly.clocks.life_time_s == 3
    before = fly.get_state()
    fly.advance(100, "FEED", [("food", 100, 5)], paused=True)
    assert fly.get_state() == before
    with pytest.raises(ValueError, match="agree"):
        SimulationClocks().advance(.1, .2)


def test_long_sleep_and_wake_steps_remain_bounded():
    fly = Organism()
    fly.advance(1e7, "SLEEP")
    assert fly.state.sleep_pressure == 0
    fly.advance(1e7, "SLEEP_ENTRY")
    assert fly.state.sleep_pressure == 1
    assert 0 <= fly.state.circadian_phase < 1


def test_json_resume_keeps_composition_ledgers_and_clocks_exactly():
    cfg = OrganismConfig(life_time_scale=5)
    continuous = Organism(cfg, {"energy": 11})
    continuous.advance(.17, "FEED", [("a", 1, 4), ("b", 2, 0)])
    resumed = Organism(cfg)
    resumed.set_state(json.loads(json.dumps(continuous.get_state())))
    for fly in (continuous, resumed):
        fly.advance(.31, "SLEEP")
        fly.advance(.12, "WALK", [("c", .2, 13)], movement_proxy=.7)
    assert resumed.get_state() == continuous.get_state()


@pytest.mark.parametrize("initial", [{"energy": float("nan")}, {"sleep_pressure": -1},
                                     {"gut_amount": 11}, {"dust_by_region": {"head": -1}}])
def test_invalid_initial_states_are_rejected(initial):
    with pytest.raises(ValueError):
        Organism(initial=initial)
