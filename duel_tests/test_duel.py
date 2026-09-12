"""Short real-physics checks of social policy integration and ordinary life."""
import copy
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_PHYSICS_TEST") != "1",
                                reason="Requires a working MuJoCo OpenGL context")


def _check_clocks_and_food(duel):
    assert duel.arena.sim.mj_data.time == pytest.approx(duel.elapsed_s, abs=1e-10)
    assert duel.arena.physics_steps == duel.steps * 100
    for contestant in duel.contestants:
        organism = contestant.organism
        assert organism.clocks.physics_time_s == pytest.approx(duel.elapsed_s)
        assert organism.clocks.neural_time_s == pytest.approx(duel.elapsed_s)
        assert organism.clocks.life_time_s == pytest.approx(
            duel.elapsed_s * organism.config.life_time_scale)
        assert organism.state.gut_amount + organism.state.digested_total == pytest.approx(
            organism.state.ingested_total, abs=1e-10)
    env = duel.environment
    assert sum(c.organism.state.ingested_total for c in duel.contestants) == pytest.approx(
        env.ingested_food, abs=1e-10)
    assert sum(p.amount for p in env.food) + env.ingested_food == pytest.approx(
        env.initial_food + env.refilled_food - env.withdrawn_food, abs=1e-10)


def test_disabled_social_sensing_matches_passive_physics_and_both_named_flies_feed():
    from fly_duel.duel import Duel, encounter_scene

    config, starts, headings = encounter_scene()
    passive = Duel(copy.deepcopy(config), seed=101, start=starts, start_headings=headings,
                   names=("amber", "cyan"), social=False)
    blind = None
    try:
        blind = Duel(copy.deepcopy(config), seed=101, start=starts, start_headings=headings,
                     names=("amber", "cyan"), social=True, social_sensing=False)
        for _ in range(100):
            passive.step()
            blind.step()
            np.testing.assert_array_equal(passive.arena.sim.mj_data.qpos, blind.arena.sim.mj_data.qpos)
            np.testing.assert_array_equal(passive.arena.sim.mj_data.qvel, blind.arena.sim.mj_data.qvel)
            np.testing.assert_array_equal(passive.arena.sim.mj_data.ctrl, blind.arena.sim.mj_data.ctrl)
            assert passive.last_transfers == blind.last_transfers
            for first, second in zip(passive.contestants, blind.contestants):
                assert first.organism.get_state() == second.organism.get_state()
                assert first.behavior.get_state() == second.behavior.get_state()
                assert first.social_decision is None and second.social_decision is None
        _check_clocks_and_food(passive)
        _check_clocks_and_food(blind)
        # A non-default namespace and a rotated second body exercise each
        # animal's own joint addresses and independently constructed mouth IK.
        assert all(c.organism.state.ingested_total > .05 for c in passive.contestants)
        assert all(c.unit.motor.owner == "FEED" for c in passive.contestants)
        assert not set(passive.contestants[0].unit.motor.qpos_ids) & set(
            passive.contestants[1].unit.motor.qpos_ids)
    finally:
        passive.close()
        if blind is not None:
            blind.close()


def test_social_mode_first_allows_both_flies_to_feed_and_obeys_support_guard(monkeypatch):
    from fly_duel.duel import Duel, encounter_scene

    config, starts, headings = encounter_scene()
    duel = Duel(config, seed=101, start=starts, start_headings=headings, social=True)
    try:
        for _ in range(100):
            duel.step()
            assert all(contestant.social_decision is None for contestant in duel.contestants)
        assert duel.arena.collision_geometry == "mesh"
        assert all(c.organism.state.ingested_total > .05 for c in duel.contestants)
        assert all(c.unit.motor.owner == "FEED" for c in duel.contestants)
        _check_clocks_and_food(duel)

        original_sense = duel.environment.sense

        def unsupported_sense(*args, **kwargs):
            frame = original_sense(*args, **kwargs)
            frame.ground_support = 0
            return frame

        monkeypatch.setattr(duel.environment, "sense", unsupported_sense)
        duel.step()
        for contestant in duel.contestants:
            assert contestant.frame.ground_support == 0
            assert not contestant.decision.gates["stable"]
            assert contestant.social_decision is None
            assert contestant.decision.action == "IDLE"
            assert contestant.unit.motor.owner == "IDLE"
    finally:
        duel.close()


def test_depleted_resource_fencing_uses_single_owner_and_cannot_feed():
    from fly_duel.duel import Duel, encounter_scene

    config, starts, headings = encounter_scene()
    config["food"][0]["amount"] = .3
    duel = Duel(config, seed=101, start=starts, start_headings=headings, social=True)
    try:
        commanded = fenced = 0
        for _ in range(350):
            duel.step()
            for contestant, transfers in zip(duel.contestants, duel.last_transfers):
                command = contestant.social_decision
                if command is not None:
                    commanded += 1
                    assert duel.elapsed_s >= 1.5
                    if command.state == "FENCE":
                        fenced += 1
                        assert contestant.decision.action == "FENCE"
                        assert contestant.unit.motor.owner in {"FENCE", "IDLE"}
                    elif command.state == "ASSESS":
                        assert contestant.decision.action == "IDLE"
                        assert contestant.unit.motor.owner == "IDLE"
                        assert command.gait == (0., 0.)
                    else:
                        assert contestant.decision.action == "WALK"
                        assert contestant.unit.motor.owner in {"WALK", "IDLE"}
                    assert not contestant.unit.motor.ingestion_enabled
                    assert transfers["intake_by_food"] == {}
        assert commanded > 0, "A hungry local resource failure must exercise the social branch"
        assert fenced > 0, "A nearby supported opponent must exercise actual foreleg fencing"
        assert all(c.organism.state.ingested_total > .01 for c in duel.contestants)
        _check_clocks_and_food(duel)
    finally:
        duel.close()
