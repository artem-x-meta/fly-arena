import json

import pytest

from fly_arena.behavior import BehaviorController
from fly_arena.organism import Organism, OrganismConfig
from fly_arena.sensors import NeuralReadout, SensorFrame


def select(controller, organism, sensors=None, neural=None, steps=1, dt=.01, mode="ethology-hybrid"):
    sensors = sensors or SensorFrame()
    for _ in range(steps):
        sensors.time_s += dt
        decision = controller.decide(sensors, organism, neural or NeuralReadout(), dt=dt, mode=mode)
    return decision


def test_feeding_uses_local_tarsal_taste_and_hunger_but_does_not_ingest():
    senses = SensorFrame(tarsal_taste={"LF": 1})
    hungry = Organism(initial={"energy": 15})
    full = Organism(initial={"energy": 95})
    hungry_decision = select(BehaviorController({"min_action_duration_s": 0}), hungry, senses)
    sated_decision = select(BehaviorController({"min_action_duration_s": 0}), full, senses)
    assert hungry_decision.action == "FEED"
    assert sated_decision.action != "FEED"
    assert hungry.state.gut_amount == hungry.state.ingested_total == 0


def test_failed_contact_times_out_and_retries_are_bounded():
    controller = BehaviorController({"min_action_duration_s": 0, "feed_contact_timeout_s": .1,
                                     "feed_contact_grace_s": .05, "feed_retry_delay_s": .2})
    organism = Organism(initial={"energy": 10})
    sensors = SensorFrame(tarsal_taste={"LF": 1})
    assert select(controller, organism, sensors).action == "FEED"
    timeout = select(controller, organism, sensors, steps=11)
    assert timeout.action != "FEED"
    assert controller.feed_retry_remaining_s > 0
    assert organism.state.gut_amount == 0


def test_lost_mouth_contact_stops_feeding_and_full_gut_overrides_dwell():
    controller = BehaviorController({"min_action_duration_s": 0})
    organism = Organism(initial={"energy": 10})
    sensor = SensorFrame(tarsal_taste={"LF": 1}, mouth_contact=True, mouth_taste=1)
    select(controller, organism, sensor, steps=2)
    organism.consume("food", 10, 10)
    decision = select(controller, organism, sensor)
    assert decision.action == "IDLE" and decision.reason == "satiety_or_full_gut"


def test_transient_mouth_overlap_during_extension_is_not_established_contact():
    controller = BehaviorController({"min_action_duration_s": 0})
    organism = Organism(initial={"energy": 15})
    sensor = SensorFrame(tarsal_taste={"LF": 1}, mouth_contact=True, mouth_taste=1)
    select(controller, organism, sensor, steps=3)
    assert not controller.feed_contact_seen
    sensor.mouth_contact = False
    decision = select(controller, organism, sensor, steps=55)
    assert decision.action == "FEED"  # lost transient overlap cannot abort extension
    sensor.mouth_contact = True
    select(controller, organism, sensor, steps=12)
    assert controller.feed_contact_seen
    sensor.mouth_contact = False
    decisions = [select(controller, organism, sensor) for _ in range(26)]
    assert any(d.reason == "mouth_contact_lost" and d.action == "IDLE" for d in decisions)


def test_grooming_switches_with_local_dust_feedback_and_stops_when_clean():
    controller = BehaviorController({"min_action_duration_s": 0})
    organism = Organism(initial={"energy": 90})
    sensor = SensorFrame(dust_afferents={"head": 1})
    assert select(controller, organism, sensor).action == "GROOM_HEAD"
    sensor.dust_afferents = {"head": .1, "front_left": .8, "front_right": .4}
    assert select(controller, organism, sensor).action == "GROOM_FRONT"
    sensor.dust_afferents = {}
    assert select(controller, organism, sensor).action == "WALK"
    assert sum(organism.state.dust_by_region.values()) == 0  # policy never edits physical dust


def test_sleep_requires_quiet_stance_and_has_higher_wake_threshold_and_delay():
    controller = BehaviorController({"min_action_duration_s": 0, "sleep_entry_quiet_s": .05, "wake_delay_s": .05})
    organism = Organism(initial={"energy": 70, "sleep_pressure": .95})
    sensor = SensorFrame()
    assert select(controller, organism, sensor).action == "SLEEP_ENTRY"
    assert select(controller, organism, sensor, steps=6).action == "SLEEP"
    sensor.local_wake_stimulus = .4  # prevents entry, below sleeping wake threshold
    assert select(controller, organism, sensor).action == "SLEEP"
    sensor.local_wake_stimulus = 1
    wake = select(controller, organism, sensor)
    assert wake.action == "WAKE" and wake.reason == "strong_local_stimulus"
    assert select(controller, organism, sensor).action == "WAKE"
    assert select(controller, organism, sensor, steps=4).action == "IDLE"


def test_disturbance_causes_sleep_deprivation_without_assigning_rebound_duration():
    cfg = OrganismConfig(tau_awake=2, tau_sleep=.5, basal_metabolic_rate=0)
    control = Organism(cfg, {"energy": 70, "sleep_pressure": .9})
    disturbed = Organism(cfg, {"energy": 70, "sleep_pressure": .9})
    controllers = [BehaviorController({"min_action_duration_s": 0, "sleep_entry_quiet_s": .02}) for _ in range(2)]
    for _ in range(100):
        for i, organism in enumerate((control, disturbed)):
            decision = select(controllers[i], organism, SensorFrame(local_wake_stimulus=.4 if i else 0))
            organism.advance(.01, decision.action)
    assert disturbed.state.sleep_pressure > control.state.sleep_pressure + .25
    assert controllers[1].action != "SLEEP"
    counts = [0, 0]
    for _ in range(100):
        for i, organism in enumerate((control, disturbed)):
            decision = select(controllers[i], organism)
            counts[i] += decision.action == "SLEEP"
            organism.advance(.01, decision.action)
    assert counts[1] > counts[0]


def test_neural_mode_never_uses_taste_dust_or_unsupported_drive_as_fallback():
    organism = Organism(initial={"energy": 20, "sleep_pressure": .2})
    sensor = SensorFrame(tarsal_taste={"LF": 1}, dust_afferents={"head": 10})
    outputs = NeuralReadout(feed_drive=100, groom_drives={"head": 100}, sleep_drive=100,
                           supported_ports={"feed": "candidate", "groom_head": "unsupported"})
    controller = BehaviorController({"min_action_duration_s": 0})
    assert select(controller, organism, sensor, outputs, mode="ethology-neural").action == "IDLE"
    outputs.supported_ports["feed"] = "confirmed"
    assert select(controller, organism, sensor, outputs, mode="ethology-neural").action == "FEED"
    outputs.feed_drive = 0
    decision = select(controller, organism, sensor, outputs, mode="ethology-neural")
    assert decision.action == "IDLE" and decision.reason == "neural_feed_request_lost"


def test_experimental_ports_require_explicit_opt_in_and_sleep_is_labelled_engineered():
    organism = Organism(initial={"energy": 20})
    sensors = SensorFrame(tarsal_taste={"LF": 1})
    readout = NeuralReadout(feed_drive=1, supported_ports={"feed": "experimental"})
    strict = BehaviorController({"min_action_duration_s": 0})
    exploratory = BehaviorController({"min_action_duration_s": 0, "allow_experimental_ports": True})
    assert select(strict, organism, sensors, readout, mode="ethology-neural").action == "IDLE"
    decision = select(exploratory, organism, sensors, readout, mode="ethology-neural")
    assert decision.action == "FEED" and decision.source == "neural_readout"
    sleepy = Organism(initial={"energy": 70, "sleep_pressure": .95})
    decision = select(strict, sleepy, SensorFrame(), NeuralReadout(), mode="ethology-neural")
    assert decision.action == "SLEEP_ENTRY" and decision.source == "engineered_policy"


def test_instability_and_critical_energy_override_competing_needs():
    organism = Organism(initial={"energy": 2, "sleep_pressure": .99})
    controller = BehaviorController({"min_action_duration_s": 0})
    sensor = SensorFrame(dust_afferents={"head": 10}, tarsal_taste={"LF": 1})
    assert select(controller, organism, sensor).action == "FEED"
    sensor.upright = .2
    guarded = select(controller, organism, sensor)
    assert guarded.action == "IDLE" and guarded.source == "physical_guard"


@pytest.mark.parametrize("action_sensor", [SensorFrame(tarsal_taste={"LF": 1}),
                                            SensorFrame(dust_afferents={"head": 1})])
def test_checkpoint_continuation_preserves_action_phases(action_sensor):
    organism = Organism(initial={"energy": 10})
    controller = BehaviorController({"min_action_duration_s": 0})
    select(controller, organism, action_sensor, steps=30)
    resumed = BehaviorController({"min_action_duration_s": 0})
    resumed.set_state(json.loads(json.dumps(controller.get_state())))
    # Same sensory time is supplied to both; policy does not use global time to
    # schedule actions, but transition event timestamps must still agree.
    for _ in range(30):
        a = controller.decide(action_sensor, organism, NeuralReadout())
        b = resumed.decide(action_sensor, organism, NeuralReadout())
        assert a == b
    assert resumed.get_state() == controller.get_state()


def test_odor_alone_does_not_open_a_feeding_attempt():
    # Odor is a distance cue for the search controller: an attempt started on it
    # probes the floor short of the patch instead of walking onto it.
    controller = BehaviorController({"min_action_duration_s": 0})
    organism = Organism(initial={"energy": 10})
    smell_only = SensorFrame(odor_left=1., odor_right=1.)
    assert select(controller, organism, smell_only, steps=3).action != "FEED"
    tasting = SensorFrame(odor_left=1., odor_right=1., tarsal_taste={"LF": 1})
    assert select(controller, organism, tasting).action == "FEED"


def test_trailing_tarsus_taste_does_not_open_a_feeding_attempt():
    controller = BehaviorController({"min_action_duration_s": 0})
    organism = Organism(initial={"energy": 10})
    behind = SensorFrame(tarsal_taste={"LH": 1, "RM": 1})
    assert select(controller, organism, behind, steps=3).action != "FEED"
    front = SensorFrame(tarsal_taste={"LH": 1, "RM": 1, "RF": 1})
    assert select(controller, organism, front).action == "FEED"


def test_failed_reach_turns_hunger_into_approach_instead_of_standing_still():
    controller = BehaviorController({"min_action_duration_s": 0, "feed_contact_timeout_s": .1,
                                     "feed_contact_grace_s": .05, "feed_retry_delay_s": .5,
                                     "dust_enter_threshold": .15})
    organism = Organism(initial={"energy": 10})
    # Dust that would otherwise outscore walking, plus food the fly can taste.
    sensors = SensorFrame(tarsal_taste={"LF": 1}, dust_afferents={"head": 1., "antenna_left": .5})
    assert select(controller, organism, sensors).action == "FEED"
    decision = select(controller, organism, sensors, steps=11)
    assert decision.action != "FEED"
    approach = select(controller, organism, sensors, steps=2)
    assert approach.action == "WALK" and approach.gates["feed_approach"]


def test_depletion_still_allows_the_approach_to_tasted_food():
    controller = BehaviorController({"min_action_duration_s": 0, "feed_contact_timeout_s": .1,
                                     "feed_contact_grace_s": .05, "feed_retry_delay_s": .5})
    organism = Organism(initial={"energy": 0})
    assert organism.exhausted
    starving = SensorFrame(tarsal_taste={"LF": 1})
    assert select(controller, organism, starving).action == "FEED"
    select(controller, organism, starving, steps=11)
    assert select(controller, organism, starving, steps=2).action == "WALK"
    # Without any local food the same depleted fly reports the explicit state.
    idle_controller = BehaviorController({"min_action_duration_s": 0})
    assert select(idle_controller, organism, SensorFrame()).action == "EXHAUSTED"


def test_walking_is_not_interrupted_by_a_tripod_stance():
    controller = BehaviorController({"min_action_duration_s": 0})
    organism = Organism(initial={"energy": 15})
    tripod = SensorFrame(ground_support=3, movement=4.)
    assert select(controller, organism, tripod, steps=3).action == "WALK"
    # Sleep entry keeps the stricter footing requirement.
    sleepy = Organism(initial={"energy": 90, "sleep_pressure": .95})
    quiet_tripod = SensorFrame(ground_support=3)
    entering = select(BehaviorController({"min_action_duration_s": 0}), sleepy, quiet_tripod, steps=80)
    assert entering.action != "SLEEP"
    quiet_stance = SensorFrame(ground_support=6)
    settled = select(BehaviorController({"min_action_duration_s": 0}), sleepy, quiet_stance, steps=80)
    assert settled.action == "SLEEP"
