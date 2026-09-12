import numpy as np

from fly_arena.behavior import BehaviorController
from fly_arena.escape import VisualLoomingDetector, looming_position, LoomingEvents
from fly_arena.organism import Organism
from fly_arena.sensors import SensorFrame, NeuralReadout


def test_expanding_image_triggers_and_static_or_blind_image_does_not():
    detector = VisualLoomingDetector()
    eyes = np.zeros((2, 96, 96, 3), np.uint8)
    detector.step(eyes)
    eyes[0, 35:61, 35:61] = [220, 0, 220]
    assert detector.step(eyes)[0] > 1
    for _ in range(60):
        detector.step(eyes)
    assert detector.scores.max() < .001
    eyes[0, 25:71, 25:71] = [220, 0, 220]
    assert detector.step(eyes, blind=True).max() == 0


def test_looming_event_position_and_resume_retain_cycle():
    obj = dict(start=[10., 0, 2], end=[4., 0, 2], radius=1., time_s=1., duration_s=1., repeat_s=3.)
    np.testing.assert_array_equal(looming_position(obj, 1.5)[0], [7, 0, 2])
    events = LoomingEvents([obj])
    assert len(events.poll(1.)) == 1
    restored = LoomingEvents([obj])
    restored.set_state(events.get_state())
    assert restored.poll(1.) == []
    assert len(restored.poll(4.)) == 1


def test_escape_preempts_feeding_without_hybrid_bypass_in_neural_mode():
    organism = Organism(initial={"energy": 20})
    policy = BehaviorController({"escape_enabled": True, "min_action_duration_s": 0})
    frame = SensorFrame(tarsal_taste={"LF": 1})
    assert policy.decide(frame, organism, NeuralReadout()).action == "FEED"
    frame.looming_left = 3.
    decision = policy.decide(frame, organism, NeuralReadout())
    assert decision.action == "ESCAPE" and decision.escape_turn < 0
    neural_policy = BehaviorController({"escape_enabled": True, "min_action_duration_s": 0})
    assert neural_policy.decide(frame, organism, NeuralReadout(), mode="ethology-neural").action != "ESCAPE"
    readout = NeuralReadout(supported_ports={"escape": "supported"}, escape_drive=3., escape_turn=1.)
    assert neural_policy.decide(frame, organism, readout, mode="ethology-neural").source == "neural_readout"


def test_sleep_has_higher_escape_threshold_and_a_wake_delay():
    organism = Organism(initial={"energy": 80, "sleep_pressure": .95})
    policy = BehaviorController({"escape_enabled": True, "min_action_duration_s": 0,
                                 "sleep_entry_quiet_s": .05})
    frame = SensorFrame()
    for _ in range(20):
        policy.decide(frame, organism, NeuralReadout())
    assert policy.action == "SLEEP"
    frame.looming_left = 1.5
    assert policy.decide(frame, organism, NeuralReadout()).action == "SLEEP"
    frame.looming_left = 3
    assert policy.decide(frame, organism, NeuralReadout()).action == "WAKE"
    frame.looming_left = 0
    for _ in range(20):
        assert policy.decide(frame, organism, NeuralReadout()).action == "WAKE"
    for _ in range(6):
        policy.decide(frame, organism, NeuralReadout())
    assert policy.action == "ESCAPE"


def test_continuous_visual_requests_obey_refractory_period():
    policy = BehaviorController({"escape_enabled": True, "min_action_duration_s": 0})
    organism = Organism(initial={"energy": 80})
    frame = SensorFrame(looming_left=4)
    starts = []
    previous = None
    for i in range(700):
        decision = policy.decide(frame, organism, NeuralReadout())
        if decision.action == "ESCAPE" and previous != "ESCAPE":
            starts.append(i * .01)
        previous = decision.action
    assert 2 <= len(starts) <= 3
    assert min(np.diff(starts)) >= 2.79
