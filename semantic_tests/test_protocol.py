"""Protocol contracts independent of expensive graph/physics backends."""

from copy import deepcopy
import json

import pytest

from fly_semantic.protocol import (
    CueTransition, Event, InputCueQueue, OutputStateMachine, ProtocolError, REGISTRY,
)


def cue(event_id="cue-1", concept_id=101, **changes):
    result = dict(schema_version=1, episode_id="episode-1", event_id=event_id,
                  direction="to_brain", concept_id=concept_id, sim_time_ms=0,
                  ttl_ms=2000, active=True, source="human")
    result.update(changes)
    return result


@pytest.mark.parametrize("changes,code", [
    ({"schema_version": True}, "unsupported_schema_version"),
    ({"schema_version": 2}, "unsupported_schema_version"),
    ({"concept_id": True}, "unknown_concept_id"),
    ({"concept_id": 999}, "unknown_concept_id"),
    ({"direction": "from_brain"}, "wrong_direction"),
    ({"source": "neural_readout"}, "invalid_source"),
    ({"source": "unknown"}, "invalid_source"),
    ({"active": 1}, "invalid_active"),
    ({"active": False}, "unsupported_input_transition"),
    ({"sim_time_ms": float("nan")}, "invalid_sim_time_ms"),
    ({"sim_time_ms": float("inf")}, "invalid_sim_time_ms"),
    ({"sim_time_ms": True}, "invalid_sim_time_ms"),
    ({"sim_time_ms": -1}, "invalid_sim_time_ms"),
    ({"ttl_ms": 0}, "invalid_ttl_ms"),
    ({"ttl_ms": float("inf")}, "invalid_ttl_ms"),
    ({"ttl_ms": "250"}, "invalid_ttl_ms"),
    ({"sim_time_ms": 1e308, "ttl_ms": 1e308}, "invalid_expiry"),
    ({"event_id": ""}, "invalid_event_id"),
    ({"event_id": " bad "}, "invalid_event_id"),
    ({"episode_id": 12}, "invalid_episode_id"),
    ({"score": 0.9}, "input_score_forbidden"),
    ({"reward": 1}, "invalid_event_fields"),
])
def test_invalid_events_are_rejected_and_journalled(changes, code):
    queue = InputCueQueue("episode-1")
    assert not queue.submit(cue(**changes), 0)
    assert queue.pending_count == 0
    assert queue.journal[-1]["reason"] == code
    restored = InputCueQueue("ignored")
    restored.set_state(json.loads(json.dumps(queue.get_state())))
    assert restored.get_state() == queue.get_state()


def test_registry_is_fixed_and_event_roundtrip_is_strict():
    assert tuple(REGISTRY) == (1, 2, 3, 101, 102, 103)
    assert REGISTRY[1].label == "Нужна еда"
    with pytest.raises(TypeError):
        REGISTRY[4] = REGISTRY[1]
    assert Event.from_dict(cue()).to_dict() == cue()
    missing = cue()
    del missing["active"]
    with pytest.raises(ProtocolError, match="invalid_event_fields"):
        Event.from_dict(missing)


def test_episode_duplicate_and_direction_rejections():
    queue = InputCueQueue("episode-1")
    assert not queue.submit(cue(episode_id="old"), 0)
    assert queue.journal[-1]["reason"] == "wrong_episode"
    assert queue.submit(cue(), 0)
    assert not queue.submit(cue(), 0)
    assert queue.journal[-1]["reason"] == "duplicate_event"
    outbound = Event(1, "episode-1", "out-1", "from_brain", 1, 0, 1500, True, "neural_readout", 0.8)
    assert not queue.submit(outbound, 0)
    assert queue.journal[-1]["reason"] == "wrong_direction"


def test_ttl_limits_delivery_and_never_truncates_delivered_pulse():
    queue = InputCueQueue("episode-1", pulse_ms=250)
    assert queue.submit(cue(ttl_ms=10), 0)
    assert queue.advance(9) == (CueTransition(101, True),)
    assert queue.advance(10) == ()
    assert queue.active_concept_id == 101
    assert queue.advance(258) == ()
    assert queue.advance(259) == (CueTransition(101, False),)
    assert queue.active_concept_id is None
    assert not queue.submit(cue("late", sim_time_ms=259, ttl_ms=10), 269)
    assert queue.journal[-1]["reason"] == "expired_event"


def test_expiry_while_waiting_does_not_create_a_pulse():
    queue = InputCueQueue("episode-1")
    assert queue.submit(cue("a"), 0)
    assert queue.submit(cue("b", 102, ttl_ms=100), 0)
    assert queue.advance(0) == (CueTransition(101, True),)
    assert queue.advance(100) == ()
    assert queue.pending_count == 0
    assert queue.advance(250) == (CueTransition(101, False),)
    assert any(entry["event_id"] == "b" and entry["reason"] == "expired_event" for entry in queue.journal)


def test_one_active_cue_ties_fifo_future_scheduling_and_pause():
    queue = InputCueQueue("episode-1")
    assert queue.submit(cue("future", 103, sim_time_ms=1000), 0)
    assert queue.submit(cue("first", 101), 0)
    assert queue.submit(cue("second", 102), 0)
    assert queue.advance(0) == (CueTransition(101, True),)
    snapshot = queue.get_state()
    for _ in range(20):
        assert queue.advance(0) == ()
    assert queue.get_state() == snapshot
    assert queue.advance(250) == (CueTransition(101, False), CueTransition(102, True))
    assert queue.advance(500) == (CueTransition(102, False),)
    assert queue.advance(999) == ()
    assert queue.advance(1000) == (CueTransition(103, True),)
    assert set(vars(CueTransition(101, True))) == {"concept_id", "active"}


def test_queue_journal_and_dedup_remain_bounded_without_replay_eviction():
    queue = InputCueQueue("episode-1", capacity=1, max_events=2, journal_size=3)
    assert queue.submit(cue("a"), 0)
    assert not queue.submit(cue("b", 102), 0)
    assert queue.journal[-1]["reason"] == "queue_full"
    queue.advance(0)
    assert queue.submit(cue("b", 102), 0)
    queue.advance(250)
    queue.advance(500)
    for i in range(10):
        assert not queue.submit(cue(f"new-{i}", sim_time_ms=500), 500)
        assert queue.journal[-1]["reason"] == "episode_event_limit"
    assert not queue.submit(cue("a", sim_time_ms=500), 500)
    assert queue.journal[-1]["reason"] == "duplicate_event"
    assert len(queue.journal) == 3
    assert queue.get_state()["seen"] == ["a", "b"]


def test_input_checkpoint_preserves_inflight_pulse_pending_order_and_dedup():
    original = InputCueQueue("episode-1")
    original.submit(cue("a", ttl_ms=2), 0)
    original.submit(cue("b", 102), 0)
    original.advance(1)
    restored = InputCueQueue("unused")
    restored.set_state(json.loads(json.dumps(original.get_state())))
    for t in (2, 250, 251, 500, 501):
        assert restored.advance(t) == original.advance(t)
        assert restored.get_state() == original.get_state()
    assert not restored.submit(cue("a", sim_time_ms=501), 501)


def test_input_reset_clears_every_episode_artifact():
    queue = InputCueQueue("episode-1")
    queue.submit(cue(), 0)
    queue.advance(0)
    queue.reset("episode-2")
    assert queue.active_concept_id is None
    assert queue.pending_count == 0
    assert queue.journal == ()
    assert queue.get_state()["seen"] == []
    assert not queue.submit(cue(), 0)
    assert queue.submit(cue(episode_id="episode-2"), 0)


def test_hysteresis_independent_needs_confirmation_and_off_transition():
    output = OutputStateMachine("episode-1", (1, 2, 3))
    assert output.update({1: 0.8, 2: 0.9, 3: 0.1}, 0) == ()
    events = output.update({1: 0.8, 2: 0.9, 3: 0.1}, 100)
    assert [(e.concept_id, e.active) for e in events] == [(1, True), (2, True)]
    assert output.update({1: 0.3, 2: 0.5, 3: 0.8}, 200) == ()
    events = output.update({1: 0.3, 2: 0.5, 3: 0.8}, 300)
    assert [(e.concept_id, e.active) for e in events] == [(1, False), (3, True)]
    state = output.poll_state(300)
    assert [state[i]["active"] for i in (1, 2, 3)] == [False, True, True]
    assert all(e.source == "neural_readout" for e in events)


def test_fast_polling_and_callbacks_cannot_supply_confirmation():
    output = OutputStateMachine("episode-1")
    output.update({1: 0.9}, 0)
    saved = output.get_state()
    for _ in range(20):
        assert output.update({1: 0.9}, 0) == ()
        assert output.poll_state(0)[1]["active"] is False
    assert output.get_state() == saved
    assert output.update({1: 0.9}, 99) == ()
    assert output.update({1: 0.9}, 100)[0].active is True


def test_missing_samples_do_not_count_as_consecutive_confirmation():
    output = OutputStateMachine("episode-1")
    output.update({1: 0.9}, 0)
    assert output.update({1: 0.9}, 1000) == ()
    assert output.update({1: 0.9}, 1100)[0].active is True


def test_heartbeat_and_staleness_do_not_clear_active_need():
    output = OutputStateMachine("episode-1")
    output.update({1: 0.9}, 0)
    first = output.update({1: 0.9}, 100)[0]
    for t in range(200, 1100, 100):
        assert output.update({1: 0.9}, t) == ()
    heartbeat = output.update({1: 0.9}, 1100)[0]
    assert heartbeat.active and heartbeat.event_id != first.event_id
    assert output.poll_state(2599)[1]["stale"] is False
    cached = output.poll_state(2600)[1]
    assert cached["stale"] is True and cached["active"] is True
    assert cached["last_event"] == heartbeat.to_dict()


def test_output_checkpoint_replays_pending_confirmation_and_heartbeat():
    original = OutputStateMachine("episode-1", (1, 3))
    original.update({1: 0.9, 3: 0.1}, 0)
    restored = OutputStateMachine("unused", (1, 3))
    restored.set_state(json.loads(json.dumps(original.get_state())))
    for t in range(100, 1300, 100):
        scores = {1: 0.9 if t < 1100 else 0.1, 3: 0.9}
        assert restored.update(scores, t) == original.update(scores, t)
        assert restored.get_state() == original.get_state()
    original.reset("episode-2")
    assert original.poll_state(0)[1] == dict(active=False, score=None, sim_time_ms=None, stale=True, last_event=None)
    assert original.get_state()["sequence"] == 0


@pytest.mark.parametrize("supported", [(), []])
def test_no_trained_readout_has_no_supported_outputs(supported):
    output = OutputStateMachine("collector-1", supported)
    assert output.supported_concepts == ()
    assert output.poll_state(0) == {}
    for t in (0, 50, 100, 1000, 2500):
        assert output.update({}, t) == ()
        assert output.poll_state(t) == {}
    saved = json.loads(json.dumps(output.get_state()))
    restored = OutputStateMachine("unused", [])
    restored.set_state(saved)
    assert restored.get_state() == output.get_state()
    assert restored.get_state()["sequence"] == 0
    with pytest.raises(ProtocolError, match="wrong_score_concepts"):
        restored.update({1: 0.9}, 2600)
    restored.reset("collector-2")
    assert restored.poll_state(0) == {}
    assert restored.update({}, 0) == ()


@pytest.mark.parametrize("scores", [{}, {2: 0.9}, {True: 0.9}, {1: float("nan")}, {1: 1.1}, {1: True}])
def test_bad_output_scores_fail_atomically(scores):
    output = OutputStateMachine("episode-1")
    before = output.get_state()
    with pytest.raises(ProtocolError):
        output.update(scores, 100)
    assert output.get_state() == before


@pytest.mark.parametrize("kind", ["input", "output"])
def test_bad_checkpoint_is_atomic_and_config_must_match(kind):
    item = InputCueQueue("episode-1") if kind == "input" else OutputStateMachine("episode-1")
    before = item.get_state()
    for field, value in (("format", "old"), ("now_ms", float("nan")), ("episode_id", "")):
        corrupted = deepcopy(before)
        corrupted[field] = value
        with pytest.raises(ProtocolError):
            item.set_state(corrupted)
        assert item.get_state() == before
    mismatched = InputCueQueue("episode-1", pulse_ms=100) if kind == "input" else OutputStateMachine("episode-1", on_threshold=0.8)
    with pytest.raises(ProtocolError):
        mismatched.set_state(before)


def test_time_reversal_fails_and_poll_is_read_only():
    queue = InputCueQueue("episode-1")
    queue.advance(100)
    with pytest.raises(ProtocolError, match="time_reversed"):
        queue.advance(99)
    output = OutputStateMachine("episode-1")
    output.update({1: 0.8}, 100)
    saved = output.get_state()
    output.poll_state(10000)
    assert output.get_state() == saved
    with pytest.raises(ProtocolError, match="time_reversed"):
        output.update({1: 0.8}, 99)
