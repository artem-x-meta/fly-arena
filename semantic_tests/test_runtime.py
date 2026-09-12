"""Fast channel/adapter contracts; real body/graph check is verify_runtime CLI."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from fly_arena.brain import BrainConfig
from fly_semantic.mapping import digest
from fly_semantic.protocol import Event
from fly_semantic.readout import LinearReadout
from fly_semantic.runtime import SemanticChannel, SemanticSimulation, _BrainTap, attach, make_features, validate_calibration


def artifacts():
    names = ("hunger", "101", "102", "103")
    mapping = {"schema_version": 1, "neuron_count": 68, "cells_per_port": 16, "dataset": "synthetic-runtime-test",
        "graph_files": {name: "0" * 64 for name in ("ids.npy", "ptr.npy", "posts.npy", "counts.npy", "signs.npy", "ports.json", "neurons.feather", "behavior_ports.json")},
        "ports": {name: [{"index": i, "body_id": i + 1000} for i in range(k * 16, (k + 1) * 16)] for k, name in enumerate(names)},
        "features": [{"index": i, "body_id": i + 1000} for i in (64, 65)],
        "excluded_indices": list(range(64)) + [66, 67]}
    mapping["digest"] = digest(mapping)
    calibration = {"schema_version": 1, "status": "CALIBRATED", "hunger_status": "CALIBRATED", "cue_status": "CALIBRATED", "mapping_digest": mapping["digest"],
        "hunger_gain_mv": 0.5, "cue_gain_mv": 0.5, "pulse_ms": 250, "brain_config": asdict(BrainConfig())}
    calibration["digest"] = digest(calibration)
    return mapping, calibration


def head(mapping, calibration):
    features = make_features(mapping, calibration)
    return LinearReadout(feature_digest=features.feature_digest, concept_ids=[1],
        mean=np.zeros(features.n_features), scale=np.ones(features.n_features),
        weights=np.zeros((features.n_features, 1)), bias=[3])


def channel(with_head=True):
    mapping, calibration = artifacts()
    return SemanticChannel(mapping, calibration, readout=head(mapping, calibration) if with_head else None, episode_id="episode")


def cue(event_id="cue", time=100):
    return Event(1, "episode", event_id, "to_brain", 101, time, 1000, True, "human")


def advance(instance, steps=25):
    for tick in range(steps):
        instance.currents(.8, tick * 10)
        instance.observe(np.ones(68), 10, (tick + 1) * 10)


def assert_same(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    else:
        assert actual == expected


def test_disabled_attach_reads_no_artifacts_or_brain():
    class Untouchable:
        def __getattr__(self, key):
            raise AssertionError("disabled channel accessed simulation")
    assert attach(Untouchable(), enabled=False, mapping=Untouchable(), calibration=Untouchable()) is None


def test_enabled_without_head_has_no_supported_outputs():
    instance = channel(with_head=False)
    assert instance.supported_concepts == []
    advance(instance)
    assert instance.last_scores == {} and instance.last_events == ()
    assert instance.outbox.poll_state(250) == {}


def test_enabled_missing_artifacts_fails_before_body_construction(monkeypatch):
    import fly_arena.ethology
    monkeypatch.setattr(fly_arena.ethology, "EthologySimulation", lambda *a, **k: pytest.fail("body was constructed"))
    with pytest.raises(ValueError, match="requires calibrated artifacts"):
        SemanticSimulation({}, enabled=True)


def test_uncalibrated_and_mismatched_head_rejected():
    mapping, calibration = artifacts()
    bad = copy.deepcopy(calibration)
    bad["status"] = "TARGET_NOT_REACHED"
    bad.pop("digest")
    bad["digest"] = digest(bad)
    with pytest.raises(ValueError):
        validate_calibration(bad, mapping)
    model = head(mapping, calibration)
    model.feature_digest = "different-port-manifest"
    with pytest.raises(ValueError, match="matching trained"):
        SemanticChannel(mapping, calibration, readout=model)


def test_no_need_or_cue_metadata_leak_into_readout():
    a, b = channel(), channel()
    assert a.inbox.submit(cue("first", 0), 0)
    assert b.inbox.submit(Event(1, "episode", "other", "to_brain", 102, 0, 999, True, "oracle"), 0)
    for tick in range(20):
        # Distinct privileged input paths, identical observed neural counts.
        a.currents(.1, tick * 10)
        b.currents(.9, tick * 10)
        a.observe(np.ones(68), 10, (tick + 1) * 10)
        b.observe(np.ones(68), 10, (tick + 1) * 10)
    assert a.last_scores == b.last_scores
    assert [e.to_dict() for e in a.last_events] == [e.to_dict() for e in b.last_events]
    assert a.last_events[0].active is True


def test_input_disconnection_changes_currents_only():
    mapping, calibration = artifacts()
    instance = SemanticChannel(mapping, calibration, readout=head(mapping, calibration), hunger_disconnected=True, cue_disconnected=True, episode_id="episode")
    assert instance.inbox.submit(cue(time=0), 0)
    assert instance.currents(.9, 0) == {}
    assert instance.inbox.active_concept_id == 101


@pytest.mark.parametrize("cue_status", [None, "TARGET_NOT_REACHED"])
def test_uncalibrated_cue_gate_is_fail_closed(cue_status):
    mapping, calibration = artifacts()
    calibration.pop("digest")
    if cue_status is None:
        calibration.pop("cue_status")
    else:
        calibration["cue_status"] = cue_status
        calibration["cue_gain_mv"] = None
    calibration["digest"] = digest(calibration)
    instance = SemanticChannel(mapping, calibration, readout=head(mapping, calibration), episode_id="episode")
    assert instance.calibrated_input_concepts == []
    with pytest.raises(ValueError, match="not calibrated"):
        instance.submit(cue(time=0), 0)
    assert instance.inbox.pending_count == 0
    assert "semantic_hunger" in instance.currents(.9, 0)
    # Exposed transport cannot bypass the runtime safety gate.
    assert instance.inbox.submit(cue(time=0), 0)
    before = instance.get_state()
    with pytest.raises(ValueError, match="not calibrated"):
        instance.currents(.9, 0)
    assert_same(instance.get_state(), before)


def test_native_direct_input_overlap_rejected_before_brain_step():
    instance = channel()
    brain = SimpleNamespace(clock=0)
    tap = _BrainTap(brain, SimpleNamespace(hunger=.8), instance)
    with pytest.raises(ValueError, match="overlaps"):
        tap.step(np.zeros(2), sensory_currents={"new_native": (np.array([64]), 1)})


def test_channel_checkpoint_restores_queued_pulse_and_readout_history():
    a, b = channel(), channel()
    assert a.inbox.submit(cue("active", 100), 0)
    assert a.inbox.submit(cue("pending", 400), 0)
    advance(a)
    assert a.inbox.active_concept_id == 101 and a.inbox.pending_count == 1
    b.set_state(a.get_state())
    assert_same(a.get_state(), b.get_state())
    for tick in range(25, 45):
        assert_same(a.currents(.8, tick * 10), b.currents(.8, tick * 10))
        np.testing.assert_array_equal(a.observe(np.ones(68), 10, (tick + 1) * 10), b.observe(np.ones(68), 10, (tick + 1) * 10))
        assert_same(a.get_state(), b.get_state())


@pytest.mark.parametrize("corruption", ["inbox_episode", "outbox_episode", "future_inbox", "future_outbox", "event_episode", "event_direction"])
def test_cross_component_corruption_rejected_atomically(corruption):
    instance = channel()
    advance(instance, 20)
    before = instance.get_state()
    broken = copy.deepcopy(before)
    if corruption == "inbox_episode":
        broken["inbox"]["episode_id"] = "other"
    elif corruption == "outbox_episode":
        broken["outbox"]["episode_id"] = "other"
        broken["outbox"]["states"]["1"]["last_event"]["episode_id"] = "other"
    elif corruption == "future_inbox":
        broken["inbox"]["now_ms"] = 9999
    elif corruption == "future_outbox":
        broken["outbox"]["now_ms"] = 9999
    elif corruption == "event_episode":
        broken["last_events"][0]["episode_id"] = "other"
    elif corruption == "event_direction":
        broken["last_events"] = [cue(time=0).to_dict()]
    with pytest.raises(ValueError):
        instance.set_state(broken)
    assert_same(instance.get_state(), before)


def test_reset_clears_all_channel_state():
    instance = channel()
    assert instance.inbox.submit(cue(time=0), 0)
    advance(instance)
    instance.reset("new-episode")
    assert instance.inbox.episode_id == instance.outbox.episode_id == "new-episode"
    assert instance.inbox.active_concept_id is None and instance.inbox.pending_count == 0
    assert instance.inbox.journal == ()
    assert instance.features.elapsed_ms == 0 and instance.last_sample_ms == 0
    assert not np.any(instance.features.values())
    assert instance.last_scores == {} and instance.last_events == ()
