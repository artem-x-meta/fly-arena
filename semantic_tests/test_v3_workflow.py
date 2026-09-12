"""Small v3 provenance/timing checks; no full graph or physics experiment."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fly_arena.brain import BrainConfig
from fly_semantic.features import CausalFeatures
from fly_semantic.mapping import digest, file_digest
from fly_semantic.runtime import _BrainTap
from semantic_tools import v3_learning, v3_workflow as workflow


class TinyBrain:
    instances = []

    def __init__(self, *args, **kwargs):
        self.weights = np.array([.1, -.1], np.float32)
        self.clock, self.history = 0, []
        self.instances.append(self)

    def step(self, luminance, milliseconds=10, sensory_currents=None, **kwargs):
        self.history.append(copy.deepcopy(sensory_currents or {}))
        self.clock += milliseconds
        return np.zeros(2), np.array([0, 0, 1, 1, 1, 0], np.int32)


def features():
    return CausalFeatures([4, 5], neuron_count=6, excluded_indices=[0, 1, 2, 3])


def seal(study):
    study.pop("digest", None)
    study["digest"] = digest(study)
    return study


def write_plan(folder, study):
    workflow.write_json(folder / "plan.json", seal(study))


@pytest.fixture
def replay_case(tmp_path, monkeypatch):
    TinyBrain.instances.clear()
    nominal = hashlib.sha256(memoryview(np.array([.1, -.1], np.float32))).hexdigest()
    ep = {"episode_id": "train-1201-limited_meal", "split": "train", "seed_group": 1201,
          "seed": 12014, "scene": "limited_meal", "duration_ms": 200, "config": {}}
    parent = tmp_path / "old"
    source = parent / "episodes/connected" / (ep["episode_id"] + ".npz")
    source.parent.mkdir(parents=True)
    metadata = {"episode": ep, "nominal_weights_sha256": nominal, "plan_digest": "old-plan",
                "source_hashes": {"old-collector": "sha"}, "body_physics": True,
                "calibration_digest": "old-calibration", "feature_digest": "old-features",
                "stimulus_channels": [{"name": "semantic_hunger", "indices": [2, 3]}]}
    hunger = np.array([.713456123789, .613456123789], np.float64)
    arrays = dict(metadata=np.array(json.dumps(metadata)), X=np.zeros((2, 4)), y=np.ones(2, np.int8),
                  time_ms=np.array([100, 200]), hunger=hunger, intake=np.zeros(2), taste=np.zeros(2),
                  action=np.array(["FEED", "FEED"]), replay_luminance=np.zeros((20, 2), np.float32),
                  replay_values_0=np.repeat(np.float32(4. * np.repeat(hunger, 10))[:, None], 2, axis=1))
    np.savez(source, **arrays)
    workflow.write_json(parent / "plan.json", {"digest": "old-plan", "source_hashes": metadata["source_hashes"]})
    folder = tmp_path / "new"
    cal = {"digest": "new-calibration", "hunger_gain_mv": 5.}
    study = dict(episodes=[ep], development_workflow_sources={"new-collector": "sha"},
                 source_records={ep["episode_id"]: {"path": str(source), "sha256": file_digest(source)}},
                 calibration_dir=str(tmp_path / "calibration"), calibration_digest=cal["digest"],
                 brain_config=asdict(BrainConfig()), hunger_gain_mv=5., nominal_weights_sha256=nominal,
                 sample_ms=100, teacher_on=.7, teacher_off=.55, teacher_initial_active=False,
                 feature_digest=features().feature_digest, source_hashes=metadata["source_hashes"])
    monkeypatch.setattr(workflow, "workflow_sources", lambda: {"new-collector": "sha"})
    monkeypatch.setattr(workflow, "SOURCE", parent)
    monkeypatch.setattr(workflow, "load_mapping", lambda *args: {"ports": {"hunger": [{"index": 2}, {"index": 3}]}})
    monkeypatch.setattr(workflow, "load_calibration", lambda *args: cal)
    monkeypatch.setattr(workflow, "make_features", lambda *args: features())
    monkeypatch.setattr(workflow, "Brain", TinyBrain)
    monkeypatch.setattr(v3_learning, "verify_test_binding", lambda *args: {"pass": True})
    write_plan(folder, study)
    return SimpleNamespace(folder=folder, source=source, study=study, ep=ep, metadata=metadata, arrays=arrays)


def change_source(case, **arrays):
    case.arrays.update(arrays)
    np.savez(case.source, **case.arrays)
    case.study["source_records"][case.ep["episode_id"]]["sha256"] = file_digest(case.source)
    write_plan(case.folder, case.study)


def fresh_case(case):
    old_id = case.ep["episode_id"]
    case.ep.update(episode_id="test-4201-limited_meal", split="test", seed_group=4201, seed=42014)
    case.study["source_records"] = {}
    write_plan(case.folder, case.study)
    case.source = case.folder / "episodes/connected" / (case.ep["episode_id"] + ".npz")
    case.source.parent.mkdir(parents=True)
    case.metadata.update(episode=case.ep, plan_digest=case.study["digest"],
                         calibration_digest=case.study["calibration_digest"], feature_digest=case.study["feature_digest"],
                         physical_workflow_sources=case.study["development_workflow_sources"])
    case.arrays.update(metadata=np.array(json.dumps(case.metadata)),
                       replay_need_float64=np.repeat(case.arrays["hunger"], 10))
    np.savez(case.source, **case.arrays)
    return case


def test_fresh_groups_and_portions_are_disjoint_and_deterministic():
    episodes = workflow.test_episodes()
    assert episodes == workflow.test_episodes()
    assert len(episodes) == len({ep["episode_id"] for ep in episodes}) == 18
    assert len({ep["seed"] for ep in episodes}) == 18
    assert {ep["seed_group"] for ep in episodes} == {4201, 4202, 4203}
    for group, amount in zip((4201, 4202, 4203), (.6, 1.2, 1.6)):
        subset = [ep for ep in episodes if ep["seed_group"] == group]
        assert len(subset) == 6
        assert all(ep["split"] == "test" for ep in subset)
        limited = next(ep for ep in subset if ep["scene"] == "limited_meal")
        assert limited["config"]["food"][0]["amount"] == amount
        assert not limited["expected_release"]


def test_development_replays_gain_from_float32_recording_and_copies_body_labels(replay_case):
    c = replay_case
    workflow.replay(c.source, c.folder, c.study, c.ep, development=True)
    actual_drive = TinyBrain.instances[-1].history[0]["semantic_hunger"][1]
    expected_drive = c.arrays["replay_values_0"][0] * np.float32(5. / 4.)
    np.testing.assert_array_equal(actual_drive, expected_drive)
    target = c.folder / "episodes/connected" / (c.ep["episode_id"] + ".npz")
    with np.load(target) as a:
        for key in ("y", "time_ms", "hunger", "intake", "taste", "action"):
            np.testing.assert_array_equal(a[key], c.arrays[key])
        meta = json.loads(str(a["metadata"]))
        assert meta["body_physics"] is False
        assert meta["source_input_sha256"] == file_digest(c.source)
        assert meta["old_plan_digest"] == "old-plan"


@pytest.mark.parametrize("mutation", ["source_bytes", "episode_config", "old_test", "condition", "workflow_source"])
def test_development_rejects_changed_provenance_or_test_leakage_before_brain(replay_case, mutation):
    c = replay_case
    condition = "connected"
    if mutation == "source_bytes":
        with c.source.open("ab") as stream:
            stream.write(b"modified")
    elif mutation == "episode_config":
        c.ep = copy.deepcopy(c.ep)
        c.ep["config"] = {"different": True}
    elif mutation == "old_test":
        c.ep["split"] = "test"
        write_plan(c.folder, c.study)
    elif mutation == "condition":
        condition = "made-up"
    else:
        c.study["development_workflow_sources"] = {"new-collector": "changed"}
        write_plan(c.folder, c.study)
    with pytest.raises(ValueError):
        workflow.replay(c.source, c.folder, c.study, c.ep, condition, development=True)
    assert not TinyBrain.instances


@pytest.mark.parametrize("mutation", ["plan", "calibration", "features", "no_exact_need", "uncoupled_hunger", "uncoupled_labels", "nonfinite_need"])
def test_fresh_replay_rejects_identity_or_label_timing_errors(replay_case, mutation):
    c = fresh_case(replay_case)
    if mutation in ("plan", "calibration", "features"):
        key = {"plan": "plan_digest", "calibration": "calibration_digest", "features": "feature_digest"}[mutation]
        c.metadata[key] = "different"
        c.arrays["metadata"] = np.array(json.dumps(c.metadata))
    elif mutation == "no_exact_need":
        c.arrays.pop("replay_need_float64")
    elif mutation == "uncoupled_hunger":
        c.arrays["hunger"] = c.arrays["hunger"] + .01
    elif mutation == "uncoupled_labels":
        c.arrays["y"] = np.zeros(2, np.int8)
    else:
        c.arrays["replay_need_float64"][3] = np.nan
    np.savez(c.source, **c.arrays)
    with pytest.raises(ValueError):
        workflow.replay(c.source, c.folder, c.study, c.ep, "homeostasis_off")
    assert not (c.folder / "episodes/homeostasis_off" / c.source.name).exists()


def test_fresh_replay_uses_original_float64_need_not_rounded_current(replay_case):
    c = fresh_case(replay_case)
    workflow.replay(c.source, c.folder, c.study, c.ep, "transmission_off")
    current = TinyBrain.instances[-1].history[0]["semantic_hunger"][1]
    assert current == 5. * c.arrays["replay_need_float64"][0]
    assert current != float(c.arrays["replay_values_0"][0, 0] * np.float32(5. / 4.))


def test_need_recorder_preserves_input_precision_and_ordinary_delivery():
    need = .713456123789
    organism = SimpleNamespace(hunger=need)
    original = TinyBrain()
    channel = SimpleNamespace(ports={"hunger": np.array([2, 3])},
                              calibration={"hunger_gain_mv": 5.}, features=features())
    channel.currents = lambda hunger, clock: {"semantic_hunger": (channel.ports["hunger"], 5. * hunger)}
    channel.observe = lambda counts, dt, clock: channel.features.update(counts, dt)
    recorder = workflow.NeedRecorder(_BrainTap(original, organism, channel), channel, organism)
    recorder.step(np.zeros(2))
    assert recorder.exact_need == [need]
    assert original.history[0]["semantic_hunger"][1] == 5. * need
    assert float(recorder.channel_values[0][0][0]) != 5. * need


def test_collect_rejects_unplanned_episode_before_binding_or_physics(replay_case, monkeypatch):
    c = fresh_case(replay_case)
    c.ep = copy.deepcopy(c.ep)
    c.ep["config"] = {"tampered": True}
    monkeypatch.setattr(workflow, "SemanticSimulation", lambda *a, **k: pytest.fail("must reject before physics"))
    with pytest.raises(ValueError, match="episode differs"):
        workflow.collect_test(c.folder, c.study, c.ep)


def test_current_graph_must_match_frozen_nominal_weights(replay_case):
    c = replay_case
    c.study["nominal_weights_sha256"] = "different"
    write_plan(c.folder, c.study)
    with pytest.raises(ValueError, match="nominal graph"):
        workflow.replay(c.source, c.folder, c.study, c.ep, development=True)
