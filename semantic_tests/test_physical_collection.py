"""Small input-recording/plan checks; these are not physical feeding results."""
from collections import Counter
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fly_arena.config import load_config
from fly_arena.organism import Organism
from fly_semantic.features import CausalFeatures
from fly_semantic.mapping import digest
from fly_semantic.runtime import _BrainTap
from semantic_tools import physical_collection as collection


class TinyBrain:
    """Deterministic fake exposes exact delivery order and RNG consumption."""
    def __init__(self):
        self.clock = 0
        self.rng = np.random.default_rng(135)
        self.history = []

    def step(self, luminance, milliseconds=10, *, sensory_currents=None, modulation=None, **kwargs):
        channels = {**(sensory_currents or {}), **(modulation or {})}
        self.history.append([(name, np.asarray(indices).copy(),
                              np.broadcast_to(np.asarray(values, np.float32), np.asarray(indices).shape).copy())
                             for name, (indices, values) in channels.items()])
        drive = np.zeros(6, np.float32)
        drive[:2] += np.asarray(luminance, np.float32)
        for _, indices, values in self.history[-1]:
            np.add.at(drive, indices, values)
        counts = self.rng.integers(0, 2, 6, dtype=np.int32)
        counts[4:] += int(abs(drive.sum()) > 4)
        self.clock += milliseconds
        return drive[:2].copy(), counts


def channel():
    result = SimpleNamespace(
        ports={"hunger": np.array([2, 3], np.int32)},
        calibration={"hunger_gain_mv": 4.},
        features=CausalFeatures([4, 5], neuron_count=6, excluded_indices=[0, 1, 2, 3]),
    )
    result.currents = lambda need, clock: {"semantic_hunger": (result.ports["hunger"], 4. * float(need))}
    result.observe = lambda counts, dt, clock: result.features.update(counts, dt)
    return result


def test_recorder_preserves_tap_outputs_features_rng_and_replay_order():
    organism = SimpleNamespace(hunger=.91)
    ordinary_brain, recorded_brain = TinyBrain(), TinyBrain()
    ordinary_channel, recorded_channel = channel(), channel()
    ordinary = _BrainTap(ordinary_brain, organism, ordinary_channel)
    recorder = collection.NeuralRecorder(_BrainTap(recorded_brain, organism, recorded_channel), recorded_channel, organism)
    expected_features = []
    for tick in range(20):
        organism.hunger = .91 - tick * .025
        kwargs = {"sensory_currents": {"native_taste": (np.array([0, 1]), .31 + tick * .013)},
                  "modulation": {"native_need": (np.array([0, 1]), np.array([.2, .1], np.float64))}}
        luminance = np.array([.33, .77], np.float64) + tick * .001
        expected = ordinary.step(luminance, **kwargs)
        actual = recorder.step(luminance, **kwargs)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(ordinary_channel.features.values(), recorded_channel.features.values())
        expected_features.append(recorded_channel.features.values())
        assert list(kwargs["sensory_currents"]) == ["native_taste"]
    assert ordinary_brain.rng.bit_generator.state == recorded_brain.rng.bit_generator.state
    assert [spec["name"] for spec in recorder.channel_specs] == ["native_taste", "semantic_hunger", "native_need"]
    replay = TinyBrain()
    replay_features = channel().features
    for tick, luminance in enumerate(recorder.luminance):
        currents = {spec["name"]: (np.array(spec["indices"]), recorder.channel_values[i][tick])
                    for i, spec in enumerate(recorder.channel_specs)}
        _, counts = replay.step(luminance, sensory_currents=currents)
        np.testing.assert_array_equal(replay_features.update(counts, 10), expected_features[tick])
        for (name_a, idx_a, val_a), (name_b, idx_b, val_b) in zip(replay.history[-1], recorded_brain.history[tick]):
            assert name_a == name_b
            np.testing.assert_array_equal(idx_a, idx_b)
            np.testing.assert_array_equal(val_a, val_b)


@pytest.mark.parametrize("kwargs", [
    {"milliseconds": 20}, {"blind": True}, {"disabled_channels": ["vision"]},
    {"blocked_outputs": np.array([1], np.int32)},
])
def test_unsupported_collection_controls_fail_before_neural_step(kwargs):
    original = TinyBrain()
    recorder = collection.NeuralRecorder(original, channel(), SimpleNamespace(hunger=.8))
    with pytest.raises(ValueError, match="10ms unablated"):
        recorder.step(np.zeros(2), **kwargs)
    assert original.clock == 0


def test_changed_input_addresses_are_not_silently_reinterpreted():
    original = TinyBrain()
    recorder = collection.NeuralRecorder(original, channel(), SimpleNamespace(hunger=.8))
    recorder.step(np.zeros(2), sensory_currents={"taste": (np.array([0]), 1.)})
    with pytest.raises(ValueError, match="Changing neural input addresses"):
        recorder.step(np.zeros(2), sensory_currents={"taste": (np.array([1]), 1.)})
    assert original.clock == 10


@pytest.fixture
def plan(tmp_path, monkeypatch):
    base_config = load_config("configs/ethology-unscaled.toml", "feeding-contact")
    monkeypatch.setattr(collection, "load_config", lambda *args: copy.deepcopy(base_config))
    monkeypatch.setattr(collection, "load_mapping", lambda *args: {"digest": "mapping"})
    monkeypatch.setattr(collection, "load_calibration", lambda *args: {"digest": "calibration", "brain_config": {}})
    monkeypatch.setattr(collection, "make_features", lambda *args: SimpleNamespace(feature_digest="features"))
    monkeypatch.setattr(collection, "file_digest", lambda *args: "f" * 64)
    monkeypatch.setattr(collection, "source_hashes", lambda: {"test_fake_collector": "f" * 64})
    study = collection.freeze_plan(tmp_path)
    return tmp_path, study


def test_physical_plan_has_disjoint_whole_episodes_and_balanced_scenes(plan):
    output, study = plan
    assert collection.load_plan(output) == study
    assert Counter(ep["split"] for ep in study["episodes"]) == {"train": 18, "validation": 6, "test": 12}
    assert len({ep["seed"] for ep in study["episodes"]}) == 36
    assert len({ep["episode_id"] for ep in study["episodes"]}) == 36
    groups = list(study["seed_groups"].values())
    assert len(set(sum(groups, []))) == sum(map(len, groups))
    for group in sum(groups, []):
        episodes = [ep for ep in study["episodes"] if ep["seed_group"] == group]
        assert {ep["scene"] for ep in episodes} == set(collection.SCENES)
        assert sum(ep["expected_release"] for ep in episodes) == 2
    assert study["label_reference_offset_ms"] == 10
    assert study["sample_ms"] == 100


def test_planned_states_cover_need_and_sated_food_without_forced_target_labels(plan):
    _, study = plan
    for ep in study["episodes"]:
        cfg = ep["config"]
        organism = Organism(cfg["organism"], cfg["initial"])
        if ep["scene"].startswith("sated"):
            assert organism.hunger < study["teacher_off"]
        else:
            assert organism.hunger >= study["teacher_on"]
        if ep["scene"].endswith("away"):
            assert cfg["food"] == []
        else:
            assert cfg["food"][0]["amount"] > 0
        if ep["scene"] == "limited_meal":
            assert cfg["food"][0]["amount"] == .8
            # Even maximal initial gut filling cannot satisfy this need.
            maximum_gut_fraction = cfg["food"][0]["amount"] / organism.config.gut_capacity
            assert organism.hunger * (1 - maximum_gut_fraction) > study["teacher_off"]
        if ep["expected_release"]:
            assert ep["duration_ms"] >= 5500


def test_frozen_plan_cannot_be_replaced(plan):
    output, _ = plan
    before = (output / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        collection.freeze_plan(output)
    assert (output / "plan.json").read_bytes() == before


@pytest.mark.parametrize("defect", ["tamper", "duplicate_seed", "duplicate_episode", "overlapping_group", "wrong_split"])
def test_plan_rejects_tampering_and_cross_split_overlap(plan, defect):
    output, original = plan
    study = copy.deepcopy(original)
    if defect == "tamper":
        study["teacher_off"] = .01
    elif defect == "duplicate_seed":
        study["episodes"][1]["seed"] = study["episodes"][0]["seed"]
    elif defect == "duplicate_episode":
        study["episodes"][1]["episode_id"] = study["episodes"][0]["episode_id"]
    elif defect == "overlapping_group":
        study["seed_groups"]["test"].append(study["seed_groups"]["train"][0])
    else:
        study["episodes"][0]["split"] = "test"
    if defect != "tamper":
        study.pop("digest")
        study["digest"] = digest(study)
    (output / "plan.json").write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(ValueError):
        collection.load_plan(output)
