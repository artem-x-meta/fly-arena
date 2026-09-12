from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from fly_arena.brain import Brain, BrainConfig, advance
from fly_arena.neural_ports import NeuralAdapter, NeuralConfig, UnsupportedPortError, build_registry, resolve_body_ids
from fly_arena.sensors import SensorFrame


@pytest.fixture
def tiny_graph(tmp_path):
    # Deliberately nonconsecutive IDs: array positions cannot masquerade as IDs.
    ids = np.array([10, 30, 70, 90, 120, 160, 200], np.int64)
    arrays = {"ids": ids, "ptr": np.array([0, 1, 1, 1, 1, 1, 1, 1], np.int64),
              "posts": np.array([1], np.int32), "counts": np.array([100], np.uint32),
              "signs": np.ones(7, np.int8)}
    for name, value in arrays.items():
        np.save(tmp_path / f"{name}.npy", value)
    ports = {"retina": [2], "eye": [0], "uv": [[.5, .5]], "lamina": [],
             "DNp09": {"L": [3], "R": [4]}, "DNa02": {"L": [5], "R": [6]}, "MDN": {"L": [], "R": []}}
    (tmp_path / "ports.json").write_text(json.dumps(ports))
    (tmp_path / "manifest.json").write_text(json.dumps({"format_version": 1, "dataset": "male-cns:v1.0"}))
    columns = ["bodyId", "type", "class", "subclass", "superclass", "somaSide", "rootSide", "synonyms", "flywireType", "mancType", "entryNerve", "receptorType", "matchingNotes"]
    rows = [{key: None for key in columns} for _ in ids]
    for row, body_id in zip(rows, ids):
        row["bodyId"] = int(body_id)
    rows[0].update(type="LgAG1", **{"class": "gustatory"}, mancType="SAch02", rootSide="L")
    rows[1].update(type="MN9", superclass="cb_motor", somaSide="L")
    rows[2].update(type="JO-FV", **{"class": "mechanosensory"}, subclass="grooming", rootSide="R")
    rows[3].update(type="DNg62", superclass="descending_neuron", synonyms="Hampel 2015: aDN1", somaSide="L")
    # Same JO type with different function must not be selected as grooming.
    rows[4].update(type="JO-FV", **{"class": "mechanosensory"}, subclass="wind_gravity", rootSide="R")
    rows[5].update(type="AOTU103m", superclass="cb_intrinsic", synonyms="Lee 2002: aDN")
    rows[6].update(type="LgLG1a", **{"class": "gustatory"}, receptorType="putative_ppk23")
    feather.write_feather(pa.Table.from_pylist(rows), tmp_path / "neurons.feather")
    return tmp_path


def test_registry_queries_keep_functional_subtypes_and_ids(tiny_graph):
    registry = build_registry(tiny_graph)
    groups = registry["groups"]
    assert groups["antennal_mechanosensory"]["body_ids"] == [70]
    assert groups["groom_output_adn"]["body_ids"] == [90]
    assert groups["taste_leg_sweet"]["body_ids"] == [10]
    assert groups["feed_output_mn9"]["body_ids"] == [30]
    assert groups["taste_leg_sweet"]["confidence"] == "candidate_homolog"
    assert all(group["status"] == "unsupported" for group in groups.values())
    assert resolve_body_ids(np.load(tiny_graph / "ids.npy"), [30]).tolist() == [1]
    for invalid in ([31], [201], [30, 30]):
        with pytest.raises(ValueError):
            resolve_body_ids(np.load(tiny_graph / "ids.npy"), invalid)


def test_strict_mode_fails_before_any_neural_step(tiny_graph):
    brain = Brain(tiny_graph)
    adapter = NeuralAdapter(brain, build_registry(tiny_graph))
    with pytest.raises(UnsupportedPortError, match="before simulation"):
        adapter.require_supported()
    assert brain.clock == 0
    readout = adapter.readout(np.full(7, 100), .1)
    assert readout.feed_drive == readout.groom_drives["head"] == 0
    assert readout.supported_ports["feeding"] == "unsupported"


def test_channel_ablation_is_separate_from_blind_and_hunger(tiny_graph):
    cfg = BrainConfig(background=0, lamina_bias=0, arousal=0, turn_bias=0)
    brain = Brain(tiny_graph, cfg)
    adapter = NeuralAdapter(brain, build_registry(tiny_graph), {"exploratory_ports": True})
    organism = SimpleNamespace(hunger=1., state=SimpleNamespace(sleep_pressure=1.))
    frame = SensorFrame(tarsal_taste={"LF": 1.}, dust_afferents={"antenna_left": .5})
    sensory, modulation = adapter.currents(frame, organism)
    brain.step(np.ones(1), blind=True, sensory_currents=sensory, modulation=modulation, disabled_channels=("dust",))
    channels = brain.channel_contributions
    assert channels["vision"]["sum_mv"] == channels["dust"]["sum_mv"] == 0
    assert channels["taste"]["sum_mv"] > 0 and channels["hunger"]["sum_mv"] > 0
    assert channels["sleep"]["sum_mv"] == 0  # no arousal bias available to subtract
    sensory, modulation = adapter.currents(SensorFrame(), organism)
    assert modulation["hunger"][1] == 0  # hunger cannot create nonexistent taste
    sensory, _ = adapter.currents(SensorFrame(tarsal_taste={"RF": 1.}), organism)
    assert sensory["taste"][1][0] == 0  # left afferent cannot smell/taste the right leg


def test_silencing_is_in_lif_and_does_not_modify_graph(tiny_graph):
    brain = Brain(tiny_graph, BrainConfig(background=0, arousal=0, turn_bias=0))
    original_weights = brain.weights.copy()
    state = brain.get_state()
    _, normal = brain.step(np.zeros(1), milliseconds=200, sensory_currents={"taste": ([0], 100.)})
    assert normal[0] > 0 and normal[1] > 0
    brain.set_state(state)
    _, blocked = brain.step(np.zeros(1), milliseconds=200, sensory_currents={"taste": ([0], 100.)}, blocked_outputs=[1])
    assert blocked[0] > 0 and blocked[1] == 0
    assert brain.voltage[1] == 0
    np.testing.assert_array_equal(brain.weights, original_weights)


def test_blocking_active_output_clears_old_filtered_motor_request(tiny_graph):
    brain = Brain(tiny_graph, BrainConfig(background=0, arousal=0, turn_bias=0))
    adapter = NeuralAdapter(brain, build_registry(tiny_graph), {"exploratory_ports": True})
    _, counts = brain.step(np.zeros(1), milliseconds=200, sensory_currents={"taste": ([0], 100.)})
    assert adapter.readout(counts, .2).feed_drive > 0
    _, counts = brain.step(np.zeros(1), milliseconds=10, sensory_currents={"taste": ([0], 100.)}, blocked_outputs=adapter.blocked_indices(("feeding",)))
    assert adapter.readout(counts, .01).feed_drive == 0


def test_checkpoint_restores_filters_and_pending_deliveries(tiny_graph):
    brain = Brain(tiny_graph)
    adapter = NeuralAdapter(brain, build_registry(tiny_graph), {"exploratory_ports": True})
    _, counts = brain.step(np.ones(1), milliseconds=17, sensory_currents={"taste": ([0], 100.)})
    adapter.readout(counts, .017)
    saved_brain, saved_adapter = brain.get_state(), adapter.get_state()
    command, counts = brain.step(np.zeros(1), milliseconds=43)
    readout = adapter.readout(counts, .043)
    final_state = brain.get_state()
    brain.set_state(saved_brain)
    adapter.set_state(saved_adapter)
    command2, counts2 = brain.step(np.zeros(1), milliseconds=43)
    readout2 = adapter.readout(counts2, .043)
    np.testing.assert_array_equal(counts, counts2)
    np.testing.assert_array_equal(command, command2)
    assert asdict(readout) == asdict(readout2)
    for name in ("queue", "current", "voltage", "filtered_luminance"):
        np.testing.assert_array_equal(final_state[name], brain.get_state()[name])


def test_original_step_matches_kernel_when_extra_channels_absent(tiny_graph):
    brain = Brain(tiny_graph)
    before = brain.get_state()
    drive = brain.base_drive.copy()
    filtered = before["filtered_luminance"] + (1 - np.exp(-1)) * (.8 - before["filtered_luminance"])
    drive[brain.retina] += brain.config.visual_gain * filtered / (.1 + filtered)
    expected = advance(brain.ptr, brain.posts, brain.weights, before["voltage"], before["current"], before["refractory"], before["queue"], 0, drive, 10)
    _, actual = brain.step(np.full(1, .8))
    np.testing.assert_array_equal(expected, actual)


def test_four_condition_artifact_does_not_promote_network_to_physical_support(tiny_graph, tmp_path):
    from scripts.neural_experiments import run_experiments

    report = run_experiments(tiny_graph, tmp_path / "experiment", seconds=.1, warmup_s=0., pathways=("feeding",),
        brain_config=BrainConfig(background=0, arousal=0, turn_bias=0),
        neural_config=NeuralConfig(exploratory_ports=True, taste_gain_mv=100.))
    path = report["pathways"]["feeding"]
    assert set(path["conditions"]) == {"neutral", "stimulus", "sensory_off", "output_off"}
    assert path["conditions"]["stimulus"]["output_mean_hz"] > 0
    assert path["conditions"]["sensory_off"]["output_mean_hz"] == 0
    assert path["conditions"]["output_off"]["output_mean_hz"] == 0
    assert path["support_status"] == "unsupported" and path["physical_status"] == "not_run"
