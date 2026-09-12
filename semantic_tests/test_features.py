import copy

import numpy as np
import pytest

from fly_semantic.features import CausalFeatures


def make_features(**kwargs):
    options = dict(indices=[2, 4], neuron_count=6, excluded_indices=[0, 1], manifest_digest="test-ports-and-graph")
    options.update(kwargs)
    return CausalFeatures(**options)


def test_exact_causal_rate_step_and_decay():
    features = make_features()
    counts = np.array([9, 9, 1, 9, 2, 9])
    observed = features.update(counts, 10)
    expected = (-np.expm1(-10 / np.array([50, 200])))[:, None] * np.array([[100, 200]])
    np.testing.assert_allclose(observed, expected.reshape(-1), rtol=1e-14)
    np.testing.assert_allclose(features.update(np.zeros(6), 10), (expected * np.exp(-10 / np.array([50, 200]))[:, None]).reshape(-1), rtol=1e-14)
    assert features.elapsed_ms == 20
    assert features.update_count == 2
    # A snapshot cannot be changed by future observations or caller mutations.
    np.testing.assert_allclose(observed, expected.reshape(-1))
    observed[:] = 123
    assert not np.all(features.values() == 123)


def test_only_selected_neural_counts_change_features():
    a, b = make_features(), make_features()
    for tick in range(10):
        counts = np.arange(6) % 3
        changed = counts.copy()
        changed[[0, 1, 3, 5]] = tick * 100
        np.testing.assert_array_equal(a.update(counts, 10), b.update(changed, 10))


def test_checkpoint_exact_continuation_and_reset():
    a, b = make_features(), make_features()
    for dt in [1, 5, 10, 11.5]:
        a.update(np.arange(6) % 2, dt)
    state = a.get_state()
    b.set_state(state)
    state["traces_hz"][:] = 0
    for dt in [3, 4.5, 7, 10]:
        np.testing.assert_array_equal(a.update(np.arange(6) % 3, dt), b.update(np.arange(6) % 3, dt))
    assert a.elapsed_ms == b.elapsed_ms
    b.reset()
    np.testing.assert_array_equal(b.values(), np.zeros(4))
    assert b.elapsed_ms == b.update_count == 0


@pytest.mark.parametrize("kwargs", [
    {"indices": [0, 4]}, {"indices": [2, 2]}, {"indices": []},
    {"indices": [2.0, 4.0]}, {"indices": [-1, 4]}, {"indices": [2, 6]},
    {"indices": list(range(513)), "neuron_count": 600, "excluded_indices": []},
    {"tau_ms": [0, 200]}, {"tau_ms": [200, 50]}, {"tau_ms": [50, np.nan]},
    {"excluded_indices": [7]}, {"neuron_count": True},
])
def test_invalid_feature_definitions(kwargs):
    with pytest.raises(ValueError):
        make_features(**kwargs)


@pytest.mark.parametrize("change", [
    {"elapsed_ms": np.nan}, {"elapsed_ms": -1}, {"update_count": -1},
    {"update_count": 0}, {"traces_hz": np.ones((2, 3))},
    {"traces_hz": np.full((2, 2), np.nan)}, {"traces_hz": -np.ones((2, 2))},
    {"identity_digest": "wrong"},
])
def test_bad_checkpoint_rejected_without_mutation(change):
    features = make_features()
    features.update(np.ones(6), 10)
    before = features.get_state()
    invalid = copy.deepcopy(before)
    invalid.update(change)
    with pytest.raises(ValueError):
        features.set_state(invalid)
    after = features.get_state()
    assert after["elapsed_ms"] == before["elapsed_ms"]
    assert after["update_count"] == before["update_count"]
    np.testing.assert_array_equal(after["traces_hz"], before["traces_hz"])


def test_changed_mapping_tau_exclusion_or_graph_rejects_resume():
    source = make_features()
    for changed in [make_features(indices=[4, 2]), make_features(tau_ms=[40, 200]), make_features(excluded_indices=[0, 1, 3]), make_features(manifest_digest="different-graph")]:
        with pytest.raises(ValueError, match="identity mismatch"):
            changed.set_state(source.get_state())


@pytest.mark.parametrize("counts,dt", [
    (np.ones(5), 10), (np.full(6, np.nan), 10), (-np.ones(6), 10),
    (np.ones(6), 0), (np.ones(6), np.nan), (np.ones(6), -1),
])
def test_invalid_observation_does_not_advance_time(counts, dt):
    features = make_features()
    with pytest.raises(ValueError):
        features.update(counts, dt)
    assert features.update_count == features.elapsed_ms == 0


def test_constant_interval_rate_is_invariant_to_subdivision():
    coarse, fine = make_features(), make_features()
    coarse.update(np.array([0, 0, 2, 0, 4, 0]), 20)
    for _ in range(2):
        fine.update(np.array([0, 0, 1, 0, 2, 0]), 10)
    np.testing.assert_allclose(coarse.values(), fine.values(), atol=1e-14)
