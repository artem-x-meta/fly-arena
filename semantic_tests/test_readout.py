import inspect

import numpy as np
import pytest

from fly_semantic.readout import LinearReadout, sigmoid


def training_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 5))
    X[:, 4] = 2.5
    y = np.column_stack((X[:, 0] > 0, X[:, 1] + 0.3 * X[:, 2] > 0)).astype(float)
    return X, y


def fit_readout():
    X, y = training_data()
    return LinearReadout.fit(X, y, feature_digest="feature-manifest", concept_ids=(1, 2), seed=11)


def test_multilabel_fit_and_unseen_episode_features():
    model = fit_readout()
    rng = np.random.default_rng(8)
    unseen = rng.normal(size=(100, 5))
    unseen[:, 4] = 2.5
    truth = np.column_stack((unseen[:, 0] > 0, unseen[:, 1] + 0.3 * unseen[:, 2] > 0))
    assert np.mean((model.predict_scores(unseen) >= 0.5) == truth) > 0.96
    assert model.diagnostics["converged"]
    assert model.diagnostics["score_calibration"] == "not_calibrated"
    both = np.array([3, 3, 0, 0, 2.5])
    assert np.all(model.predict_scores(both) > 0.9)
    neither = np.array([-3, -3, 0, 0, 2.5])
    assert np.all(model.predict_scores(neither) < 0.1)


def test_scaler_fit_on_train_only_and_inference_has_no_privileged_arguments():
    X, y = training_data()
    model = LinearReadout.fit(X, y, feature_digest="feature-manifest", concept_ids=(1, 2))
    np.testing.assert_array_equal(model.mean, X.mean(axis=0))
    assert model.scale[4] == 1.0
    before = model.predict_scores(X[:3])
    mean, scale = model.mean.copy(), model.scale.copy()
    model.predict_scores(np.full((10, 5), 1e6))
    np.testing.assert_array_equal(model.mean, mean)
    np.testing.assert_array_equal(model.scale, scale)
    np.testing.assert_array_equal(model.predict_scores(X[:3]), before)
    assert list(inspect.signature(model.predict_scores).parameters) == ["neural_features"]
    with pytest.raises(TypeError):
        model.predict_scores(X[:3], energy=0)


def test_reproducible_fit_does_not_consume_global_rng():
    np.random.seed(314)
    before = np.random.get_state()
    a, b = fit_readout(), fit_readout()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.bias, b.bias)
    assert a.diagnostics == b.diagnostics


def test_readout_npz_exact_roundtrip_and_feature_identity(tmp_path):
    model = fit_readout()
    destination = tmp_path / "readout.npz"
    model.save(destination)
    restored = LinearReadout.load(destination, expected_feature_digest="feature-manifest")
    X, _ = training_data()
    np.testing.assert_array_equal(model.predict_scores(X), restored.predict_scores(X))
    np.testing.assert_array_equal(model.weights, restored.weights)
    np.testing.assert_array_equal(model.mean, restored.mean)
    assert model.diagnostics == restored.diagnostics
    assert restored.seed == 11
    with np.load(destination, allow_pickle=False) as raw:
        assert all(raw[name].dtype.kind != "O" for name in raw.files)
    with pytest.raises(ValueError, match="identity mismatch"):
        LinearReadout.load(destination, expected_feature_digest="different-tau-or-indices")


def test_nonconvergence_remains_visible():
    X, y = training_data()
    model = LinearReadout.fit(X, y, feature_digest="manifest", concept_ids=(1, 2), max_iter=1)
    assert not model.diagnostics["converged"]
    assert model.diagnostics["iterations"] == 1
    assert np.all(np.isfinite(model.predict_scores(X)))


def test_sigmoid_does_not_overflow():
    with np.errstate(over="raise", under="ignore"):
        result = sigmoid(np.array([-1e6, -1, 0, 1, 1e6]))
    assert result[0] == 0 and result[-1] == 1 and result[2] == 0.5
    assert np.all(np.diff(result) > 0)


def test_single_label_shape_and_constant_class_diagnostic():
    X = np.arange(20, dtype=float).reshape(10, 2)
    model = LinearReadout.fit(X, np.ones(10), feature_digest="constant-state")
    assert model.predict_scores(X).shape == (10, 1)
    assert model.predict_scores(X[0]).shape == (1,)
    assert model.diagnostics["constant_label_columns"] == [0]


@pytest.mark.parametrize("change", [
    {"feature_digest": ""}, {"concept_ids": (1,)}, {"concept_ids": (1, 1)},
    {"l2": -1}, {"max_iter": 0}, {"seed": -1},
])
def test_invalid_training_options(change):
    X, y = training_data()
    options = dict(feature_digest="manifest", concept_ids=(1, 2))
    options.update(change)
    with pytest.raises(ValueError):
        LinearReadout.fit(X, y, **options)


@pytest.mark.parametrize("X,y", [
    (np.ones((3, 2)), np.array([0, 1, 0.5])),
    (np.ones((3, 2)), np.array([0, 1, np.nan])),
    (np.ones((3, 2)), np.ones((2, 1))),
    (np.full((3, 2), np.inf), np.ones(3)),
])
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        LinearReadout.fit(X, y, feature_digest="manifest")


def test_corrupt_checkpoint_nonfinite_weights_rejected(tmp_path):
    path = tmp_path / "model.npz"
    fit_readout().save(path)
    with np.load(path, allow_pickle=False) as source:
        saved = {name: source[name].copy() for name in source.files}
    saved["weights"][0, 0] = np.nan
    np.savez(path, **saved)
    with pytest.raises(ValueError, match="weights"):
        LinearReadout.load(path, expected_feature_digest="feature-manifest")
