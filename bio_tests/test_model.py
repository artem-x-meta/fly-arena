from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fly_bio.model import GradedConfig, GradedNetwork


def chain():
    # retina -> inhibitory histamine -> inhibitory L1 -> ON Mi1,
    # plus an excitatory L2 -> OFF relay on a separate branch.
    matrix = csr_matrix(([-1., -1., -1., 1.], ([1, 2, 3, 4], [0, 1, 0, 3])), shape=(5, 5), dtype=np.float32)
    return SimpleNamespace(incoming=matrix, ports={"retina": [0]})


def test_tonic_reference_is_stationary_and_not_zero_activity():
    model = GradedNetwork(chain())
    for _ in range(50):
        model.step([.5])
    np.testing.assert_array_equal(model.voltage, np.full(5, .5, np.float32))
    assert np.count_nonzero(model.last_external_drive) == 0


@pytest.mark.parametrize("luminance,sign", [(.6, 1), (.4, -1)])
def test_graded_double_inhibition_transmits_correct_on_off_polarity(luminance, sign):
    model = GradedNetwork(chain())
    model.equilibrate([luminance])
    delta = model.voltage - .5
    assert sign * delta[0] > 0
    assert sign * delta[1] < 0  # lamina inverts retinal sign
    assert sign * delta[2] > 0  # ON path's second inversion
    assert sign * delta[4] < 0  # OFF branch has one inversion
    assert model.last_external_drive[1:].tolist() == [0.] * 4


def test_freezing_relay_blocks_modulation_without_removing_tonic_release():
    model = GradedNetwork(chain())
    reference = model.voltage.copy()
    for _ in range(100):
        model.step([.6], freeze_indices=[1], freeze_voltage=reference)
    assert model.voltage[0] > .5
    assert model.voltage[1] == .5 and model.voltage[2] == .5
    assert model.voltage[4] < .5


def test_input_off_and_checkpoint_resume_are_exact():
    model = GradedNetwork(chain())
    for _ in range(15):
        model.step([.8], input_off=True)
    np.testing.assert_array_equal(model.voltage, [.5] * 5)
    for _ in range(7):
        model.step([.65])
    saved = model.get_state()
    resumed = GradedNetwork(chain())
    resumed.set_state(saved)
    for light in [.7, .3, .6, .9]:
        np.testing.assert_array_equal(model.step([light]), resumed.step([light]))


def test_recurrent_network_converges_with_bounded_input():
    matrix = csr_matrix(np.array([[.5, -.5], [1., 0]], np.float32))
    model = GradedNetwork(SimpleNamespace(incoming=matrix, ports={"retina": [0]}))
    equilibrium = model.equilibrate([1.])
    initial = model.voltage.copy()
    for _ in range(100):
        model.step([1.])
    np.testing.assert_allclose(model.voltage, initial, atol=2e-6)
    assert equilibrium["residual_au"] <= 2e-7


def test_step_matches_independent_scalar_equation():
    model = GradedNetwork(chain())
    model.voltage[:] = [.7, .4, .9, .2, .1]
    prior = model.voltage.astype(float).copy()
    drive = np.array([.15, 0, 0, 0, 0])
    expected = prior + (1 - np.exp(-.01/.05)) * (
        -prior + model.bias + .8 * (chain().incoming @ np.maximum(prior, 0)) + drive)
    np.testing.assert_allclose(model.step([.65]), expected, atol=5e-8)


def test_resume_rejects_different_graph_of_the_same_size():
    first = GradedNetwork(chain())
    changed = chain()
    changed.incoming.data[0] *= -1
    second = GradedNetwork(changed)
    with pytest.raises(ValueError, match="Incompatible"):
        second.set_state(first.get_state())


@pytest.mark.parametrize("kwargs", [{"gain": 1.}, {"dt_s": .2}, {"tau_s": 0}, {"gain": np.nan}])
def test_rejects_unstable_or_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        GradedNetwork(chain(), GradedConfig(**kwargs))
