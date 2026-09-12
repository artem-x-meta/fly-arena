from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fly_bio.model import GradedConfig, GradedNetwork
from fly_bio_selectivity.model import DelayedGradedNetwork


def chain():
    # A two-cell causal fixture; naming the input Mi1 is only to select its
    # release buffer. This fixture makes no anatomical claim about retina.
    return SimpleNamespace(incoming=csr_matrix([[0., 0.], [1., 0.]], dtype=np.float32),
                           ports={"retina": [0]}, types=np.array(["Mi1", "target"]))


def model(delay=.018):
    return DelayedGradedNetwork(chain(), delay_by_type_s={"Mi1": delay})


def equal_base(first, second):
    for name in ("voltage", "last_input", "last_external_drive", "last_recurrent_drive", "bias"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.elapsed_steps == second.elapsed_steps


@pytest.mark.parametrize("delays", [{}, {"Mi1": 0.}])
def test_zero_delay_is_bit_identical_to_parent_including_clamp_and_drives(delays):
    plain = GradedNetwork(chain())
    delayed = DelayedGradedNetwork(chain(), delay_by_type_s=delays)
    plain.equilibrate([.62])
    delayed.equilibrate([.62])
    reference = plain.voltage.copy()
    for i, light in enumerate([.1, .7, .4, .8, .6, .2, .9, .3] * 3):
        options = {"input_off": i % 5 == 0}
        if i % 4 == 0:
            options.update(freeze_indices=[0], freeze_voltage=reference)
        plain.step([light], **options)
        delayed.step([light], **options)
        equal_base(plain, delayed)


@pytest.mark.parametrize("delay,first_response_step", [(.018, 3), (.02, 4), (.013, 3)])
def test_impulse_first_arrival_respects_voltage_at_start_of_step(delay, first_response_step):
    net = model(delay)
    net.equilibrate([.5])
    recurrent = []
    for i in range(6):
        net.step([.9 if i == 0 else .5])
        recurrent.append(net.last_recurrent_drive[1])
    assert recurrent[:first_response_step - 1] == [0.] * (first_response_step - 1)
    assert recurrent[first_response_step - 1] > 0


def test_fractional_history_interpolation_uses_two_past_samples():
    net = model(.018)
    net.elapsed_steps = 2
    net.history_cursor = 2
    net.history[:, 0] = [.6, .8, .1]
    net.voltage[0] = .9
    net.step([.5])
    # At t=.02, d=.018 samples t=.002: .2*r(.01)+.8*r(0)=.64.
    np.testing.assert_allclose(net.last_recurrent_drive[1], .8 * (.64 - .5), atol=4e-8)
    assert net.history[2, 0] == np.float32(.9)
    assert net.history_cursor == 0


def test_first_image_equilibrium_fills_prehistory_without_onset_or_clock_advance():
    net = model()
    net.equilibrate([.7])
    expected = net.voltage.copy()
    assert net.elapsed_steps == 0
    np.testing.assert_array_equal(net.history[:, 0], np.full(3, expected[0], np.float32))
    for _ in range(15):
        np.testing.assert_array_equal(net.step([.7]), expected)


def test_late_clamp_preserves_already_released_signal_in_flight():
    net = model()
    for _ in range(10):
        net.step([.9])
    currents = []
    for _ in range(4):
        net.step([.9], freeze_indices=[0], freeze_voltage=np.full(2, .5, np.float32))
        assert net.voltage[0] == .5
        currents.append(net.last_recurrent_drive[1])
    assert currents[0] > 0 and currents[1] > 0
    assert currents[2:] == [0., 0.]


def test_clamp_from_stationary_start_produces_no_downstream_modulation():
    net = model()
    net.equilibrate([.6])
    initial = net.voltage.copy()
    for _ in range(10):
        net.step([.9], freeze_indices=[0], freeze_voltage=initial)
        np.testing.assert_array_equal(net.voltage, initial)


def test_resume_restores_fractional_ring_exactly_and_copies_arrays():
    first = model(.013)
    first.equilibrate([.6])
    for light in [.9, .2, .4, .8, .7]:
        first.step([light])
    state = first.get_state()
    resumed = model(.013)
    resumed.set_state(state)
    state["history"][:] = 1.
    state["base_state"]["voltage"][:] = 1.
    for i, light in enumerate([.9, .6, .1, .5, .8]):
        options = ({"freeze_indices": [0], "freeze_voltage": [.6, .58]} if i > 1 else {})
        first.step([light], **options)
        resumed.step([light], **options)
        equal_base(first, resumed)
        np.testing.assert_array_equal(first.history, resumed.history)
        assert first.history_cursor == resumed.history_cursor


def test_state_family_and_delay_configuration_cannot_be_silently_mixed():
    state = model().get_state()
    with pytest.raises(ValueError, match="Incompatible"):
        GradedNetwork(chain()).set_state(state)
    with pytest.raises(ValueError, match="Incompatible"):
        model().set_state(GradedNetwork(chain()).get_state())
    with pytest.raises(ValueError, match="Incompatible"):
        model(.013).set_state(state)


def test_same_size_changed_graph_and_changed_type_order_rejected():
    state = model().get_state()
    changed = chain()
    changed.incoming.data[0] *= -1
    with pytest.raises(ValueError, match="Incompatible"):
        DelayedGradedNetwork(changed, delay_by_type_s={"Mi1": .018}).set_state(state)
    moved = chain()
    moved.types[:] = moved.types[::-1]
    with pytest.raises(ValueError, match="indices"):
        DelayedGradedNetwork(moved, delay_by_type_s={"Mi1": .018}).set_state(state)


@pytest.mark.parametrize("field,value", [("history_cursor", 1), ("history", [[float('nan')]]),
                                         ("delayed_indices", np.array([.1]))])
def test_invalid_delay_state_rejected_without_mutating_network(field, value):
    net = model()
    saved = net.get_state()
    bad = deepcopy(saved)
    bad[field] = value
    with pytest.raises(ValueError):
        net.set_state(bad)
    np.testing.assert_array_equal(net.voltage, saved["base_state"]["voltage"])
    np.testing.assert_array_equal(net.history, saved["history"])


@pytest.mark.parametrize("delays", [{"Mi1": -.01}, {"Mi1": float("nan")},
                                  {"Mi1": float("inf")}, {"absent": .01},
                                  {"": .01}, {"Mi1": "bad"}])
def test_invalid_delays_rejected(delays):
    with pytest.raises(ValueError):
        DelayedGradedNetwork(chain(), delay_by_type_s=delays)


def test_different_dt_state_rejected_even_if_ring_shapes_match():
    state = model().get_state()
    altered = DelayedGradedNetwork(chain(), GradedConfig(dt_s=.009),
                                   delay_by_type_s={"Mi1": .018})
    with pytest.raises(ValueError, match="Incompatible"):
        altered.set_state(state)
