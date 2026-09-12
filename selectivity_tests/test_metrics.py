"""Analytic checks for metrics; these do not assert connectome function."""
import numpy as np
import pytest

from fly_bio_selectivity.metrics import direction_metrics, temporal_metrics


def times():
    # Input [2, 3.5) s, recorded immediately after each 10 ms neural step.
    return 2.01 + np.arange(150) * .01


def test_harmonics_are_per_cell_and_phase_invariant():
    t = times()
    phase = np.array([.31, .31 + np.pi])
    voltage = .5 + .2 * np.cos(2 * np.pi * 2 * t[:, None] + phase)
    result = temporal_metrics(voltage, t, 2.)
    np.testing.assert_allclose(result["h1"], [.2, .2], atol=1e-14)
    np.testing.assert_allclose(result["h2"], 0., atol=1e-14)
    np.testing.assert_allclose(np.exp(1j * result["phase_h1"]), np.exp(1j * phase), atol=1e-13)
    np.testing.assert_allclose(result["cycle_h1"], .2, atol=1e-14)
    # Opposing phases cancel at population level; both cells still respond.
    assert np.ptp(voltage.mean(axis=1)) < 1e-14
    assert result["cycle_h1"].shape == (3, 2)


def test_output_timestamp_shift_changes_phase_not_amplitude():
    t = times()
    voltage = (.5 + .1 * np.cos(2 * np.pi * 2 * (t - .01)))[:, None]
    result = temporal_metrics(voltage, t, 2.)
    np.testing.assert_allclose(result["h1"], .1, atol=1e-14)
    np.testing.assert_allclose(result["phase_h1"], -2 * np.pi * 2 * .01, atol=1e-13)


def test_second_harmonic_and_mean_are_distinct_measurements():
    t = times()
    voltage = (.7 + .1 * np.cos(2 * np.pi * 2 * t)
               + .03 * np.cos(2 * np.pi * 4 * t + .6))[:, None]
    result = temporal_metrics(voltage, t, 2.)
    np.testing.assert_allclose(result["h1"], .1, atol=1e-14)
    np.testing.assert_allclose(result["h2"], .03, atol=1e-14)
    np.testing.assert_allclose(result["mean_voltage_delta"], .2, atol=1e-14)
    np.testing.assert_allclose(result["mean_release_delta"], .2, atol=1e-14)
    np.testing.assert_allclose(result["cycle_mean_release"], .2, atol=1e-14)


def test_release_rectification_can_have_positive_mean_without_voltage_mean():
    t = times()
    voltage = np.sin(2 * np.pi * 2 * t)[:, None]
    result = temporal_metrics(voltage, t, 2., gray_voltage=0.)
    np.testing.assert_allclose(result["mean_voltage_delta"], 0., atol=1e-14)
    assert result["mean_release_delta"][0] == pytest.approx(1 / np.pi, abs=.001)
    np.testing.assert_allclose(result["h2"], 0., atol=1e-14)


def test_stationary_pattern_has_zero_harmonics_but_nonzero_gray_difference():
    t = times()
    voltage = np.broadcast_to([.5, .8], (len(t), 2))
    result = temporal_metrics(voltage, t, 2., gray_voltage=[.5, .5])
    np.testing.assert_allclose(result["h1"], 0., atol=1e-14)
    np.testing.assert_allclose(result["h2"], 0., atol=1e-14)
    np.testing.assert_allclose(result["mean_release_delta"], [0., .3], atol=1e-14)


@pytest.mark.parametrize("change", ["extra_endpoint", "incomplete_cycle", "nonuniform", "backwards", "bad_frequency"])
def test_invalid_analysis_windows_are_rejected(change):
    t = times()
    frequency = 2.
    if change == "extra_endpoint":
        t = np.append(t, 3.51)
    elif change == "incomplete_cycle":
        t = t[:-1]
    elif change == "nonuniform":
        t[12] += .002
    elif change == "backwards":
        t = t[::-1]
    else:
        frequency = 2.1
    with pytest.raises(ValueError):
        temporal_metrics(np.full((len(t), 1), .5), t, frequency)


def stable_cycles(test):
    return np.repeat(test[:, None, :, :], 3, axis=1)


def test_preferred_direction_is_locked_and_cannot_be_reselected_on_test():
    calibration = np.array([[4.], [1.], [1.], [1.]])
    test = np.array([[[2.], [.5], [.5], [.5]],
                     [[.5], [.5], [2.], [.5]]])
    result = direction_metrics(calibration, test, 1e-6, stable_cycles(test))
    np.testing.assert_array_equal(result["preferred_direction"], [0])
    np.testing.assert_array_equal(result["null_direction"], [2])
    np.testing.assert_allclose(result["phase_dsi"][:, 0], [.6, -.6])
    np.testing.assert_array_equal(result["phase_success"][:, 0], [True, False])
    assert not result["success"][0]


def test_both_phases_and_both_locked_directions_must_be_stable():
    calibration = np.array([[4., 1.], [1., 4.], [1., 1.], [1., 1.]])
    test = np.repeat((calibration / 2)[None, :, :], 2, axis=0)
    cycles = stable_cycles(test)
    # First cell's nonpreferred response still drifts, in only one phase.
    cycles[1, -2, 2, 0] = .25
    result = direction_metrics(calibration, test, 1e-6, cycles)
    np.testing.assert_array_equal(result["success"], [False, True])
    assert result["phase_cycle_relative_change"][1, 1, 0] == pytest.approx(.5)
    np.testing.assert_array_equal(result["directional_success_without_stationarity"], [True, True])


def test_unresponsive_cells_are_retained_and_never_pass():
    calibration = np.array([[0., 4., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]])
    test = np.zeros((2, 4, 3))
    test[:, :, 1] = np.array([4e-8, 1e-8, 1e-8, 1e-8])
    # An arbitrary calibration tie at zero must not become a preferred axis.
    test[:, :, 2] = np.array([4., 1., 1., 1.])
    result = direction_metrics(calibration, test, 1e-6, stable_cycles(test))
    assert result["success"].shape == (3,)
    assert not result["success"].any()
    assert np.isnan(result["phase_dsi"]).all()


def test_stationarity_evidence_cannot_be_omitted_for_full_success():
    calibration = np.array([[4.], [1.], [1.], [1.]])
    test = np.repeat(calibration[None, :, :], 2, axis=0)
    result = direction_metrics(calibration, test, 1e-6)
    assert result["directional_success_without_stationarity"][0]
    assert not result["stationarity_checked"]
    assert not result["success"][0]
    assert np.isnan(result["phase_cycle_relative_change"]).all()


def test_zero_null_response_uses_floor_for_stationarity():
    calibration = np.array([[4.], [1.], [0.], [1.]])
    test = np.repeat(calibration[None, :, :], 2, axis=0)
    cycles = stable_cycles(test)
    cycles[0, -1, 2, 0] = 5e-8
    result = direction_metrics(calibration, test, 1e-6, cycles)
    assert result["success"][0]
    assert result["phase_cycle_relative_change"][0, 1, 0] == pytest.approx(.05)


@pytest.mark.parametrize("change", ["negative", "nan", "phase_shape", "cycle_shape", "zero_floor"])
def test_invalid_direction_inputs_are_rejected(change):
    calibration = np.ones((4, 2))
    test = np.ones((2, 4, 2))
    cycles = stable_cycles(test)
    floor = 1e-6
    if change == "negative":
        calibration[0, 0] = -1
    elif change == "nan":
        test[1, 2, 1] = np.nan
    elif change == "phase_shape":
        test = test[:1]
    elif change == "cycle_shape":
        cycles = cycles[:, :1]
    else:
        floor = 0.
    with pytest.raises(ValueError):
        direction_metrics(calibration, test, floor, cycles)
