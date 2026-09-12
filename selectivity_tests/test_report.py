"""Synthetic checks for report gates, not tests of neural selectivity itself."""
import json

import numpy as np
import pytest

from bio_tools.selectivity_report import CONTROLS, dc_stable, finite_json, lplc2_metrics


def signal(mean, h1=.02):
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    return {"mean_release_delta": mean,
            "cycle_mean_release": np.repeat(mean[None, :], 3, axis=0),
            "static_release_delta": np.zeros_like(mean),
            "h1": np.full_like(mean, h1)}


def phase_signals(out=(.1, .1), comparator=.01):
    out = np.atleast_1d(np.asarray(out, dtype=float))
    return {"expanding": signal(out), **{name: signal(np.full_like(out, comparator)) for name in CONTROLS}}


def test_lplc2_requires_each_control_and_both_phases():
    first, second = phase_signals(), phase_signals()
    second["flicker"]["mean_release_delta"][1] = .06
    second["flicker"]["cycle_mean_release"][:, 1] = .06
    result = lplc2_metrics([first, second], 1e-6)
    np.testing.assert_array_equal(result["functional_success"], [True, False])
    assert not result["clamp_available"]
    assert not result["causal_success"].any()


def test_lplc2_static_pattern_is_a_required_control():
    first, second = phase_signals(), phase_signals()
    first["expanding"]["static_release_delta"][0] = .075
    result = lplc2_metrics([first, second], 1e-6)
    np.testing.assert_array_equal(result["functional_success"], [False, True])


def test_lplc2_directional_oscillation_is_not_tonic_excitation():
    phases = [phase_signals(out=(0., -1e-4), comparator=0.) for _ in range(2)]
    for phase in phases:
        phase["expanding"]["h1"][:] = 1.
        phase["contracting"]["h1"][:] = .01
    result = lplc2_metrics(phases, 1e-6)
    assert np.all(result["phase_radial_h1_dsi"] > .9)
    assert not result["functional_success"].any()


def test_lplc2_nonstationary_control_blocks_function():
    first, second = phase_signals(), phase_signals()
    second["up"]["cycle_mean_release"][-2, 0] = 0.
    result = lplc2_metrics([first, second], 1e-6)
    np.testing.assert_array_equal(result["functional_success"], [False, True])


def test_causal_claim_needs_eighty_percent_reduction_in_each_phase():
    phases = [phase_signals(), phase_signals()]
    clamps = [signal([.01, .01]), signal([.01, .03])]
    result = lplc2_metrics(phases, 1e-6, clamps)
    np.testing.assert_array_equal(result["functional_success"], [True, True])
    np.testing.assert_array_equal(result["causal_success"], [True, False])
    np.testing.assert_allclose(result["phase_clamp_reduction"], [[.9, .9], [.9, .7]])


def test_causal_claim_needs_stationary_clamp_signal():
    phases = [phase_signals(), phase_signals()]
    clamps = [signal([.01, .01]), signal([.01, .01])]
    clamps[0]["cycle_mean_release"][-2, 0] = 0.
    result = lplc2_metrics(phases, 1e-6, clamps)
    np.testing.assert_array_equal(result["causal_success"], [False, True])


def test_negative_control_does_not_count_as_excitation():
    phases = [phase_signals(out=(.01, .01), comparator=-.1) for _ in range(2)]
    result = lplc2_metrics(phases, 1e-6)
    assert result["functional_success"].all()
    np.testing.assert_allclose(result["phase_max_control_positive_release"], 0.)


def test_dc_stationarity_uses_floor_near_zero():
    cycles = np.array([[0., 0.], [0., 0.], [5e-7, 2e-6]])
    np.testing.assert_array_equal(dc_stable(cycles, 1e-6), [True, False])


def test_nonfinite_statistics_are_strict_json_null():
    result = finite_json({"values": np.array([np.nan, np.inf, -np.inf, .5]),
                          "count": np.int64(3), "yes": np.bool_(True)})
    serialized = json.dumps(result, allow_nan=False)
    assert json.loads(serialized) == {"values": [None, None, None, .5], "count": 3, "yes": True}
