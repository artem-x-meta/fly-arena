import math

import numpy as np
import pytest

from fly_circuit_lab.gf_model import evaluate, GFStream, centered_smooth, parameters
from fly_circuit_lab.protocols import looming


def test_published_coefficients_and_units_match_independent_scalar_reference():
    t = np.arange(0, .101, .0005)
    r = evaluate(t, np.full_like(t, 42.), np.full_like(t, 100.))
    assert r["lc4_mv"][-1] == pytest.approx(.0002567 * 100)
    assert r["lplc2_mv"][-1] == pytest.approx(1.7)
    i1 = -.53 + .59 / (1 + math.exp(-(42 - 66) / -11))
    i2 = -.52 * math.exp(-(42 - 26)**2 / (2 * 7.8**2))
    expected = 1.62 * .02567 + 1.45 * 1.7 + 2.27 * i1 + i2
    assert r["gf_raw_mv"][-1] == pytest.approx(expected, abs=1e-12)


def test_lc4_latency_is_19ms_and_voltage_is_linear_in_degrees_per_second():
    t = np.arange(0, .05, .0005)
    slow = evaluate(t, np.zeros_like(t), np.full_like(t, 100.))
    fast = evaluate(t, np.zeros_like(t), np.full_like(t, 200.))
    assert np.all(slow["lc4_mv"][t < .019] == 0)
    assert slow["lc4_mv"][t >= .019].min() == pytest.approx(.02567)
    np.testing.assert_allclose(fast["lc4_mv"], 2 * slow["lc4_mv"])


def test_lplc2_peaks_at_delayed_full_42_degree_size():
    t = np.arange(0, 9., .0005)
    angle = 10 * t
    r = evaluate(t, angle, np.full_like(t, 10.))
    peak_time = t[np.argmax(r["lplc2_mv"])]
    assert peak_time == pytest.approx(4.2 + .019, abs=.00051)


def test_lc4_ablation_removes_its_dependent_inhibition_only():
    t, angle, speed, _ = looming(40)
    control = evaluate(t, angle, speed)
    lc4_off = evaluate(t, angle, speed, block=["lc4"])
    assert np.all(lc4_off["lc4_mv"] == 0)
    assert np.all(lc4_off["i2_mv"] == 0)
    np.testing.assert_array_equal(lc4_off["lplc2_mv"], control["lplc2_mv"])
    np.testing.assert_array_equal(lc4_off["i1_mv"], control["i1_mv"])


def test_no_input_and_blocked_output_do_not_create_a_motor_voltage():
    t, angle, speed, _ = looming(40)
    for block in (["input"], ["output"]):
        r = evaluate(t, angle, speed, block=block)
        np.testing.assert_allclose(r["gf_delta_mv"], 0, atol=1e-14)


def test_causal_stream_matches_unsmoothed_equations_at_sample_times():
    model = GFStream(channels=1)
    t = np.arange(0, .12, .0005)
    angle = np.clip(800 * t, 0, 90)
    speed = np.where(angle < 90, 800., 0.)
    expected = evaluate(t, angle, speed)
    actual = []
    for a, v in zip(angle, speed):
        actual.append(model.step([a], [v], .0005)["gf_delta_mv"][0])
    np.testing.assert_allclose(actual, expected["gf_delta_mv"], atol=2e-14)


def test_stream_split_calls_and_checkpoint_preserve_delays_and_smoothing():
    whole, split = GFStream(), GFStream()
    a = whole.step([40., 20.], [100., 50.], .04)
    split.step([40., 20.], [100., 50.], .0135)
    restored = GFStream()
    restored.set_state(split.get_state())
    b = restored.step([40., 20.], [100., 50.], .0265)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_centered_smoothing_uses_shrinking_odd_endpoint_windows():
    np.testing.assert_allclose(centered_smooth([1, 3, 8, 1, 2], 5), [1, 4, 3, 11/3, 2])


def test_looming_protocol_uses_full_angle_and_analytic_velocity():
    t, angle, speed, meta = looming(40)
    i = np.argmin(np.abs(t + .08))
    assert angle[i] == pytest.approx(math.degrees(2 * math.atan(.04/.08)), abs=.01)
    assert speed[i] == pytest.approx(math.degrees(.08/(.08**2+.04**2)), abs=.1)
    assert meta["loom_end_s"] == pytest.approx(-.04)


@pytest.mark.parametrize("bad", [float("nan"), -1, 181])
def test_invalid_angles_rejected(bad):
    with pytest.raises(ValueError):
        evaluate([0, .01], [0, bad], [0, 1])
