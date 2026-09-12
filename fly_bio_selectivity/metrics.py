"""Predeclared per-cell periodic-response metrics; no stimulus-label fitting.

The harmonic coefficients refer to voltage. Mean release is calculated from
the model's rectifier separately: harmonic amplitude is not tonic excitation.
These helpers do not select cells, aggregate types, or rename screen axes.
"""
from __future__ import annotations

import numpy as np


def _positive_finite_scalar(value, name):
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _harmonic(voltage, times_s, frequency_hz, harmonic=1):
    centered = voltage - voltage.mean(axis=0, dtype=np.float64)
    phase = np.exp(-2j * np.pi * harmonic * frequency_hz * times_s)
    return (2.0 / len(times_s)) * (phase @ centered)


def temporal_metrics(voltage, times_s, frequency_hz, gray_voltage=.5):
    """Measure one fixed integer-cycle analysis window, independently per cell.

    ``voltage`` is T x N; ``times_s`` are the T *output* timestamps. The
    runner's +10 ms output shift changes phase but not amplitudes. Times must
    be uniform, with an integer number of samples per period and an integer
    number of complete periods. No endpoint is appended or window selected.

    ``gray_voltage`` is one scalar or a length-N stationary reference. The
    returned ``h1``, ``h2``, ``phase_h1``, ``mean_voltage_delta`` and
    ``mean_release_delta`` have shape N. ``cycle_h1`` and
    ``cycle_mean_release`` have shape C x N. The latter also subtracts the
    stationary gray release, despite its shorter key name. Harmonic phase
    is relative to a cosine; phase of a zero amplitude is noninformative.
    """
    values = np.asarray(voltage, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    frequency = _positive_finite_scalar(frequency_hz, "frequency_hz")
    if (values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1
            or not np.isfinite(values).all()):
        raise ValueError("voltage must be a finite T x N array with T>=2, N>=1")
    if (times.shape != (values.shape[0],) or not np.isfinite(times).all()):
        raise ValueError("times_s must contain one finite timestamp per sample")
    spacing = np.diff(times)
    dt = float(spacing[0])
    if dt <= 0 or not np.allclose(spacing, dt, rtol=1e-7, atol=1e-11):
        raise ValueError("times_s must be strictly increasing and uniformly sampled")
    samples_per_cycle = 1.0 / (frequency * dt)
    cycle_length = int(round(samples_per_cycle))
    # Two harmonics require more than four samples per period, so H2 cannot
    # be aliased into the fundamental or be a Nyquist singleton.
    if (cycle_length < 5 or not np.isclose(samples_per_cycle, cycle_length,
                                          rtol=1e-7, atol=1e-9)):
        raise ValueError("Each period must contain an integer number of at least five samples")
    if len(times) % cycle_length:
        raise ValueError("The analysis window must contain an integer number of full cycles")
    gray = np.asarray(gray_voltage, dtype=np.float64)
    if gray.ndim > 1 or (gray.ndim == 1 and gray.shape != (values.shape[1],)):
        raise ValueError("gray_voltage must be a scalar or one value per cell")
    if not np.isfinite(gray).all():
        raise ValueError("gray_voltage must be finite")
    gray_release = np.maximum(gray, 0.)
    first = _harmonic(values, times, frequency)
    second = _harmonic(values, times, frequency, harmonic=2)
    cycles = values.reshape(-1, cycle_length, values.shape[1])
    cycle_times = times.reshape(-1, cycle_length)
    return {
        "h1": np.abs(first),
        "h2": np.abs(second),
        "phase_h1": np.angle(first),
        "mean_voltage_delta": values.mean(axis=0, dtype=np.float64) - gray,
        "mean_release_delta": np.maximum(values, 0.).mean(axis=0, dtype=np.float64) - gray_release,
        "cycle_h1": np.stack([
            np.abs(_harmonic(cycle, sample_times, frequency))
            for cycle, sample_times in zip(cycles, cycle_times)
        ]),
        "cycle_mean_release": np.maximum(cycles, 0.).mean(axis=1, dtype=np.float64) - gray_release,
    }


def direction_metrics(cal_amplitudes, test_amplitudes_by_phase, floor,
                      cycle_amplitudes_by_phase=None):
    """Lock each cell's PD on calibration and test it without reselection.

    Calibration shape is 4 x N, with opposing directions separated by two
    indices: 0/2 and 1/3. Test shape is 2 x 4 x N. All amplitudes must be
    finite and nonnegative. Ties on calibration select the first index;
    responsiveness is checked separately, so four zeros never pass.

    Optional cycle amplitudes have shape 2 x C x 4 x N, C>=2. Both PD and
    ND must change by <=10% between the last two cycles, dividing by the
    maximum of their two amplitudes and the numerical floor. Missing cycle
    data leave DS available but cannot establish full ``success``.

    Keys ``phase_*`` have shape 2 x N except
    ``phase_cycle_relative_change``, shaped 2 x 2 x N (phase, PD/ND, cell).
    ``success`` and ``directional_success_without_stationarity`` have shape
    N and require both phases. ``phase_dsi`` is NaN wherever calibration or
    that phase's locked preferred response does not exceed the floor.
    All cells remain in the arrays, including unresponsive/unstable cells.
    """
    cal = np.asarray(cal_amplitudes, dtype=np.float64)
    test = np.asarray(test_amplitudes_by_phase, dtype=np.float64)
    response_floor = _positive_finite_scalar(floor, "floor")
    if (cal.ndim != 2 or cal.shape[0] != 4 or cal.shape[1] < 1
            or not np.isfinite(cal).all() or np.any(cal < 0)):
        raise ValueError("cal_amplitudes must be a finite nonnegative 4 x N array")
    if (test.shape != (2, 4, cal.shape[1]) or not np.isfinite(test).all()
            or np.any(test < 0)):
        raise ValueError("test_amplitudes_by_phase must be a finite nonnegative 2 x 4 x N array")
    columns = np.arange(cal.shape[1])
    pd = np.argmax(cal, axis=0)
    nd = (pd + 2) % 4
    calibration_responsive = cal[pd, columns] > response_floor
    preferred = test[:, pd, columns]
    null = test[:, nd, columns]
    responsive = (preferred > response_floor) & calibration_responsive[None, :]
    dsi = np.full(preferred.shape, np.nan, dtype=np.float64)
    np.divide(preferred - null, preferred + null, out=dsi, where=responsive)
    directional = responsive & (dsi >= .3)
    relative_change = np.full((2, 2, cal.shape[1]), np.nan, dtype=np.float64)
    stable = np.zeros((2, cal.shape[1]), dtype=bool)
    stationarity_checked = cycle_amplitudes_by_phase is not None
    if stationarity_checked:
        cycles = np.asarray(cycle_amplitudes_by_phase, dtype=np.float64)
        if (cycles.ndim != 4 or cycles.shape[0] != 2 or cycles.shape[1] < 2
                or cycles.shape[2:] != (4, cal.shape[1])
                or not np.isfinite(cycles).all() or np.any(cycles < 0)):
            raise ValueError("cycle amplitudes must be finite nonnegative 2 x C x 4 x N, C>=2")
        for direction_index, direction in enumerate((pd, nd)):
            previous = cycles[:, -2, direction, columns]
            last = cycles[:, -1, direction, columns]
            denominator = np.maximum(np.maximum(previous, last), response_floor)
            relative_change[:, direction_index, :] = np.abs(last - previous) / denominator
        stable = np.all(relative_change <= .10, axis=1)
    phase_success = directional & stable
    return {
        "preferred_direction": pd,
        "null_direction": nd,
        "calibration_responsive": calibration_responsive,
        "phase_preferred_amplitude": preferred,
        "phase_null_amplitude": null,
        "phase_dsi": dsi,
        "phase_responsive": responsive,
        "phase_stable": stable,
        "phase_cycle_relative_change": relative_change,
        "phase_success": phase_success,
        "success": np.all(phase_success, axis=0),
        "directional_success_without_stationarity": np.all(directional, axis=0),
        "stationarity_checked": stationarity_checked,
    }
