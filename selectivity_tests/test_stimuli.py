from dataclasses import replace

import numpy as np
import pytest

from fly_bio_selectivity.stimuli import (
    FrameProtocol, MotionProtocol, direction_protocols, opposite_protocol,
    radial_protocols, retinal_mean_control, retinal_movie,
)


@pytest.mark.parametrize("kind,direction", [("grating", "right"),
                                            ("grating", "up"),
                                            ("rings", "expanding")])
@pytest.mark.parametrize("phase", [0.0, np.pi])
def test_opposites_start_identically_and_hold_pre_and_post(kind, direction, phase):
    protocol = MotionProtocol(kind=kind, direction=direction, phase_rad=phase, size=16)
    opposite = opposite_protocol(protocol)
    pre = [0, protocol.pre_s / 2, protocol.pre_s]
    np.testing.assert_array_equal(protocol.frames(pre), opposite.frames(pre))
    for frame in protocol.frames(pre):
        np.testing.assert_array_equal(frame, protocol.baseline_frame())
    for frame in protocol.frames([protocol.pre_s + protocol.stimulus_s,
                                  protocol.duration_s, protocol.duration_s + .2]):
        np.testing.assert_array_equal(frame, protocol.baseline_frame())
    assert not np.array_equal(protocol.frames([protocol.pre_s + .125]),
                              opposite.frames([protocol.pre_s + .125]))


@pytest.mark.parametrize("direction,axis,shift", [("right", 2, 12), ("left", 2, -12),
                                                 ("down", 1, 12), ("up", 1, -12)])
def test_motion_sign_and_speed_are_known_pixel_translations(direction, axis, shift):
    protocol = MotionProtocol(direction=direction)
    baseline = protocol.baseline_frame()
    moved = protocol.frames([protocol.pre_s + .125])[0]
    # .125 seconds is a quarter cycle at 2 Hz: two cycles/96 px -> 12 px.
    np.testing.assert_allclose(moved, np.roll(baseline, shift, axis=axis), atol=1e-7)


def test_full_cycle_dose_equal_for_opposite_directions_after_onset():
    for protocol in [MotionProtocol(size=16),
                     MotionProtocol(kind="rings", direction="expanding", size=16)]:
        times = protocol.times(.01)
        mask = protocol.analysis_mask(times)
        assert mask.sum() == 100
        assert protocol.analysis_window() == (1.0, 2.0)
        forward = protocol.frames(times[mask])
        reverse = opposite_protocol(protocol).frames(times[mask])
        np.testing.assert_allclose(forward.mean(axis=0), .5, atol=2e-7)
        np.testing.assert_allclose(reverse.mean(axis=0), .5, atol=2e-7)
        # Spatial position cannot introduce a direction-dependent brightness dose.
        np.testing.assert_allclose((forward - .5).var(axis=0),
                                   (reverse - .5).var(axis=0), atol=2e-7)


@pytest.mark.parametrize("contrast", [.2, .8])
@pytest.mark.parametrize("eye", ["both", "left", "right"])
def test_contrast_polarity_and_eye_controls(contrast, eye):
    protocol = MotionProtocol(contrast=contrast, eye=eye, size=16)
    frames = protocol.frames([0, .625])
    assert frames.dtype == np.float32
    assert frames.shape == (2, 2, 16, 16, 3)
    assert frames.min() >= .5 * (1 - contrast) - 1e-7
    assert frames.max() <= .5 * (1 + contrast) + 1e-7
    inverted = replace(protocol, polarity=-1).frames([0, .625])
    np.testing.assert_allclose(frames + inverted, 1.0, atol=1e-7)
    if eye != "both":
        inactive = 1 if eye == "left" else 0
        np.testing.assert_array_equal(frames[:, inactive], .5)


def test_retinal_control_matches_nonuniform_sample_distribution_per_eye():
    ports = {"retina": np.arange(8),
             "eye": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
             "uv": np.array([[.5, .5], [.5, .5], [.55, .5], [.6, .5],
                             [0, 0], [.1, .1], [.2, .2], [1, 1]], np.float32)}
    protocol = MotionProtocol(kind="rings", direction="expanding", size=16)
    times = [.5, .625, .75]
    movie = protocol.frames(times)
    uniform = retinal_mean_control(protocol, times, ports)
    raw = retinal_movie(movie, ports)
    matched = retinal_movie(uniform, ports)
    for eye in (0, 1):
        mask = ports["eye"] == eye
        np.testing.assert_allclose(raw[:, mask].mean(axis=1),
                                   matched[:, mask].mean(axis=1), atol=1e-7)
        np.testing.assert_allclose(uniform[:, eye].var(axis=(1, 2, 3)), 0, atol=1e-12)
    assert not np.allclose(movie.mean(axis=(2, 3, 4)), uniform.mean(axis=(2, 3, 4)))


def test_original_dark_disk_assay_remains_available_with_original_sampler():
    from fly_bio.stimuli import FrameProtocol as OriginalProtocol
    from fly_bio.stimuli import retinal_movie as original_sampler
    assert FrameProtocol is OriginalProtocol
    assert retinal_movie is original_sampler
    protocol = FrameProtocol(kind="dark_loom", size=16)
    assert protocol.frames([0, protocol.duration_s]).shape == (2, 2, 16, 16, 3)


def test_protocol_families_record_baselines_and_repeatable_configuration():
    directions = direction_protocols(size=16)
    radial = radial_protocols(size=16)
    assert len(directions) == 8
    assert len(radial) == 4
    assert [x.phase_rad for x in radial] == [0, 0, np.pi, np.pi]
    for protocol in (*directions, *radial):
        config = protocol.protocol_config
        assert config["labels_are_model_inputs"] is False
        assert config["baseline_image"]["sha256_float32"] == (
            opposite_protocol(protocol).protocol_config["baseline_image"]["sha256_float32"])


@pytest.mark.parametrize("kwargs", [dict(kind="loom"), dict(direction="expanding"),
                                    dict(size=7), dict(size=True), dict(eye="other"),
                                    dict(contrast=1.1), dict(background=-.1),
                                    dict(phase_rad=np.nan), dict(temporal_hz=0),
                                    dict(spatial_cycles=0), dict(polarity=0),
                                    dict(pre_s=-.1), dict(post_s=-1),
                                    dict(stimulus_s=0), dict(stimulus_s=.6)])
def test_invalid_protocols_raise(kwargs):
    with pytest.raises(ValueError):
        MotionProtocol(**kwargs)


def test_invalid_times_and_analysis_windows_raise():
    protocol = MotionProtocol(size=16)
    for times in ([-.1], [np.nan], [1, 0], [[1]]):
        with pytest.raises(ValueError):
            protocol.frames(times)
    for discard in (-1, 3, .5, True):
        with pytest.raises(ValueError):
            protocol.analysis_window(discard)
    with pytest.raises(ValueError):
        protocol.times(0)
    assert protocol.frames([]).shape == (0, 2, 16, 16, 3)
