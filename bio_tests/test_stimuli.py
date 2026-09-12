"""Display and sensory-boundary tests independent of MuJoCo and graph size."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fly_bio.stimuli import (FrameProtocol, KINDS, baseline_frames,
                             matched_retinal_flash, retinal_movie, stimulus_frames)


@pytest.mark.parametrize("kind", KINDS)
def test_all_protocols_have_established_start_and_hold_end(kind):
    protocol = FrameProtocol(kind=kind, size=24)
    times = [0., .1, protocol.pre_s, protocol.pre_s + protocol.stimulus_s,
             protocol.duration_s]
    frames = protocol.frames(times)
    assert frames.shape == (5, 2, 24, 24, 3)
    assert frames.dtype == np.float32
    assert frames.min() >= 0 and frames.max() <= 1
    np.testing.assert_array_equal(frames[0], frames[1])
    np.testing.assert_array_equal(frames[1], frames[2])
    np.testing.assert_array_equal(frames[-2], frames[-1])
    np.testing.assert_array_equal(frames[0], protocol.baseline_frame())
    json.dumps(protocol.protocol_config, allow_nan=False)


def test_loom_and_contraction_are_exact_reverses_without_onset_flash():
    protocol = FrameProtocol(size=48)
    times = protocol.pre_s + np.linspace(0., protocol.stimulus_s, 17)
    approach = protocol.frames(times)
    contraction = stimulus_frames("shrinking", times, size=48)
    np.testing.assert_array_equal(approach, contraction[::-1])
    area = (approach[:, 0, :, :, 0] < protocol.background).sum(axis=(1, 2))
    assert area[0] > 0 and np.all(np.diff(area) >= 0)
    assert area[-1] > 100 * area[0]
    shrink = FrameProtocol(kind="shrinking", size=48)
    np.testing.assert_array_equal(shrink.frames([0.])[0], contraction[0])
    np.testing.assert_array_equal(baseline_frames(shrink, 3), np.repeat(contraction[:1], 3, axis=0))


@pytest.mark.parametrize("eye, active, inactive", [("left", 0, 1), ("right", 1, 0)])
def test_eye_specific_and_input_off_controls(eye, active, inactive):
    times = [0., .7, 1.1]
    frames = stimulus_frames("dark_loom", times, eye=eye, size=32)
    assert np.ptp(frames[:, active]) > .3
    np.testing.assert_array_equal(frames[:, inactive], np.full_like(frames[:, inactive], .5))
    off = stimulus_frames("input_off", times, eye=eye, size=32)
    np.testing.assert_array_equal(off, np.full_like(off, .5))


def test_matched_flash_removes_spatial_structure_and_matches_loom_mean():
    times = np.linspace(0., 1.5, 31)
    loom = stimulus_frames("dark_loom", times, size=48, eye="left")
    flash = stimulus_frames("matched_flash", times, size=48, eye="left")
    np.testing.assert_allclose(flash.mean(axis=(2, 3, 4), dtype=np.float64),
                               loom.mean(axis=(2, 3, 4), dtype=np.float64), atol=2e-8)
    assert np.all(np.ptp(flash, axis=(2, 3, 4)) == 0)
    assert flash[-1, 0, 0, 0, 0] < flash[0, 0, 0, 0, 0]


def test_bright_loom_is_same_geometry_opposite_contrast_at_gray_baseline():
    times = [0., .7, 1.1]
    dark = stimulus_frames("dark_loom", times, size=48)
    bright = stimulus_frames("bright_loom", times, size=48)
    np.testing.assert_allclose(dark + bright, 1., atol=1e-7)
    np.testing.assert_array_equal(dark < .5, bright > .5)


def test_retinal_mean_flash_matches_nonuniform_sampling_and_eye_asymmetry():
    ports = {"retina": [1, 8, 90, 40, 15], "eye": [0, 0, 0, 1, 1],
             "uv": [[.5, .5], [.51, .5], [.7, .5], [.1, .1], [.5, .5]]}
    movie = stimulus_frames("dark_loom", [0., .7, 1.1], size=48)
    flash = matched_retinal_flash(movie, ports)
    original_retina, flash_retina = retinal_movie(movie, ports), retinal_movie(flash, ports)
    for eye in (0, 1):
        mask = np.asarray(ports["eye"]) == eye
        np.testing.assert_allclose(original_retina[:, mask].mean(axis=1),
                                   flash_retina[:, mask].mean(axis=1), atol=1e-7)
    assert np.all(np.ptp(flash, axis=(2, 3, 4)) == 0)
    assert abs(flash[0, 0, 0, 0, 0] - flash[0, 1, 0, 0, 0]) > .05
    # Pixel-area matching would miss the much denser center sampling in this eye.
    pixel_flash = stimulus_frames("matched_flash", [0., .7, 1.1], size=48)
    assert abs(pixel_flash[0, 0, 0, 0, 0] - flash[0, 0, 0, 0, 0]) > .1


def test_translation_moves_center_across_display_with_constant_angular_diameter():
    protocol = FrameProtocol(kind="translating", size=96)
    frames = protocol.frames([.3, .7, 1.1])[:, 0, :, :, 0]
    centers = [np.nonzero(frame < .5)[1].mean() for frame in frames]
    assert centers[0] < 15 and 46 < centers[1] < 49 and centers[2] > 80
    # Perspective changes planar pixel area; size is fixed in angular degrees.
    assert protocol.protocol_config["translating_diameter_deg"] == 10.


def test_retinal_sampler_exactly_matches_existing_brain_rule_and_order():
    from fly_arena.brain import Brain

    rng = np.random.default_rng(719)
    frames = rng.integers(0, 256, (4, 2, 9, 13, 3), dtype=np.uint8)
    ports = {"retina": [42, 9, 503, 11, 17], "eye": [1, 0, 1, 1, 0],
             "uv": [[0, 1], [1, 0], [.5, .5], [.38, .28], [.99, .4]]}
    proxy = SimpleNamespace(eyes=np.asarray(ports["eye"]), uv=np.asarray(ports["uv"], np.float32))
    expected = np.stack([Brain.sample_eyes(proxy, frame) for frame in frames])
    actual = retinal_movie(frames, ports)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(retinal_movie(frames.astype(np.float32) / 255, ports), expected)


def test_retinal_mapping_preserves_lateralization_and_sees_disk_expansion():
    ports = {"retina": [100, 300, 200, 500], "eye": [0, 1, 0, 1],
             "uv": [[.5, .5], [.5, .5], [.7, .5], [.7, .5]]}
    movie = retinal_movie(stimulus_frames("dark_loom", [0., 1.1], size=96, eye="left"), ports)
    np.testing.assert_allclose(movie[:, [1, 3]], .5, atol=1e-7)
    np.testing.assert_allclose(movie[:, 0], .1, atol=1e-7)
    assert movie[0, 2] > .49 and movie[1, 2] < .11


@pytest.mark.parametrize("kwargs", [{"size": 7}, {"fov_deg": 180}, {"stimulus_s": 0},
                                    {"contrast": 1.1}, {"background": float("nan")},
                                    {"max_diameter_deg": 100}, {"eye": "L"}])
def test_invalid_display_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        FrameProtocol(**kwargs)


def test_sampler_rejects_ambiguous_brightness_and_bad_mapping():
    ports = {"retina": [4], "eye": [0], "uv": [[.5, .5]]}
    frames = np.full((1, 2, 8, 8, 3), 128., np.float32)
    with pytest.raises(ValueError, match="normalized"):
        retinal_movie(frames, ports)
    with pytest.raises(ValueError, match="Eye indices"):
        retinal_movie(frames / 255, {**ports, "eye": [2]})
    with pytest.raises(ValueError, match="nondecreasing"):
        stimulus_frames("dark_loom", [1., 0.])


def test_protocol_time_grid_and_empty_movie_are_well_defined():
    protocol = FrameProtocol()
    times = protocol.times(.01)
    assert len(times) == 150 and times[0] == 0 and times[-1] < protocol.duration_s
    assert protocol.frames([]).shape == (0, 2, 96, 96, 3)
    assert baseline_frames(protocol, 0).shape == (0, 2, 96, 96, 3)
