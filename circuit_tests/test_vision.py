import numpy as np
import pytest

from fly_circuit_lab.vision import PaperVision


@pytest.mark.parametrize("full_angle", [20., 40., 70.])
def test_calibrated_rays_recover_a_known_visual_cone(full_angle):
    probe = PaperVision([90., 90.])
    rays = probe.rays[0]
    radius = np.rad2deg(np.arccos(rays[..., 2]))
    image = np.zeros((96, 96, 3), np.uint8)
    image[radius <= full_angle / 2] = [220, 0, 220]
    measured = probe.measure_angle(image, 0)
    assert measured <= full_angle + 1e-10
    assert full_angle - measured < 1.8  # Raster resolution; no hidden scene size.


def test_blind_and_output_interventions_prevent_paper_requests():
    image = np.zeros((2, 96, 96, 3), np.uint8)
    for block in ((), ("output",)):
        probe = PaperVision([90., 90.], block=block)
        probe.step(image)
        image[:, 20:76, 20:76] = [220, 0, 220]
        for _ in range(10):
            result = probe.step(image, blind=not block)
        np.testing.assert_array_equal(result, [0, 0])


def test_probe_checkpoint_preserves_angle_derivative_and_delayed_output():
    a, b = PaperVision([100., 100.]), PaperVision([100., 100.])
    image = np.zeros((2, 96, 96, 3), np.uint8)
    for width in range(4, 13):
        image[0, 48-width:48+width, 48-width:48+width] = [220, 0, 220]
        a.step(image)
    b.set_state(a.get_state())
    for width in range(13, 22):
        image[0, 48-width:48+width, 48-width:48+width] = [220, 0, 220]
        np.testing.assert_array_equal(a.step(image), b.step(image))
