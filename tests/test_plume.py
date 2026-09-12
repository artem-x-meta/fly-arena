import numpy as np

from fly_arena.environment import FoodPatch
from fly_arena.plume import OdorPlume


def test_packets_advect_only_with_wind_and_sample_is_local():
    food = [FoodPatch("a", 0, 0)]
    plume = OdorPlume(food, {"prefill_s": 0, "wind_direction_deg": 0, "wind_swing_deg": 0,
                            "wind_speed_mm_s": 2, "emission_rate_hz": 0})
    plume.packets = np.array([[0., 0., 0., 1.]])
    for _ in range(100):
        plume.advance(food, .01)
    np.testing.assert_allclose(plume.packets[0, :3], [2, 0, 1], atol=1e-12)
    assert plume.sample([2, 0, 1]) > plume.sample([0, 0, 1]) * 100


def test_odor_field_resume_preserves_rng_and_next_samples_exactly():
    food = [FoodPatch("a", 0, 0)]
    plume = OdorPlume(food, seed=101)
    restored = OdorPlume(food, seed=999)
    restored.set_state(plume.get_state())
    for _ in range(100):
        plume.advance(food, .01)
        restored.advance(food, .01)
        assert plume.sample([-3, .2, 1]) == restored.sample([-3, .2, 1])
        np.testing.assert_array_equal(plume.packets, restored.packets)


def test_field_has_real_intermittency_and_bounded_storage():
    food = [FoodPatch("a", 0, 0)]
    plume = OdorPlume(food, {"wind_swing_deg": 0, "max_packets": 128}, seed=101)
    samples = []
    for _ in range(1000):
        plume.advance(food, .01)
        samples.append(plume.sample([-4, 0, 1]))
        assert len(plume.packets) <= 128
    assert min(samples) < .03 and max(samples) > .3


def test_odor_only_source_does_not_require_food_or_calories():
    source = FoodPatch("odor", 0, 0, amount=0, taste=0, energy_density=0, odor_when_empty=True)
    plume = OdorPlume([source], seed=101)
    assert len(plume.packets) > 0
    off = OdorPlume([FoodPatch("empty", 0, 0, amount=0)], seed=101)
    assert len(off.packets) == 0
