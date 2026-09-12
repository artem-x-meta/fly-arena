import numpy as np

from fly_arena.search import SurgeCastNavigator
from fly_arena.sensors import NeuralReadout, SensorFrame


def test_odor_encounter_causes_upwind_surge_and_loss_causes_cast():
    nav = SurgeCastNavigator({}, seed=101)
    sense = SensorFrame(wind_body_x=-4, wind_body_y=0)
    nav.step(sense, NeuralReadout(), "ethology-hybrid")
    assert nav.phase == "CAST"
    sense.odor_left = sense.odor_right = .4
    command = nav.step(sense, NeuralReadout(), "ethology-hybrid")
    assert nav.phase == "SURGE"
    assert command[0] == command[1] > 0
    sense.wind_body_x, sense.wind_body_y = 0, -4
    command = nav.step(sense, NeuralReadout(), "ethology-hybrid")
    assert command[1] > command[0]  # Upwind is left in the local body frame.
    sense.odor_left = sense.odor_right = 0
    for _ in range(30):
        nav.step(sense, NeuralReadout(), "ethology-hybrid")
    assert nav.phase == "CAST"


def test_no_wind_and_neural_mode_are_explicit_and_do_not_use_hidden_targets():
    nav = SurgeCastNavigator({})
    sense = SensorFrame(odor_left=.5, odor_right=.4)
    assert np.isfinite(nav.step(sense, NeuralReadout(), "ethology-hybrid")).all()
    assert nav.phase == "NO_WIND"
    np.testing.assert_array_equal(nav.step(sense, NeuralReadout(.2, .7), "ethology-neural"), [.2, .7])
    assert nav.phase == "NEURAL_ONLY"
    np.testing.assert_array_equal(nav.last_engineered, [0, 0])


def test_cast_state_rng_and_future_commands_restore_exactly():
    nav = SurgeCastNavigator({}, seed=101)
    sense = SensorFrame(wind_body_x=-4)
    for _ in range(100):
        nav.step(sense, NeuralReadout(), "ethology-hybrid")
    restored = SurgeCastNavigator({}, seed=99)
    restored.set_state(nav.get_state())
    for _ in range(200):
        np.testing.assert_array_equal(nav.step(sense, NeuralReadout(), "ethology-hybrid"),
                                      restored.step(sense, NeuralReadout(), "ethology-hybrid"))
    assert restored.phase == "CAST"


def test_cast_leg_time_counts_aligned_travel_instead_of_turning():
    nav = SurgeCastNavigator({})
    sense = SensorFrame(wind_body_x=-4)
    for _ in range(30):
        nav.step(sense, NeuralReadout(), "ethology-hybrid")
    assert nav.leg_elapsed == 0
    for _ in range(300):
        sense.wind_body_x, sense.wind_body_y = 0, 4 * nav.cast_sign
        nav.step(sense, NeuralReadout(), "ethology-hybrid")
    assert nav.cast_leg > 1
