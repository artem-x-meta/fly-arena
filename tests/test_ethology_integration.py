"""Opt-in autonomous organism/body tests; no scripted action assignments."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_ETHOLOGY_INTEGRATION") != "1",
                                reason="Explicit opt-in for bounded OpenGL/body integration experiments")


def test_autonomous_feeding_interventions(tmp_path):
    from scripts.check_ethology_scenarios import feeding_suite
    feeding_suite(tmp_path)


def test_transient_contact_after_grooming_does_not_prevent_physical_feeding(tmp_path):
    from scripts.check_ethology_scenarios import groom_to_feed_regression
    groom_to_feed_regression(tmp_path)


def test_exhaustion_local_dust_and_motor_ablation(tmp_path):
    from scripts.check_ethology_scenarios import guard_suite
    guard_suite(tmp_path)


def test_sleep_wake_deprivation_and_recovery(tmp_path):
    from scripts.check_ethology_scenarios import sleep_suite
    sleep_suite(tmp_path)


@pytest.mark.parametrize("kind", ["feeding", "groom-front", "groom-head", "sleep"])
def test_full_pause_and_fresh_model_resume(kind, tmp_path):
    from scripts.check_ethology_scenarios import feeding_config, groom_config, sleep_config, verify_resume
    cases = {"feeding": (feeding_config(), .9), "groom-front": (groom_config(), .9),
             "groom-head": (groom_config("head"), .9), "sleep": (sleep_config("control"), 1.3)}
    result = verify_resume(kind, *cases[kind], tmp_path)
    assert result["checkpoint_action"] == {"feeding": "FEED", "groom-front": "GROOM_FRONT",
                                           "groom-head": "GROOM_HEAD", "sleep": "SLEEP"}[kind]
