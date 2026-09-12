import pytest

from semantic_tools.transition_metrics import feeding_release_summary, match_need_release


def event(time, active, concept=1):
    return {"sim_time_ms": time, "active": active, "concept_id": concept}


def targets(*pairs):
    return [{"time_ms": time, "active": active} for time, active in pairs]


def test_matches_off_after_target_and_keeps_measured_latency():
    result = match_need_release(targets((0, True), (2900, False)),
        [event(300, True), event(1300, True), event(3100, False)], duration_ms=5000)
    assert result[0]["matched_neural_off_ms"] == 3100
    assert result[0]["latency_ms"] == 200
    assert result[0]["observed_active_to_inactive_release"]


def test_same_time_off_can_match():
    result = match_need_release(targets((0, True), (1000, False)),
        [event(100, True), event(1000, False)], duration_ms=2000)
    assert result[0]["latency_ms"] == 0
    assert result[0]["observed_active_to_inactive_release"]


@pytest.mark.parametrize("events", [
    [event(300, True), event(1300, False)],
    [event(300, True), event(1300, True), event(3300, True), event(4300, True)],
    [event(3000, False)],
])
def test_early_off_heartbeats_or_constant_silence_are_not_success(events):
    result = match_need_release(targets((0, True), (2900, False)), events, duration_ms=5000)
    assert result[0]["matched_neural_off_ms"] is None
    assert result[0]["latency_ms"] is None
    assert not result[0]["observed_active_to_inactive_release"]


def test_false_positive_born_after_target_release_is_not_a_supported_release():
    result = match_need_release(targets((0, True), (1000, False)),
        [event(1100, True), event(1300, False)], duration_ms=2000)
    assert result[0]["matched_neural_off_ms"] == 1300
    assert not result[0]["indicator_active_before_target_off"]
    assert not result[0]["observed_active_to_inactive_release"]


def test_off_during_next_active_period_cannot_repair_previous_miss():
    result = match_need_release(targets((0, True), (1000, False), (2000, True), (3000, False)),
        [event(100, True), event(2100, False), event(2500, True), event(3300, False)], duration_ms=4000)
    assert result[0]["matched_neural_off_ms"] is None
    assert result[1]["matched_neural_off_ms"] == 3300


def test_other_concept_off_cannot_count():
    result = match_need_release(targets((0, True), (1000, False)),
        [event(100, True), event(1500, False, concept=2)], duration_ms=2000)
    assert result[0]["matched_neural_off_ms"] is None


def test_release_flag_requires_actual_food_and_connected_cases():
    case = {"episode": "case", "seed": 1, "scene": "physical_feeding", "condition": "connected",
            "duration_s": 2, "ingested_total": 3, "target_transitions": targets((0, True), (1000, False)),
            "events": [event(100, True), event(1300, False)]}
    assert feeding_release_summary([case])["physical_release_supported"]
    case["ingested_total"] = 0
    assert not feeding_release_summary([case])["physical_release_supported"]
    assert not feeding_release_summary([])["physical_release_supported"]


@pytest.mark.parametrize("bad", [targets((100, False)), targets((200, True), (100, False)), targets((0, True), (3000, False))])
def test_bad_target_timing_or_state_is_rejected(bad):
    with pytest.raises(ValueError):
        match_need_release(bad, [], duration_ms=2000)
