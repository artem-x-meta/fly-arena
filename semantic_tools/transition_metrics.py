"""Descriptive matching of need release to subsequent neural off transitions.

This does not change a study's frozen pooled-classification acceptance rule.
"""
from __future__ import annotations

import math


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Transition time must be finite nonnegative model milliseconds")
    return float(value)


def match_need_release(target_transitions, events, *, duration_ms, concept_id=1):
    """Match real off transitions within each target's inactive time interval.

    An off before the target transition never counts. An off at/after the next
    target on belongs to another interval and cannot count either. Heartbeats
    and repeated inactive messages are not transitions. Misses retain None.
    """
    duration = _time(duration_ms)
    targets, outputs = [], []
    previous_time = -1.
    target_state = False
    for item in target_transitions:
        when = _time(item["time_ms"])
        state = item["active"]
        if type(state) is not bool or when < previous_time or when > duration or state == target_state:
            raise ValueError("Invalid target transition sequence")
        targets.append((when, state))
        previous_time, target_state = when, state
    previous_time = -1.
    for item in events:
        if item.get("concept_id") != concept_id:
            continue
        when = _time(item["sim_time_ms"])
        state = item["active"]
        if type(state) is not bool or when < previous_time or when > duration:
            raise ValueError("Invalid neural event sequence")
        outputs.append((when, state))
        previous_time = when
    actual_offs = []
    output_state = False
    for when, state in outputs:
        if output_state and not state:
            actual_offs.append(when)
        output_state = state
    matches = []
    for index, (when, active) in enumerate(targets):
        if active:
            continue
        next_on = next((time for time, state in targets[index + 1:] if state), None)
        end = duration if next_on is None else next_on
        before = False
        for output_time, state in outputs:
            if output_time >= when:
                break
            before = state
        matched = next((time for time in actual_offs if time >= when
                        and (time <= end if next_on is None else time < end)), None)
        matches.append({"target_off_ms": when, "inactive_interval_end_ms": end,
                        "indicator_active_before_target_off": before,
                        "matched_neural_off_ms": matched,
                        "latency_ms": None if matched is None else matched - when,
                        "observed_active_to_inactive_release": before and matched is not None})
    return matches


def feeding_release_summary(episodes):
    cases = []
    for episode in episodes:
        if episode["scene"] != "physical_feeding":
            continue
        matches = match_need_release(episode["target_transitions"], episode["events"],
                                     duration_ms=episode["duration_s"] * 1000)
        cases.append({"episode": episode["episode"], "seed": episode["seed"],
                      "condition": episode["condition"], "intake": episode["ingested_total"],
                      "matches": matches,
                      "release_supported": bool(episode["ingested_total"] > 0 and matches
                            and all(item["observed_active_to_inactive_release"] for item in matches))})
    connected = [item for item in cases if item["condition"] == "connected"]
    return {"scope": "Descriptive post-run release matching; frozen pooled F1 criteria are unchanged",
            "cases": cases, "connected_feeding_cases": len(connected),
            "connected_cases_with_true_off": sum(bool(item["matches"]) for item in connected),
            "connected_cases_with_supported_release": sum(item["release_supported"] for item in connected),
            "physical_release_supported": bool(connected and all(item["release_supported"] for item in connected)),
            "matching_rule": "Prior neural off events do not count; require a real neural True→False transition at/after target off, before its next on, with the indicator active before target off. Missed times remain null."}
