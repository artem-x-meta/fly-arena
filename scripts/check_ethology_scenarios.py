"""Bounded real-body acceptance experiments, explicitly without a connectome.

No experiment assigns an action. Conditions change initial stores, food,
contact permissions or scheduled external tactile stimuli only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fly_arena.checkpoint import load_checkpoint, save_checkpoint
from fly_arena.config import load_config, merge_config
from fly_arena.ethology import EthologySimulation
from fly_arena.sensors import DUST_REGIONS


def feeding_config(kind="hungry"):
    config = load_config(scenario="feeding-contact")
    if kind == "satiated":
        config["initial"]["energy"] = 98.0
        config["initial"]["gut_amount"] = 8.0
        config["initial"]["gut_energy_density"] = 0.0
    elif kind == "blocked":
        config["environment"]["mouth_contact_enabled"] = False
    elif kind == "noncaloric":
        config["food"][0]["energy_density"] = 0.0
    elif kind == "empty":
        config["food"][0]["amount"] = 0.0
    elif kind == "depletion":
        config["food"][0]["amount"] = .12
    elif kind != "hungry":
        raise ValueError(kind)
    return config


def sleep_config(kind="wake"):
    config = load_config(scenario="sleep-wake")
    if kind == "control":
        config["events"] = []
    elif kind == "deprived":
        # Non-overlapping local pulses prevent a full quiet entry interval.
        # Each pulse remains below the sleeping threshold; no sleep-duration
        # assignment or direct manipulation of S implements the response.
        config["events"] = [dict(kind="wake", time_s=round(i * .3, 4), amplitude=.5,
                                 position=[0, 0, 1], radius=30, duration_s=.15)
                             for i in range(10)]
    elif kind != "wake":
        raise ValueError(kind)
    return config


def groom_config(region="front"):
    config = load_config(scenario="sleep-wake")
    config["events"] = []
    config["initial"].update(energy=85.0, sleep_pressure=.1)
    dust = dict.fromkeys(DUST_REGIONS, 0.0)
    if region == "front":
        dust.update(front_left=1.0, front_right=1.0)
    else:
        dust.update(head=1.0, antenna_left=.5, antenna_right=.5)
    config["initial"]["dust_by_region"] = dust
    return config


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def execute_case(name, config, seconds, *, seed=1, output=None, motor_off=False):
    started = time.perf_counter()
    simulation = EthologySimulation(config, seed=seed, brain_enabled=False, motor_off=motor_off)
    rows, transitions = [], []
    physical_contact_s = permitted_contact_s = sliding_mm = 0.0
    first_intake_s = None
    initial_dust_signal = None
    peak_energy = simulation.organism.state.energy
    peak_gut = simulation.organism.state.gut_amount
    csv_stream = csv_writer = None
    try:
        if output:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            csv_stream = (output / f"{name}-seed{seed}.csv").open("w", newline="", encoding="utf-8")
        for index in range(round(seconds / .01)):
            result = simulation.step()
            if initial_dust_signal is None:
                initial_dust_signal = dict(simulation.last_frame.dust_afferents)
            row = simulation.telemetry()
            row["mouth_contact"] = simulation.last_frame.mouth_contact
            row["mouth_z_mm"] = float(simulation.arena.mouth_position[2])
            peak_energy = max(peak_energy, row["energy"])
            peak_gut = max(peak_gut, row["gut_amount"])
            physical_contact_s += sum(simulation.last_events.mouth_contact_s_by_food.values())
            permitted_contact_s += sum(simulation.last_events.contact_s_by_food.values())
            sliding_mm += sum(simulation.last_events.grooming_sliding_by_pair.values())
            if row["ingested_total"] > 0 and first_intake_s is None:
                first_intake_s = row["physics_time_s"]
            if result["transition"]:
                transitions.append(result["transition"])
            # Small bounded diagnostic collection, plus complete streamed CSV.
            if index % 10 == 0 or index == round(seconds / .01) - 1:
                rows.append(row)
            if csv_stream:
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
        final = simulation.telemetry()
        result = {"name": name, "seed": seed, "diagnostic": simulation.label,
                  "config": copy.deepcopy(config), "code_digest": simulation.source_digest,
                  "model_digest": simulation.model_digest,
                  "brain_enabled": False, "motor_off": motor_off, "physics_seconds": seconds,
                  "wall_seconds": time.perf_counter() - started,
                  "minimum_upright": simulation.minimum_upright,
                  "action_durations": dict(simulation.action_durations),
                  "transitions": transitions, "first_intake_s": first_intake_s,
                  "physical_mouth_contact_s": physical_contact_s,
                  "permitted_mouth_contact_s": permitted_contact_s,
                  "grooming_sliding_mm": sliding_mm,
                  "initial_dust_afferents": initial_dust_signal,
                  "final_dust_afferents": dict(simulation.last_frame.dust_afferents),
                  "peak_energy": peak_energy, "peak_gut": peak_gut,
                  "initial_energy": simulation.initial_energy,
                  "final": final, "samples": rows}
        assert simulation.minimum_upright > .8, f"{name}: lost stable upright stance"
        assert all(abs(value) < 1e-7 for value in simulation.resource_balances().values()), f"{name}: resource residual"
        if output:
            (output / f"{name}-seed{seed}.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
        print(f"{name} seed={seed}: {seconds:.2f}s physics, {result['wall_seconds']:.1f}s wall, "
              f"actions={result['action_durations']}, intake={final['ingested_total']:.5f}, "
              f"S={final['sleep_pressure']:.3f}", flush=True)
        return result
    finally:
        if csv_stream:
            csv_stream.close()
        simulation.close()


# Declared resume tolerance. A fresh model restores the serialized integration
# state exactly, but MuJoCo keeps no derived data (site poses, contact results)
# in that state: the uninterrupted run reads them one 0.1 ms substep stale,
# while a restored run reads them refreshed by mj_forward. The inverse-kinematic
# motor primitives read site positions, so the two runs part company in the last
# digits. Discrete outcomes -- action, reason, source, owner, phase, transition
# and event counts -- stay exact; only continuous values carry this tolerance.
RESUME_ATOL = 1e-7
RESUME_RTOL = 1e-6


def assert_nested_equal(actual, expected, path="state", atol=0.0, rtol=0.0):
    if isinstance(expected, np.ndarray):
        if atol or rtol:
            # The absolute part scales with the vector, because a state array
            # mixes physical coordinates with near-zero solver warm-start terms.
            scale = max(1.0, float(np.max(np.abs(expected)))) if expected.size else 1.0
            np.testing.assert_allclose(actual, expected, atol=atol * scale, rtol=rtol, err_msg=path)
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_nested_equal(actual[key], expected[key], f"{path}.{key}", atol, rtol)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, item in enumerate(expected):
            assert_nested_equal(actual[index], item, f"{path}[{index}]", atol, rtol)
    elif (atol or rtol) and isinstance(expected, float) and not isinstance(actual, bool):
        assert abs(actual - expected) <= atol + rtol * abs(expected),             f"{path}: {actual!r} != {expected!r} beyond the declared resume tolerance"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def assert_events_equal(actual, expected, path="events"):
    assert type(actual) is type(expected), path
    assert_nested_equal(vars(actual), vars(expected), path, RESUME_ATOL, RESUME_RTOL)
    for field_name in ("contact_s_by_food", "mouth_contact_s_by_food", "grooming_contact_s_by_pair"):
        # Contact durations accumulate a constant substep: an interrupted run
        # must not gain or lose a whole substep of feeding or grooming credit.
        assert getattr(actual, field_name) == getattr(expected, field_name), f"{path}.{field_name}"


def verify_resume(name, config, checkpoint_after_s, output, *, brain_enabled=False, graph=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = Path(graph or ROOT / "data" / "graph") if brain_enabled else None
    original = EthologySimulation(config, brain_enabled=brain_enabled, graph=graph)
    restored = None
    started = time.perf_counter()
    try:
        for _ in range(round(checkpoint_after_s / .01)):
            original.step()
        snapshot = copy.deepcopy(original.get_state())
        checkpoint_action = original.last_decision.action
        for _ in range(4):
            assert original.step(paused=True) is None
        assert_nested_equal(original.get_state(), snapshot)
        path = output / f"resume-{'full-graph-' if brain_enabled else ''}{name}.npz"
        save_checkpoint(path, snapshot, original.compatibility())
        expected_events, expected_results = [], []
        for _ in range(15):
            expected_results.append(copy.deepcopy(original.step()))
            expected_events.append(copy.deepcopy(original.last_events))
        expected = copy.deepcopy(original.get_state())
        restored = EthologySimulation(config, brain_enabled=brain_enabled, graph=graph, warmup=False)
        saved, _ = load_checkpoint(path, expected=restored.compatibility())
        restored.set_state(saved)
        for index in range(15):
            actual_result = restored.step()
            assert_nested_equal(actual_result, expected_results[index], f"step[{index}]",
                                RESUME_ATOL, RESUME_RTOL)
            assert_events_equal(restored.last_events, expected_events[index], f"physical events[{index}]")
        assert_nested_equal(restored.get_state(), expected, "state", RESUME_ATOL, RESUME_RTOL)
        result = {"name": name, "checkpoint_action": checkpoint_action,
                  "checkpoint_physics_s": checkpoint_after_s, "continuation_s": .15,
                  "comparison": f"all_serialized_state_within_atol_{RESUME_ATOL}_rtol_{RESUME_RTOL}",
                  "discrete_comparison": "exact_action_source_owner_phase_transition_and_contact_substeps",
                  "pause": "exact_all_serialized_state",
                  "physical_events": "contact_durations_exact_every_interval", "brain_enabled": brain_enabled,
                  "graph_neurons": original.brain.manifest["neurons"] if brain_enabled else None,
                  "wall_seconds": time.perf_counter() - started}
        print(f"resume-{name}: state within declared tolerance after fresh-model restore, "
              f"action={checkpoint_action}", flush=True)
        return result
    finally:
        original.close()
        if restored:
            restored.close()


def feeding_suite(output):
    results = {kind: execute_case(f"feeding-{kind}", feeding_config(kind), 1.6, output=output)
               for kind in ("hungry", "satiated", "blocked", "noncaloric", "empty", "depletion")}
    hungry = results["hungry"]
    assert hungry["final"]["ingested_total"] > .3
    assert hungry["peak_energy"] > hungry["initial_energy"]
    assert hungry["final"]["ingested_total"] > results["satiated"]["final"]["ingested_total"]
    assert results["blocked"]["final"]["ingested_total"] == 0
    assert results["empty"]["final"]["ingested_total"] == 0
    assert results["noncaloric"]["final"]["ingested_total"] > .1
    assert results["noncaloric"]["peak_energy"] <= results["noncaloric"]["initial_energy"]
    assert abs(results["depletion"]["final"]["ingested_total"] - .12) < 1e-9
    assert results["depletion"]["final"]["food_remaining"] == 0
    return results


def groom_to_feed_regression(output):
    # Head dust and hunger compete on a patch that is already under the mouth,
    # so the sequence is decided by needs, not by walking distance. The guard
    # under test is the transient mouth overlap while the grooming pose is
    # released: it must not be counted as established feeding contact, and it
    # must not prevent the real attempt that follows.
    config = feeding_config()
    dust = dict.fromkeys(DUST_REGIONS, 0.0)
    dust.update(head=1.0, antenna_left=.5, antenna_right=.5)
    config["initial"]["dust_by_region"] = dust
    result = execute_case("groom-to-feed", config, 3.0, output=output)
    actions = [t["action"] for t in result["transitions"]]
    assert "GROOM_HEAD" in actions and "FEED" in actions
    assert result["final"]["ingested_total"] > .1
    return result


def guard_suite(output):
    empty = sleep_config("control")
    empty["initial"].update(energy=.2, sleep_pressure=.05)
    exhausted = execute_case("exhaustion-no-food", empty, 1.2, output=output)
    assert exhausted["final"]["energy"] == 0
    assert exhausted["final"]["ingested_total"] == 0
    assert exhausted["action_durations"].get("EXHAUSTED", 0) > .5
    energy_samples = [row["energy"] for row in exhausted["samples"]]
    assert all(a >= b for a, b in zip(energy_samples, energy_samples[1:]))
    assert all(row["action"] == "EXHAUSTED" for row in exhausted["samples"] if row["physics_time_s"] > .5)

    dirty_cfg = groom_config("head")
    clean_cfg = copy.deepcopy(dirty_cfg)
    clean_cfg["initial"]["dust_by_region"] = dict.fromkeys(DUST_REGIONS, 0.0)
    dirty = execute_case("head-dirty", dirty_cfg, 1.6, output=output)
    clean = execute_case("head-clean", clean_cfg, 1.6, output=output)
    disabled = execute_case("head-motor-off", dirty_cfg, 1.6, output=output, motor_off=True)
    feed_off = execute_case("feeding-motor-off", feeding_config(), 1.6, output=output, motor_off=True)
    assert dirty["action_durations"].get("GROOM_HEAD", 0) > .3
    assert clean["action_durations"].get("GROOM_HEAD", 0) == 0
    assert dirty["initial_dust_afferents"]["head"] > clean["initial_dust_afferents"]["head"]
    assert dirty["final_dust_afferents"]["head"] < dirty["initial_dust_afferents"]["head"]
    assert dirty["grooming_sliding_mm"] > .01 and dirty["final"]["removed_dust"] > .01
    assert dirty["final"]["dust_head"] < dirty_cfg["initial"]["dust_by_region"]["head"]
    assert dirty["final"]["dust_front_left"] + dirty["final"]["dust_front_right"] > .01
    assert disabled["action_durations"].get("GROOM_HEAD", 0) > .3
    assert disabled["grooming_sliding_mm"] == disabled["final"]["removed_dust"] == 0
    for region, amount in dirty_cfg["initial"]["dust_by_region"].items():
        assert disabled["final"][f"dust_{region}"] == amount
    assert feed_off["action_durations"].get("FEED", 0) > .3
    assert feed_off["permitted_mouth_contact_s"] == feed_off["final"]["ingested_total"] == 0
    return {"exhausted": exhausted, "dirty": dirty, "clean": clean,
            "motor_off_head": disabled, "motor_off_feeding": feed_off}


def sleep_suite(output):
    wake = execute_case("sleep-wake", sleep_config("wake"), 3.5, output=output)
    assert any(t["action"] == "SLEEP" and t["time_s"] < 2 for t in wake["transitions"])
    assert not any(t["action"] == "WAKE" and 2 <= t["time_s"] < 2.5 for t in wake["transitions"])
    strong_wake = next(t for t in wake["transitions"] if t["action"] == "WAKE" and t["time_s"] >= 3 - 1e-8)
    wake_complete = next(t for t in wake["transitions"] if t["action"] == "IDLE" and t["time_s"] > strong_wake["time_s"])
    assert wake_complete["time_s"] - strong_wake["time_s"] >= .24
    control = execute_case("sleep-control", sleep_config("control"), 6.5, output=output)
    deprived = execute_case("sleep-deprived", sleep_config("deprived"), 6.5, output=output)
    before_release = lambda result: next(row for row in reversed(result["samples"]) if row["physics_time_s"] <= 3)
    assert before_release(deprived)["sleep_pressure"] > before_release(control)["sleep_pressure"] + .2
    recovery_sleep = lambda result: sum(row["action"] == "SLEEP" for row in result["samples"] if 3 < row["physics_time_s"] < 6.5)
    assert recovery_sleep(deprived) > recovery_sleep(control)
    return {"wake": wake, "control": control, "deprived": deprived,
            "wake_to_idle_delay_s": wake_complete["time_s"] - strong_wake["time_s"]}


def seed_suite(output, seconds=8.0):
    config = load_config()
    # Evaluation uses unchanged model defaults and untouched random seeds;
    # this is reported as bounded repertoire evidence, not validation of a
    # calibrated distribution or natural behavior.
    return [execute_case("multi-needs", config, seconds, seed=seed, output=output) for seed in (11, 12, 13)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "feeding", "groom-feed", "guards", "sleep", "resume", "full-graph-resume", "seeds", "all"), default="smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "ethology-acceptance")
    parser.add_argument("--seed-seconds", type=float, default=8.0)
    parser.add_argument("--graph", type=Path, default=ROOT / "data" / "graph")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"suite": args.suite, "brain_enabled": args.suite == "full-graph-resume", "status": "running", "results": {}}
    try:
        if args.suite == "smoke":
            results["results"]["smoke"] = execute_case("smoke", feeding_config(), .1, output=args.output)
        if args.suite in ("feeding", "all"):
            results["results"]["feeding"] = feeding_suite(args.output)
        if args.suite in ("groom-feed", "all"):
            results["results"]["groom-feed"] = groom_to_feed_regression(args.output)
        if args.suite in ("guards", "all"):
            results["results"]["guards"] = guard_suite(args.output)
        if args.suite in ("sleep", "all"):
            results["results"]["sleep"] = sleep_suite(args.output)
        if args.suite in ("resume", "all"):
            results["results"]["resume"] = [verify_resume(name, cfg, at, args.output) for name, cfg, at in (
                ("feeding", feeding_config(), .9), ("groom-front", groom_config(), .9),
                ("groom-head", groom_config("head"), .9), ("sleep", sleep_config("control"), 1.3))]
        if args.suite in ("seeds", "all"):
            results["results"]["seeds"] = seed_suite(args.output, args.seed_seconds)
        if args.suite == "full-graph-resume":
            results["results"]["resume"] = [verify_resume(name, cfg, at, args.output,
                                                         brain_enabled=True, graph=args.graph)
                                            for name, cfg, at in (
                ("feeding", feeding_config(), .9), ("groom-head", groom_config("head"), .9))]
        results["status"] = "pass"
    except Exception as error:
        results["status"], results["error"] = "fail", repr(error)
        raise
    finally:
        (args.output / f"suite-{args.suite}.json").write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")


if __name__ == "__main__":
    main()
