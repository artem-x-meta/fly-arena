"""Reproducible intervention, observer-equivalence and compatibility checks."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from fly_arena.checkpoint import load_checkpoint, save_checkpoint, code_digest
from fly_arena.config import load_config
from fly_arena.ethology import EthologySimulation
from fly_arena.ethology_run import _json_default
from scripts.check_ethology_scenarios import assert_nested_equal

from .simulation import CircuitSimulation


ROOT = Path(__file__).resolve().parents[1]


def compatibility_report(output):
    baseline = json.loads((ROOT / "runs/circuit-lab-audit/compatibility-baseline.json").read_text())
    changed = [name for name, digest in baseline["files"].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    altered_checkpoints = [name for name, digest in baseline["checkpoints"].items()
                          if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    report = {"legacy_code_digest_before": baseline["legacy_code_digest"], "legacy_code_digest_after": code_digest(),
              "changed_legacy_files": changed, "altered_legacy_checkpoints": altered_checkpoints,
              "checked_files": len(baseline["files"])}
    report["status"] = "pass" if not changed and not altered_checkpoints and code_digest() == baseline["legacy_code_digest"] else "fail"
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "compatibility.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert report["status"] == "pass", report
    return report


def interventions(output, *, seed=1, seconds=3.2, components=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ["neutral", "stimulus", "input_off", "output_off"]
    if components:
        names += ["lc4_off", "lplc2_off", "blind", "motor_off"]
    results = {}
    first_hash = None
    report = {"seed": seed, "seconds": seconds, "conditions": results, "status": "running",
              "interpretation": "Fitted GF voltage requests an engineered running action; old full-LIF GF is measured independently, never relabelled as rescued"}
    for name in names:
        cfg = load_config(ROOT / "circuit_configs/ache2019.toml")
        if name == "neutral":
            cfg["environment"]["looming"][0]["time_s"] = 100.
        block = {"input_off": ["input"], "output_off": ["output"], "lc4_off": ["lc4"], "lplc2_off": ["lplc2"]}.get(name, [])
        sim = CircuitSimulation(cfg, graph=ROOT / "data/graph", seed=seed, control="paper", block=block,
                                blind=name == "blind", motor_off=name == "motor_off")
        try:
            digest = hashlib.sha256()
            for value in (sim.brain.voltage, sim.brain.queue, sim.arena.sim.mj_data.qpos, sim.arena.sim.mj_data.qvel):
                digest.update(value.tobytes())
            state_hash = digest.hexdigest()
            if first_hash is None:
                first_hash = state_hash
            assert first_hash == state_hash
            peak_paper = np.zeros(2)
            peak_lif = np.zeros(2)
            escaped = 0.
            first_escape = None
            transitions = []
            with (output / f"{name}.jsonl").open("w", encoding="utf-8") as trace:
                for i in range(round(seconds * 100)):
                    event = sim.step()
                    peak_paper = np.maximum(peak_paper, sim.paper_probe.last["gf_online_mv"])
                    peak_lif = np.maximum(peak_lif, [sim.last_readout.raw_rates.get(f"escape_output_dnp01_{s}", 0) for s in ("L", "R")])
                    if sim.arena.motor.owner == "ESCAPE":
                        escaped += .01
                        if first_escape is None:
                            first_escape = sim.organism.clocks.physics_time_s
                    if event["transition"]:
                        transitions.append(event["transition"])
                    if i % 5 == 0:
                        trace.write(json.dumps(sim.telemetry(), default=_json_default) + "\n")
            results[name] = {"initial_brain_body_sha256": state_hash, "peak_paper_gf_mv": peak_paper.tolist(),
                             "peak_legacy_gf_hz": peak_lif.tolist(), "escape_physics_s": escaped,
                             "first_escape_s": first_escape, "minimum_upright": sim.minimum_upright,
                             "balances": sim.resource_balances(), "transitions": transitions,
                             "compatibility": sim.compatibility()}
            print(f"seed {seed} {name}: paper GF {peak_paper.round(3)}mV, old GF {peak_lif}Hz, escape {escaped:.2f}s", flush=True)
        finally:
            sim.close()
        (output / "interventions.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    checks = {
        "stimulus_requests_physical_escape": results["stimulus"]["escape_physics_s"] > .1,
        "neutral_no_escape": results["neutral"]["escape_physics_s"] == 0.,
        "input_ablation_zero_response": max(results["input_off"]["peak_paper_gf_mv"]) < 1e-10 and results["input_off"]["escape_physics_s"] == 0.,
        "output_block_no_legacy_detector_bypass": max(results["output_off"]["peak_paper_gf_mv"]) < 1e-10 and results["output_off"]["escape_physics_s"] == 0.,
        "body_stays_upright": all(r["minimum_upright"] > .8 for r in results.values()),
        "resources_conserved": all(abs(x) < 1e-7 for r in results.values() for x in r["balances"].values()),
    }
    if components:
        checks["blind_no_escape"] = results["blind"]["escape_physics_s"] == 0.
        checks["motor_off_no_escape"] = results["motor_off"]["escape_physics_s"] == 0.
    report["checks"] = checks
    report["status"] = "pass" if all(checks.values()) else "fail"
    (output / "interventions.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    assert all(checks.values()), checks
    return report


def observer_equivalence(output, *, seconds=2.5, life_profile=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / ("configs/search-renewing.toml" if life_profile else "circuit_configs/ache2019.toml"))
    original = EthologySimulation(cfg, graph=ROOT / "data/graph", seed=1)
    observer = None
    try:
        observer = CircuitSimulation(cfg, graph=ROOT / "data/graph", seed=1, control="observe")
        for _ in range(round(seconds * 100)):
            original.step()
            observer.step()
            assert asdict(original.last_decision) == asdict(observer.last_decision)
            np.testing.assert_array_equal(original.arena.sim.mj_data.qpos, observer.arena.sim.mj_data.qpos)
            np.testing.assert_array_equal(original.brain.voltage, observer.brain.voltage)
            np.testing.assert_array_equal(original.brain.queue, observer.brain.queue)
            assert original.organism.get_state() == observer.organism.get_state()
        report = {"status": "pass", "seconds": seconds, "scope": "full graph + body + organism + decisions",
                  "comparison": "exact each 10ms interval", "actions": original.action_durations}
        (output / ("life-observer-equivalence.json" if life_profile else "observer-equivalence.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Observer leaves legacy trajectory, brain and organism exactly unchanged.", flush=True)
        return report
    finally:
        original.close()
        if observer:
            observer.close()


def checkpoint_equivalence(output, *, seconds=1.4, continuation=.15):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "circuit_configs/ache2019.toml")
    sim = CircuitSimulation(cfg, graph=ROOT / "data/graph", seed=1, control="paper")
    restored = None
    try:
        for _ in range(round(seconds * 100)):
            sim.step()
        saved = sim.get_state()
        for _ in range(4):
            assert sim.step(paused=True) is None
        assert_nested_equal(sim.get_state(), saved)
        path = output / "paper-checkpoint.npz"
        save_checkpoint(path, saved, sim.compatibility())
        expected_results = [sim.step() for _ in range(round(continuation * 100))]
        expected = sim.get_state()
        restored = CircuitSimulation(cfg, graph=ROOT / "data/graph", seed=1, control="paper", warmup=False)
        loaded, _ = load_checkpoint(path, restored.compatibility())
        restored.set_state(loaded)
        for expected_step in expected_results:
            assert_nested_equal(restored.step(), expected_step)
        assert_nested_equal(restored.get_state(), expected)
        report = {"status": "pass", "checkpoint_physics_s": seconds, "continuation_s": continuation,
                  "comparison": "exact complete serialized state and each returned step", "pause": "exact",
                  "paper_clock_s": restored.paper_probe.stream.clock * .0005}
        (output / "checkpoint-equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Circuit checkpoint and pause: exact complete-state continuation.", flush=True)
        return report
    finally:
        sim.close()
        if restored:
            restored.close()
