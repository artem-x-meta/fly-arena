"""Candidate-port interventions through the real 10 ms brain/body loop.

No behavioural sequence, injected motor request, or target position is passed
to the controller. Food is a physical local surface; dust is an external event.
The result can be negative and never automatically upgrades port confidence.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly_arena.config import load_config
from fly_arena.ethology import EthologySimulation
from fly_arena.neural_ports import NeuralConfig, build_registry
from fly_arena.brain import BrainConfig


def _state_hash(sim):
    digest = hashlib.sha256()
    for values in (sim.brain.voltage, sim.brain.current, sim.brain.queue,
                   sim.arena.sim.mj_data.qpos, sim.arena.sim.mj_data.qvel):
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def run_physical_experiments(graph: Path, output: Path, *, seed=1, seconds=2., pathways=("feeding", "grooming")):
    if seconds <= 0 or round(seconds * 100) != seconds * 100:
        raise ValueError("Duration must be a positive multiple of 10 ms")
    output.mkdir(parents=True, exist_ok=True)
    base = load_config()
    base["profile"] = "unscaled"
    base["organism"]["life_time_scale"] = 1.
    base["initial"] = {"energy": 20., "sleep_pressure": 0., "dust_by_region": {
        "head": 0., "antenna_left": 0., "antenna_right": 0., "front_left": 0., "front_right": 0.}}
    # Same visible patch/stock/geometry in every condition, initially tasteless.
    # Its coordinates are only given to the environment, never the controller.
    base["food"] = [{"id": "assay_food", "x": 0., "y": 0., "radius": 3., "amount": 10.,
                     "energy_density": 12., "taste": 0., "odor": 0.}]
    base["events"] = []
    base["neural"]["exploratory_ports"] = True
    report = {"schema_version": 1, "scope": "physical_candidate_intervention", "seed": seed,
              "seconds": seconds, "base_config": base, "pathways": {},
              "interpretation": "Explicit experimental ports; no hybrid navigation or sensory-to-action fallback. Initial brain/body state is identical; only the local stimulus and specified neural intervention differ."}
    started = time.monotonic()
    first_hash = None
    for path in pathways:
        if path not in ("feeding", "grooming"):
            raise ValueError(f"Unknown pathway {path}")
        group = "feed_output_mn9" if path == "feeding" else "groom_output_adn"
        action = "FEED" if path == "feeding" else "GROOM_HEAD"
        results = {}
        for condition in ("neutral", "stimulus", "sensory_off", "output_off"):
            cfg = copy.deepcopy(base)
            disabled = (("taste", "hunger") if path == "feeding" else ("dust",)) if condition == "sensory_off" else ()
            blocked = (path,) if condition == "output_off" else ()
            sim = EthologySimulation(cfg, graph=graph, seed=seed, mode="ethology-neural",
                                     disabled_channels=disabled, blocked_outputs=blocked)
            try:
                initial_hash = _state_hash(sim)
                if first_hash is None:
                    first_hash = initial_hash
                    report["initial_brain_body_sha256"] = initial_hash
                    report["compatibility"] = sim.compatibility()
                elif initial_hash != first_hash:
                    raise AssertionError("Intervention runs did not start from identical brain/body states")
                # Apply stimulus only after the initial-state equality check.
                if condition != "neutral":
                    if path == "feeding":
                        sim.environment.food[0].taste = 1.
                    else:
                        sim.environment.events = [
                            {"time_s": 0., "kind": "dust", "region": "antenna_left", "amount": .8},
                            {"time_s": 0., "kind": "dust", "region": "antenna_right", "amount": .8}]
                rate_integral = 0.
                request_duration = relevant_sensor_duration = 0.
                mouth_contact_s = permitted_contact_s = sliding_mm = grooming_contact_s = 0.
                input_integral = 0.
                with (output / f"{path}-{condition}.jsonl").open("w", encoding="utf-8") as trace:
                    for step in range(round(seconds * 100)):
                        sim.step()
                        neural, frame, events = sim.last_readout, sim.last_frame, sim.last_events
                        rate_integral += neural.raw_rates[group] * .01
                        request = neural.feed_drive if path == "feeding" else neural.groom_drives["head"]
                        request_duration += .01 * (request > 0.)
                        level = max(frame.tarsal_taste.values(), default=0.) if path == "feeding" else max(frame.dust_afferents.get("antenna_left", 0.), frame.dust_afferents.get("antenna_right", 0.))
                        relevant_sensor_duration += .01 * (level > 0.)
                        mouth_contact_s += sum(events.mouth_contact_s_by_food.values())
                        permitted_contact_s += sum(events.contact_s_by_food.values())
                        sliding_mm += sum(events.grooming_sliding_by_pair.values())
                        grooming_contact_s += sum(events.grooming_contact_s_by_pair.values())
                        input_group = "taste_leg_sweet" if path == "feeding" else "antennal_mechanosensory"
                        input_integral += neural.raw_rates[input_group] * .01
                        if step % 10 == 0:
                            trace.write(json.dumps({**sim.telemetry(), "relevant_sensor_level": level,
                                "raw_rates": neural.raw_rates, "supported_ports": neural.supported_ports,
                                "channels": sim.brain.channel_contributions}) + "\n")
                results[condition] = {"initial_brain_body_sha256": initial_hash,
                    "input_mean_hz": input_integral / seconds, "output_mean_hz": rate_integral / seconds,
                    "request_duration_s": request_duration, "relevant_sensor_duration_s": relevant_sensor_duration,
                    "action_duration_s": sim.action_durations.get(action, 0.),
                    "all_action_durations_s": dict(sim.action_durations),
                    "ingested_food_units": sim.organism.state.ingested_total,
                    "permitted_mouth_contact_s": permitted_contact_s, "mouth_contact_s": mouth_contact_s,
                    "grooming_sliding_mm": sliding_mm, "grooming_contact_s": grooming_contact_s,
                    "removed_dust_units": sim.environment.removed_dust,
                    "final_dust_by_region": dict(sim.organism.state.dust_by_region),
                    "minimum_upright": sim.minimum_upright, "resource_balances": sim.resource_balances(),
                    "disabled_channels": list(disabled), "blocked_outputs": list(blocked)}
                print(f"{path} {condition}: input {input_integral / seconds:.2f} Hz, output {rate_integral / seconds:.2f} Hz, {action} {sim.action_durations.get(action, 0.):.2f} s, food {sim.organism.state.ingested_total:.5f}, removed dust {sim.environment.removed_dust:.5f}", flush=True)
            finally:
                sim.close()
                del sim
                gc.collect()
        stim, neutral, off, blocked = (results[name] for name in ("stimulus", "neutral", "sensory_off", "output_off"))
        measurement = "ingested_food_units" if path == "feeding" else "removed_dust_units"
        checks = {
            "relevant_local_stimulus_was_measured": stim["relevant_sensor_duration_s"] > 0.,
            "output_increases_by_1hz": stim["output_mean_hz"] >= neutral["output_mean_hz"] + 1.,
            "stimulus_causes_physical_resource_change": stim[measurement] > neutral[measurement] + 1e-6,
            "input_ablation_reduces_resource_change": off[measurement] < stim[measurement] - 1e-6,
            "output_block_reduces_resource_change": blocked[measurement] < stim[measurement] - 1e-6,
            "blocked_output_generates_no_action": blocked["action_duration_s"] == 0. and blocked["output_mean_hz"] == 0.,
        }
        report["pathways"][path] = {"conditions": results, "physical_checks": checks,
            "physical_status": "pass" if all(checks.values()) else "fail", "support_status": "unsupported"}
        report["wall_time_s"] = time.monotonic() - started
        (output / "physical_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def aggregate_evidence(graph: Path, network_root: Path, physical_root: Path, output: Path,
                       seeds=(1, 2, 3)) -> dict:
    """Separate functional model support from annotation confidence.

    Writes only to the report directory. Installing the reviewed registry in
    data/graph is an explicit integration step outside this experiment script.
    """
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three distinct seeds are required for model support")
    output.mkdir(parents=True, exist_ok=True)
    registry = build_registry(graph)
    reports = []
    for seed in seeds:
        network_path = network_root / f"seed{seed}" / "network_results.json"
        physical_path = physical_root / f"seed{seed}" / "physical_results.json"
        network = json.loads(network_path.read_text(encoding="utf-8"))
        physical = json.loads(physical_path.read_text(encoding="utf-8"))
        if network["seed"] != seed or physical["seed"] != seed:
            raise ValueError("Evidence seed does not match its directory")
        if physical["compatibility"]["graph"]["dataset"] != registry["dataset"]:
            raise ValueError("Evidence uses a different graph dataset")
        if network["graph_arrays"] != physical["compatibility"]["graph"]["arrays"]:
            raise ValueError("Network and physical evidence graph hashes differ")
        if network["brain_config"] != asdict(BrainConfig()):
            raise ValueError("Network evidence does not use the physical experiment's default BrainConfig")
        if network["neural_config"] != asdict(NeuralConfig.from_dict(physical["base_config"]["neural"])):
            raise ValueError("Network and physical neural parameters differ")
        if reports:
            previous_config = copy.deepcopy(reports[0][4]["base_config"])
            current_config = copy.deepcopy(physical["base_config"])
            # The runtime introduced this explicit key during integration; its
            # earlier implicit default was the same pair of required pathways.
            for config in (previous_config, current_config):
                config.setdefault("required_neural_pathways", ["feeding", "grooming"])
            if current_config != previous_config or physical["seconds"] != reports[0][4]["seconds"] or network["seconds"] != reports[0][3]["seconds"]:
                raise ValueError("Evidence conditions changed between seeds")
        reports.append((seed, network_path, physical_path, network, physical))
    summary = {"schema_version": 1, "seeds": list(seeds), "pathways": {},
               "status_semantics": "supported means causal sensor-to-physical-action support within the explicitly tested simplified embodied model, independently of biological annotation confidence",
               "no_parameter_tuning": True}
    for pathway in ("feeding", "grooming"):
        path = registry["pathways"][pathway]
        evidence = []
        all_pass = True
        for seed, network_path, physical_path, network, physical in reports:
            n, p = network["pathways"][pathway], physical["pathways"][pathway]
            valid = p["physical_status"] == "pass" and bool(p["physical_checks"]) and all(p["physical_checks"].values())
            condition_hashes = {entry["initial_brain_body_sha256"] for entry in p["conditions"].values()}
            valid = valid and len(condition_hashes) == 1
            all_pass = all_pass and valid
            evidence.append({"seed": seed, "network_artifact": network_path.as_posix(),
                "network_sha256": hashlib.sha256(network_path.read_bytes()).hexdigest(), "network_status": n["network_status"],
                "physical_artifact": physical_path.as_posix(),
                "physical_sha256": hashlib.sha256(physical_path.read_bytes()).hexdigest(), "physical_status": p["physical_status"],
                "physical_checks": p["physical_checks"], "source_digest": physical["compatibility"]["code"],
                "stimulus": p["conditions"]["stimulus"],
                "controls": {name: {key: value for key, value in p["conditions"][name].items()
                            if key in ("output_mean_hz", "action_duration_s", "ingested_food_units", "removed_dust_units")}
                             for name in ("neutral", "sensory_off", "output_off")}})
        confidence = "candidate_homolog" if pathway == "feeding" else "annotated_match"
        path.update(status="supported" if all_pass else "unsupported",
                    support_scope="embodied_model_only" if all_pass else "no_functional_support",
                    annotation_confidence=confidence, evidence=evidence,
                    tested_context={"physical_seconds": reports[0][4]["seconds"],
                                    "vision": "actual binocular rendered frames, default visual gain",
                                    "sensory_input": path["inputs"], "neural_output": path["outputs"],
                                    "mode": "ethology-neural with explicit exploratory candidates during validation",
                                    "brain_config": reports[0][3]["brain_config"],
                                    "neural_config": reports[0][3]["neural_config"]},
                    untested_contexts=["mouth-sugar afferents", "olfactory navigation", "other food modalities", "physiological prediction"])
        path["reason"] = ("Three-seed four-condition embodied causal tests passed; sensory annotation remains a candidate homolog; short dark-network test was negative"
                          if all_pass else "Three-seed relevant stimulation did not establish output-driven physical action")
        for name in path["inputs"] + path["outputs"]:
            group = registry["groups"][name]
            group["status"] = "supported_in_embodied_model" if all_pass else "unsupported"
            group["functional_test"] = {"status": "tested", "network": {"scope": "zero luminance, 1 second", "status_by_seed": {str(seed): n["pathways"][pathway]["network_status"] for seed, _, _, n, _ in reports}},
                "physical": {"scope": "real binocular frames, 2 seconds", "status_by_seed": {str(seed): p["pathways"][pathway]["physical_status"] for seed, _, _, _, p in reports}},
                "evidence_summary": (output / "evidence_summary.json").as_posix()}
        summary["pathways"][pathway] = path
    (output / "behavior_ports.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (output / "evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--output", type=Path, default=Path("runs/neural-physical"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=2.)
    parser.add_argument("--pathway", choices=("feeding", "grooming", "both"), default="both")
    parser.add_argument("--aggregate", action="store_true", help="Combine existing seeds 1,2,3 into a separate reviewed registry")
    parser.add_argument("--network-root", type=Path, default=Path("runs/neural-experiments"))
    parser.add_argument("--physical-root", type=Path, default=Path("runs/neural-physical"))
    args = parser.parse_args()
    if args.aggregate:
        aggregate_evidence(args.graph, args.network_root, args.physical_root, args.output)
        return
    paths = ("feeding", "grooming") if args.pathway == "both" else (args.pathway,)
    run_physical_experiments(args.graph, args.output, seed=args.seed, seconds=args.seconds, pathways=paths)


if __name__ == "__main__":
    main()
