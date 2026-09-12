"""Four-condition *network* intervention audit; no physical claim is inferred.

Run from repository root with the project Python. The full prepared graph is
used sequentially, with one allocation and identical restored states per trial.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly_arena.brain import Brain, BrainConfig
from fly_arena.neural_ports import NeuralAdapter, NeuralConfig, build_registry
from fly_arena.sensors import SensorFrame


def run_experiments(graph: Path, output: Path, *, seed: int = 1, seconds: float = 1.,
                    warmup_s: float = .1, pathways=("feeding", "grooming"),
                    brain_config: BrainConfig | None = None, neural_config: NeuralConfig | None = None) -> dict:
    if seconds <= 0 or warmup_s < 0 or round(seconds * 100) != seconds * 100 or round(warmup_s * 100) != warmup_s * 100:
        raise ValueError("Experiment duration and warmup must be multiples of 10 ms")
    output.mkdir(parents=True, exist_ok=True)
    registry = build_registry(graph, output / "behavior_ports.json")
    cfg = neural_config or NeuralConfig(exploratory_ports=True)
    brain = Brain(graph, brain_config, seed)
    adapter = NeuralAdapter(brain, registry, cfg)
    initial_brain = brain.get_state()
    initial_adapter = adapter.get_state()
    initial_hash = hashlib.sha256(initial_brain["voltage"].tobytes()).hexdigest()
    luminance = np.zeros_like(brain.filtered_luminance)
    organism = SimpleNamespace(hunger=.8, state=SimpleNamespace(sleep_pressure=0.))
    report = {
        "schema_version": 1, "scope": "network_only", "physical_status": "not_run",
        "dataset": brain.manifest["dataset"], "graph_arrays": brain.manifest.get("arrays", {}),
        "registry_sha256": hashlib.sha256((output / "behavior_ports.json").read_bytes()).hexdigest(),
        "seed": seed, "initial_voltage_sha256": initial_hash,
        "seconds": seconds, "warmup_s": warmup_s,
        "brain_config": asdict(brain.config), "neural_config": asdict(cfg),
        "interpretation": "Candidate neural requests gate engineered primitives. This network experiment does not verify ingestion, grooming contact, or physical causality.",
        "pathways": {},
    }
    conditions = ("neutral", "stimulus", "sensory_off", "output_off")
    started = time.monotonic()
    with (output / "network_trace.jsonl").open("w", encoding="utf-8") as trace:
        for path in pathways:
            if path not in ("feeding", "grooming"):
                raise ValueError(f"Unknown experiment pathway {path}")
            adapter.require_supported((path,))
            channel = "taste" if path == "feeding" else "dust"
            group = "feed_output_mn9" if path == "feeding" else "groom_output_adn"
            path_results = {}
            for condition in conditions:
                brain.set_state(initial_brain)
                adapter.set_state(initial_adapter)
                # Every condition receives identical warmup, including RNG state.
                frame = SensorFrame()
                for _ in range(round(warmup_s * 100)):
                    sensory, modulation = adapter.currents(frame, organism)
                    _, counts = brain.step(luminance, sensory_currents=sensory, modulation=modulation)
                    adapter.readout(counts, .01)
                if condition != "neutral":
                    if path == "feeding":
                        frame.tarsal_taste = {"front_left": 1., "front_right": 1.}
                    else:
                        frame.dust_afferents = {"antenna_left": 1., "antenna_right": 1.}
                disabled = (("taste", "hunger") if path == "feeding" else ("dust",)) if condition == "sensory_off" else ()
                blocked = adapter.blocked_indices((path,)) if condition == "output_off" else ()
                total = np.zeros(len(brain.voltage), np.int64)
                requested_s = 0.
                for step in range(round(seconds * 100)):
                    sensory, modulation = adapter.currents(frame, organism)
                    _, counts = brain.step(luminance, sensory_currents=sensory, modulation=modulation,
                                           disabled_channels=disabled, blocked_outputs=blocked)
                    readout = adapter.readout(counts, .01)
                    total += counts
                    request = readout.feed_drive if path == "feeding" else readout.groom_drives["head"]
                    requested_s += .01 * (request > 0.)
                    trace.write(json.dumps({"pathway": path, "condition": condition, "time_s": (step + 1) * .01,
                        "raw_rates": readout.raw_rates, "request": request,
                        "channels": brain.channel_contributions}) + "\n")
                selected = adapter.indices[group]
                rate = float(np.mean(total[selected]) / seconds)
                path_results[condition] = {"output_mean_hz": rate, "output_body_ids": registry["groups"][group]["body_ids"],
                    "output_rates_hz": (total[selected] / seconds).tolist(), "request_duration_s": requested_s,
                    "total_network_spikes": int(total.sum()), "disabled_channels": list(disabled),
                    "silenced_body_ids": [int(brain.ids[i]) for i in blocked], "physical_action": "not_measured"}
                print(f"{path} {condition}: {rate:.3f} Hz, candidate request {requested_s:.2f} s", flush=True)
            stim = path_results["stimulus"]["output_mean_hz"]
            neutral = path_results["neutral"]["output_mean_hz"]
            off = path_results["sensory_off"]["output_mean_hz"]
            blocked = path_results["output_off"]["output_mean_hz"]
            checks = {"stimulus_exceeds_neutral_by_1hz": stim >= neutral + 1.,
                      "sensory_off_equals_neutral": bool(np.isclose(off, neutral, rtol=0, atol=1e-9)),
                      "silenced_output_emits_no_spikes": blocked == 0.,
                      "stimulus_generates_candidate_request": path_results["stimulus"]["request_duration_s"] > 0.}
            report["pathways"][path] = {"conditions": path_results, "network_checks": checks,
                                       "network_status": "pass" if all(checks.values()) else "fail",
                                       "support_status": "unsupported", "physical_status": "not_run"}
            # Network evidence is recorded without upgrading behaviour support.
            registry["pathways"][path]["network_test"] = {"artifact": "network_results.json", "checks": checks}
            for name in registry["pathways"][path]["inputs"] + registry["pathways"][path]["outputs"]:
                registry["groups"][name]["functional_test"]["status"] = "network_tested_physical_pending"
                registry["groups"][name]["functional_test"]["network"] = {"artifact": "network_results.json", "status": "pass" if all(checks.values()) else "fail"}
    report["wall_time_s"] = time.monotonic() - started
    (output / "behavior_ports.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    report["registry_sha256"] = hashlib.sha256((output / "behavior_ports.json").read_bytes()).hexdigest()
    (output / "network_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--output", type=Path, default=Path("runs/neural-experiments"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=1.)
    parser.add_argument("--warmup", type=float, default=.1)
    parser.add_argument("--pathway", choices=("feeding", "grooming", "both"), default="both")
    args = parser.parse_args()
    paths = ("feeding", "grooming") if args.pathway == "both" else (args.pathway,)
    run_experiments(args.graph, args.output, seed=args.seed, seconds=args.seconds, warmup_s=args.warmup, pathways=paths)


if __name__ == "__main__":
    main()
