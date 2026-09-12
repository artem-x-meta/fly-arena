"""Four-condition full-connectome tests of a candidate retinal escape pathway."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly_arena.config import load_config
from fly_arena.ethology import EthologySimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "seconds": args.seconds, "conditions": {}, "status": "running",
              "scope": "Actual eye frames through the unchanged R1-R6 input and full LIF graph; no injected LC4/LPLC2 feature current"}
    first_hash = None
    for condition in ("neutral", "stimulus", "blind", "output_off"):
        cfg = load_config(ROOT / "configs/escape-arena.toml")
        cfg["neural"]["exploratory_ports"] = True
        cfg["behavior"]["allow_experimental_ports"] = True
        cfg["required_neural_pathways"] = ["escape"]
        if condition == "neutral":
            cfg["environment"]["looming"][0]["time_s"] = 100.
        sim = EthologySimulation(cfg, graph=ROOT / "data/graph", mode="ethology-neural", seed=args.seed,
                                  blind=condition == "blind", blocked_outputs=["escape"] if condition == "output_off" else [])
        try:
            digest = hashlib.sha256()
            for values in (sim.brain.voltage, sim.brain.current, sim.brain.queue, sim.arena.sim.mj_data.qpos, sim.arena.sim.mj_data.qvel):
                digest.update(values.tobytes())
            state_hash = digest.hexdigest()
            if first_hash is None:
                first_hash = state_hash
                report["registry"] = sim.registry
                report["compatibility"] = sim.compatibility()
            assert first_hash == state_hash
            rates = {}
            escaped = 0.
            peak_visual = 0.
            with (args.output / f"{condition}.jsonl").open("w", encoding="utf-8") as trace:
                for i in range(round(args.seconds * 100)):
                    sim.step()
                    for name, rate in sim.last_readout.raw_rates.items():
                        if name.startswith("escape_"):
                            rates[name] = rates.get(name, 0) + rate * .01 / args.seconds
                    escaped += .01 * (sim.arena.motor.owner == "ESCAPE")
                    peak_visual = max(peak_visual, sim.last_frame.looming_left, sim.last_frame.looming_right)
                    if i % 10 == 0:
                        trace.write(json.dumps({"telemetry": sim.telemetry(), "rates": sim.last_readout.raw_rates,
                                                "coverage": sim.last_readout.supported_ports}) + "\n")
            report["conditions"][condition] = {"initial_state_hash": state_hash, "mean_rates_hz": rates,
                                                "escape_physics_s": escaped, "hybrid_visual_score_diagnostic_only": peak_visual,
                                                "minimum_upright": sim.minimum_upright, "actions": sim.action_durations,
                                                "balances": sim.resource_balances()}
            print(f"seed{args.seed} {condition}: GF={rates['escape_output_dnp01']:.3f}Hz, escape={escaped:.2f}s, diagnostic score={peak_visual:.2f}", flush=True)
        finally:
            sim.close()
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    conditions = report["conditions"]
    stim, neutral, blind, blocked = (conditions[k] for k in ("stimulus", "neutral", "blind", "output_off"))
    report["checks"] = {
        "profile_output_increases": stim["mean_rates_hz"]["escape_output_dnp01"] >= neutral["mean_rates_hz"]["escape_output_dnp01"] + 1.,
        "physical_escape_increases": stim["escape_physics_s"] > neutral["escape_physics_s"],
        "blind_ablation_prevents_escape": blind["escape_physics_s"] == 0.,
        "output_block_silences_cells_and_escape": blocked["escape_physics_s"] == 0. and blocked["mean_rates_hz"]["escape_output_dnp01"] == 0.,
    }
    report["status"] = "model_causal_candidate" if all(report["checks"].values()) else "unsupported"
    report["interpretation"] = "A GF-to-running adapter is an engineering proxy; annotation and functional validity remain separate. Failed conditions are retained without retuning graph weights."
    report["registry"]["pathways"]["escape"]["functional_test"] = {"status": report["status"], "artifact": "results.json", "checks": report["checks"]}
    (args.output / "behavior_ports.json").write_text(json.dumps(report["registry"], indent=2), encoding="utf-8")
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
