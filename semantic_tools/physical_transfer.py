"""Held-out physical transfer of the frozen state head, including real feeding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fly_arena.config import load_config
from fly_semantic.mapping import file_digest
from fly_semantic.runtime import SemanticSimulation
from .state_experiment import metrics, write_json


def run(graph, calibration, model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise FileExistsError("Physical transfer is frozen; use a fresh output path")
    study = {"scope": "preliminary transfer of frozen neural-only training head into existing physical arena",
        "seeds": [701, 702, 703], "conditions": ["connected", "homeostasis_off"],
        "scenes": {"hungry_away": 1.5, "sated_away": 1.5, "physical_feeding": 5.},
        "model_sha256": file_digest(model), "calibration_sha256": file_digest(Path(calibration) / "calibration.json"),
        "teacher_on": .7, "teacher_off": .55, "score_threshold": .5,
        "no_tuning": True, "all_samples_included": True,
        "food_transfer": "existing mouth contact, feeding primitive, environment transaction and digestion only"}
    write_json(output / "plan.json", study)
    results = []
    for seed in study["seeds"]:
        for scene, duration in study["scenes"].items():
            cfg = load_config(Path("configs/ethology-unscaled.toml"), "feeding-contact")
            cfg.setdefault("motor", {})["adaptive_proboscis"] = True
            if scene != "physical_feeding":
                cfg["food"] = []
            if scene == "sated_away":
                cfg["initial"]["energy"] = 80.
            for condition in study["conditions"]:
                started = time.perf_counter()
                episode = f"{seed}-{scene}-{condition}"
                sim = SemanticSimulation(cfg, graph=graph, seed=seed, enabled=True,
                    mapping_path=Path(calibration) / "mapping.json", calibration_path=Path(calibration) / "calibration.json",
                    readout_path=model, episode_id=episode, hunger_disconnected=condition == "homeostasis_off")
                rows, events, transitions = [], [], []
                active = False
                try:
                    for tick in range(round(duration * 100)):
                        need = sim.organism.hunger
                        previous = active
                        active = bool(need >= .7 or (active and need > .55))
                        if active != previous:
                            transitions.append({"time_ms": sim.brain.clock, "active": active, "hunger": need})
                        sim.step()
                        events.extend(e.to_dict() for e in sim.channel.last_events)
                        if (tick + 1) % 10 == 0:
                            status = sim.channel.outbox.poll_state(sim.brain.clock)[1]
                            rows.append({"time_ms": sim.brain.clock, "target": active,
                                "hunger_before_step": need, "score": sim.channel.last_scores[1],
                                "active": status["active"], "intake": sim.organism.state.ingested_total,
                                "action": sim.last_decision.action, "upright": sim.arena.upright})
                    result = {"episode": episode, "seed": seed, "scene": scene, "condition": condition,
                        "duration_s": duration, "config": cfg, "wall_seconds": time.perf_counter() - started,
                        "ingested_total": sim.organism.state.ingested_total, "final_hunger": sim.organism.hunger,
                        "minimum_upright": sim.minimum_upright, "resource_balance": sim.resource_balances(),
                        "target_transitions": transitions, "events": events, "samples": rows,
                        "raw": metrics([r["target"] for r in rows], [r["score"] >= .5 for r in rows]),
                        "indicator": metrics([r["target"] for r in rows], [r["active"] for r in rows])}
                    write_json(output / f"{episode}.json", result)
                    results.append(result)
                    print(json.dumps({k: result[k] for k in ("episode", "ingested_total", "final_hunger", "raw", "indicator", "wall_seconds")}), flush=True)
                finally:
                    sim.close()
    summary = {"scope": study["scope"], "model_sha256": study["model_sha256"], "episodes": len(results), "conditions": {}}
    for condition in study["conditions"]:
        rows = [r for e in results if e["condition"] == condition for r in e["samples"]]
        summary["conditions"][condition] = {
            "raw": metrics([r["target"] for r in rows], [r["score"] >= .5 for r in rows]),
            "indicator": metrics([r["target"] for r in rows], [r["active"] for r in rows])}
    summary["feeding_cases_with_actual_intake"] = sum(e["ingested_total"] > 0 for e in results if e["scene"] == "physical_feeding")
    summary["feeding_cases_with_target_off_transition"] = sum(any(not t["active"] for t in e["target_transitions"]) for e in results if e["scene"] == "physical_feeding")
    summary["raw_target_reached"] = summary["conditions"]["connected"]["raw"]["f1"] >= .8
    summary["indicator_target_reached"] = summary["conditions"]["connected"]["indicator"]["f1"] >= .8
    summary["status"] = "PRELIMINARY_TARGET_REACHED" if summary["raw_target_reached"] and summary["indicator_target_reached"] else "TARGET_NOT_REACHED"
    write_json(output / "results.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--calibration", type=Path, default=Path("runs/semantic-v1/calibration-stage-b"))
    parser.add_argument("--model", type=Path, default=Path("runs/semantic-v1/state-study/state-readout.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/semantic-v1/physical-transfer"))
    args = parser.parse_args()
    run(args.graph, args.calibration, args.model, args.output)
