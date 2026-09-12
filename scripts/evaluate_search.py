"""Fixed scene comparison; calibration seeds and held-out seeds are explicit.

Positions below are for external measurement only; never given to a controller.
Each case streams CSV and commits its JSON even if a later seed fails.
"""
from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly_arena.config import load_config, config_digest
from fly_arena.ethology import EthologySimulation


def run_case(config_path, seed, seconds, distance, output, full_brain=False):
    cfg = copy.deepcopy(config_path) if isinstance(config_path, dict) else load_config(config_path)
    if distance is not None:
        for patch, angle in zip(cfg["food"], (40., 200.)):
            radians = np.radians(angle)
            patch["x"], patch["y"] = round(float(distance * np.cos(radians)), 3), round(float(distance * np.sin(radians)), 3)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sim = EthologySimulation(cfg, seed=seed, brain_enabled=full_brain, graph=ROOT / "data/graph")
    positions = np.array([[p.x, p.y] for p in sim.environment.food])
    previous = sim.arena.position[:2]
    length = 0.
    closest = float(np.linalg.norm(positions - previous, axis=1).min())
    first_contact = first_intake = first_exhaustion = None
    phases, events, changes = {}, {}, []
    last_action = None
    with (output / f"seed{seed}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        try:
            for step in range(round(seconds * 100)):
                result = sim.step()
                frame = sim.last_frame
                t = sim.organism.clocks.physics_time_s
                position = sim.arena.position[:2]
                length += float(np.linalg.norm(position - previous))
                closest = min(closest, float(np.linalg.norm(positions - position, axis=1).min()))
                previous = position
                if first_contact is None and max([frame.mouth_taste, *frame.tarsal_taste.values()]) > 0:
                    first_contact = t
                if first_intake is None and sim.organism.state.ingested_total > 0:
                    first_intake = t
                if first_exhaustion is None and sim.organism.exhausted:
                    first_exhaustion = t
                phase = getattr(sim.navigator, "phase", "legacy")
                if sim.last_decision.action == "WALK":
                    phases[phase] = phases.get(phase, 0) + .01
                for event in result["external_events"]:
                    events[event["kind"]] = events.get(event["kind"], 0) + 1
                if sim.last_decision.action != last_action:
                    changes.append({"t": t, "action": sim.last_decision.action})
                    last_action = sim.last_decision.action
                if step % 10 == 0 or step == round(seconds * 100) - 1:
                    row = sim.telemetry()
                    row.update(search_phase=phase, closest_approach_mm=closest, path_length_mm=length)
                    if writer is None:
                        writer = csv.DictWriter(stream, list(row))
                        writer.writeheader()
                    writer.writerow(row)
            report = {"seed": seed, "physics_seconds": seconds, "food_distance_mm": distance,
                      "brain_enabled": full_brain, "config": cfg, "config_digest": config_digest(cfg),
                      "source_digest": sim.source_digest, "ingested_total": sim.organism.state.ingested_total,
                      "first_taste_s": first_contact, "first_intake_s": first_intake,
                      "first_exhaustion_s": first_exhaustion,
                      "contact_before_exhaustion": first_contact is not None and (first_exhaustion is None or first_contact < first_exhaustion),
                      "closest_approach_mm": closest, "path_length_mm": length,
                      "walk_s": sim.action_durations.get("WALK", 0), "final_energy": sim.organism.state.energy,
                      "minimum_upright": sim.minimum_upright, "search_phase_durations": phases,
                      "events": events, "action_durations": sim.action_durations, "transitions": changes,
                      "balances": sim.resource_balances(), "wall_seconds": time.perf_counter() - started}
            (output / f"seed{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"seed {seed}: intake={report['ingested_total']:.3f} first={first_intake} closest={closest:.2f} "
                  f"path={length:.2f} walk={report['walk_s']:.2f} E={report['final_energy']:.2f} "
                  f"upright={sim.minimum_upright:.3f}", flush=True)
            return report
        finally:
            sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-brain", action="store_true")
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise SystemExit("Use a new output directory; previous trials must be retained")
    results = []
    frozen_config = load_config(args.config)
    for seed in args.seeds:
        results.append(run_case(frozen_config, seed, args.seconds, args.distance, args.output, args.full_brain))
        fed = [r for r in results if r["first_intake_s"] is not None]
        summary = {"seeds": [r["seed"] for r in results], "physics_seconds": args.seconds,
                   "food_distance_mm": args.distance, "brain_enabled": args.full_brain,
                   "reached_food_fraction": f"{len(fed)}/{len(results)}",
                   "median_first_intake_s": float(np.median([r["first_intake_s"] for r in fed])) if fed else None,
                   "median_path_length_mm": float(np.median([r["path_length_mm"] for r in results])),
                   "seeds_contact_before_exhaustion": sum(r["contact_before_exhaustion"] for r in results),
                   "seeds_ever_exhausted": sum(r["first_exhaustion_s"] is not None for r in results)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
