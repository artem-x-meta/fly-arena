"""Fixed metric for odor search, measured before any change to the plume.

Runs the unchanged default scene on several seeds and reports, per seed, whether
the fly ever transferred food, how long that took and how far it walked. The
numbers are the baseline the next iteration has to beat; they are recorded here
so the target cannot be chosen after seeing the result.

    .venv/Scripts/python.exe scripts/search_baseline.py --seconds 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fly_arena.config import load_config
from fly_arena.ethology import EthologySimulation


def scene(distance_mm: float | None) -> dict:
    """Default scene, or one where food is out of reach of a standing fly.

    The default scene spawns the fly less than a millimetre from a patch, so it
    measures whether the feeding loop closes, not whether the fly can find food.
    A distance moves both patches away and makes navigation the only route.
    """
    config = load_config()
    if distance_mm is None:
        return config
    for patch, angle in zip(config["food"], (40.0, 200.0)):
        radians = np.radians(angle)
        patch["x"] = round(float(distance_mm * np.cos(radians)), 3)
        patch["y"] = round(float(distance_mm * np.sin(radians)), 3)
    return config


def run_seed(seed: int, seconds: float, distance_mm: float | None = None) -> dict:
    started = time.perf_counter()
    simulation = EthologySimulation(scene(distance_mm), seed=seed, brain_enabled=False)
    patches = [(p.x, p.y, p.radius) for p in simulation.environment.food]
    track = [simulation.arena.position[:2].copy()]
    first_intake_s = first_taste_s = None
    nearest = min(float(np.hypot(track[0][0] - x, track[0][1] - y)) for x, y, _ in patches)
    try:
        for _ in range(round(seconds / .01)):
            simulation.step()
            position = simulation.arena.position[:2].copy()
            track.append(position)
            nearest = min(nearest, min(float(np.hypot(position[0] - x, position[1] - y))
                                       for x, y, _ in patches))
            frame = simulation.last_frame
            if first_taste_s is None and max([frame.mouth_taste, *frame.tarsal_taste.values()]) > 0:
                first_taste_s = simulation.organism.clocks.physics_time_s
            if first_intake_s is None and simulation.organism.state.ingested_total > 0:
                first_intake_s = simulation.organism.clocks.physics_time_s
        final = simulation.telemetry()
        path_mm = float(np.abs(np.diff(np.asarray(track), axis=0)).sum())
        return {"seed": seed, "physics_seconds": seconds,
                "wall_seconds": round(time.perf_counter() - started, 1),
                "first_taste_s": first_taste_s, "first_intake_s": first_intake_s,
                "ingested_total": final["ingested_total"],
                "closest_approach_mm": round(nearest, 3),
                "path_length_mm": round(path_mm, 2),
                "walk_s": round(simulation.action_durations.get("WALK", 0.), 2),
                "exhausted_s": round(simulation.action_durations.get("EXHAUSTED", 0.), 2),
                "final_energy": round(final["energy"], 3),
                "action_durations": {k: round(v, 2) for k, v in simulation.action_durations.items()}}
    finally:
        simulation.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "search-baseline")
    parser.add_argument("--food-distance-mm", type=float, default=None,
                        help="Move both patches this far from the start; omit to keep the default scene")
    parser.add_argument("--name", default="baseline", help="Output file stem")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        result = run_seed(seed, args.seconds, args.food_distance_mm)
        results.append(result)
        print(f"seed {seed}: intake={result['ingested_total']:.3f} "
              f"first_intake={result['first_intake_s']} closest={result['closest_approach_mm']}mm "
              f"path={result['path_length_mm']}mm walk={result['walk_s']}s "
              f"exhausted={result['exhausted_s']}s", flush=True)
    fed = [r for r in results if r["ingested_total"] > 0]
    summary = {"seeds": args.seeds, "physics_seconds": args.seconds,
               "food_distance_mm": args.food_distance_mm,
               "reached_food_fraction": f"{len(fed)}/{len(results)}",
               "median_first_intake_s": (float(np.median([r["first_intake_s"] for r in fed]))
                                         if fed else None),
               "median_closest_approach_mm": float(np.median([r["closest_approach_mm"] for r in results])),
               "median_path_length_mm": float(np.median([r["path_length_mm"] for r in results])),
               "seeds_ending_exhausted": sum(r["exhausted_s"] > 0 for r in results),
               "results": results}
    (args.output / f"{args.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
