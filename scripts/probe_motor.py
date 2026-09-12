"""Bounded actuator/contact diagnostic, not an autonomous behavior policy.

Run from the repository with .venv/Scripts/python.exe scripts/probe_motor.py.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import json
import time
import numpy as np

from fly_arena.ethology_arena import EthologyArena


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("FEED", "GROOM_FRONT", "GROOM_HEAD", "SLEEP", "WALK"), default="FEED")
    parser.add_argument("--seconds", type=float, default=1.3)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--blocked", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/motor-probe.json"))
    args = parser.parse_args()
    arena = EthologyArena(food_patches=[dict(id="test", x=.9, y=0., radius=2., surface_z=.01)],
                         mouth_contact_enabled=not args.blocked, grooming_contacts_enabled=not args.blocked)
    writer = None
    if args.video:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=30, codec="libx264", quality=7, macro_block_size=16)
    summary = {"action": args.action, "blocked": args.blocked, "min_upright": 1.,
               "max_contact_force_model_units": 0., "physics_seconds": 0.,
               "contact_s_by_food": defaultdict(float), "grooming_sliding_by_pair": defaultdict(float),
               "grooming_contact_s_by_pair": defaultdict(float), "ik_errors_mm": arena.motor.ik_errors}
    start = time.perf_counter()
    due = 0.
    try:
        for _ in range(round(args.seconds / .01)):
            events = arena.step_behavior(args.action, [.8, .8])
            summary["physics_seconds"] += .01
            summary["min_upright"] = min(summary["min_upright"], arena.upright)
            summary["max_contact_force_model_units"] = max(summary["max_contact_force_model_units"], arena.max_groom_contact_force)
            for key in ("contact_s_by_food", "grooming_sliding_by_pair", "grooming_contact_s_by_pair"):
                for pair, value in getattr(events, key).items():
                    summary[key][pair] += value
            if writer and summary["physics_seconds"] >= due:
                writer.append_data(arena.render(closeup=True))
                due += 1 / 30
        summary.update(wall_seconds=time.perf_counter() - start, mouth_position_mm=arena.mouth_position.tolist(),
                       final_position_mm=arena.position.tolist(), finite=bool(np.isfinite(arena.sim.mj_data.qpos).all()))
    finally:
        if writer:
            writer.close()
        arena.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
