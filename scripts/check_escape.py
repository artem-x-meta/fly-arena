"""Physical visual-escape interventions with explicit engineered detection."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly_arena.config import load_config, merge_config
from fly_arena.ethology import EthologySimulation
from fly_arena.escape import looming_position
from scripts.check_ethology_scenarios import feeding_config, sleep_config, groom_config, verify_resume


def case_config(kind):
    stimulus = load_config(ROOT / "configs/escape-arena.toml")
    if kind == "feeding":
        cfg = feeding_config()
    elif kind == "sleep":
        cfg = sleep_config("control")
    elif kind == "grooming":
        cfg = groom_config("head")
    else:
        cfg = stimulus
    cfg = merge_config(cfg, {"escape": stimulus["escape"], "environment": {"looming": stimulus["environment"]["looming"]}})
    if kind == "weak":
        cfg["environment"]["looming"][0]["radius"] = .15
    if kind == "repeated":
        cfg["environment"]["looming"][0]["repeat_s"] = 1.5
    return cfg


def run_case(kind, output, *, seed=101, seconds=3.2, video=False):
    cfg = case_config(kind)
    sim = EthologySimulation(cfg, seed=seed, brain_enabled=False, blind=kind == "blind", motor_off=kind == "motor-off")
    peak = 0.
    actual_escape_s = 0.
    escape_resource = 0.
    transitions = []
    escape_start = escape_position = away = None
    escaped_position = None
    writer = None
    frame_due = 0.
    output.mkdir(parents=True, exist_ok=True)
    try:
        if video:
            import imageio.v2 as imageio
            from fly_arena.display import compose_ethology_frame
            writer = imageio.get_writer(output / f"{kind}.mp4", fps=30, codec="libx264", macro_block_size=16)
        with (output / f"{kind}-trace.jsonl").open("w", encoding="utf-8") as trace:
            for i in range(round(seconds * 100)):
                before = sim.organism.state.ingested_total
                result = sim.step()
                peak = max(peak, sim.last_frame.looming_left, sim.last_frame.looming_right)
                if result["transition"]:
                    transitions.append(result["transition"])
                if sim.arena.motor.owner == "ESCAPE":
                    actual_escape_s += .01
                    escape_resource += sim.organism.state.ingested_total - before
                    if escape_start is None:
                        escape_start = sim.organism.clocks.physics_time_s
                        escape_position = sim.arena.position.copy()
                        position = looming_position(cfg["environment"]["looming"][0], escape_start)[0]
                        away = escape_position[:2] - position[:2]
                        away /= np.linalg.norm(away)
                    escaped_position = sim.arena.position.copy()
                if i % 10 == 0:
                    trace.write(json.dumps(sim.telemetry()) + "\n")
                if writer and sim.organism.clocks.physics_time_s >= frame_due:
                    writer.append_data(compose_ethology_frame(sim))
                    frame_due += 1 / 30
        report = {"kind": kind, "seed": seed, "config": cfg, "peak_visual_score": peak,
                  "first_escape_s": escape_start, "actual_escape_s": actual_escape_s,
                  "escape_ingestion": escape_resource, "transitions": transitions,
                  "minimum_upright": sim.minimum_upright, "actions": sim.action_durations,
                  "final": sim.telemetry(),
                  "away_displacement_mm": float((escaped_position[:2] - escape_position[:2]) @ away) if escape_start is not None else 0,
                  "escape_distance_mm": float(np.linalg.norm(escaped_position[:2] - escape_position[:2])) if escape_start is not None else 0,
                  "brain_enabled": False, "detector": "engineered colour-area expansion from both rendered eyes"}
        (output / f"{kind}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{kind}: peak={peak:.3f} escape={actual_escape_s:.2f}s first={escape_start} upright={sim.minimum_upright:.3f} away={report['away_displacement_mm']:.2f}mm", flush=True)
        return report
    finally:
        if writer:
            writer.close()
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/escape-acceptance")
    parser.add_argument("--cases", nargs="+", default=["strong", "weak", "blind", "motor-off", "feeding", "sleep", "grooming", "repeated"])
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    results = {}
    for kind in args.cases:
        results[kind] = run_case(kind, args.output, seconds=7 if kind == "repeated" else 3.2, video=args.video)
    for kind in ("weak", "blind", "motor-off"):
        if kind in results:
            assert results[kind]["actual_escape_s"] == 0, kind
    for kind in ("strong", "feeding", "sleep", "grooming", "repeated"):
        if kind in results:
            r = results[kind]
            assert r["actual_escape_s"] > .1, kind
            assert r["minimum_upright"] > .8, kind
            assert r["escape_ingestion"] == 0, kind
    if "strong" in results:
        assert results["strong"]["away_displacement_mm"] > .1
        cfg = case_config("strong")
        results["resume"] = verify_resume("escape-burst", cfg, results["strong"]["first_escape_s"] + .05, args.output)
    if "sleep" in results:
        r = results["sleep"]
        wake = next(t for t in r["transitions"] if t["action"] == "WAKE")
        assert r["first_escape_s"] - wake["time_s"] >= .25
    if "repeated" in results:
        starts = [t["time_s"] for t in results["repeated"]["transitions"] if t["action"] == "ESCAPE"]
        assert len(starts) >= 1
        assert all(b - a >= 2 for a, b in zip(starts, starts[1:]))
    (args.output / "results.json").write_text(json.dumps({"status": "pass", "results": results}, indent=2), encoding="utf-8")
    print("ESCAPE CONTROLS: PASS", flush=True)


if __name__ == "__main__":
    main()
