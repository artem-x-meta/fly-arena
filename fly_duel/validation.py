"""Bounded software/physics checks of the engineered fight, not physiology."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .runner import run


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "fly_duel").glob("*.py"))}


def run_case(condition, *, seed, seconds, output, video=None):
    social = condition != "passive"
    report = run(seconds=seconds, seed=seed, scene_name="encounter", social=social,
                 social_sensing=condition != "sensing_off", body_contact=condition != "no_contact",
                 swap=condition == "swapped", output=output / condition, video=video)
    checks = {
        "completed": report["stop_reason"] == "duration",
        "food_conserved": abs(report["food_balance_error"]) < 1e-7,
        "both_upright": all(c["min_upright"] > .8 for c in report["contestants"]),
    }
    if condition in {"passive", "sensing_off"}:
        checks["no_social_actions"] = all(not any(c["social"]["counters"].values()) for c in report["contestants"])
    if condition == "no_contact":
        checks["no_physical_contact"] = report["any_contact_exchange_s"] == 0
    if condition in {"social", "swapped"}:
        checks["physical_push"] = sum(c["contact_push_s"] for c in report["contestants"]) > .1
        checks["retreat"] = any(c["social"]["counters"]["retreats"] > 0 for c in report["contestants"])
        checks["physical_withdrawal"] = any(c["social"]["max_retreat_mm"] > .25 for c in report["contestants"])
        checks["food_after_retreat"] = any(c["food_after_retreat"] > .05 for c in report["contestants"])
    result = {"condition": condition, "seed": seed, "checks": checks, "report": report,
              "sources": source_manifest()}
    path = output / f"validation-{condition}-seed{seed}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"condition": condition, "seed": seed, "checks": checks}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["social", "swapped", "passive", "sensing_off", "no_contact"], required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--output", type=Path, default=Path("runs/food-fight-validation"))
    parser.add_argument("--video", type=Path, help="Optional video, requires exactly one seed")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.video and len(args.seed) != 1:
        parser.error("--video requires one seed")
    before = source_manifest()
    results = [run_case(args.condition, seed=seed, seconds=args.seconds, output=args.output, video=args.video) for seed in args.seed]
    if source_manifest() != before:
        raise RuntimeError("Sources changed during validation; rerun the affected cases")
    if not all(all(row["checks"].values()) for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
