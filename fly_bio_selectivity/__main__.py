from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preregistered motion selectivity: unchanged vs 18/13ms delay proxy")
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--output", type=Path, default=Path("runs/bio-selectivity-v1"))
    parser.add_argument("--model", choices=("baseline", "delayed", "all"), default="all")
    parser.add_argument("--trial", type=int, nargs="+", help="Run only these preregistered trial indices")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    from .experiment import make_plan, run
    if args.plan_only:
        plan = make_plan(args.output)
        print(f"Frozen {len(plan['trials'])} trials in {args.output / 'plan.json'}")
    else:
        run(args.output, args.graph, model_filter=args.model, indices=args.trial)


if __name__ == "__main__":
    main()
