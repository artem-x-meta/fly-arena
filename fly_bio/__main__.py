from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Independent graded MaleCNS visual benchmark; original fly unchanged")
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--output", type=Path, default=Path("runs/bio-visual"))
    parser.add_argument("--gain", type=float, default=.8)
    parser.add_argument("--case", action="append", help="condition[:eye[:intervention]], repeatable; default display-control suite")
    args = parser.parse_args()
    cases = None
    if args.case:
        cases = []
        for value in args.case:
            fields = value.split(":")
            if len(fields) > 3:
                parser.error("--case is condition[:eye[:intervention]]")
            cases.append((fields[0], fields[1] if len(fields) > 1 else "both", fields[2] if len(fields) > 2 else "none"))
    from .benchmark import benchmark
    benchmark(args.graph, args.output, cases=cases, gain=args.gain)


if __name__ == "__main__":
    main()
