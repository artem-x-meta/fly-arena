from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Strict copy migration for the known optional fly-name factory change")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("migrate")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--graph", type=Path)
    command.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix(".migration.json")
    if report_path.resolve() in (args.input.resolve(), args.output.resolve()) or report_path.exists():
        parser.error("Migration report must be a new, separate file")
    from .migration import migrate
    try:
        report = migrate(args.input, args.output, graph=args.graph)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Verified migration copy: {args.output}; physical time {report['physics_time_s']:.2f}s.", flush=True)
    print(f"Input unchanged. Exact state arrays preserved. Audit: {report_path}", flush=True)


if __name__ == "__main__":
    main()
