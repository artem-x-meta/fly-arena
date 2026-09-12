"""Reproduce the declared contrast/gain checks without fitting parameters."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fly_bio.benchmark import run_case, sources
from fly_bio.graph import Connectome
from fly_bio.model import GradedConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--output", type=Path, default=Path("runs/bio-visual-v1/sensitivity"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = sources()
    cases = [(kind, .8, contrast) for contrast in (.2, .4)
             for kind in ("dark_loom", "retinal_mean_flash")]
    cases += [("dark_loom", gain, .8) for gain in (.6, .9)]
    (args.output / "plan.json").write_text(json.dumps({"sources": frozen,
        "cases": [{"condition": kind, "model": asdict(GradedConfig(gain=gain)), "contrast": contrast}
                  for kind, gain, contrast in cases],
        "fit": "No fitting or case-dependent parameter selection"}, indent=2), encoding="utf-8")
    graph = Connectome(args.graph)
    reports = [run_case(graph, args.output, kind=kind, config=GradedConfig(gain=gain), contrast=contrast)
               for kind, gain, contrast in cases]
    if sources() != frozen:
        raise RuntimeError("Neural model sources changed during sensitivity checks")
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
