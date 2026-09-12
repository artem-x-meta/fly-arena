from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Opt-in circuit laboratory; Fly Arena v0.3 remains unchanged")
    sub = parser.add_subparsers(dest="action", required=True)
    reproduce = sub.add_parser("reproduce", help="Reimplement the published GF equations and save curves")
    reproduce.add_argument("--output", type=Path, default=ROOT / "runs/circuit-lab/paper")
    audit = sub.add_parser("audit", help="Read-only MaleCNS LC4/LPLC2 to GF audit")
    audit.add_argument("--data", type=Path, default=ROOT / "data")
    audit.add_argument("--output", type=Path, default=ROOT / "runs/circuit-lab/anatomy")
    propagate = sub.add_parser("propagate", help="Diagnostic direct LC4/LPLC2 injections into the unchanged LIF graph")
    propagate.add_argument("--data", type=Path, default=ROOT / "data")
    propagate.add_argument("--output", type=Path, default=ROOT / "runs/circuit-lab/propagation")
    propagate.add_argument("--seed", type=int, default=1)
    run = sub.add_parser("run", help="Observe the paper model, or explicitly let it request escape")
    run.add_argument("--control", choices=["observe", "paper"], default="observe")
    run.add_argument("--config", type=Path, default=ROOT / "circuit_configs/ache2019.toml")
    run.add_argument("--data", type=Path, default=ROOT / "data")
    run.add_argument("--output", type=Path, default=ROOT / "runs/circuit-lab/live")
    run.add_argument("--seconds", type=float, default=4.)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--headless", action="store_true")
    run.add_argument("--blind", action="store_true")
    run.add_argument("--motor-off", action="store_true")
    run.add_argument("--threshold-mv", type=float, default=1., help="Engineered body trigger, not a paper spike threshold")
    run.add_argument("--block", action="append", choices=["input", "lc4", "lplc2", "i1", "i2", "output"], default=[])
    run.add_argument("--video", type=Path)
    run.add_argument("--resume", type=Path)
    diagnose = sub.add_parser("diagnose", help="Locate where the visual signal is lost before LC4/LPLC2")
    diagnose.add_argument("--data", type=Path, default=ROOT / "data")
    diagnose.add_argument("--config", type=Path, default=ROOT / "circuit_configs/ache2019.toml")
    diagnose.add_argument("--output", type=Path, default=ROOT / "runs/circuit-lab/diagnosis")
    diagnose.add_argument("--seconds", type=float, default=2.5)
    diagnose.add_argument("--seed", type=int, default=1)
    diagnose.add_argument("--skip-structure", action="store_true", help="Reuse a stored structural audit")
    diagnose.add_argument("--ceiling", type=float, default=None,
                          help="Make the arena lid visible at this grey level; omit to keep the blank sky")
    diagnose.add_argument("--free-body", action="store_true",
                          help="Let both coverage runs move; they then diverge as soon as one escapes")
    retina = sub.add_parser("retina", help="Compare retinal codes on the cells the object covers")
    retina.add_argument("--data", type=Path, default=ROOT / "data")
    retina.add_argument("--config", type=Path, default=ROOT / "circuit_configs/ache2019.toml")
    retina.add_argument("--diagnosis", type=Path, default=ROOT / "runs/circuit-lab/diagnosis/diagnosis.json")
    retina.add_argument("--output", type=Path, default=ROOT / "runs/circuit-lab/retina")
    retina.add_argument("--mode", action="append", default=[], help="Repeat to select codes; default is all three")
    retina.add_argument("--seconds", type=float, default=3.)
    retina.add_argument("--seed", type=int, default=1)
    retina.add_argument("--ceiling", type=float, default=None)
    flash = sub.add_parser("flash", help="Uniform illumination step: does a code hold a sustained offset")
    flash.add_argument("--data", type=Path, default=ROOT / "data")
    flash.add_argument("--config", type=Path, default=ROOT / "circuit_configs/ache2019.toml")
    flash.add_argument("--output", type=Path, default=ROOT / "runs/circuit-lab/retina")
    flash.add_argument("--mode", action="append", default=[])
    flash.add_argument("--seconds", type=float, default=3.)
    flash.add_argument("--at", type=float, default=1., dest="at_s")
    flash.add_argument("--factor", type=float, default=1.6)
    flash.add_argument("--seed", type=int, default=1)
    verify = sub.add_parser("verify", help="Circuit interventions, exact observer/resume and legacy file integrity")
    verify.add_argument("--suite", choices=["compatibility", "interventions", "observer", "checkpoint", "all"], default="all")
    verify.add_argument("--seed", type=int, default=1)
    verify.add_argument("--components", action="store_true")
    verify.add_argument("--output", type=Path, default=ROOT / "runs/circuit-lab/validation")
    args = parser.parse_args()
    if args.action == "reproduce":
        from .reproduce import reproduce as work
        work(args.output)
    elif args.action == "audit":
        from .anatomy import audit as work
        work(args.data / "graph", args.output)
    elif args.action == "propagate":
        from .propagation import propagation
        propagation(args.data / "graph", args.output, seed=args.seed)
    elif args.action == "diagnose":
        from .diagnose import diagnose as work
        work(args.data, args.output, config=args.config, seconds=args.seconds,
             seed=args.seed, skip_structure=args.skip_structure, hold_body=not args.free_body,
             ceiling=args.ceiling)
    elif args.action == "retina":
        from .retina import experiment, MODES
        experiment(args.data, args.output, config_path=args.config, diagnosis=args.diagnosis,
                   modes=tuple(args.mode) or MODES, seconds=args.seconds, seed=args.seed,
                   ceiling=args.ceiling)
    elif args.action == "flash":
        from .retina import flash as work, MODES
        work(args.data, args.output, config_path=args.config, modes=tuple(args.mode) or MODES,
             seconds=args.seconds, seed=args.seed, at_s=args.at_s, factor=args.factor)
    elif args.action == "verify":
        from .validation import compatibility_report, interventions, observer_equivalence, checkpoint_equivalence
        if args.suite in ("compatibility", "all"):
            compatibility_report(args.output)
        if args.suite in ("interventions", "all"):
            interventions(args.output / f"seed{args.seed}", seed=args.seed, components=args.components)
        if args.suite in ("observer", "all"):
            observer_equivalence(args.output)
        if args.suite in ("checkpoint", "all"):
            checkpoint_equivalence(args.output)
    else:
        import math
        if not math.isfinite(args.seconds) or args.seconds < 0:
            parser.error("seconds must be finite and >=0")
        from .runner import run as work
        try:
            work(args)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
