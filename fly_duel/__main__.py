"""Two flies sharing food: passive competition or an optional social controller."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=_nonnegative, default=25., help="Model seconds; 0 = until stopped")
    parser.add_argument("--seed", type=int, nargs="+", help="Default: 1,2,3 in batch mode; 1 with --single")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/duel")
    parser.add_argument("--no-contact-control", action="store_true",
                        help="Also run each seed with the bodies able to pass through each other")
    parser.add_argument("--social", action="store_true", help="Enable the engineered social controller")
    parser.add_argument("--scene", choices=("search", "encounter"),
                        help="Default: encounter with --social, otherwise the original search scene")
    parser.add_argument("--no-social-sensing", action="store_true", help="Hide the rival, retaining both bodies")
    parser.add_argument("--amount", type=_nonnegative, help="Initial food amount (scene default if omitted)")
    parser.add_argument("--resources", choices=("scarce", "ample"), default="scarce",
                        help="Encounter food budget: scarce=4 or ample=40; --amount overrides it")
    parser.add_argument("--single", action="store_true", help="One seed and position instead of a paired batch")
    parser.add_argument("--swap", action="store_true", help="Swap starting sides in a --single run")
    parser.add_argument("--viewer", action="store_true", help="Open a live window; Space pauses")
    parser.add_argument("--headless", action="store_true", help="Disable the viewer, including launcher defaults")
    parser.add_argument("--video", type=Path, help="Save a 30 fps video; requires --single")
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = [1] if args.single else [1, 2, 3]
    if args.swap and not args.single:
        parser.error("--swap requires --single; batch mode already runs both starting positions")
    args.viewer = args.viewer and not args.headless
    if (args.video or args.viewer or args.seconds == 0) and not args.single:
        parser.error("--video, --viewer and unlimited --seconds 0 require --single")
    if args.single and (len(args.seed) != 1 or args.no_contact_control):
        parser.error("--single accepts exactly one seed and no --no-contact-control")
    scene_name = args.scene or ("encounter" if args.social else "search")
    from .runner import run
    rows = []
    conditions = [True, False] if args.no_contact_control else [True]
    stop = False
    for contact in conditions:
        for seed in args.seed:
            for swap in ((args.swap,) if args.single else (False, True)):
                report = run(seconds=args.seconds, seed=seed, body_contact=contact,
                             output=args.output, swap=swap, scene_name=scene_name,
                             social=args.social, social_sensing=not args.no_social_sensing,
                             amount=args.amount, viewer_enabled=args.viewer, video=args.video,
                             resources=args.resources)
                # Resource budget changes the world; the social controller
                # observes only its own actual feeding history.
                rows.append(report)
                a, b = report["contestants"]
                print(f"seed {seed} contact={str(contact):5} swap={str(swap):5} "
                      f"| eaten {report['eaten_total']:6.2f} | share {report['winner_share']} "
                      f"| touching {report['body_contact_steps']:4} "
                      f"| {a['name']} {a['ingested']:6.2f} ({a['patch_time_s']:5.1f}s, odor {a['mean_odor']:.3f}) "
                      f"| {b['name']} {b['ingested']:6.2f} ({b['patch_time_s']:5.1f}s, odor {b['mean_odor']:.3f})",
                      flush=True)
                stop = report["stop_reason"] in ("interrupted", "window_closed")
                if stop:
                    break
            if stop:
                break
        if stop:
            break
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
