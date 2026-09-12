"""Run a specifically validated experimental head through the existing demo.

This adapter changes only the obsolete v1-specific announcement. All simulation,
console-menu, telemetry, checkpoint and resume behavior stays in the original
fly_semantic entry point. Readiness is an artifact identity check before startup;
this module never writes it, enables a default, trains a head, or updates files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from fly_semantic import __main__ as original


OLD_DIAGNOSTIC_NOTICE = (
    "Диагностическая версия: в проверенном прогоне сигнал «Нужна еда» сохранялся после кормления. "
    "Телеметрия организма доступна отдельно: t."
)


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def content_digest(value):
    # Match the existing semantic artifact format without loading a neural backend.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_readiness(path, model, calibration):
    """Check a small explicit readiness document once, before any simulation."""
    path, model, calibration = Path(path), Path(model), Path(calibration)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1 or document.get("status") != "DIAGNOSTIC_ENABLED"):
        raise ValueError("Readiness must be schema_version=1 with status DIAGNOSTIC_ENABLED")
    if document.get("model_sha256") != file_sha256(model):
        raise ValueError("Readiness model SHA-256 differs from the requested head")
    calibrated = json.loads((calibration / "calibration.json").read_text(encoding="utf-8"))
    plain = dict(calibrated)
    if plain.pop("digest", None) != content_digest(plain):
        raise ValueError("Requested calibration content digest is invalid")
    if document.get("calibration_digest") != calibrated["digest"]:
        raise ValueError("Readiness calibration digest differs from the requested calibration")
    if calibrated.get("status") != "CALIBRATED" or calibrated.get("cue_status") == "CALIBRATED":
        raise ValueError("This diagnostic demo requires calibrated hunger and disabled directional cues")
    if not isinstance(document.get("report"), str) or not document["report"].strip():
        raise ValueError("Readiness must name the report describing the diagnostic limits")
    return document


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True, help="Explicit DIAGNOSTIC_ENABLED artifact")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True, help="Directory with mapping.json and calibration.json")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--interactive", action="store_true", help="Existing fixed console menu, initially paused")
    parser.add_argument("--output", type=Path, default=Path("runs/semantic-v3/demo"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--disabled", action="store_true", help="Existing legacy-only mode; semantic artifact checks are skipped")
    return parser


def run_demo(args, *, input_fn=input, print_fn=print):
    # Construct exactly the existing entry point's argument shape. Extra readiness
    # provenance stays in summary.arguments; it is never given to a neural head.
    forwarded = argparse.Namespace(**vars(args))
    forwarded.command, forwarded.semantic_config = "demo", None
    if args.disabled:
        return original.run_demo(forwarded, input_fn=input_fn, print_fn=print_fn)
    readiness = validate_readiness(args.readiness, args.model, args.calibration)
    notice = ("Экспериментальный выход «Нужна еда»: результаты и ограничения смотри в отчёте "
              f"{readiness['report']}. Телеметрия организма доступна отдельно: t.")

    def diagnostic_print(*values, **kwargs):
        if len(values) == 1 and values[0] == OLD_DIAGNOSTIC_NOTICE:
            return print_fn(notice, **kwargs)
        return print_fn(*values, **kwargs)

    return original.run_demo(forwarded, input_fn=input_fn, print_fn=diagnostic_print)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_demo(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
