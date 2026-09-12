"""Console demo for the trained semantic readout and existing physical arena.

Experiment commands remain in semantic_tools.audit, semantic_tools.calibrate and
semantic_tools.state_experiment; this entry point does not retrain anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import tomllib


DEFAULT_CALIBRATION = Path("runs/semantic-v1/calibration-selected")
DEFAULT_MODEL = Path("runs/semantic-v1/state-selected/state-readout.npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the existing arena with a trained neural state readout")
    demo.add_argument("--graph", type=Path, default=Path("data/graph"))
    demo.add_argument("--semantic-config", type=Path,
                      help="Optional TOML with [semantic].enabled; explicit demo enables without this file")
    demo.add_argument("--calibration", type=Path,
                      help=f"Directory containing mapping.json and calibration.json; overrides TOML (default {DEFAULT_CALIBRATION})")
    demo.add_argument("--model", type=Path,
                      help=f"Required trained readout; overrides TOML, ignored when disabled (default {DEFAULT_MODEL})")
    demo.add_argument("--config", type=Path, default=Path("configs/ethology-unscaled.toml"))
    demo.add_argument("--seconds", type=float, default=10,
                      help="Additional simulation seconds in batch mode (default 10); interactive mode advances 1 s per request")
    demo.add_argument("--interactive", action="store_true", help="Start paused at a fixed console menu; no separate viewer")
    demo.add_argument("--output", type=Path, default=Path("runs/semantic-v1/demo"))
    demo.add_argument("--resume", type=Path, help="Continue a complete compatible checkpoint")
    demo.add_argument("--disabled", action="store_true", help="Run unchanged legacy simulation without semantic artifacts")
    demo.add_argument("--seed", type=int, default=1)
    return parser


def resolve_arguments(args):
    """Resolve the actual toggle and artifact paths without importing a backend.

    Paths in TOML are relative to the TOML directory. Explicit CLI artifact paths
    override them. --disabled always wins; the plain demo command enables the
    channel when no semantic configuration was supplied. The semantic seed is
    checked against a saved mapping and never reseeds the legacy simulation.
    """
    resolved = argparse.Namespace(**vars(args))
    settings = {}
    if args.semantic_config is not None:
        with args.semantic_config.open("rb") as stream:
            document = tomllib.load(stream)
        settings = document.get("semantic")
        allowed = {"enabled", "core_mode", "seed", "supported_concepts", "pulse_ms", "calibration", "model"}
        if (set(document) != {"semantic"} or not isinstance(settings, dict)
                or set(settings) - allowed or type(settings.get("enabled")) is not bool):
            raise ValueError("Semantic TOML requires a [semantic] table with enabled=true/false and known fields")
        if settings.get("core_mode", "frozen_core") != "frozen_core":
            raise ValueError("This semantic stage supports frozen_core only")
        seed = settings.get("seed", 42)
        if type(seed) is not int or seed < 0:
            raise ValueError("Semantic mapping seed must be a nonnegative integer")
        supported = settings.get("supported_concepts", [1])
        if not isinstance(supported, list) or len(supported) != 1 or type(supported[0]) is not int or supported[0] != 1:
            raise ValueError("This trained demo supports NEED_FOOD [1] only")
        pulse = settings.get("pulse_ms", 250)
        if type(pulse) not in (int, float) or pulse != 250:
            raise ValueError("This calibrated semantic stage requires pulse_ms=250")
        for key in ("calibration", "model"):
            if key in settings and (not isinstance(settings[key], str) or not settings[key].strip()):
                raise ValueError(f"semantic.{key} must be a nonempty path string")
    resolved.disabled = args.disabled or settings.get("enabled", True) is False
    resolved.semantic_settings = dict(settings)
    for key, fallback in (("calibration", DEFAULT_CALIBRATION), ("model", DEFAULT_MODEL)):
        value = getattr(args, key)
        if value is None:
            value = args.semantic_config.parent / settings[key] if key in settings else fallback
        setattr(resolved, key, value)
    return resolved


def validate_arguments(args) -> None:
    if not math.isfinite(args.seconds) or args.seconds < 0:
        raise ValueError("--seconds must be finite and nonnegative")
    steps = args.seconds * 100
    if not math.isfinite(steps) or abs(steps - round(steps)) > 1e-7:
        raise ValueError("--seconds must be a multiple of the existing 0.01 s simulation step")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    for path in (args.config, args.graph / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required arena artifact is missing: {path}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {args.resume}")
    if not args.disabled:
        for path in (args.model, args.calibration / "mapping.json", args.calibration / "calibration.json"):
            if not path.is_file():
                raise FileNotFoundError(f"Required semantic artifact is missing: {path}. Run the documented calibration/training commands first.")
        if "seed" in args.semantic_settings:
            mapping = json.loads((args.calibration / "mapping.json").read_text(encoding="utf-8"))
            if mapping.get("seed") != args.semantic_settings["seed"]:
                raise ValueError("Semantic configuration seed differs from the saved input mapping")


def _build_simulation(args):
    # Keep --help and argument/artifact failures independent of expensive imports.
    from fly_arena.config import load_config
    from .runtime import SemanticSimulation

    kwargs = {"graph": args.graph, "seed": args.seed}
    if not args.disabled:
        from fly_arena.brain import BrainConfig

        calibration = json.loads((args.calibration / "calibration.json").read_text(encoding="utf-8"))
        try:
            kwargs["brain_config"] = BrainConfig(**calibration["brain_config"])
        except (TypeError, KeyError) as error:
            raise ValueError("Calibration does not contain a valid BrainConfig") from error
    return SemanticSimulation(load_config(args.config), enabled=not args.disabled,
        mapping_path=args.calibration / "mapping.json", calibration_path=args.calibration / "calibration.json",
        readout_path=None if args.disabled else args.model, episode_id=f"demo-seed-{args.seed}", **kwargs)


def _save_checkpoint(path, simulation) -> None:
    from fly_arena.checkpoint import save_checkpoint

    save_checkpoint(path, simulation.get_state(), simulation.compatibility())


def _restore_checkpoint(path, simulation) -> None:
    from fly_arena.checkpoint import load_checkpoint

    state, _ = load_checkpoint(path, expected=simulation.compatibility())
    simulation.set_state(state)


def _write_json(stream, value) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    stream.flush()


class DemoSession:
    """Blocking console I/O is called only between completed simulation steps."""

    def __init__(self, simulation, run_dir: Path, event_stream, telemetry_stream, delivery_stream, *, print_fn=print):
        self.simulation = simulation
        self.run_dir = run_dir
        self.events = event_stream
        self.telemetry = telemetry_stream
        self.delivery = delivery_stream
        self.print = print_fn
        self.last_logged_sample = -1
        self.last_logged_body_ms = -1
        self.last_logged_journal = None
        self.event_count = 0
        self.input_sequence = 0
        self.checkpoint = run_dir / "checkpoint.npz"

    @property
    def now_ms(self) -> int:
        return int(self.simulation.brain.clock)

    def announce(self) -> None:
        from .protocol import REGISTRY

        channel = self.simulation.channel
        if channel is None:
            self.print("Семантический канал отключён. Работает прежняя симуляция.")
            return
        labels = ", ".join(REGISTRY[item].label for item in channel.supported_concepts)
        self.print(f"Считыватель нейронной активности: {labels or 'нет обученных выходов'}.")
        self.print("Score — оценка внешнего считывателя; это не калиброванная вероятность и не величина потребности.")
        self.print("Диагностическая версия: в проверенном прогоне сигнал «Нужна еда» сохранялся после кормления. Телеметрия организма доступна отдельно: t.")
        if not channel.calibrated_input_concepts:
            self.print("Еда слева / справа / впереди: входы отключены, калибровка ещё не пройдена.")

    def record(self) -> None:
        from .protocol import REGISTRY

        channel = self.simulation.channel
        if self.now_ms % 100 == 0 and self.now_ms != self.last_logged_body_ms:
            _write_json(self.telemetry, self.body_telemetry())
            self.last_logged_body_ms = self.now_ms
        if channel is None:
            return
        if channel.last_sample_ms and channel.last_sample_ms != self.last_logged_sample:
            self.last_logged_sample = channel.last_sample_ms
            _write_json(self.telemetry, {"kind": "neural_readout_scores", "sim_time_ms": channel.last_sample_ms,
                                         "scores": channel.last_scores, "score_semantics": "uncalibrated_decoder_score"})
        for event in channel.last_events:
            _write_json(self.events, event.to_dict())
            self.event_count += 1
            state = "активно" if event.active else "неактивно"
            self.print(f"{event.sim_time_ms / 1000:.2f} с | {REGISTRY[event.concept_id].label}: {state} | score={event.score:.3f}")
        # The queue owns a bounded ring; stream only the newly appended suffix.
        journal = list(channel.inbox.journal)
        offset = 0
        if self.last_logged_journal is not None:
            for index in range(len(journal) - 1, -1, -1):
                if journal[index] == self.last_logged_journal:
                    offset = index + 1
                    break
        for row in journal[offset:]:
            _write_json(self.delivery, {"kind": "semantic_delivery", **row})
        if journal:
            self.last_logged_journal = journal[-1]

    def advance(self, steps: int) -> None:
        for _ in range(steps):
            self.simulation.step()
            self.record()

    def poll(self) -> None:
        from .protocol import REGISTRY

        channel = self.simulation.channel
        if channel is None:
            self.print("Семантический канал отключён.")
            return
        cached = channel.outbox.poll_state(self.now_ms)
        if not cached:
            self.print("Нет обученных выходов.")
        for concept, value in cached.items():
            if value["score"] is None:
                status = "считыватель ещё не выдал результат"
            else:
                status = "активно" if value["active"] else "неактивно"
                status += f", score={value['score']:.3f}"
            if value["stale"]:
                status += ", данные устарели или ещё не получены"
            self.print(f"{REGISTRY[concept].label}: {status}")

    def body_telemetry(self) -> dict:
        """Read simulator diagnostics for logging/UI only, never for the decoder."""
        source = self.simulation.telemetry()
        return {"kind": "simulator_telemetry", "sim_time_ms": self.now_ms,
                "semantic_enabled": self.simulation.channel is not None,
                **{key: float(source[key]) for key in ("energy", "gut_amount", "hunger", "ingested_total")},
                "action": str(source["action"])}

    def show_body_telemetry(self) -> None:
        value = self.body_telemetry()
        self.print(f"Телеметрия организма, {value['sim_time_ms'] / 1000:.2f} с: "
                   f"энергия={value['energy']:.3f}; пища в кишечнике={value['gut_amount']:.3f}; "
                   f"голод={value['hunger']:.3f}; съедено={value['ingested_total']:.3f}; "
                   f"действие={value['action']}.")

    def submit(self, concept_id: int) -> bool:
        from .protocol import Event, SCHEMA_VERSION

        channel = self.simulation.channel
        if channel is None or concept_id not in channel.calibrated_input_concepts:
            self.print("Этот вход отключён: семантический канал или калибровка недоступны.")
            _write_json(self.delivery, {"kind": "semantic_delivery", "sim_time_ms": self.now_ms,
                "concept_id": concept_id, "status": "rejected", "reason": "input_unavailable"})
            return False
        self.input_sequence += 1
        # Distinguish a resumed console session without supplying this ID to neurons.
        event_id = f"human-{self.run_dir.name}-{self.input_sequence}"
        event = Event(SCHEMA_VERSION, channel.inbox.episode_id, event_id, "to_brain", concept_id,
                      self.now_ms, 2000, True, "human")
        accepted = channel.submit(event, self.now_ms)
        _write_json(self.delivery, {"kind": "human_input", "event": event.to_dict(), "accepted": accepted})
        self.print("Подсказка поставлена в очередь следующего нейрошага." if accepted else "Подсказка отклонена; причина записана в delivery.jsonl.")
        return accepted

    def save(self) -> None:
        _save_checkpoint(self.checkpoint, self.simulation)
        self.print(f"Checkpoint: {self.checkpoint}")

    def interactive(self, input_fn=input) -> None:
        while True:
            self.print(f"\nПауза на {self.now_ms / 1000:.2f} с. 1: шаг 1 с; p: нейронные сигналы; t: телеметрия организма; l/r/a: еда слева/справа/впереди; s: сохранить; q: выйти.")
            try:
                choice = input_fn("> ").strip().lower()
            except EOFError:
                return
            if choice == "q":
                return
            if choice == "1":
                self.advance(100)
            elif choice == "p":
                self.poll()
            elif choice == "t":
                self.show_body_telemetry()
            elif choice in ("l", "r", "a"):
                self.submit({"l": 101, "r": 102, "a": 103}[choice])
            elif choice == "s":
                self.save()
            else:
                self.print("Допустимы только пункты меню: 1, p, t, l, r, a, s, q.")


def run_demo(args, *, input_fn=input, print_fn=print) -> dict:
    args = resolve_arguments(args)
    validate_arguments(args)
    simulation = _build_simulation(args)
    try:
        if args.resume is not None:
            _restore_checkpoint(args.resume, simulation)
        # Provenance only: wall time is never passed to the channel or simulation.
        run_dir = args.output / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S-%fZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        start_ms = int(simulation.brain.clock)
        status = "completed"
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as events, \
             (run_dir / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry, \
             (run_dir / "delivery.jsonl").open("w", encoding="utf-8") as delivery:
            session = DemoSession(simulation, run_dir, events, telemetry, delivery, print_fn=print_fn)
            session.announce()
            try:
                if args.interactive:
                    session.interactive(input_fn)
                else:
                    session.advance(round(args.seconds * 100))
                    session.poll()
            except KeyboardInterrupt:
                status = "interrupted"
                print_fn("Остановлено пользователем; сохраняю состояние.")
            session.save()
            summary = {"status": status, "semantic_enabled": not args.disabled,
                "start_sim_time_ms": start_ms, "end_sim_time_ms": session.now_ms,
                "supported_concepts": [] if simulation.channel is None else simulation.channel.supported_concepts,
                "calibrated_input_concepts": [] if simulation.channel is None else simulation.channel.calibrated_input_concepts,
                "outbound_event_count": session.event_count, "checkpoint": str(session.checkpoint),
                "resumed_from": None if args.resume is None else str(args.resume),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "compatibility": simulation.compatibility(),
                "limitations": "Fixed labels from a trained external neural readout. Existing engineered body control remains. Cue delivery does not imply learned cue-guided navigation."}
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            print_fn(f"Готово: {session.now_ms / 1000:.2f} с модельного времени. Логи: {run_dir}")
            return summary
    finally:
        simulation.close()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_demo(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
