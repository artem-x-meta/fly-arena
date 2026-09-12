"""Console contracts use a dummy simulator, never a physical experiment."""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fly_semantic import __main__ as cli
from fly_semantic.protocol import InputCueQueue, OutputStateMachine


def arguments(tmp_path, *extra):
    graph = tmp_path / "graph"
    graph.mkdir(exist_ok=True)
    (graph / "manifest.json").write_text("{}")
    config = tmp_path / "config.toml"
    config.write_text("")
    calibration = tmp_path / "calibration"
    calibration.mkdir(exist_ok=True)
    (calibration / "mapping.json").write_text("{}")
    (calibration / "calibration.json").write_text("{}")
    model = tmp_path / "state-readout.npz"
    model.write_bytes(b"dummy-test-artifact")
    return cli.build_parser().parse_args(["demo", "--graph", str(graph), "--config", str(config),
        "--calibration", str(calibration), "--model", str(model), "--output", str(tmp_path / "output"), *extra])


class DummyChannel:
    def __init__(self, calibrated=()):
        self.inbox = InputCueQueue("dummy-episode")
        self.outbox = OutputStateMachine("dummy-episode", [1])
        self.supported_concepts = [1]
        self.calibrated_input_concepts = list(calibrated)
        self.last_scores = {}
        self.last_sample_ms = 0
        self.last_events = ()
        self.submitted = []

    def submit(self, event, now):
        self.submitted.append(event)
        return self.inbox.submit(event, now)

    def step(self, now):
        self.inbox.advance(now - 10)
        self.last_events = ()
        if now % 100 == 0:
            self.last_sample_ms = now
            self.last_scores = {1: 0.9}
            self.last_events = self.outbox.update(self.last_scores, now)


class DummySimulation:
    """Synthetic body diagnostics intentionally disagree with the dummy readout."""
    def __init__(self, enabled=True, calibrated=()):
        self.brain = SimpleNamespace(clock=0)
        self.channel = DummyChannel(calibrated) if enabled else None
        self.closed = False
        self.telemetry_calls = 0

    def step(self):
        self.brain.clock += 10
        if self.channel is not None:
            self.channel.step(self.brain.clock)

    def close(self):
        self.closed = True

    def compatibility(self):
        return {"dummy_test_only": True, "semantic_enabled": self.channel is not None}

    def telemetry(self):
        self.telemetry_calls += 1
        return {"energy": 95.0, "gut_amount": 3.0, "hunger": 0.05,
                "ingested_total": 7.0, "action": "FEED"}

    def get_state(self):
        result = {"brain": {"clock": self.brain.clock}, "body_test_marker": [1, 2, 3]}
        if self.channel is not None:
            result["semantic"] = {"inbox": self.channel.inbox.get_state(), "outbox": self.channel.outbox.get_state(),
                "last_scores": self.channel.last_scores, "last_sample_ms": self.channel.last_sample_ms}
        return result

    def set_state(self, state):
        self.brain.clock = state["brain"]["clock"]
        assert state["body_test_marker"] == [1, 2, 3]
        if self.channel is not None:
            semantic = state["semantic"]
            self.channel.inbox.set_state(semantic["inbox"])
            self.channel.outbox.set_state(semantic["outbox"])
            self.channel.last_scores = {int(k): v for k, v in semantic["last_scores"].items()}
            self.channel.last_sample_ms = semantic["last_sample_ms"]


def test_parser_defaults_and_help_do_not_create_simulation(capsys, monkeypatch):
    parsed = cli.resolve_arguments(cli.build_parser().parse_args(["demo"]))
    assert parsed.seconds == 10
    assert parsed.calibration == Path("runs/semantic-v1/calibration-selected")
    assert parsed.model.name == "state-readout.npz"
    monkeypatch.setattr(cli, "_build_simulation", lambda args: pytest.fail("help built a simulator"))
    with pytest.raises(SystemExit) as stop:
        cli.main(["demo", "--help"])
    assert stop.value.code == 0
    assert "--resume" in capsys.readouterr().out


def test_semantic_toml_false_disables_without_reading_model(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--seconds", "0")
    args.semantic_config = tmp_path / "semantic.toml"
    args.semantic_config.write_text('[semantic]\nenabled = false\n')
    args.model = tmp_path / "missing.npz"
    args.calibration = tmp_path / "missing-calibration"
    constructed = []

    def build(resolved):
        constructed.append(resolved.disabled)
        return DummySimulation(enabled=not resolved.disabled)

    monkeypatch.setattr(cli, "_build_simulation", build)
    summary = cli.run_demo(args, print_fn=lambda *items: None)
    assert constructed == [True]
    assert summary["semantic_enabled"] is False


def test_semantic_toml_paths_cli_overrides_and_explicit_disabled(tmp_path):
    semantic = tmp_path / "semantic.toml"
    semantic.write_text('[semantic]\nenabled = true\ncalibration = "calibration-a"\nmodel = "head-a.npz"\n')
    parsed = cli.build_parser().parse_args(["demo", "--semantic-config", str(semantic)])
    resolved = cli.resolve_arguments(parsed)
    assert resolved.disabled is False
    assert resolved.calibration == tmp_path / "calibration-a"
    assert resolved.model == tmp_path / "head-a.npz"
    parsed.model = Path("explicit.npz")
    parsed.disabled = True
    overridden = cli.resolve_arguments(parsed)
    assert overridden.model == Path("explicit.npz")
    assert overridden.disabled is True


@pytest.mark.parametrize("body", [
    '[semantic]\nenabled = "false"\n',
    '[semantic]\nseed = 42\n',
    '[semantic]\nenabled = true\nunknown = 1\n',
    '[semantic]\nenabled = true\ncore_mode = "plastic_core"\n',
    '[semantic]\nenabled = true\nseed = true\n',
    '[semantic]\nenabled = true\nsupported_concepts = [true]\n',
    '[semantic]\nenabled = true\npulse_ms = 100\n',
])
def test_invalid_semantic_configuration_is_rejected(tmp_path, body):
    args = arguments(tmp_path)
    args.semantic_config = tmp_path / "semantic.toml"
    args.semantic_config.write_text(body)
    with pytest.raises(ValueError):
        cli.resolve_arguments(args)


def test_semantic_config_seed_binds_mapping_without_changing_arena_seed(tmp_path):
    args = arguments(tmp_path, "--seed", "7")
    args.semantic_config = tmp_path / "semantic.toml"
    args.semantic_config.write_text('[semantic]\nenabled = true\nseed = 42\n')
    (args.calibration / "mapping.json").write_text('{"seed": 41}')
    resolved = cli.resolve_arguments(args)
    assert resolved.seed == 7
    with pytest.raises(ValueError, match="seed differs"):
        cli.validate_arguments(resolved)
    (args.calibration / "mapping.json").write_text('{"seed": 42}')
    cli.validate_arguments(resolved)


@pytest.mark.parametrize("seconds", ["nan", "inf", "-1", "0.001"])
def test_bad_duration_fails_before_building_body(tmp_path, monkeypatch, seconds):
    args = arguments(tmp_path, "--seconds", seconds)
    monkeypatch.setattr(cli, "_build_simulation", lambda args: pytest.fail("invalid arguments built a simulator"))
    with pytest.raises(ValueError):
        cli.run_demo(args)


def test_missing_trained_artifact_fails_without_silent_readout_fallback(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.model.unlink()
    monkeypatch.setattr(cli, "_build_simulation", lambda args: pytest.fail("missing model built a simulator"))
    with pytest.raises(FileNotFoundError, match="Required semantic artifact"):
        cli.run_demo(args)
    assert not args.output.exists()


def test_disabled_mode_ignores_all_semantic_artifact_paths(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--disabled", "--seconds", "0.1")
    args.model = tmp_path / "does-not-exist.npz"
    args.calibration = tmp_path / "also-missing"
    simulator = DummySimulation(enabled=False)
    monkeypatch.setattr(cli, "_build_simulation", lambda args: simulator)
    summary = cli.run_demo(args, print_fn=lambda *items: None)
    assert summary["semantic_enabled"] is False
    assert summary["outbound_event_count"] == 0
    assert summary["end_sim_time_ms"] == 100
    assert simulator.closed
    output = Path(summary["checkpoint"]).parent
    assert (output / "events.jsonl").read_text() == ""
    assert json.loads((output / "telemetry.jsonl").read_text())["semantic_enabled"] is False


def test_batch_streams_neural_outputs_separately_and_resumes_full_checkpoint(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--seconds", "0.2")
    created = []

    def build(args):
        simulator = DummySimulation()
        created.append(simulator)
        return simulator

    monkeypatch.setattr(cli, "_build_simulation", build)
    summary = cli.run_demo(args, print_fn=lambda *items: None)
    assert created[-1].closed
    output = Path(summary["checkpoint"]).parent
    outbound = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    telemetry = [json.loads(line) for line in (output / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(outbound) == 1
    assert outbound[0]["source"] == "neural_readout"
    assert outbound[0]["concept_id"] == 1 and outbound[0]["active"]
    neural = [row for row in telemetry if row["kind"] == "neural_readout_scores"]
    body = [row for row in telemetry if row["kind"] == "simulator_telemetry"]
    assert [row["sim_time_ms"] for row in neural] == [100, 200]
    assert [row["sim_time_ms"] for row in body] == [100, 200]
    assert all(row["hunger"] == 0.05 and row["energy"] == 95.0 and row["action"] == "FEED" for row in body)
    assert all("scores" not in row for row in body)
    assert all("energy" not in row and row["scores"] == {"1": 0.9} for row in neural)
    args.resume = Path(summary["checkpoint"])
    args.seconds = 0.1
    continued = cli.run_demo(args, print_fn=lambda *items: None)
    assert continued["start_sim_time_ms"] == 200
    assert continued["end_sim_time_ms"] == 300
    assert continued["outbound_event_count"] == 0
    assert Path(summary["checkpoint"]).exists()
    assert continued["checkpoint"] != summary["checkpoint"]


def test_interactive_input_and_poll_leave_simulation_paused(tmp_path):
    simulator = DummySimulation(calibrated=[101, 102, 103])
    events, telemetry, delivery = io.StringIO(), io.StringIO(), io.StringIO()
    session = cli.DemoSession(simulator, tmp_path, events, telemetry, delivery, print_fn=lambda *items: None)
    choices = iter(["p", "t", "l", "r", "a", "1", "p", "t", "q"])
    observed_clocks = []

    def user_input(prompt):
        observed_clocks.append(simulator.brain.clock)
        return next(choices)

    session.interactive(user_input)
    assert observed_clocks == [0, 0, 0, 0, 0, 0, 1000, 1000, 1000]
    assert [e.concept_id for e in simulator.channel.submitted] == [101, 102, 103]
    assert all(e.sim_time_ms == 0 and e.source == "human" for e in simulator.channel.submitted)
    assert simulator.brain.clock == 1000


def test_body_telemetry_is_explicitly_separate_from_cached_neural_poll(tmp_path):
    simulator = DummySimulation()
    output = []
    session = cli.DemoSession(simulator, tmp_path, io.StringIO(), io.StringIO(), io.StringIO(), print_fn=output.append)
    saved = simulator.channel.outbox.get_state()
    session.poll()
    assert simulator.telemetry_calls == 0
    session.show_body_telemetry()
    assert simulator.telemetry_calls == 1
    assert simulator.channel.outbox.get_state() == saved
    assert simulator.brain.clock == 0
    assert output[-1].startswith("Телеметрия организма")
    assert "энергия=95.000" in output[-1]
    assert "голод=0.050" in output[-1]


def test_uncalibrated_menu_inputs_are_rejected_without_submission(tmp_path):
    simulator = DummySimulation()
    outputs = []
    delivery = io.StringIO()
    session = cli.DemoSession(simulator, tmp_path, io.StringIO(), io.StringIO(), delivery, print_fn=outputs.append)
    session.announce()
    for concept in (101, 102, 103):
        assert session.submit(concept) is False
    assert simulator.channel.submitted == []
    assert simulator.brain.clock == 0
    assert all(json.loads(row)["reason"] == "input_unavailable" for row in delivery.getvalue().splitlines())
    assert any("отключены" in row for row in outputs)


def test_resume_checks_expected_compatibility_and_closes_on_failure(tmp_path, monkeypatch):
    from fly_arena.checkpoint import save_checkpoint

    args = arguments(tmp_path, "--seconds", "0")
    args.resume = tmp_path / "wrong.npz"
    save_checkpoint(args.resume, {"dummy": True}, {"wrong": "identity"})
    simulator = DummySimulation()
    monkeypatch.setattr(cli, "_build_simulation", lambda args: simulator)
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        cli.run_demo(args)
    assert simulator.closed
    assert not args.output.exists()
