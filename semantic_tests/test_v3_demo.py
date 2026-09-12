"""New diagnostic launcher delegates to the unchanged demo with exact artifacts."""
import copy
import json
from pathlib import Path

import pytest

from fly_semantic.mapping import digest
from semantic_tools import v3_demo as demo


@pytest.fixture
def ready(tmp_path):
    model = tmp_path / "arbitrary-new-head.npz"
    model.write_bytes(b"head bytes")
    calibration = tmp_path / "arbitrary-calibration"
    calibration.mkdir()
    calibrated = {"schema_version": 1, "status": "CALIBRATED", "cue_status": "UNVALIDATED",
                  "hunger_gain_mv": 5., "mapping_digest": "mapping"}
    calibrated["digest"] = digest(calibrated)
    (calibration / "calibration.json").write_text(json.dumps(calibrated), encoding="utf-8")
    document = {"schema_version": 1, "status": "DIAGNOSTIC_ENABLED",
                "model_sha256": demo.file_sha256(model), "calibration_digest": calibrated["digest"],
                "report": "reports/semantic_channel_v3/diagnostic-results.md"}
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    args = demo.build_parser().parse_args([
        "--readiness", str(path), "--model", str(model), "--calibration", str(calibration),
        "--config", str(tmp_path / "arena.toml"), "--seconds", "1.25", "--interactive",
        "--resume", str(tmp_path / "checkpoint.npz"), "--output", str(tmp_path / "output"),
        "--seed", "83", "--graph", str(tmp_path / "graph")])
    return args, document, calibrated


def test_digest_matches_existing_runtime_without_importing_a_backend():
    value = {"hunger_gain_mv": 5., "nested": {"text": "муха"}, "array": [1, 2, None]}
    assert demo.content_digest(value) == digest(value)


def test_forwards_args_and_menu_io_and_replaces_only_obsolete_notice(ready, monkeypatch):
    args, _, _ = ready
    before = copy.deepcopy(vars(args))
    printed, received = [], {}
    input_fn = lambda prompt: "q"
    score_notice = "Score — оценка внешнего считывателя; это не калиброванная вероятность и не величина потребности."
    cue_notice = "Еда слева / справа / впереди: входы отключены, калибровка ещё не пройдена."

    def original(forwarded, *, input_fn, print_fn):
        received.update(args=forwarded, input_fn=input_fn)
        print_fn(score_notice)
        print_fn(demo.OLD_DIAGNOSTIC_NOTICE)
        print_fn(cue_notice)
        print_fn("Пауза: 1 / p / t / s / q")
        print_fn("Checkpoint: saved")
        return {"unchanged": "summary"}

    monkeypatch.setattr(demo.original, "run_demo", original)
    result = demo.run_demo(args, input_fn=input_fn, print_fn=printed.append)
    assert result == {"unchanged": "summary"}
    assert vars(args) == before
    assert all(getattr(received["args"], key) == value for key, value in before.items())
    assert received["args"].command == "demo"
    assert received["args"].semantic_config is None
    assert received["input_fn"] is input_fn
    assert printed[0] == score_notice and printed[2] == cue_notice
    assert printed[3:] == ["Пауза: 1 / p / t / s / q", "Checkpoint: saved"]
    assert "Экспериментальный выход" in printed[1]
    assert "diagnostic-results.md" in printed[1]
    assert demo.OLD_DIAGNOSTIC_NOTICE not in printed


@pytest.mark.parametrize("mutation", ["status", "schema", "model", "calibration_content", "calibration_identity", "cue_enabled", "report"])
def test_readiness_failures_stop_before_existing_demo(ready, monkeypatch, mutation):
    args, document, calibrated = ready
    if mutation == "status":
        document["status"] = "TARGET_NOT_REACHED"
    elif mutation == "schema":
        document["schema_version"] = True
    elif mutation == "model":
        args.model.write_bytes(b"replacement head")
    elif mutation == "calibration_content":
        calibrated["hunger_gain_mv"] = 6.
    elif mutation == "calibration_identity":
        document["calibration_digest"] = "different"
    elif mutation == "cue_enabled":
        calibrated["cue_status"] = "CALIBRATED"
        calibrated.pop("digest")
        calibrated["digest"] = digest(calibrated)
        document["calibration_digest"] = calibrated["digest"]
    else:
        document["report"] = ""
    args.readiness.write_text(json.dumps(document), encoding="utf-8")
    (args.calibration / "calibration.json").write_text(json.dumps(calibrated), encoding="utf-8")
    monkeypatch.setattr(demo.original, "run_demo", lambda *a, **k: pytest.fail("must fail before startup"))
    with pytest.raises(ValueError):
        demo.run_demo(args)


def test_readiness_is_read_once_before_simulation_not_by_print_callback(ready, monkeypatch):
    args, _, _ = ready
    printed = []

    def original(forwarded, *, input_fn, print_fn):
        # Simulate readiness storage becoming unavailable after startup validation.
        args.readiness.unlink()
        for _ in range(3):
            print_fn(demo.OLD_DIAGNOSTIC_NOTICE)
        return {"ok": True}

    monkeypatch.setattr(demo.original, "run_demo", original)
    assert demo.run_demo(args, print_fn=printed.append) == {"ok": True}
    assert len(printed) == 3


def test_disabled_mode_preserves_legacy_artifact_independence(ready, monkeypatch):
    args, _, _ = ready
    args.disabled = True
    args.readiness.unlink()
    args.model.unlink()
    printed = []

    def original(forwarded, *, input_fn, print_fn):
        assert forwarded.disabled is True
        assert print_fn is printed.append or print_fn == printed.append
        print_fn("Семантический канал отключён. Работает прежняя симуляция.")
        return {"semantic_enabled": False}

    monkeypatch.setattr(demo.original, "run_demo", original)
    assert demo.run_demo(args, print_fn=printed.append) == {"semantic_enabled": False}
    assert len(printed) == 1 and "отключён" in printed[0]


def test_explicit_readiness_argument_is_required():
    with pytest.raises(SystemExit) as error:
        demo.build_parser().parse_args(["--model", "head", "--calibration", "cal", "--config", "arena"])
    assert error.value.code == 2
