"""Bounded real-graph/real-body compatibility and semantic checkpoint smoke.

The cue-capable calibration and fixed readout created here are TEST FIXTURES.
They exercise transport/serialization only and are not evidence of learned
needs, validated cue encoding, semantic behavior, or successful food choice.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from fly_arena.checkpoint import code_digest, load_checkpoint, save_checkpoint
from fly_arena.config import load_config
from fly_arena.ethology import EthologySimulation
from fly_semantic.mapping import digest, file_digest, load_mapping
from fly_semantic.protocol import Event
from fly_semantic.readout import LinearReadout
from fly_semantic.runtime import SemanticSimulation, load_calibration, make_features, semantic_code_digest


def assert_state_equal(actual, expected, path="state"):
    """Exact values, permitting checkpoint list/tuple normalization only."""
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise AssertionError(f"State array differs at {path}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"State keys differ at {path}")
        for key in expected:
            # Wall duration is telemetry, never simulation dynamics. Native MuJoCo
            # timer durations are already canonicalized by the unchanged core.
            if key != "wall_time_s":
                assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(f"State sequence differs at {path}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_state_equal(a, b, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"State value differs at {path}: {actual!r} != {expected!r}")


def run(graph, calibration_dir, output, stage_b_calibration=None):
    graph, calibration_dir, output = Path(graph), Path(calibration_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mapping_path = calibration_dir / "mapping.json"
    mapping = load_mapping(mapping_path, graph)
    calibration = load_calibration(calibration_dir / "calibration.json", mapping)
    result = {"format": "fly_semantic.runtime_verification.v1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING", "checks": {}, "seed": 2718,
        "old_core_sha256_before": code_digest(), "semantic_code_before": semantic_code_digest(),
        "mapping_digest": mapping["digest"], "hunger_calibration_digest": calibration["digest"],
        "neuron_count": mapping["neuron_count"],
        "physical_seconds_requested_excluding_existing_constructor_warmup": .57,
        "comparison": "exact full nested state; wall_time_s excluded; old core canonicalizes native MuJoCo timers",
        "test_fixture_warning": "Fixed synthetic readout and explicitly synthetic cue calibration validate mechanics only; no learned-result or cue-calibration claim"}
    start = time.perf_counter()
    active = None
    try:
        cfg = load_config()
        kwargs = {"graph": graph, "seed": result["seed"], "warmup": True}
        active = EthologySimulation(cfg, **kwargs)
        initial = copy.deepcopy(active.get_state())
        baseline_compatibility = active.compatibility()
        for _ in range(10):
            active.step()
        baseline = copy.deepcopy(active.get_state())
        active.close()
        active = None
        print("baseline complete", flush=True)

        active = SemanticSimulation(cfg, enabled=False, mapping_path="intentionally-nonexistent", **kwargs)
        assert active.channel is None
        assert_state_equal(active.get_state(), initial)
        assert active.compatibility() == baseline_compatibility
        for _ in range(10):
            active.step()
        assert_state_equal(active.get_state(), baseline)
        result["checks"]["disabled_full_state_equivalence_100ms"] = "PASS"
        paused = copy.deepcopy(active.get_state())
        for _ in range(3):
            assert active.step(paused=True) is None
        assert_state_equal(active.get_state(), paused)
        result["checks"]["disabled_pause_no_change"] = "PASS"
        active.close()
        active = None
        del initial, baseline, paused
        print("disabled equivalence complete", flush=True)

        test_calibration = copy.deepcopy(calibration)
        test_calibration.pop("digest")
        test_calibration["cue_status"] = "CALIBRATED"
        test_calibration["cue_gain_mv"] = calibration["hunger_gain_mv"]
        test_calibration["verification_fixture"] = "SYNTHETIC MECHANICS ONLY: cue calibration overridden solely for queued-pulse resume test"
        test_calibration["digest"] = digest(test_calibration)
        fixture_path = output / "SYNTHETIC-mechanics-calibration.json"
        fixture_path.write_text(json.dumps(test_calibration, indent=2), encoding="utf-8")
        features = make_features(mapping, test_calibration)
        synthetic_head = LinearReadout(feature_digest=features.feature_digest, concept_ids=[1],
            mean=np.zeros(features.n_features), scale=np.ones(features.n_features),
            weights=np.full((features.n_features, 1), 1e-5), bias=[3.0], seed=0,
            diagnostics={"test_fixture": "fixed manually defined head, never fitted; not a learned result"})
        head_path = output / "SYNTHETIC-mechanics-head.npz"
        synthetic_head.save(head_path)
        enabled = {"enabled": True, "mapping_path": mapping_path, "calibration_path": fixture_path,
                   "readout_path": head_path, "episode_id": "runtime-test"}
        active = SemanticSimulation(cfg, **enabled, **kwargs)
        initial_enabled = copy.deepcopy(active.get_state())
        for concept, when, event_id in [(101, 100, "active-at-checkpoint"), (102, 400, "queued-at-checkpoint")]:
            assert active.channel.submit(Event(1, "runtime-test", event_id, "to_brain", concept, when, 1000, True, "schedule"), 0)
        for _ in range(25):
            active.step()
        saved_state = copy.deepcopy(active.get_state())
        assert active.brain.clock == 250 and active.channel.features.elapsed_ms == 250
        assert active.channel.inbox.active_concept_id == 101 and active.channel.inbox.pending_count == 1
        assert active.channel.outbox.poll_state(250)[1]["active"]
        assert np.any(active.channel.features.values())
        checkpoint_path = output / "fullbody-semantic-at-250ms.npz"
        save_checkpoint(checkpoint_path, saved_state, active.compatibility())
        assert active.step(paused=True) is None
        assert_state_equal(active.get_state(), saved_state)
        result["checks"]["enabled_pause_neural_body_queue_readout_unchanged"] = "PASS"
        for _ in range(5):
            active.step()
        uninterrupted = copy.deepcopy(active.get_state())
        active.reset("fresh-runtime-test")
        assert_state_equal(active.legacy.get_state(), initial_enabled["legacy"])
        assert active.channel.inbox.episode_id == "fresh-runtime-test"
        assert active.channel.inbox.pending_count == 0 and active.channel.inbox.active_concept_id is None
        assert active.channel.inbox.journal == () and active.channel.features.elapsed_ms == 0
        assert active.channel.last_scores == {} and active.channel.last_events == ()
        result["checks"]["reset_exact_initial_body_and_cleared_semantics"] = "PASS"
        active.step()
        result["checks"]["reset_next_step_runs"] = "PASS"
        active.close()
        active = None
        print("enabled checkpoint saved and reset complete", flush=True)

        active = SemanticSimulation(cfg, **enabled, **kwargs)
        restored, _ = load_checkpoint(checkpoint_path, active.compatibility())
        active.set_state(restored)
        assert_state_equal(active.get_state(), saved_state)
        for _ in range(5):
            active.step()
        assert_state_equal(active.get_state(), uninterrupted)
        result["checks"]["npz_fullbody_neural_queues_filters_exact_250_to_300ms"] = "PASS"
        result["checkpoint_sha256"] = file_digest(checkpoint_path)
        active.close()
        active = None
        del restored, saved_state, uninterrupted, initial_enabled
        print("exact physical continuation complete", flush=True)

        # Actual Stage B artifact: no synthetic cue permission and no model.
        actual_calibration_path = Path(stage_b_calibration) if stage_b_calibration else calibration_dir / "calibration.json"
        active = SemanticSimulation(cfg, enabled=True, mapping_path=mapping_path,
            calibration_path=actual_calibration_path, episode_id="hunger-only", **kwargs)
        assert active.channel.supported_concepts == [] and active.channel.calibrated_input_concepts == []
        try:
            active.channel.submit(Event(1, "hunger-only", "rejected", "to_brain", 101, 0, 1000, True, "human"), 0)
        except ValueError as error:
            assert "not calibrated" in str(error)
        else:
            raise AssertionError("Uncalibrated cue was accepted")
        active.step()
        assert active.channel.last_scores == {} and active.channel.last_events == ()
        result["checks"]["real_hunger_only_calibration_no_fake_outputs_or_cue_permission"] = "PASS"
        result["stage_b_calibration_digest"] = active.channel.calibration["digest"]
        active.close()
        active = None

        protected_path = Path("runs/food-fight-audit/compatibility-baseline.json")
        protected = json.loads(protected_path.read_text(encoding="utf-8"))
        changed = [item["path"] for item in protected if file_digest(item["path"]) != item["sha256"]]
        result["protected_baseline_files"] = len(protected)
        result["changed_protected_files"] = changed
        if changed:
            raise AssertionError(f"Protected files changed: {changed}")
        result["checks"]["legacy_sources_configs_tests_launchers_unchanged"] = "PASS"
        result["status"] = "PASS"
    except Exception:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        raise
    finally:
        if active is not None:
            active.close()
        result["wall_seconds"] = time.perf_counter() - start
        result["old_core_sha256_after"] = code_digest()
        result["semantic_code_after"] = semantic_code_digest()
        result["semantic_sources_changed_during_smoke"] = result["semantic_code_before"] != result["semantic_code_after"]
        (output / "runtime-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "checks": result["checks"], "wall_seconds": result["wall_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/graph"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("runs/semantic-v1/calibration-low"))
    parser.add_argument("--stage-b-calibration", type=Path, default=Path("runs/semantic-v1/calibration-stage-b/calibration.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/semantic-v1/verification"))
    args = parser.parse_args()
    run(args.graph, args.calibration_dir, args.output, args.stage_b_calibration)
