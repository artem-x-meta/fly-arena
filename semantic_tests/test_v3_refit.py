"""Refit data lineage, bounded candidates and genuinely fresh test identities."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from fly_semantic.mapping import digest, file_digest
from fly_semantic.readout import LinearReadout
from semantic_tools import physical_learning as physical
from semantic_tools import v3_learning as v3
from semantic_tools import v3_refit as refit


def _seal(value):
    return value | {"digest": digest(value)}


def old_plan(source):
    episodes = []
    for split, group in (("train", 1201), ("train", 1202), ("train", 1203),
                         ("validation", 2201), ("test", 4201), ("test", 4202), ("test", 4203)):
        for index, scene in enumerate(refit.SCENES):
            cfg = {"initial": {"energy": 10., "sleep_pressure": .1}, "environment": {"light": .8},
                "organism": {"intake_rate": 2.8}, "food": [] if scene.endswith("away") else [{"amount": .8}]}
            episodes.append({"episode_id": f"{split}-{group}-{scene}", "split": split, "seed_group": group,
                "seed": group * 10 + index, "scene": scene, "duration_ms": 2000,
                "expected_release": scene in ("meal_fast", "meal_slow"), "config": cfg})
    return _seal({**v3.protocol_fields(), "episodes": episodes, "hunger_gain_mv": 5.,
        "feature_digest": "fixture-features-5", "calibration_digest": "fixture-calibration-5",
        "nominal_weights_sha256": "fixture-weights", "mapping_digest": "fixture-mapping",
        "brain_config": {}, "max_iter": 500, "source_hashes": {str(source): file_digest(source)},
        "collection_conditions": ["connected", "homeostasis_off", "transmission_off"]})


def row_fixture(study, ep, *, body=True, condition="connected", source=None):
    times = np.arange(100, ep["duration_ms"] + 1, 100, dtype=np.int64)
    high = ep["scene"] in ("hungry_away", "limited_meal", "meal_fast", "meal_slow")
    transitions = [{"time_ms": 0, "active": True}] if high else []
    if ep["expected_release"]: transitions.append({"time_ms": 790, "active": False})
    y = physical._teacher_at(transitions, times - 10)
    X = np.zeros((len(times), 1024))
    X[:, 0] = 30 * y if condition == "connected" else 0.
    exact = np.repeat(np.where(y, .8, .3), 10).astype(np.float64)
    # Restore precise transition at tick79 rather than end-of-sample approximation.
    if ep["expected_release"]: exact[:79] = .8; exact[79:] = .3
    meta = {"episode": ep, "condition": condition, "plan_digest": study["digest"],
        "feature_digest": study["feature_digest"], "calibration_digest": study["calibration_digest"],
        "nominal_weights_sha256": study["nominal_weights_sha256"], "target_transitions": transitions,
        "source_hashes": study["source_hashes"], "body_physics": body,
        "stimulus_channels": [{"name": "semantic_hunger", "indices": [0]}]}
    if source is not None:
        meta.update(matched_original_hunger=True, source_input_path=str(source), source_input_sha256=file_digest(source))
    return {"X": X, "y": y, "time_ms": times, "hunger": exact[9::10],
        "intake": np.linspace(0, 4 if ep["expected_release"] else .8 if ep["scene"] == "limited_meal" else 0, len(times)),
        "taste": np.zeros(len(times)), "action": np.full(len(times), "WALK"),
        "replay_need_float64": exact, "replay_luminance": np.zeros((len(exact), 2)),
        "replay_values_0": (exact * 5).astype(np.float32)[:, None], "metadata": meta}


def save_row(output, row):
    meta = row["metadata"]
    path = output / "episodes" / meta["condition"] / (meta["episode"]["episode_id"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{k: v for k, v in row.items() if k != "metadata"},
            metadata=np.asarray(json.dumps(meta)))
    return path


def prepared_fixture(tmp_path, monkeypatch):
    source = tmp_path / "old-collector.py"
    source.write_text("source", encoding="utf-8")
    parent, replay, root = tmp_path / "parent", tmp_path / "replay", tmp_path / "refit"
    study = old_plan(source)
    parent.mkdir(); replay.mkdir()
    for folder in (parent, replay):
        (folder / "plan.json").write_text(json.dumps(study), encoding="utf-8")
    for ep in study["episodes"]:
        if ep["split"] == "test": save_row(parent, row_fixture(study, ep))
        else: save_row(replay, row_fixture(study, ep, body=False))
    X = np.zeros((30, 1024)); y = np.tile([0, 1], 15); X[:, 0] = y * 30
    baseline = LinearReadout.fit(X, y, feature_digest=study["feature_digest"], l2=.01)
    baseline.save(parent / "state-readout.npz")
    selected = {"model_sha256": file_digest(parent / "state-readout.npz")}
    for name, value in (("results.json", {"status": "TARGET_NOT_REACHED"}),
                         ("selection.json", selected), ("test-binding.json", {"frozen": True})):
        (parent / name).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(refit, "_require_finished_parent", lambda path: (study, selected))
    prepared = refit.prepare(root, parent=parent, replay_source=replay)
    return root, parent, replay, prepared


def test_fresh_test_generation_is_deterministic_varied_and_leaves_parent_unchanged(tmp_path):
    source = tmp_path / "source.py"; source.write_text("fixture")
    study = old_plan(source); before = deepcopy(study)
    episodes = refit.fresh_test_episodes(study)
    assert episodes == refit.fresh_test_episodes(study)
    assert study == before
    assert len(episodes) == 18
    assert {ep["seed_group"] for ep in episodes} == {5201, 5202, 5203}
    old_seeds = {ep["seed"] for ep in study["episodes"]}
    assert not old_seeds & {ep["seed"] for ep in episodes}
    limited = [ep for ep in episodes if ep["scene"] == "limited_meal"]
    assert [ep["config"]["food"][0]["amount"] for ep in limited] == [.6, 1.2, 1.6]
    assert len({ep["config"]["initial"]["energy"] for ep in episodes}) == 18
    assert len({ep["config"]["environment"]["light"] for ep in episodes}) == 18


def test_no_refit_preparation_before_parent_result_exists(tmp_path):
    root = tmp_path / "refit"
    with pytest.raises(FileNotFoundError, match="Finish and record"):
        refit.prepare(root, parent=tmp_path / "unfinished", replay_source=tmp_path / "replay")
    assert not root.exists()


def test_source_data_reassigned_only_in_memory_and_all_origins_retained(tmp_path, monkeypatch):
    root, _, _, study = prepared_fixture(tmp_path, monkeypatch)
    before = {item["path"]: file_digest(item["path"]) for item in study["lineage"]}
    rows = refit.load_development(refit.load_plan(root))
    assert {k: len(v) for k, v in rows.items()} == {"physical_train": 12, "physical_validation": 6, "replay_train": 24}
    for role, selected in rows.items():
        for row in selected:
            meta = row["metadata"]
            assert meta["episode"]["split"] == ("validation" if role == "physical_validation" else "train")
            if role.startswith("physical"):
                assert meta["original_metadata"]["episode"]["split"] == "test"
            else: assert meta["original_metadata"]["episode"]["split"] in ("train", "validation")
            with np.load(row["path"], allow_pickle=False) as archive:
                for key in ("X", "y", "intake", "hunger", "time_ms"):
                    assert np.array_equal(row[key], archive[key])
    assert before == {path: file_digest(path) for path in before}


def test_six_candidate_refit_then_new_test_and_same_gain_baseline(tmp_path, monkeypatch):
    root, _, _, study = prepared_fixture(tmp_path, monkeypatch)
    selected = refit.train(root)
    assert selected["candidate_count"] == 6
    candidates = v3._read_sealed(root / "training.json")["candidates"]
    assert [c["unique_train_episodes"] for c in candidates] == [12] * 3 + [36] * 3
    assert all(c["unique_validation_episodes"] == 6 for c in candidates)
    assert all(c["fresh_test_seen"] is False for c in candidates)
    assert refit.verify_test_binding(root)[2]["fresh_test_seen"] is False
    for ep in study["episodes"]:
        if ep["split"] != "test": continue
        path = save_row(root, row_fixture(study, ep))
        for condition in ("homeostasis_off", "transmission_off"):
            save_row(root, row_fixture(study, ep, body=False, condition=condition, source=path))
    result = refit.evaluate(root)
    assert result["status"] == "PRELIMINARY_TARGET_REACHED"
    assert result["new_test_groups"] == [5201, 5202, 5203]
    assert result["conditions"]["connected"]["episodes"] == 18
    assert result["baseline_conditions"]["connected"]["episodes"] == 18
    assert result["conditions"]["transmission_off"]["indicator"]["f1"] == 0
    assert result["reused_v3a_test_is_development"] is True
    assert result["test_tuned"] is False
    with pytest.raises(FileExistsError): refit.evaluate(root)
    with pytest.raises(FileExistsError): refit.train(root)


@pytest.mark.parametrize("change", ["duplicate", "physical_role", "replay_role", "old_test", "criteria"])
def test_lineage_or_final_protocol_cannot_be_silently_redefined(tmp_path, monkeypatch, change):
    _, _, _, study = prepared_fixture(tmp_path, monkeypatch)
    if change == "duplicate": study["lineage"][1]["path"] = study["lineage"][0]["path"]
    elif change == "physical_role": study["lineage"][0]["assigned_split"] = "validation"
    elif change == "replay_role": study["lineage"][-1]["assigned_split"] = "validation"
    elif change == "old_test":
        next(ep for ep in study["episodes"] if ep["split"] == "test")["seed_group"] = 4201
    else: study["acceptance"]["minimum_limited_case_recall"] = .2
    study = _seal({k: v for k, v in study.items() if k != "digest"})
    with pytest.raises(ValueError): refit.validate_plan(study)


@pytest.mark.parametrize("change", ["parent_result", "input_data", "origin_plan", "new_plan", "model"])
def test_frozen_sources_fail_before_opening_any_new_test(tmp_path, monkeypatch, change):
    root, parent, replay, study = prepared_fixture(tmp_path, monkeypatch)
    refit.train(root)
    if change == "parent_result": path = parent / "results.json"
    elif change == "input_data": path = Path(study["lineage"][0]["path"])
    elif change == "origin_plan": path = replay / "plan.json"
    elif change == "model": path = root / "state-readout.npz"
    else:
        study["extra_after_binding"] = True
        study = _seal({k: v for k, v in study.items() if k != "digest"})
        (root / "plan.json").write_text(json.dumps(study), encoding="utf-8")
        path = None
    if path:
        with path.open("ab") as stream: stream.write(b"changed")
    def forbid_test(*args, **kwargs): raise AssertionError("New test opened before validating frozen bindings")
    monkeypatch.setattr(refit.np, "load", forbid_test)
    with pytest.raises(ValueError): refit.evaluate(root)


@pytest.mark.parametrize("change", ["need_missing", "need_precision", "need_labels", "visual", "current_shape", "channel_duplicate", "episode", "body"])
def test_replay_requires_exact_aligned_recorded_physical_inputs(tmp_path, monkeypatch, change):
    root, _, _, study = prepared_fixture(tmp_path, monkeypatch)
    ep = next(ep for ep in study["episodes"] if ep["split"] == "test" and ep["scene"] == "meal_fast")
    row = row_fixture(study, ep)
    if change == "need_missing": row.pop("replay_need_float64")
    elif change == "need_precision": row["replay_need_float64"] = row["replay_need_float64"].astype(np.float32)
    elif change == "need_labels": row["replay_need_float64"][9] = .2
    elif change == "visual": row["replay_luminance"] = row["replay_luminance"][:-1]
    elif change == "current_shape": row["replay_values_0"] = row["replay_values_0"][:-1]
    elif change == "channel_duplicate": row["metadata"]["stimulus_channels"] *= 2
    elif change == "episode": row["metadata"]["episode"] = dict(ep, seed=-1)
    else: row["metadata"]["body_physics"] = False
    path = save_row(root, row)
    with pytest.raises(ValueError): refit._replay_source(path, study, ep)


def test_replay_source_validates_unmodified_exact_recording(tmp_path, monkeypatch):
    root, _, _, study = prepared_fixture(tmp_path, monkeypatch)
    ep = next(ep for ep in study["episodes"] if ep["split"] == "test" and ep["scene"] == "meal_fast")
    path = save_row(root, row_fixture(study, ep))
    meta, arrays = refit._replay_source(path, study, ep)
    assert meta["episode"] == ep
    assert np.array_equal(arrays["hunger"], arrays["replay_need_float64"][9::10])


def test_direct_collector_or_replay_cannot_replace_frozen_episode(monkeypatch):
    ep = {"episode_id": "test-5201-hungry_away", "split": "test", "seed_group": 5201, "seed": 52010}
    study = {"episodes": [ep]}
    monkeypatch.setattr(refit, "verify_test_binding", lambda root: (study, {}, {}))
    refit._validate_test_operation(Path("fixture"), study, ep)
    with pytest.raises(ValueError, match="episode differs"):
        refit._validate_test_operation(Path("fixture"), study, dict(ep, seed=1))
    with pytest.raises(ValueError, match="study differs"):
        refit._validate_test_operation(Path("fixture"), study | {"changed": True}, ep)
    with pytest.raises(ValueError, match="predeclared fresh"):
        refit._test_episodes(study, 4201)
