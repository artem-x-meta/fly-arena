"""One bounded live-data refit after v3a, with an entirely new physical test.

Former v3a test episodes are explicitly reused as development here. The original
v3a result, model, code and data remain frozen. This helper never rewrites them.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fly_semantic.mapping import digest, file_digest
from fly_semantic.readout import LinearReadout
from semantic_tools import physical_learning as physical
from semantic_tools import v3_learning as v3


ROOT = Path("runs/semantic-v3/refit")
PARENT = Path("runs/semantic-v3/selected")
REPLAY_SOURCE = Path("runs/semantic-v3/development/gain-5")
REGIMES = ("actual_only", "mixed_replay")
TRAIN_GROUPS = (4201, 4202)
VALIDATION_GROUPS = (4203,)
TEST_GROUPS = (5201, 5202, 5203)
SCENES = ("hungry_away", "sated_away", "meal_fast", "meal_slow", "limited_meal", "sated_food")


def source_hashes():
    from semantic_tools.v3_workflow import workflow_sources
    return {**v3.source_hashes(), **workflow_sources(),
        "semantic_tools/v3_refit.py": file_digest(Path(__file__))}


def protocol_fields():
    stable = {k: deepcopy(v) for k, v in v3.protocol_fields().items()
        if not k.startswith("v3_")}
    return {**stable, "refit_training_regimes": list(REGIMES),
        "refit_selection_rule": v3.SELECTION_RULE, "candidate_count": 6,
        "feature_mask": "all", "hunger_gain_mv": 5.,
        "refit_physical_train_groups": list(TRAIN_GROUPS),
        "refit_reused_validation_groups": list(VALIDATION_GROUPS),
        "refit_fresh_test_groups": list(TEST_GROUPS),
        "validation_status": "Former v3a test, now explicitly development; not a fresh independent confirmatory set",
        "mixed_replay_policy": "Append all 24 old gain-5 development episodes exactly once to TRAIN; no duplication or sample weighting",
        "fresh_test_rng_offset": 610000, "limited_test_portions": [.6, 1.2, 1.6]}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_digest(path)}


def fresh_test_episodes(parent_plan):
    """Same six families and parameter bounds; new body and variation RNG seeds."""
    templates = {ep["scene"]: ep for ep in parent_plan["episodes"]
        if ep["split"] == "test" and ep["seed_group"] == 4201}
    if set(templates) != set(SCENES):
        raise ValueError("Parent must supply all six frozen scene templates")
    result = []
    for group, portion in zip(TEST_GROUPS, (.6, 1.2, 1.6)):
        for index, scene in enumerate(SCENES):
            ep = deepcopy(templates[scene])
            seed = group * 10 + index
            rng = np.random.default_rng(seed + 610000)
            cfg = ep["config"]
            cfg["initial"]["energy"] = float(rng.uniform(75, 85) if scene.startswith("sated") else rng.uniform(10, 20))
            cfg["initial"]["sleep_pressure"] = float(rng.uniform(.02, .15))
            cfg["environment"]["light"] = float(rng.uniform(.65, 1.))
            cfg["organism"]["intake_rate"] = float(rng.uniform(1.25, 1.5) if scene == "meal_slow" else rng.uniform(2.6, 3.))
            if scene.endswith("away"):
                cfg["food"] = []
            else:
                cfg["food"][0].update(x=float(rng.uniform(1.52, 1.68)), y=float(rng.uniform(-.06, .06)),
                    radius=float(rng.uniform(1.05, 1.17)), amount=portion if scene == "limited_meal" else 5.,
                    taste=float(rng.uniform(.75, 1.)), odor=float(rng.uniform(.6, 1.)))
            ep.update(episode_id=f"test-{group}-{scene}", seed_group=group, seed=seed, split="test")
            result.append(ep)
    return result


def _development_episode(original, role, origin):
    ep = deepcopy(original)
    ep.update(episode_id=f"refit-{role}-{origin}-{original['episode_id']}", split=role)
    return ep


def _require_finished_parent(parent):
    if not (parent / "results.json").exists():
        raise FileNotFoundError("Finish and record v3a results.json before preparing or running a refit")
    v3.verify_test_binding(parent)
    result = v3._read_sealed(parent / "results.json")
    study = v3._read_sealed(parent / "plan.json")
    selection = v3._read_sealed(parent / "selection.json")
    if (result["plan_digest"] != study["digest"] or result["model_sha256"] != selection["model_sha256"]
            or study["hunger_gain_mv"] != 5.):
        raise ValueError("Finished parent result/model/gain differs from the declared refit source")
    return study, selection


def prepare(root=ROOT, *, parent=PARENT, replay_source=REPLAY_SOURCE):
    """Freeze six candidates and every data origin before fitting any head."""
    root, parent, replay_source = map(Path, (root, parent, replay_source))
    if root.exists():
        raise FileExistsError("Use a fresh refit root; do not overwrite an existing plan or run")
    parent_plan, parent_selection = _require_finished_parent(parent)
    replay_plan = v3._read_sealed(replay_source / "plan.json")
    if any(parent_plan[k] != replay_plan[k] for k in ("feature_digest", "calibration_digest", "nominal_weights_sha256")):
        raise ValueError("Mixed replay changes gain, features, or graph")
    physical_rows = physical.load_rows(parent, parent_plan, "test")
    replay_rows = physical.load_rows(replay_source, replay_plan, "train") + physical.load_rows(replay_source, replay_plan, "validation")
    if len(physical_rows) != 18 or len(replay_rows) != 24:
        raise ValueError("Refit requires all 18 physical and 24 replay development episodes")
    episodes, lineage = [], []
    for origin, rows, source_plan_path in (("physical", physical_rows, parent / "plan.json"),
                                           ("replay", replay_rows, replay_source / "plan.json")):
        for row in rows:
            old_ep = row["metadata"]["episode"]
            if origin == "physical":
                group = old_ep["seed_group"]
                if group not in TRAIN_GROUPS + VALIDATION_GROUPS or row["metadata"].get("body_physics") is not True:
                    raise ValueError("Unexpected parent physical episode or group")
                role = "train" if group in TRAIN_GROUPS else "validation"
            else:
                if old_ep["split"] not in ("train", "validation") or row["metadata"].get("body_physics") is not False:
                    raise ValueError("Mixed source must remain declared neural replay")
                role = "train"
            ep = _development_episode(old_ep, role, origin)
            episodes.append(ep)
            lineage.append({"episode_id": ep["episode_id"], "origin": origin, "assigned_split": role,
                "original_episode": old_ep, "original_split": old_ep["split"],
                "source_plan_path": str(source_plan_path), "source_plan_sha256": file_digest(source_plan_path),
                "path": row["path"], "sha256": row["sha256"]})
    episodes.extend(fresh_test_episodes(parent_plan))
    protected = [_artifact(parent / name) for name in ("results.json", "plan.json", "selection.json", "test-binding.json", "state-readout.npz")]
    protected += [_artifact(replay_source / "plan.json")]
    protected += v3._calibration_files(parent_plan)
    stable_keys = ("calibration_dir", "mapping_digest", "calibration_digest", "feature_digest",
        "nominal_weights_sha256", "brain_config", "max_iter")
    study = {key: deepcopy(parent_plan[key]) for key in stable_keys if key in parent_plan}
    study.update(protocol_fields())
    study.update(schema_version=2, scope="Bounded live-gain-5 refit; v3a test explicitly reassigned to development; new 5201-5203 physical test",
        source_hashes=source_hashes(), episodes=episodes, lineage=lineage,
        frozen_parent_artifacts=protected, parent_path=str(parent), replay_source_path=str(replay_source),
        parent_results_sha256=file_digest(parent / "results.json"),
        baseline_model_path=str(parent / "state-readout.npz"), baseline_model_sha256=parent_selection["model_sha256"],
        collection_conditions=["connected", "homeostasis_off", "transmission_off"],
        seed_groups={"train": sorted(set(TRAIN_GROUPS + (1201, 1202, 1203, 2201))),
            "validation": list(VALIDATION_GROUPS), "test": list(TEST_GROUPS)},
        expected_counts={"physical_train_episodes": 12, "physical_reused_validation_episodes": 6,
            "mixed_replay_train_episodes": 24, "fresh_physical_test_episodes": 18},
        source_roles="No source arrays rewritten. Old test/validation labels are development for this new plan only.")
    study = v3._sealed(study)
    validate_plan(study)
    root.mkdir(parents=True, exist_ok=False)
    v3._write_new(root / "plan.json", study)
    return study


def validate_plan(study):
    physical.validate_plan(study)
    for key, value in protocol_fields().items():
        if study.get(key) != value:
            raise ValueError(f"Frozen refit protocol differs: {key}")
    if len(study["lineage"]) != 42 or len({r["path"] for r in study["lineage"]}) != 42:
        raise ValueError("Development lineage must contain 42 unique source episodes")
    if len({r["episode_id"] for r in study["lineage"]}) != 42:
        raise ValueError("Duplicate assigned development episode")
    counts = {(origin, split): sum(r["origin"] == origin and r["assigned_split"] == split for r in study["lineage"])
        for origin, split in (("physical", "train"), ("physical", "validation"), ("replay", "train"))}
    if counts != {("physical", "train"): 12, ("physical", "validation"): 6, ("replay", "train"): 24}:
        raise ValueError("Refit development counts or roles changed")
    for item in study["lineage"]:
        original = item["original_episode"]
        if item["origin"] == "physical":
            expected = "train" if original["seed_group"] in TRAIN_GROUPS else "validation" if original["seed_group"] in VALIDATION_GROUPS else None
            if original["split"] != "test" or expected != item["assigned_split"]:
                raise ValueError("Former physical test role reassignment changed")
        elif item["origin"] != "replay" or item["assigned_split"] != "train" or original["split"] not in ("train", "validation"):
            raise ValueError("Replay may only supplement TRAIN")
    test = [ep for ep in study["episodes"] if ep["split"] == "test"]
    if len(test) != 18 or {ep["seed_group"] for ep in test} != set(TEST_GROUPS):
        raise ValueError("Fresh test requires exactly three new six-scene seed groups")


def load_plan(root=ROOT):
    study = v3._read_sealed(Path(root) / "plan.json")
    validate_plan(study)
    if study["source_hashes"] != source_hashes():
        raise ValueError("Refit source changed after plan freeze")
    v3._verify_sources(study["source_hashes"])
    v3._verify_data(study["frozen_parent_artifacts"])
    v3._verify_data(study["lineage"])
    for item in study["lineage"]:
        if file_digest(item["source_plan_path"]) != item["source_plan_sha256"]:
            raise ValueError("Development origin plan changed")
    return study


def load_development(study):
    """Validate original files under original plans, then relabel roles in memory."""
    cache = {}
    for item in study["lineage"]:
        key = (item["source_plan_path"], item["original_split"])
        if key not in cache:
            plan_path = Path(key[0])
            original_plan = v3._read_sealed(plan_path)
            rows = physical.load_rows(plan_path.parent, original_plan, key[1])
            cache[key] = {row["metadata"]["episode"]["episode_id"]: row for row in rows}
    episodes = {ep["episode_id"]: ep for ep in study["episodes"]}
    result = {"physical_train": [], "physical_validation": [], "replay_train": []}
    for item in study["lineage"]:
        original = cache[(item["source_plan_path"], item["original_split"])][item["original_episode"]["episode_id"]]
        if (original["sha256"] != item["sha256"] or original["metadata"]["episode"] != item["original_episode"]
                or Path(original["path"]).resolve() != Path(item["path"]).resolve()
                or any(original["metadata"][k] != study[k] for k in ("feature_digest", "calibration_digest", "nominal_weights_sha256"))):
            raise ValueError("Development arrays or original provenance changed")
        # Only scoring metadata changes; X/y/intake/hunger/time arrays are retained.
        row = dict(original)
        row["metadata"] = dict(original["metadata"], episode=episodes[item["episode_id"]],
            refit_lineage=deepcopy(item), original_metadata=deepcopy(original["metadata"]))
        role_key = f"{item['origin']}_{item['assigned_split']}"
        result[role_key].append(row)
    return result


def train(root=ROOT):
    root = Path(root)
    if (root / "candidates").exists() or (root / "selection.json").exists() or (root / "state-readout.npz").exists():
        raise FileExistsError("Refit candidates already exist; no further tuning in this round")
    study = load_plan(root)
    v3._assert_test_absent(root, study)
    rows = load_development(study)
    validation = rows["physical_validation"]
    destination = root / "candidates"
    destination.mkdir()
    candidates = []
    for regime in REGIMES:
        training = rows["physical_train"] + (rows["replay_train"] if regime == "mixed_replay" else [])
        physical._consistent(training + validation)
        X, y = np.concatenate([r["X"] for r in training]), np.concatenate([r["y"] for r in training])
        if X.shape[1] != 1024:
            raise ValueError("Refit must keep the full 1024-feature layout")
        for l2 in v3.L2_GRID:
            index = len(candidates)
            name = f"{index:02d}-{regime}-l2-{l2:g}"
            folder = destination / name
            folder.mkdir()
            model = LinearReadout.fit(X, y, feature_digest=study["feature_digest"],
                l2=l2, max_iter=study.get("max_iter", 500), seed=42)
            if not model.diagnostics["converged"]:
                raise RuntimeError(f"Refit optimizer failed: {name}")
            model.save(folder / "state-readout.npz")
            scored = physical.score_rows(model, validation, study)
            candidate = v3._sealed({"candidate_index": index, "name": name, "regime": regime,
                "l2": l2, "model_path": str(folder / "state-readout.npz"),
                "model_sha256": file_digest(folder / "state-readout.npz"),
                "train_rows": len(X), "unique_train_episodes": len(training),
                "physical_train_episodes": len(rows["physical_train"]),
                "replay_train_episodes": len(rows["replay_train"]) if regime == "mixed_replay" else 0,
                "unique_validation_episodes": len(validation), "train_positive_fraction": float(y.mean()),
                "train_data": physical._data_manifest(training), "validation_data": physical._data_manifest(validation),
                "nonzero_weight_columns": np.flatnonzero(np.any(model.weights != 0, axis=1)).tolist(),
                "optimizer": model.diagnostics, "validation": scored, "assessment": v3.assess_acceptance(scored, study),
                "validation_status": study["validation_status"], "fresh_test_seen": False})
            v3._write_new(folder / "validation.json", candidate)
            candidates.append(candidate)
    chosen = max(candidates, key=v3.selection_key)
    with (root / "state-readout.npz").open("xb") as stream:
        stream.write(Path(chosen["model_path"]).read_bytes())
    selection = v3._sealed({"schema_version": 1, "plan_digest": study["digest"],
        "plan_sha256": file_digest(root / "plan.json"), "source_hashes": source_hashes(),
        "selection_rule": v3.SELECTION_RULE, "candidate_count": 6, "candidate_name": chosen["name"],
        "regime": chosen["regime"], "selected_l2": chosen["l2"], "model_sha256": chosen["model_sha256"],
        "feature_digest": study["feature_digest"], "calibration_digest": study["calibration_digest"],
        "nominal_weights_sha256": study["nominal_weights_sha256"], "gain": 5., "mask": "all",
        "assessment": chosen["assessment"], "validation": chosen["validation"],
        "validation_status": study["validation_status"], "train_positive_fraction": chosen["train_positive_fraction"],
        "nonzero_weight_columns": chosen["nonzero_weight_columns"],
        "candidate_artifacts": [{**_artifact(destination / c["name"] / "validation.json"),
            "model_path": c["model_path"], "model_sha256": c["model_sha256"]} for c in candidates],
        "fresh_test_seen": False, "all_candidates_reported": True})
    v3._write_new(root / "selection.json", selection)
    v3._write_new(root / "training.json", v3._sealed({"schema_version": 1,
        "plan_digest": study["digest"], "selection_digest": selection["digest"], "candidates": candidates}))
    absent = v3._assert_test_absent(root, study)
    binding = v3._sealed({"schema_version": 1, "plan_digest": study["digest"],
        "plan_sha256": file_digest(root / "plan.json"), "selection_digest": selection["digest"],
        "selection_sha256": file_digest(root / "selection.json"),
        "model_sha256": selection["model_sha256"], "source_hashes": source_hashes(),
        "future_test_outputs_absent": absent, "fresh_test_seen": False})
    v3._write_new(root / "test-binding.json", binding)
    return selection


def verify_test_binding(root=ROOT):
    root = Path(root)
    study = load_plan(root)
    selection = v3._read_sealed(root / "selection.json")
    binding = v3._read_sealed(root / "test-binding.json")
    if (binding["plan_sha256"] != file_digest(root / "plan.json")
            or binding["selection_sha256"] != file_digest(root / "selection.json")
            or binding["model_sha256"] != file_digest(root / "state-readout.npz")
            or binding["model_sha256"] != selection["model_sha256"]
            or binding["source_hashes"] != source_hashes() or selection["source_hashes"] != source_hashes()
            or selection["plan_digest"] != study["digest"] or selection["candidate_count"] != 6
            or selection["fresh_test_seen"] is not False or binding["fresh_test_seen"] is not False):
        raise ValueError("Frozen refit selection or test binding changed")
    for item in selection["candidate_artifacts"]:
        v3._verify_data([item])
        if file_digest(item["model_path"]) != item["model_sha256"]:
            raise ValueError("Refit candidate model changed")
    return study, selection, binding


def _test_episodes(study, group=None):
    if group is not None and group not in TEST_GROUPS:
        raise ValueError("Only the predeclared fresh test groups can be collected")
    return [ep for ep in study["episodes"] if ep["split"] == "test" and (group is None or ep["seed_group"] == group)]


def _validate_test_operation(root, study, ep):
    bound_study, _, _ = verify_test_binding(root)
    if study != bound_study:
        raise ValueError("Collection/replay study differs from the frozen plan")
    planned = {item["episode_id"]: item for item in _test_episodes(study)}
    if planned.get(ep["episode_id"]) != ep:
        raise ValueError("Collection/replay episode differs from the frozen fresh test")


def _collect_episode(root, study, ep):
    from fly_semantic.runtime import SemanticSimulation
    from semantic_tools.v3_workflow import NeedRecorder
    target = root / "episodes/connected" / (ep["episode_id"] + ".npz")
    if target.exists():
        raise FileExistsError(target)
    _validate_test_operation(root, study, ep)
    cal_dir = Path(study["calibration_dir"])
    sim = SemanticSimulation(ep["config"], graph=Path("data/graph"), seed=ep["seed"], enabled=True,
        mapping_path=cal_dir / "mapping.json", calibration_path=cal_dir / "calibration.json",
        readout_path=root / "state-readout.npz", episode_id=ep["episode_id"])
    recorder = NeedRecorder(sim.legacy.brain, sim.channel, sim.organism)
    sim.legacy.brain = recorder
    started = time.perf_counter()
    X, y, times, needs, intake, taste, actions, scores, events, transitions = ([] for _ in range(10))
    active = False
    try:
        nominal = hashlib.sha256(memoryview(sim.brain.weights)).hexdigest()
        if (nominal != study["nominal_weights_sha256"] or sim.channel.features.feature_digest != study["feature_digest"]
                or sim.channel.calibration["digest"] != study["calibration_digest"]):
            raise ValueError("Fresh refit runtime graph/calibration/features differ from frozen plan")
        for tick in range(ep["duration_ms"] // 10):
            need = sim.organism.hunger
            previous = active
            active = bool(need >= study["teacher_on"] or (active and need > study["teacher_off"]))
            if active != previous:
                transitions.append({"time_ms": sim.brain.clock, "active": active, "hunger": need})
            sim.step()
            events.extend(e.to_dict() for e in sim.channel.last_events)
            if (tick + 1) * 10 % study["sample_ms"] == 0:
                X.append(sim.channel.features.values()); y.append(active); times.append(sim.brain.clock); needs.append(need)
                intake.append(sim.organism.state.ingested_total)
                taste.append(max([sim.last_frame.mouth_taste, *sim.last_frame.tarsal_taste.values()]))
                actions.append(sim.last_decision.action); scores.append(sim.channel.last_scores[1])
        meta = {"episode": ep, "condition": "connected", "plan_digest": study["digest"],
            "feature_digest": sim.channel.features.feature_digest, "calibration_digest": sim.channel.calibration["digest"],
            "nominal_weights_sha256": nominal, "target_transitions": transitions, "source_hashes": source_hashes(),
            "body_physics": True, "refit_scope": "New free physical trajectory; former test data are development only",
            "resource_balance": sim.resource_balances(), "minimum_upright": sim.minimum_upright,
            "ingested_total": sim.organism.state.ingested_total, "final_hunger": sim.organism.hunger,
            "stimulus_channels": recorder.channel_specs, "live_events": events, "wall_seconds": time.perf_counter() - started}
        arrays = {"X": np.array(X), "y": np.array(y, np.int8), "time_ms": np.array(times, np.int64),
            "hunger": np.array(needs), "intake": np.array(intake), "taste": np.array(taste), "action": np.array(actions),
            "live_scores": np.array(scores), "replay_luminance": np.array(recorder.luminance),
            "replay_need_float64": np.array(recorder.exact_need, np.float64),
            "diagnostic_input_rates": np.array(recorder.input_rates), "metadata": np.array(json.dumps(meta, allow_nan=False))}
        arrays.update({f"replay_values_{i}": np.array(values) for i, values in enumerate(recorder.channel_values)})
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        print(json.dumps({"episode": ep["episode_id"], "intake": meta["ingested_total"],
            "hunger": meta["final_hunger"], "seconds": round(meta["wall_seconds"], 2)}), flush=True)
        return meta
    finally:
        sim.close()


def collect(group=None, *, root=ROOT):
    root = Path(root)
    study, _, _ = verify_test_binding(root)
    return [_collect_episode(root, study, ep) for ep in _test_episodes(study, group)]


def _replay_source(source, study, ep):
    """Read and validate exact recorded native inputs; no hidden-state fallback."""
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        meta = json.loads(str(archive["metadata"].item()))
    if (meta["episode"] != ep or meta.get("body_physics") is not True or meta["condition"] != "connected"
            or meta["plan_digest"] != study["digest"] or meta["source_hashes"] != study["source_hashes"]
            or any(meta[k] != study[k] for k in ("feature_digest", "calibration_digest", "nominal_weights_sha256"))):
        raise ValueError("Replay source is not the matching frozen connected physical episode")
    ticks = ep["duration_ms"] // 10
    sample_ticks = study["sample_ms"] // 10
    exact = arrays.get("replay_need_float64")
    if (exact is None or exact.dtype != np.float64 or exact.shape != (ticks,)
            or not np.isfinite(exact).all() or np.any((exact < 0) | (exact > 1))):
        raise ValueError("Replay requires exact finite float64 hunger for every neural tick")
    visual = arrays["replay_luminance"]
    if visual.ndim < 2 or len(visual) != ticks or not np.isfinite(visual).all():
        raise ValueError("Replay visual input shape or values are invalid")
    specs = meta["stimulus_channels"]
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names) or names.count("semantic_hunger") != 1:
        raise ValueError("Replay needs unique channels and exactly one hunger input")
    for index, spec in enumerate(specs):
        indices = np.asarray(spec["indices"])
        values = arrays[f"replay_values_{index}"]
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                or values.shape != (ticks, len(indices)) or not np.isfinite(values).all()):
            raise ValueError("Replay current addresses or recorded values are invalid")
    expected_times = np.arange(study["sample_ms"], ep["duration_ms"] + 1, study["sample_ms"])
    if not np.array_equal(arrays["time_ms"], expected_times):
        raise ValueError("Replay sample timing differs from the frozen grid")
    active, labels = False, []
    for tick, need in enumerate(exact):
        active = bool(need >= study["teacher_on"] or (active and need > study["teacher_off"]))
        if (tick + 1) % sample_ticks == 0: labels.append(active)
    if (not np.array_equal(arrays["hunger"], exact[sample_ticks - 1::sample_ticks])
            or not np.array_equal(arrays["y"], labels)):
        raise ValueError("Replay hunger and teacher are not aligned to the same neural ticks")
    return meta, arrays


def _replay_episode(root, study, ep, condition):
    from fly_arena.brain import Brain, BrainConfig
    from fly_semantic.mapping import load_mapping
    from fly_semantic.runtime import load_calibration, make_features
    if condition not in ("homeostasis_off", "transmission_off", "replay_check"):
        raise ValueError("Unknown frozen replay condition")
    _validate_test_operation(root, study, ep)
    target = root / "episodes" / condition / (ep["episode_id"] + ".npz")
    if target.exists(): raise FileExistsError(target)
    source = root / "episodes/connected" / (ep["episode_id"] + ".npz")
    meta, arrays = _replay_source(source, study, ep)
    cal_dir = Path(study["calibration_dir"])
    mapping = load_mapping(cal_dir / "mapping.json", Path("data/graph"))
    calibration = load_calibration(cal_dir / "calibration.json", mapping)
    brain = Brain(Path("data/graph"), BrainConfig(**study["brain_config"]), seed=ep["seed"])
    nominal = hashlib.sha256(memoryview(brain.weights)).hexdigest()
    features = make_features(mapping, calibration)
    if nominal != study["nominal_weights_sha256"] or features.feature_digest != study["feature_digest"]:
        raise ValueError("Replay graph or features differ from frozen refit")
    if condition == "transmission_off": brain.weights = np.zeros_like(brain.weights)
    started = time.perf_counter()
    X, input_rates = [], []
    for tick, visual in enumerate(arrays["replay_luminance"]):
        channels = {}
        for index, spec in enumerate(meta["stimulus_channels"]):
            if condition == "homeostasis_off" and spec["name"] == "semantic_hunger": continue
            current = (study["hunger_gain_mv"] * arrays["replay_need_float64"][tick]
                if spec["name"] == "semantic_hunger" else arrays[f"replay_values_{index}"][tick])
            channels[spec["name"]] = (np.asarray(spec["indices"], np.int32), current)
        _, counts = brain.step(visual, sensory_currents=channels)
        z = features.update(counts, 10)
        input_rates.append(counts[[r["index"] for r in mapping["ports"]["hunger"]]] * 100.)
        if (tick + 1) * 10 % study["sample_ms"] == 0: X.append(z)
    X = np.asarray(X)
    if condition == "replay_check": np.testing.assert_array_equal(X, arrays["X"])
    updated = {**meta, "condition": condition, "body_physics": False,
        "matched_original_hunger": True, "source_input_path": str(source), "source_input_sha256": file_digest(source),
        "replay_scope": "Recorded physical trajectory held fixed; this is a neural intervention, not free-body behavior",
        "wall_seconds": time.perf_counter() - started}
    output = {k: arrays[k] for k in ("y", "time_ms", "hunger", "intake", "taste", "action")}
    output.update(X=X, diagnostic_input_rates=np.asarray(input_rates), metadata=np.asarray(json.dumps(updated, allow_nan=False)))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream: np.savez_compressed(stream, **output)
    print(json.dumps({"episode": ep["episode_id"], "condition": condition,
        "seconds": round(updated["wall_seconds"], 2)}), flush=True)
    return updated


def replay(condition, group=None, *, root=ROOT):
    root = Path(root)
    study, _, _ = verify_test_binding(root)
    return [_replay_episode(root, study, ep, condition) for ep in _test_episodes(study, group)]


def evaluate(root=ROOT):
    root = Path(root)
    if (root / "results.json").exists(): raise FileExistsError("Fresh refit test already evaluated")
    study, selection, binding = verify_test_binding(root)
    recorded = {condition: physical.load_rows(root, study, "test", condition) for condition in study["collection_conditions"]}
    physical._consistent([row for rows in recorded.values() for row in rows])
    for condition, rows in recorded.items():
        for row in rows:
            meta = row["metadata"]
            if condition == "connected":
                if meta.get("body_physics") is not True: raise ValueError("Acceptance requires fresh physical episodes")
            else:
                connected = root / "episodes/connected" / (meta["episode"]["episode_id"] + ".npz")
                if (meta.get("body_physics") is not False or meta.get("matched_original_hunger") is not True
                        or Path(meta["source_input_path"]).resolve() != connected.resolve()
                        or meta["source_input_sha256"] != file_digest(connected)):
                    raise ValueError("Control does not retain its matched physical source")
    model = LinearReadout.load(root / "state-readout.npz", expected_feature_digest=study["feature_digest"])
    baseline = LinearReadout.load(study["baseline_model_path"], expected_feature_digest=study["feature_digest"])
    shuffled, permutation = physical._shuffle_rows(recorded["connected"], study.get("shuffle_seed", 81817))
    conditions = recorded | {"cross_episode_shuffle": shuffled}
    scores = {key: physical.score_rows(model, rows, study) for key, rows in conditions.items()}
    old_scores = {key: physical.score_rows(baseline, rows, study) for key, rows in conditions.items()}
    constant = physical._ConstantHead(model.feature_digest, float(selection["train_positive_fraction"] >= .5))
    scores["train_constant"] = physical.score_rows(constant, recorded["connected"], study)
    result = v3._sealed({"schema_version": 1, **v3.assess_acceptance(scores["connected"], study),
        "plan_digest": study["digest"], "selection_digest": selection["digest"], "binding_digest": binding["digest"],
        "model_sha256": selection["model_sha256"], "baseline_model_sha256": study["baseline_model_sha256"],
        "gain": 5., "mask": "all", "regime": selection["regime"], "selected_l2": selection["selected_l2"],
        "conditions": scores, "baseline_conditions": old_scores,
        "paired_refit_minus_v3a": {key: physical.paired_seed_bootstrap(scores[key], old_scores[key]) for key in conditions},
        "paired_connected_minus_control": {key: physical.paired_seed_bootstrap(scores["connected"], value)
            for key, value in scores.items() if key != "connected"},
        "shuffle_permutation": permutation,
        "shuffle_method": "Deranged whole-episode donors; monotone normalized-time nearest-neighbor alignment; diagnostic only",
        "test_data": physical._data_manifest([row for rows in recorded.values() for row in rows]),
        "all_planned_episodes_included": True, "test_tuned": False,
        "development_lineage": study["lineage"], "reused_v3a_test_is_development": True,
        "new_test_groups": list(TEST_GROUPS), "source_hashes": source_hashes(),
        "scope": "Six-candidate refit; actual live-gain development; new three-seed physical test; same-gain paired frozen-head baseline"})
    v3._write_new(root / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "collect", "replay", "evaluate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--group", type=int)
    parser.add_argument("--condition", choices=("homeostasis_off", "transmission_off", "replay_check"), default="replay_check")
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.root)
    elif args.command == "train": result = train(args.root)
    elif args.command == "collect": result = collect(args.group, root=args.root)
    elif args.command == "replay": result = replay(args.condition, args.group, root=args.root)
    else: result = evaluate(args.root)
    if isinstance(result, dict):
        print(json.dumps({k: result[k] for k in ("digest", "status", "candidate_count", "regime", "selected_l2") if k in result}), flush=True)


if __name__ == "__main__": main()
