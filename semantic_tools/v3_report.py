"""Recompute/report complete v3 evidence; never collect, train or promote.

Final artifacts must exist before this module opens any test recording. Every
candidate and every planned condition is retained, including behavioral failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tomllib

import numpy as np

from fly_semantic.mapping import digest, file_digest
from fly_semantic.readout import LinearReadout
from semantic_tools import physical_learning as physical
from semantic_tools import v3_integrity as integrity
from semantic_tools import v3_learning as learning


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("connected", "homeostasis_off", "transmission_off", "cross_episode_shuffle", "train_constant")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sealed(path):
    value = read(path)
    if value.get("digest") != digest({k: v for k, v in value.items() if k != "digest"}):
        raise ValueError(f"Report artifact digest mismatch: {path}")
    return value


def same(left, right, message):
    if digest(left) != digest(right):
        raise ValueError(message)


def check_identities(plan, selection, binding, result, calibration, mapping):
    """Small pure fail-closed checks suitable for independent fixture tests."""
    for name in ("feature_digest", "calibration_digest", "nominal_weights_sha256"):
        if plan[name] != selection[name]:
            raise ValueError(f"Selected physical plan identity differs: {name}")
    if (result["plan_digest"] != plan["digest"] or binding["plan_digest"] != plan["digest"]
            or result["selection_digest"] != selection["digest"] or binding["selection_digest"] != selection["digest"]
            or result["test_binding_digest"] != binding["digest"]
            or result["model_sha256"] != selection["model_sha256"]
            or result["gain"] != selection["gain"] or plan["hunger_gain_mv"] != selection["gain"]
            or result["mask"] != selection["mask"]["kind"] or result["selected_l2"] != selection["selected_l2"]
            or calibration["hunger_gain_mv"] != selection["gain"]
            or calibration["digest"] != plan["calibration_digest"]
            or mapping["digest"] != plan["mapping_digest"] or calibration["mapping_digest"] != mapping["digest"]):
        raise ValueError("Report mixes model/gain/mask/calibration/plan bindings")
    if (result["test_tuned"] is not False or result["all_planned_episodes_included"] is not True
            or result["historical_v2_is_paired_baseline"] is not False):
        raise ValueError("Final report must retain prospective complete non-paired-baseline status")


def limited_coverage(rows):
    """Independent observation coverage, not a changed primary learning gate."""
    cases = []
    for row in rows:
        ep = row["metadata"]["episode"]
        if ep["scene"] != "limited_meal":
            continue
        amount = float(sum(p["amount"] for p in ep["config"]["food"]))
        intake = float(row["intake"][-1])
        if not np.isfinite(amount) or amount <= 0 or not np.isfinite(intake) or intake < 0:
            raise ValueError("Invalid physical portion or measured intake")
        cases.append({"episode_id": ep["episode_id"], "planned_portion": amount, "actual_intake": intake,
            "fraction_ingested": intake / amount, "coverage_pass": intake >= .95 * amount - 1e-12,
            "final_hunger": float(row["metadata"]["final_hunger"]),
            "target_active_final_sample": bool(row["y"][-1])})
    if len(cases) != 3 or sorted(c["planned_portion"] for c in cases) != [.6, 1.2, 1.6]:
        raise ValueError("Report requires exactly the three predeclared limited portions")
    return {"criterion": "Each limited_meal actual intake >=95% of its planned portion",
        "role": "Independent physical observation coverage required before diagnostic promotion; primary scores/gates remain unchanged",
        "all_pass": all(c["coverage_pass"] for c in cases), "cases": cases}


def verify_candidates(folder, selection, training):
    candidates = [sealed(item["path"]) for item in selection["candidate_artifacts"]]
    expected = [(g, m, l2) for g in learning.GAINS for m in learning.MASKS for l2 in learning.L2_GRID]
    actual = [(c["gain"], c["mask"]["kind"], c["l2"]) for c in candidates]
    if (actual != expected or [c["candidate_index"] for c in candidates] != list(range(18))
            or selection["candidate_count"] != 18 or selection["all_candidates_reported"] is not True):
        raise ValueError("Candidate grid omitted, duplicated or reordered")
    same(training["candidates"], candidates, "Training report differs from every frozen candidate")
    if training["selection_digest"] != selection["digest"]:
        raise ValueError("Training report refers to another selection")
    same(training["protocol"], learning.protocol_fields(), "Training protocol changed")
    chosen = max(candidates, key=learning.selection_key)
    if chosen["name"] != selection["candidate_name"] or chosen["model_sha256"] != selection["model_sha256"]:
        raise ValueError("Selected model is not the prescribed validation winner")
    development = {}
    for record in selection["development"]:
        path = Path(record["path"])
        plan = sealed(path / "plan.json")
        learning.validate_plan(plan)
        train = physical.load_rows(path, plan, "train")
        validation = physical.load_rows(path, plan, "validation")
        if len(train) != 18 or len(validation) != 6:
            raise ValueError("Development split is incomplete")
        X = np.concatenate([row["X"] for row in train])
        development[record["gain"]] = (plan, train, validation, X)
    for candidate in candidates:
        plan, train, validation, X = development[candidate["gain"]]
        keep, mask = learning.training_mask(X, candidate["mask"]["kind"])
        same(mask, candidate["mask"], "Candidate mask was not derived from TRAIN alone")
        model = LinearReadout.load(candidate["model_path"], expected_feature_digest=plan["feature_digest"])
        if (candidate["calibration_digest"] != plan["calibration_digest"]
                or candidate["nominal_weights_sha256"] != plan["nominal_weights_sha256"]
                or candidate["development_plan_digest"] != plan["digest"]
                or candidate["test_seen"] is not False or not candidate["optimizer"]["converged"]
                or np.any(model.weights[~keep] != 0) or np.any(model.mean[~keep] != 0)):
            raise ValueError("Candidate model/mask/optimizer provenance mismatch")
        # Scaler must be exactly that of masked TRAIN data, never validation/test.
        masked = X.copy(); masked[:, ~keep] = 0.
        np.testing.assert_array_equal(model.mean, masked.mean(axis=0))
        expected_scale = masked.std(axis=0)
        expected_scale = np.where(expected_scale > 1e-12, expected_scale, 1.)
        np.testing.assert_array_equal(model.scale, expected_scale)
        scored = physical.score_rows(model, validation, plan)
        same(scored, candidate["validation"], "Candidate validation cannot be recomputed from X")
        same(learning.assess_acceptance(scored, plan), candidate["assessment"], "Candidate validation gates changed")
    return candidates


def verify_probe(root):
    folder = Path(root) / "calibration"
    plan, result = sealed(folder / "gain-probe-plan.json"), sealed(folder / "gain-probe-results.json")
    if result["plan_digest"] != plan["digest"] or file_digest(result["raw_path"]) != result["raw_sha256"]:
        raise ValueError("Calibration curve bytes differ from frozen probe")
    if (result["selected_gain_mv"] is not None or result["validation_or_test_opened"] is not False
            or plan["gains_mv"] != [4., 4.5, 5., 6.] or not plan["v3_rate_guard"]["declared_before_results"]):
        raise ValueError("Calibration was not the predeclared TRAIN-only gain sweep")
    with np.load(result["raw_path"], allow_pickle=False) as a:
        curves = []
        for row in result["rows"]:
            if row["kind"] != "matched":
                continue
            direct = a[row["key"] + "__input_hz"]
            downstream = a[row["key"] + "__feature_hz"]
            if direct.shape != (32,) or downstream.shape != (512,) or not np.isfinite(direct).all() or not np.isfinite(downstream).all():
                raise ValueError("Calibration curve arrays malformed")
            curves.append({"gain": row["gain_mv"], "hunger": row["hunger_override"],
                "input_mean_hz": float(direct.mean()), "input_max_hz": float(direct.max()),
                "feature_mean_hz": float(downstream.mean()),
                "feature_fraction_at_least_300hz": float((downstream >= 300).mean())})
    if len(curves) != 16:
        raise ValueError("Incomplete matched calibration curves")
    return {"plan": plan, "result": result, "curves": curves}


def verify_evidence(folder):
    folder = Path(folder)
    # Fail BEFORE preservation hashes, model reads or any test NPZ access.
    final = [folder / name for name in ("selection.json", "plan.json", "training.json", "test-binding.json", "candidate-binding.json", "results.json")]
    if not all(p.is_file() for p in final):
        raise FileNotFoundError("Final v3 artifacts incomplete; report will not open test recordings")
    guard = integrity.check()
    if not guard["pass"] or guard["checked_files"] != 403:
        raise ValueError("Report requires all 403 old files and original runtime digests intact")
    binding = learning.verify_test_binding(folder)
    candidate_binding = read(folder / "candidate-binding.json")
    candidate_guard = integrity.check_candidate(folder / "candidate-binding.json")
    if not candidate_guard["pass"]:
        raise ValueError("Independent prospective candidate binding failed")
    plan, selection, training, result = [sealed(folder / name) for name in ("plan.json", "selection.json", "training.json", "results.json")]
    cal_dir = Path(plan["calibration_dir"])
    calibration, mapping = sealed(cal_dir / "calibration.json"), sealed(cal_dir / "mapping.json")
    check_identities(plan, selection, binding, result, calibration, mapping)
    for role, path in (("model", folder / "state-readout.npz"), ("study_plan", folder / "plan.json"),
                       ("mapping", cal_dir / "mapping.json"), ("calibration", cal_dir / "calibration.json")):
        record = candidate_binding["artifacts"].get(role)
        if record is None or (ROOT / record["path"]).resolve() != path.resolve() or file_digest(path) != record["sha256"]:
            raise ValueError(f"Independent candidate binding has wrong role: {role}")
    candidates = verify_candidates(folder, selection, training)
    episodes = [ep for ep in plan["episodes"] if ep["split"] == "test"]
    if len(episodes) != 18 or Counter(ep["seed_group"] for ep in episodes) != {4201: 6, 4202: 6, 4203: 6}:
        raise ValueError("Report requires all 18 predeclared fresh physical cases")
    expected_paths = { (folder / "episodes" / condition / (ep["episode_id"] + ".npz")).resolve()
        for condition in ("connected", "homeostasis_off", "transmission_off") for ep in episodes }
    bound_paths = {Path(path).resolve() for path in binding["future_test_outputs_absent"]}
    manifest = {Path(item["path"]).resolve(): item["sha256"] for item in result["test_data"]}
    candidate_future = {(ROOT / path).resolve() for path in candidate_binding["evaluation_outputs_absent_at_binding"]}
    if (candidate_future != expected_paths | {(folder / "results.json").resolve()}
            or len(candidate_future) != len(candidate_binding["evaluation_outputs_absent_at_binding"])):
        raise ValueError("Independent candidate binding must cover all 54 test recordings and final result")
    if (bound_paths != expected_paths or set(manifest) != expected_paths
            or len(manifest) != len(result["test_data"]) or len(bound_paths) != len(binding["future_test_outputs_absent"])):
        raise ValueError("Report test manifest/binding omits or duplicates planned recordings")
    if any(file_digest(path) != manifest[path] for path in expected_paths):
        raise ValueError("Recorded test bytes changed after evaluation")
    recorded = {condition: physical.load_rows(folder, plan, "test", condition)
        for condition in ("connected", "homeostasis_off", "transmission_off")}
    physical._consistent([row for rows in recorded.values() for row in rows])
    model = LinearReadout.load(folder / "state-readout.npz", expected_feature_digest=selection["feature_digest"])
    keep = np.zeros(1024, bool); keep[selection["mask"]["kept_columns"]] = True
    if np.any(model.weights[~keep] != 0) or np.any(model.mean[~keep] != 0):
        raise ValueError("Selected mask bypassed in exported model")
    live_checks = []
    connected = {row["metadata"]["episode"]["episode_id"]: row for row in recorded["connected"]}
    for condition, rows in recorded.items():
        for row in rows:
            meta = row["metadata"]
            if (meta.get("body_physics") is True) != (condition == "connected"):
                raise ValueError("Physical and fixed-body replay roles are mixed")
            workflow_key = "physical_workflow_sources" if condition == "connected" else "replay_workflow_sources"
            if meta.get(workflow_key) != plan["development_workflow_sources"]:
                raise ValueError("Test recording execution sources differ")
            if condition == "connected":
                with np.load(row["path"], allow_pickle=False) as a:
                    live = a["live_scores"]
                    predicted = np.array([model.predict_scores(x[None, :])[0, 0] for x in row["X"]])
                    np.testing.assert_array_equal(live, predicted, err_msg="Live scores differ from identical per-row model(X)")
                machine = physical._machine(plan, meta["episode"]["episode_id"])
                expected_events = []
                for score, when in zip(predicted, row["time_ms"]):
                    expected_events.extend(event.to_dict() for event in machine.update({1: float(score)}, int(when)))
                if expected_events != meta.get("live_events"):
                    raise ValueError("Actual live output events differ from exact model scores and hysteresis")
                live_checks.append({"episode_id": meta["episode"]["episode_id"], "samples": len(live),
                    "exact_per_row_scores": True, "exact_live_hysteresis_events": True})
            else:
                original = connected[meta["episode"]["episode_id"]]
                if (Path(meta["source_input_path"]).resolve() != Path(original["path"]).resolve()
                        or meta["source_input_sha256"] != original["sha256"]):
                    raise ValueError("Replay control does not bind its same physical trajectory")
                for name in ("y", "hunger", "intake", "time_ms"):
                    np.testing.assert_array_equal(row[name], original[name])
    shuffled, permutation = physical._shuffle_rows(recorded["connected"], plan.get("shuffle_seed", 81817))
    scored = {name: physical.score_rows(model, rows, plan)
              for name, rows in (recorded | {"cross_episode_shuffle": shuffled}).items()}
    constant = physical._ConstantHead(model.feature_digest, float(selection["train_positive_fraction"] >= .5))
    scored["train_constant"] = physical.score_rows(constant, recorded["connected"], plan)
    same(result["conditions"], scored, "Published test scores differ from frozen model(X)")
    same(result["shuffle_permutation"], permutation, "Shuffle donor identity mismatch")
    same(result["paired_connected_minus_control"], {name: physical.paired_seed_bootstrap(scored["connected"], value)
        for name, value in scored.items() if name != "connected"}, "Paired seed uncertainty differs")
    assessment = learning.assess_acceptance(scored["connected"], plan)
    same({key: result[key] for key in assessment}, assessment, "Result changes predeclared primary gates")
    meals = [c for c in scored["connected"]["cases"] if c["feeding_case"] or c["scene"] == "limited_meal"]
    if len(meals) != 9:
        raise ValueError("Report must show all six sufficient and three limited feeding cases")
    return {"plan": plan, "selection": selection, "training": training, "results": result,
        "calibration": calibration, "mapping": mapping, "preservation": guard, "test_binding": binding,
        "candidate_binding_guard": candidate_guard,
        "candidates": candidates, "recorded": recorded, "live_checks": live_checks,
        "limited_coverage": limited_coverage(recorded["connected"]), "probe": verify_probe(folder.parent),
        "study_dir": str(folder)}


def promotion_status(path, evidence):
    if path is None:
        return {"status": "PENDING", "reason": "Отдельный диагностический запуск не включён этим генератором."}
    value = read(path)
    if value.get("status") not in ("PENDING", "NOT_PROMOTED", "DIAGNOSTIC_ENABLED", "PROMOTED"):
        raise ValueError("Unrecognized diagnostic-promotion status")
    if (value.get("model_sha256") != evidence["selection"]["model_sha256"]
            or value.get("calibration_digest") != evidence["plan"]["calibration_digest"]):
        raise ValueError("Promotion binds a different model/calibration")
    if value["status"] in ("DIAGNOSTIC_ENABLED", "PROMOTED"):
        if (not all(evidence["results"]["checks"].values()) or not evidence["limited_coverage"]["all_pass"]):
            raise ValueError("Diagnostic promotion requires primary gates and actual limited-portion coverage")
        for role in ("launcher", "config"):
            if file_digest(value[role + "_path"]) != value[role + "_sha256"]:
                raise ValueError("Diagnostic launcher/config bytes changed")
        config = Path(value["config_path"])
        with config.open("rb") as stream:
            settings = tomllib.load(stream)["semantic"]
        if (settings["enabled"] is not True
                or file_digest((config.parent / settings["model"]).resolve()) != value["model_sha256"]
                or (config.parent / settings["calibration"]).resolve() != Path(evidence["plan"]["calibration_dir"]).resolve()):
            raise ValueError("Diagnostic configuration does not use the evaluated model/calibration")
    return value


def plot(evidence, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = evidence["results"]
    meals = [c for c in result["conditions"]["connected"]["cases"] if c["feeding_case"] or c["scene"] == "limited_meal"]
    rows = {r["metadata"]["episode"]["episode_id"]: r for r in evidence["recorded"]["connected"]}
    fig = plt.figure(figsize=(15, 17), constrained_layout=True)
    grid = fig.add_gridspec(5, 3)
    ax = fig.add_subplot(grid[0, :2])
    x = np.arange(len(CONDITIONS))
    raw = [result["conditions"][name]["raw"]["f1"] for name in CONDITIONS]
    indicated = [result["conditions"][name]["indicator"]["f1"] for name in CONDITIONS]
    ax.bar(x - .18, raw, .36, label="Raw F1", color="#7caabe")
    ax.bar(x + .18, indicated, .36, label="Visible indicator F1", color="#08788d")
    ax.axhline(.8, color="black", ls=":", lw=1)
    ax.set(xticks=x, xticklabels=["Physical", "Hunger off", "Edges off", "Shuffle", "Constant"], ylim=(0, 1.06), ylabel="F1", title="Fresh test; controls hold recorded body inputs fixed")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[0, 2]); ax.axis("off")
    check_lines = [f"{'PASS' if yes else 'FAIL'} {name}" for name, yes in result["checks"].items()]
    check_lines.append(f"{'PASS' if evidence['limited_coverage']['all_pass'] else 'FAIL'} actual portion coverage")
    ax.text(0, .95, "Predeclared gates + coverage\n\n" + "\n".join(check_lines), va="top", fontsize=8)
    for column, (field, ylabel) in enumerate((("input_mean_hz", "Mean input rate (Hz)"), ("feature_mean_hz", "Mean selected-cell rate (Hz)"), ("feature_fraction_at_least_300hz", "Selected cells >=300Hz (fraction)"))):
        ax = fig.add_subplot(grid[1, column])
        for gain in (4., 4.5, 5., 6.):
            curve = sorted([c for c in evidence["probe"]["curves"] if c["gain"] == gain], key=lambda c: c["hunger"])
            ax.plot([c["hunger"] for c in curve], [c[field] for c in curve], "o-", label=f"{gain:g}mV")
        ax.set(xlabel="Artificial hunger override", ylabel=ylabel, title="TRAIN calibration: same body-input window")
        ax.set_ylim(bottom=0); ax.legend(fontsize=7)
    arrays = {"condition_names": np.array(CONDITIONS), "raw_f1": np.array(raw), "indicator_f1": np.array(indicated)}
    for i, case in enumerate(meals):
        ax = fig.add_subplot(grid[2 + i // 3, i % 3])
        row = rows[case["episode_id"]]
        t = np.asarray(case["time_ms"]) / 1000
        ax.plot(t, case["scores"], color="#08788d", label="Neural score")
        ax.plot(t, row["hunger"], color="#d19029", alpha=.8, label="Hunger telemetry")
        ax.step(t, case["targets"], where="post", color="black", lw=1, label="Teacher target")
        ax.step(t, np.array(case["indicator_samples"]) * .10 - .16, where="post", color="#08788d", label="Visible indicator")
        if case["feeding_case"]:
            text = f"Stable release: {case['release_supported']}"
        else:
            portion = sum(p["amount"] for p in row["metadata"]["episode"]["config"]["food"])
            text = f"Portion {portion:g}; recall {case['indicator']['recall']:.3f}"
        ax.set(title=f"{case['episode_id']}\nIntake {case['intake']:.3f}; {text}", xlabel="Simulation time (s)", ylim=(-.2, 1.05))
        ax.set_yticks([0, .5, 1])
        if i == 0: ax.legend(fontsize=6)
        prefix = f"meal_{i}_"
        arrays.update({prefix + "episode_id": np.array(case["episode_id"]), prefix + "time_ms": np.asarray(case["time_ms"]),
            prefix + "scores": np.asarray(case["scores"]), prefix + "targets": np.asarray(case["targets"]),
            prefix + "indicator": np.asarray(case["indicator_samples"]), prefix + "hunger": row["hunger"], prefix + "intake": row["intake"]})
    fig.savefig(output / "summary.png", dpi=160); fig.savefig(output / "summary.svg"); plt.close(fig)
    arrays["calibration_curves_json"] = np.array(json.dumps(evidence["probe"]["curves"]))
    arrays["metadata"] = np.array(json.dumps({"model_sha256": result["model_sha256"], "plan_digest": result["plan_digest"], "historical_v2_is_paired_baseline": False}))
    np.savez_compressed(output / "summary.npz", **arrays)


def render(evidence, promotion, output):
    selection, result = evidence["selection"], evidence["results"]
    connected = result["conditions"]["connected"]
    coverage = evidence["limited_coverage"]
    rows = {r["metadata"]["episode"]["episode_id"]: r for r in evidence["recorded"]["connected"]}
    lines = ["# NEED_FOOD v3: оставшийся голод после небольшой порции", "",
        f"Статус основного физического пилота: **{result['status']}**. Отдельный диагностический запуск: **{promotion['status']}**. "
        f"Покрытие фактически съеденных ограниченных порций: **{'PASS' if coverage['all_pass'] else 'FAIL'}**.", "",
        f"Выбран gain **{selection['gain']:g} mV**, маска **{selection['mask']['kind']}**, L2 **{selection['selected_l2']:g}**; "
        f"осталось {len(selection['mask']['kept_cells'])}/512 клеток. Raw F1 **{connected['raw']['f1']:.3f}**, "
        f"F1 видимого индикатора **{connected['indicator']['f1']:.3f}**. "
        f"Устойчивое снятие после достаточной еды: {connected['feeding_cases_with_release']}/{connected['feeding_cases']}; "
        f"задержки, мс: {connected['supported_release_latency_ms']}.", "",
        "![Все девять кормлений, нейронные контроли и калибровка](summary.png)", "",
        "## Что изменено и что проверялось", "",
        "В v2 индикатор научился замолкать после достаточной еды, но мог выключаться после малой порции при оставшемся голоде. "
        "V3 меняет усиление искусственного входа `semantic_hunger` и заново обучает небольшой линейный считыватель. "
        "Физическое тело, существующие моторные механизмы и исходный LIF с весами полного коннектома сохранены. "
        "Считыватель получает только нейронные признаки; голод, пища, время и учитель доступны оценщику, но не модели.", "",
        "Разработка использовала 18 прежних физических TRAIN и 6 validation эпизодов: записанные входы органов чувств "
        "и тела воспроизводились через мозг при новом gain. Поэтому это нейронные повторы при фиксированном телесном фоне, "
        "а не новая физическая жизнь каждого кандидата. Проверены ровно 18 кандидатов: 3 gain × 2 маски × 3 L2. "
        "Маска `low_saturation` вычисляется только из TRAIN: удаляется клетка, у которой максимум двух внешних трасс "
        "≥300 Hz не менее чем в 25% TRAIN-отсчётов; удаляются обе колонки, экспортируются нулевые веса. "
        "Нормализация также вычисляется только на TRAIN. Маска не меняет нейроны или связи мозга.", "",
        "После выбора сохранена точная модель и привязаны будущие файлы; затем выполнены 18 новых свободных физических "
        "эпизодов на seed-группах 4201/4202/4203. В каждой группе шесть сцен. Ограниченные порции заранее заданы "
        "0,6 / 1,2 / 1,6; достаточные кормления включают быстрый и медленный приём пищи. Все случаи входят в оценку, "
        "включая отсутствие ожидаемого питания, пропуск сигнала или неудачное снятие.", "",
        "Потребность учителя включается при hunger≥0,70 и снимается при hunger≤0,55. Видимый индикатор включает "
        "сигнал при score≥0,70, выключает при score≤0,40, подтверждает два отсчёта по 100 ms. "
        "Score — выход обученного классификатора, не измеренная вероятность голода. Трассы 50/200 ms — внешняя память считывателя.", "",
        "## Все условия и критерии", "", "| Условие | Raw F1 | Индикатор F1 |", "|---|---:|---:|"]
    for name in CONDITIONS:
        score = result["conditions"][name]
        lines.append(f"| {name} | {score['raw']['f1']:.3f} | {score['indicator']['f1']:.3f} |")
    lines += ["", "`homeostasis_off` удаляет искусственный вход голода; `transmission_off` обнуляет передачу связей "
        "только в памяти процесса. Оба контроля повторяют записанные входы новой физической траектории при фиксированном "
        "теле. Это не свободное поведение после вмешательства. Перестановка меняет целого донора нейронной траектории, "
        "с выравниванием по относительному времени; постоянная голова определяется TRAIN. Головы на контролях не переобучаются.", "",
        "| Основной критерий | Заданный предел | Результат |", "|---|---|---|"]
    for key, passed in result["checks"].items():
        lines.append(f"| {key} | {learning.ACCEPTANCE[key]} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "Проверка покрытия ограниченной порции — отдельная наблюдательная проверка перед включением "
        "диагностического запуска: фактически съедено не менее 95% запланированной порции в каждом из трёх случаев. "
        "Она не изменяет F1 или основные критерии пилота и не позволяет выдавать отсутствие контакта с едой за успех после еды.", "",
        "| Ограниченная порция | Фактически съедено | Остаточный голод | Recall индикатора | Покрытие |", "|---:|---:|---:|---:|---|"]
    cases = {c["episode_id"]: c for c in connected["cases"]}
    for c in coverage["cases"]:
        lines.append(f"| {c['planned_portion']:g} | {c['actual_intake']:.4f} | {c['final_hunger']:.4f} | {cases[c['episode_id']]['indicator']['recall']:.3f} | {'PASS' if c['coverage_pass'] else 'FAIL'} |")
    lines += ["", "## Каждый физический test", "", "| Эпизод | Приём пищи | Raw F1 | Индикатор F1 | Recall | Устойчивое снятие |", "|---|---:|---:|---:|---:|---|"]
    for c in connected["cases"]:
        release = str(c["release_supported"]) if c["feeding_case"] else "не ожидается"
        lines.append(f"| {c['episode_id']} | {c['intake']:.3f} | {c['raw']['f1']:.3f} | {c['indicator']['f1']:.3f} | {c['indicator']['recall']:.3f} | {release} |")
    lines += ["", "## Все кандидаты на validation", "",
        "Выбор сделан до новых физических test по сохранённому правилу: сначала все validation-критерии, "
        "затем худший recall ограниченного питания, indicator F1, raw F1, более сильная L2 и исходный порядок. "
        "Ни один кандидат не пропущен; только выбранная голова проходит новый test.", "",
        "| Gain | Маска | L2 | Клеток | Raw F1 | Индикатор F1 | Мин. limited recall | Все критерии |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    for c in evidence["candidates"]:
        v = c["validation"]; a = c["assessment"]
        lines.append(f"| {c['gain']:g} | {c['mask']['kind']} | {c['l2']:g} | {len(c['mask']['kept_cells'])} | "
            f"{v['raw']['f1']:.3f} | {v['indicator']['f1']:.3f} | {a['minimum_limited_case_recall']:.3f} | {all(a['checks'].values())} |")
    lines += ["", "## Калибровка и пределы модели", "",
        "Калибровка выполнена до выбора gain: 52 пробы, 36,4 s модельного времени. Взяты последние 700 ms пяти "
        "TRAIN-фонов группы 1201 с холодным стартом мозга; оценено последнее окно 300 ms. В отдельном сравнении "
        "при одном фоне limited_meal заменялся только искусственный вход голода на .45/.60/.75/.85. "
        "Это проверка канала, не точный повтор полной истории голода и не доказательство причины всех ошибок v2.", "",
        "| Gain, mV | Входные частоты для .45/.60/.75/.85, Hz | Старый предел <200 Hz | Заранее заданный v3-предел |", "|---:|---|---|---|"]
    for gain in (4., 4.5, 5., 6.):
        curve = sorted([c for c in evidence["probe"]["curves"] if c["gain"] == gain], key=lambda c: c["hunger"])
        gate = next(g for g in evidence["probe"]["result"]["gain_gates"] if g["gain_mv"] == gain)
        lines.append(f"| {gain:g} | {' / '.join(f'{c["input_mean_hz"]:.2f}' for c in curve)} | {gate['historical_gate_pass']} | {gate['v3_all_probe_rate_guards_pass']} |")
    maximum_feature_saturation = max(c["feature_fraction_at_least_300hz"] for c in evidence["probe"]["curves"])
    lines += ["", "Старый предел входной частоты <200 Hz был инженерным выбором v1. До новых результатов для v3 "
        "заданы конечность состояний, <1% входных клеток и <1% всех клеток с частотой ≥300 Hz, близкой к пределу "
        "исходного refractory-механизма ≈333 Hz. Оба результата сохранены: gain6 достигает ровно200 Hz и нарушает "
        "старый предел, но проходит новый. Это явное изменение критерия нового исследования, не переписывание старой калибровки.", "",
        f"Глобальная доля ≥300 Hz мала, однако среди выбранных 512 признаковых клеток в активном режиме достигает "
        f"**{100*maximum_feature_saturation:.2f}%** уже при gain4. Поэтому ответ имеет выраженное переключение и локальное "
        "насыщение. Маскирование обучаемых колонок не исправляет биофизическую модель. Успех классификации не доказывает "
        "биологическую семантику или найденный естественный центр голода.", "",
        "95% интервалы разностей F1 рассчитаны по парным seed-группам, а не по нейронным кадрам; исходные значения "
        "и параметры bootstrap находятся в overall.json. Три группы — предварительный пилот. Нет проверки множества "
        "геометрий, длительной жизни, новых потребностей, подсказок направления или языкового понимания.", "",
        "Исторический v2 indicator F1≈0,732 получен на другом test при другом входном gain. Его нельзя вычитать "
        "из текущего F1 как парный эффект новой головы. Здесь парными являются только текущие connected и его "
        "контроли с сохранёнными траекториями.", "",
        "## Проверяемость и воспроизведение", "",
        f"Генератор подтвердил неизменность **{evidence['preservation']['checked_files']} старых файлов**, все 18 "
        "кандидатов, TRAIN-маски и нормализацию, 54 test-записи, привязки модели/gain/калибровки и повторный расчёт "
        "всех метрик из X. Во всех 18 физических эпизодах сохранённые live_scores точно совпадают с предсказанием "
        "той же модели для каждой строки X. Это технические проверки; качество определяется показанными выше результатами.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m semantic_tools.v3_report --help", "```", "",
        "Генератор читает уже завершённые результаты, не запускает обучение и не включает демонстрацию. "
        "Для повторного рендера нужен новый каталог вывода. Исходные артефакты не перезаписываются. "
        "Новый эксперимент после подстройки требует нового заранее выделенного test."]
    return "\n".join(lines) + "\n"


def run(study, output, promotion=None, docs=None):
    output = Path(output)
    names = ("results.md", "overall.json", "summary.png", "summary.svg", "summary.npz")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Frozen report exists; use a new report directory")
    evidence = verify_evidence(study)
    promoted = promotion_status(promotion, evidence)
    output.mkdir(parents=True, exist_ok=True)
    plot(evidence, output)
    text = render(evidence, promoted, output)
    with (output / "results.md").open("x", encoding="utf-8") as stream: stream.write(text)
    overall = {"schema_version": 3, "results": evidence["results"], "selection": evidence["selection"],
        "preservation": evidence["preservation"], "candidate_binding_guard": evidence["candidate_binding_guard"],
        "live_checks": evidence["live_checks"],
        "limited_coverage": evidence["limited_coverage"], "promotion": promoted,
        "calibration_curves": evidence["probe"]["curves"], "candidate_count_verified": len(evidence["candidates"]),
        "test_recordings_verified": sum(len(rows) for rows in evidence["recorded"].values()),
        "generator_sha256": file_digest(Path(__file__)), "historical_v2_is_paired_baseline": False,
        "artifacts": {name: file_digest(output / name) for name in names if name != "overall.json"}}
    with (output / "overall.json").open("x", encoding="utf-8") as stream:
        json.dump(overall, stream, indent=2, ensure_ascii=False, allow_nan=False)
    if docs is not None:
        path = Path(docs)
        if path.resolve() != (ROOT / "docs/SEMANTIC_CHANNEL_V3.md").resolve():
            raise ValueError("Only the new V3 guide can be finalized")
        summary = (f"# NEED_FOOD v3\n\nСтатус: **{evidence['results']['status']} / {promoted['status']}**.\n\n"
            f"Канал использует gain {evidence['selection']['gain']:g} mV, маску `{evidence['selection']['mask']['kind']}`, "
            f"L2={evidence['selection']['selected_l2']:g}. F1 видимого индикатора на 18 новых физических эпизодах: "
            f"**{evidence['results']['conditions']['connected']['indicator']['f1']:.3f}**. "
            f"Покрытие съеденных ограниченных порций: **{'PASS' if evidence['limited_coverage']['all_pass'] else 'FAIL'}**.\n\n"
            "[Подробный отчёт со всеми кормлениями и ограничениями](../reports/semantic_channel_v3/results.md).\n\n"
            "Разработка: прежние TRAIN/validation записи тела воспроизводились при новом gain; test: новая физическая жизнь. "
            "Голова читает нейронные признаки и сообщает потребность, текущими действиями тела управляют существующие механизмы. "
            "Старые режимы и веса коннектома сохранены. Результат — инженерный пилот, не восстановленная биологическая модель.\n\n"
            "Генератор отчёта: `.\\.venv\\Scripts\\python.exe -m semantic_tools.v3_report --help`. Он не обучает и не включает демонстрацию.\n")
        path.write_text(summary, encoding="utf-8")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/semantic-v3/selected"))
    parser.add_argument("--output", type=Path, default=Path("reports/semantic_channel_v3"))
    parser.add_argument("--promotion", type=Path)
    parser.add_argument("--docs", type=Path)
    args = parser.parse_args()
    value = run(args.study, args.output, args.promotion, args.docs)
    print(json.dumps({"status": value["results"]["status"], "promotion": value["promotion"]["status"],
        "limited_coverage_pass": value["limited_coverage"]["all_pass"]}, allow_nan=False))
