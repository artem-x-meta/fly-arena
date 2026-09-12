"""Report the separate physical refit, retaining the completed negative v3a.

No fresh test record is read before complete final artifacts exist. Both heads
are rescored on the SAME new X because gain, graph, ports and feature identities
remain identical. This is distinct from comparing historical aggregate scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from fly_semantic.mapping import file_digest
from fly_semantic.readout import LinearReadout
from semantic_tools import physical_learning as physical
from semantic_tools import v3_learning as learning
from semantic_tools import v3_refit as refit
from semantic_tools import v3_report as prior_report


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = prior_report.CONDITIONS
read, sealed, same = prior_report.read, prior_report.sealed, prior_report.same


def release_boundaries(scored):
    """Describe failed release timing without changing the frozen success rule."""
    rows=[]
    for case in scored["cases"]:
        if not case["feeding_case"] or case["release_supported"]:
            continue
        previous=False
        actual_offs=[]
        for event in case["events"]:
            if previous and not event["active"]:
                actual_offs.append(float(event["sim_time_ms"]))
            previous=event["active"]
        for match in case["release_matches"]:
            if match["stable_release_supported"]:
                continue
            target=float(match["target_off_ms"])
            prior_off=max((when for when in actual_offs if when<target),default=None)
            early=prior_off is not None and not match["indicator_active_before_target_off"]
            rows.append({"episode_id":case["episode_id"],"target_off_ms":target,
                "previous_neural_off_ms":prior_off,"early_by_ms":target-prior_off if early else None,
                "failure_kind":"early_off" if early else "other_unsupported_release",
                "primary_success_unchanged":False})
    return rows


def verify_candidates(study, selected, training):
    candidates = [sealed(a["path"]) for a in selected["candidate_artifacts"]]
    expected = [(regime, l2) for regime in refit.REGIMES for l2 in learning.L2_GRID]
    if ([(c["regime"], c["l2"]) for c in candidates] != expected
            or [c["candidate_index"] for c in candidates] != list(range(6))
            or selected["candidate_count"] != 6 or selected["all_candidates_reported"] is not True):
        raise ValueError("Refit report requires exactly all six predeclared candidates")
    same(candidates, training["candidates"], "Refit candidate records differ from training report")
    if training["selection_digest"] != selected["digest"] or training["plan_digest"] != study["digest"]:
        raise ValueError("Training report refers to another selection or plan")
    chosen = max(candidates, key=learning.selection_key)
    if (chosen["name"] != selected["candidate_name"] or chosen["model_sha256"] != selected["model_sha256"]
            or chosen["regime"] != selected["regime"] or chosen["l2"] != selected["selected_l2"]):
        raise ValueError("Chosen refit differs from the frozen selection rule")
    data = refit.load_development(study)
    for c in candidates:
        train = data["physical_train"] + (data["replay_train"] if c["regime"] == "mixed_replay" else [])
        validation = data["physical_validation"]
        X, y = np.concatenate([r["X"] for r in train]), np.concatenate([r["y"] for r in train])
        same(c["train_data"], physical._data_manifest(train), "Candidate TRAIN inputs differ from declared regime")
        same(c["validation_data"], physical._data_manifest(validation), "Candidate validation inputs differ")
        if (c["unique_train_episodes"] != len(train) or c["unique_validation_episodes"] != 6
                or c["train_rows"] != len(X) or c["train_positive_fraction"] != float(y.mean())
                or c["fresh_test_seen"] is not False or not c["optimizer"]["converged"]):
            raise ValueError("Refit training counts, optimization or held-out status differ")
        model = LinearReadout.load(c["model_path"], expected_feature_digest=study["feature_digest"])
        np.testing.assert_array_equal(model.mean, X.mean(axis=0))
        std = X.std(axis=0)
        np.testing.assert_array_equal(model.scale, np.where(std > 1e-12, std, 1.))
        score = physical.score_rows(model, validation, study)
        same(score, c["validation"], "Candidate validation not reproducible from its own model(X)")
        same(learning.assess_acceptance(score, study), c["assessment"], "Refit candidate gates changed")
    return candidates


def verify_parent(study):
    parent = Path(study["parent_path"])
    evidence = prior_report.verify_evidence(parent)
    if (evidence["results"]["status"] != "TARGET_NOT_REACHED"
            or evidence["results"]["model_sha256"] != study["baseline_model_sha256"]):
        raise ValueError("Refit must retain the documented negative v3a and its frozen head")
    audit_path = parent.parent / "verification/report-final-checks.json"
    audit = read(audit_path)
    if audit.get("status") != "PASS" or audit.get("study_status") != "TARGET_NOT_REACHED":
        raise ValueError("Completed v3a final report audit is missing or invalid")
    changed = [path for path, sha in audit["artifact_sha256"].items() if file_digest(path) != sha]
    if changed:
        raise ValueError(f"Completed v3a report/docs/plots changed: {changed}")
    for field in ("calibration_digest", "feature_digest", "nominal_weights_sha256"):
        if evidence["plan"][field] != study[field]:
            raise ValueError("Same-X baseline claim changes calibration, features or graph")
    if evidence["plan"]["hunger_gain_mv"] != study["hunger_gain_mv"] or study["hunger_gain_mv"] != 5.:
        raise ValueError("Same-X comparison requires unchanged gain5")
    return {"old_preservation": evidence["preservation"], "v3a_candidate_binding": evidence["candidate_binding_guard"],
        "v3a_test_recordings_checked": sum(len(rows) for rows in evidence["recorded"].values()),
        "v3a_report_audit_path": str(audit_path), "v3a_report_audit_sha256": file_digest(audit_path),
        "v3a_report_artifacts_checked": len(audit["artifact_sha256"]), "v3a_results_sha256": file_digest(parent / "results.json")}


def verify_evidence(root):
    root = Path(root)
    required = [root / name for name in ("plan.json", "selection.json", "training.json", "test-binding.json", "results.json")]
    if not all(p.is_file() for p in required):
        raise FileNotFoundError("Refit final artifacts incomplete; no fresh test recording will be opened")
    study, selected, binding = refit.verify_test_binding(root)
    result, training = sealed(root / "results.json"), sealed(root / "training.json")
    if (result["plan_digest"] != study["digest"] or result["selection_digest"] != selected["digest"]
            or result["binding_digest"] != binding["digest"] or result["model_sha256"] != selected["model_sha256"]
            or result["baseline_model_sha256"] != study["baseline_model_sha256"]
            or result["gain"] != selected["gain"] or result["gain"] != study["hunger_gain_mv"]
            or result["mask"] != "all" or selected["mask"] != "all"
            or result["selected_l2"] != selected["selected_l2"] or result["regime"] != selected["regime"]
            or result["all_planned_episodes_included"] is not True or result["test_tuned"] is not False
            or result["reused_v3a_test_is_development"] is not True or result["new_test_groups"] != [5201, 5202, 5203]):
        raise ValueError("Refit result/selection/plan identity or prospective status mismatch")
    same(result["development_lineage"], study["lineage"], "Refit report changed development origins")
    cal_dir = Path(study["calibration_dir"])
    calibration, mapping = sealed(cal_dir / "calibration.json"), sealed(cal_dir / "mapping.json")
    if (calibration["digest"] != study["calibration_digest"] or mapping["digest"] != study["mapping_digest"]
            or calibration["mapping_digest"] != mapping["digest"] or calibration["hunger_gain_mv"] != 5.):
        raise ValueError("Physical refit calibration or mapping changed")
    parent_guard = verify_parent(study)
    candidates = verify_candidates(study, selected, training)
    episodes = [ep for ep in study["episodes"] if ep["split"] == "test"]
    if Counter(ep["seed_group"] for ep in episodes) != {5201: 6, 5202: 6, 5203: 6}:
        raise ValueError("Fresh refit test is not the full 18-episode frozen set")
    expected = {(root / "episodes" / condition / (ep["episode_id"] + ".npz")).resolve()
        for condition in ("connected", "homeostasis_off", "transmission_off") for ep in episodes}
    bound = {Path(p).resolve() for p in binding["future_test_outputs_absent"]}
    manifest = {Path(a["path"]).resolve(): a["sha256"] for a in result["test_data"]}
    if (bound != expected or set(manifest) != expected or len(bound) != len(binding["future_test_outputs_absent"])
            or len(manifest) != len(result["test_data"]) or any(file_digest(p) != manifest[p] for p in expected)):
        raise ValueError("Refit test data missing, duplicated, unbound or changed after evaluation")
    # If root also supplied the independent v3 integrity binding, verify it.
    independent = None
    if (root / "candidate-binding.json").exists():
        independent = prior_report.integrity.check_candidate(root / "candidate-binding.json")
        if not independent["pass"]:
            raise ValueError("Independent refit candidate binding failed")
        ib = read(root / "candidate-binding.json")
        future = {(ROOT / p).resolve() for p in ib["evaluation_outputs_absent_at_binding"]}
        if not expected | {(root / "results.json").resolve()} <= future:
            raise ValueError("Independent refit binding omits planned tests/result")
    recorded = {condition: physical.load_rows(root, study, "test", condition)
        for condition in ("connected", "homeostasis_off", "transmission_off")}
    physical._consistent([row for rows in recorded.values() for row in rows])
    model = LinearReadout.load(root / "state-readout.npz", expected_feature_digest=study["feature_digest"])
    if file_digest(study["baseline_model_path"]) != study["baseline_model_sha256"]:
        raise ValueError("Frozen v3a baseline model changed")
    baseline = LinearReadout.load(study["baseline_model_path"], expected_feature_digest=study["feature_digest"])
    originals = {r["metadata"]["episode"]["episode_id"]: r for r in recorded["connected"]}
    live_checks = []
    for condition, rows in recorded.items():
        for row in rows:
            meta = row["metadata"]
            if (meta.get("body_physics") is True) != (condition == "connected"):
                raise ValueError("Fresh physical and fixed-body control roles mixed")
            if condition == "connected":
                with np.load(row["path"], allow_pickle=False) as archive:
                    live = archive["live_scores"]
                    predictions = np.array([model.predict_scores(x[None, :])[0, 0] for x in row["X"]])
                    np.testing.assert_array_equal(live, predictions)
                machine = physical._machine(study, meta["episode"]["episode_id"])
                events = []
                for score, when in zip(predictions, row["time_ms"]):
                    events.extend(e.to_dict() for e in machine.update({1: float(score)}, int(when)))
                if events != meta.get("live_events"):
                    raise ValueError("Refit live events do not follow saved model scores/hysteresis")
                live_checks.append({"episode_id": meta["episode"]["episode_id"], "exact_scores": True,
                    "exact_events": True, "samples": len(live)})
            else:
                original = originals[meta["episode"]["episode_id"]]
                if (meta.get("matched_original_hunger") is not True
                        or Path(meta["source_input_path"]).resolve() != Path(original["path"]).resolve()
                        or meta["source_input_sha256"] != original["sha256"]):
                    raise ValueError("Control is not paired to its exact physical source")
                for field in ("y", "hunger", "intake", "time_ms"):
                    np.testing.assert_array_equal(row[field], original[field])
    shuffled, permutation = physical._shuffle_rows(recorded["connected"], study.get("shuffle_seed", 81817))
    conditions = recorded | {"cross_episode_shuffle": shuffled}
    scores = {name: physical.score_rows(model, rows, study) for name, rows in conditions.items()}
    baseline_scores = {name: physical.score_rows(baseline, rows, study) for name, rows in conditions.items()}
    constant = physical._ConstantHead(model.feature_digest, float(selected["train_positive_fraction"] >= .5))
    scores["train_constant"] = physical.score_rows(constant, recorded["connected"], study)
    same(scores, result["conditions"], "Refit scores differ from saved model and X")
    same(baseline_scores, result["baseline_conditions"], "Paired v3a baseline was not scored on same X")
    same(permutation, result["shuffle_permutation"], "Refit shuffled donors changed")
    same(result["paired_refit_minus_v3a"], {name: physical.paired_seed_bootstrap(scores[name], baseline_scores[name])
        for name in conditions}, "Paired refit-baseline uncertainty differs")
    same(result["paired_connected_minus_control"], {name: physical.paired_seed_bootstrap(scores["connected"], value)
        for name, value in scores.items() if name != "connected"}, "Paired refit-control uncertainty differs")
    acceptance = learning.assess_acceptance(scores["connected"], study)
    same(acceptance, {key: result[key] for key in acceptance}, "Refit status changes frozen acceptance")
    return {"plan": study, "selection": selected, "results": result, "candidates": candidates,
        "recorded": recorded, "live_checks": live_checks, "limited_coverage": prior_report.limited_coverage(recorded["connected"]),
        "release_boundary_diagnostics": release_boundaries(scores["connected"]),
        "parent_guard": parent_guard, "independent_binding_guard": independent,
        "calibration": calibration, "mapping": mapping}


def plot(evidence, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = evidence["results"]
    current, old = result["conditions"]["connected"], result["baseline_conditions"]["connected"]
    meals = [c for c in current["cases"] if c["feeding_case"] or c["scene"] == "limited_meal"]
    if len(meals) != 9:
        raise ValueError("Figure requires all six sufficient and three limited feedings")
    old_cases = {c["episode_id"]: c for c in old["cases"]}
    body_rows = {r["metadata"]["episode"]["episode_id"]: r for r in evidence["recorded"]["connected"]}
    fig = plt.figure(figsize=(15, 14), constrained_layout=True)
    grid = fig.add_gridspec(4, 3)
    ax = fig.add_subplot(grid[0, :2])
    labels = CONDITIONS[:-1]
    x = np.arange(len(labels))
    before = [result["baseline_conditions"][name]["indicator"]["f1"] for name in labels]
    after = [result["conditions"][name]["indicator"]["f1"] for name in labels]
    ax.bar(x-.18, before, .36, color="#8c6cbb", label="Frozen v3a head, same new X")
    ax.bar(x+.18, after, .36, color="#08788d", label="Physical refit head, same new X")
    ax.axhline(.8, ls=":", color="black", lw=1)
    ax.set(xticks=x, xticklabels=["Physical", "Hunger off", "Edges off", "Shuffle"], ylabel="Indicator F1", ylim=(0,1.06),
        title="New 520x test; same gain5 / features / body trajectories")
    ax.legend(fontsize=8)
    ax=fig.add_subplot(grid[0,2]); ax.axis("off")
    lines=[f"{'PASS' if passed else 'FAIL'} {name}" for name,passed in result["checks"].items()]
    lines.append(f"{'PASS' if evidence['limited_coverage']['all_pass'] else 'FAIL'} actual limited-portion coverage")
    ax.text(0,.95,"Frozen gates + observation coverage\n\n"+"\n".join(lines),va="top",fontsize=8)
    arrays={"conditions":np.array(labels),"v3a_indicator_f1":np.array(before),"refit_indicator_f1":np.array(after)}
    for i,case in enumerate(meals):
        ax=fig.add_subplot(grid[1+i//3,i%3]); previous=old_cases[case["episode_id"]]; body=body_rows[case["episode_id"]]
        t=np.array(case["time_ms"])/1000
        ax.plot(t,previous["scores"],color="#8c6cbb",alpha=.7,label="v3a score")
        ax.plot(t,case["scores"],color="#08788d",label="Refit score")
        ax.plot(t,body["hunger"],color="#d19029",alpha=.6,label="Hunger telemetry")
        ax.step(t,case["targets"],where="post",color="black",lw=1,label="Teacher target")
        ax.step(t,np.array(previous["indicator_samples"])*.08-.15,where="post",color="#8c6cbb",label="v3a indicator")
        ax.step(t,np.array(case["indicator_samples"])*.08-.28,where="post",color="#08788d",label="Refit indicator")
        if case["feeding_case"]: detail=f"Stable release {previous['release_supported']} -> {case['release_supported']}"
        else: detail=f"Recall {previous['indicator']['recall']:.2f} -> {case['indicator']['recall']:.2f}"
        ax.set(title=f"{case['episode_id']}\nIntake {case['intake']:.3f}; {detail}",xlabel="Simulation time (s)",ylim=(-.33,1.05))
        ax.set_yticks([0,.5,1])
        if i==0:ax.legend(fontsize=6)
        prefix=f"meal_{i}_"
        arrays.update({prefix+"episode_id":np.array(case["episode_id"]),prefix+"time_ms":np.array(case["time_ms"]),
            prefix+"v3a_scores":np.array(previous["scores"]),prefix+"refit_scores":np.array(case["scores"]),
            prefix+"targets":np.array(case["targets"]),prefix+"v3a_indicator":np.array(previous["indicator_samples"]),
            prefix+"refit_indicator":np.array(case["indicator_samples"]),prefix+"hunger":body["hunger"],prefix+"intake":body["intake"]})
    fig.savefig(output/"summary.png",dpi=160);fig.savefig(output/"summary.svg");plt.close(fig)
    arrays["metadata"]=np.array(json.dumps({"plan_digest":result["plan_digest"],"model_sha256":result["model_sha256"],
        "baseline_model_sha256":result["baseline_model_sha256"],"paired_same_X":True,"gain_mv":5.}))
    np.savez_compressed(output/"summary.npz",**arrays)


def render(evidence,promotion):
    selection,result=evidence["selection"],evidence["results"]
    current,old=result["conditions"]["connected"],result["baseline_conditions"]["connected"]
    chosen=next(c for c in evidence["candidates"] if c["name"]==selection["candidate_name"])
    rows={r["metadata"]["episode"]["episode_id"]:r for r in evidence["recorded"]["connected"]}
    old_cases={c["episode_id"]:c for c in old["cases"]}
    lines=["# NEED_FOOD: отдельное переобучение на физической жизни", "",
        f"Статус: **{result['status']} / {promotion['status']}**. "
        f"Новый физический test: raw F1 **{current['raw']['f1']:.3f}**, indicator F1 **{current['indicator']['f1']:.3f}**. "
        f"Покрытие фактически съеденных ограниченных порций: **{'PASS' if evidence['limited_coverage']['all_pass'] else 'FAIL'}**.", "",
        f"Выбрана схема **{selection['regime']}**, L2 **{selection['selected_l2']:g}**, gain остаётся **5 mV**, "
        "используются все 512 клеток /1024 внешних трассы. Веса коннектома и физический контроллер сохранены. "
        "Обучается только небольшой внешний линейный считыватель потребности.", "",
        "Основная проверка оставшегося голода после малой порции: "
        + ", ".join(f"{c['planned_portion']:g} единицы пищи — recall {next(case for case in current['cases'] if case['episode_id']==c['episode_id'])['indicator']['recall']:.3f}"
            for c in evidence["limited_coverage"]["cases"]) + ". "
        "Эти показатели относятся к текущему пилоту; общий статус сохраняет все отдельные критерии достаточного кормления.", "",
        "![Обе головы на одинаковых новых данных: все девять кормлений](summary.png)", "",
        "## История и разделение данных", "",
        "[Предыдущий завершённый пилот v3a](../semantic_channel_v3/results.md) имел indicator F1≈0,854, "
        "но не выполнил критерии остаточного голода и задержки снятия сигнала. Его отрицательный результат, "
        "модель и все артефакты сохранены. Новый этап повторно обучает голову с нуля на физических записях при уже выбранном gain5.", "",
        "Все 12 прежних физических случаев групп4201/4202 назначены новым TRAIN. Все6 случаев4203 используются "
        "как validation. Их прежний статус test относится только к v3a: в новом исследовании это явные данные разработки, "
        "уже виденные до выбора рецепта. Они не объявляются независимой проверкой нового результата.", "",
        "Ровно шесть кандидатов заданы заранее: `actual_only` и `mixed_replay`, каждый с L2=0,001/0,01/0,1. "
        "Первый использует12 физических TRAIN-эпизодов; второй добавляет все24 прежних повторных нейронных записей "
        "gain5 к TRAIN по одному разу, включая прежнюю validation2201. Новая validation4203 остаётся отдельной. "
        "Ни одна исходная NPZ не переписана: назначения ролей хранятся в новом плане и применяются в памяти.", "",
        f"Выбранная схема использует {chosen['unique_train_episodes']} уникальных TRAIN-эпизодов "
        f"({chosen['physical_train_episodes']} физических + {chosen['replay_train_episodes']} повторных) и6 validation. "
        "Нормализация считается только на соответствующем TRAIN. Отбор использует прежние заранее заданные "
        "validation-критерии, худший recall ограниченного питания, indicator F1, raw F1, L2 и порядок кандидатов.", "",
        "После выбора модель и будущие файлы привязаны по SHA. Независимый новый физический test — группы "
        "5201/5202/5203, по шесть сцен, всего18 эпизодов. Начальные состояния и вариации задаются новыми RNG seeds. "
        "Ограниченные порции0,6/1,2/1,6 заданы до test. Все случаи входят в знаменатели; ошибки и отсутствие нужного эффекта не исключаются.", "",
        "## Парное сравнение голов и нейронные контроли", "",
        "Обе головы повторно оценены на **одинаковых новых X** из групп520x. Входной gain, калибровка, "
        "признаки и коннектом совпадают, поэтому это корректное парное сравнение считывателей. "
        "Ниже приведены именно новые оценки старой головы на520x, а не её исторический F1 на420x.", "",
        "| Условие | v3a raw F1 | Новая raw F1 | v3a indicator F1 | Новая indicator F1 |", "|---|---:|---:|---:|---:|"]
    for name in CONDITIONS:
        score=result["conditions"][name]
        baseline=result["baseline_conditions"].get(name)
        before_raw="—" if baseline is None else f"{baseline['raw']['f1']:.3f}"
        before_ind="—" if baseline is None else f"{baseline['indicator']['f1']:.3f}"
        lines.append(f"| {name} | {before_raw} | {score['raw']['f1']:.3f} | {before_ind} | {score['indicator']['f1']:.3f} |")
    lines += ["", "`homeostasis_off` удаляет искусственный вход голода; `transmission_off` отключает передачу "
        "внутренних связей в памяти процесса. Контроли удерживают записанные телесные и сенсорные входы "
        "постоянными: свободное тело после вмешательства здесь не моделируется. Перестановка использует целые "
        "эпизоды-доноры с выравниванием относительного времени. Постоянный baseline определяется только долей TRAIN. "
        "Головы на контролях не переобучаются.", "",
        "| Парный эффект на connected | Средняя разность F1 по seed | 95% bootstrap по seed |", "|---|---:|---|"]
    for metric in ("raw","indicator"):
        row=result["paired_refit_minus_v3a"]["connected"][metric]
        lines.append(f"| {metric} | {row['mean_seed_f1_difference']:.4f} | {row['paired_seed_bootstrap_95pct']} |")
    lines += ["", "Три seed-группы дают только предварительную оценку. Интервалы сгруппированы по seed; "
        "нейронные кадры не считаются независимыми повторениями. Общий F1 и средняя парная разность по seed — разные агрегаты.", "",
        "## Критерии и фактическое питание", "", "| Критерий | Предел | Результат |", "|---|---|---|"]
    for name,passed in result["checks"].items():lines.append(f"| {name} | {learning.ACCEPTANCE[name]} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "Учитель: включение hunger≥0,70, снятие hunger≤0,55. Индикатор: score≥0,70 /score≤0,40, "
        "два подтверждающих отсчёта по100ms. Устойчивое снятие требует500ms без повторного включения; допустимая "
        "задержка≤1000ms. Пропуск первоначальной потребности и раннее выключение не считаются успешным снятием.", "",
        "Дополнительно перед включением диагностической демонстрации проверяется фактическое потребление "
        "≥95% каждой ограниченной порции. Это отдельная проверка физического покрытия, не изменение F1 или основных критериев.", "",
        "| Порция | Съедено | Остаточный голод | v3a recall | Новый recall | Покрытие |", "|---:|---:|---:|---:|---:|---|"]
    current_cases={c["episode_id"]:c for c in current["cases"]}
    for c in evidence["limited_coverage"]["cases"]:
        new=current_cases[c["episode_id"]];before=old_cases[c["episode_id"]]
        lines.append(f"| {c['planned_portion']:g} | {c['actual_intake']:.4f} | {c['final_hunger']:.4f} | "
            f"{before['indicator']['recall']:.3f} | {new['indicator']['recall']:.3f} | {'PASS' if c['coverage_pass'] else 'FAIL'} |")
    lines += ["", "| Итог достаточного кормления / ошибок | v3a на новых X | Новая голова |", "|---|---|---|"]
    for key in ("feeding_cases_with_release","supported_release_latency_ms","initial_high_misses","false_activation_events","maximum_false_active_fraction_on_sated"):
        lines.append(f"| {key} | {old[key]} | {current[key]} |")
    if evidence["release_boundary_diagnostics"]:
        lines += ["", "Неудачные снятия разобраны отдельно; раннее выключение не становится успехом после изменения допуска.", "",
            "| Эпизод | Исчезновение потребности, ms | Предыдущее выключение, ms | Опережение, ms |", "|---|---:|---:|---:|"]
        for boundary in evidence["release_boundary_diagnostics"]:
            lines.append(f"| {boundary['episode_id']} | {boundary['target_off_ms']:g} | {boundary['previous_neural_off_ms']} | {boundary['early_by_ms']} |")
        lines += ["", "Даже опережение на один нейронный шаг остаётся ошибкой по исходному правилу. "
            "Порог времени после просмотра test не изменялся. Критерий требует реального перехода "
            "из включённого состояния в выключенное после target-off и500ms последующей устойчивости."]
    lines += ["", "## Все физические случаи", "", "| Эпизод | Съедено | v3a F1 | Новый F1 | Новый recall | Пропуск начального сигнала | Ложные включения | Снятие / задержки, ms |", "|---|---:|---:|---:|---:|---|---:|---|"]
    for c in current["cases"]:
        delays=[m["stable_latency_ms"] for m in c["release_matches"]]
        release=f"{c['release_supported']} / {delays}" if c["feeding_case"] else "не ожидается"
        lines.append(f"| {c['episode_id']} | {c['intake']:.3f} | {old_cases[c['episode_id']]['indicator']['f1']:.3f} | "
            f"{c['indicator']['f1']:.3f} | {c['indicator']['recall']:.3f} | {c['initial_high_missed']} | {c['false_activation_events']} | {release} |")
    lines += ["", "F1 отдельной полностью сытой сцены равен0 по принятому соглашению при отсутствии положительного класса. "
        "Для таких сцен содержательная проверка — ложные включения и доля ложной активности, а не F1 отдельно.", "",
        "## Все шесть кандидатов", "", "| Схема | L2 | TRAIN эпизодов | Validation raw F1 | Indicator F1 | Мин. limited recall | Все критерии |", "|---|---:|---:|---:|---:|---:|---|"]
    for c in evidence["candidates"]:
        lines.append(f"| {c['regime']} | {c['l2']:g} | {c['unique_train_episodes']} | {c['validation']['raw']['f1']:.3f} | "
            f"{c['validation']['indicator']['f1']:.3f} | {c['assessment']['minimum_limited_case_recall']:.3f} | {all(c['assessment']['checks'].values())} |")
    lines += ["", "## Проверки и ограничения", "",
        "Проверены403 старых файла, обе привязки и все54 test-записи завершённого v3a, его отчёт/графики/инструкция, "
        "точные источники42 новых development-записей и шесть кандидатов. Все54 новых test-артефакта привязаны к плану; "
        "обе головы, все метрики и парные интервалы повторно вычислены из X. Живые scores и полные списки событий "
        "снятия/подтверждения совпадают с точным построчным вызовом новой модели во всех18 физических эпизодах.", "",
        "Первичный вход голода — искусственная инженерная стимуляция32 клеток; получатели не объявляются естественным "
        "центром голода. Признаки не содержат прямо стимулируемые клетки. Обнаруженное ранее насыщение примерно22% "
        "выбранных клеток сохраняется; новая голова не исправляет модель LIF. Внешние трассы50/200ms остаются памятью "
        "считывателя, score не объявляется калиброванной вероятностью. Успех ограниченного пилота не доказывает биологическое "
        "понимание или общую коммуникацию мухи.", "",
        "Подсказки направления, другие потребности, новые моторные политики, пластичность коннектома и широкая "
        "проверка длительной жизни не входят в этот этап. Закрытая биологическая ветка не возобновляется.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m semantic_tools.v3_refit_report --help", "```", "",
        "Генератор читает готовые финальные артефакты, не собирает данные, не обучает и не включает запуск. "
        "Повторный рендер требует нового каталога отчёта. Повторная подстройка после520x требует ещё одного нового test."]
    return "\n".join(lines)+"\n"


def run(root,output,promotion=None,docs=None):
    root,output=Path(root),Path(output)
    names=("results.md","overall.json","summary.png","summary.svg","summary.npz")
    if any((output/name).exists() for name in names):raise FileExistsError("Completed refit report is immutable; choose new output")
    if docs is not None and Path(docs).resolve()!=(ROOT/"docs/SEMANTIC_CHANNEL_V3_REFIT.md").resolve():
        raise ValueError("Only the new separate refit guide can be finalized")
    evidence=verify_evidence(root)
    promotion_value=prior_report.promotion_status(promotion,evidence)
    output.mkdir(parents=True,exist_ok=True)
    plot(evidence,output)
    with (output/"results.md").open("x",encoding="utf-8") as stream:stream.write(render(evidence,promotion_value))
    overall={"schema_version":1,"results":evidence["results"],"selection":evidence["selection"],
        "parent_guard":evidence["parent_guard"],"independent_binding_guard":evidence["independent_binding_guard"],
        "limited_coverage":evidence["limited_coverage"],"live_checks":evidence["live_checks"],"promotion":promotion_value,
        "release_boundary_diagnostics":evidence["release_boundary_diagnostics"],
        "candidate_count_verified":6,"new_test_recordings_verified":54,"paired_same_new_X":True,
        "generator_sha256":file_digest(Path(__file__)),
        "artifacts":{name:file_digest(output/name) for name in names if name!="overall.json"}}
    with (output/"overall.json").open("x",encoding="utf-8") as stream:json.dump(overall,stream,indent=2,ensure_ascii=False,allow_nan=False)
    if docs is not None:
        new=evidence["results"]["conditions"]["connected"];old=evidence["results"]["baseline_conditions"]["connected"]
        lines=["# NEED_FOOD: отдельное переобучение на физической жизни", "",
            f"Статус: **{evidence['results']['status']} / {promotion_value['status']}**.", "",
            f"На18 новых физических эпизодах indicator F1 составляет **{new['indicator']['f1']:.3f}**; "
            f"у сохранённой головы v3a на **тех же новых X** — {old['indicator']['f1']:.3f}. "
            f"Выбрана схема `{evidence['selection']['regime']}`, L2={evidence['selection']['selected_l2']:g}; gain остаётся5mV.", "",
            f"Покрытие фактически съеденных ограниченных порций: **{'PASS' if evidence['limited_coverage']['all_pass'] else 'FAIL'}**. "
            f"Устойчивое снятие после достаточной еды: {new['feeding_cases_with_release']}/{new['feeding_cases']}; "
            f"задержки, ms: {new['supported_release_latency_ms']}.", "",
            "После ограниченных порций: "+", ".join(f"{c['planned_portion']:g} — recall {next(case for case in new['cases'] if case['episode_id']==c['episode_id'])['indicator']['recall']:.3f}"
                for c in evidence["limited_coverage"]["cases"])+". Все случаи остаются в оценке, включая ошибки достаточного кормления.", "",
            "[Полный отчёт: все случаи, контроли, шесть кандидатов и ограничения](../reports/semantic_channel_v3_refit/results.md).", "",
            "Прежние4201/4202 стали TRAIN,4203 — validation. Это виденные данные разработки. Только новые5201/5202/5203 "
            "образуют независимый test этого этапа. [Отрицательный v3a](SEMANTIC_CHANNEL_V3.md) сохранён отдельно. "
            "Модель читает только нейронные признаки; граф, gain и физические механизмы сохранены. "
            "Это предварительный инженерный результат, не восстановленная биологическая модель.", "",
            "Генератор: `.\\.venv\\Scripts\\python.exe -m semantic_tools.v3_refit_report --help`. Он не обучает и не включает демонстрацию."]
        if promotion_value["status"] in ("DIAGNOSTIC_ENABLED","PROMOTED"):
            launcher=Path(promotion_value["launcher_path"]).name
            lines += ["", f"Отдельный диагностический запуск: [{launcher}](../{launcher}). Прежний запуск18 сохранён."]
        else:
            lines += ["", "Отдельный запуск19 и переключение прежнего режима не добавлены. Для явного исследовательского "
                "просмотра этой головы можно использовать существующее консольное меню; оно запускается на паузе:", "",
                "```powershell",
                f'.\\.venv\\Scripts\\python.exe -m fly_semantic demo --model "{root.as_posix()}/state-readout.npz" '
                f'--calibration "{Path(evidence["plan"]["calibration_dir"]).as_posix()}" --interactive '
                f'--output "{root.as_posix()}/experimental-demo"',
                "```", "", "Это явный исследовательский запуск модели с указанными ограничениями, а не перевод в штатный режим."]
        if evidence["release_boundary_diagnostics"]:
            lines += ["", "Неудачные снятия сигнала:"]
            for boundary in evidence["release_boundary_diagnostics"]:
                lines.append(f"- `{boundary['episode_id']}`: выключение {boundary['previous_neural_off_ms']}ms при "
                    f"target-off {boundary['target_off_ms']:g}ms; опережение {boundary['early_by_ms']}ms.")
            lines += ["", "Раннее выключение остаётся ошибкой, даже если опережает target-off всего на10ms. Исходный критерий не смягчался после test."]
        Path(docs).write_text("\n".join(lines)+"\n",encoding="utf-8")
    return overall


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=refit.ROOT)
    parser.add_argument("--output",type=Path,default=Path("reports/semantic_channel_v3_refit"))
    parser.add_argument("--promotion",type=Path)
    parser.add_argument("--docs",type=Path)
    args=parser.parse_args()
    result=run(args.root,args.output,args.promotion,args.docs)
    print(json.dumps({"status":result["results"]["status"],"promotion":result["promotion"]["status"],
        "limited_coverage_pass":result["limited_coverage"]["all_pass"]},allow_nan=False))
