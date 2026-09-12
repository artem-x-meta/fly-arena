"""Render the physical NEED_FOOD study only from complete, bound evidence.

No simulations or training run here. Saved neural feature matrices are scored
again with both frozen heads to verify the reported same-trajectory comparison.
Reports cannot claim a passed pilot unless candidate and preservation guards
pass; promotion to a separate demo is an independent, explicit artifact.
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
from semantic_tools import physical_learning as learning
from semantic_tools import v2_integrity as integrity


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _same(expected, actual, message):
    if digest(expected) != digest(actual):
        raise ValueError(message)


def validation_history(study_dir):
    """Retain an explicitly rejected validation-only recipe without testing it."""
    alternative = Path(study_dir).parent / "mixed-study"
    if not (alternative / "selection.json").exists():
        return []
    plan = read_json(alternative / "plan.json")
    learning.validate_plan(plan)
    selected = learning._verify_selection(alternative, plan)
    decision = read_json(alternative / "candidate-decision.json")
    if (decision.get("status") != "NOT_SELECTED_ON_VALIDATION" or decision.get("test_seen") is not False
            or not plan.get("include_synthetic_train") or (alternative / "results.json").exists()):
        raise ValueError("Alternative recipe history is not an untested validation rejection")
    return [{"path": str(alternative), "plan_digest": plan["digest"], "model_sha256": selected["model_sha256"],
             "selection_sha256": file_digest(alternative / "selection.json"), "selected_l2": selected["selected_l2"],
             "supplemental_training_episodes": len(plan["supplemental_train"]), "validation": selected["validation"],
             "decision": decision}]


def verify_evidence(study_dir, binding_path):
    """Fail closed on changed files, incomplete trials or unreproducible metrics."""
    study_dir, binding_path = Path(study_dir), Path(binding_path)
    plan = read_json(study_dir / "plan.json")
    learning.validate_plan(plan)
    binding = read_json(binding_path)
    guard = integrity.check_candidate(binding_path)
    if not guard["pass"] or guard["preservation"]["checked_files"] != 254:
        raise ValueError("Report needs the intact candidate and all 254 protected files")
    roles = binding["artifacts"]
    for role, path in (("model", study_dir / "state-readout.npz"), ("study_plan", study_dir / "plan.json"),
                       ("mapping", Path(plan["calibration_dir"]) / "mapping.json"),
                       ("calibration", Path(plan["calibration_dir"]) / "calibration.json")):
        if role not in roles or (ROOT / roles[role]["path"]).resolve() != path.resolve() or roles[role]["sha256"] != file_digest(path):
            raise ValueError(f"Candidate binding has the wrong {role} artifact")
    selected = learning._verify_selection(study_dir, plan)
    training, result = read_json(study_dir / "training.json"), read_json(study_dir / "results.json")
    if (result["plan_digest"] != plan["digest"] or result["selection_digest"] != selected["digest"]
            or result["model_sha256"] != selected["model_sha256"] or result["test_tuned"] is not False
            or result["all_planned_episodes_included"] is not True
            or training["selection_digest"] != selected["digest"] or training["model_sha256"] != selected["model_sha256"]):
        raise ValueError("Training, selection and evaluation identity mismatch")
    calibration_dir = Path(plan["calibration_dir"])
    mapping, calibration = read_json(calibration_dir / "mapping.json"), read_json(calibration_dir / "calibration.json")
    if (mapping["digest"] != plan["mapping_digest"] or calibration["digest"] != plan["calibration_digest"]
            or calibration["mapping_digest"] != mapping["digest"]):
        raise ValueError("Frozen input/feature calibration identity mismatch")
    split_rows = {split: learning.load_rows(study_dir, plan, split) for split in ("train", "validation", "test")}
    recorded = {"connected": split_rows["test"]}
    for condition in plan["collection_conditions"]:
        if condition != "connected":
            recorded[condition] = learning.load_rows(study_dir, plan, "test", condition)
    paths = [Path(row["path"]) for rows in split_rows.values() for row in rows]
    paths += [Path(row["path"]) for key, rows in recorded.items() if key != "connected" for row in rows]
    collection_guard = integrity.validate_collection_artifacts(plan, ROOT / "data/graph", episodes=paths)
    absent = {(ROOT / name).resolve() for name in binding["evaluation_outputs_absent_at_binding"]}
    test_paths = {Path(row["path"]).resolve() for rows in recorded.values() for row in rows}
    if not test_paths | {(study_dir / "results.json").resolve()} <= absent:
        raise ValueError("Candidate was not prospectively bound before every required test artifact")
    manifest_paths = {Path(item["path"]).resolve(): item["sha256"] for item in result["test_data"]}
    if len(manifest_paths) != len(result["test_data"]) or set(manifest_paths) != test_paths:
        raise ValueError("Evaluation omitted or duplicated planned test artifacts")
    if any(file_digest(path) != manifest_paths[path] for path in test_paths):
        raise ValueError("Test data changed after evaluation")
    baseline_path = Path(plan["baseline_model_path"])
    if file_digest(baseline_path) != result["baseline_model_sha256"] or result["baseline_model_sha256"] != plan["baseline_model_sha256"]:
        raise ValueError("Historical baseline model changed")
    candidate = LinearReadout.load(study_dir / "state-readout.npz", expected_feature_digest=plan["feature_digest"])
    baseline = LinearReadout.load(baseline_path, expected_feature_digest=plan["feature_digest"])
    shuffled, permutation = learning._shuffle_rows(recorded["connected"], plan.get("shuffle_seed", 81817))
    conditions = recorded | {"cross_episode_shuffle": shuffled}
    scores = {key: learning.score_rows(candidate, rows, plan) for key, rows in conditions.items()}
    old_scores = {key: learning.score_rows(baseline, rows, plan) for key, rows in conditions.items()}
    constant = learning._ConstantHead(plan["feature_digest"], float(selected["train_positive_fraction"] >= .5))
    scores["train_constant"] = learning.score_rows(constant, recorded["connected"], plan)
    _same(result["conditions"], scores, "Candidate report metrics differ from saved neural features")
    _same(result["baseline_conditions"], old_scores, "Baseline report metrics differ from the same saved neural features")
    _same(result["shuffle_permutation"], permutation, "Reported donor permutation differs")
    _same(result["paired_candidate_minus_baseline"],
        {key: learning.paired_seed_bootstrap(scores[key], old_scores[key]) for key in conditions},
        "Reported paired baseline uncertainty differs from seed groups")
    _same(result["paired_connected_minus_control"],
        {key: learning.paired_seed_bootstrap(scores["connected"], value) for key, value in scores.items() if key != "connected"},
        "Reported paired control uncertainty differs from seed groups")
    acceptance = learning.assess_acceptance(scores["connected"], scores, plan, old_scores["connected"])
    _same({"checks": result["checks"], "status": result["status"]}, acceptance, "Report status differs from frozen criteria")
    input_diagnostics = []
    for row in recorded["connected"]:
        if row["metadata"]["episode"]["scene"] in ("limited_meal", "meal_fast", "meal_slow"):
            with np.load(row["path"], allow_pickle=False) as archive:
                input_diagnostics.append({"episode_id": row["metadata"]["episode"]["episode_id"],
                    "last_second_input_mean_hz": float(archive["diagnostic_input_rates"][-100:].mean())})
    smoke_path = Path(study_dir).parent / "verification/model-smoke.json"
    smoke = read_json(smoke_path) if smoke_path.exists() else None
    if smoke is not None and (smoke.get("status") != "PASS" or smoke.get("model_sha256") != selected["model_sha256"]
            or not smoke.get("checks") or not all(value is True for value in smoke["checks"].values())):
        raise ValueError("Current candidate integration smoke identity/status mismatch")
    return {"plan": plan, "selection": selected, "training": training, "results": result,
            "binding_guard": guard, "collection_guard": collection_guard, "split_rows": split_rows,
            "recorded": recorded, "mapping": mapping, "calibration": calibration,
            "binding_path": str(binding_path), "study_dir": str(study_dir), "validation_history": validation_history(study_dir),
            "input_diagnostics": input_diagnostics, "integration_smoke": smoke,
            "integration_smoke_sha256": None if smoke is None else file_digest(smoke_path)}


def promotion_status(path, evidence):
    if path is None:
        return {"status": "PENDING", "reason": "Решение о включении отдельной демонстрации ещё не записано."}
    value = read_json(path)
    if value.get("status") not in ("PROMOTED", "NOT_PROMOTED", "PENDING"):
        raise ValueError("Unknown demo promotion status")
    if value.get("model_sha256") != evidence["selection"]["model_sha256"]:
        raise ValueError("Demo promotion references a different model")
    if value["status"] == "PROMOTED":
        if evidence["results"]["status"] != "PRELIMINARY_TARGET_REACHED":
            raise ValueError("A failed pilot cannot be advertised as a passed promoted demo")
        config = Path(value["config_path"])
        if file_digest(config) != value["config_sha256"]:
            raise ValueError("Promoted demo config changed")
        with config.open("rb") as stream:
            settings = tomllib.load(stream)["semantic"]
        model = (config.parent / settings["model"]).resolve()
        calibration = (config.parent / settings["calibration"]).resolve()
        if (settings["enabled"] is not True or file_digest(model) != value["model_sha256"]
                or calibration != Path(evidence["plan"]["calibration_dir"]).resolve()):
            raise ValueError("Promoted demo config does not use the evaluated model/calibration")
    return value


def _latencies(case):
    return ", ".join("нет" if item["stable_latency_ms"] is None else f"{item['stable_latency_ms']:g}" for item in case["release_matches"]) or "нет target-off"


def _plot(evidence, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = evidence["results"]
    meals = [case for case in result["conditions"]["connected"]["cases"]
             if case["feeding_case"] or case["scene"] == "limited_meal"]
    previous = {case["episode_id"]: case for case in result["baseline_conditions"]["connected"]["cases"]}
    rows = {row["metadata"]["episode"]["episode_id"]: row for row in evidence["recorded"]["connected"]}
    chart_rows = 1 + (len(meals) + 1) // 2
    fig = plt.figure(figsize=(13, 3.1 * chart_rows), constrained_layout=True)
    grid = fig.add_gridspec(chart_rows, 2)
    ax = fig.add_subplot(grid[0, :])
    names = ["connected", "homeostasis_off", "transmission_off", "cross_episode_shuffle"]
    positions = np.arange(len(names))
    old = np.array([result["baseline_conditions"][name]["indicator"]["f1"] for name in names])
    new = np.array([result["conditions"][name]["indicator"]["f1"] for name in names])
    ax.bar(positions - .19, old, .38, label="v1 head", color="#8c6cbb")
    ax.bar(positions + .19, new, .38, label="physical head", color="#197b95")
    ax.axhline(evidence["plan"]["indicator_target_f1"], color="black", linestyle=":", linewidth=1)
    ax.set(xticks=positions, xticklabels=["Physical input", "Hunger port off", "Transmission off", "Episode shuffle"],
           ylabel="Indicator F1", ylim=(0, 1.05), title="Same recorded trajectories; replay controls hold body inputs fixed")
    ax.legend(loc="upper right")
    arrays = {"condition_names": np.asarray(names), "old_indicator_f1": old, "new_indicator_f1": new}
    for index, meal in enumerate(meals):
        ax = fig.add_subplot(grid[1 + index // 2, index % 2])
        old_case, actual = previous[meal["episode_id"]], rows[meal["episode_id"]]
        time = np.asarray(meal["time_ms"]) / 1000
        ax.plot(time, old_case["scores"], color="#8c6cbb", alpha=.8, label="v1 score")
        ax.plot(time, meal["scores"], color="#197b95", alpha=.8, label="new score")
        ax.step(time, meal["targets"], where="post", color="black", linewidth=1.3, label="Need target")
        ax.step(time, np.asarray(old_case["indicator_samples"]) * .10 - .18, where="post", color="#8c6cbb", label="v1 indicator")
        ax.step(time, np.asarray(meal["indicator_samples"]) * .10 - .34, where="post", color="#197b95", label="new indicator")
        outcome = (f"new stable release {meal['release_supported']}" if meal["feeding_case"] else
                   f"need remains; new indicator recall {meal['indicator']['recall']:.3f}")
        ax.set(xlabel="Simulation time (s)", ylabel="Score / target", ylim=(-.38, 1.04),
               title=f"{meal['episode_id']}\nActual intake {meal['intake']:.3f}; {outcome}")
        ax.set_yticks([0, .5, 1])
        if index == 0:
            ax.legend(loc="upper right", fontsize=7)
        prefix = f"meal_{index}_"
        arrays.update({prefix + "episode_id": np.asarray(meal["episode_id"]), prefix + "scene": np.asarray(meal["scene"]),
            prefix + "release_expected": np.asarray(meal["feeding_case"]), prefix + "time_ms": np.asarray(meal["time_ms"]),
            prefix + "old_scores": np.asarray(old_case["scores"]), prefix + "new_scores": np.asarray(meal["scores"]),
            prefix + "targets": np.asarray(meal["targets"]), prefix + "old_indicator": np.asarray(old_case["indicator_samples"]),
            prefix + "new_indicator": np.asarray(meal["indicator_samples"]), prefix + "actual_intake": actual["intake"]})
    fig.savefig(destination / "physical-head-comparison.png", dpi=160)
    fig.savefig(destination / "physical-head-comparison.svg")
    plt.close(fig)
    arrays["metadata"] = np.asarray(json.dumps({"plan_digest": evidence["plan"]["digest"],
        "model_sha256": result["model_sha256"], "baseline_model_sha256": result["baseline_model_sha256"]}))
    np.savez_compressed(destination / "physical-head-comparison.npz", **arrays)


def render_report(evidence, promotion, output):
    plan, selection, result = evidence["plan"], evidence["selection"], evidence["results"]
    current, old = result["conditions"]["connected"], result["baseline_conditions"]["connected"]
    splits = Counter(ep["split"] for ep in plan["episodes"])
    seeds = sorted({ep["seed_group"] for ep in plan["episodes"] if ep["split"] == "test"})
    prior = {case["episode_id"]: case for case in old["cases"]}
    physical = {row["metadata"]["episode"]["episode_id"]: row for row in evidence["recorded"]["connected"]}
    table = []
    for name, value in result["conditions"].items():
        baseline = result["baseline_conditions"].get(name)
        table.append(f"| {name} | {baseline['raw']['f1']:.3f} | {value['raw']['f1']:.3f} | {baseline['indicator']['f1']:.3f} | {value['indicator']['f1']:.3f} |" if baseline else
                     f"| {name} | — | {value['raw']['f1']:.3f} | — | {value['indicator']['f1']:.3f} |")
    meals = []
    for case in current["cases"]:
        if case["feeding_case"]:
            baseline = prior[case["episode_id"]]
            meals.append(f"| {case['episode_id']} | {case['intake']:.4f} | {'да' if case['release_matches'] else 'нет'} | {'да' if baseline['release_supported'] else 'нет'} | {_latencies(baseline)} | {'да' if case['release_supported'] else 'нет'} | {_latencies(case)} |")
    case_table = []
    for case in current["cases"]:
        row = physical[case["episode_id"]]
        need = row["metadata"].get("final_hunger", float(row["hunger"][-1]))
        case_table.append(f"| {case['episode_id']} | {need:.3f} | {case['intake']:.3f} | {sum(case['targets'])}/{len(case['targets'])} | "
            f"{prior[case['episode_id']]['indicator']['recall']:.3f} | {case['indicator']['recall']:.3f} | {case['indicator']['fn']} |")
    limited = [case for case in current["cases"] if case["scene"] == "limited_meal"]
    timely_releases = sum(case["feeding_case"] and case["release_supported"] and
        all(item["stable_latency_ms"] is not None and item["stable_latency_ms"] <= plan["release_max_latency_ms"]
            for item in case["release_matches"]) for case in current["cases"])
    diagnostics = {item["episode_id"]: item for item in evidence.get("input_diagnostics", [])}
    limited_text = ""
    if limited:
        limited_text = ("Особый отрицательный результат дают заранее включённые сцены `limited_meal`. "
            "После маленькой порции сохраняется высокая потребность; здесь правильный сигнал должен оставаться включённым. "
            "Показатель recall отражает долю истинно активных отсчётов, которую индикатор действительно показал.\n\n")
        limited_text += "\n".join(f"- `{case['episode_id']}`: съедено {case['intake']:.3f}, голод в конце "
            f"{physical[case['episode_id']]['metadata'].get('final_hunger', float(physical[case['episode_id']]['hunger'][-1])):.3f}; "
            f"recall индикатора v1 **{prior[case['episode_id']]['indicator']['recall']:.3f}**, v2 **{case['indicator']['recall']:.3f}** "
            f"({case['indicator']['fn']} пропущенных активных отсчётов из {sum(case['targets'])})." for case in limited)
        limited_text += ("\n\nПоэтому снятие сигнала после достаточной еды нельзя переносить на любое кормление. "
            "Голова должна одновременно удерживать сигнал при оставшемся голоде. Возможное несоответствие порога "
            "искусственного нейронного входа шкале голода — отдельная гипотеза для будущей проверки; "
            "этот опыт не менял вход и не устанавливает причину ошибки.")
        if all(case["episode_id"] in diagnostics for case in limited):
            measured = ", ".join(f"{diagnostics[case['episode_id']]['last_second_input_mean_hz']:.2f}" for case in limited)
            limited_text += (f" В частности, средняя частота непосредственно стимулируемых клеток за последнюю секунду "
                f"этих эпизодов равна **{measured} Гц**: входная активность не исчезла. "
                "Оценка только постоянного внешнего тока не учитывает рекуррентные токи полного графа и не доказывает потерю информации.")
    intake = [float(row["intake"][-1]) for row in evidence["recorded"]["connected"]]
    static = result["paired_candidate_minus_baseline"]["connected"]["indicator"]
    limit_checks = "\n".join(f"| {key} | {'PASS' if passed else 'FAIL'} |" for key, passed in result["checks"].items())
    validation = selection["validation"]
    history = (f"Физический кандидат уже на validation имел raw F1={validation['raw']['f1']:.3f}, "
        f"indicator F1={validation['indicator']['f1']:.3f}, снятие {validation['feeding_cases_with_release']}/{validation['feeding_cases']} "
        f"и максимальную ложную активность в сытых сценах {validation['maximum_false_active_fraction_on_sated']:.3f}. "
        "Удачное выключение после двух кормлений само по себе не означало готовности всего канала.")
    for trial in evidence.get("validation_history", []):
        measured = trial["validation"]
        history += (f"\n\nДо открытия test была рассмотрена отдельная смешанная схема: те же физические train/validation "
            f"и {trial['supplemental_training_episodes']} старых синтетических эпизода только из train v1. "
            f"Она выбрала L2={trial['selected_l2']:g}, получила validation raw F1={measured['raw']['f1']:.3f}, "
            f"indicator F1={measured['indicator']['f1']:.3f}, снятие {measured['feeding_cases_with_release']}/{measured['feeding_cases']} "
            f"и ложную активность в сытых сценах {measured['maximum_false_active_fraction_on_sated']:.3f}. "
            f"Смешанная схема не улучшила основной validation indicator F1 и была отклонена; её test не запускался. "
            f"Артефакты сохранены отдельно в `{trial['path']}`. Для итогового test осталась исходная физическая голова "
            "с лучшим validation indicator F1. Дополнительной настройки по этому test не проводилось.")
    if promotion["status"] == "PROMOTED":
        demo = (f"Отдельная демонстрация включена конфигурацией `{promotion['config_path']}`. Она использует проверенную голову; старый запуск остаётся самостоятельным артефактом.\n\n"
                f"```powershell\n.\\.venv\\Scripts\\python.exe -m fly_semantic demo --semantic-config {promotion['config_path']} --interactive --output runs/semantic-v2/demo\n```")
    else:
        explanation = ("Качество и максимальная задержка не выполнили заданные критерии; ограниченная порция оставляет необнаруженный голод."
                       if promotion["status"] == "NOT_PROMOTED" else promotion.get("reason", ""))
        demo = f"Статус отдельной демонстрации: **{promotion['status']}**. {explanation} Отдельный штатный запуск v2 не добавлен; явное подключение модели остаётся исследовательским действием, описанным в `docs/SEMANTIC_CHANNEL_V2.md`."
    conclusion = "Предварительные критерии достигнуты." if result["status"] == "PRELIMINARY_TARGET_REACHED" else "Предварительные критерии не достигнуты; модель остаётся экспериментальной."
    promotion_argument = f" --promotion-json {evidence['promotion_path']}" if evidence.get("promotion_path") else ""
    smoke_text = ("Отдельный smoke с выбранной головой подтвердил точное совпадение живых признаков с записью на 300 мс "
        "и после восстановления полной контрольной точки ещё на 400 мс. Это проверка подключения и восстановления состояния; "
        "она не отменяет неудачные критерии качества." if evidence.get("integration_smoke") is not None else
        "Отдельная свежая проверка живого подключения выбранной головы в этот отчёт не включена.")
    return f"""# Физическое обучение сигнала «Нужна еда»: v2

{conclusion} Статус сохранённого теста: **{result['status']}**. Старая и новая головы прочитали одни и те же нейронные признаки одной и той же физической жизни. На connected-test raw F1: **{old['raw']['f1']:.3f} → {current['raw']['f1']:.3f}**, F1 видимого индикатора: **{old['indicator']['f1']:.3f} → {current['indicator']['f1']:.3f}**. Устойчивое выключение после достаточной еды: **{old['feeding_cases_with_release']}/{old['feeding_cases']} → {current['feeding_cases_with_release']}/{current['feeding_cases']}**. Эта доля не включает ограничение по максимальной задержке: оно проверяется отдельно. Полноценный сигнал также обязан сохраняться после недостаточной порции; эти отрицательные случаи показаны наравне с удачными кормлениями.

## Что изменено и за счёт чего

Обучены только коэффициенты небольшого линейного считывателя NEED_FOOD: {selection['trained_parameters']} весов и смещений, плюс нормализация по train. На вход головы поступает `X`, то есть причинные следы активности внутренних нейронов. Энергия, насыщение, количество пищи, идентификатор сцены и целевая метка доступны только сборщику/оценщику. Они не являются признаками модели. Голод организма продолжает поступать через ранее откалиброванный искусственный вход; его амплитуда, адреса клеток, динамика Brain, коннектом и признаки не изменены.

Новое обучение использует настоящие переходы тела при кормлении. Пища попадает внутрь через существующий контакт ротового аппарата, поведение FEED, транзакцию ресурса и пищеварение. Голова лишь сообщает результат и не управляет ногами, кормлением, навигацией или временем остановки еды. В test {sum(value > 0 for value in intake)} из {len(intake)} сцен имели фактическое поступление пищи; нулевое поступление в сценах без еды ожидаемо и не исключается.

## Данные и выбор модели

Всего {len(plan['episodes'])} полных физических эпизодов: **{splits['train']} train / {splits['validation']} validation / {splits['test']} test**. Внутри эпизода строки не разделяются между выборками; группы seed не пересекаются. Сцены: голод вдали от пищи, сытость вдали, быстрое и медленное достаточное кормление, ограниченная порция и сытая муха рядом с едой. Длительности и начальные состояния заданы до сбора.

Рассмотрены только L2 `{plan['l2_grid']}`. Выбран **L2={selection['selected_l2']:g}**: максимальный validation indicator F1, затем raw F1, затем более сильная регуляризация. Нормализация обучена только на train, синтетические эпизоды v1 не примешивались. Пороги raw={plan['score_threshold']}, indicator on={plan['indicator_on']}/off={plan['indicator_off']}, подтверждение {plan['confirm_samples']} отсчётами с периодом {plan['sample_ms']} мс фиксированы заранее. Шкала sigmoid — оценка модели, а не калиброванная вероятность голода.

Перед test зафиксированы модель, план, mapping, калибровка, train/validation-артефакты и исходники; все будущие test-выходы тогда отсутствовали. Генератор отчёта повторно проверил эту привязку, прочитал все запланированные файлы и заново получил приведённые выходы из сохранённого `X`.

{history}

## Сравнение и контроли

| Условие | v1 raw F1 | v2 raw F1 | v1 indicator F1 | v2 indicator F1 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

`homeostasis_off` убирает только дополнительный искусственный вход голода. Обычные сенсорные и модулирующие пути могут сохранять косвенную информацию о состоянии. `transmission_off` обнуляет межнейронную передачу в отдельном экземпляре Brain. Оба контроля воспроизводят записанные физические сенсорные входы при фиксированной траектории тела: **это нейронный replay, а не новая свободная жизнь мухи после вмешательства**. Это различие ограничивает выводы о поведении тела.

`cross_episode_shuffle` переставляет целые эпизоды-доноры и монотонно сопоставляет их относительное время ближайшими отсчётами; этот искусственный контроль не является физической траекторией или критерием приёмки. `train_constant` всегда выдаёт класс большинства train. Прямые входные частоты `diagnostic_input_rates` сохранены отдельно для диагностики и не входят в признаки головы.

## Что означает «перестала просить еду»

Эталон включается при голоде ≥{plan['teacher_on']} и выключается при ≤{plan['teacher_off']}. Метка каждого отсчёта соответствует состоянию перед последним шагом {plan['label_reference_offset_ms']} мс; признаки снимаются после него. Сам индикатор работает только от нейронного score через фиксированный автомат с гистерезисом.

Успешное снятие требует фактического поступления пищи, настоящего target-off, активного индикатора непосредственно перед этим моментом и последующего нейронного перехода True→False, сохраняющегося не менее **{plan['release_stable_ms']} мс** в том же интервале отсутствия потребности. Раннее выключение, постоянное молчание, пропущенный голод и выключение после нового target-on не засчитываются. В знаменателе остаются все заранее назначенные достаточные кормления, даже если пища не попала внутрь или target-off не наступил. Ограниченная порция заранее не относится к случаям, где требуется исчезновение голода.

| Достаточное кормление | Реально съедено | target-off | v1 снятие | v1 задержка, мс | v2 снятие | v2 задержка, мс |
|---|---:|---|---|---|---|---|
{chr(10).join(meals)}

Устойчивое снятие уложилось в заранее заданные {plan['release_max_latency_ms']} мс в **{timely_releases}/{current['feeding_cases']}** достаточных кормлений. Максимальная наблюдённая задержка новой головы — **{current['maximum_supported_release_latency_ms']} мс**; наличие перехода и выполнение ограничения времени оцениваются отдельно.

Голод в начале пропущен: v1 **{old['initial_high_misses']}/{old['initial_high_cases']}**, v2 **{current['initial_high_misses']}/{current['initial_high_cases']}**. Ложных включений: v1 **{old['false_activation_events']}**, v2 **{current['false_activation_events']}**. Максимальная доля ложного активного индикатора в сытых сценах: v1 **{old['maximum_false_active_fraction_on_sated']:.3f}**, v2 **{current['maximum_false_active_fraction_on_sated']:.3f}**.

{limited_text}

Все test-сцены, включая пропуски активной потребности:

| Эпизод | Голод в конце | Съедено | Истинно активные отсчёты | v1 recall | v2 recall | v2 пропуски |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(case_table)}

Recall в сценах без положительных меток по принятому правилу равен 0 и не означает пропуск потребности. Пропуски включают начальное ожидание подтверждения индикатора; оно не исключается из F1. Задержка после достаточного кормления и раннее выключение при сохраняющемся голоде — разные ошибки.

## Заранее заданные критерии

| Критерий | Результат |
|---|---|
{limit_checks}

Полный набор численных порогов сохранён в `plan.json → acceptance`. Неудачные условия, сцены и seed не исключались. Качество классификации само по себе не заменяет успешного выключения после еды.

## Неопределённость и ограничения

В test только **{len(seeds)} группы seed: {seeds}**. Среднее по этим группам парное изменение indicator F1 равно **{static['mean_seed_f1_difference']:+.3f}**, 95% bootstrap-интервал **[{static['paired_seed_bootstrap_95pct'][0]:+.3f}, {static['paired_seed_bootstrap_95pct'][1]:+.3f}]**. Пересэмплируются группы, не кадры; при двух группах этот интервал является описанием маленького пилота и не подтверждает широкую обобщаемость. Не выполнен большой протокол 5 seed × 100 test-эпизодов на условие.

Проверена одна поддерживаемая потребность и заранее заданные сцены. Нет обучения языка, свободного разговора, понимания намерений, доказательства субъективного состояния или биологически достоверного механизма голода. Входящие FOOD_LEFT/RIGHT/AHEAD, считыватели отдыха и чистки, новая моторная политика, память сообщений и пластичность коннектома не реализованы этим этапом. Закрытая биологическая ветка не возобновлялась.

Время отдельных сборов записано в метаданных. Этот этап не проводит новый парный замер RAM/VRAM и накладных расходов живого канала; старые показатели нельзя автоматически объявлять свежим ресурсным результатом v2.

## Сохранность и идентичность

Повторная проверка подтвердила неизменность **{evidence['binding_guard']['preservation']['checked_files']} защищённых файлов**, включая старые режимы, голову v1, конфигурации и контрольные точки. Проверены исходники сборщика и вычисление исходных весов графа. Результаты и изображения имеют следующие привязки:

{smoke_text}

| Артефакт | Идентификатор |
|---|---|
| План | `{plan['digest']}` |
| Модель v2 | `{result['model_sha256']}` |
| Модель v1 | `{result['baseline_model_sha256']}` |
| Mapping | `{plan['mapping_digest']}` |
| Калибровка | `{plan['calibration_digest']}` |
| Признаки | `{plan['feature_digest']}` |
| Исходные веса Brain | `{selection['nominal_weights_sha256']}` |

## Запуск и воспроизведение

{demo}

Отчёт воспроизводится без нового физического эксперимента и обучения:

```powershell
.\\.venv\\Scripts\\python.exe -m semantic_tools.physical_report --study {evidence['study_dir']} --binding {evidence['binding_path']} --output {output}{promotion_argument}
```

Команды `physical_collection plan/collect/replay` и `physical_learning train/evaluate` предназначены для нового явно выбранного каталога исследования. Существующие эпизоды, модель и test-результат защищены от перезаписи; изменение модели после test требует новой независимой тестовой выборки.

![Сравнение голов: достаточные и недостаточные кормления](physical-head-comparison.png)

Числа графика: `physical-head-comparison.npz`; переносимый рисунок: `physical-head-comparison.svg`; машинный отчёт и актуальные проверки: `overall.json`.
"""


def run(study_dir, binding_path, output, promotion_path=None):
    evidence = verify_evidence(study_dir, binding_path)
    promotion = promotion_status(promotion_path, evidence)
    evidence["promotion_path"] = None if promotion_path is None else str(promotion_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _plot(evidence, output)
    report = render_report(evidence, promotion, output)
    (output / "results.md").write_text(report, encoding="utf-8")
    result = evidence["results"]
    summary = {"schema_version": 1, "status": result["status"], "promotion": promotion,
        "plan_digest": evidence["plan"]["digest"], "model_sha256": result["model_sha256"],
        "baseline_model_sha256": result["baseline_model_sha256"], "test_result_sha256": file_digest(Path(study_dir) / "results.json"),
        "report_sha256": file_digest(output / "results.md"), "generator_sha256": file_digest(__file__),
        "checked_candidate": evidence["binding_guard"], "checked_collection": evidence["collection_guard"],
        "counts": dict(Counter(ep["split"] for ep in evidence["plan"]["episodes"])),
        "old_connected": {key: value for key, value in result["baseline_conditions"]["connected"].items() if key not in ("cases", "by_seed")},
        "new_connected": {key: value for key, value in result["conditions"]["connected"].items() if key not in ("cases", "by_seed")},
        "checks": result["checks"], "all_planned_cases_included": True,
        "validation_history": evidence.get("validation_history", []),
        "input_diagnostics": evidence.get("input_diagnostics", []),
        "integration_smoke": evidence.get("integration_smoke"),
        "integration_smoke_sha256": evidence.get("integration_smoke_sha256"),
        "scope": "Output-only NEED_FOOD head on a small physical pilot; same-X comparison, controls replay fixed body inputs"}
    learning.write_json(output / "overall.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/semantic-v2/physical-study"))
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/semantic_channel_v2"))
    parser.add_argument("--promotion-json", type=Path)
    args = parser.parse_args()
    result = run(args.study, args.binding, args.output, args.promotion_json)
    print(json.dumps({"status": result["status"], "promotion": result["promotion"]["status"],
        "protected_files": result["checked_candidate"]["preservation"]["checked_files"], "report": str(args.output / "results.md")}), flush=True)


if __name__ == "__main__":
    main()
