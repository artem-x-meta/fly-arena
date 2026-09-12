"""Write the Stage A/B implementation and measurement report from saved evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fly_semantic.mapping import file_digest, validate_mapping
from fly_semantic.runtime import semantic_code_digest, validate_calibration
from .state_experiment import load_plan, metrics
from .transition_metrics import feeding_release_summary


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_bindings(training, results, physical, study, physical_plan, mapping,
                      calibration, verification, *, model_sha, calibration_sha, current_code):
    if any(item.get("model_sha256") != model_sha for item in (training, results, physical, physical_plan)):
        raise ValueError("Report would mix different trained model artifacts")
    if any(item.get("plan_digest") != study["digest"] for item in (training, results)):
        raise ValueError("Report study/training/evaluation plans differ")
    if (study["calibration_digest"] != calibration["digest"]
            or physical_plan["calibration_sha256"] != calibration_sha
            or calibration["mapping_digest"] != mapping["digest"]):
        raise ValueError("Report would mix different calibration/mapping artifacts")
    if (verification.get("status") != "PASS"
            or verification.get("semantic_code_before") != current_code
            or verification.get("semantic_code_after") != current_code
            or verification.get("mapping_digest") != mapping["digest"]
            or not verification.get("checks")
            or any(result != "PASS" for result in verification["checks"].values())):
        raise ValueError("Report needs passing runtime verification for the current semantic code and mapping")


def physical_cases(directory, plan, summary):
    cases = []
    for seed in plan["seeds"]:
        for scene, duration in plan["scenes"].items():
            for condition in plan["conditions"]:
                episode = f"{seed}-{scene}-{condition}"
                row = read_json(directory / f"{episode}.json")
                if (row["episode"] != episode or row["seed"] != seed or row["scene"] != scene
                        or row["condition"] != condition or row["duration_s"] != duration):
                    raise ValueError("Physical episode does not match the frozen plan")
                cases.append(row)
    if len(cases) != summary["episodes"]:
        raise ValueError("Physical result episode count differs from plan")
    for condition in plan["conditions"]:
        samples = [r for case in cases if case["condition"] == condition for r in case["samples"]]
        recomputed = {
            "raw": metrics([r["target"] for r in samples], [r["score"] >= plan["score_threshold"] for r in samples]),
            "indicator": metrics([r["target"] for r in samples], [r["active"] for r in samples])}
        if summary["conditions"][condition] != recomputed:
            raise ValueError("Physical pooled metrics differ from their episode evidence")
    return cases


def resource_evidence(directory):
    disabled_path, enabled_path = directory / "profile-disabled.json", directory / "profile-enabled.json"
    if not disabled_path.exists() or not enabled_path.exists():
        return {"status": "NOT_MEASURED", "vram_peak_bytes": None}, (
            "Парные накладные расходы runtime и пик RAM с обученной головой пока не измерены; "
            "цели +20% времени/+1GiB RAM не подтверждены. VRAM не измерялся. "
            "Исторический исходный замер приведён в baseline.md.")
    disabled, enabled = read_json(disabled_path), read_json(enabled_path)
    if (disabled["enabled"] is not False or enabled["enabled"] is not True
            or disabled["simulation_seconds"] != enabled["simulation_seconds"]
            or disabled["loop_wall_seconds"] <= 0 or enabled["loop_wall_seconds"] <= 0):
        raise ValueError("Resource profiles do not form a matched disabled/enabled comparison")
    overhead = enabled["loop_wall_seconds"] / disabled["loop_wall_seconds"] - 1
    extra_ram = enabled["ram_peak_bytes"] - disabled["ram_peak_bytes"]
    result = {"status": "SHORT_PAIRED_SAMPLE", "disabled": disabled, "enabled": enabled,
              "loop_overhead_fraction": overhead, "additional_peak_working_set_bytes": extra_ram,
              "within_20pct_loop_target_in_this_sample": overhead <= .2,
              "within_1gib_added_ram_target_in_this_sample": extra_ram <= 1024 ** 3,
              "vram_peak_bytes": None}
    text = (f"Отдельные процессы, одинаковые настройки и {enabled['simulation_seconds']:g}s симуляции: "
            f"цикл disabled {disabled['loop_wall_seconds']:.3f}s, enabled {enabled['loop_wall_seconds']:.3f}s "
            f"(разница {overhead:+.1%}). Peak working set disabled {disabled['ram_peak_bytes']/1024**2:.1f}MiB, "
            f"enabled {enabled['ram_peak_bytes']/1024**2:.1f}MiB, прибавка {extra_ram/1024**2:+.1f}MiB. "
            "Это один короткий парный замер с рендерингом глаз; он включает влияние входа на спайковую "
            "активность и не является оценкой вариации времени длинных прогонов. VRAM не измерялся.")
    return result, text


def run(study_dir, physical_dir, calibration_dir):
    study_dir, physical_dir, calibration_dir = map(Path, (study_dir, physical_dir, calibration_dir))
    training = read_json(study_dir / "training.json")
    results = read_json(study_dir / "results.json")
    physical = read_json(physical_dir / "results.json")
    study, physical_plan = load_plan(study_dir), read_json(physical_dir / "plan.json")
    mapping = read_json(calibration_dir / "mapping.json")
    calibration = read_json(calibration_dir / "calibration.json")
    first = read_json("runs/semantic-v1/state-study/training.json")
    verification_dir = Path("runs/semantic-v1/verification")
    verification = read_json(verification_dir / "runtime-verification.json")
    validate_mapping(mapping, Path("data/graph"))
    validate_calibration(calibration, mapping)
    validate_bindings(training, results, physical, study, physical_plan, mapping, calibration, verification,
        model_sha=file_digest(study_dir / "state-readout.npz"),
        calibration_sha=file_digest(calibration_dir / "calibration.json"), current_code=semantic_code_digest())
    output = Path("reports/semantic_channel")
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for name, condition in results["conditions"].items():
        if name == "train_constant":
            table.append(f"| {name} | {condition['f1']:.3f} | — | {condition['accuracy']:.3f} |")
        else:
            table.append(f"| {name} | {condition['raw']['f1']:.3f} | {condition['indicator']['f1']:.3f} | {condition['raw']['accuracy']:.3f} |")
    physical_rows = physical_cases(physical_dir, physical_plan, physical)
    releases = feeding_release_summary(physical_rows)
    release_by_episode = {row["episode"]: row for row in releases["cases"]}
    physical_ready = (physical["raw_target_reached"] and physical["indicator_target_reached"]
                      and releases["physical_release_supported"])
    end_to_end_ready = results["quality_target_reached"] and results["control_gain_target_reached"] and physical_ready
    resources, resource_text = resource_evidence(verification_dir)
    conclusion = ("Предварительный выход NEED_FOOD прошёл стенд и эти физические случаи. Длительная свободная жизнь остаётся непроверенной."
        if end_to_end_ready else "**Выход NEED_FOOD пока не готов для надёжного использования в свободной жизни мухи.** Успех на нейронном стенде отделён от физического переноса и снятия нужды после еды.")
    release_text = (f"Подтверждено снятие после эталонного перехода в {releases['connected_cases_with_supported_release']}/"
                    f"{releases['connected_feeding_cases']} connected-случаях питания. "
                    "Для зачёта индикатор должен быть активен до снятия нужды и перейти True→False в последующем "
                    "неактивном окне эталона. Ранний off, постоянное молчание и heartbeat не засчитываются. "
                    "Отсутствующее время и задержка сохраняются как null; это описательный анализ после прогона, "
                    "не изменение замороженного критерия pooled F1.")
    feed_lines = []
    for row in physical_rows:
        if row.get("scene") == "physical_feeding":
            offs = [t["time_ms"] for t in row["target_transitions"] if not t["active"]]
            matches = release_by_episode[row["episode"]]["matches"]
            off_events = [e["sim_time_ms"] for e in row["events"] if not e["active"]]
            matched = [item["matched_neural_off_ms"] for item in matches]
            delays = [item["latency_ms"] for item in matches]
            feed_lines.append(f"| {row['seed']} | {row['condition']} | {row['ingested_total']:.3f} | {row['final_hunger']:.3f} | {offs} | {off_events} | {matched} | {delays} |")
    text = f"""# Семантический канал: первый этап A/B

Это технический отчёт по новому искусственному интерфейсу, 12 сентября 2026. Спецификация адаптирована к действующей арене. Закрытая био-ветка не возобновлялась. Ни текстовой модели, ни обучения человеческому языку здесь нет.

## Результат и объём

Нейронный стенд: **{results['status']}**. Замороженный критерий суммарной классификации в физической арене: **{physical['status']}**. Это предварительные короткие прогоны; результат не заменяет предусмотренную расширенную проверку 100 эпизодов на каждый из пяти seed.

{conclusion}

На стенде test raw F1={results['conditions']['connected']['raw']['f1']:.3f}, F1 индикатора={results['conditions']['connected']['indicator']['f1']:.3f}; в физической арене соответственно {physical['conditions']['connected']['raw']['f1']:.3f}/{physical['conditions']['connected']['indicator']['f1']:.3f}. Корректное снятие индикатора после питания подтверждено в {releases['connected_cases_with_supported_release']}/{releases['connected_feeding_cases']} connected-случаях. Это разные проверки, и сильная первая метрика не компенсирует пропуск последней.

Реализованы отключаемая обёртка действующей мухи, протокол шести ID, ограниченные очереди, искусственный непрерывный вход голода, признаки активности, обучаемый линейный выход NEED_FOOD, журнал, CLI, pause/reset и полное продолжение checkpoint. В рабочем декодере доступен только NEED_FOOD. Отдых и загрязнение есть в организме, но соответствующие считыватели пока не обучены и не отображаются как поддерживаемые.

Входящие FOOD_LEFT/RIGHT/AHEAD зарегистрированы, однако их доставка в демонстрации отключена до отдельной калибровки и обучения поведения. Считыватель выбора пути, свободный поиск пищи по подсказке и двусторонний интерфейс ещё не реализованы. Это завершённый первый этап инфраструктуры и выходного эксперимента, не весь MVP спецификации.

## За счёт чего формируется сигнал

Существующий Organism вычисляет потребность как `(1 − energy/energy_capacity) × (1 − gut_amount/gut_capacity)`, ограниченную диапазоном 0..1. Пища реально поступает через контакт рта и транзакцию среды; заполнение кишечника сразу снижает голод, усвоение энергии происходит позже. Эту модель новый канал не заменяет.

Прежний нейронный канал усиливает вкус пропорционально hunger×taste и исчезает вдали от еды. Добавлен отдельно именованный `semantic_hunger`: потребность умножается на **{calibration['hunger_gain_mv']:g} mV** и через штатный `Brain.step(sensory_currents=...)` подаётся в 32 сенсорные клетки. Это искусственная инженерная стимуляция; выбранные клетки не объявляются обнаруженным центром голода.

Состав клеток внутри каждого candidate mapping выбирается воспроизводимо с seed 42 по структуре: положительный знак модели, аннотированный класс `{mapping.get('selection_pool', 'cb_intrinsic')}`, верхняя четверть по числу исходящих записей связей, исключение штатных прямых входов и известных моторных портов. Конкретные клетки не отбирались по активности или меткам; выбор самого класса получателей и амплитуды выполнен по validation, как описано ниже. У каждого из четырёх портов по 32 разные клетки. Индексы, реальные bodyId, типы, правила и SHA исходных массивов сохранены в mapping.json.

Считыватель видит 512 внутренних клеток, выбранных по существующим контактам после этих входов. Все непосредственно стимулируемые клетки, штатные сенсорные входы и известные моторные порты исключены. Из числа спайков строятся причинные экспоненциальные следы 50/200 ms — 1024 признака в Hz. Эти следы являются внешней памятью считывания.

Линейный слой имеет **{training['trained_parameters']} обучаемых весов и смещений**; дополнительно сохранена нормализация, рассчитанная только по train. Обучение: BCE+L2={training['diagnostics']['l2']}, L-BFGS из уже установленного SciPy. В runtime слой получает только вектор нейронных признаков: энергии, ID подсказки, координат еды, номера эпизода или эталона там нет.

Оценка sigmoid — некалиброванный score. Индикатор включается при score≥0.7 и выключается при score≤0.4 после двух последовательных отсчётов 10Hz. Повтор активного события — не чаще одного heartbeat в секунду, срок актуальности 1500ms модельного времени. Отсутствие heartbeat означает устаревшие данные. Молчание не переводится как «всё хорошо».

## Что заморожено и что управляет телом

Полный `male-cns:v1.0`: 166700 клеток, 25582938 записей связей, 124177617 контактов. Исходный LIF, его gain0.075, порог7mV, постоянные20/5ms, шаг1ms и задержка2ms сохранены. Основной обмен мозга и тела — 10ms. В нормальных прогонах не меняются веса, знаки или топология. Контроль transmission_off использует отдельный нулевой массив передачи в памяти процесса; файлы графа остаются исходными.

Выбор еды/отдыха/чистки, навигация, походка, стабилизация и контактное питание остаются нынешними инженерными механизмами арены. Новый выходной считыватель сообщает оценку состояния и не выбирает движение. Дополнительная стимуляция может влиять на существующий нейронный вклад в гибридное управление; поэтому физический перенос проверен отдельно.

## Обучение и контрольные эпизоды

32 целых тренировочных эпизода и 8 validation. После выбора настройки — 20 test: пять новых групп seed по четыре эпизода. Каждый длится1.5s, даёт15 отсчётов, все они включены в метрики, включая начало и молчание. Первоначальные запасы, сон, пыль, сенсорный фон и начальное нейронное состояние варьируются. Высокие/низкие состояния сбалансированы по эпизодам. Teacher использует пороги потребности0.7/0.55 с отдельным гистерезисом; пороги вывода другие.

Учебный стенд использует настоящий полный Brain и существующий Organism, но искусственные локальные сенсорные фоны и прокси движения. Физического тела и поступления пищи в этих эпизодах нет. Это не скрывается за названием «арена». Переход после настоящего питания измеряется в следующем разделе. Поддерживается один смысл, поэтому сочетания нескольких потребностей пока не проверены.

Первый вариант с32 `cb_intrinsic` клетками и0.5mV отклонён на validation: train raw F1={first['train']['raw']['f1']:.3f}, validation raw F1={first['validation']['raw']['f1']:.3f}, индикатор={first['validation']['indicator']['f1']:.3f}. Его test не собирался. Артефакты сохранены. Затем изменили искусственную группу получателей на `{mapping.get('selection_pool', 'cb_intrinsic')}` и амплитуду на {calibration['hunger_gain_mv']:g}mV, повторили калибровку и обучение; текущий test не использовался для выбора параметров. Сильные калибровочные токи, провалившие заранее заданный предел частоты входных клеток, также сохранены.

Фиксированная цель первичного стенда — raw F1≥0.80 и выигрыш≥0.15 над отключением homeostasis и перестановкой эпизодов. Отдельно приведён F1 индикатора, поскольку пользователь видит именно его. Стандартная точность сама по себе не является доказательством передачи через нейросеть.

| Условие test | F1 score≥0.5 | F1 индикатора | Accuracy score |
|---|---:|---:|---:|
{chr(10).join(table)}

homeostasis_off удаляет новый ток при тех же начальных условиях. В учебном стенде вкус равен нулю, поэтому штатный hunger×taste тоже не передаёт голод. transmission_off удаляет внутреннюю передачу, сохраняя входы и ту же обученную голову. cross_episode_shuffle переставляет целые нейронные траектории между эпизодами; временной порядок внутри них сохраняется. train_constant выбирается по доле положительных train, не по test. Модели на контролях не переобучаются.

Парные разности F1 и95% интервалы bootstrap по пяти группам seed сохранены в results.json. При n=5 это описательная предварительная оценка, не оценка независимости сотен кадров или клеток. Даже падение без связей показывает зависимость данного считывателя от передачи; преимущества настоящей топологии над случайной и семантического «понимания» оно не доказывает.

## Перенос в тело и настоящая еда

Замороженная голова проверена на трёх новых seed: голодная муха вдали от еды1.5s, сытая вдали от еды1.5s, голодная возле пищи5s. Каждый случай повторён с отключённым новым входом. Сцена питания использует существующий feeding-contact и адаптивный хоботок; телепортации еды в организм и награды за сообщение нет.

| Условие физики | Raw F1 | F1 индикатора |
|---|---:|---:|
| connected | {physical['conditions']['connected']['raw']['f1']:.3f} | {physical['conditions']['connected']['indicator']['f1']:.3f} |
| homeostasis_off | {physical['conditions']['homeostasis_off']['raw']['f1']:.3f} | {physical['conditions']['homeostasis_off']['indicator']['f1']:.3f} |

Реальное поступление еды зарегистрировано в {physical['feeding_cases_with_actual_intake']}/6 случаях питания; переход эталона в неактивное состояние — в {physical['feeding_cases_with_target_off_transition']}/6. Значения времени ниже относятся к симуляции; пустой список означает отсутствие перехода в окне, а не успешное распознавание.

{release_text}

| Seed | Вход | Съедено food_u | Итоговый голод | Эталон off, ms | Все neural off, ms | Подходящий off, ms | Задержка, ms |
|---:|---|---:|---:|---|---|---|---|
{chr(10).join(feed_lines)}

Эти короткие случаи не устанавливают устойчивость на длинной жизни, всех геометриях и всех потребностях. Если физический индикатор не достигает цели, успешный нейронный стенд не объявляется готовым сообщением в свободной жизни.

## Совместимость, время и воспроизведение

Соседний пакет fly_semantic оборачивает экземпляр Brain и существующую EthologySimulation. Исходники старых пакетов не редактировались. Отключённый attach возвращается до чтения артефактов и не создаёт RNG или токи. Собственные RNG используются только при подготовке mapping/эпизодов; вывод и очередь детерминированы.

На полном графе и физическом теле проверены точное совпадение отключённого режима, пауза, reset и продолжение NPZ, включая MuJoCo, спайки, очереди, следы, гистерезис и кэш событий. Для проверки mechanics очереди использовались явно помеченные синтетические голова и cue-калибровка; они не являются обученным результатом и не используются демонстрацией. Обычный checkpoint защищён хешами кода, модели, mapping, калибровки и старой совместимостью. Артефакты разных голов не смешиваются.

Исходная скорость и память — в [baseline.md](baseline.md), финальные проверки — в runs/semantic-v1/verification. {resource_text} Старый core digest04a519… и исходные файлы графа проверяются по protected-manifest. Новые проверки не заменяют старые критерии биологического правдоподобия.

Перед генерацией отчёта сверяются SHA обученной головы во всех этапах, калибровка, mapping, план, результаты каждого физического эпизода и успешный runtime-smoke на текущей версии семантического кода. Разные или устаревшие артефакты приводят к ошибке генерации. Машиночитаемая сводка этапа находится в [implementation-summary.json](implementation-summary.json), сопоставление снятия нужды — в [food-release-analysis.json](food-release-analysis.json).

Запуск из D:\\fly-arena: `18_run_semantic_channel.cmd`. Это фиксированное консольное меню поверх физической арены: один шаг жизни1s, просмотр последнего нейронного выхода, сохранение и выход. Пока меню ждёт ввода, время не идёт. Кнопки направления перечислены и отключены до следующего этапа. Телеметрия организма показана отдельно от нейронного сообщения. `--semantic-config semantic_configs/default.toml` отключает канал; `--disabled` имеет приоритет. Базовые launchers1–17 остаются прежними.

Подробные команды и пути приведены в [docs/SEMANTIC_CHANNEL.md](../../docs/SEMANTIC_CHANNEL.md). Выходной checkpoint расположен в каталоге конкретного запуска demo, а обучение на старте демонстрации не выполняется.
"""
    (output / "results.md").write_text(text, encoding="utf-8")
    (output / "food-release-analysis.json").write_text(json.dumps(releases, indent=2), encoding="utf-8")
    implementation = {"implemented": ["disabled-compatible wrapper", "versioned protocol", "hunger input", "linear NEED_FOOD readout", "CLI", "complete runtime checkpoint"],
        "not_implemented": ["NEED_REST/NEED_GROOMING heads", "calibrated inbound cues", "receiver policy", "two-way behavioral interface", "validated free-life hunger release"],
        "base_connectome_and_version": "male-cns:v1.0; 166700 neurons, 25582938 stored edges, 124177617 contacts",
        "supported_outbound_concepts": ["NEED_FOOD"], "supported_inbound_concepts": [],
        "frozen_parameters": "Existing LIF/core weights and body mechanics; trained head frozen before held-out assessment",
        "trained_parameters": training["trained_parameters"],
        "scripted_body_components": ["behavior selection", "odor navigation", "gait", "stabilization", "contact feeding", "sleep/grooming primitives"],
        "artificial_sensory_ports": {"mapping_digest": mapping["digest"], "selection_pool": mapping.get("selection_pool", "cb_intrinsic"), "hunger_gain_mv": calibration["hunger_gain_mv"]},
        "training_data_and_splits": {"train": 32, "validation": 8, "test": 20, "split_unit": "whole episode", "training_physics": False},
        "state_readout_metrics": results["conditions"]["connected"],
        "receiver_behavior_metrics": {"status": "NOT_IMPLEMENTED"}, "control_results": results["conditions"],
        "physical_classification": physical, "physical_release_supported": releases["physical_release_supported"],
        "end_to_end_need_output_ready": bool(end_to_end_ready),
        "peak_ram_and_vram": resources, "performance_vs_baseline": resources,
        "limitations": ["preliminary small study", "synthetic sensory background during training", "physical feeding release not reliable", "VRAM unmeasured", "no biological semantic-understanding claim"],
        "reproduce_commands": "docs/SEMANTIC_CHANNEL.md", "model_sha256": training["model_sha256"],
        "runtime_verification_semantic_code": verification["semantic_code_after"]}
    (output / "implementation-summary.json").write_text(json.dumps(implementation, indent=2), encoding="utf-8")
    plot(output, results, physical)
    print(str(output / "results.md"))


def plot(output, results, physical):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, title, conditions in zip(axes, ["Full LIF state assay (20 held-out episodes)", "Physical transfer (18 short episodes)"],
        [{k: v for k, v in results["conditions"].items() if k != "train_constant"}, physical["conditions"]]):
        labels = list(conditions)
        x = np.arange(len(labels))
        ax.bar(x - .18, [conditions[k]["raw"]["f1"] for k in labels], .36, label="Score F1")
        ax.bar(x + .18, [conditions[k]["indicator"]["f1"] for k in labels], .36, label="Indicator F1")
        ax.axhline(.8, ls="--", color="black", lw=1, label="0.80 target")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.set_xticks(x, [s.replace("_", "\n") for s in labels], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Artificial NEED_FOOD channel — preliminary, fixed model", fontsize=12)
    fig.savefig(output / "state-results.png", dpi=170)
    fig.savefig(output / "state-results.svg")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", default="runs/semantic-v1/state-selected")
    p.add_argument("--physical", default="runs/semantic-v1/physical-selected")
    p.add_argument("--calibration", default="runs/semantic-v1/calibration-selected")
    a = p.parse_args()
    run(a.study, a.physical, a.calibration)
