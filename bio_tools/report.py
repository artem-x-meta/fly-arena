r"""Rebuild the graded-vision report from completed, frozen benchmark batches.

Run from the project root: .\.venv\Scripts\python.exe bio_tools/report.py
No simulation runs, graph mutations or runtime imports are performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
LABELS = {
    "uniform": "Серый фон", "on_step": "Светлее: 0.5 → 0.6",
    "off_step": "Темнее: 0.5 → 0.4", "dark_loom": "Тёмный диск расширяется",
    "bright_loom": "Светлый диск расширяется", "shrinking": "Тёмный диск сужается",
    "translating": "Тёмный диск перемещается", "retinal_mean_flash": "Равномерное затемнение",
}
INTERVENTIONS = {
    "input_off": "Вход = 0", "retina": "Сетчатка",
    "lamina": "L1–L5", "lc4_lplc2": "LC4 + LPLC2", "gf": "GF",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_batches(root):
    records, statuses, provenance = [], {}, {}
    for family in ("controls", "interventions", "sensitivity"):
        directory = root / family
        paths = sorted(directory.rglob("summary.json")) if directory.exists() else []
        count = 0
        for path in paths:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Expected a completed list of cases: {path}")
            plan = path.with_name("plan.json")
            if plan.exists():
                plan_data = json.loads(plan.read_text(encoding="utf-8"))
                planned_sources = plan_data.get("sources", plan_data.get("sources_before"))
                if planned_sources is None:
                    raise ValueError(f"The plan has no frozen source manifest: {plan}")
                if any(row["sources"] != planned_sources for row in rows):
                    raise ValueError(f"Case sources do not match the frozen plan: {path}")
                provenance[str(plan.relative_to(root))] = digest(plan)
            provenance[str(path.relative_to(root))] = digest(path)
            for row in rows:
                trace_path = path.parent / row["trace"]
                if not trace_path.is_file():
                    raise FileNotFoundError(trace_path)
                with np.load(trace_path) as archive:
                    trace = {key: archive[key].copy() for key in archive.files}
                provenance[str(trace_path.relative_to(root))] = digest(trace_path)
                records.append({"family": family, "report": row, "trace": trace,
                                "summary_path": str(path.relative_to(root)),
                                "trace_path": str(trace_path.relative_to(root))})
                count += 1
        statuses[family] = {"status": "AVAILABLE" if count else "PENDING", "completed_cases": count}
    return records, statuses, provenance


def select(records, condition, intervention="none", eye="both", family="controls", gain=.8):
    rows = [x for x in records if x["family"] == family
            and x["report"]["condition"] == condition
            and x["report"]["intervention"] == intervention
            and x["report"]["eye"] == eye
            and x["report"]["model"]["gain"] == gain]
    if len(rows) > 1:
        raise ValueError(f"Ambiguous baseline case: {condition}/{eye}/{intervention}/{gain}")
    return rows[0] if rows else None


def gf_peak(record):
    return record["report"]["populations"]["GF"]["peak_rms_delta_au"]


def proof_bundle(root):
    proofs = {}
    for name in ("compatibility.json", "compatibility-graph.json", "fullgraph-checkpoint.json"):
        path = root / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            proofs[name] = {"status": "AVAILABLE", "sha256": digest(path), "data": value}
        else:
            proofs[name] = {"status": "PENDING"}
    tests = root / "tests.xml"
    if tests.exists():
        tree = ET.parse(tests).getroot()
        suites = [tree] if tree.tag == "testsuite" else list(tree.iter("testsuite"))
        counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                  for key in ("tests", "errors", "failures", "skipped")}
        proofs["tests.xml"] = {"status": "AVAILABLE", "sha256": digest(tests), **counts}
    else:
        proofs["tests.xml"] = {"status": "PENDING"}
    return proofs


def analyze(records, statuses, provenance):
    data = {"format": 1, "units": "arbitrary voltage/release units; no spikes, Hz or measured mV",
            "status": statuses, "input_sha256": provenance,
            "report_script_sha256": digest(Path(__file__)),
            "controls": [], "interventions": [], "sensitivity": [],
            "on_off_antisymmetry": {}, "comparisons": {}}
    source_variants = {json.dumps(x["report"]["sources"], sort_keys=True) for x in records}
    data["all_available_cases_same_runtime_sources"] = len(source_variants) <= 1
    for record in records:
        row, trace = record["report"], record["trace"]
        gf = row["populations"]["GF"]
        deviation = np.max(np.abs(trace["retinal_luminance"] - trace["retinal_luminance"][0]), axis=1)
        changed = np.flatnonzero(deviation > .02)
        item = {"condition": row["condition"], "eye": row["eye"],
                "intervention": row["intervention"], "model": row["model"],
                "display_contrast": row["display"]["contrast"],
                "gf_peak_rms_delta_au": gf["peak_rms_delta_au"],
                "gf_signed_mean_at_rms_peak_au": gf["mean_at_peak_au"],
                "gf_peak_at_s": gf["peak_rms_at_s"],
                "gf_maximum_is_last_recorded_sample": bool(np.argmax(trace["GF_rms"]) == len(trace["GF_rms"]) - 1),
                "changing_retinal_inputs": row["changing_retinal_inputs"],
                "first_retinal_frame_change_gt_002_at_s": float(trace["times_s"][changed[0]] - row["model"]["dt_s"]) if len(changed) else None,
                "initial_retina_minmax": [float(trace["retinal_luminance"][0].min()), float(trace["retinal_luminance"][0].max())],
                "operating_regime": row.get("operating_regime", "PENDING"),
                "trace": record["trace_path"]}
        data[record["family"]].append(item)
    on, off = select(records, "on_step"), select(records, "off_step")
    if on and off:
        for name in on["report"]["populations"]:
            if f"{name}_mean" not in off["trace"]:
                continue
            a, b = on["trace"][f"{name}_mean"], off["trace"][f"{name}_mean"]
            data["on_off_antisymmetry"][name] = {
                "max_abs_sum_signed_mean_au": float(np.max(np.abs(a + b))),
                "on_max_abs_mean_au": float(np.max(np.abs(a))),
                "off_max_abs_mean_au": float(np.max(np.abs(b)))}
    loom = select(records, "dark_loom")
    if loom:
        intact = gf_peak(loom)
        for condition in ("retinal_mean_flash", "shrinking", "translating", "bright_loom"):
            other = select(records, condition)
            if other:
                peak = gf_peak(other)
                data["comparisons"][condition] = {"other_over_dark_peak_rms": peak / intact if intact else None,
                                                  "dark_over_other_peak_rms": intact / peak if peak else None}
        for row in data["interventions"]:
            row["gf_peak_rms_fraction_of_intact"] = row["gf_peak_rms_delta_au"] / intact if intact else None
    return data


def plot(records, root):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "fly-bio-v1-report",
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.8), layout="constrained")
    fig.suptitle("Градуальная модель MaleCNS: передача зрительного контраста", fontsize=17, weight="bold")
    ax = axes[0, 0]
    names = ["L1", "L2", "Mi1", "Tm1", "T4", "T5", "LC4", "LPLC2", "GF"]
    positions = np.arange(len(names))
    for shift, condition, color, label in ((-.19, "on_step", "#2369a3", "Светлее (+0.1)"),
                                            (.19, "off_step", "#c46a21", "Темнее (−0.1)")):
        record = select(records, condition)
        if record:
            values = [record["report"]["populations"][name]["mean_at_peak_au"] for name in names]
            ax.bar(positions + shift, values, width=.36, color=color, label=label)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.axhline(0, color="#777777", linewidth=.7)
    ax.set_xticks(positions, names, rotation=30)
    ax.set_ylabel("Среднее ΔV при максимуме RMS, усл. ед.\nСимметричная логарифмическая шкала")
    ax.set_title("A. ON/OFF: противоположные знаки ответа", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.15)
    ax = axes[0, 1]
    for condition, color, style in (("dark_loom", "#b95d13", "-"),
                                   ("retinal_mean_flash", "#2477a6", "--"),
                                   ("shrinking", "#7a4b97", "-"),
                                   ("bright_loom", "#888888", ":")):
        record = select(records, condition)
        if record:
            ax.plot(record["trace"]["times_s"], record["trace"]["GF_mean"] * 1e6,
                    color=color, linestyle=style, linewidth=2, label=LABELS[condition])
    ax.axvspan(.3, 1.1, color="#dddddd", alpha=.35)
    ax.axhline(0, color="#777777", linewidth=.7)
    ax.set_xlabel("Время, с; серым показано изменение изображения")
    ax.set_ylabel("GF: среднее ΔV двух клеток × 10⁶, усл. ед.")
    ax.set_title("B. GF: расширение и равномерное затемнение", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(alpha=.15)
    loom = select(records, "dark_loom")
    ax = axes[1, 0]
    if loom:
        groups = ["retina", "L1", "L2", "Mi1", "Tm1", "T4", "T5", "LC4", "LPLC2", "GF"]
        values = [loom["report"]["populations"][name]["peak_rms_delta_au"] for name in groups]
        ax.bar(groups, values, color="#4b7771")
        ax.set_yscale("log")
    else:
        ax.text(.5, .5, "PENDING", transform=ax.transAxes, ha="center")
    ax.set_ylabel("Максимум RMS(ΔV) в окне, усл. ед.")
    ax.set_xlabel("Популяции; максимумы измерены независимо")
    ax.set_title("C. Передача тёмного контраста по популяциям", loc="left", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=.15)
    ax = axes[1, 1]
    values, labels = [], []
    if loom:
        values.append(100.)
        labels.append("Без\nвмешательства")
        for intervention in INTERVENTIONS:
            record = select(records, "dark_loom", intervention=intervention, family="interventions")
            if record:
                values.append(100 * gf_peak(record) / gf_peak(loom))
                labels.append(INTERVENTIONS[intervention])
    if values:
        ax.bar(np.arange(len(values)), values, color=["#777777"] + ["#b75b58"] * (len(values) - 1))
        for i, value in enumerate(values):
            ax.text(i, value + 2, f"{value:.3g}%", ha="center", fontsize=9)
        ax.set_xticks(np.arange(len(values)), labels, rotation=20)
        ax.set_ylim(0, 116)
    else:
        ax.text(.5, .5, "PENDING", transform=ax.transAxes, ha="center")
    ax.set_ylabel("Максимум RMS GF / исходный максимум, %")
    ax.set_xlabel("Клетки удержаны при исходном V; тонический выход сохранён")
    ax.set_title("D. Остаток GF после вмешательств", loc="left", fontsize=11)
    ax.grid(axis="y", alpha=.15)
    fig.get_layout_engine().set(rect=(0, .09, 1, .83))
    fig.text(.5, .025,
             "Все 166 700 клеток градуальные, включая GF. Единицы условные; спайков и моторного контроля нет.\n"
             "Максимум ограничен окном 1.5 с. Удержание LC4/LPLC2 — вмешательство в модель, не биологическое выключение клетки.",
             ha="center", fontsize=10, color="#444444")
    fig.savefig(root / "response-summary.png", dpi=170)
    fig.savefig(root / "response-summary.svg", metadata={"Date": None})
    plt.close(fig)


def markdown(data):
    lines = ["# Результаты градуальной зрительной модели MaleCNS", "",
             "Автоматически собрано из завершённых серий. Обновление: "
             r"`.\.venv\Scripts\python.exe bio_tools/report.py` из корня проекта.", "",
             "**Получена передача контраста от изображения через граф к LC4/LPLC2 и GF. "
             "Избирательный детектор приближения и замена моторного контроллера не получены.**", "",
             "Напряжение и выделение медиатора имеют условные единицы. GF здесь также градуальный; спайки не вычисляются.", "",
             "![Измеренные ответы и вмешательства](response-summary.png)", "",
             "## Набор данных", ""]
    for family, state in data["status"].items():
        lines.append(f"- {family}: **{state['status']}**, завершённых случаев {state['completed_cases']}.")
    lines += [f"- Один и тот же runtime во всех доступных случаях: {data['all_available_cases_same_runtime_sources']}.", "",
              "Хэши входных отчётов и трасс сохранены в [results-analysis.json](results-analysis.json). "
              "Каждый случай сравнивается с собственным начальным изображением, удерживаемым неподвижным.", "",
              "## Внешние стимулы", "",
              "RMS — величина ответа без знака, усреднённая по двум GF. Знак показан отдельно. "
              "Максимум вычислен только в зарегистрированном окне; это не обязательно полный максимум переходного процесса.", "",
              "| Стимул | max RMS GF | Среднее ΔV GF в тот же момент | Время, с | Меняющиеся ретинальные входы |",
              "|---|---:|---:|---:|---:|"]
    for row in data["controls"]:
        lines.append(f"| {LABELS.get(row['condition'], row['condition'])} | {row['gf_peak_rms_delta_au']:.8g} | "
                     f"{row['gf_signed_mean_at_rms_peak_au']:+.8g} | {row['gf_peak_at_s']:.2f} | {row['changing_retinal_inputs']} |")
    if not data["controls"]:
        lines.append("| PENDING | — | — | — | — |")
    comparisons = data["comparisons"]
    if "retinal_mean_flash" in comparisons:
        ratio = comparisons["retinal_mean_flash"]["dark_over_other_peak_rms"]
        lines += ["", f"Расширение / равномерное затемнение по max RMS GF: **{ratio:.5f}**. "
                  "Равномерное изображение сохраняет среднюю яркость на фактических ретинальных входах каждого глаза. "
                  "Такой близкий ответ не подтверждает выделение геометрии приближения."]
    if data["on_off_antisymmetry"]:
        gf_error = data["on_off_antisymmetry"]["GF"]["max_abs_sum_signed_mean_au"]
        lines += ["", f"Для ON/OFF шагов max|средний GF_ON + средний GF_OFF| = {gf_error:.8g}. "
                  "Почти противоположные ответы согласуются с линейной передачей контраста в данном диапазоне."]
    lines += ["", "Сужающийся тёмный диск даёт отрицательное изменение GF; его RMS нельзя трактовать как "
              "возбуждение или вероятность побега. Его начальная рабочая точка отличается от расширения. "
              "Перемещающийся диск имеет диаметр 10°, расширяющийся достигает 70°: это не сравнение при равной площади.", "",
              "## Вмешательства", "",
              "Удержание клетки сохраняет её исходное напряжение и тонический выход, удаляя модуляцию. "
              "Это отличается от удаления клетки, обнуления выделения медиатора или биологической инактивации.", "",
              "| Глаз / вмешательство | max RMS GF | Остаток от исходного max RMS |",
              "|---|---:|---:|"]
    for row in data["interventions"]:
        ratio = row.get("gf_peak_rms_fraction_of_intact")
        value = f"{ratio * 100:.5f}%" if ratio is not None else "PENDING"
        lines.append(f"| {row['eye']} / {row['intervention']} | {row['gf_peak_rms_delta_au']:.8g} | {value} |")
    if not data["interventions"]:
        lines.append("| PENDING | — | — |")
    lines += ["", "Отношения независимых максимумов не являются разложением ответа на причинные доли. "
              "Небольшой остаток при удержании промежуточных групп требует проверки других путей и численной точности. "
              "Асимметрия левого и правого условий включает разную представленность изображения в существующих UV-портах.", "",
              "## Чувствительность к параметрам", "",
              "| Условие | Контраст | gain | max RMS GF | Среднее GF в тот же момент |",
              "|---|---:|---:|---:|---:|"]
    for row in data["sensitivity"]:
        lines.append(f"| {row['condition']} / {row['intervention']} | {row['display_contrast']:.3g} | {row['model']['gain']:.3g} | "
                     f"{row['gf_peak_rms_delta_au']:.8g} | {row['gf_signed_mean_at_rms_peak_au']:+.8g} |")
    if not data["sensitivity"]:
        lines.append("| PENDING — серия ещё не доступна | — | — | — | — |")
    proofs = data.get("proof_bundle", {})
    lines += ["", "## Проверки и совместимость", ""]
    tests = proofs.get("tests.xml", {})
    if tests.get("status") == "AVAILABLE":
        lines.append(f"[Тесты](tests.xml): {tests['tests']}, ошибок {tests['errors']}, "
                     f"неудач {tests['failures']}, пропусков {tests['skipped']}.")
    else:
        lines.append("Тесты: PENDING.")
    for name in ("compatibility.json", "compatibility-graph.json", "fullgraph-checkpoint.json"):
        proof = proofs.get(name, {})
        value = proof.get("data", {})
        status = value.get("passed", value.get("all_match", "PENDING"))
        lines.append(f"- [{name}]({name}): {status}.")
    lines += ["", "Совместимость здесь проверена совпадением байтов; старую физику повторно не запускали. "
              "Сохранение новой сети проверено через API состояния на полном графе; отдельного CLI resume этот отчёт не добавляет.", "",
              "Полное объяснение модели и границ вывода: "
              "[BIO_GRADED_V1.md](../../docs/BIO_GRADED_V1.md). "
              "Первичные источники: [BIO_MODEL_SOURCES.md](../../docs/BIO_MODEL_SOURCES.md).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "runs" / "bio-visual-v1")
    args = parser.parse_args()
    root = args.root.resolve()
    records, statuses, provenance = load_batches(root)
    if not records:
        raise SystemExit("No completed benchmark summaries found; simulations were not started.")
    data = analyze(records, statuses, provenance)
    data["proof_bundle"] = proof_bundle(root)
    (root / "results-analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (root / "RESULTS.md").write_text(markdown(data), encoding="utf-8")
    plot(records, root)
    print(json.dumps({"output": str(root), "cases": len(records), "status": statuses}, ensure_ascii=False))


if __name__ == "__main__":
    main()
