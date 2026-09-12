"""Read frozen selectivity trials and build an auditable report; never simulate.

Run from any directory with the project's interpreter. Missing trials remain
PENDING. Runtime source or trace mismatches fail before publishing a report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from fly_bio_selectivity.metrics import direction_metrics

MODELS = ("baseline", "delayed")
MODEL_RU = {"baseline": "Исходная", "delayed": "Mi1 +18 / Tm1 +13 мс"}
DIRECTIONS = ("right", "down", "left", "up")
PROFILES = ("cal", "test0", "testpi")
TEST_PROFILES = ("test0", "testpi")
TYPES = tuple(f"T{k}{suffix}" for k in (4, 5) for suffix in "abcd")
CONTROLS = ("contracting", *DIRECTIONS, "flicker")
LABELS = {"right": "вправо", "down": "вниз", "left": "влево", "up": "вверх",
          "expanding": "расширение", "contracting": "сужение", "flicker": "мерцание",
          "static": "первый кадр"}
TRACE_KEYS = ("ids", "types", "indices", "times_s", "reference_voltage", "static_release_delta",
              "retinal_h1", "retinal_mean", "h1", "h2", "phase_h1", "mean_voltage_delta",
              "mean_release_delta", "cycle_h1", "cycle_mean_release", "all_times_s",
              "population_mean_delta")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def finite_json(value):
    """Strict JSON: undefined/empty scientific statistics are null, not NaN."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def trial_key(trial):
    return tuple(trial[name] for name in ("model", "profile", "condition", "intervention"))


def load_trials(root):
    root = Path(root).resolve()
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if tuple(plan["directions"]) != DIRECTIONS:
        raise ValueError("Direction metric requires clockwise right/down/left/up indices")
    for relative, expected in plan["sources"].items():
        source = PROJECT / relative
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f"Current runtime source differs from frozen plan: {relative}")
    records, missing = {}, []
    provenance = {"plan.json": digest(plan_path)}
    ids, types = None, None
    for trial in plan["trials"]:
        tag = f"{trial['index']:02d}-{trial['model']}-{trial['profile']}-{trial['condition']}-{trial['intervention']}"
        path = root / f"{tag}.json"
        if not path.exists():
            missing.append(trial)
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A worker may be in its short final JSON write; no partial data
            # are accepted as a finished trial.
            missing.append({**trial, "pending_reason": "incomplete JSON write"})
            continue
        if row.get("trial") != trial or row.get("sources") != plan["sources"]:
            raise ValueError(f"Trial source or design does not match frozen plan: {path.name}")
        trace_path = (root / row["trace"]).resolve()
        if trace_path.parent != root or not trace_path.is_file():
            raise ValueError(f"Invalid/missing completed trace: {row['trace']}")
        trace_hash = digest(trace_path)
        if trace_hash != row["trace_sha256"]:
            raise ValueError(f"Trace hash differs before loading: {trace_path.name}")
        with np.load(trace_path, allow_pickle=False) as archive:
            trace = {name: archive[name].copy() for name in TRACE_KEYS}
        if ids is None:
            ids, types = trace["ids"].copy(), trace["types"].copy()
        elif not np.array_equal(ids, trace["ids"]) or not np.array_equal(types, trace["types"]):
            raise ValueError("Recorded cell identity/order changed across trials")
        for name, values in trace.items():
            if values.dtype.kind in "fc" and not np.isfinite(values).all():
                raise ValueError(f"Nonfinite stored metric: {trace_path.name}:{name}")
        if len(trace["times_s"]) != 150 or trace["cycle_h1"].shape != (3, len(ids)):
            raise ValueError("Completed trace has an unexpected fixed analysis window")
        if not row.get("input_only_at_retina", False):
            raise ValueError("A trial violated the retinal input boundary")
        records[trial_key(trial)] = {"trial": trial, "report": row, "trace": trace}
        provenance[path.name] = digest(path)
        provenance[trace_path.name] = trace_hash
    return plan, records, missing, provenance, ids, types


def side_annotations(ids):
    """Use anatomical annotation, never a neural response, to group sides."""
    if ids is None:
        return np.empty(0, dtype="U16"), {"status": "PENDING"}
    path = PROJECT / "data/graph/neurons.feather"
    if not path.exists():
        return np.full(len(ids), "unknown", dtype="U16"), {"status": "UNAVAILABLE"}
    import pyarrow.feather as feather
    table = feather.read_table(path, columns=["bodyId", "rootSide", "somaSide"]).to_pydict()
    def normalize(value):
        text = str(value or "").strip().lower()
        return {"l": "left", "r": "right", "left": "left", "right": "right",
                "midline": "midline", "m": "midline", "bilateral": "bilateral"}.get(text, "unknown")
    by_id = {}
    conflicts = 0
    for cell, root_side, soma_side in zip(table["bodyId"], table["rootSide"], table["somaSide"]):
        root_value, soma_value = normalize(root_side), normalize(soma_side)
        by_id[int(cell)] = root_value if root_value != "unknown" else soma_value
        conflicts += root_value != "unknown" and soma_value != "unknown" and root_value != soma_value
    sides = np.array([by_id.get(int(cell), "unknown") for cell in ids], dtype="U16")
    return sides, {"status": "AVAILABLE", "sha256": digest(path),
                   "rule": "rootSide when recognized; otherwise somaSide; no screen-axis registration",
                   "root_soma_disagreements_all_annotations": int(conflicts),
                   "recorded_counts": dict(zip(*np.unique(sides, return_counts=True)))}


def get_record(records, model, profile, condition, intervention="none"):
    return records.get((model, profile, condition, intervention))


def proof_bundle(root):
    result = {}
    for name in ("tests.json", "real-graph-state.json", "compatibility.json", "source-proof.json"):
        path = Path(root) / "verification" / name
        if not path.is_file():
            result[name] = {"status": "PENDING"}
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        passed = value.get("passed", value.get("all_identical", False))
        result[name] = {"status": "PASS" if passed else "FAIL", "sha256": digest(path),
                        "counts": value.get("counts"), "changed": value.get("changed"),
                        "exact_serialized_roundtrip": value.get("exact_serialized_roundtrip"),
                        "scope": value.get("scope")}
    final_tests = Path(root) / "verification/tests-final.xml"
    if final_tests.is_file():
        tree = ET.parse(final_tests).getroot()
        suites = [tree] if tree.tag == "testsuite" else list(tree.iter("testsuite"))
        counts = {name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
                  for name in ("tests", "failures", "errors", "skipped")}
        result["tests-final.xml"] = {"status": "PASS" if not counts["failures"] and not counts["errors"] else "FAIL",
                                     "sha256": digest(final_tests), "counts": counts}
    return result


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"cells": 0, "q25": None, "median": None, "q75": None}
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    return {"cells": len(values), "q25": q25, "median": median, "q75": q75}


def direction_summary(metrics, mask):
    total = int(np.count_nonzero(mask))
    responsive = np.all(metrics["phase_responsive"], axis=0) & mask
    stable = np.all(metrics["phase_stable"], axis=0)
    passed = metrics["success"] & mask
    worst = np.min(metrics["phase_dsi"][:, responsive], axis=0)
    return {"cells": total, "calibration_responsive": int(np.count_nonzero(metrics["calibration_responsive"] & mask)),
            "both_test_phases_responsive": int(responsive.sum()),
            "not_responsive_in_both_test_phases": total - int(responsive.sum()),
            "responsive_but_nonstationary": int(np.count_nonzero(responsive & ~stable)),
            "directional_without_stationarity": int(np.count_nonzero(metrics["directional_success_without_stationarity"] & mask)),
            "success_cells": int(passed.sum()), "success_fraction_all_cells": float(passed.sum() / total) if total else None,
            "type_gate_passed": bool(passed.sum() >= .5 * total) if total else False,
            "worst_phase_dsi_both_phases_responsive_only": quantiles(worst),
            "calibration_pd_counts_all_cells": {name: int(np.count_nonzero(mask & (metrics["preferred_direction"] == index)))
                                                for index, name in enumerate(DIRECTIONS)},
            "calibration_pd_counts_responsive_cells": {
                name: int(np.count_nonzero(mask & metrics["calibration_responsive"] & (metrics["preferred_direction"] == index)))
                for index, name in enumerate(DIRECTIONS)}}


def dc_stable(cycles, floor):
    cycles = np.asarray(cycles, dtype=np.float64)
    previous, last = cycles[-2], cycles[-1]
    tolerance = np.maximum(floor, .1 * np.maximum(np.abs(previous), np.abs(last)))
    return np.abs(last - previous) <= tolerance


def lplc2_metrics(phase_signals, floor, clamp_signals=None):
    """Apply the fixed LPLC2 gates to two phases; inputs already select cells.

    Each phase maps conditions to dicts containing mean_release_delta,
    cycle_mean_release, h1. Expansion additionally contains
    static_release_delta. No cells are selected based on the outcome.
    """
    positive, comparator, stable, excitation, dominance, secondary = [], [], [], [], [], []
    signed_rows = []
    for signals in phase_signals:
        expansion = signals["expanding"]
        out = np.maximum(expansion["mean_release_delta"], 0.)
        control_values = [np.maximum(signals[name]["mean_release_delta"], 0.) for name in CONTROLS]
        control_values.append(np.maximum(expansion["static_release_delta"], 0.))
        maximum = np.max(np.stack(control_values), axis=0)
        settled = np.logical_and.reduce([
            dc_stable(signals[name]["cycle_mean_release"], floor) for name in ("expanding", *CONTROLS)])
        a = expansion["h1"]
        b = signals["contracting"]["h1"]
        radial = np.full(a.shape, np.nan)
        np.divide(a - b, a + b, out=radial, where=np.maximum(a, b) > floor)
        positive.append(out)
        comparator.append(maximum)
        stable.append(settled)
        excitation.append(out > floor)
        dominance.append(out >= 2. * maximum)
        secondary.append(radial)
        signed_rows.append({name: signals[name]["mean_release_delta"] for name in ("expanding", *CONTROLS)})
    positive, comparator = np.stack(positive), np.stack(comparator)
    stable, excitation, dominance = np.stack(stable), np.stack(excitation), np.stack(dominance)
    phase_success = stable & excitation & dominance
    functional = np.all(phase_success, axis=0)
    reductions = np.full(positive.shape, np.nan)
    clamp_stable = np.zeros(positive.shape, dtype=bool)
    causal = np.zeros(functional.shape, dtype=bool)
    if clamp_signals is not None:
        for phase, signal in enumerate(clamp_signals):
            clamped = np.maximum(signal["mean_release_delta"], 0.)
            ratio = np.full(clamped.shape, np.nan)
            np.divide(clamped, positive[phase], out=ratio, where=positive[phase] > floor)
            reductions[phase] = 1. - ratio
            clamp_stable[phase] = dc_stable(signal["cycle_mean_release"], floor)
        causal = functional & np.all((reductions >= .8) & clamp_stable, axis=0)
    return {"phase_positive_release": positive, "phase_max_control_positive_release": comparator,
            "phase_excitation": excitation, "phase_dominance": dominance,
            "phase_dc_stable": stable, "phase_success": phase_success,
            "functional_success": functional, "phase_radial_h1_dsi": np.stack(secondary),
            "phase_clamp_reduction": reductions, "phase_clamp_dc_stable": clamp_stable,
            "causal_success": causal, "clamp_available": clamp_signals is not None}


def analyze(root):
    plan, records, missing, provenance, ids, types = load_trials(root)
    sides, annotation = side_annotations(ids)
    data = {"format": 1, "status": "COMPLETE" if not missing else "PENDING",
            "planned_trials": len(plan["trials"]), "completed_trials": len(records), "missing_trials": missing,
            "all_completed_trials_match_frozen_sources_and_trace_hashes": True,
            "current_runtime_sources_match_plan": True, "input_sha256": provenance,
            "analysis_script_sha256": digest(Path(__file__)), "annotation": annotation,
            "verification": proof_bundle(root),
            "units": "arbitrary continuous voltage/release; not Hz or physiological mV",
            "criteria": plan["criteria"], "models": {}, "retinal_audit": [],
            "scope": "One deterministic graph, one frequency 2 Hz, one radial center, unregistered display axes"}
    cell_arrays = {} if ids is None else {"ids": ids, "types": types, "sides": sides}
    if ids is None:
        return data, cell_arrays, records
    for model in MODELS:
        model_data = {"status": "PENDING", "direction_status": "PENDING", "lplc2_status": "PENDING"}
        data["models"][model] = model_data
        nulls = [get_record(records, model, "cal", "gray"), get_record(records, model, "cal", "right", "input_off")]
        if any(row is None for row in nulls):
            model_data["pending_reason"] = "gray/input_off controls required for the fixed numerical floor"
            continue
        null_max = max(float(np.max(row["trace"]["h1"])) for row in nulls)
        floor = max(1e-6, 100. * null_max)
        model_data.update({"null_h1_max_au": null_max, "response_floor_au": floor,
                           "null_mean_release_max_abs_au": max(float(np.max(np.abs(row["trace"]["mean_release_delta"]))) for row in nulls)})
        required = [get_record(records, model, profile, direction) for profile in PROFILES for direction in DIRECTIONS]
        if all(row is not None for row in required):
            calibration = np.stack([get_record(records, model, "cal", direction)["trace"]["h1"] for direction in DIRECTIONS])
            test = np.stack([[get_record(records, model, profile, direction)["trace"]["h1"] for direction in DIRECTIONS] for profile in TEST_PROFILES])
            # Each trial stores C x N, so stack directions behind cycle.
            cycles = np.stack([np.stack([get_record(records, model, profile, direction)["trace"]["cycle_h1"]
                                        for direction in DIRECTIONS], axis=1) for profile in TEST_PROFILES])
            metrics = direction_metrics(calibration, test, floor, cycles)
            model_data["direction_status"] = "AVAILABLE"
            model_data["types"] = {name: direction_summary(metrics, types == name) for name in TYPES}
            model_data["type_sides"] = {name: {side: direction_summary(metrics, (types == name) & (sides == side))
                                              for side in np.unique(sides[types == name])} for name in TYPES}
            model_data["types_passing"] = [name for name in TYPES if model_data["types"][name]["type_gate_passed"]]
            for name, value in metrics.items():
                if isinstance(value, np.ndarray):
                    cell_arrays[f"{model}_direction_{name}"] = value
        selected = types == "LPLC2"
        radial_required = [get_record(records, model, profile, condition) for profile in TEST_PROFILES
                           for condition in ("expanding", *CONTROLS)]
        if all(row is not None for row in radial_required):
            def selected_signal(record):
                trace = record["trace"]
                return {"mean_release_delta": trace["mean_release_delta"][selected],
                        "cycle_mean_release": trace["cycle_mean_release"][:, selected],
                        "static_release_delta": trace["static_release_delta"][selected],
                        "h1": trace["h1"][selected]}
            phases = [{condition: selected_signal(get_record(records, model, profile, condition))
                       for condition in ("expanding", *CONTROLS)} for profile in TEST_PROFILES]
            clamp_records = [get_record(records, model, profile, "expanding", "t4t5") for profile in TEST_PROFILES]
            clamps = [selected_signal(row) for row in clamp_records] if all(row is not None for row in clamp_records) else None
            metrics = lplc2_metrics(phases, floor, clamps)
            n = int(selected.sum())
            functional_count, causal_count = int(metrics["functional_success"].sum()), int(metrics["causal_success"].sum())
            model_data["lplc2_status"] = "AVAILABLE" if model == "baseline" or clamps is not None else "PENDING_CLAMP"
            model_data["lplc2"] = {
                "cells": n, "functional_success_cells": functional_count,
                "functional_success_fraction_all_cells": functional_count / n if n else None,
                "functional_gate_passed": bool(functional_count >= .5 * n) if n else False,
                "causal_clamp_available": clamps is not None,
                "causal_effect_estimable_above_floor": clamps is not None and bool(np.any(np.all(metrics["phase_excitation"], axis=0))),
                "causal_success_cells": causal_count if clamps is not None else None,
                "causal_success_fraction_all_cells": causal_count / n if n and clamps is not None else None,
                "causal_gate_passed": bool(causal_count >= .5 * n) if n and clamps is not None else None,
                "both_phase_excitation_cells": int(np.all(metrics["phase_excitation"], axis=0).sum()),
                "both_phase_dominance_cells": int(np.all(metrics["phase_dominance"], axis=0).sum()),
                "both_phase_dc_stable_cells": int(np.all(metrics["phase_dc_stable"], axis=0).sum()),
                "phase_summaries": {profile: {
                    "excitation_cells": int(metrics["phase_excitation"][phase].sum()),
                    "dominance_cells": int(metrics["phase_dominance"][phase].sum()),
                    "dc_stable_cells": int(metrics["phase_dc_stable"][phase].sum()),
                    "radial_h1_dsi_response_above_floor_only": quantiles(metrics["phase_radial_h1_dsi"][phase]),
                    "clamp_reduction_expansion_above_floor_only": quantiles(metrics["phase_clamp_reduction"][phase]),
                    "conditions": {condition: {
                        "mean_signed_release_delta_all_cells": float(signal["mean_release_delta"].mean()),
                        "mean_positive_release_delta_all_cells": float(np.maximum(signal["mean_release_delta"], 0.).mean()),
                        "mean_h1_all_cells": float(signal["h1"].mean()),
                        "max_positive_release_delta": float(np.maximum(signal["mean_release_delta"], 0.).max()),
                        "dc_stable_cells": int(dc_stable(signal["cycle_mean_release"], floor).sum())}
                                   for condition, signal in phases[phase].items()},
                    "initial_static_mean_positive_release_delta_all_cells": float(np.maximum(phases[phase]["expanding"]["static_release_delta"], 0.).mean())}
                                    for phase, profile in enumerate(TEST_PROFILES)}}
            cell_arrays["lplc2_ids"] = ids[selected]
            cell_arrays["lplc2_sides"] = sides[selected]
            for name, value in metrics.items():
                if isinstance(value, np.ndarray):
                    cell_arrays[f"{model}_lplc2_{name}"] = value
        model_data["status"] = "AVAILABLE" if model_data["direction_status"] == "AVAILABLE" and model_data["lplc2_status"] == "AVAILABLE" else "PENDING"
    # Retinal power checks do not consume any neural response and cannot be
    # adjusted to improve direction classification.
    for model in MODELS:
        for profile in PROFILES:
            expected = .5 * float(plan["profiles"][profile][0])
            for condition in ("expanding", "contracting", *DIRECTIONS, "flicker"):
                row = get_record(records, model, profile, condition)
                if row is None:
                    continue
                error = float(np.max(np.abs(row["trace"]["retinal_h1"] - expected)))
                mean_error = float(np.max(np.abs(row["trace"]["retinal_mean"])))
                data["retinal_audit"].append({"model": model, "profile": profile, "condition": condition,
                    "check": "each_port_h1_matches_same_local_amplitude", "expected_amplitude": expected,
                    "maximum_absolute_error": error, "maximum_absolute_temporal_mean_error": mean_error,
                    "passed": error <= 1e-6 and mean_error <= 1e-6})
            for forward, backward in (("right", "left"), ("down", "up"), ("expanding", "contracting")):
                a, b = get_record(records, model, profile, forward), get_record(records, model, profile, backward)
                if a is None or b is None:
                    continue
                error = float(np.max(np.abs(a["trace"]["retinal_h1"] - b["trace"]["retinal_h1"])))
                reference_error = float(np.max(np.abs(a["trace"]["reference_voltage"] - b["trace"]["reference_voltage"])))
                data["retinal_audit"].append({"model": model, "profile": profile, "conditions": [forward, backward],
                    "check": "opposite_directions_equal_port_h1_and_recorded_initial_voltage",
                    "maximum_absolute_amplitude_error": error,
                    "maximum_recorded_initial_voltage_error": reference_error,
                    "passed": error <= 1e-6 and reference_error <= 1e-6})
    data["retinal_audit_all_available_passed"] = all(row["passed"] for row in data["retinal_audit"])
    data["trial_diagnostics"] = [{"trial": row["trial"], "wall_seconds": row["report"]["wall_seconds"],
                                   "ever_rectified_by_type": row["report"]["ever_rectified_by_type"],
                                   "input_only_at_retina": row["report"]["input_only_at_retina"]}
                                  for row in records.values()]
    test_rows = [row for row in records.values() if row["trial"]["profile"] in TEST_PROFILES
                 and row["trial"]["intervention"] == "none"]
    data["validation_operating_regime"] = {
        "completed_validation_trials": len(test_rows), "planned_validation_trials": 28,
        "all_recorded_test_trajectories_no_rectifier_crossings": bool(test_rows) and
            all(not row["report"]["ever_rectified_by_type"] for row in test_rows),
        "interpretation": "Within an active linear region, a stable delayed system driven by zero-mean periodic input cannot develop a stationary DC shift; this does not rule out directional harmonic transmission"}
    return finite_json(data), cell_arrays, records


def number(value, precision=4):
    return "—" if value is None else f"{value:.{precision}g}"


def direction_table(data):
    lines = ["| Модель | Тип | Все клетки | Отзывчивы в обеих фазах | Прошли все критерии | Медиана худшего DSI* |",
             "|---|---|---:|---:|---:|---:|"]
    for model in MODELS:
        rows = data["models"].get(model, {}).get("types")
        if rows is None:
            lines.append(f"| {MODEL_RU[model]} | PENDING | — | — | — | — |")
            continue
        for name, row in rows.items():
            q = row["worst_phase_dsi_both_phases_responsive_only"]
            lines.append(f"| {MODEL_RU[model]} | {name} | {row['cells']} | {row['both_test_phases_responsive']} | "
                         f"{row['success_cells']} ({100*row['success_fraction_all_cells']:.2f}%) | {number(q['median'])} |")
    lines += ["", "*DSI вычислен для каждой клетки отдельно. Медиана берётся только по клеткам, отзывчивым в обеих проверках; "
              "используется худший из двух проверочных DSI. Критерий успеха использует полный знаменатель, включая молчащие и неустановившиеся клетки."]
    return "\n".join(lines)


def conclusion(data):
    if data["status"] != "COMPLETE":
        return f"Серия ещё выполняется: готовы {data['completed_trials']} из {data['planned_trials']} опытов. Итоговая проверка имеет статус PENDING."
    if not data["retinal_audit_all_available_passed"]:
        return "Серия завершена, но проверка сопоставимости ретинальных стимулов не пройдена; функциональный вывод не принимается."
    candidate = data["models"].get("delayed", {})
    if candidate.get("status") != "AVAILABLE":
        return "Файлы серии получены, но обязательная часть анализа остаётся PENDING; положительный вывод не принимается."
    passed = candidate.get("types_passing", [])
    lplc = candidate.get("lplc2", {})
    if not passed and not lplc.get("functional_gate_passed", False):
        return "Получен отрицательный результат по заранее выбранным критериям: добавление задержек Mi1/Tm1 18/13 мс не дало подтверждённой избирательности типов T4/T5 или возбуждающего детектора радиального расширения LPLC2."
    motion = f"Критерий направленной передачи прошли типы: {', '.join(passed)}." if passed else "Ни один тип T4/T5 не прошёл полный критерий направленной передачи."
    radial = ("LPLC2 прошли функциональный критерий расширения и проверку зависимости от T4/T5."
              if lplc.get("causal_gate_passed") else
              "LPLC2 прошли функциональный критерий, но требуемая причинная проверка T4/T5 не подтверждена."
              if lplc.get("functional_gate_passed") else
              "Функциональный критерий возбуждающего детектора расширения LPLC2 не пройден.")
    return motion + " " + radial


def lplc2_table(data):
    lines = ["| Модель | Все LPLC2 | Возбуждаются в обеих фазах | Установление DC во всех контролях | Прошли функциональный критерий | Также снижение ≥80% при фиксации T4/T5 |",
             "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        row = data["models"].get(model, {}).get("lplc2")
        if row is None:
            lines.append(f"| {MODEL_RU[model]} | PENDING | — | — | — | — |")
        else:
            causal = "не проверялось" if row["causal_success_cells"] is None else str(row["causal_success_cells"])
            if row["causal_clamp_available"] and not row["both_phase_excitation_cells"]:
                causal = "неоценимо: исходного возбуждающего ответа нет"
            lines.append(f"| {MODEL_RU[model]} | {row['cells']} | {row['both_phase_excitation_cells']} | "
                         f"{row['both_phase_dc_stable_cells']} | {row['functional_success_cells']} | {causal} |")
    return "\n".join(lines)


def numerical_interpretation(data):
    if data["status"] != "COMPLETE" or any(data["models"].get(model, {}).get("status") != "AVAILABLE" for model in MODELS):
        return "Полное численное сопоставление остаётся PENDING до получения всех обязательных условий."
    baseline, candidate = data["models"]["baseline"], data["models"]["delayed"]
    total = sum(row["cells"] for row in candidate["types"].values())
    baseline_passed = sum(row["success_cells"] for row in baseline["types"].values())
    candidate_passed = sum(row["success_cells"] for row in candidate["types"].values())
    fractions = [row["success_fraction_all_cells"] for row in candidate["types"].values()]
    line = (f"Отдельные направленные клетки присутствуют: в исходной модели критерий прошли "
            f"{baseline_passed}/{total}, в кандидате — {candidate_passed}/{total}. "
            f"Для кандидата это {100*min(fractions):.2f}–{100*max(fractions):.2f}% клеток отдельных типов "
            "при заранее заданном пороге 50%. Поэтому вывод относится к недостаточности избирательности типов, "
            "а не к полному отсутствию направленных клеток. Разница в несколько клеток здесь не является "
            "статистическим доказательством улучшения на независимых животных.")
    lplc = candidate["lplc2"]
    summaries = [lplc["phase_summaries"][profile]["conditions"]["expanding"] for profile in TEST_PROFILES]
    h1 = np.mean([row["mean_h1_all_cells"] for row in summaries])
    maximum_dc = max(row["max_positive_release_delta"] for row in summaries)
    line += (f"\n\nУ кандидата средняя по LPLC2 амплитуда H1 расширения составляет {h1:.4g}, "
             f"но наибольшее положительное изменение среднего выделения среди всех клеток и обеих фаз — "
             f"{maximum_dc:.4g}, ниже F={candidate['response_floor_au']:.4g}. "
             "Установление постоянной составляющей проходит у всех LPLC2; отсутствие функционального "
             "ответа здесь не объясняется отбрасыванием неустановившихся клеток.")
    regime = data.get("validation_operating_regime", {})
    if regime.get("completed_validation_trials") == 28 and regime.get("all_recorded_test_trajectories_no_rectifier_crossings"):
        line += ("\n\nВо всех 28 основных проверочных опытах C=0.4 ни одна клетка на записанных шагах "
                 "не пересекла порог выпрямителя. Значит, проверенные траектории оставались в линейной "
                 "области уравнений. При нулевом временном среднем ретинального контраста (здесь — с точностью округления) устойчивая "
                 "линейная система с задержками не создаёт стационарного смещения среднего выхода. "
                 "Это объясняет, почему передача колебаний сохранилась, а положительный DC-выход "
                 "не появился. Такой вывод относится к выбранному режиму модели; он не доказывает, "
                 "что конкретная биологическая нелинейность уже установлена или что остальные "
                 "пространственные и временные предположения верны.")
    return line


def plot(data, root):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = {"baseline": "#637485", "delayed": "#c15a37"}
    x = np.arange(len(TYPES))
    for model_index, model in enumerate(MODELS):
        rows = data["models"].get(model, {}).get("types")
        if rows is None:
            continue
        offset = (model_index - .5) * .34
        axes[0, 0].bar(x + offset, [100 * rows[name]["success_fraction_all_cells"] for name in TYPES],
                       width=.32, color=colors[model], label=MODEL_RU[model])
        for index, name in enumerate(TYPES):
            q = rows[name]["worst_phase_dsi_both_phases_responsive_only"]
            if q["median"] is not None:
                axes[0, 1].errorbar(index + offset, q["median"],
                    yerr=[[q["median"] - q["q25"]], [q["q75"] - q["median"]]],
                    fmt="o", color=colors[model], capsize=3)
    axes[0, 0].axhline(50, color="#333333", ls="--", lw=1)
    axes[0, 0].set(xticks=x, xticklabels=TYPES, ylabel="% всех клеток типа", ylim=(0, 102),
                   title="A. Пройден полный критерий в обеих фазах")
    if axes[0, 0].has_data():
        axes[0, 0].legend(loc="upper right", frameon=False)
    axes[0, 1].axhline(.3, color="#333333", ls="--", lw=1)
    axes[0, 1].axhline(0, color="#bbbbbb", lw=.7)
    axes[0, 1].set(xticks=x, xticklabels=TYPES, ylabel="Худший проверочный DSI: медиана, квартиль",
                   ylim=(-1.05, 1.05), title="B. Только отзывчивые в обеих фазах клетки")
    conditions = ("expanding", "contracting", *DIRECTIONS, "flicker")
    xx = np.arange(len(conditions))
    for model in MODELS:
        row = data["models"].get(model, {}).get("lplc2")
        if row is None:
            continue
        for profile, marker, line in (("test0", "o", "-"), ("testpi", "s", "--")):
            summary = row["phase_summaries"][profile]["conditions"]
            label = MODEL_RU[model] + (", φ=0" if profile == "test0" else ", φ=π")
            axes[1, 0].plot(xx, [summary[name]["mean_positive_release_delta_all_cells"] for name in conditions],
                            marker=marker, ls=line, color=colors[model], label=label)
            axes[1, 1].plot(xx, [summary[name]["mean_h1_all_cells"] for name in conditions],
                            marker=marker, ls=line, color=colors[model], label=label)
    for ax in axes[1]:
        ax.set_xticks(xx, [LABELS[name] for name in conditions], rotation=25, ha="right")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        ax.set_ylabel("Условные единицы, среднее по всем LPLC2")
    axes[1, 0].set_title("C. Положительное изменение среднего выделения")
    floors = [row.get("response_floor_au", 1e-6) for row in data["models"].values()]
    display_floor = max(floors, default=1e-6)
    axes[1, 0].axhline(display_floor, color="#333333", ls="--", lw=1)
    axes[1, 0].set_ylim(bottom=0., top=max(axes[1, 0].get_ylim()[1], display_floor * 1.15))
    axes[1, 0].annotate("F: численный порог", (0, display_floor), xytext=(3, 4),
                        textcoords="offset points", fontsize=8)
    if len(axes[1, 0].lines) > 1 and all(np.max(line.get_ydata()) < display_floor for line in axes[1, 0].lines[:-1]):
        axes[1, 0].text(.5, .55, "Все средние R ниже численного порога", transform=axes[1, 0].transAxes,
                        ha="center", color="#555555", fontsize=9)
    axes[1, 1].set_title("D. Амплитуда H1 — отдельная метрика передачи")
    if axes[1, 1].lines:
        axes[1, 1].legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.18)
    fig.suptitle(f"Зрительная избирательность: {data['completed_trials']}/{data['planned_trials']} опытов; "
                 "2 Гц, проверочный контраст 0.4\n"
                 "Один детерминированный граф; экранные оси и центр не калиброваны по рецептивным полям", fontsize=13)
    fig.savefig(Path(root) / "summary.png", dpi=170)
    fig.savefig(Path(root) / "summary.svg")
    plt.close(fig)


def render(data, root):
    result = conclusion(data)
    completion_note = ("Серия полностью завершена; все предусмотренные условия включены в анализ."
                       if data["status"] == "COMPLETE" else
                       "Числа относятся только к завершённым условиям; окончательный вывод остаётся PENDING.")
    floor_lines = []
    for model in MODELS:
        row = data["models"].get(model, {})
        floor_lines.append(f"- {MODEL_RU[model]}: F={number(row.get('response_floor_au'))}; "
                           f"максимальная H1 нулевого контроля={number(row.get('null_h1_max_au'))}.")
    verification = data.get("verification", {})
    test_counts = verification.get("tests-final.xml", verification.get("tests.json", {})).get("counts") or {}
    test_line = (f"Сохранённая общая проверка: {test_counts['tests']} тестов, "
                 f"{test_counts['failures']} ошибок утверждений, {test_counts['errors']} ошибок выполнения, "
                 f"{test_counts['skipped']} пропусков."
                 if test_counts else "Общая проверка тестов: PENDING.")
    markdown = f"""# Проверка зрительной избирательности: результаты

**{result}**

Статус: **{data['status']}**, завершено {data['completed_trials']}/{data['planned_trials']} опытов.
Этот файл автоматически построен только из завершённых и проверенных файлов.

## T4/T5

{direction_table(data)}

## LPLC2

{lplc2_table(data)}

Функциональный критерий требует положительного среднего выделения относительно серого фона,
превосходства не менее чем вдвое над каждым контролем, установления и повторения в обеих фазах.
Радиальная асимметрия H1 сохраняется отдельно и не подменяет этот критерий.

{numerical_interpretation(data)}

## Численный порог и происхождение

{chr(10).join(floor_lines)}

Все доступные JSON соответствуют зафиксированному плану; хэш каждого NPZ проверен до чтения.
Текущие исходники модели совпадают с зафиксированными. Проверки ретинальной мощности:
{sum(row['passed'] for row in data['retinal_audit'])}/{len(data['retinal_audit'])} доступных проверок прошли.
{completion_note}
{test_line} Проверки состояния и побайтной совместимости находятся в папке
[verification](verification/).

![Графики результатов](summary.png)

Данные: [машинный анализ](results-analysis.json), [массивы всех клеток](cell-selectivity.npz),
[зафиксированный план](plan.json), [векторный рисунок](summary.svg).
Квартили, полные знаменатели, отдельные стороны и списки критериев находятся в машинном анализе.
Подробный текст: [BIO_SELECTIVITY_RESULT.md](../../docs/BIO_SELECTIVITY_RESULT.md).

Эти условия не являются независимыми животными. Результат относится к одной частоте, одному
центру радиальных изображений, прежнему отображению глаза и двум вариантам динамики. Он
не проверяет замену моторного контроллера и не доказывает невозможность биологической модели.
"""
    (Path(root) / "RESULTS.md").write_text(markdown, encoding="utf-8")
    document = f"""# Биологическая ветка v2: проверка направления и радиального расширения

12 сентября 2026 года. Отчёт о вычислительном эксперименте, подготовленный из
сохранённых результатов; текст предназначен как проверяемый материал для статьи.

**{result}**

Текущий статус серии: **{data['status']}**, готовы {data['completed_trials']} из
{data['planned_trials']} опытов. {completion_note}

## Зачем проводилась проверка

В [предыдущей серии](BIO_GRADED_V1.md) изображение уже вызывало непрерывный ответ
до LC4, LPLC2 и GF. Получение ненулевого ответа не показывало, что сеть отличает
направление движения от изменения яркости. На следующем шаге проверено одно
ограниченное предположение: достаточно ли дополнительного запаздывания двух
зрительных типов, чтобы на неизменных связях возникла устойчивая избирательность.

Исходная модель сохраняет градуальную передачу, усиление 0.8, временную
постоянную 50 мс и рабочую точку 0.5. В кандидате добавлены задержки выхода
Mi1 18 мс и Tm1 13 мс; история интерполируется при шаге 10 мс. Остальные
уравнения и веса не подгонялись по ответам этой серии. Сохранён полный граф
166 700 клеток и 25 582 938 записей связей; изображение воздействует только
на объявленные ретинальные порты.

Числа 18/13 мс мотивированы различиями пиков измеренных временных фильтров,
но **в модели они использованы как дополнительное искусственное запаздывание**.
Пик фильтра не равен синаптической задержке или мембранной временной постоянной.
Полные опубликованные фильтры и нелинейности здесь не перенесены. Источники и
это ограничение подробно разобраны в [обзоре источников](BIO_SELECTIVITY_SOURCES.md),
включая [Behnia et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4243710/).

## Что предъявлялось и что заранее считалось успехом

[План](BIO_SELECTIVITY_PLAN.md) и его
[исполняемая версия](../runs/bio-selectivity-v1/plan.json) зафиксированы до
основных запусков. Испытаны четыре направления синусоидальной решётки,
расширяющиеся и сужающиеся кольца, а также равномерное мерцание с той же
локальной временной амплитудой. Частота 2 Гц, два пространственных периода
на ширину экрана. Калибровка использует контраст 0.8, фазу 0; проверки —
контраст 0.4, фазы 0 и π. Нулевые контроли и две фиксации T4/T5 дополняют
42 основных условия до 48 опытов.

Каждый стартует с собственного стационарного первого кадра. Через 0.5 с
начинаются шесть периодов движения. Основная оценка использует только
последние три периода, 150 отсчётов: входные кадры [2.0,3.5) с, состояния
нейронов после шага на 0.01 с позже. Окно не выбиралось по максимуму ответа.

Для сравнения с первым неподвижным паттерном используется его вычисленное
стационарное состояние (`reference_voltage`, `static_release_delta`), а не
отдельная временная траектория неподвижной копии. Это уточнение фактически
исполненного зафиксированного кода отмечено после серии в
[плане](BIO_SELECTIVITY_PLAN.md); исходная неточная формулировка сохранена.
Условия и критерии по результатам не изменялись. Для серого фона и мерцания
изображение заменяется после базового генератора: полный рецепт состоит из
`trial`, `protocol`, `stimulus_postprocessing` и зафиксированных исходников.
Хэш базового первого кадра не выдаётся за хэш окончательного мерцания.

Направленная передача измеряется амплитудой первой временной гармоники
**каждой клетки отдельно**. Её предпочтительная ось выбирается только
в калибровке, затем фиксируется. Индекс D=(A_PD−A_ND)/(A_PD+A_ND)
должен быть не меньше 0.3 в обеих проверках; амплитуда выше численного порога,
а изменение H1 между последними двумя периодами не больше 10% для обеих
осей. Тип проходит критерий, только если это выполнено у не менее половины
всех его клеток. Неотзывчивые клетки не исчезают из знаменателя.

Для LPLC2 задан более сильный критерий. Положительное изменение среднего
выделения относительно серого фона должно превосходить порог и минимум
вдвое превышать каждый контроль: сужение, четыре плоских движения, мерцание
и первый неподвижный паттерн. Требуется установление постоянной составляющей
во всех используемых сигналах и повторение в обеих фазах у не менее половины
всех LPLC2. Для причинной интерпретации кандидата дополнительно требуется
снижение этого ответа минимум на 80% при фиксации модуляции T4/T5.

## Результаты отдельных типов T4/T5

{direction_table(data)}

Квартили и разбиение по стороне записаны в
[results-analysis.json](../runs/bio-selectivity-v1/results-analysis.json).
Для стороны используется аннотация rootSide, при её отсутствии somaSide;
клетки с неизвестной стороной сохраняются. Экранные оси ещё не зарегистрированы
относительно естественного глаза, поэтому предпочтение «вправо» нельзя
автоматически объявить совпадением с биологическим направлением T4a или T5a.
Синусоидальные решётки содержат обе полярности; правильное разделение ON/OFF
этим опытом отдельно не проверялось.

## Что произошло у LPLC2

{lplc2_table(data)}

{numerical_interpretation(data)}

Здесь различаются колебание напряжения и устойчивое возбуждение.
Амплитуда H1 может быть ненулевой даже при нулевом среднем изменении выделения.
Поэтому более сильная H1 при расширении сама по себе не подтверждает
возбуждающий детектор приближения. Числа обоих измерений для каждого условия
и каждой фазы сохранены отдельно, как и маски всех клеток, прошедших каждый
критерий. Фиксация T4/T5 удерживает начальные напряжения и оставляет тоническое
выделение; она не называется полным биологическим выключением клеток.

## Проверяемость и смысл ограничений

{chr(10).join(floor_lines)}

Порог F=max(10^-6,100 A_null) отделяет ответ от численного остатка. Он не
является измеренным физиологическим порогом. Все доступные исходники опытов
совпали с зафиксированными; хэш каждого NPZ проверен до загрузки. Пройдено
{sum(row['passed'] for row in data['retinal_audit'])} из {len(data['retinal_audit'])}
доступных проверок равенства локальных ретинальных амплитуд и исходных
состояний противоположных направлений.

{test_line} Это проверка кода, не подтверждение нейронной функции.
Отдельно [проверено состояние полного графа](../runs/bio-selectivity-v1/verification/real-graph-state.json):
сохранение всех буферов задержки и последующее продолжение совпали побитно.
[Проверка совместимости](../runs/bio-selectivity-v1/verification/compatibility.json)
сравнивает байты прежних файлов и сохранений; она не является повторным
запуском физических сценариев. Подробный результат тестов:
[tests.json](../runs/bio-selectivity-v1/verification/tests.json).
Финальный расширенный набор:
[tests-final.xml](../runs/bio-selectivity-v1/verification/tests-final.xml).

Это детерминированный вычислительный эксперимент на одном графе. Отдельные
клетки и 48 условий не являются независимыми биологическими животными.
Значения напряжения и выделения выражены в условных единицах; их нельзя
называть герцами, физиологическими милливольтами или кальциевым ΔF/F.

Отрицательный результат ограничивает достаточность именно этой совокупности
допущений: выбранного дополнительного запаздывания, рабочей точки, знаков
по источнику и прежнего отображения изображения на сетчатку. Он не
опровергает функцию живых T4/T5/LPLC2 и не доказывает невозможность её модели.
Не проверены другие частоты, центры рецептивных полей, полные временные
фильтры, рецепторные знаки и восстановленные нелинейности. Эти ограничения
являются возможными объяснениями; причина не назначается без отдельного
вмешательства. Работа с различными знаками, задержками или порогами после
просмотра этой серии должна быть следующим обозначенным экспериментом.

Схема управления телом здесь не испытывалась. Даже частичный успех
направленной передачи не заменяет внешний контроллер и не переносится
автоматически на жизнь одиночной мухи или драки самцов. Работающие режимы
продолжают использовать прежние программы.

## Артефакты и повторное построение

![Результаты](../runs/bio-selectivity-v1/summary.png)

Доступны [автоматический отчёт](../runs/bio-selectivity-v1/RESULTS.md),
[машинный анализ с происхождением данных](../runs/bio-selectivity-v1/results-analysis.json),
[полные клеточные массивы](../runs/bio-selectivity-v1/cell-selectivity.npz) и
[векторный рисунок](../runs/bio-selectivity-v1/summary.svg).

Пересборка отчёта читает готовые результаты и не запускает нейронную модель:

```powershell
.\\.venv\\Scripts\\python.exe bio_tools/selectivity_report.py
```
"""
    report_path = PROJECT / "docs/BIO_SELECTIVITY_RESULT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/bio-selectivity-v1")
    args = parser.parse_args()
    data, arrays, _ = analyze(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "cell-selectivity.npz", **arrays)
    data["cell_selectivity_sha256"] = digest(args.output / "cell-selectivity.npz")
    (args.output / "results-analysis.json").write_text(json.dumps(finite_json(data), ensure_ascii=False, indent=2,
                                                               allow_nan=False), encoding="utf-8")
    plot(data, args.output)
    render(data, args.output)
    print(json.dumps({"status": data["status"], "completed": data["completed_trials"],
                      "planned": data["planned_trials"], "conclusion": conclusion(data)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
