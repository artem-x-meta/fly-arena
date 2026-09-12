"""Numerical replication report for the published GF equations and interventions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .gf_model import evaluate, centered_smooth, parameters
from .protocols import looming, linear_expansion


def reproduce(output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    p = parameters()
    reports = []
    protocols = (10, 40, 70, 100, 140)  # Explicit protocol values reported in STAR Methods.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ratio, ax in zip(protocols, axes.flat):
        t, angle, speed, protocol = looming(ratio)
        base = evaluate(t, angle, speed)
        report = {"protocol": protocol, "conditions": {}}
        for label, block, color in (("control", (), "#247c88"), ("LC4 off (+I2 off)", ("lc4",), "#a57940"),
                                     ("LPLC2 off", ("lplc2",), "#94709a"), ("input off", ("input",), "#777777"),
                                     ("output blocked", ("output",), "#bb465c")):
            r = base if not block else evaluate(t, angle, speed, block=block)
            smoothed = centered_smooth(r["gf_delta_mv"], 5)
            during = (t >= protocol["loom_start_s"]) & (t <= protocol["loom_end_s"] + .05)
            peak = np.flatnonzero(during)[np.argmax(smoothed[during])]
            lf_peak = np.argmax(r["lplc2_mv"])
            delayed_size_at_lf_peak = float(np.interp(t[lf_peak] - p["delay_lplc2_s"], t, angle))
            report["conditions"][label] = {"peak_gf_delta_mv": float(smoothed[peak]),
                                           "peak_time_relative_collision_s": float(t[peak]),
                                           "lplc2_delayed_angle_at_peak_deg": delayed_size_at_lf_peak,
                                           "rmse_vs_control_equations_mv": float(np.sqrt(np.mean((r["gf_delta_mv"] - base["gf_delta_mv"])**2)))}
            ax.plot(t * 1000, smoothed, color=color, label=label, lw=1.2)
            name = label.lower().split()[0] + ("_off" if block and "output" not in block else "")
            r["gf_centered_mv"] = smoothed
            with (output / f"rv{ratio}_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(r)
                writer.writerows(zip(*r.values()))
        ax.set(title=f"r/v = {ratio} ms", xlabel="Time relative to collision (ms)", ylabel="GF change from model rest (mV)")
        ax.axvline(protocol["loom_end_s"] * 1000, color="#bbbbbb", ls=":")
        ax.grid(alpha=.2)
        ax.set_xlim(max(protocol["loom_start_s"], -ratio / 1000 * 5) * 1000, 70)
        reports.append(report)
    t, angle, speed = linear_expansion(100)
    r = evaluate(t, angle, speed)
    ax = axes.flat[-1]
    for name, color in (("lc4_mv", "#357fb3"), ("lplc2_mv", "#dd9c43"), ("i1_mv", "#888888"), ("i2_mv", "#b95b80")):
        ax.plot(angle, r[name], label=name, color=color)
    ax.set(title="100 deg/s: separated fitted inputs", xlabel="Full angular diameter (degrees)", ylabel="Component contribution (mV)", xlim=(5,90))
    ax.legend(fontsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Ache et al. 2019, equations 3-7 / independent numerical reimplementation / no experimental recordings fitted")
    fig.savefig(output / "gf-reproduction.png", dpi=170)
    fig.savefig(output / "gf-reproduction.svg")
    plt.close(fig)
    checks = {
        "size_component_peaks_at_42_deg": all(abs(r["conditions"]["control"]["lplc2_delayed_angle_at_peak_deg"] - 42) < 1.0 for r in reports),
        "input_ablation_zero_delta": all(abs(r["conditions"]["input off"]["peak_gf_delta_mv"]) < 1e-12 for r in reports),
        "output_block_zero_delta": all(abs(r["conditions"]["output blocked"]["peak_gf_delta_mv"]) < 1e-12 for r in reports),
        "component_ablations_change_trace": all(r["conditions"]["LC4 off (+I2 off)"]["rmse_vs_control_equations_mv"] > .001 and
                                               r["conditions"]["LPLC2 off"]["rmse_vs_control_equations_mv"] > .001 for r in reports),
    }
    report = {"model": p, "protocols": reports, "checks": checks,
              "status": "equations_reproduced" if all(checks.values()) else "failed",
              "recordings_validation": "not_run: original individual patch-clamp recordings were not obtained",
              "original_code_comparison": "not_run: advertised ModelDB deposit was not located during this bounded audit",
              "smoothing": "2.5 ms centered 5-sample average on a 0.5-ms grid; online adapter uses trailing average"}
    (output / "reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)
    if not all(checks.values()):
        raise AssertionError("Published-equation replication checks failed")
    return report
