"""Matched-image neural experiments; no body/behaviour controller is run."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .graph import Connectome
from .model import GradedConfig, GradedNetwork
from .stimuli import FrameProtocol, matched_retinal_flash, retinal_movie


GROUPS = {"L1": ["L1"], "L2": ["L2"], "L3": ["L3"], "Mi1": ["Mi1"],
          "Tm1": ["Tm1"], "Tm2": ["Tm2"], "Tm3": ["Tm3"], "T2": ["T2"],
          "T4": [f"T4{x}" for x in "abcd"], "T5": [f"T5{x}" for x in "abcd"],
          "LC4": ["LC4"], "LPLC2": ["LPLC2"], "GF": ["DNp01"]}


def sources():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def make_movie(kind, graph, *, eye="both", contrast=.8):
    base_kind = "dark_loom" if kind in {"retinal_mean_flash", "on_step", "off_step"} else kind
    protocol = FrameProtocol(kind=base_kind, eye=eye, contrast=contrast)
    times = protocol.times(.01)
    frames = protocol.frames(times)
    metadata = protocol.protocol_config
    if kind == "retinal_mean_flash":
        frames = matched_retinal_flash(frames, graph.ports)
        metadata["postprocess"] = "per-eye mean over exact retinal ports, no neural output used"
    elif kind in {"on_step", "off_step"}:
        frames.fill(.5)
        value = .6 if kind == "on_step" else .4
        for eye_index in ([0, 1] if eye == "both" else [0] if eye == "left" else [1]):
            frames[times >= protocol.pre_s, eye_index] = value
        metadata["postprocess"] = f"uniform display luminance .5 -> {value} at {protocol.pre_s}s"
    metadata["condition"] = kind
    metadata["actual_initial_frame_sha256"] = hashlib.sha256(frames[0].tobytes()).hexdigest()
    return times, frames, metadata


def run_case(graph, output, *, kind="dark_loom", eye="both", config=None,
             contrast=.8, intervention="none"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cfg = config or GradedConfig()
    times, frames, display = make_movie(kind, graph, eye=eye, contrast=contrast)
    luminance = retinal_movie(frames, graph.ports)
    if cfg.dt_s != .01:
        raise ValueError("This display benchmark is sampled at exactly 10 ms")
    freeze = {
        "none": np.empty(0, np.int32), "input_off": np.empty(0, np.int32),
        "retina": np.asarray(graph.ports["retina"], np.int32),
        "lamina": graph.indices(["L1", "L2", "L3", "L4", "L5"]),
        "lc4_lplc2": graph.indices(["LC4", "LPLC2"]),
        "gf": graph.indices("DNp01"),
    }
    if intervention not in freeze:
        raise ValueError(f"Unknown intervention: {intervention}")
    frozen_indices = freeze[intervention]
    model, control = GradedNetwork(graph, cfg), GradedNetwork(graph, cfg)
    resting = model.equilibrate(luminance[0], input_off=intervention == "input_off")
    initial = model.get_state()
    reference = initial["voltage"].copy()
    control.set_state(initial)
    groups = {"retina": np.asarray(graph.ports["retina"], np.int32),
              **{label: graph.indices(types) for label, types in GROUPS.items()}}
    covered = np.flatnonzero(np.max(np.abs(luminance - luminance[0]), axis=0) > .02)
    groups["retina_covered"] = groups["retina"][covered]
    series = {name: {key: [] for key in ("mean", "rms", "maximum_abs", "release_mean")}
              for name in groups if len(groups[name])}
    regime = {name: {"min_voltage_au": float(np.min(reference[index])), "max_fraction_below_release": 0.}
              for name, index in groups.items() if len(index)}
    total_minimum = float(reference.min())
    max_clipped_cells = int(np.count_nonzero(reference <= 0))
    ever_clipped = reference <= 0
    external_guard = np.ones(model.n, bool)
    external_guard[model.retina] = False
    for light in luminance:
        model.step(light, input_off=intervention == "input_off",
                   freeze_indices=frozen_indices, freeze_voltage=reference)
        control.step(luminance[0], input_off=intervention == "input_off",
                     freeze_indices=frozen_indices, freeze_voltage=reference)
        delta = model.voltage - control.voltage
        release_delta = np.maximum(model.voltage, 0.) - np.maximum(control.voltage, 0.)
        total_minimum = min(total_minimum, float(model.voltage.min()))
        max_clipped_cells = max(max_clipped_cells, int(np.count_nonzero(model.voltage <= 0)))
        ever_clipped |= model.voltage <= 0
        if np.any(model.last_external_drive[external_guard] != 0):
            raise ArithmeticError("A visual stimulus was injected beyond the retinal ports")
        for label, metrics in series.items():
            values = delta[groups[label]].astype(np.float64)
            regime[label]["min_voltage_au"] = min(regime[label]["min_voltage_au"], float(model.voltage[groups[label]].min()))
            regime[label]["max_fraction_below_release"] = max(regime[label]["max_fraction_below_release"],
                                                               float(np.mean(model.voltage[groups[label]] <= 0)))
            metrics["mean"].append(float(np.mean(values)))
            metrics["rms"].append(float(np.sqrt(np.mean(values * values))))
            metrics["maximum_abs"].append(float(np.max(np.abs(values))))
            metrics["release_mean"].append(float(np.mean(release_delta[groups[label]], dtype=np.float64)))
    summary = {}
    active = times >= display["pre_s"]
    for label, metrics in series.items():
        mean, rms = np.array(metrics["mean"]), np.array(metrics["rms"])
        peak = int(np.argmax(rms))
        summary[label] = {"cells": int(len(groups[label])),
                          "baseline_voltage_mean_au": float(np.mean(reference[groups[label]])),
                          "peak_rms_delta_au": float(rms[peak]),
                          "peak_rms_at_s": float(times[peak] + cfg.dt_s),
                          "mean_at_peak_au": float(mean[peak]),
                          "active_mean_delta_au": float(mean[active].mean()),
                          "final_mean_delta_au": float(mean[-1]),
                          "maximum_cell_abs_delta_au": float(max(metrics["maximum_abs"]))}
        summary[label].update(regime[label])
    setting_digest = hashlib.sha256(json.dumps({"model": asdict(cfg), "display": display,
                                               "intervention": intervention}, sort_keys=True).encode()).hexdigest()[:10]
    tag = f"{kind}-{eye}-{intervention}-g{cfg.gain:g}-{setting_digest}"
    matrices = {f"{name}_{key}": np.asarray(values) for name, row in series.items() for key, values in row.items()}
    np.savez_compressed(output / f"{tag}.npz", times_s=times + cfg.dt_s,
                        retinal_luminance=luminance, covered_retina_port_indices=covered,
                        final_voltage=model.voltage, reference_voltage=reference, **matrices)
    from PIL import Image
    # Representative stimulus, not an invented illustration of neural output.
    Image.fromarray(np.rint(frames[-1, 0] * 255).astype(np.uint8)).save(output / f"{tag}.png")
    clipped_types, clipped_counts = np.unique(graph.types[ever_clipped], return_counts=True)
    result = {"condition": kind, "eye": eye, "intervention": intervention,
              "intervention_definition": ("none" if intervention == "none" else
                "zero all external retinal drive; tonic/recurrent neural drive remains" if intervention == "input_off" else
                "hold selected cells at their own pre-stimulus voltage; removes modulation, not tonic release"),
              "frozen_cells": int(len(frozen_indices)), "model": asdict(cfg),
              "model_family": "graded voltage with rectified release, normalized incoming anatomical counts, balanced tonic drive",
              "units": "arbitrary continuous voltage/release; no spikes, Hz or physiological mV",
              "display": display, "resting_state": resting,
              "retinal_inputs": len(model.retina), "changing_retinal_inputs": int(len(covered)),
              "input_only_at_retina": True,
              "operating_regime": {"min_voltage_au": total_minimum,
                                    "maximum_cells_below_release_threshold": max_clipped_cells,
                                    "ever_below_release_by_type": dict(zip(clipped_types.tolist(), clipped_counts.tolist())),
                                    "rectifier_active": max_clipped_cells > 0,
                                    "interpretation": "If no voltage crosses zero, this operating regime is a linear dynamical relay"},
              "all_nodes_and_edges_retained": {"nodes": graph.n, "edges": graph.edge_count},
              "populations": summary, "sources": sources(),
              "trace": f"{tag}.npz", "wall_seconds": time.perf_counter() - started,
              "scope": "Matched initial-image control; a nonzero response is transmission, not proof of looming selectivity or a motor policy replacement"}
    (output / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"case": tag, "covered": len(covered),
                      **{x: summary[x]["peak_rms_delta_au"] for x in ("Mi1", "Tm1", "T4", "T5", "LC4", "LPLC2", "GF")},
                      "wall_s": round(result["wall_seconds"], 2)}), flush=True)
    return result


def benchmark(graph_path, output, *, cases=None, gain=.8):
    graph = Connectome(graph_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = sources()
    config = GradedConfig(gain=gain)
    selected = cases or [(kind, "both", "none") for kind in
        ("uniform", "on_step", "off_step", "dark_loom", "bright_loom", "shrinking", "translating", "retinal_mean_flash")]
    (output / "plan.json").write_text(json.dumps({"sources": original, "model": asdict(config),
        "cases": selected, "graph": graph.metadata,
        "interpretation_rule": "Do not equate transmission with looming selectivity; compare matched static, shrinking, translating and dose-matched flash"}, indent=2), encoding="utf-8")
    reports = [run_case(graph, output, kind=kind, eye=eye, intervention=intervention, config=config)
               for kind, eye, intervention in selected]
    if sources() != original:
        raise RuntimeError("Source changed during the batch; retain as development evidence, do not call it frozen validation")
    (output / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports
