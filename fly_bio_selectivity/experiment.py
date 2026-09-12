"""Preregistered comparison of unchanged and additional-delay neural models."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fly_bio.graph import Connectome
from fly_bio.model import GradedConfig, GradedNetwork
from .model import DelayedGradedNetwork, BEHNIA_DELAY_BY_TYPE_S
from .stimuli import MotionProtocol, retinal_movie
from .metrics import temporal_metrics

ROOT = Path(__file__).resolve().parents[1]
# Clockwise display axes. Opposite indices are (0,2) and (1,3).
DIRECTIONS = ("right", "down", "left", "up")
TYPES = tuple(f"T{k}{s}" for k in (4, 5) for s in "abcd")
RECORD_TYPES = ("L1", "L2", "Mi1", "Tm1", "Tm2", "Tm3", *TYPES, "LC4", "LPLC2", "DNp01")
PROFILES = {"cal": (.8, 0.), "test0": (.4, 0.), "testpi": (.4, float(np.pi))}


def sources():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("fly_bio", "fly_bio_selectivity")
            for p in sorted((ROOT / directory).glob("*.py"))}


def trials():
    rows = [{"model": model, "profile": profile, "condition": condition, "intervention": "none"}
            for model in ("baseline", "delayed") for profile in PROFILES
            for condition in (*DIRECTIONS, "expanding", "contracting", "flicker")]
    rows += [{"model": model, "profile": "cal", "condition": condition,
              "intervention": "input_off" if condition == "right" else "none"}
             for model in ("baseline", "delayed") for condition in ("gray", "right")]
    rows += [{"model": "delayed", "profile": profile, "condition": "expanding", "intervention": "t4t5"}
             for profile in ("test0", "testpi")]
    return [{"index": index, **row} for index, row in enumerate(rows)]


def protocol_for(trial):
    contrast, phase = PROFILES[trial["profile"]]
    condition = trial["condition"]
    radial = condition in {"expanding", "contracting"}
    return MotionProtocol(kind="rings" if radial else "grating",
        direction=condition if condition in (*DIRECTIONS, "expanding", "contracting") else "right",
        contrast=contrast, phase_rad=phase, temporal_hz=2., spatial_cycles=2.,
        pre_s=.5, stimulus_s=3., post_s=0.)


def make_inputs(trial, ports):
    protocol = protocol_for(trial)
    times = protocol.times(.01)
    frames = protocol.frames(times)
    if trial["condition"] == "flicker":
        motion_time = np.clip(times - protocol.pre_s, 0, protocol.stimulus_s)
        level = .5 + .5 * protocol.contrast * np.cos(protocol.phase_rad - 2 * np.pi * protocol.temporal_hz * motion_time)
        frames[:] = level[:, None, None, None, None]
    elif trial["condition"] == "gray":
        frames.fill(.5)
    light = retinal_movie(frames, ports)
    mask = protocol.analysis_mask(times, discard_cycles=3)
    if int(mask.sum()) != 150:
        raise ValueError("The fixed analysis requires exactly 150 samples / three cycles")
    return protocol, times, frames, light, mask


def make_plan(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = {"format": 1, "sources": sources(), "trials": trials(),
            "model_config": asdict(GradedConfig()), "candidate_delays_s": dict(BEHNIA_DELAY_BY_TYPE_S),
            "delay_scope": "Sensitivity proxy of published filter peak differences; not measured axonal delays, membrane taus or complete fitted filters",
            "directions": list(DIRECTIONS), "profiles": PROFILES,
            "protocol": {"temporal_hz": 2., "spatial_cycles_per_display_width": 2., "pre_s": .5,
                         "motion_s": 3., "post_s": 0., "discard_cycles": 3, "analyze_cycles": 3,
                         "frame_window_s": [2., 3.5], "voltage_sample_window_s": [2.01, 3.5]},
            "criteria": {"absolute_response_floor_au": 1e-6, "null_multiplier": 100.,
                "dsi_minimum": .3, "per_type_all_cell_success_fraction": .5,
                "stationarity_h1_relative_maximum": .1,
                "lplc2_expansion_over_each_control_minimum": 2.,
                "lplc2_dc_stationarity_absolute": "max(F, .1*max(abs(last_cycle),abs(previous_cycle)))",
                "lplc2_t4t5_clamp_reduction_minimum": .8},
            "interpretation": ["PD selected only at calibration contrast .8/phase0 and locked for both test phases at .4",
                "No weights, delays, thresholds or axes tuned from the main trial outputs",
                "Nonresponsive/nonstationary cells remain in full population denominators",
                "Per-cell harmonic direction transmission is weaker than an excitatory radial-expansion detector",
                "Display axes/center are not independently calibrated anatomical receptive fields",
                "Deterministic cells and trials are not independent biological animals"]}
    path = output / "plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(plan)):
            raise ValueError("An incompatible plan exists; choose a new output directory")
    else:
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def run_trial(graph, trial, output, plan):
    output = Path(output)
    tag = f"{trial['index']:02d}-{trial['model']}-{trial['profile']}-{trial['condition']}-{trial['intervention']}"
    result_path = output / f"{tag}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result["sources"] != plan["sources"] or result["trial"] != trial
                or hashlib.sha256((output / result["trace"]).read_bytes()).hexdigest() != result["trace_sha256"]):
            raise ValueError("Existing trial differs from the frozen source/plan or its trace")
        print(f"Verified existing {tag}", flush=True)
        return result
    if sources() != plan["sources"]:
        raise RuntimeError("Model/protocol sources changed after preregistration")
    started = time.perf_counter()
    protocol, times, frames, luminance, mask = make_inputs(trial, graph.ports)
    config = GradedConfig()
    model = GradedNetwork(graph, config) if trial["model"] == "baseline" else DelayedGradedNetwork(graph, config)
    input_off = trial["intervention"] == "input_off"
    resting = model.equilibrate(luminance[0], input_off=input_off)
    reference = model.voltage.copy()
    selected = graph.indices(RECORD_TYPES)
    frozen = graph.indices(TYPES) if trial["intervention"] == "t4t5" else np.empty(0, np.int32)
    recorded = np.empty((int(mask.sum()), len(selected)), np.float32)
    compact = np.empty((len(times), 3), np.float64)
    compact_indices = [graph.indices(name) for name in ("LPLC2", "LC4", "DNp01")]
    clipped = np.zeros(graph.n, bool)
    recording_index = 0
    other_cells = np.ones(graph.n, bool)
    other_cells[np.asarray(graph.ports["retina"], int)] = False
    for index, light in enumerate(luminance):
        model.step(light, input_off=input_off, freeze_indices=frozen, freeze_voltage=reference)
        if np.any(model.last_external_drive[other_cells]):
            raise AssertionError("Visual input bypassed the retinal boundary")
        clipped |= model.voltage <= 0
        compact[index] = [float(np.mean(model.voltage[cells] - .5, dtype=np.float64)) for cells in compact_indices]
        if mask[index]:
            recorded[recording_index] = model.voltage[selected]
            recording_index += 1
    late_times = times[mask] + config.dt_s
    metrics = temporal_metrics(recorded, late_times, protocol.temporal_hz, gray_voltage=config.resting_voltage)
    retinal_metrics = temporal_metrics(luminance[mask], late_times, protocol.temporal_hz, gray_voltage=.5)
    path = output / f"{tag}.npz"
    np.savez_compressed(path, ids=graph.ids[selected], types=graph.types[selected], indices=selected,
        times_s=late_times, voltage=recorded, reference_voltage=reference[selected],
        static_release_delta=np.maximum(reference[selected], 0) - config.resting_voltage,
        retinal_h1=retinal_metrics["h1"], retinal_mean=retinal_metrics["mean_voltage_delta"],
        retinal_luminance=luminance[mask], all_times_s=times+config.dt_s,
        population_mean_delta=compact, **metrics)
    clipped_types, clipped_counts = np.unique(graph.types[clipped], return_counts=True)
    result = {"trial": trial, "sources": sources(), "model_config": asdict(config),
        "extra_output_delays_s": dict(BEHNIA_DELAY_BY_TYPE_S) if trial["model"] == "delayed" else {},
        "graph": graph.metadata, "protocol": protocol.protocol_config, "resting_state": resting,
        "stimulus_postprocessing": ("uniform same-amplitude temporal flicker" if trial["condition"] == "flicker" else
                                    "constant .5 gray" if trial["condition"] == "gray" else "none"),
        "analysis_samples": recording_index, "analysis_cycles": 3,
        "neurons_recorded": len(selected), "all_neurons_simulated": graph.n,
        "all_edges_retained": graph.edge_count, "input_only_at_retina": True,
        "freeze_definition": "selected T4/T5 voltages held at own stationary initial-image baseline; tonic release retained",
        "frozen_cells": len(frozen), "ever_rectified_by_type": dict(zip(clipped_types.tolist(), clipped_counts.tolist())),
        "trace": path.name, "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "wall_seconds": time.perf_counter()-started,
        "units": "continuous voltage/release in arbitrary units, not spikes/Hz/physiological mV"}
    if result["sources"] != plan["sources"]:
        raise RuntimeError("Sources changed during trial; no validated result will be published")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed {tag}; {recording_index} samples; {result['wall_seconds']:.2f}s wall", flush=True)
    return result


def run(output, graph_path, *, model_filter="all", indices=None):
    plan = make_plan(output)
    graph = Connectome(graph_path)
    chosen = [trial for trial in plan["trials"] if
              (model_filter == "all" or trial["model"] == model_filter)
              and (indices is None or trial["index"] in indices)]
    return [run_trial(graph, trial, output, plan) for trial in chosen]
