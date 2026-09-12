"""Where the visual signal is lost between the retina and LC4/LPLC2.

Read-only measurement. Nothing here changes the graph, `fly_arena`, or any
stored parameter; the module records what the unchanged model already does so
that a later fix can be judged against fixed numbers.

Four independent parts:

* ``structure`` — does a path exist at all, and what does LC4/LPLC2/GF receive.
* ``retina_coverage`` — how much of the retinal input the looming object moves,
  isolated from the fly's own motion by a matched run without the object.
* ``pathway`` — firing rate, membrane voltage and synaptic current per
  population during a real approach, including the retinal cells the object
  actually covers.
* ``transfer`` — the retinal current curve and its slope at the scene's
  operating point.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from fly_arena.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PATHWAY_TYPES = ("L1", "L2", "L3", "L5", "Mi1", "Tm2", "Tm3", "Tm4", "TmY3", "T2",
                 "T4c", "T5c", "Tm5Y", "Tm20", "LC4", "LPLC2", "DNp01")


def _types(graph: Path) -> np.ndarray:
    ids = np.load(graph / "ids.npy")
    rows = feather.read_table(graph / "neurons.feather", columns=["bodyId", "type"]).to_pydict()
    by_id = dict(zip(rows["bodyId"], rows["type"]))
    return np.array([by_id.get(int(body)) or "" for body in ids])


def structure(graph: Path, *, max_hops: int = 6) -> dict:
    """Breadth-first reachability from the retinal ports, and the input budget."""
    ptr = np.load(graph / "ptr.npy")
    posts = np.load(graph / "posts.npy", mmap_mode="r")
    counts = np.load(graph / "counts.npy", mmap_mode="r")
    signs = np.load(graph / "signs.npy")
    types = _types(graph)
    ports = json.loads((graph / "ports.json").read_text())
    retina = np.asarray(ports["retina"], np.int64)
    targets = {kind: np.flatnonzero(types == kind) for kind in ("LC4", "LPLC2", "DNp01")}

    seen = np.zeros(len(ptr) - 1, bool)
    seen[retina] = True
    frontier = retina.copy()
    hops, reached = [], {kind: None for kind in targets}
    for hop in range(1, max_hops + 1):
        nxt = np.unique(np.concatenate([posts[ptr[i]:ptr[i + 1]] for i in frontier])) if len(frontier) else np.empty(0, np.int64)
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        frontier = nxt
        hops.append({"hop": hop, "new_cells": int(len(nxt)), "cells_seen": int(seen.sum()),
                     **{kind: f"{int(seen[idx].sum())}/{len(idx)}" for kind, idx in targets.items()}})
        for kind, idx in targets.items():
            if reached[kind] is None and seen[idx].all():
                reached[kind] = hop
        if not len(nxt) or all(value is not None for value in reached.values()):
            break

    gain = 0.075  # BrainConfig default; the audit reports weights, it does not set them
    pre = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
    weight = np.asarray(counts, np.float64) * signs[pre] * gain
    budget = {}
    for kind, idx in targets.items():
        mask = np.isin(np.asarray(posts), idx)
        values, sources = weight[mask], pre[mask]
        excitatory = float(values[values > 0].sum())
        inhibitory = float(values[values < 0].sum())
        by_type: dict[str, float] = {}
        for source, value in zip(sources, values):
            name = types[source] or "(no type)"
            by_type[name] = by_type.get(name, 0.) + float(value)
        top = sorted(by_type.items(), key=lambda item: -abs(item[1]))[:8]
        budget[kind] = {"cells": int(len(idx)), "incoming_edges": int(mask.sum()),
                        "presynaptic_cells": int(len(np.unique(sources))),
                        "excitatory_sum": round(excitatory, 1), "inhibitory_sum": round(inhibitory, 1),
                        "net_per_cell": round((excitatory + inhibitory) / len(idx), 2),
                        "strongest_presynaptic_types": [[name, round(value, 1)] for name, value in top]}
    return {"synapse_gain_used_for_report": gain, "hops": hops,
            "first_hop_fully_reached": reached, "input_budget": budget,
            "interpretation": "Structural reachability and stored weights only. A path and a positive budget "
                              "are not evidence that the running model transmits anything."}


def make_ceiling(simulation, level):
    """Give the arena a visible lid, so the eyes stop staring at empty white.

    The arena already carries a lid geometry at the top; it ships fully
    transparent, so everything above the 3 mm wall renders as the blank
    background of the renderer. Measured on the unchanged scene, that leaves
    69.8% of the retinal inputs pinned at luminance 1.0 with a median of exactly
    1.000, and no code downstream can recover what those inputs never carried.

    This changes the scene, not the model: no gain, threshold or weight is
    touched. Both conditions are always reported side by side.
    """
    if level is None:
        return None
    if not 0 <= level <= 1:
        raise ValueError("Ceiling grey must lie in [0, 1]")
    import mujoco as mj
    model = simulation.arena.sim.mj_model
    geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "invisible_lid")
    if geom < 0:
        raise ValueError("This arena has no lid geometry to make visible")
    model.geom_rgba[geom] = [level, level, level, 1.]
    return float(level)


def _eye_sampler(graph: Path):
    ports = json.loads((graph / "ports.json").read_text())
    eyes = np.asarray(ports["eye"], np.int64)
    uv = np.asarray(ports["uv"], np.float32)
    luma = np.array([.2126, .7152, .0722], np.float32)

    def sample(frames: np.ndarray) -> np.ndarray:
        height, width = frames.shape[1:3]
        x = np.rint(uv[:, 0] * (width - 1)).astype(int)
        y = np.rint(uv[:, 1] * (height - 1)).astype(int)
        return (frames[eyes, y, x].astype(np.float32) / 255) @ luma

    return sample


def retina_coverage(config_path: Path, graph: Path, *, seconds: float, seed: int,
                    hold_body: bool = True, ceiling=None) -> dict:
    """Object-only retinal change, from two matched runs without the connectome.

    The second run removes the looming object and changes nothing else, so the
    difference is the object's contribution for as long as the two bodies agree.
    They stop agreeing as soon as one of them escapes, which happens well before
    the object is large, so ``hold_body`` clamps behavioural movement in both
    runs. The fly then holds its stance, the difference stays purely optical for
    the whole approach, and the divergence time is measured rather than assumed.
    """
    from .simulation import CircuitSimulation

    sample = _eye_sampler(graph)
    steps = round(seconds / .01)
    traces, positions, angles = [], [], []
    for present in (True, False):
        config = load_config(config_path)
        if not present:
            config["environment"]["looming"] = []
        simulation = CircuitSimulation(config, control="observe", seed=seed, brain_enabled=False,
                                       motor_off=hold_body)
        make_ceiling(simulation, ceiling)
        luminance, track, angle = [], [], []
        try:
            for _ in range(steps):
                simulation.step()
                luminance.append(sample(simulation.arena.eyes()))
                track.append(simulation.arena.position[:2].copy())
                angle.append(float(np.max(simulation.paper_probe.angle_deg)))
        finally:
            simulation.close()
        traces.append(np.asarray(luminance))
        positions.append(np.asarray(track))
        angles.append(np.asarray(angle))

    drift = np.linalg.norm(positions[0] - positions[1], axis=1)
    matched = int(np.argmax(drift > .05)) if np.any(drift > .05) else steps
    difference = np.abs(traces[0] - traces[1])
    window = difference[:matched]
    peak = int(np.argmax((window > .02).sum(axis=1))) if matched else 0
    covered = np.flatnonzero(window.max(axis=0) > .2) if matched else np.empty(0, int)
    return {"seconds": seconds, "retinal_inputs": int(difference.shape[1]),
            "body_held_still": hold_body, "ceiling_grey": ceiling,
            "matched_until_s": round(matched * .01, 2),
            "body_drift_threshold_mm": .05,
            "peak_time_s": round(peak * .01, 2),
            "peak_inputs_changed_over_2_percent": int((window[peak] > .02).sum()) if matched else 0,
            "peak_max_delta": round(float(window[peak].max()), 3) if matched else 0.,
            "covered_inputs": int(len(covered)),
            "covered_mean_peak_delta": round(float(window.max(axis=0)[covered].mean()), 3) if len(covered) else 0.,
            "max_angle_deg": round(float(np.max(angles[0])), 2),
            "covered_indices": covered.tolist(),
            "interpretation": "Luminance reaching the retinal ports. Says nothing about whether the model encodes it."}


def measurement_groups(graph: Path, brain, types: np.ndarray, covered: np.ndarray) -> dict:
    """Population index sets shared by every measurement, so they cannot drift.

    A population mean over thousands of cells cannot show a change carried by a
    few dozen, so the retinal inputs the object covers and the lamina cells
    directly postsynaptic to them are kept as their own groups.
    """
    groups = {"retina": brain.retina}
    if len(covered):
        covered_cells = brain.retina[covered]
        groups["retina_covered"] = covered_cells
        groups["retina_uncovered"] = np.setdiff1d(brain.retina, covered_cells)
        ptr = np.load(graph / "ptr.npy")
        posts = np.load(graph / "posts.npy", mmap_mode="r")
        targets = np.unique(np.concatenate([posts[ptr[i]:ptr[i + 1]] for i in covered_cells]))
        lamina = np.flatnonzero(np.isin(types, ["L1", "L2", "L3", "L5"]))
        groups["lamina_covered"] = np.intersect1d(targets, lamina)
        groups["lamina_uncovered"] = np.setdiff1d(lamina, groups["lamina_covered"])
    for name in PATHWAY_TYPES:
        index = np.flatnonzero(types == name)
        if len(index):
            groups[name] = index
    return {name: index for name, index in groups.items() if len(index)}


def pathway(config_path: Path, graph: Path, *, seconds: float, seed: int, covered: np.ndarray,
            matched_until_s: float, hold_body: bool = True, ceiling=None) -> dict:
    """Rate, voltage and current per population during one real approach.

    ``hold_body`` clamps behavioural movement for the same reason as in the
    coverage measurement: an escape part way through changes what the eyes see,
    which would mix a behavioural change into a sensory measurement. Held still,
    the whole approach is observable and the quiet baseline is a real baseline.
    """
    from .simulation import CircuitSimulation

    types = _types(graph)
    config = load_config(config_path)
    simulation = CircuitSimulation(config, control="observe", seed=seed, brain_enabled=True, graph=graph,
                                   motor_off=hold_body)
    make_ceiling(simulation, ceiling)
    brain = simulation.brain
    groups = measurement_groups(graph, brain, types, covered)

    captured: dict[str, np.ndarray] = {}
    original = brain.step

    def recording(*args, **kwargs):
        command, spikes = original(*args, **kwargs)
        captured["spikes"] = spikes
        return command, spikes

    brain.step = recording
    rows = []
    try:
        for _ in range(round(seconds / .01)):
            simulation.step()
            spikes = captured["spikes"]
            row = {"t": round(simulation.organism.clocks.physics_time_s, 3),
                   "angle_deg": round(float(np.max(simulation.paper_probe.angle_deg)), 2),
                   "gf_paper_mv": round(float(np.max(simulation.paper_probe.last["gf_online_mv"])), 4)}
            for name, index in groups.items():
                row[f"hz_{name}"] = round(float(spikes[index].sum() / len(index) / .01), 2)
            rows.append(row)
        final = {name: {"cells": int(len(index)),
                        "mean_voltage_mv": round(float(brain.voltage[index].mean()), 3),
                        "mean_current_mv": round(float(brain.current[index].mean()), 3)}
                 for name, index in groups.items()}
    finally:
        simulation.close()

    time = np.array([row["t"] for row in rows])
    approach = np.array([row["angle_deg"] for row in rows]) > 0
    quiet = (time > .1) & (time < time[approach].min() - .1) if approach.any() else time < time.max()
    looming = approach if hold_body else approach & (time <= matched_until_s)
    comparison = {}
    for name in groups:
        values = np.array([row[f"hz_{name}"] for row in rows])
        # Consecutive exchanges alternate; compare like parity with like parity.
        parity = {}
        for offset, label in ((0, "even"), (1, "odd")):
            rest = values[offset::2][quiet[offset::2]]
            active = values[offset::2][looming[offset::2]]
            if len(rest) and len(active):
                parity[label] = {"rest_hz": round(float(rest.mean()), 2),
                                 "looming_hz": round(float(active.mean()), 2),
                                 "change_hz": round(float(active.mean() - rest.mean()), 2),
                                 "rest_sd_hz": round(float(rest.std()), 2)}
        alternation = abs(float(values[0::2].mean() - values[1::2].mean()))
        comparison[name] = {"period_2_alternation_hz": round(alternation, 2), **parity}
    return {"seconds": seconds, "seed": seed, "body_held_still": hold_body,
            "ceiling_grey": ceiling, "final_state": final,
            "quiet_vs_looming": comparison, "trace": rows,
            "interpretation": "Population means during one approach on one seed. A near-zero change in a "
                              "population the object does not cover is expected; the covered subset is the test."}


def transfer(visual_gain: float = 28.0, half: float = .1, scene_luminance: float = .77,
             covered_drop: float = .44) -> dict:
    """The retinal current curve and its slope where the scene actually sits."""
    curve = lambda value: visual_gain * value / (half + value)
    slope = lambda value: visual_gain * half / (half + value) ** 2
    samples = [round(x, 2) for x in (0., .05, .1, .2, .33, .5, scene_luminance, 1.)]
    return {"formula": f"{visual_gain} * L / ({half} + L)",
            "curve": [{"luminance": value, "drive_mv": round(curve(value), 2),
                       "slope_mv_per_luminance": round(slope(value), 2)} for value in samples],
            "scene_luminance": scene_luminance,
            "drive_at_scene_mv": round(curve(scene_luminance), 2),
            "covered_drop": covered_drop,
            "drive_after_drop_mv": round(curve(scene_luminance - covered_drop), 2),
            "drive_change_mv": round(curve(scene_luminance - covered_drop) - curve(scene_luminance), 2),
            "slope_at_scene": round(slope(scene_luminance), 2),
            "slope_at_dark_end": round(slope(half), 2),
            "sensitivity_ratio": round(slope(half) / slope(scene_luminance), 1)}


def diagnose(data: Path, output: Path, *, config: Path, seconds: float = 2.5, seed: int = 1,
             skip_structure: bool = False, hold_body: bool = True, ceiling=None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = Path(data) / "graph"
    report: dict = {"schema_version": 1, "config": str(config), "seed": seed, "seconds": seconds,
                    "ceiling_grey": ceiling}

    if not skip_structure:
        print("structure: breadth-first reachability and input budget", flush=True)
        report["structure"] = structure(graph)
        for row in report["structure"]["hops"]:
            print("   ", row, flush=True)

    print("retina: isolating the object with a matched run that has no object", flush=True)
    coverage = retina_coverage(config, graph, seconds=seconds, seed=seed, hold_body=hold_body,
                               ceiling=ceiling)
    report["retina_coverage"] = coverage
    print(f"    {coverage['covered_inputs']} of {coverage['retinal_inputs']} inputs covered, "
          f"mean peak delta {coverage['covered_mean_peak_delta']}, "
          f"bodies matched until {coverage['matched_until_s']} s", flush=True)

    print("pathway: recording every population through one real approach", flush=True)
    report["pathway"] = pathway(config, graph, seconds=seconds, seed=seed,
                                covered=np.asarray(coverage["covered_indices"], int),
                                matched_until_s=coverage["matched_until_s"], hold_body=hold_body,
                                ceiling=ceiling)
    report["transfer"] = transfer()

    (output / "diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'population':18} {'cells':>6} {'rest Hz':>8} {'loom Hz':>8} {'change':>8} {'V mV':>7} {'I mV':>7}")
    for name, state in report["pathway"]["final_state"].items():
        compare = report["pathway"]["quiet_vs_looming"].get(name, {}).get("odd", {})
        print(f"{name:18} {state['cells']:6} {compare.get('rest_hz', float('nan')):8.2f} "
              f"{compare.get('looming_hz', float('nan')):8.2f} {compare.get('change_hz', float('nan')):+8.2f} "
              f"{state['mean_voltage_mv']:7.3f} {state['mean_current_mv']:7.3f}")
    print(f"\nretinal drive at the scene: {report['transfer']['drive_at_scene_mv']} mV, "
          f"a {report['transfer']['covered_drop']} luminance drop buys "
          f"{report['transfer']['drive_change_mv']} mV; slope there is "
          f"{report['transfer']['sensitivity_ratio']}x below the dark end")
    print(f"saved {output / 'diagnosis.json'}", flush=True)
    return report
