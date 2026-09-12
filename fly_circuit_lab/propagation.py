"""Direct diagnostic injections isolate graph transmission from visual encoding.

28 mV is an explicitly engineered test current, not a coefficient from the
paper. Near-333-Hz responses reflect saturation of the legacy LIF model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from fly_arena.brain import Brain


def propagation(graph: Path, output: Path, *, seed=1):
    graph, output = Path(graph), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    brain = Brain(graph, seed=seed)
    table = feather.read_table(graph / "neurons.feather", columns=["bodyId", "type"])
    mapping = dict(zip(table["bodyId"].to_pylist(), table["type"].to_pylist()))
    types = np.array([mapping[int(i)] for i in brain.ids])
    selected = {kind: np.flatnonzero(types == kind).astype(np.int32) for kind in ("LC4", "LPLC2", "DNp01")}
    initial = brain.get_state()
    results = {}
    cases = [("neutral", [], []), ("lc4", ["LC4"], []), ("lplc2", ["LPLC2"], []),
             ("both", ["LC4", "LPLC2"], []), ("both_gf_off", ["LC4", "LPLC2"], ["DNp01"]),
             ("both_inputs_off", ["LC4", "LPLC2"], ["LC4", "LPLC2"])]
    for name, inputs, blocked in cases:
        brain.set_state(initial)
        total = np.zeros(len(brain.ids), np.int64)
        currents = {kind: (selected[kind], 28.) for kind in inputs}
        mask = np.concatenate([selected[k] for k in blocked]) if blocked else np.empty(0, np.int32)
        for _ in range(30):
            _, counts = brain.step(np.zeros(len(brain.retina)), sensory_currents=currents, blocked_outputs=mask)
            total += counts
        results[name] = {kind + "_Hz": float(total[indices].mean() / .3) for kind, indices in selected.items()}
        print(name, results[name], flush=True)
    checks = {"direct_lc4_reaches_gf": results["lc4"]["DNp01_Hz"] > results["neutral"]["DNp01_Hz"],
              "direct_lplc2_reaches_gf": results["lplc2"]["DNp01_Hz"] > results["neutral"]["DNp01_Hz"],
              "blocked_source_and_target_are_silent": results["both_inputs_off"]["DNp01_Hz"] == results["both_gf_off"]["DNp01_Hz"] == 0}
    report = {"seed": seed, "seconds_per_condition": .3, "injection_mv": 28.,
              "protocol": "Direct engineering current into selected LC4/LPLC2; zero retinal luminance; full graph unchanged",
              "initial_voltage_sha256": hashlib.sha256(initial["voltage"].tobytes()).hexdigest(),
              "body_ids": {kind: brain.ids[indices].tolist() for kind, indices in selected.items()},
              "graph_arrays": brain.manifest["arrays"], "results": results, "checks": checks,
              "interpretation": "Tests transmission under artificial input only. It does not reproduce visual tuning, biophysical amplitudes or natural spike rates, and is never wired into the existing fly's runtime."}
    (output / "propagation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert all(checks.values()), checks
    return report
