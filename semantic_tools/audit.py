"""Read-only baseline audit of the existing arena; writes reports only.

Run from the project root with the existing .venv. No model/graph changes,
downloads, benchmark training, video or checkpoint writes occur here.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "semantic_channel"
SNAPSHOTS = (
    "runs/search/checkpoints/latest.npz",
    "runs/escape-final/full-graph/checkpoints/latest.npz",
    "runs/circuit-lab/live/checkpoints/latest.npz",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(name, document):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(document, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def protected_paths():
    selected = []
    for directory in ("fly_arena", "fly_circuit_lab", "fly_compat", "fly_duel",
                      "fly_bio", "fly_bio_selectivity", "bio_tools", "tests",
                      "circuit_tests", "compat_tests", "duel_tests", "bio_tests",
                      "selectivity_tests"):
        selected.extend((ROOT / directory).glob("*.py"))
    for directory in ("configs", "scenarios", "circuit_configs"):
        selected.extend((ROOT / directory).glob("*.toml"))
    selected.extend(ROOT.glob("[0-1][0-9]_*.cmd"))
    selected.extend(ROOT / name for name in ("pyproject.toml", "constraints-tested.txt"))
    selected.extend(ROOT / name for name in SNAPSHOTS)
    selected.extend((ROOT / "data" / "graph").glob("*"))
    return sorted(set(path for path in selected if path.is_file()))


def freeze():
    if (OUTPUT / "protected-manifest.json").exists():
        raise FileExistsError("Baseline already frozen; use check instead of replacing it")
    import numpy as np
    from fly_arena.brain import BrainConfig
    from fly_arena.checkpoint import code_digest
    from fly_arena.config import config_digest, load_config

    cfg = load_config(ROOT / "configs" / "ethology-unscaled.toml")
    graph = ROOT / "data" / "graph"
    manifest = json.loads((graph / "manifest.json").read_text(encoding="utf-8"))
    protected = {path.relative_to(ROOT).as_posix(): sha256(path) for path in protected_paths()}
    prior_core = "04a519569f0a52b75d6b6666f5f3fe6523b03236cfd5fcd8daacc3a831c789ff"
    document = {"schema_version": 1, "created_utc": timestamp(), "core_code_digest": code_digest(),
                "expected_previous_core_digest": prior_core,
                "previous_core_matches": code_digest() == prior_core,
                "files": protected}
    write_json("protected-manifest.json", document)
    write_json("baseline-config.json", {"arena": cfg, "brain": asdict(BrainConfig())})
    arrays = {}
    for name in ("ids.npy", "ptr.npy", "posts.npy", "counts.npy", "signs.npy"):
        a = np.load(graph / name, mmap_mode="r")
        arrays[name] = {"dtype": str(a.dtype), "shape": list(a.shape), "data_bytes": a.nbytes,
                        "sha256": protected[f"data/graph/{name}"]}
    ids = np.load(graph / "ids.npy", mmap_mode="r")
    ptr = np.load(graph / "ptr.npy", mmap_mode="r")
    counts = np.load(graph / "counts.npy", mmap_mode="r")
    registry = json.loads((graph / "behavior_ports.json").read_text(encoding="utf-8"))
    packages = {}
    for name in ("male-cns-arena", "flygym", "mujoco", "numpy", "numba", "scipy",
                 "pyarrow", "imageio", "imageio-ffmpeg", "pillow", "pytest", "psutil", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json("baseline.json", {
        "schema_version": 1, "created_utc": timestamp(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(), "dependencies": packages,
        "graph_manifest": manifest, "graph_arrays": arrays,
        "verified_manifest_arrays": all(protected[f"data/graph/{name}"] == digest
                                        for name, digest in manifest["arrays"].items()),
        "id_mapping": {"description": "ids[i] is source bodyId of CSR neuron i",
                       "strictly_increasing": bool(np.all(ids[1:] > ids[:-1])),
                       "first_body_id": int(ids[0]), "last_body_id": int(ids[-1])},
        "stored_edges": int(ptr[-1]), "synaptic_contacts_from_counts": int(counts.sum(dtype=np.uint64)),
        "core_code_digest": code_digest(), "config_digest": config_digest(cfg),
        "brain_config": asdict(BrainConfig()), "protected_files_count": len(protected),
        "existing_pathway_statuses": {name: item["status"] for name, item in registry["pathways"].items()},
        "tests": {name: len(list((ROOT / name).glob("test_*.py"))) for name in
                  ("tests", "circuit_tests", "compat_tests", "duel_tests", "bio_tests", "selectivity_tests")},
        "neuron_model": {"backend": "CPU numba homogeneous LIF", "voltage_units": "mV relative to rest",
                         "dt_ms": 1, "membrane_tau_ms": 20, "synaptic_tau_ms": 5,
                         "threshold_mv": 7, "reset_mv": 0, "refractory_ms": 2,
                         "synaptic_delay_ms": 2, "control_interval_ms": 10,
                         "physics_dt_ms": 0.1, "weight_rule": "count * presynaptic sign * 0.075"},
    })
    print(json.dumps({"frozen_files": len(protected), "core": code_digest(),
                      "previous_core_matches": document["previous_core_matches"]}), flush=True)


def memory():
    """Windows kernel lifetime peak working set, not a sampled RSS maximum."""
    if platform.system() != "Windows":
        return {"ram_peak_bytes": None, "ram_measurement": "unavailable outside Windows"}
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"ram_peak_bytes": counters.PeakWorkingSetSize, "ram_current_bytes": counters.WorkingSetSize,
            "private_commit_bytes": counters.PrivateUsage,
            "ram_measurement": "Windows GetProcessMemoryInfo process lifetime PeakWorkingSetSize"}


def benchmark(seconds=1.0):
    if not (0 < seconds <= 2.0) or abs(seconds * 100 - round(seconds * 100)) > 1e-8:
        raise ValueError("Short baseline must be in (0, 2] s and multiple of 10 ms")
    from fly_arena.config import load_config
    from fly_arena.ethology import EthologySimulation
    cfg = load_config(ROOT / "configs" / "ethology-unscaled.toml")
    started = time.perf_counter()
    sim = EthologySimulation(cfg, graph=ROOT / "data" / "graph", seed=1, warmup=True)
    setup = time.perf_counter() - started
    print(f"Baseline arena ready after {setup:.3f} s; running {seconds:.2f} sim s", flush=True)
    try:
        loop = time.perf_counter()
        for _ in range(round(seconds * 100)):
            sim.step()
        wall = time.perf_counter() - loop
        document = {"created_utc": timestamp(), "mode": sim.mode, "brain_enabled": True,
                    "seed": 1, "config_path": "configs/ethology-unscaled.toml", "warmup": True,
                    "seconds_simulated": sim.organism.clocks.physics_time_s,
                    "setup_wall_seconds": setup, "loop_wall_seconds": wall,
                    "sim_seconds_per_wall_second": seconds / wall,
                    "wall_seconds_per_sim_second": wall / seconds,
                    "renderer": sim.gl_renderer, "physics_timestep_seconds": sim.arena.sim.timestep,
                    "runtime_weights_sha256": hashlib.sha256(memoryview(sim.brain.weights)).hexdigest(),
                    "runtime_weights_dtype": str(sim.brain.weights.dtype),
                    "runtime_weights_shape": list(sim.brain.weights.shape),
                    "runtime_weights_hash_encoding": "SHA256 of C-order array payload only (no NPY header)",
                    "neural_clock_ms": sim.brain.clock, "action_durations": sim.action_durations,
                    "telemetry_final": sim.telemetry(), **memory(),
                    "vram_peak_bytes": None,
                    "vram_measurement": "Unavailable: no per-process peak VRAM meter was used. Offscreen eyes still use OpenGL.",
                    "limitations": "Single short full-graph+body headless sample, includes offscreen eyes, no video/log streaming/checkpoint compression. Numba caches may be warm; process RAM peak includes imports and setup."}
        write_json("baseline-performance.json", document)
        print(json.dumps({key: document[key] for key in ("loop_wall_seconds", "ram_peak_bytes", "renderer")}), flush=True)
    finally:
        sim.arena.close()


def check():
    frozen = json.loads((OUTPUT / "protected-manifest.json").read_text(encoding="utf-8"))
    changes = [name for name, digest in frozen["files"].items()
               if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    result = {"checked_utc": timestamp(), "checked_files": len(frozen["files"]),
              "changed_or_missing": changes, "pass": not changes}
    write_json("protected-check.json", result)
    print(json.dumps(result), flush=True)
    if changes:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "benchmark", "check"))
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "benchmark":
        benchmark(args.seconds)
    else:
        check()


if __name__ == "__main__":
    main()
