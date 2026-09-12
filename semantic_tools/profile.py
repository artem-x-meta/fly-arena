"""Short separate-process runtime resource samples; no training or file mutation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from fly_arena.config import load_config
from fly_semantic.runtime import SemanticSimulation
from .audit import memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enabled", action="store_true")
    p.add_argument("--seconds", type=float, default=1.)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    started = time.perf_counter()
    config = load_config(Path("configs/ethology-unscaled.toml"))
    sim = SemanticSimulation(config, graph=Path("data/graph"), seed=1, enabled=a.enabled,
        mapping_path=Path("runs/semantic-v1/calibration-selected/mapping.json"),
        calibration_path=Path("runs/semantic-v1/calibration-selected/calibration.json"),
        readout_path=Path("runs/semantic-v1/state-selected/state-readout.npz"))
    setup = time.perf_counter() - started
    try:
        started = time.perf_counter()
        for _ in range(round(a.seconds * 100)):
            sim.step()
        elapsed = time.perf_counter() - started
        result = {"enabled": a.enabled, "simulation_seconds": a.seconds, "setup_wall_seconds": setup,
            "loop_wall_seconds": elapsed, "sim_seconds_per_wall_second": a.seconds / elapsed,
            "renderer": sim.gl_renderer, **memory(), "vram_peak_bytes": None,
            "limitation": "Single short process; includes offscreen eye render and actual change in spiking from artificial input, no video/disk logging; no VRAM meter"}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
