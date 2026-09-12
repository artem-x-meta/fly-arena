from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="MaleCNS + NeuroMechFly experimental arena")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare", help="Download official data and build sparse graph")
    prepare.add_argument("--data", type=Path, default=Path("data"))
    prepare.add_argument("--raw", type=Path, help="Reuse an existing directory of official Feather files")
    run = sub.add_parser("run", help="Run body-demo or experimental connectome mode")
    run.add_argument("--mode", choices=["body-demo", "connectome", "ethology-hybrid", "ethology-neural"], default="connectome")
    run.add_argument("--config", type=Path, help="Ethology TOML settings")
    run.add_argument("--scenario", help="Ethology initial conditions and external events")
    run.add_argument("--resume", type=Path, help="Restore a complete ethology checkpoint")
    run.add_argument("--wall-seconds", type=float, default=0, help="Optional wall-time limit after loading, 0 disables")
    run.add_argument("--disable-channel", action="append", default=[], help="Ablate a named sensory/modulation channel")
    run.add_argument("--block-output", action="append", default=[], help="Silence a named neural output/pathway")
    run.add_argument("--diagnostics", action="store_true", help="Show neural rates and stream channel diagnostics")
    run.add_argument("--data", type=Path, default=Path("data"))
    run.add_argument("--seconds", type=float, default=10, help="Simulation seconds; 0 means until window closes/Ctrl+C")
    run.add_argument("--headless", action="store_true")
    run.add_argument("--mujoco-ui", action="store_true",
                     help="Show MuJoCo's own side panels; they blank the scene on some drivers")
    run.add_argument("--gl", choices=["glfw", "egl", "osmesa"], help="Headless Linux: egl; Windows: glfw (default)")
    run.add_argument("--video", type=Path, help="Stream an MP4, without keeping frames in RAM")
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--blind", action="store_true", help="Ablate visual current; keep tonic inputs")
    run.add_argument("--motor-off", action="store_true", help="Clamp descending gait commands to zero")
    run.add_argument("--arousal", type=float, default=12, help="Additional constant current to DNp09; 0 removes it")
    run.add_argument("--turn-bias", type=float, default=6, help="Additional constant current to DNa02")
    run.add_argument("--background", type=float, default=4, help="Constant background drive to all neurons")
    run.add_argument("--synapse-gain", type=float, default=.075)
    args = parser.parse_args()
    if args.action == "prepare":
        from .data import prepare as prepare_data
        prepare_data(args.data, args.raw)
        return
    if args.seconds < 0 or args.wall_seconds < 0 or not all(map(math.isfinite, [args.seconds, args.wall_seconds, args.arousal, args.turn_bias, args.background, args.synapse_gain])):
        parser.error("Parameters must be finite; seconds must be >= 0")
    if args.headless and args.seconds == 0 and not args.video:
        print("Running without a time limit; Ctrl+C stops and saves logs.", flush=True)
    if args.gl:
        os.environ["MUJOCO_GL"] = args.gl
    if args.headless and platform.system() == "Linux" and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    if args.mode.startswith("ethology-") or args.config or args.scenario or args.resume:
        from .ethology_run import run_ethology
        try:
            run_ethology(args)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    else:
        run_simulation(args)


def run_simulation(args):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from .arena import Arena, BodyDemo
    from .brain import Brain, BrainConfig
    from .display import compose_frame

    brain = None
    if args.mode == "connectome":
        cfg = BrainConfig(arousal=args.arousal, turn_bias=args.turn_bias, background=args.background,
                          synapse_gain=args.synapse_gain)
        print("Loading the full prepared graph. First Numba compilation may take a moment.", flush=True)
        brain = Brain(args.data / "graph", cfg, args.seed)
        print(f"{brain.manifest['neurons']:,} neurons / {brain.manifest['edges']:,} edges. "
              f"Tonic drive: background {cfg.background:g} + DNp09 {cfg.arousal:g} mV.", flush=True)
    else:
        print("BODY DEMO: scripted wandering for testing the body. No connectome is running.", flush=True)
    arena = Arena(seed=args.seed)
    demo = BodyDemo(args.seed) if brain is None else None
    viewer = writer = None
    paused = [False]
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000:06d}-{args.mode}"
    log_path = args.output / f"{run_id}.csv"
    metadata_path = args.output / f"{run_id}.json"
    video_due = 0.0
    start = time.perf_counter()
    last_report = start
    t = 0.0
    spike_total = 0
    frames = 0
    interrupted = False
    failure = None
    initial_position = arena.position
    try:
        if not args.headless:
            import mujoco.viewer
            def key_callback(key):
                if key == 32:
                    paused[0] = not paused[0]
            # MuJoCo's own side panels are hidden by default. On some OpenGL drivers
            # their rendering leaves the 3D viewport clipped to a narrow strip and the
            # scene disappears; the panels carry no state this program needs, because
            # the overlay already reports it. --mujoco-ui restores them.
            viewer = mujoco.viewer.launch_passive(arena.sim.mj_model, arena.sim.mj_data,
                                                  key_callback=key_callback,
                                                  show_left_ui=args.mujoco_ui, show_right_ui=args.mujoco_ui)
            with viewer.lock():
                viewer.cam.distance = 10
                viewer.cam.azimuth = 125
                viewer.cam.elevation = -35
                viewer.cam.lookat[:] = arena.position
            print("Mouse: orbit / zoom. Space: pause. Close window: stop.", flush=True)
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=30, codec="libx264", quality=7, macro_block_size=16)
        with log_path.open("w", newline="", encoding="utf-8") as log:
            fields = ["time_s", "wall_s", "x_mm", "y_mm", "z_mm", "upright", "command_L", "command_R", "spikes",
                      "retina_spikes", "DNp09_L_Hz", "DNp09_R_Hz", "DNa02_L_Hz", "DNa02_R_Hz", "left_light", "right_light"]
            csv_writer = csv.DictWriter(log, fields)
            csv_writer.writeheader()
            while args.seconds == 0 or t < args.seconds - 1e-8:
                if viewer and not viewer.is_running():
                    break
                if paused[0]:
                    viewer.sync()
                    time.sleep(.02)
                    continue
                eyes = arena.eyes()
                if brain:
                    command, counts = brain.step(brain.sample_eyes(eyes), blind=args.blind)
                    spikes = int(counts.sum())
                else:
                    command, spikes = demo.step(arena), 0
                if args.motor_off:
                    command = np.zeros(2)
                arena.step(command)
                t = arena.physics_steps * arena.sim.timestep
                wall = time.perf_counter() - start
                spike_total += spikes
                row = dict(zip(fields[:6], [t, wall, *arena.position, arena.upright]))
                row.update(command_L=command[0], command_R=command[1], spikes=spikes,
                           left_light=float(eyes[0].mean() / 255), right_light=float(eyes[1].mean() / 255))
                if brain:
                    row["retina_spikes"] = int(counts[brain.retina].sum())
                    for kind in ("DNp09", "DNa02"):
                        for i, side in enumerate(("L", "R")):
                            row[f"{kind}_{side}_Hz"] = brain.decoder.rates[kind][i]
                csv_writer.writerow(row)
                if viewer:
                    with viewer.lock():
                        viewer.cam.lookat[:] = arena.position
                    viewer.sync()
                if writer and t + 1e-8 >= video_due:
                    writer.append_data(compose_frame(arena, eyes, args.mode, t, spikes, command, wall))
                    frames += 1
                    # 1x playback; use a video player's speed control to slow it down.
                    video_due += 1 / 30
                if time.perf_counter() - last_report > 5:
                    print(f"sim {t:.2f}s | wall {wall:.1f}s | {spikes:,} spikes/10ms | "
                          f"gait {command.round(2)} | upright {arena.upright:.2f}", flush=True)
                    log.flush()
                    last_report = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        if viewer:
            viewer.close()
        if writer:
            writer.close()
        metadata = {
            "mode": args.mode, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "simulation_seconds": t, "wall_seconds": time.perf_counter() - start,
            "total_spikes": spike_total, "video_frames": frames,
            "start_position_mm": initial_position.tolist(), "end_position_mm": arena.position.tolist(),
            "interrupted": interrupted, "error": failure,
            "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__,
            "brain_config": asdict(brain.config) if brain else None,
            "graph": brain.manifest if brain else None,
            "limitations": "Simplified experimental LIF, approximate vision, tonic DN inputs, engineered CPG gait. No learned natural behavior.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if brain:
            brain.save(args.output / f"{run_id}-neural-state.npz")
        arena.close()
    print(f"Finished {t:.2f} simulated seconds in {time.perf_counter() - start:.1f}s. Log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
