"""Interactive/headless observation, streamed logs, videos and graceful resume."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class StreamingLog:
    def __init__(self, directory, run_id, rotate_rows):
        self.directory, self.run_id = Path(directory), run_id
        self.rotate_rows = rotate_rows
        self.rows = self.event_rows = 0
        self.csv = self.events = None
        self.writer = None
        self.files = []

    def write_row(self, row):
        if self.rows % self.rotate_rows == 0:
            if self.csv:
                self.csv.close()
            path = self.directory / f"{self.run_id}-{self.rows // self.rotate_rows:04d}.csv"
            self.csv = path.open("w", newline="", encoding="utf-8")
            self.files.append(str(path))
            self.writer = csv.DictWriter(self.csv, list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows += 1

    def event(self, event):
        if self.event_rows % self.rotate_rows == 0:
            if self.events:
                self.events.close()
            path = self.directory / f"{self.run_id}-events-{self.event_rows // self.rotate_rows:04d}.jsonl"
            self.events = path.open("w", encoding="utf-8")
            self.files.append(str(path))
        self.events.write(json.dumps(event, default=_json_default, allow_nan=False) + "\n")
        self.event_rows += 1

    def flush(self):
        for stream in (self.csv, self.events):
            if stream:
                stream.flush()

    def close(self):
        for stream in (self.csv, self.events):
            if stream:
                stream.close()


def run_ethology(args):
    import imageio.v2 as imageio
    import mujoco
    from .brain import BrainConfig
    from .checkpoint import load_checkpoint, save_checkpoint
    from .config import load_config
    from .display import compose_ethology_frame
    from .ethology import EthologySimulation

    saved = None
    if args.resume:
        saved, expected = load_checkpoint(args.resume)
        if args.config or args.scenario:
            raise ValueError("Resume uses its saved configuration; omit --config and --scenario")
        cfg, mode, seed = saved["config"], saved["mode"], saved["seed"]
        brain_config = BrainConfig(**saved["brain_config"])
        flags = {key: saved[key] for key in ("blind", "motor_off", "disabled_channels", "blocked_outputs")}
        if saved["diagnostic_body_only"]:
            raise ValueError("Body-only diagnostic checkpoints cannot launch as a connectome simulation")
    else:
        cfg = load_config(args.config, args.scenario)
        mode = args.mode if args.mode.startswith("ethology-") else "ethology-hybrid"
        seed = args.seed
        brain_config = BrainConfig(arousal=args.arousal, turn_bias=args.turn_bias,
                                   background=args.background, synapse_gain=args.synapse_gain)
        flags = {"blind": args.blind, "motor_off": args.motor_off,
                 "disabled_channels": args.disable_channel, "blocked_outputs": args.block_output}
    paused, stopping, wake_requested = [False], [False], [False]
    previous_handlers = {}
    for name in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            signum = getattr(signal, name)
            previous_handlers[signum] = signal.signal(signum, lambda *_: stopping.__setitem__(0, True))
    sim = viewer = writer = logs = None
    frames = 0
    failure = None
    run_start = time.perf_counter()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000:06d}-{mode}"
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoints" / "latest.npz"
    try:
        print("Loading full MaleCNS and the extended physical body...", flush=True)
        sim = EthologySimulation(cfg, graph=args.data / "graph", seed=seed, mode=mode,
                                  brain_config=brain_config, warmup=saved is None, **flags)
        if saved:
            actual = sim.compatibility()
            different = [key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)]
            if different:
                raise ValueError("Incompatible checkpoint: " + ", ".join(sorted(different)))
            sim.set_state(saved)
        print(sim.label, flush=True)
        logs = StreamingLog(args.output, run_id, cfg["logging"]["rotate_rows"])
        logs.event({"kind": "resume" if saved else "start", "mode": mode, "label": sim.label,
                    "physics_time_s": sim.organism.clocks.physics_time_s,
                    "checkpoint": str(args.resume) if saved else None})
        if not args.headless:
            import mujoco.viewer
            def key_callback(key):
                if key == 32:
                    paused[0] = not paused[0]
                elif key in (87, 119):
                    wake_requested[0] = True
            # MuJoCo's own side panels are hidden by default. On some OpenGL drivers
            # their rendering leaves the 3D viewport clipped to a narrow strip and the
            # scene disappears; the panels carry no state this program needs, because
            # the overlay already reports it. --mujoco-ui restores them.
            viewer = mujoco.viewer.launch_passive(sim.arena.sim.mj_model, sim.arena.sim.mj_data,
                                                  key_callback=key_callback,
                                                  show_left_ui=args.mujoco_ui,
                                                  show_right_ui=args.mujoco_ui)
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7, 125, -35
                viewer.cam.lookat[:] = sim.arena.position
            print("Space: pause. W: local wake pulse. Close window / Ctrl+C: checkpoint and stop.", flush=True)
        video_path = args.video
        if video_path:
            if saved:
                video_path = video_path.with_name(f"{video_path.stem}-resume-{run_id}{video_path.suffix}")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=7, macro_block_size=16)
        start_physics = sim.organism.clocks.physics_time_s
        start_wall = sim.organism.clocks.wall_time_s
        loop_start = time.perf_counter()
        video_due = log_due = start_physics
        checkpoint_due = loop_start + cfg["logging"]["checkpoint_wall_s"]
        last_report = loop_start
        last_paused = False
        while not stopping[0]:
            t = sim.organism.clocks.physics_time_s
            now = time.perf_counter()
            if args.seconds and t - start_physics >= args.seconds - 1e-8:
                break
            if args.wall_seconds and now - loop_start >= args.wall_seconds:
                break
            if viewer and not viewer.is_running():
                break
            sim.organism.clocks.wall_time_s = start_wall + now - loop_start
            if paused[0] != last_paused:
                logs.event({"kind": "pause" if paused[0] else "unpause", "physics_time_s": t})
                logs.flush()
                last_paused = paused[0]
            if paused[0]:
                if viewer:
                    viewer.sync()
                time.sleep(.02)
                continue
            if wake_requested[0]:
                sim.environment.add_wake(t, sim.arena.position, amplitude=2)
                logs.event({"kind": "manual_local_wake", "physics_time_s": t, "amplitude": 2})
                wake_requested[0] = False
            result = sim.step()
            if result["transition"]:
                logs.event({"kind": "transition", **result["transition"]})
            for event in result["external_events"]:
                logs.event({"kind": "external", "event": event})
            t = sim.organism.clocks.physics_time_s
            if t + 1e-9 >= log_due:
                logs.write_row(sim.telemetry())
                if args.diagnostics and sim.brain:
                    logs.event({"kind": "neural_diagnostic", "physics_time_s": t,
                                "rates": sim.last_readout.raw_rates,
                                "channels": sim.brain.channel_contributions,
                                "coverage": sim.last_readout.supported_ports})
                log_due += 1 / cfg["logging"]["telemetry_hz"]
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = sim.arena.position
                state = sim.organism.state
                text = (f"{sim.label}\n{sim.last_decision.action} / {sim.arena.motor.primitive_phase}\n"
                        f"energy {state.energy:.1f}   gut {state.gut_amount:.2f}   sleep {state.sleep_pressure:.2f}\n"
                        f"dust {sum(state.dust_by_region.values()):.2f}\n"
                        f"physics {t:.2f}s   neural {sim.organism.clocks.neural_time_s:.2f}s   life {sim.organism.clocks.life_time_s:.1f}s")
                if hasattr(sim.navigator, "phase"):
                    text += f"\nSearch: {sim.navigator.phase}  odor L/R {sim.last_frame.odor_left:.2f}/{sim.last_frame.odor_right:.2f}"
                if args.diagnostics:
                    text += "\n" + " / ".join(f"{key} {value:.1f}Hz" for key, value in sim.last_readout.raw_rates.items()
                                             if key in ("feed_output_mn9", "groom_output_adn"))
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ""))
                viewer.sync()
            if writer and t + 1e-9 >= video_due:
                writer.append_data(compose_ethology_frame(sim, resumed=bool(saved)))
                frames += 1
                video_due += 1 / 30
            now = time.perf_counter()
            if cfg["logging"]["checkpoint_wall_s"] and now >= checkpoint_due:
                save_checkpoint(checkpoint_path, sim.get_state(), sim.compatibility())
                logs.event({"kind": "checkpoint", "physics_time_s": t, "path": str(checkpoint_path)})
                checkpoint_due = now + cfg["logging"]["checkpoint_wall_s"]
            if now - last_report >= 5:
                state = sim.organism.state
                print(f"physics {t:.2f}s | wall {now - loop_start:.1f}s | {sim.last_decision.action} | "
                      f"E {state.energy:.2f} gut {state.gut_amount:.2f} sleep {state.sleep_pressure:.2f} | "
                      f"upright {sim.arena.upright:.3f}", flush=True)
                logs.flush()
                last_report = now
        sim.organism.clocks.wall_time_s = start_wall + time.perf_counter() - loop_start
        save_checkpoint(checkpoint_path, sim.get_state(), sim.compatibility())
        logs.event({"kind": "stop", "physics_time_s": sim.organism.clocks.physics_time_s,
                    "interrupted": stopping[0], "checkpoint": str(checkpoint_path)})
        print(f"Saved {checkpoint_path}; physical time {sim.organism.clocks.physics_time_s:.2f}s.", flush=True)
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        if viewer:
            viewer.close()
        if writer:
            writer.close()
        if logs:
            logs.close()
        if sim:
            report = {"mode": mode, "label": sim.label, "error": failure,
                      "resumed_from": str(args.resume) if args.resume else None,
                      "wall_seconds_including_load": time.perf_counter() - run_start,
                      "video_frames": frames, "telemetry": sim.telemetry(),
                      "minimum_upright": sim.minimum_upright,
                      "opengl_renderer": sim.gl_renderer,
                      "action_durations_physics_s": sim.action_durations,
                      "neural_coverage": sim.last_readout.supported_ports,
                      "config": cfg, "brain_config": asdict(sim.brain_config),
                      "compatibility": sim.compatibility(), "logs": logs.files if logs else []}
            (args.output / f"{run_id}.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
            sim.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
