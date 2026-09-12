"""Separate CLI runtime: old configurations, code digest and checkpoints stay intact."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import time

import numpy as np

from fly_arena.brain import BrainConfig
from fly_arena.checkpoint import save_checkpoint, load_checkpoint
from fly_arena.config import load_config
from fly_arena.ethology_run import StreamingLog, _json_default

from .simulation import CircuitSimulation


def run(args):
    import imageio.v2 as imageio
    import mujoco
    from fly_arena.display import compose_ethology_frame
    from PIL import Image, ImageDraw, ImageFont

    saved = None
    if args.resume:
        saved, expected = load_checkpoint(args.resume)
        if saved.get("circuit_lab_format") != 1:
            raise ValueError("This is a legacy checkpoint. Continue it with 08_resume_search.cmd or python -m fly_arena run --resume.")
        base = saved["legacy_state"]
        config, seed = base["config"], base["seed"]
        control, block, threshold = saved["control"], saved["block"], saved["threshold_mv"]
        brain_config = BrainConfig(**base["brain_config"])
        flags = {k: base[k] for k in ("blind", "motor_off", "disabled_channels", "blocked_outputs")}
    else:
        config, seed = load_config(args.config), args.seed
        control, block, threshold = args.control, args.block, args.threshold_mv
        brain_config = BrainConfig()
        flags = {"blind": args.blind, "motor_off": args.motor_off}
    sim = viewer = writer = log = None
    stopping, paused = [False], [False]
    handlers = {}
    for name in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            code = getattr(signal, name)
            handlers[code] = signal.signal(code, lambda *_: stopping.__setitem__(0, True))
    start = time.perf_counter()
    failure = None
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()%1000000:06d}-{control}"
    checkpoint = output / "checkpoints/latest.npz"
    frames = 0
    try:
        sim = CircuitSimulation(config, graph=args.data / "graph", seed=seed, control=control,
                                block=block, threshold_mv=threshold, brain_config=brain_config,
                                warmup=saved is None, **flags)
        if saved:
            if sim.compatibility() != expected:
                raise ValueError("Incompatible circuit-lab model/code/parameters or legacy runtime")
            sim.set_state(saved)
        print(sim.label, flush=True)
        print("Paper voltage is a fitted subthreshold response, not full-graph GF spiking.", flush=True)
        log = StreamingLog(output, run_id, 36000)
        log.event({"kind": "resume" if saved else "start", "control": control, "block": block,
                   "threshold_mv_engineered": threshold, "physics_time_s": sim.organism.clocks.physics_time_s})
        if not args.headless:
            import mujoco.viewer
            def key(keycode):
                if keycode == 32:
                    paused[0] = not paused[0]
            viewer = mujoco.viewer.launch_passive(sim.arena.sim.mj_model, sim.arena.sim.mj_data,
                                                   key_callback=key, show_left_ui=False, show_right_ui=False)
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7, 125, -35
        if args.video:
            video_path = args.video
            if saved:
                video_path = video_path.with_name(video_path.stem + "-resume-" + run_id + video_path.suffix)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=30, codec="libx264", macro_block_size=16)
        t_start = sim.organism.clocks.physics_time_s
        wall_base = sim.organism.clocks.wall_time_s
        loop_start = time.perf_counter()
        video_due = log_due = t_start
        last_report = loop_start
        checkpoint_due = loop_start + 120
        while not stopping[0] and (not args.seconds or sim.organism.clocks.physics_time_s - t_start < args.seconds - 1e-9):
            if viewer and not viewer.is_running():
                break
            if paused[0]:
                if viewer:
                    viewer.sync()
                time.sleep(.02)
                continue
            result = sim.step()
            sim.organism.clocks.wall_time_s = wall_base + time.perf_counter() - loop_start
            t = sim.organism.clocks.physics_time_s
            if result["transition"]:
                log.event({"kind": "transition", **result["transition"]})
            for event in result["external_events"]:
                log.event({"kind": "event", **event})
            if t >= log_due:
                log.write_row(sim.telemetry())
                log_due += .1
            voltage = sim.paper_probe.last["gf_online_mv"]
            legacy = [sim.last_readout.raw_rates.get(f"escape_output_dnp01_{s}", 0.) for s in ("L", "R")]
            status = f"paper GF {voltage[0]:.2f}/{voltage[1]:.2f} mV | full LIF GF {legacy[0]:.1f}/{legacy[1]:.1f} Hz"
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = sim.arena.position
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                 f"{sim.label}\n{sim.last_decision.action}\n{status}\nTime {t:.2f}s / Space: pause", ""))
                viewer.sync()
            if writer and t >= video_due:
                canvas = Image.fromarray(compose_ethology_frame(sim, resumed=saved is not None))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 735, 1024, 768), fill="#101c23")
                draw.text((20, 740), status, font=ImageFont.load_default(size=13), fill="#e7bc72")
                writer.append_data(np.asarray(canvas))
                frames += 1
                video_due += 1/30
            now = time.perf_counter()
            if now >= checkpoint_due:
                save_checkpoint(checkpoint, sim.get_state(), sim.compatibility())
                checkpoint_due = now + 120
            if now - last_report > 5:
                print(f"{t:.2f}s {sim.last_decision.action} | {status}", flush=True)
                log.flush()
                last_report = now
        save_checkpoint(checkpoint, sim.get_state(), sim.compatibility())
        print(f"Saved laboratory checkpoint {checkpoint}", flush=True)
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        if viewer:
            viewer.close()
        if writer:
            writer.close()
        if log:
            log.close()
        if sim:
            report = {"label": sim.label, "error": failure, "control": control, "block": block,
                      "wall_s": time.perf_counter() - start, "video_frames": frames,
                      "actions": sim.action_durations, "minimum_upright": sim.minimum_upright,
                      "telemetry": sim.telemetry(), "compatibility": sim.compatibility(),
                      "model_scope": "Published fitted GF input model with a separate engineered running threshold; full LIF graph retained independently"}
            (output / f"{run_id}.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
            sim.close()
        for code, handler in handlers.items():
            signal.signal(code, handler)
