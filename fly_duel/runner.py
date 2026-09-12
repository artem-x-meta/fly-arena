"""Two-fly telemetry, video and viewer. Reports are not resumable checkpoints."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import signal
import time

import numpy as np


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def telemetry(duel):
    rows = []
    for contestant in duel.contestants:
        state, unit = contestant.organism.state, contestant.unit
        controller = getattr(contestant, "social", None)
        row = {"name": contestant.name, "position_mm": unit.position,
               "heading_rad": unit.heading, "upright": unit.upright,
               "ground_support": unit.ground_support,
               "action": contestant.decision.action if contestant.decision else "IDLE",
               "energy": state.energy, "ingested": state.ingested_total,
               "social_state": controller.state if controller is not None else "DISABLED"}
        if controller is not None:
            row.update(fatigue=controller.fatigue, food_memory_s=controller.food_memory_s,
                       social_counters=dict(controller.counters), social_reason=controller.last_reason,
                       resource_motivation=controller.motivation.summary(),
                       motor_owner=unit.motor.owner, motor_phase=unit.motor.primitive_phase)
            observation = getattr(contestant, "social_frame", None)
            if observation is not None:
                row["social_senses"] = asdict(observation)
        rows.append(row)
    return {"physics_time_s": duel.elapsed_s,
            "food_remaining": sum(patch.amount for patch in duel.environment.food),
            "body_contacts": duel.arena.contacts_between_flies(),
            "max_core_penetration_mm": duel.arena.max_core_penetration_mm,
            "max_interfly_penetration_mm": duel.arena.max_interfly_penetration_mm,
            "resource_transfers": getattr(duel, "last_transfers", []), "contestants": rows}


def _status(duel):
    lines = []
    for c in duel.contestants:
        controller = getattr(c, "social", None)
        social = controller.state if controller is not None else "DISABLED"
        action = c.decision.action if c.decision else "IDLE"
        lines.append(f"{c.name}: {social} / {action} | eaten {c.organism.state.ingested_total:.2f}")
    return lines


def _camera(duel):
    points = [unit.position for unit in duel.arena.units]
    points += [np.array([patch.x, patch.y, .2]) for patch in duel.environment.food]
    points = np.array(points)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    centre[2] = .4
    extent = np.max(np.linalg.norm(points[:, :2] - centre[:2], axis=1))
    return centre, max(7., float(extent) * 3.2 + 4.)


def video_frame(duel, *, social, social_sensing):
    from PIL import Image, ImageDraw, ImageFont
    centre, distance = _camera(duel)
    canvas = Image.new("RGB", (1024, 768), "#101c23")
    scene = duel.arena.render(distance=distance, azimuth=90., elevation=-58.,
                              lookat=centre, size=(1024, 576))
    canvas.paste(Image.fromarray(scene), (0, 112))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=25)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    draw.text((20, 12), "FLY ARENA / TWO MALES",
              font=title, fill="#e1eee8")
    resources = duel.config.get("duel", {}).get("resources")
    label = ("AMPLE FOOD" if resources == "ample" else "LIMITED FOOD" if resources == "scarce"
             else "SHARED FOOD")
    if not social:
        label += " / SOCIAL REACTIONS OFF"
    if social and not social_sensing:
        label += " / RIVAL SENSING OFF"
    draw.text((20, 48), label + " / physical bodies and shared food", font=small, fill="#e7bc72")
    remaining = sum(p.amount for p in duel.environment.food)
    draw.text((20, 78), f"Time {duel.elapsed_s:.2f} s  |  food remaining {remaining:.2f}  |  "
              f"body contacts {duel.arena.contacts_between_flies()}", font=font, fill="#94b4b9")
    for index, (line, colour) in enumerate(zip(_status(duel), ("#f29e2e", "#8cc7f2"))):
        draw.text((20, 698 + index * 28), line, font=font, fill=colour)
    return np.asarray(canvas)


def run(seconds=25., *, seed=1, body_contact=True, output=None, swap=False,
        scene_name="search", social=False, social_sensing=True, amount=None,
        viewer_enabled=False, video=None, resources="scarce"):
    from . import duel as module

    scene_kwargs = {"swap": swap}
    if amount is not None:
        scene_kwargs["amount"] = amount
    if scene_name == "encounter":
        scene_kwargs["resources"] = resources
        config, start, headings = module.encounter_scene(**scene_kwargs)
    elif scene_name == "search":
        config, start = module.scene(**scene_kwargs)
        headings = None
    else:
        raise ValueError(f"Unknown scene {scene_name!r}")
    output = Path(output) if output is not None else Path("runs/duel")
    output.mkdir(parents=True, exist_ok=True)
    tag = (f"seed{seed}{'-swapped' if swap else ''}{'' if body_contact else '-nocontact'}"
           f"{'-social' if social else ''}{'-blind' if not social_sensing else ''}"
           f"{'-encounter' if scene_name == 'encounter' else ''}")
    if scene_name == "encounter":
        tag += f"-{resources}"
    report_path, trace_path = output / f"duel-{tag}.json", output / f"duel-{tag}.jsonl"
    duel = viewer = writer = log = None
    handlers = {}
    stopping, paused = [False], [False]
    frames, stop_reason, failure = 0, "duration", None
    started = time.perf_counter()
    for name in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            code = getattr(signal, name)
            handlers[code] = signal.signal(code, lambda *_: stopping.__setitem__(0, True))
    try:
        kwargs = {"seed": seed, "body_contact": body_contact, "start": start}
        if social or not social_sensing:
            kwargs.update(social=social, social_sensing=social_sensing)
        if headings is not None:
            kwargs["start_headings"] = headings
        duel = module.Duel(config, **kwargs)
        log = trace_path.open("w", encoding="utf-8", buffering=1024 * 128)
        log.write(json.dumps({"kind": "start", "seed": seed, "scene": scene_name,
                              "social": social, "social_sensing": social_sensing,
                              "body_contact": body_contact, "sides_swapped": swap,
                              "start_positions": start, "start_headings": headings,
                              "resources": resources, "config": config}, default=_json_default) + "\n")
        if viewer_enabled:
            import mujoco.viewer
            def key(keycode):
                if keycode == 32:
                    paused[0] = not paused[0]
            viewer = mujoco.viewer.launch_passive(duel.arena.sim.mj_model, duel.arena.sim.mj_data,
                key_callback=key, show_left_ui=False, show_right_ui=False)
            with viewer.lock():
                centre, distance = _camera(duel)
                viewer.cam.lookat[:] = centre
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = distance, 90., -58.
        if video:
            import imageio.v2 as imageio
            video = Path(video)
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video, fps=30, codec="libx264", quality=7, macro_block_size=16)
            writer.append_data(video_frame(duel, social=social, social_sensing=social_sensing))
            frames = 1
        print(f"Running {'social' if social else 'passive'} {scene_name}: seed {seed}, "
              f"{seconds:g}s. Ctrl+C closes the run and writes its report.", flush=True)
        print("Space pauses the viewer. Run reports are not resumable checkpoints.", flush=True)
        video_due, last_report = 1 / 30, time.perf_counter()
        pause_recorded = False
        max_steps = round(seconds / .01)
        while seconds == 0 or duel.steps < max_steps:
            if stopping[0]:
                stop_reason = "interrupted"
                break
            if viewer is not None and not viewer.is_running():
                stop_reason = "window_closed"
                break
            if paused[0] != pause_recorded:
                pause_recorded = paused[0]
                log.write(json.dumps({"kind": "pause" if pause_recorded else "unpause",
                                      "physics_time_s": duel.elapsed_s}) + "\n")
                log.flush()
            if paused[0]:
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"PAUSED / {duel.elapsed_s:.2f}s / Space: continue\n" + "\n".join(_status(duel)), ""))
                viewer.sync()
                time.sleep(.02)
                continue
            duel.step()
            log.write(json.dumps({"kind": "step", **telemetry(duel)}, default=_json_default) + "\n")
            if viewer is not None:
                import mujoco
                centre, distance = _camera(duel)
                with viewer.lock():
                    viewer.cam.lookat[:] = centre
                    viewer.cam.distance = distance
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"{'SOCIAL' if social else 'PASSIVE'} / {duel.elapsed_s:.2f}s / Space: pause\n"
                    + "\n".join(_status(duel)), ""))
                viewer.sync()
            if writer is not None and duel.elapsed_s >= video_due - 1e-9:
                writer.append_data(video_frame(duel, social=social, social_sensing=social_sensing))
                frames += 1
                video_due += 1 / 30
            now = time.perf_counter()
            if now - last_report >= 5:
                print(f"{duel.elapsed_s:.2f}s | " + " | ".join(_status(duel)), flush=True)
                log.flush()
                last_report = now
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as error:
        failure, stop_reason = repr(error), "error"
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            finally:
                try:
                    if duel is not None:
                        report = duel.summary()
                        report.update(body_contact_enabled=body_contact, seed=seed,
                            sides_swapped=swap, solo_control=False,
                            start_positions=[list(point) for point in start],
                            start_headings=headings, social_enabled=social,
                            social_sensing=social_sensing, scene=scene_name,
                            resources=resources,
                            wall_s=round(time.perf_counter() - started, 3),
                            video_frames=frames, video=str(video) if video else None,
                            telemetry=str(trace_path), stop_reason=stop_reason, error=failure)
                        report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
                        if log is not None:
                            log.write(json.dumps({"kind": "end", "report": report}, default=_json_default) + "\n")
                finally:
                    try:
                        if log is not None:
                            log.close()
                        if duel is not None:
                            duel.close()
                    finally:
                        for code, handler in handlers.items():
                            signal.signal(code, handler)
    if duel is None:
        raise KeyboardInterrupt
    return report
