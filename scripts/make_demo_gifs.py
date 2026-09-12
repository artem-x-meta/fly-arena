"""Short annotated GIFs of the fly's behaviour, for documentation and articles.

Each scene is an ordinary run of the simulator with a camera pointed at it and a
caption strip drawn on top. No scene scripts an action: the arbiter still chooses
every 10 ms, and the captions report what it chose rather than what we wanted.

    .venv/Scripts/python.exe scripts/make_demo_gifs.py --scene all

Scenes render without the connectome by default, because it contributes only a
weighted part of the walking command and costs roughly twice the wall time;
`--brain` turns it on and the caption says which was used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fly_arena.config import load_config
from fly_arena.ethology import EthologySimulation

WIDTH, HEIGHT, STRIP = 520, 380, 54
FPS = 15
# The simulator exchanges every 10 ms, so one captured frame in seven plays back
# at roughly real speed. Scenes that need detail capture more often and therefore
# run slower than life; the caption says so.
REAL_TIME = 7


class Camera:
    """A view on the scene: fixed, or tracking the fly."""

    def __init__(self, distance, azimuth, elevation, *, track=True, lookat=(0., 0., .1), lift=0.):
        self.distance, self.azimuth, self.elevation = distance, azimuth, elevation
        self.track, self.lookat, self.lift = track, np.asarray(lookat, float), lift

    def apply(self, camera, simulation):
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.azimuth, camera.elevation = self.distance, self.azimuth, self.elevation
        if self.track:
            camera.lookat[:] = simulation.arena.position + np.array([0., 0., self.lift])
        else:
            camera.lookat[:] = self.lookat


def caption(frame, lines, colours):
    image = Image.new("RGB", (WIDTH, HEIGHT + STRIP), "#0e171c")
    image.paste(Image.fromarray(frame).resize((WIDTH, HEIGHT)), (0, STRIP))
    draw = ImageDraw.Draw(image)
    big = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=13)
    draw.text((12, 7), lines[0], font=big, fill=colours[0])
    if len(lines) > 1:
        draw.text((12, 31), lines[1], font=small, fill=colours[1])
    return np.asarray(image)


def record(name, config, camera, seconds, *, seed=1, brain=False, every=REAL_TIME,
           initial=None, label=None, output=ROOT / "vis-demo", warm=0.):
    """Run one scene and write an annotated GIF."""
    settings = load_config(config) if isinstance(config, (str, Path)) else config
    if initial:
        settings.setdefault("initial", {}).update(initial)
    started = time.perf_counter()
    graph = ROOT / "data" / "graph" if brain else None
    simulation = EthologySimulation(settings, seed=seed, brain_enabled=brain, graph=graph)
    renderer = mj.Renderer(simulation.arena.sim.mj_model, HEIGHT, WIDTH)
    view = mj.MjvCamera()
    frames, actions = [], {}
    try:
        for index in range(round((warm + seconds) / .01)):
            simulation.step()
            if index * .01 < warm or index % every:
                continue
            camera.apply(view, simulation)
            renderer.update_scene(simulation.arena.sim.mj_data, view)
            state, decision = simulation.organism.state, simulation.last_decision
            action = decision.action if decision else "IDLE"
            actions[action] = actions.get(action, 0) + every * .01
            second = (f"energy {state.energy:5.1f}   gut {state.gut_amount:4.2f}   "
                      f"sleep {state.sleep_pressure:4.2f}   dust {sum(state.dust_by_region.values()):4.2f}")
            speed = FPS * every / 100
            pace = "" if .9 <= speed <= 1.1 else f"   [{speed:.2g}x]"
            frames.append(caption(renderer.render(),
                                  [f"{action} / {simulation.arena.motor.primitive_phase}{pace}", second],
                                  ["#f0d9a8" if action != "ESCAPE" else "#ff8f6e", "#93b3b8"]))
    finally:
        renderer.close()
        simulation.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.gif"
    import imageio.v2 as imageio
    imageio.mimsave(path, frames, duration=1000 / FPS, loop=0)
    size = path.stat().st_size / 1048576
    order = sorted(actions.items(), key=lambda item: -item[1])
    print(f"{name:16} {len(frames):4} frames  {size:5.2f} MiB  {time.perf_counter() - started:5.1f}s wall  "
          f"| {'  '.join(f'{a} {t:.2f}s' for a, t in order)}", flush=True)
    return path


def dusty(head=1.0, antenna=.5, front=.0):
    return {"dust_by_region": {"head": head, "antenna_left": antenna, "antenna_right": antenna,
                               "front_left": front, "front_right": front}}


def scenes():
    near = load_config(ROOT / "configs/search-arena.toml")
    near["initial"] = {"energy": 12.0, "sleep_pressure": .05, **dusty(0, 0, 0)}
    near["food"] = [{"id": "patch", "x": 1.6, "y": 0., "radius": 1.1, "amount": 6.,
                     "energy_density": 12., "taste": 1., "odor": 1.}]

    far = load_config(ROOT / "configs/search-arena.toml")
    far["initial"] = {"energy": 40.0, "sleep_pressure": .05, **dusty(0, 0, 0)}
    far["food"] = [{"id": "far_patch", "x": 11.0, "y": 4.5, "radius": 1.2, "amount": 8.,
                    "energy_density": 12., "taste": 1., "odor": 1.4}]

    groom = load_config(ROOT / "configs/search-arena.toml")
    groom["food"] = []
    groom["initial"] = {"energy": 85.0, "sleep_pressure": .1, **dusty()}

    rub = load_config(ROOT / "configs/search-arena.toml")
    rub["food"] = []
    rub["initial"] = {"energy": 85.0, "sleep_pressure": .1, **dusty(0, 0, 1.0)}

    sleep = load_config(ROOT / "scenarios/sleep-wake.toml")

    day = load_config(ROOT / "configs/search-renewing.toml")
    day["initial"] = {"energy": 22.0, "sleep_pressure": .45, **dusty(.9, .45, .3)}

    return {
        "proboscis": dict(config=near, seconds=4.0, every=3,
                          camera=Camera(3.0, 150, -12, lift=.30)),
        "feeding-wide": dict(config=near, seconds=5.0,
                             camera=Camera(6.5, 130, -28, lift=.2)),
        "groom-head": dict(config=groom, seconds=5.0, every=4,
                           camera=Camera(4.0, 158, -14, lift=.30)),
        "groom-front": dict(config=rub, seconds=5.0, every=4,
                            camera=Camera(3.4, 178, -10, lift=.30)),
        "escape": dict(config=ROOT / "configs/escape-arena.toml", seconds=3.4, every=4,
                       camera=Camera(11., 140, -22, lift=.4)),
        "sleep-wake": dict(config=sleep, seconds=4.2,
                           camera=Camera(4.5, 140, -18, lift=.2)),
        "search": dict(config=far, seconds=20.0, every=20,
                       camera=Camera(26., 135, -55, track=False, lookat=(5., 2., .1))),
        "ethogram": dict(config=day, seconds=50.0, every=30,
                         camera=Camera(9., 135, -30, lift=.3)),
    }


def main():
    available = scenes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", default=[],
                        choices=[*available, "all"], help="Repeat to pick several; default is all")
    parser.add_argument("--brain", action="store_true", help="Run with the full connectome")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "vis-demo")
    args = parser.parse_args()
    chosen = [name for name in available if not args.scene or "all" in args.scene or name in args.scene]
    for name in chosen:
        record(name, brain=args.brain, seed=args.seed, output=args.output, **available[name])


if __name__ == "__main__":
    main()
