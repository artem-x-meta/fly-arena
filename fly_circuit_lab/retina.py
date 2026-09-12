"""Alternative retinal codes, measured against the unchanged one.

The diagnosis showed two independent reasons the looming object never reaches
LC4/LPLC2. This module addresses the first one only: the cells that do see the
object barely change their firing, because the stored current curve
``28 * L / (0.1 + L)`` is almost flat where the scene sits.

Nothing in ``fly_arena`` is modified. The replacement is injected by disabling
the built-in ``vision`` channel and supplying a channel of our own over the same
retinal ports, which ``Brain.step`` already supports. Every constant here is an
engineered choice of this laboratory, not a measured property of a fly.

Three codes, so the measurement can separate two different claims:

``legacy``
    The unchanged model. Saturating function of absolute smoothed luminance.
``linear``
    Removes the saturation, keeps absolute brightness coding. Isolates how much
    of the failure is the flat slope alone.
``adaptive``
    Removes the saturation and the absolute coding: each input reports Weber
    contrast against its own slowly adapting background, which is what
    photoreceptors do. Isolates the benefit of adaptation on top of slope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

MODES = ("legacy", "linear", "adaptive")


@dataclass
class RetinaConfig:
    """Engineered constants. None of these is fitted to a recording."""
    mode: str = "adaptive"
    # Current assigned to an input sitting exactly at its adapted background.
    midpoint_mv: float = 14.0
    # Weber contrast of 1.0 moves the current by this much.
    contrast_gain_mv: float = 24.0
    # Background adaptation time constant, in milliseconds of physical time.
    adapt_ms: float = 300.0
    # Luminance added to the denominator so a black background cannot divide by zero.
    floor: float = .05
    # Hard bounds, so no code can inject an unbounded current.
    clip_mv: tuple = (0.0, 28.0)
    # linear mode only: current per unit luminance, centred on the scene mean.
    linear_gain_mv: float = 28.0

    def validate(self) -> "RetinaConfig":
        if self.mode not in MODES:
            raise ValueError(f"Unknown retinal code {self.mode!r}; choose one of {MODES}")
        if not (self.adapt_ms > 0 and self.floor > 0 and self.contrast_gain_mv >= 0):
            raise ValueError("Retinal code constants must be positive")
        low, high = self.clip_mv
        if not low < high:
            raise ValueError("Retinal clip range must be ordered")
        if not low <= self.midpoint_mv <= high:
            raise ValueError("Retinal midpoint must lie inside the clip range")
        return self


class RetinaCode:
    """Turns sampled retinal luminance into an injected current, with state."""

    def __init__(self, size: int, config: RetinaConfig | None = None):
        self.config = (config or RetinaConfig()).validate()
        self.size = int(size)
        self.background = np.full(self.size, -1., np.float32)   # -1 marks "not yet seen"
        self.last_current = np.zeros(self.size, np.float32)
        self.last_contrast = np.zeros(self.size, np.float32)

    def step(self, luminance: np.ndarray, seconds: float = .01) -> np.ndarray:
        luminance = np.clip(np.asarray(luminance, np.float32), 0, 1)
        if luminance.shape != (self.size,):
            raise ValueError("Retinal luminance has the wrong shape")
        first = self.background < 0
        if first.any():
            self.background = np.where(first, luminance, self.background)
        alpha = 1 - np.exp(-seconds * 1000 / self.config.adapt_ms)
        contrast = (luminance - self.background) / (self.background + self.config.floor)
        if self.config.mode == "adaptive":
            current = self.config.midpoint_mv + self.config.contrast_gain_mv * contrast
        elif self.config.mode == "linear":
            # Absolute brightness, but on a straight line rather than a saturating one.
            current = self.config.linear_gain_mv * luminance
        else:
            raise ValueError("legacy mode injects nothing; the built-in channel stays enabled")
        self.background += alpha * (luminance - self.background)
        self.last_contrast = contrast.astype(np.float32)
        self.last_current = np.clip(current, *self.config.clip_mv).astype(np.float32)
        return self.last_current

    def get_state(self) -> dict:
        return {"version": 1, "config": asdict(self.config), "background": self.background.copy(),
                "last_current": self.last_current.copy(), "last_contrast": self.last_contrast.copy()}

    def set_state(self, saved: dict) -> None:
        if saved.get("version") != 1 or saved["config"] != asdict(self.config):
            raise ValueError("Incompatible retinal code checkpoint")
        for name in ("background", "last_current", "last_contrast"):
            value = np.asarray(saved[name], np.float32)
            if value.shape != (self.size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid retinal code checkpoint field {name}")
            setattr(self, name, value.copy())


class RetinaChannelAdapter:
    """Wraps the neural adapter and adds the laboratory's retinal channel."""

    def __init__(self, original, code: RetinaCode, sampler, retina_indices):
        self.original, self.code, self.sampler = original, code, sampler
        self.retina_indices = np.asarray(retina_indices, np.int32)

    def currents(self, sensor_frame, organism):
        sensory, modulation = self.original.currents(sensor_frame, organism)
        frames = getattr(sensor_frame, "vision", None)
        if frames is None:
            raise ValueError("The laboratory retinal channel needs the eye images")
        current = self.code.step(self.sampler(np.asarray(frames)))
        sensory = dict(sensory)
        sensory["retina_lab"] = (self.retina_indices, current)
        return sensory, modulation

    def __getattr__(self, name):
        return getattr(self.original, name)


def install(simulation, config: RetinaConfig, graph: Path) -> RetinaCode | None:
    """Replace the built-in visual channel on a running lab simulation.

    Returns the installed code, or ``None`` for ``legacy``, where the built-in
    channel is deliberately left in place so the control is the unchanged model.
    """
    config = config.validate()
    if config.mode == "legacy":
        return None
    if simulation.brain is None:
        raise ValueError("A retinal code needs the connectome enabled")
    from .diagnose import _eye_sampler

    graph = Path(graph)
    code = RetinaCode(len(simulation.brain.retina), config)
    simulation.adapter = RetinaChannelAdapter(simulation.adapter, code, _eye_sampler(graph),
                                              simulation.brain.retina)
    # The built-in channel is emptied by handing it a black image rather than by
    # listing "vision" in disabled_channels: that flag also reaches the optical
    # probe as `blind`, which would stop the paper model from seeing the object
    # and silently remove the very stimulus the experiment is about.
    brain = simulation.brain
    zeros = np.zeros(len(brain.retina), np.float32)
    sample_eyes = brain.sample_eyes
    brain.sample_eyes = lambda frames, _zeros=zeros, _original=sample_eyes: _zeros
    simulation.retina_code = code
    return code


def experiment(data: Path, output: Path, *, config_path: Path, diagnosis: Path,
               modes=MODES, seconds: float = 3.0, seed: int = 1,
               retina_config: RetinaConfig | None = None, ceiling=None) -> dict:
    """Run each retinal code on the same scene and compare the same numbers.

    The set of retinal inputs the object covers is read from the stored
    diagnosis, so every code is judged on the cells that were identified before
    any of them existed. The body is held still, as in the diagnosis, because a
    behavioural difference would change what the eyes see.
    """
    import numpy as np
    from fly_arena.config import load_config
    from .diagnose import _types, make_ceiling, measurement_groups
    from .simulation import CircuitSimulation

    graph = Path(data) / "graph"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stored = json.loads(Path(diagnosis).read_text(encoding="utf-8"))
    covered = np.asarray(stored["retina_coverage"]["covered_indices"], int)
    if not len(covered):
        raise ValueError("The stored diagnosis has no covered retinal inputs to measure")
    types = _types(graph)

    results = {}
    for mode in modes:
        settings = RetinaConfig(**{**asdict(retina_config or RetinaConfig()), "mode": mode})
        simulation = CircuitSimulation(load_config(config_path), control="observe", seed=seed,
                                       brain_enabled=True, graph=graph, motor_off=True)
        make_ceiling(simulation, ceiling)
        code = install(simulation, settings, graph)
        brain = simulation.brain
        substitution_checked = code is None
        groups = measurement_groups(graph, brain, types, covered)
        captured: dict = {}
        original = brain.step

        def recording(*args, **kwargs):
            command, spikes = original(*args, **kwargs)
            captured["spikes"] = spikes
            return command, spikes

        brain.step = recording
        rows, track = [], []
        try:
            for _ in range(round(seconds / .01)):
                simulation.step()
                spikes = captured["spikes"]
                if not substitution_checked:
                    built_in = brain.channel_contributions["vision"]
                    if built_in["enabled"] and abs(built_in["sum_mv"]) > 1e-6:
                        raise ValueError("The built-in visual channel still injects current")
                    if "retina_lab" not in brain.channel_contributions:
                        raise ValueError("The laboratory retinal channel was not delivered")
                    substitution_checked = True
                track.append(simulation.arena.position[:2].copy())
                row = {"t": round(simulation.organism.clocks.physics_time_s, 3),
                       "angle_deg": round(float(np.max(simulation.paper_probe.angle_deg)), 2)}
                if code is not None:
                    row["injected_mv_covered"] = round(float(code.last_current[covered].mean()), 3)
                    row["contrast_covered"] = round(float(code.last_contrast[covered].mean()), 4)
                for name, index in groups.items():
                    row[f"hz_{name}"] = round(float(spikes[index].sum() / len(index) / .01), 2)
                rows.append(row)
            state = {name: {"mean_voltage_mv": round(float(brain.voltage[index].mean()), 3),
                            "mean_current_mv": round(float(brain.current[index].mean()), 3)}
                     for name, index in groups.items()}
        finally:
            simulation.close()

        time = np.array([row["t"] for row in rows])
        angle = np.array([row["angle_deg"] for row in rows])
        approach = angle > 0
        # A mode that never sees the object cannot be compared; record why.
        window = {"approach_steps": int(approach.sum()), "max_angle_deg": round(float(angle.max()), 2),
                  "first_approach_s": round(float(time[approach].min()), 2) if approach.any() else None,
                  "body_travel_mm": round(float(np.linalg.norm(
                      np.asarray(track[-1]) - np.asarray(track[0]))), 4) if track else None}
        quiet = (time > .1) & (time < time[approach].min() - .1) if approach.any() else np.zeros_like(approach)
        summary = {}
        for name in groups:
            values = np.array([row[f"hz_{name}"] for row in rows])
            rest = values[1::2][quiet[1::2]]
            active = values[1::2][approach[1::2]]
            summary[name] = {"rest_hz": round(float(rest.mean()), 2) if len(rest) else None,
                             "looming_hz": round(float(active.mean()), 2) if len(active) else None,
                             "change_hz": round(float(active.mean() - rest.mean()), 2) if len(rest) and len(active) else None,
                             "rest_sd_hz": round(float(rest.std()), 2) if len(rest) else None,
                             **state[name]}
        selectivity = None
        if summary["retina_covered"]["change_hz"] is not None:
            selectivity = round(summary["retina_covered"]["change_hz"] - summary["retina_uncovered"]["change_hz"], 2)
        results[mode] = {"settings": asdict(settings), "populations": summary, "window": window,
                         "covered_minus_uncovered_hz": selectivity, "trace": rows}
        show = lambda value, width=7: f"{value:+{width}.2f}" if value is not None else " " * (width - 3) + "n/a"
        print(f"{mode:9} covered {show(summary['retina_covered']['change_hz'])} Hz | "
              f"uncovered {show(summary['retina_uncovered']['change_hz'])} Hz | "
              f"difference {show(selectivity)} Hz | lamina {show(summary['lamina_covered']['change_hz'], 6)} | "
              f"Mi1 {show(summary['Mi1']['looming_hz'], 6)} | "
              f"LC4 {show(summary['LC4']['looming_hz'], 6)} | GF {show(summary['DNp01']['looming_hz'], 6)} | "
              f"approach {window['approach_steps']} steps, max angle {window['max_angle_deg']}, "
              f"body moved {window['body_travel_mm']} mm", flush=True)

    report = {"schema_version": 1, "seed": seed, "seconds": seconds, "ceiling_grey": ceiling,
              "covered_inputs": int(len(covered)),
              "covered_mean_peak_delta": stored["retina_coverage"]["covered_mean_peak_delta"],
              "diagnosis_source": str(diagnosis), "modes": results,
              "interpretation": "Whether a retinal code makes the covered inputs report the object. A larger "
                                "covered-minus-uncovered difference means the change is carried by the cells "
                                "that see the object rather than by the whole population. This measures the "
                                "first failure only; the medulla still has no operating point."}
    (output / "retina-codes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {output / 'retina-codes.json'}", flush=True)
    return report


def flash(data: Path, output: Path, *, config_path: Path, modes=MODES, seconds: float = 3.0,
          seed: int = 1, at_s: float = 1.0, factor: float = 1.6,
          retina_config: RetinaConfig | None = None) -> dict:
    """Uniform illumination step, with no object anywhere in the scene.

    This separates the two codes that the looming scene cannot tell apart.
    ``linear`` reports absolute brightness, so a brighter world must shift it and
    keep it shifted. ``adaptive`` reports contrast against its own background, so
    it should answer the moment the light changes and then return towards its
    midpoint as the background catches up. The test is therefore the *sustained*
    offset once adaptation has had several time constants, not the peak.

    The step is applied to the MuJoCo headlight because this arena defines no
    light sources: ``nlight`` is zero, so the environment's own ``light`` event
    scales an empty array and changes nothing. That makes the built-in event
    useless as a control here, which is why the laboratory moves the headlight
    itself. Note that the headlight is not part of the stored checkpoint.
    """
    import numpy as np
    from fly_arena.config import load_config
    from .diagnose import _types, measurement_groups
    from .simulation import CircuitSimulation

    graph = Path(data) / "graph"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    types = _types(graph)
    results = {}
    for mode in modes:
        settings = RetinaConfig(**{**asdict(retina_config or RetinaConfig()), "mode": mode})
        config = load_config(config_path)
        config["environment"]["looming"] = []          # nothing approaches; only the light steps
        simulation = CircuitSimulation(config, control="observe", seed=seed, brain_enabled=True,
                                       graph=graph, motor_off=True)
        code = install(simulation, settings, graph)
        brain = simulation.brain
        groups = measurement_groups(graph, brain, types, np.empty(0, int))
        groups["lamina"] = np.flatnonzero(np.isin(types, ["L1", "L2", "L3", "L5"]))
        model = simulation.arena.sim.mj_model
        base_diffuse = model.vis.headlight.diffuse.copy()
        base_ambient = model.vis.headlight.ambient.copy()
        captured: dict = {}
        original = brain.step

        def recording(*args, **kwargs):
            command, spikes = original(*args, **kwargs)
            captured["spikes"] = spikes
            return command, spikes

        brain.step = recording
        rows, stepped = [], False
        try:
            for index in range(round(seconds / .01)):
                now = index * .01
                if not stepped and now >= at_s:
                    model.vis.headlight.diffuse[:] = np.clip(base_diffuse * factor, 0, 1)
                    model.vis.headlight.ambient[:] = np.clip(base_ambient * factor, 0, 1)
                    stepped = True
                simulation.step()
                spikes = captured["spikes"]
                row = {"t": round(simulation.organism.clocks.physics_time_s, 3)}
                for name, series in groups.items():
                    row[f"hz_{name}"] = round(float(spikes[series].sum() / len(series) / .01), 2)
                rows.append(row)
        finally:
            model.vis.headlight.diffuse[:] = base_diffuse
            model.vis.headlight.ambient[:] = base_ambient
            simulation.close()

        time = np.array([row["t"] for row in rows])
        before = (time > .3) & (time < at_s - .05)
        peak = (time >= at_s) & (time < at_s + .15)
        settled = time > at_s + 3 * settings.adapt_ms / 1000
        summary = {}
        for name in groups:
            values = np.array([row[f"hz_{name}"] for row in rows])
            base = values[1::2][before[1::2]].mean()
            summary[name] = {"baseline_hz": round(float(base), 2),
                             "peak_change_hz": round(float(values[1::2][peak[1::2]].mean() - base), 2),
                             "sustained_change_hz": round(float(values[1::2][settled[1::2]].mean() - base), 2)}
        results[mode] = {"settings": asdict(settings), "populations": summary, "trace": rows}
        retina_row, lamina_row = summary["retina"], summary["lamina"]
        print(f"{mode:9} retina peak {retina_row['peak_change_hz']:+7.2f} sustained {retina_row['sustained_change_hz']:+7.2f} | "
              f"lamina peak {lamina_row['peak_change_hz']:+7.2f} sustained {lamina_row['sustained_change_hz']:+7.2f}",
              flush=True)

    report = {"schema_version": 1, "seed": seed, "seconds": seconds, "flash_at_s": at_s,
              "headlight_factor": factor, "modes": results,
              "interpretation": "A code that reports absolute brightness keeps a sustained offset after a "
                                "uniform illumination step; a code that adapts answers the step and returns. "
                                "Neither outcome is a statement about looming selectivity deeper in the circuit."}
    (output / "flash-control.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {output / 'flash-control.json'}", flush=True)
    return report
