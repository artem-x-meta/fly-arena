"""Causal 10 ms integration: sense -> neural -> decision -> physics -> resources."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from .behavior import BehaviorController
from .brain import Brain, BrainConfig
from .checkpoint import code_digest
from .config import config_digest
from .environment import Environment
from .ethology_arena import EthologyArena
from .neural_ports import NeuralAdapter, build_registry
from .organism import Organism
from .sensors import NeuralReadout, PhysicalEvents, SensorFrame


class LocalNavigator:
    """Explicit engineered odor tracking, using local concentration and history."""
    def __init__(self, config, seed=1):
        self.config = dict(config)
        self.rng = np.random.default_rng(seed)
        self.previous_odor = 0.0
        self.turn = 0.0
        self.last_engineered = np.zeros(2)
        self.last_neural = np.zeros(2)

    def step(self, frame, neural, mode, dt=.01, *, active=True):
        self.last_neural = np.array([neural.walk_left, neural.walk_right])
        if mode == "ethology-neural":
            self.last_engineered = np.zeros(2)
            return self.last_neural.copy()
        odor = (frame.odor_left + frame.odor_right) / 2
        self.turn += -self.turn * dt + .25 * np.sqrt(dt) * self.rng.normal()
        gradient = (frame.odor_left - frame.odor_right) / max(.05, odor)
        # Casting persists locally when the concentration falls; no target pose.
        casting = self.turn * (1.5 if odor < self.previous_odor else .6)
        turn = np.clip(self.config["odor_turn_gain"] * gradient + casting, -.5, .5)
        speed = self.config["search_speed"]
        self.last_engineered = np.clip([speed - turn, speed + turn], 0, 1.2)
        self.previous_odor = odor
        weight = self.config["neural_weight"]
        return np.clip(weight * self.last_neural + (1 - weight) * self.last_engineered, 0, 1.2)

    def get_state(self):
        return {"rng": self.rng.bit_generator.state, "previous_odor": self.previous_odor,
                "turn": self.turn, "last_engineered": self.last_engineered.copy(),
                "last_neural": self.last_neural.copy()}

    def set_state(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.previous_odor, self.turn = state["previous_odor"], state["turn"]
        self.last_engineered = np.array(state["last_engineered"])
        self.last_neural = np.array(state["last_neural"])


class EthologySimulation:
    def __init__(self, config, *, graph=None, seed=1, mode="ethology-hybrid",
                 brain_config=None, brain_enabled=True, warmup=True,
                 blind=False, motor_off=False, disabled_channels=(), blocked_outputs=()):
        self.config = copy.deepcopy(config)
        self.mode, self.seed = mode, seed
        self.blind, self.motor_off = bool(blind), bool(motor_off)
        self.disabled_channels, self.blocked_outputs = list(disabled_channels), list(blocked_outputs)
        self.organism = Organism(config["organism"], config["initial"])
        self.environment = Environment(config["food"], config["environment"], config["events"], seed=seed)
        self.environment.initial_dust = sum(self.organism.state.dust_by_region.values())
        policy_config = dict(config["behavior"])
        policy_config["allow_experimental_ports"] = config["neural"].get("exploratory_ports", False)
        policy_config["escape_enabled"] = config.get("escape", {}).get("enabled", False)
        self.behavior = BehaviorController(policy_config, seed=seed)
        if config["navigation"].get("strategy", "legacy") == "surge-cast":
            from .search import SurgeCastNavigator
            self.navigator = SurgeCastNavigator(config["navigation"], seed)
        else:
            self.navigator = LocalNavigator(config["navigation"], seed)
        self.brain = self.adapter = None
        self.registry = None
        self.brain_config = brain_config or BrainConfig()
        if brain_enabled:
            graph = Path(graph)
            if not (graph / "manifest.json").exists():
                raise FileNotFoundError(f"No prepared connectome at {graph}. Run: python -m fly_arena prepare")
            registry_path = graph / "behavior_ports.json"
            self.registry = (json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists()
                             else build_registry(graph, registry_path))
            if config.get("escape", {}).get("enabled", False):
                from .escape_ports import augment_escape_registry
                self.registry = augment_escape_registry(self.registry, graph)
            self.brain = Brain(graph, self.brain_config, seed)
            self.adapter = NeuralAdapter(self.brain, self.registry, config["neural"])
            if mode == "ethology-neural":
                self.adapter.require_supported(config.get("required_neural_pathways", ["feeding", "grooming"]))
        elif mode != "ethology-hybrid":
            raise ValueError("Body-only diagnostics do not support neural mode")
        self.arena = EthologyArena(seed=seed, food_patches=config["food"], warmup=warmup,
                                   mouth_contact_enabled=self.environment.config["mouth_contact_enabled"],
                                   grooming_contacts_enabled=self.environment.config["cleaning_enabled"],
                                   looming_objects=self.environment.config.get("looming", []))
        self.looming_detector = None
        if config.get("escape", {}).get("enabled", False):
            from .escape import VisualLoomingDetector
            self.looming_detector = VisualLoomingDetector(config["escape"].get("detector", {}))
        self.arena.mouth_contact_thickness = self.environment.config["contact_thickness_mm"]
        self.arena.motor.adaptive_proboscis = config.get("motor", {}).get("adaptive_proboscis", False)
        try:
            from OpenGL.GL import GL_RENDERER, glGetString
            self.gl_renderer = glGetString(GL_RENDERER).decode()
        except (ImportError, AttributeError):
            self.gl_renderer = "unavailable"
        model = self.arena.sim.mj_model
        model_buffer = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
        mj.mj_saveModel(model, buffer=model_buffer)
        self.model_digest = hashlib.sha256(model_buffer).hexdigest()
        self.source_digest = code_digest()
        self.base_light_diffuse = model.light_diffuse.copy()
        self.base_light_ambient = model.light_ambient.copy()
        self.visual_ids = {}
        for region, name in {"head": "clean_head", "antenna_left": "clean_antenna_left",
                             "antenna_right": "clean_antenna_right", "front_left": "lf_rub",
                             "front_right": "rf_rub"}.items():
            self.visual_ids[region] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"fly/{name}")
        self.steps = 0
        self.spike_total = 0
        self.minimum_upright = self.arena.upright
        self.action_durations = {}
        self.last_frame = self.last_decision = None
        self.last_readout = NeuralReadout()
        self.last_events = None
        self.last_command = np.zeros(2)
        self.transitions = 0
        self.initial_energy = self.organism.state.energy
        self.initial_gut = self.organism.state.gut_amount
        self.initial_gut_energy = self.organism.state.gut_food_energy

    @property
    def diagnostic(self):
        return self.brain is None

    @property
    def label(self):
        if self.diagnostic:
            return "BODY / ORGANISM DIAGNOSTIC - NO CONNECTOME"
        if self.mode == "ethology-hybrid":
            if self.config["navigation"].get("strategy") == "surge-cast":
                return "HYBRID / ENGINEERED ODOR SEARCH + CONNECTOME + MOTOR PRIMITIVES"
            return "HYBRID: ENGINEERED POLICY + CONNECTOME + MOTOR PRIMITIVES"
        if self.config["neural"].get("exploratory_ports", False):
            return "EXPERIMENTAL NEURAL CANDIDATES / UNCONFIRMED PATHWAYS"
        return "NEURAL REQUESTS / EXPLICIT ORGANISM + ENGINEERED MOTOR PRIMITIVES"

    def step(self, *, paused=False):
        if paused:
            return None
        dt = .01
        time_s = self.organism.clocks.physics_time_s
        external_events = self.environment.apply_scheduled_events(time_s, self.organism)
        self.arena.update_looming(time_s, refresh=True)
        self._update_visuals()
        eyes = self.arena.eyes()
        frame = self.environment.sense(self.arena, self.organism, eyes)
        if self.looming_detector is not None:
            frame.looming_left, frame.looming_right = map(float, self.looming_detector.step(eyes, dt, blind=self.blind or "vision" in self.disabled_channels))
        if self.brain:
            sensory, modulation = self.adapter.currents(frame, self.organism)
            _, counts = self.brain.step(self.brain.sample_eyes(eyes), sensory_currents=sensory,
                                       modulation=modulation, blind=self.blind,
                                       disabled_channels=self.disabled_channels,
                                       blocked_outputs=self.adapter.blocked_indices(self.blocked_outputs))
            neural = self.adapter.readout(counts, dt)
            spikes = int(counts.sum())
        else:
            neural, spikes = NeuralReadout(), 0
        decision = self.behavior.decide(frame, self.organism, neural, dt=dt, mode=self.mode)
        previous_search_phase = getattr(self.navigator, "phase", "legacy")
        gait = self.navigator.step(frame, neural, self.mode, dt, active=decision.action == "WALK" and not self.motor_off)
        search_phase = getattr(self.navigator, "phase", "legacy")
        if search_phase != previous_search_phase:
            external_events.append({"kind": "search_phase", "time_s": time_s, "phase": search_phase,
                                    "source": "engineered_policy", "odor_left": frame.odor_left,
                                    "odor_right": frame.odor_right})
        if decision.action != "WALK" or self.motor_off:
            gait = np.zeros(2)
        self.arena.mouth_contact_enabled = self.environment.config["mouth_contact_enabled"]
        self.arena.grooming_contacts_enabled = self.environment.config["cleaning_enabled"]
        events = self.arena.step_behavior(decision, gait, motor_off=self.motor_off)
        resources = self.environment.apply_physical_events(events, self.organism, dt)
        balance = self.organism.advance(dt, decision.action, movement_proxy=max(0, frame.movement))
        self.environment.advance_field(dt)
        if self.brain and abs(self.brain.clock / 1000 - self.organism.clocks.neural_time_s) > 1e-7:
            raise ArithmeticError("Neural and organism clocks diverged")
        if abs(self.arena.physics_steps * self.arena.sim.timestep - self.organism.clocks.physics_time_s) > 1e-7:
            raise ArithmeticError("Physics and organism clocks diverged")
        changed = (self.last_decision is None or decision.action != self.last_decision.action
                   or decision.target_body_region != self.last_decision.target_body_region)
        transition = {"time_s": time_s, **asdict(decision)} if changed else None
        self.transitions += int(changed)
        self.steps += 1
        self.spike_total += spikes
        self.minimum_upright = min(self.minimum_upright, self.arena.upright)
        self.action_durations[decision.action] = self.action_durations.get(decision.action, 0.0) + dt
        self.last_frame, self.last_readout = frame, neural
        self.last_decision, self.last_events, self.last_command = decision, events, gait
        return {"transition": transition, "external_events": external_events, "spikes": spikes,
                "resources": resources, "balance": asdict(balance)}

    def _update_visuals(self):
        model = self.arena.sim.mj_model
        for region, gid in self.visual_ids.items():
            if gid >= 0:
                amount = self.organism.state.dust_by_region[region]
                model.geom_rgba[gid] = [.52, .29, .08, min(.85, max(0.0, amount))]
        brightness = float(self.environment.config["light"])
        model.light_diffuse[:] = self.base_light_diffuse * brightness
        model.light_ambient[:] = self.base_light_ambient * brightness

    def telemetry(self):
        o, clocks = self.organism, self.organism.clocks
        frame = self.last_frame
        row = {"physics_time_s": clocks.physics_time_s, "neural_time_s": clocks.neural_time_s,
               "life_time_s": clocks.life_time_s, "wall_time_s": clocks.wall_time_s,
               "energy": o.state.energy, "gut_amount": o.state.gut_amount,
               "gut_food_energy": o.state.gut_food_energy, "hunger": o.hunger,
               "sleep_pressure": o.state.sleep_pressure, "circadian_phase": o.state.circadian_phase,
               "action": self.last_decision.action if self.last_decision else "IDLE",
               "source": self.last_decision.source if self.last_decision else "physical_guard",
               "upright": self.arena.upright, "movement_mm_s": frame.movement if frame else 0,
               "ground_support": frame.ground_support if frame else 0,
               "food_remaining": sum(p.amount for p in self.environment.food),
               "ingested_total": o.state.ingested_total, "digested_total": o.state.digested_total,
               "assimilated_energy_total": o.state.assimilated_energy_total,
               "metabolic_spent_total": o.state.metabolic_spent_total,
               "metabolic_unmet_total": o.state.metabolic_unmet_total,
               "energy_overflow_total": o.state.energy_overflow_total,
               "removed_dust": self.environment.removed_dust,
               "odor_left": frame.odor_left if frame else 0, "odor_right": frame.odor_right if frame else 0,
               "mouth_taste": frame.mouth_taste if frame else 0,
               "wake_stimulus": frame.local_wake_stimulus if frame else 0,
               "command_L": self.last_command[0], "command_R": self.last_command[1],
               "engineered_L": self.navigator.last_engineered[0], "engineered_R": self.navigator.last_engineered[1],
               "neural_L": self.navigator.last_neural[0], "neural_R": self.navigator.last_neural[1],
               "feed_drive": self.last_readout.feed_drive,
               "groom_head_drive": self.last_readout.groom_drives.get("head", 0),
               "spikes_total": self.spike_total, "actuator_owner": self.arena.motor.owner,
               "primitive_phase": self.arena.motor.primitive_phase,
               "max_groom_contact_force": self.arena.max_groom_contact_force}
        row.update(looming_left=frame.looming_left if frame else 0.,
                   looming_right=frame.looming_right if frame else 0.,
                   escape_drive=self.last_readout.escape_drive,
                   escape_cooldown_s=self.behavior.escape_cooldown_s)
        row.update(search_phase=getattr(self.navigator, "phase", "legacy"),
                   search_heading_error=getattr(self.navigator, "heading_error", 0.),
                   wind_body_x=frame.wind_body_x if frame else 0.,
                   wind_body_y=frame.wind_body_y if frame else 0.)
        row.update(dict(zip(("x_mm", "y_mm", "z_mm"), self.arena.position)))
        row.update({f"dust_{key}": value for key, value in o.state.dust_by_region.items()})
        row.update(self.resource_balances())
        return row

    def resource_balances(self):
        o = self.organism.state
        balances = {**self.environment.check_balances(self.organism),
                    "gut_mass_residual": self.initial_gut + o.ingested_total - o.gut_amount - o.digested_total,
                    "energy_residual": self.initial_energy + o.assimilated_energy_total - o.energy - o.metabolic_spent_total - o.energy_overflow_total}
        if any(abs(value) > 1e-7 for value in balances.values()):
            raise ArithmeticError(f"Organism balance violated: {balances}")
        return balances

    def compatibility(self):
        import importlib.metadata
        return {"code": self.source_digest, "config": config_digest(self.config), "model": self.model_digest,
                "graph": self.brain.manifest if self.brain else None,
                "ports": config_digest(self.registry) if self.registry else None,
                "versions": {name: importlib.metadata.version(name) for name in ("flygym", "mujoco", "numpy", "numba")},
                "mode": self.mode, "diagnostic_body_only": self.diagnostic}

    def get_state(self):
        model = self.arena.sim.mj_model
        return {"config": self.config, "mode": self.mode, "seed": self.seed,
                "brain_config": asdict(self.brain_config), "diagnostic_body_only": self.diagnostic,
                "blind": self.blind, "motor_off": self.motor_off,
                "disabled_channels": self.disabled_channels, "blocked_outputs": self.blocked_outputs,
                "organism": self.organism.get_state(), "environment": self.environment.get_state(),
                "behavior": self.behavior.get_state(), "navigator": self.navigator.get_state(),
                "arena": self.arena.get_state(), "brain": self.brain.get_state() if self.brain else None,
                "adapter": self.adapter.get_state() if self.adapter else None,
                "looming_detector": self.looming_detector.get_state() if self.looming_detector else None,
                "model_mutable": {name: getattr(model, name).copy() for name in ("geom_rgba", "light_diffuse", "light_ambient")},
                "stats": {"steps": self.steps, "spike_total": self.spike_total,
                          "minimum_upright": self.minimum_upright, "action_durations": self.action_durations,
                          "transitions": self.transitions, "initial_energy": self.initial_energy,
                          "initial_gut": self.initial_gut, "initial_gut_energy": self.initial_gut_energy},
                "last_decision": asdict(self.last_decision) if self.last_decision else None,
                "last_frame": asdict(self.last_frame) if self.last_frame else None,
                "last_readout": asdict(self.last_readout),
                "last_events": asdict(self.last_events) if self.last_events else None,
                "last_command": self.last_command.copy()}

    def set_state(self, state):
        self.organism.set_state(state["organism"])
        self.environment.set_state(state["environment"])
        self.behavior.set_state(state["behavior"])
        self.navigator.set_state(state["navigator"])
        if self.looming_detector:
            self.looming_detector.set_state(state["looming_detector"])
        for name, values in state["model_mutable"].items():
            getattr(self.arena.sim.mj_model, name)[:] = values
        self.arena.set_state(state["arena"])
        if self.brain:
            self.brain.set_state(state["brain"])
            self.adapter.set_state(state["adapter"])
        for name, value in state["stats"].items():
            setattr(self, name, value)
        if state["last_decision"]:
            from .behavior import BehaviorDecision
            self.last_decision = BehaviorDecision(**state["last_decision"])
        self.last_frame = SensorFrame(**state["last_frame"]) if state["last_frame"] else None
        self.last_readout = NeuralReadout(**state["last_readout"])
        self.last_events = PhysicalEvents(**state["last_events"]) if state["last_events"] else None
        self.last_command = np.array(state["last_command"])

    def close(self):
        self.arena.close()
