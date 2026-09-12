"""Composition around the existing LIF/physical simulation, with no core edits."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .features import CausalFeatures
from .mapping import digest, file_digest, load_mapping, validate_mapping
from .protocol import InputCueQueue, OutputStateMachine
from .readout import LinearReadout


def load_calibration(path, mapping):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_calibration(document, mapping)
    return document


def validate_calibration(document, mapping):
    value = dict(document)
    expected = value.pop("digest", None)
    if expected != digest(value) or value.get("schema_version") != 1 or value.get("status") != "CALIBRATED":
        raise ValueError("Enabled semantic channel requires successful, intact calibration")
    if value["mapping_digest"] != mapping["digest"]:
        raise ValueError("Calibration and semantic mapping differ")
    hunger_gain = value.get("hunger_gain_mv")
    if type(hunger_gain) not in (float, int) or not np.isfinite(hunger_gain) or hunger_gain <= 0:
        raise ValueError("Hunger input gain must be explicitly calibrated")
    cue_gain = value.get("cue_gain_mv")
    cue_calibrated = value.get("cue_status") == "CALIBRATED"
    if cue_calibrated and (type(cue_gain) not in (float, int) or not np.isfinite(cue_gain) or cue_gain <= 0):
        raise ValueError("Cue input gain must be explicitly calibrated")
    if not cue_calibrated and cue_gain is not None and (type(cue_gain) not in (float, int) or not np.isfinite(cue_gain) or cue_gain <= 0):
        raise ValueError("Unverified cue gain must be positive finite or null")
    if value["pulse_ms"] != 250:
        raise ValueError("Unsupported calibrated pulse duration")


def make_features(mapping, calibration):
    return CausalFeatures([r["index"] for r in mapping["features"]],
        neuron_count=mapping["neuron_count"], excluded_indices=mapping["excluded_indices"],
        manifest_digest=digest({"mapping": mapping["digest"], "calibration": calibration["digest"]}))


class SemanticChannel:
    """Readout receives counts only; organism need is confined to input encoding."""
    def __init__(self, mapping, calibration, *, readout=None, episode_id="demo-0001",
                 hunger_disconnected=False, cue_disconnected=False):
        validate_mapping(mapping)
        validate_calibration(calibration, mapping)
        self.mapping = copy.deepcopy(mapping)
        self.calibration = copy.deepcopy(calibration)
        self.features = make_features(mapping, calibration)
        self.readout = readout
        if readout is not None and (readout.feature_digest != self.features.feature_digest or list(readout.concept_ids) != [1]):
            raise ValueError("This stage supports a matching trained NEED_FOOD head only")
        self.hunger_disconnected = bool(hunger_disconnected)
        self.cue_disconnected = bool(cue_disconnected)
        self.ports = {name: np.array([r["index"] for r in entries], np.int32)
                      for name, entries in mapping["ports"].items()}
        self.reset(episode_id)

    @property
    def supported_concepts(self):
        return [1] if self.readout is not None else []

    @property
    def calibrated_input_concepts(self):
        # Historical calibration only checked hunger; never infer cue calibration.
        return [101, 102, 103] if self.calibration.get("cue_status") == "CALIBRATED" else []

    def submit(self, event, sim_time_ms):
        """Use this gate for human/oracle input; unverified cues cannot be queued."""
        if not self.calibrated_input_concepts:
            raise ValueError("Semantic cue input is not calibrated")
        return self.inbox.submit(event, sim_time_ms)

    def reset(self, episode_id):
        self.features.reset()
        self.inbox = InputCueQueue(episode_id, pulse_ms=self.calibration["pulse_ms"])
        self.outbox = OutputStateMachine(episode_id, supported_concepts=self.supported_concepts)
        self.last_scores = {}
        self.last_events = ()
        self.last_sample_ms = 0

    def currents(self, need, sim_time_ms):
        if isinstance(need, bool) or not np.isscalar(need) or not np.isfinite(need) or not 0 <= need <= 1:
            raise ValueError("Existing organism hunger must be finite in [0,1]")
        if not self.calibrated_input_concepts and (self.inbox.pending_count or self.inbox.active_concept_id is not None):
            # Defend against callers bypassing submit through the exposed inbox.
            candidate = copy.deepcopy(self.inbox)
            candidate.advance(sim_time_ms)
            if candidate.active_concept_id is not None:
                raise ValueError("Semantic cue input is not calibrated")
        self.inbox.advance(sim_time_ms)
        result = {}
        if not self.hunger_disconnected:
            result["semantic_hunger"] = (self.ports["hunger"], self.calibration["hunger_gain_mv"] * float(need))
        cue = self.inbox.active_concept_id
        if cue is not None and not self.cue_disconnected:
            result["semantic_cue"] = (self.ports[str(cue)], self.calibration["cue_gain_mv"])
        return result

    def observe(self, counts, dt_ms, sim_time_ms):
        if (type(sim_time_ms) not in (int, float) or not np.isfinite(sim_time_ms)
                or dt_ms != 10 or sim_time_ms != self.features.elapsed_ms + dt_ms):
            raise ValueError("Semantic observation must follow the next 10ms neural interval")
        values = self.features.update(counts, dt_ms)
        self.last_events = ()
        if self.readout is not None and sim_time_ms - self.last_sample_ms >= 100:
            scores = self.readout.predict_scores(values[None, :])[0]
            self.last_scores = {int(k): float(v) for k, v in zip(self.readout.concept_ids, scores)}
            self.last_events = self.outbox.update(self.last_scores, sim_time_ms)
            self.last_sample_ms = int(sim_time_ms)
        return values

    def identity(self):
        model = None if self.readout is None else digest({
            "weights": self.readout.weights.tolist(), "bias": self.readout.bias.tolist(),
            "mean": self.readout.mean.tolist(), "scale": self.readout.scale.tolist(),
            "feature_digest": self.readout.feature_digest, "concept_ids": self.readout.concept_ids.tolist()})
        return {"mapping": self.mapping["digest"], "calibration": self.calibration["digest"],
            "model": model, "hunger_disconnected": self.hunger_disconnected, "cue_disconnected": self.cue_disconnected}

    def get_state(self):
        return {"format": "fly_semantic.channel.v1", "identity": self.identity(),
            "features": self.features.get_state(), "inbox": self.inbox.get_state(), "outbox": self.outbox.get_state(),
            "last_scores": self.last_scores.copy(), "last_events": [e.to_dict() for e in self.last_events],
            "last_sample_ms": self.last_sample_ms}

    def set_state(self, state):
        from .protocol import Event
        if state.get("format") != "fly_semantic.channel.v1" or state.get("identity") != self.identity():
            raise ValueError("Incompatible semantic channel checkpoint")
        # Validate all components in detached instances before committing anything.
        features = make_features(self.mapping, self.calibration)
        features.set_state(state["features"])
        inbox = copy.deepcopy(self.inbox)
        inbox.set_state(state["inbox"])
        outbox = copy.deepcopy(self.outbox)
        outbox.set_state(state["outbox"])
        scores = {int(k): float(v) for k, v in state["last_scores"].items()}
        if (set(scores) not in (set(), set(self.supported_concepts)) or
                any(not np.isfinite(v) or not 0 <= v <= 1 for v in scores.values())):
            raise ValueError("Invalid cached semantic scores")
        sample = state["last_sample_ms"]
        if type(sample) is not int or sample < 0 or sample > features.elapsed_ms or sample % 100:
            raise ValueError("Invalid semantic sample clock")
        events = tuple(Event.from_dict(e) for e in state["last_events"])
        in_state, out_state = inbox.get_state(), outbox.get_state()
        if in_state["episode_id"] != out_state["episode_id"]:
            raise ValueError("Semantic checkpoint episode IDs diverged")
        if in_state["now_ms"] > features.elapsed_ms or out_state["now_ms"] != sample:
            raise ValueError("Semantic transport and feature clocks diverged")
        if out_state["last_sample_ms"] != (sample if sample else None):
            raise ValueError("Semantic output sample clocks diverged")
        if (bool(scores) != bool(sample) or (self.readout is None and sample != 0)
                or (self.readout is not None and features.elapsed_ms - sample >= 100)):
            raise ValueError("Semantic cached score/sample state is inconsistent")
        if any(out_state["states"][str(k)]["score"] != v for k, v in scores.items()):
            raise ValueError("Semantic cached and output scores differ")
        if len(events) > len(self.supported_concepts) or len({e.concept_id for e in events}) != len(events):
            raise ValueError("Invalid cached semantic event count")
        for event in events:
            if (event.episode_id != in_state["episode_id"] or event.direction != "from_brain"
                    or event.concept_id not in self.supported_concepts
                    or event.sim_time_ms != sample or event.sim_time_ms != features.elapsed_ms
                    or event.to_dict() != out_state["states"][str(event.concept_id)]["last_event"]):
                raise ValueError("Invalid cached semantic event")
        self.features, self.inbox, self.outbox = features, inbox, outbox
        self.last_scores, self.last_events, self.last_sample_ms = scores, events, sample


class _BrainTap:
    """Private instance proxy; old adapters keep their original Brain reference."""
    def __init__(self, brain, organism, channel):
        self._brain, self._organism, self._channel = brain, organism, channel

    def __getattr__(self, name):
        return getattr(self._brain, name)

    def step(self, luminance, milliseconds=10, **kwargs):
        if milliseconds != 10:
            raise ValueError("Semantic v0.1 arena requires the existing 10ms exchange")
        sensory = dict(kwargs.get("sensory_currents") or {})
        for source in (sensory, kwargs.get("modulation") or {}):
            for indices, _ in source.values():
                if np.intersect1d(indices, self._channel.features.indices).size:
                    raise ValueError("Native direct input overlaps primary semantic features")
        additions = self._channel.currents(self._organism.hunger, self._brain.clock)
        if set(sensory) & set(additions):
            raise ValueError("Semantic channel name collision")
        sensory.update(additions)
        kwargs["sensory_currents"] = sensory
        command, counts = self._brain.step(luminance, milliseconds, **kwargs)
        self._channel.observe(counts, milliseconds, self._brain.clock)
        return command, counts


def semantic_code_digest():
    return digest({p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def attach(simulation, *, enabled=False, mapping=None, calibration=None, readout=None,
           episode_id="demo-0001", hunger_disconnected=False, cue_disconnected=False):
    """Disabled path returns None before reading artifacts, allocating RNG or wrapping."""
    if not enabled:
        return None
    if simulation.brain is None or isinstance(simulation.brain, _BrainTap):
        raise ValueError("Semantic channel needs an unattached full brain")
    channel = SemanticChannel(mapping, calibration, readout=readout, episode_id=episode_id,
        hunger_disconnected=hunger_disconnected, cue_disconnected=cue_disconnected)
    if asdict(simulation.brain.config) != calibration["brain_config"]:
        raise ValueError("Brain configuration differs from calibration")
    if simulation.brain.clock != 0:
        raise ValueError("Attach at episode initialization; use semantic checkpoint to resume")
    simulation.brain = _BrainTap(simulation.brain, simulation.organism, channel)
    return channel


class SemanticSimulation:
    """Owns legacy simulation and semantic state; creates neither a new gait nor policy."""
    def __init__(self, config, *, enabled=False, mapping_path=None, calibration_path=None,
                 readout_path=None, episode_id="demo-0001", hunger_disconnected=False,
                 cue_disconnected=False, **legacy_kwargs):
        from fly_arena.ethology import EthologySimulation
        mapping = calibration = model = None
        # Fail invalid enabled configurations before building the body or stimulating.
        if enabled:
            if mapping_path is None or calibration_path is None:
                raise ValueError("Enabled semantic channel requires calibrated artifacts")
            mapping = load_mapping(mapping_path, legacy_kwargs.get("graph"))
            calibration = load_calibration(calibration_path, mapping)
            if readout_path is not None:
                model = LinearReadout.load(readout_path, expected_feature_digest=make_features(mapping, calibration).feature_digest)
        self.legacy = EthologySimulation(config, **legacy_kwargs)
        try:
            self.channel = attach(self.legacy, enabled=enabled, mapping=mapping, calibration=calibration,
                readout=model, episode_id=episode_id, hunger_disconnected=hunger_disconnected,
                cue_disconnected=cue_disconnected)
            self._initial_legacy_state = copy.deepcopy(self.legacy.get_state())
        except Exception:
            self.legacy.close()
            raise

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def step(self, *, paused=False):
        return self.legacy.step(paused=paused)

    def reset(self, episode_id):
        self.legacy.set_state(copy.deepcopy(self._initial_legacy_state))
        # Legacy restore preserves a prior decision when the supplied one is None.
        # Reset this wrapper's initial episode explicitly without editing old code.
        if self._initial_legacy_state["last_decision"] is None:
            self.legacy.last_decision = None
        if self.channel is not None:
            self.channel.reset(episode_id)

    def compatibility(self):
        if self.channel is None:
            return self.legacy.compatibility()
        return {**self.legacy.compatibility(), "semantic_code": semantic_code_digest(),
                "semantic": self.channel.identity()}

    def get_state(self):
        if self.channel is None:
            return self.legacy.get_state()
        return {"format": "fly_semantic.simulation.v1", "legacy": self.legacy.get_state(),
                "semantic": self.channel.get_state()}

    def set_state(self, state):
        if self.channel is None:
            return self.legacy.set_state(state)
        if state.get("format") != "fly_semantic.simulation.v1":
            raise ValueError("Not a semantic simulation checkpoint")
        candidate = copy.deepcopy(self.channel)
        candidate.set_state(state["semantic"])
        if state["legacy"]["brain"]["clock"] != candidate.features.elapsed_ms:
            raise ValueError("Neural/semantic checkpoint clocks diverged")
        self.legacy.set_state(state["legacy"])
        self.channel.set_state(state["semantic"])

    def close(self):
        self.legacy.close()
