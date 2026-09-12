"""Optional paper-model driver composed around the unmodified legacy arena."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from fly_arena.ethology import EthologySimulation

from .gf_model import PARAMETER_PATH
from .vision import PaperVision, CircuitVisionAdapter


def lab_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(Path(__file__).parent)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class CircuitSimulation(EthologySimulation):
    def __init__(self, config, *, control="observe", block=(), threshold_mv=1.0, **kwargs):
        if control not in ("observe", "paper") or not math.isfinite(threshold_mv) or threshold_mv <= 0:
            raise ValueError("Invalid circuit control or engineering threshold")
        if set(block) - {"input", "lc4", "lplc2", "i1", "i2", "output"}:
            raise ValueError("Unknown circuit intervention")
        if control == "paper" and not config.get("escape", {}).get("enabled", False):
            raise ValueError("Circuit experiments require an explicit escape-enabled scene")
        super().__init__(config, **kwargs)
        if self.brain is not None and "escape" not in self.registry["pathways"]:
            # Read-only observers may attach to any existing life profile. Add
            # GF rate monitors in this instance only; legacy input currents and
            # the brain state are unchanged, and no on-disk registry is edited.
            from fly_arena.escape_ports import augment_escape_registry
            from fly_arena.neural_ports import NeuralAdapter
            self.registry = augment_escape_registry(self.registry, kwargs["graph"])
            self.adapter = NeuralAdapter(self.brain, self.registry, config["neural"])
        self.lab_control, self.lab_block, self.lab_threshold = control, tuple(block), float(threshold_mv)
        self.lab_source_digest = lab_digest()
        calibration = self.arena.sim.mj_model.cam_fovy[self.arena.eye_ids]
        self.paper_probe = PaperVision(calibration, block=block, threshold_mv=threshold_mv)
        self.circuit_adapter = CircuitVisionAdapter(self.paper_probe, self.looming_detector, control)
        self.looming_detector = self.circuit_adapter

    @property
    def label(self):
        control = getattr(self, "lab_control", "observe")
        if control == "paper":
            return "PAPER GF MODEL / FITTED INPUTS + ENGINEERED MOTOR ADAPTER"
        return "CIRCUIT OBSERVER / ORIGINAL v0.3 POLICY CONTROLS THE BODY"

    def telemetry(self):
        row = super().telemetry()
        if not hasattr(self, "paper_probe"):
            return row
        p = self.paper_probe
        row.update(paper_time_s=p.stream.clock * p.stream.timestep, paper_control=self.lab_control,
                   paper_threshold_mv=p.threshold_mv, paper_block="+".join(self.lab_block))
        for i, side in enumerate(("L", "R")):
            row[f"paper_angle_{side}_deg"] = p.angle_deg[i]
            row[f"paper_angular_speed_{side}_deg_s"] = p.omega_deg_s[i]
            for name in ("lc4_mv", "lplc2_mv", "i1_mv", "i2_mv", "gf_online_mv"):
                row[f"paper_{name}_{side}"] = float(p.last.get(name, [0., 0.])[i])
            row[f"legacy_GF_{side}_Hz"] = self.last_readout.raw_rates.get(f"escape_output_dnp01_{side}", 0.)
            row[f"legacy_detector_{side}"] = self.circuit_adapter.legacy_scores[i]
        return row

    def step(self, *, paused=False):
        result = super().step(paused=paused)
        if result is not None and result["transition"]:
            result["transition"]["circuit_driver"] = self.lab_control
            result["transition"]["paper_gf_mv"] = self.paper_probe.last["gf_online_mv"].tolist()
            result["transition"]["legacy_gf_hz"] = [self.last_readout.raw_rates.get(f"escape_output_dnp01_{s}", 0.) for s in ("L", "R")]
        return result

    def compatibility(self):
        return {**super().compatibility(), "circuit_lab_code": self.lab_source_digest,
                "paper_parameters": hashlib.sha256(PARAMETER_PATH.read_bytes()).hexdigest(),
                "lab_control": self.lab_control, "lab_block": list(self.lab_block),
                "lab_threshold_mv": self.lab_threshold}

    def get_state(self):
        return {"circuit_lab_format": 1, "control": self.lab_control, "block": list(self.lab_block),
                "threshold_mv": self.lab_threshold, "legacy_state": super().get_state()}

    def set_state(self, saved):
        if saved.get("circuit_lab_format") != 1 or saved["control"] != self.lab_control or saved["block"] != list(self.lab_block) or saved["threshold_mv"] != self.lab_threshold:
            raise ValueError("Incompatible circuit laboratory checkpoint")
        super().set_state(saved["legacy_state"])
