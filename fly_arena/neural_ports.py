"""Auditable MaleCNS behaviour ports; annotations are not functional proof.

The runtime graph is never copied or rewired here. Persistent identities are
body IDs, resolved to CSR indices only when an adapter is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .sensors import NeuralReadout


DATASET = "male-cns:v1.0"
ANNOTATIONS_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
SOURCES = {
    "annotations": ANNOTATIONS_URL,
    "mn9": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/",
    "antennal": "https://elifesciences.org/articles/08758",
    "sweet_manc": "https://elifesciences.org/reviewed-preprints/97766",
}


class UnsupportedPortError(ValueError):
    """Requested behaviour lacks a validated sensor-to-physical-action path."""


@dataclass
class NeuralConfig:
    exploratory_ports: bool = False
    taste_gain_mv: float = 28.0
    dust_gain_mv: float = 28.0
    hunger_gain_mv: float = 4.0
    sleep_arousal_reduction_mv: float = 12.0
    feed_threshold_hz: float = 5.0
    groom_threshold_hz: float = 5.0
    rate_tau_s: float = .1
    escape_threshold_hz: float = 5.0

    def __post_init__(self):
        if not isinstance(self.exploratory_ports, bool):
            raise ValueError("neural.exploratory_ports must be boolean")
        for field in fields(self):
            if field.name == "exploratory_ports":
                continue
            value = getattr(self, field.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"neural.{field.name} must be finite and nonnegative")
        if self.rate_tau_s <= 0 or min(self.feed_threshold_hz, self.groom_threshold_hz, self.escape_threshold_hz) <= 0:
            raise ValueError("Neural time constant and readout thresholds must be positive")

    @classmethod
    def from_dict(cls, values: dict | None):
        values = values or {}
        known = {field.name for field in fields(cls)}
        unexpected = set(values) - known
        if unexpected:
            raise ValueError(f"Unknown neural config keys: {sorted(unexpected)}")
        return cls(**values)


def resolve_body_ids(ids: np.ndarray, body_ids) -> np.ndarray:
    """Reject unknown or ambiguous IDs instead of attaching to a nearby row."""
    ids = np.asarray(ids)
    requested = np.asarray(body_ids, dtype=np.int64)
    if ids.ndim != 1 or requested.ndim != 1 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Graph IDs must be unique and increasing; requested IDs must be a vector")
    if len(np.unique(requested)) != len(requested):
        raise ValueError("Duplicate body ID in port")
    if not len(requested):
        return np.empty(0, np.int32)
    indices = np.searchsorted(ids, requested)
    if not len(ids) or np.any(indices >= len(ids)) or np.any(ids[np.minimum(indices, len(ids) - 1)] != requested):
        raise ValueError("Port contains body IDs absent from this graph")
    return indices.astype(np.int32)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_registry(graph: Path, output: Path | None = None) -> dict:
    """Select ports from the prepared release, preserving annotation provenance."""
    import pyarrow.feather as feather

    graph = Path(graph)
    manifest = json.loads((graph / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET:
        raise ValueError(f"Behaviour queries are reviewed only for {DATASET}")
    ids = np.load(graph / "ids.npy", mmap_mode="r")
    annotation_path = graph / "neurons.feather"
    columns = ["bodyId", "type", "class", "subclass", "superclass", "somaSide", "rootSide",
               "synonyms", "flywireType", "mancType", "entryNerve", "receptorType", "matchingNotes"]
    table = feather.read_table(annotation_path)
    absent = set(columns) - set(table.column_names)
    if absent:
        raise ValueError(f"Annotations lack required provenance fields: {sorted(absent)}")
    records = table.select(columns).to_pylist()
    choices = [
        ("feed_output_mn9", "motor_output", "annotated_match", "type == 'MN9' AND superclass == 'cb_motor'", SOURCES["mn9"],
         lambda row: row["type"] == "MN9" and row["superclass"] == "cb_motor"),
        ("groom_output_adn", "descending_output", "annotated_match", "synonyms IN ['Hampel 2015: aDN1', 'Hampel 2015: aDN2'] AND superclass == 'descending_neuron'", SOURCES["antennal"],
         lambda row: row["synonyms"] in ("Hampel 2015: aDN1", "Hampel 2015: aDN2") and row["superclass"] == "descending_neuron"),
        ("antennal_mechanosensory", "sensory_input", "annotated_match", "class == 'mechanosensory' AND subclass == 'grooming' AND type IN ['JO-FV', 'JO-FD1']", SOURCES["antennal"],
         lambda row: row["class"] == "mechanosensory" and row["subclass"] == "grooming" and row["type"] in ("JO-FV", "JO-FD1")),
        ("taste_leg_sweet", "sensory_input", "candidate_homolog", "type == 'LgAG1' AND class == 'gustatory' AND mancType == 'SAch02'", SOURCES["sweet_manc"],
         lambda row: row["type"] == "LgAG1" and row["class"] == "gustatory" and row["mancType"] == "SAch02"),
    ]
    groups = {}
    for name, role, confidence, query, source, predicate in choices:
        selected = sorted((row for row in records if predicate(row)), key=lambda row: row["bodyId"])
        body_ids = [row["bodyId"] for row in selected]
        resolve_body_ids(ids, body_ids)
        groups[name] = {
            "dataset": DATASET, "release": "v1.0", "role": role, "confidence": confidence,
            "query": query, "body_ids": body_ids, "neurons": selected,
            "sources": [SOURCES["annotations"], source],
            "status": "unsupported", "annotation_status": "resolved" if selected else "unresolved",
            "functional_test": {"status": "not_run", "network": None, "physical": None},
        }
    for name, role, reason in [
        ("taste_mouth_sweet", "sensory_input", "Labellar modality labels not verified in this MaleCNS release"),
        ("odor_navigation", "sensory_input", "No validated local odour-to-navigation chain"),
        ("sleep_output", "output", "Sleep is an explicit organism model, not a reconstructed neural centre"),
        ("front_groom_output", "output", "Antennal descending outputs do not establish front-leg rubbing control"),
    ]:
        groups[name] = {"dataset": DATASET, "release": "v1.0", "role": role,
                        "confidence": "unresolved", "query": None, "body_ids": [], "neurons": [],
                        "sources": [], "status": "unsupported", "annotation_status": "unresolved",
                        "reason": reason, "functional_test": {"status": "not_run", "network": None, "physical": None}}
    registry = {
        "schema_version": 1, "dataset": DATASET, "release": "v1.0",
        "graph_arrays": manifest.get("arrays", {}),
        "graph_ids_sha256": _digest(graph / "ids.npy"),
        "prepared_annotations_sha256": _digest(annotation_path),
        "original_annotations": manifest.get("sources", {}).get("annotations", {}),
        "groups": groups,
        "pathways": {
            "feeding": {"status": "unsupported", "inputs": ["taste_leg_sweet"], "outputs": ["feed_output_mn9"],
                        "reason": "Candidate sweet homology; network and physical four-condition experiments required"},
            "grooming": {"status": "unsupported", "inputs": ["antennal_mechanosensory"], "outputs": ["groom_output_adn"],
                         "reason": "Annotated ports; network and physical four-condition experiments required"},
        },
        "engineering": {
            "dust": "Local head dirt 0..1 -> current in annotated antennal mechanosensory cells; engineering proxy for antennal irritation",
            "taste": "Local tarsal attractive taste 0..1 -> current in candidate sweet leg afferents",
            "hunger": "Extra current only in taste afferents already receiving local taste; no direct MN9 drive",
            "sleep": "Sleep pressure reduces engineered DNp09 arousal bias; it does not erase sensory channels",
            "decoder": "Thresholded low-pass rates gate engineered motor primitives; no neuron-to-muscle registration",
        },
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def _level(value: Any) -> float:
    if isinstance(value, dict):
        value = max((_level(item) for item in value.values()), default=0.)
    elif isinstance(value, (list, tuple, np.ndarray)):
        value = np.max(value) if len(value) else 0.
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Neural sensory levels must be finite")
    return float(np.clip(value, 0, 1))


class NeuralAdapter:
    def __init__(self, brain, registry: dict | Path, config: NeuralConfig | dict | None = None):
        self.brain = brain
        self.registry = json.loads(Path(registry).read_text(encoding="utf-8")) if isinstance(registry, (str, Path)) else registry
        if self.registry.get("schema_version") != 1 or self.registry.get("dataset") != brain.manifest.get("dataset"):
            raise ValueError("Behaviour registry schema or dataset does not match graph")
        expected_arrays = self.registry.get("graph_arrays")
        if expected_arrays is not None and expected_arrays != brain.manifest.get("arrays", {}):
            raise ValueError("Behaviour registry graph fingerprints do not match this graph")
        self.config = config if isinstance(config, NeuralConfig) else NeuralConfig.from_dict(config)
        ids = np.asarray(brain.ids)
        self.indices = {name: resolve_body_ids(ids, group["body_ids"]) for name, group in self.registry["groups"].items()}
        for name, group in self.registry["groups"].items():
            if [row["bodyId"] for row in group["neurons"]] != group["body_ids"]:
                raise ValueError(f"Port {name} annotation rows do not match its body IDs")
        self.sides = {name: np.array([row.get("rootSide") or row.get("somaSide") or "?" for row in group["neurons"]])
                      for name, group in self.registry["groups"].items()}
        self.filtered_rates = {name: 0. for name in self.indices}
        self.last_readout = None

    def require_supported(self, pathways=("feeding", "grooming")):
        missing = [name for name in pathways if self.registry.get("pathways", {}).get(name, {}).get("status") != "supported"]
        if missing and not self.config.exploratory_ports:
            detail = "; ".join(f"{name}: {self.registry.get('pathways', {}).get(name, {}).get('reason', 'unresolved')}" for name in missing)
            raise UnsupportedPortError(f"Unsupported required neural pathways before simulation: {detail}. Explicit exploratory_ports=true permits experimental candidates.")
        for name in pathways:
            path = self.registry.get("pathways", {}).get(name, {})
            ports = path.get("inputs", []) + path.get("outputs", [])
            if not ports or any(not len(self.indices.get(port, [])) for port in ports):
                raise UnsupportedPortError(f"Neural pathway {name} has unresolved/empty ports, including in exploratory mode")

    def currents(self, sensor_frame, organism) -> tuple[dict, dict]:
        tarsal = getattr(sensor_frame, "tarsal_taste", {})
        if isinstance(tarsal, dict):
            by_side = {side: _level({key: value for key, value in tarsal.items()
                       if key in (("LF", "LM", "LH", "front_left", "middle_left", "hind_left") if side == "L"
                                  else ("RF", "RM", "RH", "front_right", "middle_right", "hind_right"))}) for side in ("L", "R")}
        else:
            by_side = dict.fromkeys(("L", "R"), _level(tarsal))
        taste = np.asarray([by_side.get(side, 0.) for side in self.sides["taste_leg_sweet"]], np.float32)
        dust = getattr(sensor_frame, "dust_afferents", {})
        dust_by_side = {side: _level({"head": dust.get("head", 0.), "antenna": dust.get("antenna_left" if side == "L" else "antenna_right", 0.)}) for side in ("L", "R")} if isinstance(dust, dict) else {"L": 0., "R": 0.}
        head_dust = np.asarray([dust_by_side.get(side, 0.) for side in self.sides["antennal_mechanosensory"]], np.float32)
        hunger = getattr(organism, "hunger", 0.)
        hunger = _level(hunger() if callable(hunger) else hunger)
        sleep = _level(getattr(getattr(organism, "state", organism), "sleep_pressure", 0.))
        sensory = {
            "taste": (self.indices["taste_leg_sweet"], self.config.taste_gain_mv * taste),
            "dust": (self.indices["antennal_mechanosensory"], self.config.dust_gain_mv * head_dust),
        }
        arousal_indices = np.asarray(self.brain.ports["DNp09"]["L"] + self.brain.ports["DNp09"]["R"], np.int32)
        reduction = min(self.config.sleep_arousal_reduction_mv, max(0., self.brain.config.arousal))
        modulation = {
            "hunger": (self.indices["taste_leg_sweet"], self.config.hunger_gain_mv * hunger * taste),
            "sleep": (arousal_indices, -reduction * sleep),
        }
        return sensory, modulation

    def blocked_indices(self, names) -> np.ndarray:
        result = []
        for name in names:
            if name in self.registry.get("pathways", {}):
                for output in self.registry["pathways"][name]["outputs"]:
                    result.extend(self.indices[output])
            elif name in self.indices:
                result.extend(self.indices[name])
            else:
                raise ValueError(f"Unknown neural output port {name}")
        return np.unique(result).astype(np.int32)

    def readout(self, counts, dt: float) -> NeuralReadout:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Neural readout dt must be positive")
        counts = np.asarray(counts)
        if counts.shape != self.brain.voltage.shape or np.any(counts < 0):
            raise ValueError("Invalid per-neuron spike counts")
        alpha = -np.expm1(-dt / self.config.rate_tau_s)
        raw_rates = {}
        for name, selected in self.indices.items():
            rate = float(np.mean(counts[selected]) / dt) if len(selected) else 0.
            raw_rates[name] = rate
            self.filtered_rates[name] += alpha * (rate - self.filtered_rates[name])
            blocked = getattr(self.brain, "last_blocked_outputs", ())
            if len(selected) and len(blocked) and np.isin(selected, blocked).all():
                # A readout filter must not resurrect a deliberately blocked
                # output as a delayed motor request after the intervention.
                self.filtered_rates[name] = 0.
            for side in ("L", "R"):
                side_ids = selected[self.sides[name] == side] if len(selected) else selected
                raw_rates[f"{name}_{side}"] = float(np.mean(counts[side_ids]) / dt) if len(side_ids) else 0.
        statuses = {}
        for path in ("feeding", "grooming"):
            original = self.registry["pathways"][path]["status"]
            statuses[path] = "supported" if original == "supported" else "experimental" if self.config.exploratory_ports else "unsupported"
        def drive(path, group, threshold):
            if statuses[path] == "unsupported":
                return 0.
            return float(np.clip((self.filtered_rates[group] - threshold) / threshold, 0, 1))
        command = np.asarray(getattr(self.brain, "last_command", np.zeros(2)))
        statuses.update({"walking": "engineered_decoder", "sleep": "unsupported", "front_groom": "unsupported", "odor_navigation": "unsupported", "mouth_taste": "unsupported"})
        statuses.update({"feed": statuses["feeding"], "groom_head": statuses["grooming"], "groom_front": "unsupported"})
        self.last_readout = NeuralReadout(float(command[0]), float(command[1]),
            drive("feeding", "feed_output_mn9", self.config.feed_threshold_hz),
            {"head": drive("grooming", "groom_output_adn", self.config.groom_threshold_hz), "front": 0.},
            0., statuses, raw_rates)
        if "escape" in self.registry["pathways"]:
            status = self.registry["pathways"]["escape"]["status"]
            status = "supported" if status == "supported" else "experimental" if self.config.exploratory_ports else "unsupported"
            self.last_readout.supported_ports["escape"] = status
            if status != "unsupported":
                rate = self.filtered_rates["escape_output_dnp01"]
                self.last_readout.escape_drive = float(np.clip((rate - self.config.escape_threshold_hz) / self.config.escape_threshold_hz, 0, 4))
                left = raw_rates["escape_output_dnp01_L"]
                right = raw_rates["escape_output_dnp01_R"]
                self.last_readout.escape_turn = -1. if left >= right else 1.
        return self.last_readout

    def get_state(self) -> dict:
        return {"filtered_rates": self.filtered_rates.copy()}

    def set_state(self, state: dict):
        rates = state["filtered_rates"]
        if set(rates) != set(self.filtered_rates) or any(not np.isfinite(value) or value < 0 for value in rates.values()):
            raise ValueError("Invalid neural adapter checkpoint rates")
        self.filtered_rates = dict(rates)
