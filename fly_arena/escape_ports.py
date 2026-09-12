"""MaleCNS looming candidates; retinal stimulation only, no injected threat drive."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .neural_ports import resolve_body_ids


def augment_escape_registry(registry, graph: Path):
    import pyarrow.feather as feather
    graph = Path(graph)
    result = copy.deepcopy(registry)
    columns = ["bodyId", "type", "somaSide", "rootSide", "superclass", "hemibrainType", "flywireType", "synonyms"]
    rows = feather.read_table(graph / "neurons.feather", columns=columns).to_pylist()
    ids = np.load(graph / "ids.npy", mmap_mode="r")
    groups = result["groups"]
    for kind in ("LPLC2", "LC4", "LPLC4", "DNp01"):
        selected = sorted((r for r in rows if r["type"] == kind), key=lambda r: r["bodyId"])
        name = "escape_output_dnp01" if kind == "DNp01" else f"escape_candidate_{kind.lower()}"
        body_ids = [r["bodyId"] for r in selected]
        resolve_body_ids(ids, body_ids)
        sources = ["https://male-cns.janelia.org/download/"]
        if kind == "LPLC2":
            sources.append("https://www.nature.com/articles/nature24626")
        elif kind in ("LC4", "DNp01"):
            sources.append("https://www.janelia.org/publication/neural-basis-for-looming-size-and-velocity-encoding-in-the-drosophila-giant-fiber-escape")
        groups[name] = {"dataset": result["dataset"], "release": result["release"],
                        "role": "descending_output_candidate" if kind == "DNp01" else "visual_interneuron_candidate",
                        "confidence": "annotated_type_engineered_running_proxy" if kind == "DNp01" else "annotated_type",
                        "query": f"type == '{kind}'", "body_ids": body_ids, "neurons": selected,
                        "sources": sources,
                        "status": "unsupported", "functional_test": {"status": "not_run"}}
    retina = json.loads((graph / "ports.json").read_text())["retina"]
    retina_ids = ids[np.asarray(retina, dtype=int)].tolist()
    selected_ids = set(retina_ids)
    groups["escape_retinal_input"] = {
        "dataset": result["dataset"], "release": result["release"], "role": "existing_visual_input",
        "confidence": "engineering_camera_registration", "query": "Existing ports.json retina indexes resolved through ids.npy",
        "body_ids": retina_ids, "neurons": [],
        "sources": ["https://male-cns.janelia.org/download/"], "status": "unsupported",
        "functional_test": {"status": "not_run"}}
    by_id = {r["bodyId"]: r for r in rows if r["bodyId"] in selected_ids}
    groups["escape_retinal_input"]["neurons"] = [by_id[body_id] for body_id in retina_ids]
    result["pathways"]["escape"] = {
        "status": "unsupported", "inputs": ["escape_retinal_input"], "outputs": ["escape_output_dnp01"],
        "intermediates": ["escape_candidate_lplc2", "escape_candidate_lc4", "escape_candidate_lplc4"],
        "reason": "Named MaleCNS candidates; no validated camera-to-output-to-running escape chain",
        "engineering_limit": "Giant Fiber's documented takeoff role does not validate the engineered running primitive"}
    result.setdefault("engineering", {})["escape"] = "Existing camera-to-retina currents only; GF output gates a labelled running proxy. No direct current from the hybrid image detector."
    return result
