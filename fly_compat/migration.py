"""Copy v0.3 checkpoints across the optional fly-name factory argument only.

This migration is deliberately pinned to an archived whole-package digest.
It cannot authorize unrelated or future source changes. Neither source files
nor the input checkpoint are edited, and all normal runtime fingerprints are
checked against a freshly compiled single-fly simulation before publication.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

ARCHIVED_CODE = "e4ae8ad23ead9725ddab53bee0529448b7916ffebc9a62d470263667de4817ca"
ROOT = Path(__file__).resolve().parents[1]
FACTORY_REVERTS = (
    (b'def make_ethology_fly(name="fly"):', b'def make_ethology_fly():'),
    (b'NeuroMechFly(name=name)', b'NeuroMechFly(name="fly")'),
)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prove_factory_delta(source_dir=None):
    """Prove the only source difference is the known default-preserving API."""
    source_dir = Path(source_dir) if source_dir is not None else ROOT / "fly_arena"
    current, normalized = hashlib.sha256(), hashlib.sha256()
    calls = []
    files = sorted(source_dir.glob("*.py"))
    if not files:
        raise ValueError("No single-fly source package found")
    for path in files:
        content = path.read_bytes()
        reverted = content
        if path.name == "ethology_arena.py":
            for new, old in FACTORY_REVERTS:
                if reverted.count(new) != 1:
                    raise ValueError("Factory source is not the exact known optional-name change")
                reverted = reverted.replace(new, old, 1)
        current.update(path.name.encode())
        current.update(content)
        normalized.update(path.name.encode())
        normalized.update(reverted)
        for node in ast.walk(ast.parse(content, filename=str(path))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "make_ethology_fly":
                if node.args or node.keywords:
                    raise ValueError("Single-fly factory call does not use its unchanged default")
                calls.append({"path": path.name, "line": node.lineno, "arguments": []})
    if normalized.hexdigest() != ARCHIVED_CODE:
        raise ValueError("Source differs beyond the exact known factory change; migration refused")
    if not calls:
        raise ValueError("No default single-fly factory call found")
    return {"archived_code": ARCHIVED_CODE, "current_code": current.hexdigest(),
            "canonicalized_code": normalized.hexdigest(), "source_files": len(files),
            "in_memory_replacements": [
                {"current": new.decode(), "archived": old.decode()} for new, old in FACTORY_REVERTS],
            "default_single_fly_calls": calls}


def check_fingerprints(expected, actual, proof):
    if expected.get("code") != ARCHIVED_CODE:
        raise ValueError("Only the archived v0.3 code fingerprint is eligible")
    if actual.get("code") != proof["current_code"]:
        raise ValueError("Runtime source fingerprint changed during verification")
    differing = [key for key in set(expected) | set(actual)
                 if key != "code" and expected.get(key) != actual.get(key)]
    if differing:
        raise ValueError("Incompatible checkpoint beyond code: " + ", ".join(sorted(differing)))
    return sorted(key for key in actual if key != "code")


def assert_same_state(first, second, path="state"):
    """Exact recursive state equality, including native MuJoCo and neural arrays."""
    if isinstance(first, np.ndarray):
        if (not isinstance(second, np.ndarray) or first.dtype != second.dtype
                or first.shape != second.shape or first.tobytes() != second.tobytes()):
            raise ValueError(f"Restored state differs at {path}")
    elif isinstance(first, dict):
        if not isinstance(second, dict) or set(first) != set(second):
            raise ValueError(f"Restored state keys differ at {path}")
        for key in first:
            assert_same_state(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        if not isinstance(second, (list, tuple)) or len(first) != len(second):
            raise ValueError(f"Restored state sequence differs at {path}")
        for index, (left, right) in enumerate(zip(first, second)):
            assert_same_state(left, right, f"{path}[{index}]")
    elif first != second:
        raise ValueError(f"Restored state differs at {path}")


def _paths(input_path, output_path):
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if source == destination or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError("Migration must create a separate copy; input cannot be overwritten")
    if destination.exists():
        raise FileExistsError(f"Migration destination already exists: {destination}")
    if not source.is_file():
        raise FileNotFoundError(source)
    return source, destination


def _read_document(archive):
    return json.loads(str(np.load(io.BytesIO(archive.read("metadata_json.npy")), allow_pickle=False)))


def copy_checkpoint_metadata(input_path, output_path, expected, replacement):
    """Internal copy operation: change only code metadata, preserve array bytes.

    The public migration always obtains both fingerprints from its source proof
    and real runtime validation first. No CLI option accepts arbitrary hashes.
    """
    source, destination = _paths(input_path, output_path)
    if set(expected) != set(replacement) or any(expected[k] != replacement[k] for k in expected if k != "code"):
        raise ValueError("Only compatibility.code may change")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".migration-", suffix=".npz", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        members = []
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(temporary, "w") as new:
            document = _read_document(old)
            if document.get("format_version") != 1 or document.get("compatibility") != expected:
                raise ValueError("Checkpoint metadata changed during migration")
            if len(old.namelist()) != len(set(old.namelist())):
                raise ValueError("Duplicate checkpoint archive members")
            document["compatibility"] = copy.deepcopy(replacement)
            metadata = io.BytesIO()
            np.save(metadata, np.array(json.dumps(document, allow_nan=False, sort_keys=True)), allow_pickle=False)
            for info in old.infolist():
                payload = metadata.getvalue() if info.filename == "metadata_json.npy" else old.read(info.filename)
                new.writestr(copy.copy(info), payload)
                if info.filename != "metadata_json.npy":
                    members.append({"member": info.filename, "sha256": hashlib.sha256(payload).hexdigest()})
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(temporary) as new:
            old_document, new_document = _read_document(old), _read_document(new)
            old_document["compatibility"]["code"] = replacement["code"]
            if old_document != new_document or old.namelist() != new.namelist():
                raise ValueError("Migration changed metadata beyond compatibility.code")
            if any(old.read(row["member"]) != new.read(row["member"]) for row in members):
                raise ValueError("Migration changed checkpoint state arrays")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        # Unlike os.replace, atomic hard-link publication never overwrites an
        # existing file, including one created after the initial path check.
        os.link(temporary, destination)
        return members
    finally:
        temporary.unlink(missing_ok=True)


def migrate(input_path, output_path, *, graph=None):
    source, destination = _paths(input_path, output_path)
    proof = prove_factory_delta()
    source_hash = file_digest(source)
    from fly_arena.checkpoint import load_checkpoint
    saved, expected = load_checkpoint(source)
    if "circuit_lab_format" in saved or saved.get("diagnostic_body_only", True):
        raise ValueError("This migration supports normal v0.3 single-fly checkpoints; lab/body-only is unsupported")
    if expected.get("code") != ARCHIVED_CODE:
        raise ValueError("Only the archived v0.3 code fingerprint is eligible")
    graph = Path(graph).resolve() if graph is not None else ROOT / "data/graph"
    for name in ("manifest.json", "behavior_ports.json"):
        if not (graph / name).is_file():
            raise FileNotFoundError(f"Existing prepared graph and port registry required: {graph / name}")
    from fly_arena.brain import BrainConfig
    from fly_arena.ethology import EthologySimulation
    flags = {key: saved[key] for key in ("blind", "motor_off", "disabled_channels", "blocked_outputs")}
    sim = EthologySimulation(saved["config"], graph=graph, seed=saved["seed"], mode=saved["mode"],
                            brain_config=BrainConfig(**saved["brain_config"]), warmup=False, **flags)
    try:
        actual = sim.compatibility()
        checked = check_fingerprints(expected, actual, proof)
        sim.set_state(saved)
        assert_same_state(saved, sim.get_state())
        time_s = sim.organism.clocks.physics_time_s
    finally:
        sim.close()
    if prove_factory_delta() != proof or file_digest(source) != source_hash:
        raise ValueError("Source code or input checkpoint changed during verification")
    members = copy_checkpoint_metadata(source, destination, expected, actual)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "input": str(source),
              "output": str(destination), "input_sha256": source_hash,
              "input_sha256_after": file_digest(source), "output_sha256": file_digest(destination),
              "source_proof": proof, "verified_fingerprints": checked,
              "old_compatibility": expected, "new_compatibility": actual,
              "physics_time_s": time_s, "native_state_restored_exactly": True,
              "changed_metadata_paths": ["compatibility.code"],
              "array_members_unchanged": members}
    return report
