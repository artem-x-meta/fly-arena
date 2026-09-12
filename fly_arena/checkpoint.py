"""Atomic, non-pickle checkpoints with explicit compatibility fingerprints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FORMAT_VERSION = 1


def code_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not checkpoint data")
        key = f"array_{len(arrays)}"
        arrays[key] = value
        return {"__ndarray__": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _pack(v, arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(v, arrays) for v in value]
    return value


def _unpack(value, archive):
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            return archive[value["__ndarray__"]].copy()
        return {k: _unpack(v, archive) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v, archive) for v in value]
    return value


def save_checkpoint(path: Path, state: dict, compatibility: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    document = {"format_version": FORMAT_VERSION, "compatibility": compatibility,
                "state": _pack(state, arrays)}
    arrays["metadata_json"] = np.array(json.dumps(document, allow_nan=False, sort_keys=True))
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path, expected: dict | None = None):
    with np.load(path, allow_pickle=False) as archive:
        document = json.loads(str(archive["metadata_json"]))
        if document.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format")
        if expected is not None:
            actual = document["compatibility"]
            different = [key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key)]
            if different:
                raise ValueError("Incompatible checkpoint: " + ", ".join(sorted(different)))
        return _unpack(document["state"], archive), document["compatibility"]
