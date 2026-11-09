"""Portable run identities and non-overwriting output records."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import uuid
from datetime import datetime, timezone

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value, *, exclusive=False):
    """Publish a complete JSON record atomically, optionally refusing replacement."""
    payload = json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    created = False
    try:
        # Normal inherited permissions also keep the record readable by later
        # restricted Windows processes. Never truncate the destination in place.
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def read_config(path):
    import yaml
    try:
        with Path(path).open(encoding="utf-8") as stream:
            result = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML/JSON configuration in {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a mapping")
    return result


def runtime():
    packages = {}
    for name in ("maintenance-arena", "numpy", "gymnasium", "PyYAML", "torch", "stable-baselines3", "sb3-contrib", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def begin_run(output, *, command, profile=None, seed=None, inputs=None, nominal_calendar_steps=None, arguments=None):
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    from . import __version__
    record = {
        "schema": "maintenance_arena_run_v1", "package_version": __version__,
        "command": command, "arguments": sys.argv[1:] if arguments is None else json_value(arguments), "profile": profile,
        "seed": seed, "inputs": inputs or {}, "nominal_calendar_steps": nominal_calendar_steps,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime(), "status": "running",
    }
    source_root = Path(__file__).parent
    record["implementation_sha256"] = canonical_hash({p.relative_to(source_root).as_posix(): file_hash(p) for p in sorted(source_root.rglob("*.py"))})
    write_json(directory / "run.json", record)
    return directory, record


def finish_run(directory, record, *, elapsed_seconds, result=None, error=None):
    status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed" if error is not None else "complete"
    record.update(status=status, elapsed_seconds=elapsed_seconds,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    if error is not None:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    else:
        record["result"] = result
    write_json(Path(directory) / "run.json", record)
