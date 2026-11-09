"""Versioned task profiles; a profile defines a task, not the identity of a past run."""
from __future__ import annotations
import copy
from importlib.resources import files
import json

from .records import canonical_hash


def all_profiles():
    return json.loads(files("maintenance_arena").joinpath("data/profiles.json").read_text(encoding="utf-8"))


def _merge(destination, overrides, *, strict=False):
    for name, value in overrides.items():
        if strict and name not in destination:
            raise ValueError(f"Unknown profile configuration field: {name}")
        if isinstance(destination.get(name), dict) and isinstance(value, dict):
            _merge(destination[name], value, strict=strict)
        else:
            destination[name] = copy.deepcopy(value)


def resolve_profile(identifier, overrides=None):
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Profile overrides must be a mapping")
    profiles = all_profiles()
    if identifier not in profiles:
        raise ValueError(f"Unknown profile {identifier!r}; choose from {', '.join(profiles)}")
    result = copy.deepcopy(profiles[identifier])
    if overrides:
        allowed = {"env_config", "training", "policy_defaults"}
        if set(overrides) - allowed:
            raise ValueError(f"Only {sorted(allowed)} can be overridden")
        for section, values in overrides.items():
            if not isinstance(values, dict):
                raise ValueError(f"{section} must be a mapping")
            _merge(result[section], values, strict=section == "policy_defaults")
    result["base_profile"] = identifier
    result["configuration_sha256"] = canonical_hash(result)
    result["modified"] = bool(overrides)
    result["id"] = identifier if not overrides else f"{identifier}+{result['configuration_sha256'][:12]}"
    if overrides:
        result["base_evidence_scope"] = result["evidence_scope"]
        result["evidence_scope"] = "Modified configuration; published validation is not inherited. New executions require their own evidence."
    return result
