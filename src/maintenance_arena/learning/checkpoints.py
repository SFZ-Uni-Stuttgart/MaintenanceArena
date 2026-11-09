"""Bound checkpoint loading and complete native continuation state.

Weight-only checkpoints contain policy weights without optimizer or collector
state. They can be evaluated or used to initialize a new run, but they never
resume training exactly.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
import torch

from .ppo import ActorCritic, canonical_hash, observation_array, parameter_hash, recipe_config

SCHEMA = "maintenance_arena_custom_ppo_checkpoint_v1"


def atomic_write(path, write, validate=None):
    """Publish a complete file atomically without replacing an existing file.

    The temporary file is on the same filesystem. A hard-link publication is
    exclusive on both Windows and Linux; removing the temporary name leaves
    one ordinary final file. No symbolic links or overwrite window are used.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if validate is not None:
            validate(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    """Keep checkpoints compatible with Torch's restricted weights-only reader."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint state type: {type(value).__name__}")


def observation_contract(env):
    return {"shape": list(env.observation_space.shape), "dtype": str(env.observation_space.dtype),
            "names": list(env.observation_names), "actions": int(env.action_space.n),
            "lower": env.observation_space.low.tolist(), "upper": env.observation_space.high.tolist()}


def _environment_config(profile, config):
    if profile == "paper-gearbox-v1":
        from ..gearbox import GearboxEnv
        env = GearboxEnv(config)
        try:
            return env.config
        finally:
            env.close()
    from ..core import validate_config
    return validate_config(config)


def save_native(path, *, agent, env, collector):
    """No-clobber checkpoint after a completed optimizer update."""
    path = Path(path)
    if agent.rollout_updates < 1:
        raise ValueError("Native checkpoint requires a completed update")
    contract = observation_contract(env)
    state = agent.snapshot()
    payload = {"schema_version": SCHEMA, "profile": agent.profile, "algorithm": "PPO",
               "candidate_id": "P-C03", "recipe": deepcopy(agent.config),
               "recipe_sha256": canonical_hash(agent.config), "env_config": _plain(env.config),
               "env_config_sha256": canonical_hash(_plain(env.config)),
               "observation_contract": contract, "observation_sha256": canonical_hash(contract),
               "learner": state, "policy_parameter_sha256": parameter_hash(state["policy_state"]),
               "trainable_parameter_count": sum(p.numel() for p in agent.network.parameters()),
               "environment": env.snapshot(), "collector": deepcopy(collector),
               "torch_rng_state": torch.get_rng_state().clone(),
               "runtime": {"python": platform.python_version(), "numpy": str(np.__version__), "torch": str(torch.__version__)},
               "resume_boundary": "completed_optimizer_update", "normalization": "none; fixed four-field physical representation",
               "value_offset_scaled": agent.config["center_rate"] / (1000. * (1. - agent.config["gamma"]))}
    def validate(candidate):
        check = load_native(candidate, profile=agent.profile, env_config=env.config, recipe=agent.config)
        if check["policy_parameter_sha256"] != payload["policy_parameter_sha256"]:
            raise ArithmeticError("Native checkpoint roundtrip changed weights")
    atomic_write(path, lambda handle: torch.save(_plain(payload), handle), validate)
    return path


def _read(path, expected_sha256=None):
    path = Path(path)
    if path.is_dir():
        path = path / "final.pt"
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Checkpoint file SHA-256 differs from the supplied artifact binding")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a structured payload")
    return payload


def load_native(path, *, profile, env_config=None, recipe=None):
    recipe_config(profile)
    payload = _read(path)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("Weight-only checkpoints are for evaluation and initialization; exact resume requires a native checkpoint")
    if payload.get("profile") != profile or payload.get("resume_boundary") != "completed_optimizer_update":
        raise ValueError("Native checkpoint profile/boundary mismatch")
    for field, binding in (("recipe", "recipe_sha256"), ("env_config", "env_config_sha256"),
                           ("observation_contract", "observation_sha256")):
        if canonical_hash(payload[field]) != payload[binding]:
            raise ValueError(f"Native checkpoint {field} binding failed")
    if env_config is not None and canonical_hash(_plain(_environment_config(profile, env_config))) != payload["env_config_sha256"]:
        raise ValueError("Checkpoint environment configuration differs")
    if recipe is not None and canonical_hash(recipe) != payload["recipe_sha256"]:
        raise ValueError("Checkpoint PPO recipe differs")
    expected = 9220 if profile == "paper-gearbox-v1" else 9155
    learner = payload["learner"]
    if (payload.get("trainable_parameter_count") != expected
            or parameter_hash(learner["policy_state"]) != payload.get("policy_parameter_sha256")):
        raise ValueError("Native checkpoint parameter geometry/hash failed")
    if payload["collector"]["calendar_steps"] < 1 or learner["rollout_updates"] < 1:
        raise ValueError("Incomplete native collector/update state")
    return payload


class Policy:
    def __init__(self, state, action_count, metadata):
        # Local RNG fork avoids changing a caller's sampling stream merely by loading.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.network = ActorCritic(action_count).cpu()
        self.network.load_state_dict(state, strict=True)
        if parameter_hash(self.network.state_dict()) != parameter_hash(state):
            raise ArithmeticError("Policy reload changed parameter bytes")
        self.network.eval()
        self.metadata = metadata
        self.action_count = action_count

    def probabilities(self, observation):
        with torch.no_grad():
            logits, _ = self.network(torch.from_numpy(observation_array(observation)).reshape(1, -1))
            return torch.softmax(logits, dim=-1)[0].numpy().copy()

    def probabilities_batch(self, observations):
        """Float32 batch inference in chunks of 32,768 rows.

        Matrix batches can round differently from repeated single-observation
        matrix/vector products, so batch callers use this path explicitly rather
        than looping ``predict``.
        """
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim < 2 or values.shape[-1] != 4 or not np.isfinite(values).all():
            raise ValueError("Expected finite (...,4) policy inputs")
        flat = values.reshape(-1, 4)
        if not len(flat):
            return np.empty((*values.shape[:-1], self.action_count), dtype=np.float32)
        output = []
        with torch.inference_mode():
            for start in range(0, len(flat), 32768):
                logits, _ = self.network(torch.from_numpy(flat[start:start + 32768].copy()))
                output.append(torch.softmax(logits, -1).numpy())
        return np.concatenate(output).reshape(*values.shape[:-1], self.action_count)

    def predict(self, observation):
        with torch.no_grad():
            logits, _ = self.network(torch.from_numpy(observation_array(observation)).reshape(1, -1))
            scores = torch.softmax(logits, dim=-1)[0] if self.action_count == 2 else logits[0]
            return int(torch.argmax(scores).item())

    __call__ = predict

    def value(self, observation, *, raw=False):
        """Return value in training units; raw=True removes only reward centering.

        Both coordinates keep the reward scale of 1/1000. Multiplying by 1000
        converts the uncentered value to discounted currency units.
        """
        with torch.no_grad():
            _, values = self.network(torch.from_numpy(observation_array(observation)).reshape(1, -1))
            value = float(values[0])
        return value + (float(self.metadata.get("value_offset_scaled", 0.)) if raw else 0.)


def load_policy(checkpoint, *, profile, config=None, expected_sha256=None):
    """Load native or weight-only checkpoints with explicit profile provenance.

    ``config`` is the resolved environment configuration when supplied. Unversioned
    weight-only gearbox files require the external manifest's complete file hash.
    No observed tensor geometry is taken as proof of training provenance.
    """
    recipe_config(profile)
    checkpoint = Path(checkpoint)
    if checkpoint.suffix.lower() == ".json":
        manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
        if manifest.get("profile") != profile:
            raise ValueError("Initialization/checkpoint manifest profile differs")
        if expected_sha256 is not None and file_sha256(checkpoint) != expected_sha256:
            raise ValueError("Checkpoint manifest SHA-256 differs")
        entry = manifest.get("files", {}).get("checkpoint")
        if not isinstance(entry, Mapping) or not {"path", "sha256"}.issubset(entry):
            raise ValueError("Checkpoint manifest lacks the bound model")
        resolved = (checkpoint.parent / entry["path"]).resolve()
        if not resolved.is_relative_to(checkpoint.parent.resolve()):
            raise ValueError("Checkpoint manifest path escapes its directory")
        checkpoint, expected_sha256 = resolved, entry["sha256"]
    payload = _read(checkpoint, expected_sha256)
    action_count = 3 if profile == "paper-gearbox-v1" else 2
    schema = payload.get("schema_version")
    metadata = {"profile": profile, "checkpoint": str(Path(checkpoint).resolve()), "format": schema,
                "evaluation_only": True, "normalization": "none"}
    if schema == SCHEMA:
        payload = load_native(checkpoint, profile=profile, env_config=config)
        state = payload["learner"]["policy_state"]
        metadata.update(evaluation_only=False, value_offset_scaled=payload["value_offset_scaled"],
                        recipe_sha256=payload["recipe_sha256"], observation_sha256=payload["observation_sha256"])
    elif schema == "maintenance_causal_slack_bounded_agent_model_v1":
        if action_count != 2 or payload.get("algorithm") != "PPO" or payload.get("candidate_id") != "P-C03" or payload.get("candidate_index") != 3:
            raise ValueError("Weight-only core checkpoint algorithm/recipe differs")
        state = payload.get("policy_state")
        if not isinstance(state, Mapping) or payload.get("trainable_parameter_count") != 9155:
            raise ValueError("Weight-only core checkpoint geometry differs")
        if parameter_hash(state) != payload.get("policy_parameter_sha256"):
            raise ValueError("Weight-only core tensor hash failed")
        recorded_schema = str(payload.get("metadata", {}).get("schema_version", ""))
        if not recorded_schema:
            raise ValueError("Weight-only core checkpoint lacks source-study identity")
        continuous = "continuous_exact_state_source" in recorded_schema
        if continuous != (profile == "paper-core-continuous-v1"):
            raise ValueError("Weight-only core source distribution differs from selected profile")
        metadata["legacy_metadata"] = deepcopy(payload.get("metadata", {}))
    elif schema == "bounded_three_action_gearbox_PPO_checkpoint_v1":
        if (action_count != 3 or payload.get("algorithm") != "PPO" or payload.get("candidate_id") != "P-C03"
                or payload.get("candidate_index") != 3 or payload.get("parameter_counts", {}).get("total") != 9220):
            raise ValueError("Weight-only gearbox checkpoint identity/geometry differs")
        state = payload.get("state_dict")
        if not isinstance(state, Mapping) or parameter_hash(state) != payload.get("policy_parameter_sha256"):
            raise ValueError("Weight-only gearbox tensor hash failed")
        metadata.update(objective="historical episodic renewal", value_offset_scaled=0.)
    elif schema is None and "state_dict" in payload:
        if action_count != 3 or expected_sha256 is None:
            raise ValueError("Unversioned gearbox weights require a matching profile and external SHA-256 binding")
        if not {"arm", "model_seed", "role_seeds", "actual_calendar_steps", "nominal_calendar_steps"}.issubset(payload):
            raise ValueError("Unversioned gearbox payload lacks retained study metadata")
        if payload["arm"] not in ("episodic", "continuing", "raw", "centered"):
            raise ValueError("Unknown retained gearbox study arm")
        state = payload["state_dict"]
        center = float(payload.get("center_rate_raw_per_calendar_step", 0.))
        offset = center / (1000. * (1. - .999))
        if payload["arm"] == "centered" and (center != 233.01899356282811 or payload.get("value_offset_scaled") != offset):
            raise ValueError("Retained centered gearbox value coordinates differ")
        metadata.update(objective=payload["arm"], value_offset_scaled=offset,
                        model_seed=payload["model_seed"], external_sha256=expected_sha256)
    else:
        raise ValueError("Unsupported checkpoint format; a raw state_dict does not establish compatibility")
    if not isinstance(state, Mapping) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint policy is not a tensor state dictionary")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Checkpoint contains nonfinite policy weights")
    result = Policy(state, action_count, metadata)
    metadata["parameter_sha256"] = parameter_hash(state)
    return result
