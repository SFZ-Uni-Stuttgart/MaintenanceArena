"""Masked PPO for the unchanged resource-maintenance simulator.

Training randomizes known economic and logistics contexts at episode resets.
Validation uses the existing cashflow evaluator, exact public observations,
deterministic masked actions and frozen training normalization statistics.
Neither context names nor lifecycle truth become network inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from functools import wraps
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from ..resource_evaluation import run_episode
from ..resource import ResourceMaintenanceConfig, ResourceMaintenanceEnv
from ..resource_lifetime import ResourceLifetimeEnv


def _preserve_caller_random_state(function):
    """SB3 setup owns global RNGs temporarily, including when loading a model."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        try:
            return function(*args, **kwargs)
        finally:
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])
            torch.set_num_threads(threads)
    return wrapped


def _check_profile_environment(profile, env_class):
    expected = {"paper-resource-v1": ResourceMaintenanceEnv,
                "paper-resource-lifetime-v1": ResourceLifetimeEnv}
    if profile not in expected or env_class is not expected[profile]:
        raise ValueError("resource profile does not match the 55/58-field environment task")


DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
}


def _configuration(value: ResourceMaintenanceConfig | Mapping[str, Any]) -> dict[str, Any]:
    return asdict(value) if isinstance(value, ResourceMaintenanceConfig) else dict(value)


class ContextSamplingEnv(gym.Env[np.ndarray, int]):
    """Uniform episode-level contexts with separate context and fleet RNGs.

    Explicit ``reset(seed=...)`` reproduces both streams. Automatic resets
    advance each stream once irrespective of decisions or delivery draws.
    Fleet seeds lie in [1_000_000, 2**32), disjoint from the experiment's
    reserved low-range calibration and evaluation seeds.
    Spaces are invariant: context changes may not resize assets or inventory.
    The optional context label in info is for audit, never an observation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base: ResourceMaintenanceConfig | Mapping[str, Any],
        contexts: Mapping[str, Mapping[str, Any]],
        horizon: int = 1440,
        env_class=ResourceMaintenanceEnv,
    ):
        super().__init__()
        if not contexts:
            raise ValueError("at least one training context is required")
        self.context_names = tuple(contexts)
        self.env_class = env_class
        self._configurations = tuple(
            ResourceMaintenanceConfig.from_mapping(_configuration(base) | dict(overrides) | {"horizon": horizon})
            for overrides in contexts.values()
        )
        self.env = self.env_class(self._configurations[0])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.observation_names = self.env.observation_names
        self.render_mode = None
        for cfg in self._configurations[1:]:
            candidate = self.env_class(cfg)
            try:
                if (candidate.observation_space != self.observation_space
                        or candidate.action_space != self.action_space
                        or candidate.observation_names != self.observation_names):
                    raise ValueError("training contexts must preserve observation and action spaces")
            finally:
                candidate.close()
        self._context_rng: np.random.Generator | None = None
        self._fleet_rng: np.random.Generator | None = None
        self.current_context: str | None = None
        self.reset_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if options:
            raise ValueError("training context is sampled, not selected by reset options")
        super().reset(seed=seed)
        if seed is not None or self._context_rng is None:
            key = seed if seed is not None else int(self.np_random.integers(0, 2**32))
            context_key, fleet_key = np.random.SeedSequence(key).spawn(2)
            self._context_rng = np.random.default_rng(context_key)
            self._fleet_rng = np.random.default_rng(fleet_key)
        index = int(self._context_rng.integers(len(self._configurations)))
        fleet_seed = int(self._fleet_rng.integers(1_000_000, 2**32))
        self.env.close()
        self.env = self.env_class(self._configurations[index])
        self.current_context = self.context_names[index]
        self.reset_count += 1
        observation, info = self.env.reset(seed=fleet_seed)
        info["training_context"] = self.current_context
        info["training_fleet_seed"] = fleet_seed
        return observation, info

    def step(self, action: int):
        return self.env.step(action)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    def policy_state(self) -> dict[str, Any]:
        return self.env.policy_state()

    def close(self) -> None:
        self.env.close()

    def snapshot(self):
        if self.current_context is None:
            raise RuntimeError("training environment has not been reset")
        return copy.deepcopy({"context": self.current_context,
            "context_rng": self._context_rng.bit_generator.state,
            "fleet_rng": self._fleet_rng.bit_generator.state,
            "rng": self.np_random.bit_generator.state,
            "reset_count": self.reset_count, "environment": self.env.snapshot()})

    def restore(self, state):
        index = self.context_names.index(state["context"])
        self.current_context = state["context"]
        self.env.close()
        self.env = self.env_class(self._configurations[index])
        self.env.restore(state["environment"])
        self._context_rng, self._fleet_rng = np.random.default_rng(), np.random.default_rng()
        self._context_rng.bit_generator.state = copy.deepcopy(state["context_rng"])
        self._fleet_rng.bit_generator.state = copy.deepcopy(state["fleet_rng"])
        self.np_random.bit_generator.state = copy.deepcopy(state["rng"])
        self.reset_count = state["reset_count"]


def make_training_env(
    base: ResourceMaintenanceConfig | Mapping[str, Any],
    contexts: Mapping[str, Mapping[str, Any]],
    seed: int,
    n_envs: int = 8,
    horizon: int = 1440,
    gamma: float = 0.995,
    env_class=ResourceMaintenanceEnv,
) -> VecNormalize:
    """Build eight synchronous masked environments and training-only scaling."""
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs < 1:
        raise ValueError("n_envs must be a positive integer")
    torch.set_num_threads(1)
    vector = DummyVecEnv([
        lambda: Monitor(ContextSamplingEnv(base, contexts, horizon=horizon, env_class=env_class))
        for _ in range(n_envs)
    ])
    vector.seed(int(seed))
    return VecNormalize(
        vector, training=True, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=float(gamma), epsilon=1e-8,
    )


def create_model(
    env: VecNormalize,
    seed: int,
    hyperparams: Mapping[str, Any] | None = None,
) -> MaskablePPO:
    """The single fixed recipe; no hidden hyperparameter search."""
    torch.set_num_threads(1)
    parameters = dict(DEFAULT_HYPERPARAMETERS)
    parameters.update(dict(hyperparams or {}))
    if float(parameters["gamma"]) != env.gamma:
        raise ValueError("PPO and reward normalization must use the same gamma")
    return MaskablePPO(
        "MlpPolicy", env, seed=int(seed), device="cpu", verbose=0,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                       "activation_fn": torch.nn.Tanh},
        **parameters,
    )


def public_observation(env: ResourceMaintenanceEnv) -> np.ndarray:
    """Reconstruct the returned vector using only its public field names.

    This deliberately works with a proxy exposing only ``policy_state`` and
    ``observation_names``. No private observation helper or simulator state is
    required by the shared evaluator's action callback.
    """
    state = env.policy_state()
    values = []
    for name in env.observation_names:
        if name.startswith("asset_"):
            asset, field = name.split(".", 1)
            values.append(state["assets"][int(asset.removeprefix("asset_"))][field])
        elif name.startswith("pipeline_"):
            slot, field = name.split(".", 1)
            index = int(slot.removeprefix("pipeline_"))
            occupied = index < len(state["pipeline"])
            values.append(int(occupied) if field == "occupied" else
                          state["pipeline"][index][field] if occupied else 0)
        elif name.startswith("config."):
            values.append(state["config"][name.removeprefix("config.")])
        else:
            values.append(state[name])
    observation = np.asarray(values, dtype=np.float32)
    if not np.isfinite(observation).all():
        raise ValueError("public policy observation contains nonfinite values")
    return observation


@dataclass(frozen=True, slots=True)
class RLPolicyMetadata:
    kind: str = "maskable_ppo"
    model_seed: int | None = None
    deterministic: bool = True


def predict_action(model: MaskablePPO, normalizer: VecNormalize, env: ResourceMaintenanceEnv) -> int:
    """Inference never updates observation or reward running statistics."""
    observation = normalizer.normalize_obs(public_observation(env))
    mask = env.action_masks()
    action, _ = model.predict(observation, action_masks=mask, deterministic=True)
    result = int(np.asarray(action).item())
    if result < 0 or result >= len(mask) or not mask[result]:
        raise AssertionError("masked PPO produced an infeasible action")
    return result


def evaluate_model(
    model: MaskablePPO,
    normalizer: VecNormalize,
    baseconfig: ResourceMaintenanceConfig | Mapping[str, Any],
    envseeds: list[int],
    windows: list[tuple[int, int]],
    *,
    model_seed: int | None = None,
    trace_directory: Path | None = None,
    env_class=ResourceMaintenanceEnv,
) -> list[dict[str, Any]]:
    """Use unchanged economics and uninterrupted evaluation windows.

    Flags are restored after evaluation so a staged training run can resume.
    Evaluation never steps or resets the training vector environment, and
    terminal-value rewards are not substituted for raw economic measurements.
    """
    evaluator = run_episode
    if env_class is ResourceLifetimeEnv:
        from ..resource_lifetime import run_episode as evaluator
    elif env_class is not ResourceMaintenanceEnv:
        raise ValueError("unknown resource evaluation environment")
    flags = normalizer.training, normalizer.norm_reward
    normalizer.training, normalizer.norm_reward = False, False
    metadata = RLPolicyMetadata(model_seed=model_seed if model_seed is not None else model.seed)
    try:
        return [evaluator(
            _configuration(baseconfig), metadata, int(seed), windows,
            trace_path=(Path(trace_directory) / f"fleet_{seed}.jsonl") if trace_directory else None,
            action_fn=lambda env, _: predict_action(model, normalizer, env),
        ) for seed in envseeds]
    finally:
        normalizer.training, normalizer.norm_reward = flags


DEFAULT_CONTEXTS = {
    "nominal": {},
    "slow_supply": {"normal_lead_range": [36, 72], "express_lead_range": [6, 12]},
    "double_part_cost": {"part_cost": 1200.0},
    "continuous_staffing": {"shift_hours": 24},
}
_MONITOR_FIELDS = ("rewards", "needs_reset", "episode_returns", "episode_lengths",
                   "episode_times", "total_steps", "current_reset_info")
_VECTOR_FIELDS = ("_seeds", "_options", "reset_infos", "buf_obs", "buf_dones",
                  "buf_rews", "buf_infos", "actions")


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _versions():
    return {"python": ".".join(map(str, sys.version_info[:3])), **{
        package: importlib.metadata.version(package)
        for package in ("numpy", "gymnasium", "torch", "stable-baselines3", "sb3-contrib")}}


def _binding(path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _check_rollout(calendar_steps, n_envs, n_steps):
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs < 1:
        raise ValueError("resource n_envs must be a positive integer")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("resource n_steps must be a positive integer")
    rollout = n_envs * n_steps
    if rollout < 2:
        raise ValueError(f"resource n_envs * n_steps = {n_envs} * {n_steps} must be at least 2")
    if calendar_steps % rollout:
        below = calendar_steps // rollout * rollout
        examples = " or ".join(str(value) for value in (below, below + rollout) if value > 0)
        raise ValueError(f"calendar_steps must be an exact multiple of n_envs * n_steps = {n_envs} * {n_steps} "
                         f"= {rollout}; got {calendar_steps}, use for example {examples}")
    return rollout


def check_calendar_budget(calendar_steps, config=None):
    """Require whole PPO rollouts of ``n_envs * n_steps`` calendar steps.

    ``config`` is a profile's ``training`` section. With the default 8
    environments of 256 steps each, a budget must be a multiple of 2048.
    Returns the rollout size in calendar steps.
    """
    if isinstance(calendar_steps, bool) or not isinstance(calendar_steps, int) or calendar_steps < 1:
        raise ValueError("calendar_steps must be a positive cumulative integer budget")
    supplied = dict(config or {})
    n_steps = dict(DEFAULT_HYPERPARAMETERS, **supplied.get("hyperparameters", {}))["n_steps"]
    return _check_rollout(calendar_steps, supplied.get("n_envs", 8), n_steps)


def _write_json(path, payload):
    from ..records import write_json
    write_json(path, payload, exclusive=True)


def _manifest_path(path):
    path = Path(path)
    return path / "checkpoint.json" if path.is_dir() else path


def _read_manifest(path, expected_kind=None):
    manifest_path = _manifest_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") not in ("resource_native", "resource_legacy"):
        raise ValueError("resource policy requires a paired resource checkpoint manifest")
    if expected_kind and manifest["kind"] != expected_kind:
        raise ValueError("legacy weights cannot resume training; a native resource checkpoint is required")
    required = {"model", "normalizer"} | ({"state"} if expected_kind == "resource_native" else set())
    if not required <= manifest.get("files", {}).keys():
        raise ValueError("resource checkpoint is missing its model, matching normalizer or resume state")
    paths = {}
    for name in required:
        item = manifest["files"][name]
        relative = Path(item["path"])
        resolved = (manifest_path.parent / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("checkpoint file must have a contained relative path")
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"resource checkpoint {name} hash mismatch")
        paths[name] = resolved
    return manifest, paths


def _vector_snapshot(normalizer):
    vector = normalizer.venv
    return {"discounted_reward_returns": normalizer.returns.copy(),
            "vector": {field: copy.deepcopy(getattr(vector, field)) for field in _VECTOR_FIELDS
                       if hasattr(vector, field)},
            "workers": [{"monitor": {field: copy.deepcopy(getattr(worker, field)) for field in _MONITOR_FIELDS},
                         "monitor_elapsed": time.time() - worker.t_start,
                         "environment": worker.env.snapshot()} for worker in vector.envs]}


def _restore_vector(normalizer, snapshot):
    vector = normalizer.venv
    normalizer.returns = np.asarray(snapshot["discounted_reward_returns"]).copy()
    if len(vector.envs) != len(snapshot["workers"]):
        raise ValueError("native resource vector size changed")
    for field, value in snapshot["vector"].items():
        setattr(vector, field, copy.deepcopy(value))
    for worker, stored in zip(vector.envs, snapshot["workers"], strict=True):
        worker.env.restore(stored["environment"])
        for field, value in stored["monitor"].items():
            setattr(worker, field, copy.deepcopy(value))
        worker.t_start = time.time() - stored["monitor_elapsed"]


@_preserve_caller_random_state
def train(env_factory, *, profile, seed, calendar_steps, output, resume=None, config=None, initialize=None):
    """Run the fixed masked-PPO recipe; native continuation is at full updates.

    A lifetime-environment run is a new training run, never a replay of an
    earlier study. Local native
    checkpoints contain trusted Python training state and must not be loaded
    from untrusted third parties.

    Explicit ``initialize`` transfers actor/critic weights together with their
    paired observation/reward normalization statistics. Optimizer state,
    exposure, pending discounted returns and all environment/global RNG streams
    start fresh from the new run seed. This is a recorded warm-start method,
    not inference, exact resumption or reproduction of a published fit.

    ``calendar_steps`` must be a multiple of ``n_envs * n_steps`` (see
    ``check_calendar_budget``). The result's ``checkpoint`` is the manifest path;
    ``checkpoint_relative`` names it relative to ``output``, for shared records.
    """
    started = time.perf_counter()
    if resume is not None and initialize is not None:
        raise ValueError("resume and initialize are mutually exclusive")
    if isinstance(calendar_steps, bool) or not isinstance(calendar_steps, int) or calendar_steps < 1:
        raise ValueError("calendar_steps must be a positive cumulative integer budget")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("resource seed must be an integer in [0, 2**32)")
    supplied = copy.deepcopy(dict(config or {}))
    allowed = {"n_envs", "horizon", "hyperparameters", "contexts", "total_steps", "checkpoint_interval"}
    if set(supplied) - allowed:
        raise ValueError(f"unknown resource training settings: {sorted(set(supplied) - allowed)}")
    sample = env_factory()
    try:
        base = asdict(sample.unwrapped.config)
        env_class = type(sample.unwrapped)
        if env_class not in (ResourceMaintenanceEnv, ResourceLifetimeEnv):
            raise ValueError("resource PPO requires ResourceEnv or ResourceLifetimeEnv")
        _check_profile_environment(profile, env_class)
        observation_names = list(sample.unwrapped.observation_names)
    finally:
        sample.close()
    n_envs = supplied.get("n_envs", 8)
    horizon = supplied.get("horizon", 1440)
    parameters = dict(DEFAULT_HYPERPARAMETERS, **supplied.get("hyperparameters", {}))
    contexts = supplied.get("contexts", DEFAULT_CONTEXTS)
    for name in ("n_steps", "batch_size", "n_epochs"):
        if isinstance(parameters[name], bool) or not isinstance(parameters[name], int) or parameters[name] < 1:
            raise ValueError(f"resource {name} must be a positive integer")
    if float(parameters["gamma"]) != 0.995:
        raise ValueError("resource profile fixes calendar gamma at 0.995")
    # Exact continuation requires constant schedules: arbitrary callables are
    # not an interchangeable rendering of the recipe's learning-rate contract.
    _json_value(parameters)
    _check_rollout(calendar_steps, n_envs, parameters["n_steps"])
    contract = _json_value({"profile": profile, "seed": int(seed), "environment_class": env_class.__name__,
        "environment": base, "observation_names": observation_names, "n_envs": n_envs,
        "horizon": horizon, "contexts": contexts, "hyperparameters": parameters})
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    targets = {name: directory / name for name in ("model.zip", "normalizer.pkl", "training_state.pt", "checkpoint.json")}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("resource output already contains checkpoint files")
    torch.set_num_threads(1)
    normalizer = make_training_env(base, contexts, int(seed), n_envs=n_envs,
        horizon=horizon, gamma=0.995, env_class=env_class)
    initialization = None
    try:
        if resume is None:
            model = create_model(normalizer, int(seed), parameters)
            initial_steps = 0
            if initialize is not None:
                initialization = _initialize_from_pair(model, normalizer, initialize, profile,
                                                      env_class, base)
        else:
            manifest, paths = _read_manifest(resume, "resource_native")
            if manifest.get("contract") != contract:
                raise ValueError("resume cannot change profile, seed, environment or training configuration")
            if manifest.get("runtime") != _versions():
                raise ValueError("exact native resume requires the saved Python and library versions")
            initialization = manifest.get("initialization")
            state = torch.load(paths["state"], map_location="cpu", weights_only=False)
            if state.get("schema") != "maintenance_arena_resource_resume_v1":
                raise ValueError("unsupported resource resume state")
            normalizer = VecNormalize.load(str(paths["normalizer"]), normalizer.venv)
            if not normalizer.training or not normalizer.norm_reward:
                raise ValueError("native training normalizer flags changed")
            model = MaskablePPO.load(str(paths["model"]), env=normalizer, device="cpu", force_reset=False)
            # Model loading reseeds its vector environment during setup.
            # Restore pending auto-reset seeds and worker state afterwards.
            _restore_vector(normalizer, state["environment"])
            if model._last_obs is None or not np.array_equal(model._last_obs, state["last_observation"]):
                raise ValueError("native model lost its pending last observation")
            initial_steps = model.num_timesteps
            if initial_steps != manifest["calendar_steps"]:
                raise ValueError("native model exposure differs from checkpoint manifest")
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
        if calendar_steps <= initial_steps:
            raise ValueError("resume budget must exceed the saved cumulative calendar steps")
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(folder=None, format_strings=[]))
        model.learn(total_timesteps=calendar_steps - initial_steps, reset_num_timesteps=resume is None,
                    progress_bar=False, log_interval=None)
        if model.num_timesteps != calendar_steps:
            raise AssertionError("resource PPO did not finish its exact update-boundary budget")
        state = {"schema": "maintenance_arena_resource_resume_v1",
                 "environment": _vector_snapshot(normalizer),
                 "last_observation": copy.deepcopy(model._last_obs),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state()}
        model.save(str(targets["model.zip"]))
        normalizer.save(str(targets["normalizer.pkl"]))
        torch.save(state, targets["training_state.pt"])
        manifest = {"schema_version": 1, "kind": "resource_native", "profile": profile,
                    "model_seed": int(seed), "calendar_steps": calendar_steps,
                    "contract": contract, "observation_names": observation_names,
                    "runtime": _versions(), "files": {"model": _binding(targets["model.zip"]),
                    "normalizer": _binding(targets["normalizer.pkl"]), "state": _binding(targets["training_state.pt"])},
                    "historical_lifetime_rl_result": False,
                    "initialization": initialization,
                    "resume_boundary": "complete PPO update, no pending transition"}
        _write_json(targets["checkpoint.json"], manifest)
        return {"status": "complete", "algorithm": "maskable_ppo", "profile": profile,
                "seed": int(seed), "calendar_steps": calendar_steps,
                "new_calendar_steps": calendar_steps - initial_steps,
                "optimizer_updates": max((int(value["step"].item()) for value in model.policy.optimizer.state.values()
                                          if "step" in value), default=0),
                "ppo_epochs": int(model._n_updates),
                "initialization": initialization,
                "checkpoint": str(targets["checkpoint.json"]),
                "checkpoint_relative": targets["checkpoint.json"].relative_to(directory).as_posix(),
                "elapsed_seconds": time.perf_counter() - started,
                "historical_reproduction": False}
    finally:
        normalizer.close()


def _initialize_from_pair(model, normalizer, path, profile, env_class, base):
    """Keep weight coordinates coherent while resetting the entire learner."""
    manifest, paths = _read_manifest(path)
    if manifest.get("profile") != profile:
        raise ValueError("initializer profile differs from the new resource training task")
    fresh_random = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    observation_env = env_class(base)
    policy = None
    try:
        policy = load_policy(path, profile=profile, env=observation_env)
        for key in ("gamma", "epsilon", "clip_obs", "clip_reward", "norm_obs"):
            if getattr(policy.normalizer, key) != getattr(normalizer, key):
                raise ValueError(f"initializer normalization contract differs: {key}")
        model.policy.load_state_dict(policy.model.policy.state_dict(), strict=True)
        if model.policy.optimizer.state:
            raise AssertionError("initialization must precede the first optimizer update")
        normalizer.obs_rms = copy.deepcopy(policy.normalizer.obs_rms)
        normalizer.ret_rms = copy.deepcopy(policy.normalizer.ret_rms)
        normalizer.returns = np.zeros(normalizer.num_envs)
        normalizer.training = normalizer.norm_reward = True
        return {"mode": "weights_and_paired_normalization_fresh_training_state",
                "source_kind": manifest["kind"], "source_profile": manifest["profile"],
                "source_model_seed": manifest.get("model_seed"),
                "manifest_sha256": hashlib.sha256(_manifest_path(path).read_bytes()).hexdigest(),
                "model_sha256": manifest["files"]["model"]["sha256"],
                "normalizer_sha256": manifest["files"]["normalizer"]["sha256"],
                "preserved": ["actor_critic_weights", "observation_statistics", "reward_statistics"],
                "reset": ["optimizer", "calendar_exposure", "discounted_returns", "environment_states",
                          "global_rngs", "context_fleet_streams", "monitor_state"]}
    finally:
        if policy is not None:
            policy.close()
        observation_env.close()
        random.setstate(fresh_random[0])
        np.random.set_state(fresh_random[1])
        torch.set_rng_state(fresh_random[2])


class ResourcePolicy:
    """Frozen model/normalizer pair; the policy only receives public masks."""

    def __init__(self, model, normalizer, observation_names, env=None):
        self.model, self.normalizer = model, normalizer
        self.observation_names = tuple(observation_names)
        self.env = env
        self.normalizer.training = False
        self.normalizer.norm_reward = False

    def predict(self, observation, *, action_mask=None, deterministic=True):
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (len(self.observation_names),) or not np.isfinite(vector).all():
            raise ValueError("resource policy observation shape/values differ from checkpoint contract")
        if action_mask is None and self.env is not None:
            action_mask = self.env.action_masks()
        if action_mask is None:
            raise ValueError("resource prediction requires the current public action mask")
        mask = np.asarray(action_mask, dtype=bool)
        if mask.shape != (self.model.action_space.n,) or not mask.any():
            raise ValueError("invalid resource action mask")
        scaled = self.normalizer.normalize_obs(vector)
        action, _ = self.model.predict(scaled, action_masks=mask, deterministic=deterministic)
        result = int(np.asarray(action).item())
        if not mask[result]:
            raise AssertionError("masked policy produced an infeasible action")
        return result

    def close(self):
        self.normalizer.close()


@_preserve_caller_random_state
def load_policy(path, *, profile=None, env=None):
    """Load one hash-bound pair; never silently substitute a normalizer."""
    manifest, paths = _read_manifest(path)
    if profile is not None and manifest.get("profile") != profile:
        raise ValueError("resource policy profile mismatch")
    selected_profile = manifest.get("profile")
    expected_class = {"paper-resource-v1": ResourceMaintenanceEnv,
                      "paper-resource-lifetime-v1": ResourceLifetimeEnv}.get(selected_profile)
    if expected_class is None:
        raise ValueError("resource checkpoint has an unsupported profile")
    if env is None:
        if manifest["kind"] == "resource_legacy":
            env = ResourceMaintenanceEnv()
        else:
            cls = ResourceLifetimeEnv if manifest["contract"]["environment_class"] == "ResourceLifetimeEnv" else ResourceMaintenanceEnv
            env = cls(manifest["contract"]["environment"])
        action_env = None
    else:
        action_env = env.unwrapped
        env = env.unwrapped
    names = tuple(manifest.get("observation_names", ResourceMaintenanceEnv().observation_names))
    if tuple(env.observation_names) != names:
        raise ValueError("checkpoint observation order differs from environment (55 vs 58 is not interchangeable)")
    _check_profile_environment(selected_profile, type(env))
    cls, base = type(env), asdict(env.config)
    carrier = DummyVecEnv([lambda: cls(base)])
    try:
        normalizer = VecNormalize.load(str(paths["normalizer"]), carrier)
        if normalizer.observation_space.shape != (len(names),):
            raise ValueError("normalizer shape differs from the observation contract")
        model = MaskablePPO.load(str(paths["model"]), device="cpu")
        if model.observation_space.shape != normalizer.observation_space.shape or model.action_space != carrier.action_space:
            raise ValueError("model and normalizer/action spaces do not match")
        return ResourcePolicy(model, normalizer, names, action_env)
    except Exception:
        carrier.close()
        raise
