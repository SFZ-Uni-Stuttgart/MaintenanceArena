"""Reference CPU PPO recipe for the core and gearbox tasks.

Core cycles stop the learning continuation at renewal. Gearbox renewal remains
continuing. Gamma is per calendar step and lambda is per decision in both cases.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional


PROFILES = ("paper-core-continuous-v1", "paper-core-discrete-v1", "paper-gearbox-v1")
DEFAULTS = {"learning_rate": .0003, "entropy_coefficient": .001,
            "rollout_calendar_steps": 1024, "batch_size": 128, "epochs": 10,
            "gamma": .999, "gae_lambda": .95, "advantage_epsilon": 1e-8,
            "clip_range": .2, "value_coefficient": .5, "max_gradient_norm": .5,
            "optimizer": {"betas": [.9, .999], "epsilon": 1e-8, "weight_decay": 0.}}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def parameter_hash(state):
    """Tensor hash of a state dict: names, dtype, shape and contiguous CPU bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps({"name": str(name), "dtype": str(value.dtype),
                               "shape": list(value.shape)}, ensure_ascii=True,
                              separators=(",", ":"), sort_keys=True).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = value.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def recipe_config(profile, config=None):
    if profile not in PROFILES:
        raise ValueError(f"No retained custom PPO recipe for {profile!r}")
    values = deepcopy(dict(config or {}))
    aliases = {"entropy": "entropy_coefficient", "clip": "clip_range",
               "rollout_local_calendar_quota": "rollout_calendar_steps",
               "gae_lambda_per_decision": "gae_lambda",
               "advantage_normalization_epsilon": "advantage_epsilon",
               "center_rate_raw_per_calendar_step": "center_rate"}
    for old, new in aliases.items():
        if old in values:
            if new in values and values[new] != values[old]:
                raise ValueError(f"Conflicting recipe keys {old}/{new}")
            values[new] = values.pop(old)
    recipe = deepcopy(DEFAULTS)
    recipe.update(center_rate=233.01899356282811 if profile == "paper-gearbox-v1" else 0.,
                  seed_namespace=608 if profile == "paper-gearbox-v1" else 3)
    # Public examples keep recipe numerics. Runtime allocation is outside this
    # mapping; changing a scientific hyperparameter needs a new profile.
    harmless = {"algorithm", "recipe_id", "evidence_status", "checkpoint", "description"}
    values = {key: value for key, value in values.items() if key not in harmless}
    if set(values) - set(recipe):
        raise ValueError(f"Unknown PPO settings: {sorted(set(values) - set(recipe))}")
    for key, value in values.items():
        if key == "optimizer":
            value = {k: v for k, v in value.items() if k != "name"}
        if value != recipe[key]:
            raise ValueError(f"{profile} fixes {key}={recipe[key]!r}; use a separately specified recipe")
    return recipe


class ActorCritic(nn.Module):
    def __init__(self, action_count=2):
        super().__init__()
        if action_count not in (2, 3):
            raise ValueError("The retained network has two or three actions")
        def body():
            return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.policy_body, self.value_body = body(), body()
        self.action_head, self.value_head = nn.Linear(64, action_count), nn.Linear(64, 1)
        for module in (*self.policy_body, *self.value_body):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.action_head.weight, gain=.01)
        nn.init.zeros_(self.action_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observations):
        return self.action_head(self.policy_body(observations)), self.value_head(self.value_body(observations)).squeeze(-1)


def observation_array(value):
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("Expected one finite four-field observation")
    if np.any(result < np.asarray([0., 0., 0., -1.], dtype=np.float32)) or np.any(result > 1.):
        raise ValueError("Observation is outside the retained four-field bounds")
    return result.copy()


@dataclass(frozen=True)
class GAEComputation:
    deltas: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    bootstrap_discounts: np.ndarray


def duration_aware_gae(*, macro_rewards, values, next_values, durations,
                       bootstrap_masks, trace_masks, gamma, gae_lambda_per_decision):
    rewards, current, successor = (np.asarray(x, dtype=np.float64) for x in (macro_rewards, values, next_values))
    lengths = np.asarray(durations)
    boot, trace = (np.asarray(x, dtype=np.float64) for x in (bootstrap_masks, trace_masks))
    if rewards.ndim != 1 or rewards.size == 0 or any(x.shape != rewards.shape for x in (current, successor, lengths, boot, trace)):
        raise ValueError("GAE arrays must be equally sized nonempty vectors")
    if not all(np.isfinite(x).all() for x in (rewards, current, successor, lengths, boot, trace)):
        raise ValueError("GAE inputs must be finite")
    if np.any(lengths < 1) or np.any(lengths != np.floor(lengths)):
        raise ValueError("GAE durations must be positive integers")
    if not all(np.logical_or(x == 0, x == 1).all() for x in (boot, trace)):
        raise ValueError("GAE masks must be binary")
    if not 0 < gamma <= 1 or not 0 <= gae_lambda_per_decision <= 1:
        raise ValueError("Invalid gamma/lambda")
    discounts = np.power(gamma, lengths.astype(np.int64), dtype=np.float64)
    deltas = rewards + discounts * boot * successor - current
    advantages = np.empty_like(deltas)
    next_advantage = 0.
    for index in range(rewards.size - 1, -1, -1):
        next_advantage = float(deltas[index] + discounts[index] * gae_lambda_per_decision * trace[index] * next_advantage)
        advantages[index] = next_advantage
    returns = advantages + current
    if not all(np.isfinite(x).all() for x in (deltas, advantages, returns)):
        raise ArithmeticError("Nonfinite GAE result")
    return GAEComputation(deltas, advantages, returns, discounts)


@dataclass(frozen=True)
class RolloutEntry:
    observation: np.ndarray
    action: int
    old_log_probability: float
    value: float
    discounted_macro_reward: float
    next_observation: np.ndarray
    next_value: float
    duration_calendar_steps: int
    bootstrap_mask: float
    trace_mask: float
    raw_profit_sum: float
    calendar_step_start: int
    calendar_step_end: int


class PPOAgent:
    """Deterministic initialization, action sampling, minibatch order and update math of the recipe."""
    def __init__(self, profile, seed, config=None, *, role_seeds=None):
        self.profile = profile
        self.config = recipe_config(profile, config)
        self.action_count = 3 if profile == "paper-gearbox-v1" else 2
        self.seed = int(seed)
        if isinstance(seed, bool) or self.seed != seed or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.roles = {name: int(np.random.SeedSequence([self.seed, self.config["seed_namespace"], code]).generate_state(1, dtype=np.uint32)[0])
                      for name, code in (("network", 1), ("environment", 2), ("action", 3), ("shuffle", 5))}
        if role_seeds is not None:
            if set(role_seeds) != set(self.roles):
                raise ValueError("Role seed keys must be network/environment/action/shuffle")
            self.roles = {key: int(value) for key, value in role_seeds.items()}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.roles["network"])
            self.network = ActorCritic(self.action_count).cpu()
        optimizer = self.config["optimizer"]
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"],
                                          betas=tuple(optimizer["betas"]), eps=optimizer["epsilon"],
                                          weight_decay=optimizer["weight_decay"])
        self.action_rng = np.random.Generator(np.random.PCG64(self.roles["action"]))
        self.shuffle_rng = np.random.Generator(np.random.PCG64(self.roles["shuffle"]))
        self.optimizer_updates = self.rollout_updates = self.gradient_sample_presentations = 0

    def batch_output(self, observations):
        array = np.ascontiguousarray(np.asarray(observations, dtype=np.float32))
        if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
            raise ValueError("Expected finite shape-(n,4) policy inputs")
        with torch.no_grad():
            logits, values = self.network(torch.from_numpy(array))
            probabilities = torch.softmax(logits, dim=-1)
        return tuple(x.detach().cpu().numpy().astype(np.float32, copy=True) for x in (logits, probabilities, values))

    def probabilities(self, observation):
        return self.batch_output(observation_array(observation).reshape(1, -1))[1][0]

    def predict(self, observation):
        logits, probabilities, _ = self.batch_output(observation_array(observation).reshape(1, -1))
        return int(np.argmax(probabilities[0] if self.action_count == 2 else logits[0]))

    deterministic_action = predict

    def value(self, observation):
        return float(self.batch_output(observation_array(observation).reshape(1, -1))[2][0])

    def sample_action(self, observation):
        logits, probabilities, values = self.batch_output(observation_array(observation).reshape(1, -1))
        uniform = float(self.action_rng.random())
        if self.action_count == 2:
            action = 0 if uniform < float(probabilities[0, 0]) else 1
            log = math.log(max(float(probabilities[0, action]), float(np.finfo(np.float32).tiny)))
        else:
            cumulative = np.cumsum(probabilities[0].astype(np.float64), dtype=np.float64)
            cumulative[-1] = 1.
            action = min(int(np.searchsorted(cumulative, uniform, side="right")), 2)
            log = float(torch.log_softmax(torch.from_numpy(logits[0]), dim=-1).numpy()[action])
        return action, log, float(values[0])

    def update(self, entries):
        rows = tuple(entries)
        if not rows or rows[-1].trace_mask != 0:
            raise ValueError("A nonempty rollout must close its final trace")
        for left, right in zip(rows, rows[1:]):
            if left.calendar_step_end != right.calendar_step_start:
                raise ValueError("Rollout calendar rows are not contiguous")
            if left.trace_mask == 1 and left.next_observation.tobytes() != right.observation.tobytes():
                raise ValueError("Linked successor observations differ")
        cfg = self.config
        self.network.train()
        gae = duration_aware_gae(macro_rewards=[x.discounted_macro_reward for x in rows],
                               values=[x.value for x in rows], next_values=[x.next_value for x in rows],
                               durations=[x.duration_calendar_steps for x in rows],
                               bootstrap_masks=[x.bootstrap_mask for x in rows], trace_masks=[x.trace_mask for x in rows],
                               gamma=cfg["gamma"], gae_lambda_per_decision=cfg["gae_lambda"])
        mean, std = float(np.mean(gae.advantages)), float(np.std(gae.advantages))
        normalized = (gae.advantages - mean) / (std + cfg["advantage_epsilon"])
        observations = torch.from_numpy(np.stack([x.observation for x in rows]).astype(np.float32, copy=False))
        actions = torch.as_tensor([x.action for x in rows], dtype=torch.int64)
        old_logs = torch.as_tensor([x.old_log_probability for x in rows], dtype=torch.float32)
        advantages = torch.from_numpy(normalized.astype(np.float32, copy=False))
        returns = torch.from_numpy(gae.returns.astype(np.float32, copy=False))
        with torch.no_grad():
            initial, _ = self.network(observations)
            initial_logs = torch.log_softmax(initial, dim=-1).gather(1, actions[:, None]).squeeze(1)
            ratio_error = float(torch.max(torch.abs(torch.exp(initial_logs - old_logs) - 1.)).item())
        if ratio_error > 1e-5:
            raise ArithmeticError("Stored PPO logs differ from current policy")
        count = len(rows)
        losses, value_losses, entropies, norms = [], [], [], []
        for _ in range(cfg["epochs"]):
            permutation = np.asarray(self.shuffle_rng.permutation(count), dtype=np.int64)
            for start in range(0, count, cfg["batch_size"]):
                indices = torch.from_numpy(permutation[start:start + cfg["batch_size"]])
                logits, values = self.network(observations[indices])
                distribution = torch.distributions.Categorical(logits=logits)
                ratios = torch.exp(distribution.log_prob(actions[indices]) - old_logs[indices])
                unclipped = ratios * advantages[indices]
                clipped = torch.clamp(ratios, 1. - cfg["clip_range"], 1. + cfg["clip_range"]) * advantages[indices]
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                value_loss = functional.mse_loss(values, returns[indices])
                entropy = torch.mean(distribution.entropy())
                total = policy_loss + cfg["value_coefficient"] * value_loss - cfg["entropy_coefficient"] * entropy
                if not torch.isfinite(total).item():
                    raise ArithmeticError("Nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg["max_gradient_norm"], error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer_updates += 1
                self.gradient_sample_presentations += int(indices.numel())
                losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                entropies.append(float(entropy.detach().item()))
                norms.append(float(norm.detach().item()))
        self.rollout_updates += 1
        return {"rollout_update_ordinal": self.rollout_updates, "sample_count": count,
                "calendar_exposure": sum(x.duration_calendar_steps for x in rows),
                "optimizer_updates_cumulative": self.optimizer_updates,
                "gradient_sample_presentations_this_rollout": count * cfg["epochs"],
                "gradient_sample_presentations_cumulative": self.gradient_sample_presentations,
                "advantage_mean": mean, "advantage_std": std,
                "policy_loss_mean": statistics.fmean(losses), "value_loss_mean": statistics.fmean(value_losses),
                "entropy_mean": statistics.fmean(entropies), "gradient_norm_before_clip_mean": statistics.fmean(norms),
                "pre_update_ratio_max_abs_error": ratio_error}

    def policy_state(self):
        return {key: value.detach().cpu().contiguous().clone() for key, value in self.network.state_dict().items()}

    def snapshot(self):
        return {"policy_state": self.policy_state(), "optimizer_state": deepcopy(self.optimizer.state_dict()),
                "action_rng": deepcopy(self.action_rng.bit_generator.state), "shuffle_rng": deepcopy(self.shuffle_rng.bit_generator.state),
                "roles": deepcopy(self.roles), "seed": self.seed, "optimizer_updates": self.optimizer_updates,
                "rollout_updates": self.rollout_updates, "gradient_sample_presentations": self.gradient_sample_presentations}

    def restore(self, state):
        if state["roles"] != self.roles or state["seed"] != self.seed:
            raise ValueError("Checkpoint RNG roles differ")
        self.network.load_state_dict(state["policy_state"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.action_rng.bit_generator.state = deepcopy(state["action_rng"])
        self.shuffle_rng.bit_generator.state = deepcopy(state["shuffle_rng"])
        for name in ("optimizer_updates", "rollout_updates", "gradient_sample_presentations"):
            setattr(self, name, int(state[name]))


def train(env_factory, *, profile, seed, calendar_steps, output, resume=None, config=None,
          initialize=None):
    """Train the retained recipe; native resumes use a final cumulative target.

    A checkpoint is emitted after every complete update. No buffer is left
    partially trained, and no macro is shortened to meet a time quota.
    The summary's ``checkpoint`` is an absolute path; ``checkpoint_relative``
    names the same file relative to ``output``, for records that are shared.
    """
    from .checkpoints import load_native, load_policy, save_native
    if resume is not None and initialize is not None:
        raise ValueError("resume and initialize are mutually exclusive")
    if isinstance(calendar_steps, bool) or not isinstance(calendar_steps, int) or calendar_steps < 1:
        raise ValueError("calendar_steps must be a positive integer")
    started = time.perf_counter()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob("checkpoint_*.pt")) or (directory / "final.pt").exists() or (directory / "training.json").exists():
        raise FileExistsError("Training outputs already exist; use a new output directory")
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    old_torch_rng = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    wrapped_env = None
    try:
        wrapped_env = env_factory()
        env = wrapped_env.unwrapped
        agent = PPOAgent(profile, seed, config)
        if env.action_space.n != agent.action_count or env.observation_space.shape != (4,):
            raise ValueError("Environment geometry does not match selected PPO profile")
        initialization = None
        if initialize is not None:
            initial_policy = load_policy(initialize, profile=profile, config=env.config)
            target_offset = agent.config["center_rate"] / (1000. * (1. - agent.config["gamma"]))
            source_offset = float(initial_policy.metadata.get("value_offset_scaled", 0.))
            if source_offset != target_offset:
                raise ValueError("Initialization critic centering differs from the target recipe")
            agent.network.load_state_dict(initial_policy.network.state_dict(), strict=True)
            initialization = {"source": str(Path(initialize).resolve()),
                              "parameter_sha256": parameter_hash(agent.policy_state()),
                              "source_metadata": deepcopy(initial_policy.metadata),
                              "optimizer": "fresh", "random_streams": "fresh from requested seed",
                              "collector_and_source": "fresh", "is_exact_resume": False}
        total = decisions = renewals = failures = 0
        raw_profit = 0.
        updates = []
        if resume is None:
            training_source = {"kind": "exact_finite_gearbox", "environment_seed": agent.roles["environment"],
                               "opportunity_stream": "PCG64"}
            if agent.action_count == 2:
                training_source = env.configure_training(seed=agent.roles["environment"], profile=profile)
            observation, _ = env.reset(seed=agent.roles["environment"])
        else:
            payload = load_native(resume, profile=profile, env_config=env.config, recipe=agent.config)
            import platform
            if (payload["runtime"]["python"] != platform.python_version()
                    or payload["runtime"]["numpy"] != str(np.__version__)
                    or payload["runtime"]["torch"] != str(torch.__version__)):
                raise ValueError("Exact resume requires the checkpoint's Python, NumPy and Torch versions")
            if payload["learner"]["seed"] != seed:
                raise ValueError("Resume seed conflicts with checkpoint")
            if payload["collector"]["calendar_steps"] >= calendar_steps:
                raise ValueError("Resume target must exceed checkpoint exposure")
            agent.restore(payload["learner"])
            observation = env.restore(payload["environment"])
            collector = payload["collector"]
            training_source = deepcopy(collector["training_source"])
            initialization = deepcopy(collector.get("initialization"))
            total, decisions, renewals, failures = (collector[x] for x in ("calendar_steps", "decisions", "renewals", "failures"))
            raw_profit, updates = collector["raw_profit"], deepcopy(collector["updates"])
            torch.set_rng_state(payload["torch_rng_state"])
        initial_hash = parameter_hash(agent.policy_state())
        rows, local = [], 0
        while total < calendar_steps:
            action, log, value = agent.sample_action(observation)
            successor, reward, terminated, truncated, info = env.step(action)
            duration = int(info["duration_calendar_steps"])
            if duration < 1 or duration > 19 or info["gamma_pow_duration"] != agent.config["gamma"] ** duration:
                raise ArithmeticError("Environment calendar-discount contract differs")
            next_total = total + duration
            local += duration
            close = local >= agent.config["rollout_calendar_steps"] or next_total >= calendar_steps or truncated
            # A time limit bootstraps the returned actual successor. Only an
            # episodic Core renewal removes the continuation value.
            bootstrap = float(not terminated)
            trace = float(not (terminated or truncated or close))
            center = agent.config["center_rate"] / (1000. * (1 - agent.config["gamma"])) * (1 - agent.config["gamma"] ** duration)
            rows.append(RolloutEntry(observation_array(observation), action, log, value,
                                     float(reward) - center, observation_array(successor),
                                     agent.value(successor) * bootstrap, duration, bootstrap, trace,
                                     float(info["raw_profit_sum"]), total, next_total))
            total = next_total
            decisions += 1
            renewals += int(info.get("renewal", terminated))
            failures += int(info.get("failures", 0))
            raw_profit += float(info["raw_profit_sum"])
            observation = successor
            if terminated or truncated:
                observation, _ = env.reset()
            if close:
                diagnostics = agent.update(rows)
                diagnostics["calendar_steps"] = total
                updates.append(diagnostics)
                rows, local = [], 0
                collector = {"calendar_steps": total, "decisions": decisions, "renewals": renewals,
                             "failures": failures, "raw_profit": raw_profit, "updates": updates,
                             "training_source": training_source, "initialization": initialization}
                save_native(directory / f"checkpoint_{total}.pt", agent=agent, env=env, collector=collector)
        if rows or not 0 <= total - calendar_steps <= 18:
            raise ArithmeticError("Training stopped outside a complete update/macro")
        final = directory / "final.pt"
        save_native(final, agent=agent, env=env, collector=collector)
        summary = {"schema_version": "maintenance_arena_training_v1", "profile": profile,
                   "seed": seed, "nominal_calendar_steps": calendar_steps, "actual_calendar_steps": total,
                   "calendar_overshoot": total - calendar_steps, "decisions": decisions,
                   "renewals": renewals, "training_failures": failures, "training_raw_profit": raw_profit,
                   "optimizer_updates": agent.optimizer_updates, "rollout_updates": agent.rollout_updates,
                   "gradient_sample_presentations": agent.gradient_sample_presentations,
                   "initial_parameter_sha256": initial_hash, "final_parameter_sha256": parameter_hash(agent.policy_state()),
                   "recipe": agent.config, "recipe_sha256": canonical_hash(agent.config),
                   "training_source": training_source,
                   "initialization": initialization,
                   "checkpoint": str(final.resolve()), "checkpoint_relative": final.relative_to(directory).as_posix(),
                   "resume_source": str(Path(resume).resolve()) if resume else None,
                   "runtime_seconds": time.perf_counter() - started,
                   "evidence_status": "software training run; not a confirmation of published results"}
        with (directory / "training.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
        return summary
    finally:
        try:
            if wrapped_env is not None:
                wrapped_env.close()
        finally:
            torch.set_rng_state(old_torch_rng)
            torch.set_num_threads(old_threads)
            torch.use_deterministic_algorithms(old_deterministic)
