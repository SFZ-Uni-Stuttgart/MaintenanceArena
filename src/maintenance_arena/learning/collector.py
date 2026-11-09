"""PPO training on a finite scenario bank, with milestones, initialization and exact resumption.

``BankTrainer`` is the training procedure of the article's synthetic studies. It
drives the core recipe's ``PPOAgent`` (Table A1 settings of the article) on a
``ScenarioBankEnv``:

* Role seeds: ``SeedSequence([20260926, seed, code])`` for the network (code 1),
  the environment and measurement replay (2), action sampling (3) and minibatch
  shuffling (5). The package's ``train`` uses other role seeds and its own bank.
* A rollout collects at least 1,024 calendar steps but stops at the next
  milestone, so ``train_until(m)`` ends at ``m`` (or the few steps beyond it that
  the last committed order takes) and every milestone is an update boundary.
* ``initialize`` loads policy weights only: the optimizer, the random streams and
  the bank start fresh (fine-tuning). ``resume`` continues a saved trainer exactly:
  weights, optimizer, random streams, environment and counters.
* ``input_set`` multiplies the observations by one of the ablation's input masks
  (``maintenance_arena.wrappers.INPUT_SETS``) before the agent sees them.

Checkpoints are written once and never replaced; they load with Torch's
weights-only reader. Load them only from trusted sources.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import platform
from typing import Any, Iterator

import numpy as np
import torch

from ..banks import ScenarioBankEnv
from ..wrappers import INPUT_SETS, input_set as apply_input_set
from .checkpoints import _plain, atomic_write, file_sha256
from .ppo import PPOAgent, RolloutEntry, canonical_hash, parameter_hash

SCHEMA = "maintenance_arena_bank_ppo_checkpoint_v1"
PROFILE = "paper-core-continuous-v1"
# Root of the role seeds of the article's synthetic runs.
ROLE_SEED_ROOT = 20260926
ROLE_CODES = (("network", 1), ("environment", 2), ("action", 3), ("shuffle", 5))
COUNTERS = ("calendar", "decisions", "episodes", "failures", "raw_profit")


def role_seeds(seed: int) -> dict[str, int]:
    """Network, environment, action and shuffle seeds of one training run."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return {name: int(np.random.SeedSequence([ROLE_SEED_ROOT, int(seed), code]).generate_state(1)[0])
            for name, code in ROLE_CODES}


@contextmanager
def single_thread() -> Iterator[None]:
    """Run Torch on one CPU thread, as in the article's runs, and restore the setting afterwards."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _runtime() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": str(np.__version__), "torch": str(torch.__version__)}


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a trainer checkpoint with Torch's weights-only reader and check its bindings."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA:
        raise ValueError("not a bank trainer checkpoint")
    if canonical_hash(payload["recipe"]) != payload["recipe_sha256"]:
        raise ValueError("checkpoint recipe binding failed")
    if parameter_hash(payload["agent"]["policy_state"]) != payload["policy_parameter_sha256"]:
        raise ValueError("checkpoint parameter hash failed")
    return dict(payload)


def policy_agent(source: str | Path | Mapping[str, Any], seed: int | None = None) -> PPOAgent:
    """A ``PPOAgent`` holding the weights of a trainer checkpoint (or a policy state dictionary), for evaluation.

Eight-input weights (both causal estimates, ``learning.dual``) give a ``DualPPOAgent``.
"""
    if isinstance(source, Mapping) and "schema_version" not in source:
        state, seed = source, 0 if seed is None else seed
    else:
        payload = load_checkpoint(source) if not isinstance(source, Mapping) else source
        state, seed = payload["agent"]["policy_state"], payload["seed"] if seed is None else seed
    with single_thread():
        if tuple(state["policy_body.0.weight"].shape) == (64, 8):
            from .dual import DualPPOAgent
            agent = DualPPOAgent(int(seed), role_seeds=role_seeds(int(seed)))
        else:
            agent = PPOAgent(PROFILE, int(seed), role_seeds=role_seeds(int(seed)))
    agent.network.load_state_dict(state, strict=True)
    return agent


class BankTrainer:
    """Collector around the core ``PPOAgent`` on one finite training bank."""

    def __init__(self, bank: Mapping[str, Any], seed: int, *, env_config: Mapping[str, Any] | None = None,
                 input_set: str | None = None, initialize: str | Path | Mapping[str, Any] | None = None,
                 resume: str | Path | None = None):
        if initialize is not None and resume is not None:
            raise ValueError("initialize and resume are mutually exclusive")
        if input_set is not None and input_set not in INPUT_SETS:
            raise ValueError(f"input set must be one of {sorted(INPUT_SETS)}")
        self.seed = int(seed)
        self.roles = role_seeds(self.seed)
        self.input_set = input_set
        with single_thread():
            self.agent = self._make_agent()
        self.env = self._make_env(bank, env_config)
        self.bank_sha256 = bank["sha256"]
        self.calendar = self.decisions = self.episodes = self.failures = 0
        self.raw_profit = 0.
        self.logs: list[dict[str, Any]] = []
        self.needs_reset = False
        self.initialized_from: dict[str, Any] | None = None
        self.observation = self._mask(self.env.reset(seed=self.roles["environment"])[0])
        if initialize is not None:
            self._initialize(initialize)
        if resume is not None:
            self._resume(resume)

    def _make_agent(self) -> PPOAgent:
        return PPOAgent(PROFILE, self.seed, role_seeds=self.roles)

    def _make_env(self, bank: Mapping[str, Any], env_config: Mapping[str, Any] | None) -> ScenarioBankEnv:
        return ScenarioBankEnv(bank, env_config)

    def _load_weights(self, state: Mapping[str, Any]) -> None:
        self.agent.network.load_state_dict(state, strict=True)

    def _mask(self, observation: np.ndarray) -> np.ndarray:
        return observation if self.input_set is None else apply_input_set(observation, self.input_set)

    def _initialize(self, source: str | Path | Mapping[str, Any]) -> None:
        if isinstance(source, Mapping):
            state, origin = source, {"source": "policy state"}
        else:
            payload = load_checkpoint(source)
            if payload["recipe"] != self.agent.config:
                raise ValueError("the initialization checkpoint has another PPO recipe")
            state = payload["agent"]["policy_state"]
            origin = {"source": "trainer checkpoint", "file_sha256": file_sha256(source),
                      "calendar_steps": payload["collector"]["calendar"]}
        self._load_weights(state)
        self.initialized_from = {**origin, "parameter_sha256": parameter_hash(self.agent.policy_state()),
                                 "mode": "weights only; fresh optimizer, random streams and bank"}

    def _resume(self, path: str | Path) -> None:
        payload = load_checkpoint(path)
        expected = {"seed": self.seed, "bank_sha256": self.bank_sha256, "input_set": self.input_set,
                    "recipe": self.agent.config}
        for key, value in expected.items():
            if payload[key] != value:
                raise ValueError(f"the checkpoint's {key} differs from this trainer")
        if payload["runtime"] != _runtime():
            raise ValueError("exact resumption requires the checkpoint's Python, NumPy and Torch versions")
        self.agent.restore(payload["agent"])
        restored = self._mask(self.env.restore(payload["environment"]))
        stored = np.asarray(payload["observation"], dtype=np.float32)
        if restored.tobytes() != stored.tobytes():
            raise ValueError("the stored observation differs from the restored environment")
        self.observation = stored
        collector = payload["collector"]
        for name in COUNTERS:
            setattr(self, name, collector[name])
        self.raw_profit = float(self.raw_profit)
        self.logs = deepcopy(list(collector["logs"]))
        self.needs_reset = bool(collector["needs_reset"])
        self.initialized_from = deepcopy(collector["initialized_from"])

    def train_until(self, milestone: int) -> dict[str, Any]:
        """Train until at least ``milestone`` calendar steps, ending every rollout at the milestone."""
        if isinstance(milestone, bool) or not isinstance(milestone, int) or milestone <= self.calendar:
            raise ValueError("the milestone must be an integer beyond the steps already trained")
        quota = self.agent.config["rollout_calendar_steps"]
        with single_thread():
            while self.calendar < milestone:
                limit = min(milestone, self.calendar + quota)
                entries = []
                while self.calendar < limit:
                    if self.needs_reset:
                        self.observation = self._mask(self.env.reset()[0])
                        self.needs_reset = False
                    old = self.observation.copy()
                    action, log_probability, value = self.agent.sample_action(old)
                    successor, reward, terminal, truncated, info = self.env.step(action)
                    if truncated:
                        raise RuntimeError("training cycles must not be truncated")
                    successor = self._mask(successor)
                    duration = int(info["duration_calendar_steps"])
                    end = self.calendar + duration
                    entries.append(RolloutEntry(
                        observation=old, action=action, old_log_probability=log_probability, value=value,
                        discounted_macro_reward=reward, next_observation=successor.copy(),
                        next_value=0. if terminal else self.agent.value(successor),
                        duration_calendar_steps=duration, bootstrap_mask=float(not terminal),
                        trace_mask=float(not terminal), raw_profit_sum=info["raw_profit_sum"],
                        calendar_step_start=self.calendar, calendar_step_end=end))
                    self.calendar = end
                    self.decisions += 1
                    self.raw_profit += info["raw_profit_sum"]
                    self.observation = successor
                    self.needs_reset = bool(terminal)
                    if terminal:
                        self.episodes += 1
                        self.failures += info["failures"]
                entries[-1] = replace(entries[-1], trace_mask=0.)
                update = self.agent.update(entries)
                self.logs.append({**update, "actual_calendar_steps": self.calendar, "episodes": self.episodes,
                                  "failures": self.failures, "raw_profit": self.raw_profit})
        return {"nominal_calendar_steps": milestone, "actual_calendar_steps": self.calendar,
                "calendar_overshoot": self.calendar - milestone, "decisions": self.decisions,
                "completed_cycles": self.episodes, "failures_during_training": self.failures,
                "raw_profit_during_training": self.raw_profit, "rollout_updates": self.agent.rollout_updates,
                "optimizer_minibatches": self.agent.optimizer_updates}

    def save(self, path: str | Path) -> dict[str, Any]:
        """Write a checkpoint at the current update boundary; an existing file is never replaced."""
        if self.agent.rollout_updates < 1:
            raise RuntimeError("a checkpoint requires a completed update")
        state = self.agent.snapshot()
        payload = {"schema_version": SCHEMA, "profile": PROFILE, "seed": self.seed, "roles": self.roles,
                   "recipe": deepcopy(self.agent.config), "recipe_sha256": canonical_hash(self.agent.config),
                   "bank_sha256": self.bank_sha256, "env_config": deepcopy(self.env.config),
                   "input_set": self.input_set, "agent": state,
                   "policy_parameter_sha256": parameter_hash(state["policy_state"]),
                   "environment": self.env.snapshot(), "observation": self.observation.tolist(),
                   "collector": {**{name: getattr(self, name) for name in COUNTERS}, "logs": self.logs,
                                 "needs_reset": self.needs_reset, "initialized_from": self.initialized_from},
                   "runtime": _runtime()}
        target = Path(path)
        atomic_write(target, lambda handle: torch.save(_plain(payload), handle), load_checkpoint)
        return {"path": target.name, "sha256": file_sha256(target),
                "policy_parameter_sha256": payload["policy_parameter_sha256"]}

    def close(self) -> None:
        self.env.close()


__all__ = ["BankTrainer", "ROLE_SEED_ROOT", "load_checkpoint", "policy_agent", "role_seeds", "single_thread"]
