"""The three learners of the article's learner comparison: PPO, DQN and double DQN.

All three learn on the same finite scenario bank, with the same four observations,
the same rewards (gamma = 0.999 per calendar step, no continuation after a
renewal) and the same calendar budget. Each has one fixed compact recipe
(``recipe``), the one of Appendix A of the article:

* ``PPO``: the core recipe (``paper-core-continuous-v1``), trained by ``BankTrainer``.
* ``DQN`` and ``DDQN``: two ReLU layers of 64 units, a replay of 100,000 decisions,
  2,048 warm-up steps, a gradient step every 4 calendar steps, a target copy every
  1,024, epsilon from 1 to 0.03 over the first half of the budget, Huber loss and a
  gradient norm of at most 10 (``learning.q_learning``). They differ only in the
  choice of the successor action of the target.

``make_trainer`` returns a trainer with ``train_until``, ``save`` and ``close``.
The environment of every learner replays the bank with the measurement seed of
``BankTrainer`` (``role_seeds(seed)["environment"]``), so runs with the same seed
start on the same components. The exploration of DQN and DDQN depends on the full
budget, so ``total_calendar_steps`` stays fixed when training stops at milestones.
``load_policy`` and ``greedy_first_orders`` replay saved checkpoints greedily on
whole decision series, one decision of every life at a time.

Checkpoints are written once and load with Torch's weights-only reader. Load them
only from trusted sources.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..banks import ScenarioBankEnv
from ..replay import DecisionSeries
from .checkpoints import _plain, atomic_write, file_sha256
from .collector import (PROFILE, SCHEMA as PPO_SCHEMA, BankTrainer, _runtime, load_checkpoint, policy_agent,
                        role_seeds, single_thread)
from .ppo import canonical_hash, parameter_hash, recipe_config
from .q_learning import QLearner

ALGORITHMS = ("PPO", "DQN", "DDQN")
LABELS = {"PPO": "PPO", "DQN": "DQN", "DDQN": "Double DQN"}
SCHEMA = "maintenance_arena_bank_learner_checkpoint_v1"
GAMMA = .999
COUNTERS = ("calendar", "decisions", "episodes", "failures", "raw_profit")


def recipe(algorithm: str) -> dict[str, Any]:
    """The fixed recipe of one learner: architecture, settings, gamma and input scaling."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    settings: dict[str, Any] = recipe_config(PROFILE)
    if algorithm in ("DQN", "DDQN"):
        settings = {"learning_rate": 3e-4, "activation": "relu", "adam_epsilon": 1e-8, "weight_decay": 0.,
                    "initialization": "pytorch_default", "learning_rate_schedule": "constant", "batch_size": 128,
                    "replay_capacity": 100000, "warmup_calendar_steps": 2048, "gradient_calendar_interval": 4,
                    "gradient_steps_per_tick": 1, "target_copy_calendar_interval": 1024,
                    "epsilon_decay_fraction": .5, "epsilon_final": .03, "huber_beta": 1., "max_gradient_norm": 10.,
                    "dueling": False}
    return {"algorithm": algorithm, "architecture": {"width": 64, "depth": 2}, "settings": settings,
            "gamma": GAMMA, "normalization": "none; the four observations enter unscaled"}


def make_agent(algorithm: str, seed: int) -> QLearner:
    """A fresh DQN or DDQN learner with its seeded initialization."""
    if algorithm not in ("DQN", "DDQN"):
        raise ValueError("make_agent builds DQN and DDQN; PPO is trained by BankTrainer")
    values = recipe(algorithm)
    with single_thread():
        return QLearner(4, 2, values["architecture"], values["settings"], values["gamma"], int(seed),
                        double=algorithm == "DDQN")


def parameter_count(algorithm: str) -> int:
    """Trainable parameters: actor and critic for PPO, the online network for DQN and DDQN."""
    if algorithm == "PPO":
        from .ppo import ActorCritic
        with torch.random.fork_rng(devices=[]):
            return sum(parameter.numel() for parameter in ActorCritic(2).parameters())
    return make_agent(algorithm, 0).parameters_count


def make_trainer(algorithm: str, bank: Mapping[str, Any], seed: int, *, total_calendar_steps: int,
                 env_config: Mapping[str, Any] | None = None, resume: str | Path | None = None
                 ) -> BankTrainer | LearnerTrainer:
    """The trainer of one learner on one bank; PPO uses ``BankTrainer``."""
    if algorithm == "PPO":
        return BankTrainer(bank, seed, env_config=env_config, resume=resume)
    return LearnerTrainer(bank, seed, algorithm=algorithm, total_calendar_steps=total_calendar_steps,
                          env_config=env_config, resume=resume)


def _checked(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA:
        raise ValueError("not a learner checkpoint of this package")
    if canonical_hash(payload["recipe"]) != payload["recipe_sha256"] or payload["recipe"] != recipe(payload["algorithm"]):
        raise ValueError("the checkpoint recipe differs from the recipe of its learner")
    if parameter_hash(payload["agent"]["online"]) != payload["policy_parameter_sha256"]:
        raise ValueError("checkpoint parameter hash failed")
    return dict(payload)


def load_learner_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a DQN or DDQN checkpoint with Torch's weights-only reader and check its bindings."""
    return _checked(torch.load(Path(path), map_location="cpu", weights_only=True))


class LearnerTrainer:
    """Collector of DQN or DDQN on one finite training bank, with milestones and exact resumption."""

    def __init__(self, bank: Mapping[str, Any], seed: int, *, algorithm: str, total_calendar_steps: int,
                 env_config: Mapping[str, Any] | None = None, resume: str | Path | None = None):
        if algorithm not in ("DQN", "DDQN"):
            raise ValueError("LearnerTrainer trains DQN and DDQN; PPO uses BankTrainer")
        if (isinstance(total_calendar_steps, bool) or not isinstance(total_calendar_steps, int)
                or total_calendar_steps < 1):
            raise ValueError("total_calendar_steps must be a positive integer")
        self.algorithm = algorithm
        self.recipe = recipe(algorithm)
        self.seed = int(seed)
        self.total_calendar_steps = total_calendar_steps
        self.agent = make_agent(algorithm, self.seed)
        self.env = ScenarioBankEnv(bank, env_config)
        self.bank_sha256 = bank["sha256"]
        self.calendar = self.decisions = self.episodes = self.failures = 0
        self.raw_profit = 0.
        self.logs: list[dict[str, Any]] = []
        self.needs_reset = False
        self.observation = self.env.reset(seed=role_seeds(self.seed)["environment"])[0]
        if resume is not None:
            self._resume(resume)

    def _resume(self, path: str | Path) -> None:
        payload = load_learner_checkpoint(path)
        expected = {"algorithm": self.algorithm, "seed": self.seed, "bank_sha256": self.bank_sha256,
                    "total_calendar_steps": self.total_calendar_steps}
        for key, value in expected.items():
            if payload[key] != value:
                raise ValueError(f"the checkpoint's {key} differs from this trainer")
        if payload["runtime"] != _runtime():
            raise ValueError("exact resumption requires the checkpoint's Python, NumPy and Torch versions")
        self.agent.restore(payload["agent"])
        restored = self.env.restore(payload["environment"])  # also checks the environment configuration
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

    def train_until(self, milestone: int) -> dict[str, Any]:
        """Train until at least ``milestone`` calendar steps; the last decision may run past it."""
        if (isinstance(milestone, bool) or not isinstance(milestone, int)
                or not self.calendar < milestone <= self.total_calendar_steps):
            raise ValueError("the milestone must lie beyond the steps already trained and within the budget")
        total = self.total_calendar_steps
        update: dict[str, Any] = {}
        with single_thread():
            while self.calendar < milestone:
                if self.needs_reset:
                    self.observation = self.env.reset()[0]
                    self.needs_reset = False
                old = self.observation.copy()
                action = int(self.agent.act(old[None, :], self.calendar, total)[0])
                successor, reward, terminal, truncated, info = self.env.step(action)
                if truncated:
                    raise RuntimeError("training cycles must not be truncated")
                duration = int(info["duration_calendar_steps"])
                if info["gamma_pow_duration"] != GAMMA ** duration:
                    raise ValueError(f"the environment's gamma differs from the learners' {GAMMA}")
                before = self.calendar
                self.calendar += duration
                self.decisions += 1
                self.raw_profit += info["raw_profit_sum"]
                self.observation = successor
                self.needs_reset = bool(terminal)
                if terminal:
                    self.episodes += 1
                    self.failures += info["failures"]
                self.agent.observe(old, action, reward, successor.copy(), duration, terminal)
                update = self.agent.learn_due(before, self.calendar, total)
            self.logs.append({**update, "actual_calendar_steps": self.calendar, "episodes": self.episodes,
                              "failures": self.failures, "raw_profit": self.raw_profit})
        return {"nominal_calendar_steps": milestone, "actual_calendar_steps": self.calendar,
                "calendar_overshoot": self.calendar - milestone, "decisions": self.decisions,
                "completed_cycles": self.episodes, "failures_during_training": self.failures,
                "raw_profit_during_training": self.raw_profit, "rollout_updates": 0,
                "optimizer_minibatches": self.agent.optimizer_updates}

    def save(self, path: str | Path) -> dict[str, Any]:
        """Write a checkpoint; an existing file is never replaced."""
        if self.calendar < 1:
            raise RuntimeError("a checkpoint requires training")
        state = self.agent.snapshot()
        payload = {"schema_version": SCHEMA, "algorithm": self.algorithm, "seed": self.seed,
                   "recipe": deepcopy(self.recipe), "recipe_sha256": canonical_hash(self.recipe),
                   "total_calendar_steps": self.total_calendar_steps, "bank_sha256": self.bank_sha256,
                   "env_config": deepcopy(self.env.config), "agent": state,
                   "policy_parameter_sha256": parameter_hash(state["online"]),
                   "environment": self.env.snapshot(), "observation": self.observation.tolist(),
                   "collector": {**{name: getattr(self, name) for name in COUNTERS}, "logs": self.logs,
                                 "needs_reset": self.needs_reset},
                   "runtime": _runtime()}
        target = Path(path)
        atomic_write(target, lambda handle: torch.save(_plain(payload), handle), load_learner_checkpoint)
        return {"path": target.name, "sha256": file_sha256(target),
                "policy_parameter_sha256": payload["policy_parameter_sha256"]}

    def close(self) -> None:
        self.env.close()


class GreedyPolicy:
    """Greedy decisions of a trained learner for a batch of observations."""

    def __init__(self, algorithm: str, agent: Any, checkpoint_sha256: str | None = None):
        self.algorithm, self.agent, self.checkpoint_sha256 = algorithm, agent, checkpoint_sha256

    def actions(self, observations: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(np.asarray(observations, dtype=np.float32))
        if self.algorithm == "PPO":
            return np.argmax(self.agent.batch_output(values)[1], axis=1).astype(np.int64)
        return self.agent.greedy(values)


def load_policy(path: str | Path) -> GreedyPolicy:
    """The greedy policy of a ``BankTrainer`` (PPO) or ``LearnerTrainer`` (DQN, DDQN) checkpoint."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and payload.get("schema_version") == PPO_SCHEMA:
        return GreedyPolicy("PPO", policy_agent(load_checkpoint(path)), file_sha256(path))
    payload = _checked(payload)
    agent = make_agent(payload["algorithm"], payload["seed"])
    agent.online.load_state_dict(payload["agent"]["online"], strict=True)
    return GreedyPolicy(payload["algorithm"], agent, file_sha256(path))


def greedy_first_orders(policy: GreedyPolicy, series: DecisionSeries) -> np.ndarray:
    """First orders (-1 without one) of a greedy policy on all lives of ``series``.

All lives advance together, one decision per call of the policy, as in the
article's evaluation. A life no longer counts after its last productive step.
"""
    lengths = np.asarray([stop - start for start, stop in series.slices])
    steps = np.zeros((int(lengths.max()), series.lives, series.states.shape[1]), dtype=np.float32)
    for life, (start, stop) in enumerate(series.slices):
        steps[:stop - start, life] = series.states[start:stop]
    orders = np.full(series.lives, -1, dtype=np.int64)
    with single_thread():
        for age, observations in enumerate(steps):
            new = (age < lengths) & (orders < 0) & (policy.actions(observations) == 1)
            orders[new] = age
            if np.all(orders >= 0):
                break
    return orders


__all__ = ["ALGORITHMS", "LABELS", "GreedyPolicy", "LearnerTrainer", "greedy_first_orders", "load_learner_checkpoint",
           "load_policy", "make_agent", "make_trainer", "parameter_count", "recipe"]
