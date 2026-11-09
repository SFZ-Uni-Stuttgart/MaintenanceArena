"""Bearing PPO: the core task's duration-aware PPO recipe on replayed bearing lives.

Network, optimizer, sampling, minibatch order and update numerics are those of
``maintenance_arena.learning.ppo`` with profile ``paper-core-continuous-v1``; only the
first layers of actor and critic are replaced to take the bearing inputs. Training draws
a training bearing uniformly with replacement for every episode, starts at minute 1 and
centers every calendar minute's cash on a moving estimate of calendar profit. Requires
PyTorch (the ``rl`` extra).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from maintenance_arena.learning.ppo import PPOAgent, RolloutEntry

from .economics import ReplayEconomics, pooled_rate
from .env import BearingReplayEnv

RECIPE_PROFILE = "paper-core-continuous-v1"
CHECKPOINT_SCHEMA = "maintenance_arena_bearing_policy_v1"


class BearingPPOAgent(PPOAgent):
    """The core PPO agent with an ``n_inputs``-wide first layer in actor and critic.

    Seed roles and the remaining layers follow the core recipe. The new first layers are
    drawn after reseeding Torch with the network role, policy layer first, and a fresh
    Adam optimizer covers all parameters.
    """

    def __init__(self, seed: int, n_inputs: int):
        if isinstance(n_inputs, bool) or not isinstance(n_inputs, int) or n_inputs < 1:
            raise ValueError("n_inputs must be a positive integer")
        super().__init__(RECIPE_PROFILE, seed)
        self.n_inputs = n_inputs
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.roles["network"])
            for body in (self.network.policy_body, self.network.value_body):
                layer = nn.Linear(n_inputs, 64)
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
                body[0] = layer
        optimizer = self.config["optimizer"]
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"],
                                          betas=tuple(optimizer["betas"]), eps=optimizer["epsilon"],
                                          weight_decay=optimizer["weight_decay"])

    def batch_output(self, observations):
        array = np.ascontiguousarray(np.asarray(observations, dtype=np.float32))
        if array.ndim != 2 or array.shape[1] != self.n_inputs or not np.isfinite(array).all():
            raise ValueError(f"expected finite shape-(n, {self.n_inputs}) policy inputs")
        with torch.no_grad():
            logits, values = self.network(torch.from_numpy(array))
            probabilities = torch.softmax(logits, dim=-1)
        return tuple(x.detach().cpu().numpy().astype(np.float32, copy=True) for x in (logits, probabilities, values))

    @staticmethod
    def _row(observation) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32).reshape(1, -1)

    def probabilities(self, observation):
        return self.batch_output(self._row(observation))[1][0]

    def predict(self, observation):
        """Greedy action; WAIT on ties."""
        return int(np.argmax(self.probabilities(observation)))

    deterministic_action = predict

    def value(self, observation):
        return float(self.batch_output(self._row(observation))[2][0])

    def sample_action(self, observation):
        _, probabilities, values = self.batch_output(self._row(observation))
        action = 0 if float(self.action_rng.random()) < float(probabilities[0, 0]) else 1
        log = math.log(max(float(probabilities[0, action]), float(np.finfo(np.float32).tiny)))
        return action, log, float(values[0])

    def order_probabilities(self, observations) -> np.ndarray:
        """P(ORDER) at every row of an input matrix."""
        return self.batch_output(observations)[1][:, 1]

    def greedy_order_minute(self, observations) -> int | None:
        """First minute whose ORDER probability exceeds the WAIT probability, or None."""
        probabilities = self.batch_output(observations)[1]
        hits = np.flatnonzero(probabilities[:, 1] > probabilities[:, 0])
        return None if hits.size == 0 else int(hits[0] + 1)


@dataclass(frozen=True)
class TrainingSettings:
    """Reward centering and bookkeeping of bearing PPO training (the article's values).

    With ``own_rate`` centering the rate starts at the pooled reactive rate of the training
    lives, weighted as ``prior_lives`` mean lives of calendar time, and follows
    A <- decay A + cash, W <- decay W + calendar after every episode, rate = A / W.
    """
    reward_center: str = "own_rate"
    reward_scale: float = 1000.0
    rate_decay: float = 0.99
    prior_lives: float = 100.0
    log_every_rollouts: int = 25
    torch_threads: int = 1

    def __post_init__(self):
        if self.reward_center not in ("own_rate", "none"):
            raise ValueError("reward_center must be own_rate or none")
        if not self.reward_scale > 0 or not 0 <= self.rate_decay <= 1 or not self.prior_lives > 0:
            raise ValueError("reward_scale and prior_lives must be positive and rate_decay within [0, 1]")
        for name in ("log_every_rollouts", "torch_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class TrainingResult:
    agent: BearingPPOAgent
    calendar_steps: int
    episodes: int
    failures: int
    seconds: float
    logs: list[dict[str, float]] = field(default_factory=list)


def _training_banks(observations: Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(observations, np.ndarray) or len(observations) == 0:
        raise ValueError("training needs a nonempty sequence of input matrices")
    banks = [np.asarray(x) for x in observations]
    width = banks[0].shape[1] if banks[0].ndim == 2 else 0
    for bank in banks:
        if bank.dtype != np.float32 or bank.ndim != 2 or len(bank) == 0 or bank.shape[1] != width or width == 0:
            raise ValueError("training inputs must be nonempty float32 matrices of equal width")
    return banks


def train_policy(observations: Sequence[np.ndarray], seed: int, calendar_steps: int, *,
                 economics: ReplayEconomics | None = None,
                 settings: TrainingSettings | None = None) -> TrainingResult:
    """Train one bearing actor on replayed training lives.

    ``observations`` holds one float32 (T_i, n) input matrix per training bearing, in the
    order that indexes bearing sampling. Rollouts close at multiples of 1,024 calendar
    minutes, and training ends with the first completed update at or beyond
    ``calendar_steps``. Torch runs with ``settings.torch_threads`` threads during training.
    """
    started = time.perf_counter()
    settings = settings or TrainingSettings()
    economics = economics or ReplayEconomics()
    if isinstance(calendar_steps, bool) or not isinstance(calendar_steps, int) or calendar_steps < 1:
        raise ValueError("calendar_steps must be a positive integer")
    banks = _training_banks(observations)
    width = banks[0].shape[1]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(settings.torch_threads)
    try:
        agent = BearingPPOAgent(seed, width)
        gamma, quota = agent.config["gamma"], agent.config["rollout_calendar_steps"]
        env = BearingReplayEnv(banks, economics=economics, gamma=gamma, reward_scale=settings.reward_scale)
        lives = env.lives
        centered = settings.reward_center == "own_rate"
        rho = rate_cash = rate_calendar = 0.0
        if centered:
            rate_calendar = settings.prior_lives * float(np.mean(lives))
            rate_cash = pooled_rate(economics.outcome(life, None) for life in lives) * rate_calendar
            rho = rate_cash / rate_calendar
        observation, _ = env.reset(seed=agent.roles["environment"])
        terminal = np.zeros(width, dtype=np.float32)
        calendar, history, logs = 0, [], []
        while calendar < calendar_steps:
            limit = min(calendar_steps, (calendar // quota + 1) * quota)
            entries = []
            while calendar < limit:
                action, log_probability, value = agent.sample_action(observation)
                successor, _, terminated, _, info = env.step(action)
                cash = [row["raw_profit"] for row in info["ledger"]]
                duration = len(cash)
                discounted = math.fsum(gamma ** j * (c - rho) / settings.reward_scale for j, c in enumerate(cash))
                if terminated:
                    following, following_value = terminal, 0.0
                else:
                    following, following_value = successor, agent.value(successor)
                entries.append(RolloutEntry(observation.copy(), action, log_probability, value, discounted,
                                            following.copy(), following_value, duration, float(not terminated),
                                            float(not terminated), float(math.fsum(cash)), calendar,
                                            calendar + duration))
                calendar += duration
                if terminated:
                    outcome = env.episode_outcome()
                    history.append((outcome.cash, outcome.calendar, int(outcome.failed)))
                    if centered:
                        rate_cash = settings.rate_decay * rate_cash + outcome.cash
                        rate_calendar = settings.rate_decay * rate_calendar + outcome.calendar
                        rho = rate_cash / rate_calendar
                    observation, _ = env.reset()
                else:
                    observation = successor
            entries[-1] = replace(entries[-1], trace_mask=0.0)
            update = agent.update(entries)
            if update["rollout_update_ordinal"] % settings.log_every_rollouts == 0 and history:
                recent = history[-200:]
                logs.append({"rollout": update["rollout_update_ordinal"], "calendar": calendar,
                             "train_rate_recent": math.fsum(c for c, _, _ in recent) / math.fsum(k for _, k, _ in recent),
                             "train_failure_share_recent": sum(f for *_, f in recent) / len(recent),
                             "entropy": update["entropy_mean"], "value_loss": update["value_loss_mean"], "rho": rho})
    finally:
        torch.set_num_threads(previous_threads)
    return TrainingResult(agent, calendar, len(history), sum(f for *_, f in history),
                          time.perf_counter() - started, logs)


def save_policy(path: str | Path, agent: BearingPPOAgent, **metadata: Any) -> None:
    """Save the network weights with JSON-compatible metadata; an existing file is never replaced."""
    payload = {"schema": CHECKPOINT_SCHEMA, "seed": agent.seed, "n_inputs": agent.n_inputs,
               "roles": dict(agent.roles), "recipe": agent.config, "optimizer_updates": agent.optimizer_updates,
               "rollout_updates": agent.rollout_updates, "metadata": metadata,
               "network_state_dict": agent.policy_state()}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(payload, stream)


def load_policy(path: str | Path) -> tuple[BearingPPOAgent, dict[str, Any]]:
    """Load an actor saved by ``save_policy`` and return it with its metadata."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path}: not a bearing policy checkpoint")
    agent = BearingPPOAgent(payload["seed"], payload["n_inputs"])
    if payload["roles"] != agent.roles or payload["recipe"] != agent.config:
        raise ValueError(f"{path}: seed roles or recipe differ from this package")
    agent.network.load_state_dict(payload["network_state_dict"], strict=True)
    return agent, payload["metadata"]
