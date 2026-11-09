"""Duration-aware DQN and double DQN for the core task (comparison learners of the article).

One decision of the core task lasts one or more calendar steps. The replay stores
each decision with its duration ``k``, and the target is ``r + gamma**k * Q'(s')``
with ``Q'`` the target network, or zero after the last decision of a cycle. Double
DQN chooses the successor action with the online network and evaluates it with
the target network. Exploration, gradient steps and target copies follow the
calendar, not the number of decisions:

* epsilon falls linearly from 1 to ``epsilon_final`` over the first
  ``epsilon_decay_fraction`` of the calendar budget;
* after ``warmup_calendar_steps``, one gradient step on a minibatch drawn without
  replacement is due at every multiple of ``gradient_calendar_interval``;
* the target network copies the online network at every multiple of
  ``target_copy_calendar_interval``, after the gradient steps due with the same
  decision.

Seeds: ``SeedSequence([seed, code])`` gives the network initialization (code 1),
exploration (2) and replay sampling (3). Observations enter unscaled.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SETTINGS = {
    "learning_rate", "activation", "adam_epsilon", "weight_decay", "initialization",
    "learning_rate_schedule", "batch_size", "replay_capacity", "warmup_calendar_steps",
    "gradient_calendar_interval", "gradient_steps_per_tick", "target_copy_calendar_interval",
    "epsilon_decay_fraction", "epsilon_final", "huber_beta", "max_gradient_norm", "dueling",
}
REPLAY = {"observation": np.float32, "next_observation": np.float32, "action": np.int64, "reward": np.float32,
          "duration": np.int32, "bootstrap_mask": np.float32}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, *, positive=False):
    result = float(value)
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


class QNetwork(nn.Module):
    """Hidden layers of ``width`` units and one linear output per action."""

    def __init__(self, obs_dim, action_count, architecture, settings):
        super().__init__()
        activation = nn.Tanh if settings["activation"] == "tanh" else nn.ReLU
        modules, width = [], obs_dim
        for _ in range(architecture["depth"]):
            modules.extend((nn.Linear(width, architecture["width"]), activation()))
            width = architecture["width"]
        modules.append(nn.Linear(width, action_count))
        self.network = nn.Sequential(*modules)
        if settings["initialization"] == "orthogonal":
            for module in self.network:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1. if module is self.network[-1] else math.sqrt(2.))
                    nn.init.zeros_(module.bias)

    def forward(self, observations):
        return self.network(observations)


class QLearner:
    """DQN (``double=False``) or double DQN on one stream of core decisions."""

    def __init__(self, obs_dim, action_count, architecture, settings, gamma, seed, *, double=False):
        self.obs_dim = _integer(obs_dim, "obs_dim", 1)
        self.action_count = _integer(action_count, "action_count", 2)
        self.seed = _integer(seed, "seed")
        if type(double) is not bool:
            raise ValueError("double must be True or False")
        self.algorithm = "DDQN" if double else "DQN"
        if set(architecture) != {"width", "depth"}:
            raise ValueError("the Q architecture has the fields width and depth")
        self.architecture = {key: _integer(architecture[key], key, 1) for key in ("width", "depth")}
        if set(settings) != SETTINGS:
            raise ValueError(f"Q settings differ: missing {sorted(SETTINGS - set(settings))}, "
                             f"unknown {sorted(set(settings) - SETTINGS)}")
        self.settings = deepcopy(dict(settings))
        for key in ("batch_size", "replay_capacity", "gradient_calendar_interval", "gradient_steps_per_tick",
                    "target_copy_calendar_interval"):
            self.settings[key] = _integer(settings[key], key, 1)
        self.settings["warmup_calendar_steps"] = _integer(settings["warmup_calendar_steps"], "warmup_calendar_steps")
        for key in ("learning_rate", "adam_epsilon", "huber_beta", "max_gradient_norm", "epsilon_decay_fraction"):
            self.settings[key] = _number(settings[key], key, positive=True)
        for key in ("weight_decay", "epsilon_final"):
            self.settings[key] = _number(settings[key], key)
        if settings["activation"] not in ("tanh", "relu") or settings["initialization"] not in ("orthogonal",
                                                                                              "pytorch_default"):
            raise ValueError("unsupported activation or initialization")
        if settings["learning_rate_schedule"] not in ("constant", "linear_to_10_percent"):
            raise ValueError("unsupported learning-rate schedule")
        if settings["dueling"] is not False:
            raise ValueError("only the plain Q head is implemented (dueling = False)")
        if self.settings["epsilon_final"] > 1 or self.settings["epsilon_decay_fraction"] > 1:
            raise ValueError("epsilon_final and epsilon_decay_fraction must not exceed 1")
        if self.settings["batch_size"] > self.settings["replay_capacity"]:
            raise ValueError("the replay must hold at least one minibatch")
        self.gamma = _number(gamma, "gamma", positive=True)
        if self.gamma > 1:
            raise ValueError("gamma must lie in (0, 1]")
        self.roles = {name: int(np.random.SeedSequence([self.seed, code]).generate_state(1, dtype=np.uint64)[0]) % 2**63
                      for name, code in (("initialization", 1), ("action", 2), ("replay", 3))}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.roles["initialization"])
            self.online = QNetwork(self.obs_dim, self.action_count, self.architecture, self.settings)
        self.target = deepcopy(self.online).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.settings["learning_rate"],
                                          betas=(.9, .999), eps=self.settings["adam_epsilon"],
                                          weight_decay=self.settings["weight_decay"])
        self.action_rng = np.random.Generator(np.random.PCG64(self.roles["action"]))
        self.replay_rng = np.random.Generator(np.random.PCG64(self.roles["replay"]))
        self.capacity = self.settings["replay_capacity"]
        self.replay = {key: np.empty((self.capacity, self.obs_dim) if "observation" in key else self.capacity,
                                     dtype=dtype) for key, dtype in REPLAY.items()}
        self.replay_size = self.replay_cursor = self.observed_transitions = 0
        self.pending_calendar_steps = self.calendar_steps = 0
        self.phase_budget = None
        self.optimizer_updates = self.target_copies = self.replay_sample_presentations = 0
        self.gradient_ticks_processed = self.gradient_ticks_skipped = 0
        self.last_metrics = {}

    @property
    def parameters_count(self):
        """Trainable parameters of the online network; the target network is a copy."""
        return sum(parameter.numel() for parameter in self.online.parameters())

    def _inputs(self, observations):
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.obs_dim or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"observations must be finite with shape (n, {self.obs_dim})")
        return torch.as_tensor(np.ascontiguousarray(values))

    def _budget(self, total):
        total = _integer(total, "calendar budget", 1)
        if self.phase_budget is not None and total != self.phase_budget:
            raise ValueError("the calendar budget cannot change during training")
        self.phase_budget = total
        return total

    def epsilon(self, calendar_steps, total_calendar_steps):
        exposure = _integer(calendar_steps, "calendar steps")
        total = _integer(total_calendar_steps, "calendar budget", 1)
        progress = min(exposure / (total * self.settings["epsilon_decay_fraction"]), 1.)
        return 1. + (self.settings["epsilon_final"] - 1.) * progress

    def greedy(self, observations):
        """Greedy actions of a batch of observations (the first action on a tie)."""
        with torch.no_grad():
            q_values = self.online(self._inputs(observations))
        if not torch.isfinite(q_values).all():
            raise ArithmeticError("nonfinite Q values")
        return q_values.max(dim=-1).indices.numpy().copy()

    def act(self, observations, calendar_steps, total_calendar_steps):
        """Epsilon-greedy actions; one uniform draw per observation, and a second for a random action."""
        actions = self.greedy(observations)
        self._budget(total_calendar_steps)
        epsilon = self.epsilon(calendar_steps, total_calendar_steps)
        for index in range(len(actions)):
            if float(self.action_rng.random()) < epsilon:
                actions[index] = int(self.action_rng.integers(self.action_count))
        return actions

    def observe(self, observation, action, reward, next_observation, duration, terminal):
        """Insert one completed decision into the replay."""
        action = _integer(action, "action")
        duration = _integer(duration, "duration", 1)
        if action >= self.action_count or not math.isfinite(float(reward)):
            raise ValueError("invalid action or reward")
        row = {"observation": observation, "next_observation": next_observation, "action": action,
               "reward": np.float32(reward), "duration": duration, "bootstrap_mask": float(not terminal)}
        for key, value in row.items():
            self.replay[key][self.replay_cursor] = value
        self.replay_cursor = (self.replay_cursor + 1) % self.capacity
        self.replay_size = min(self.replay_size + 1, self.capacity)
        self.observed_transitions += 1
        self.pending_calendar_steps += duration

    def _targets(self, batch):
        rewards = torch.as_tensor(batch["reward"], dtype=torch.float32)
        durations = torch.as_tensor(batch["duration"], dtype=torch.int64)
        bootstrap = torch.as_tensor(batch["bootstrap_mask"], dtype=torch.bool)
        successors = torch.as_tensor(batch["next_observation"], dtype=torch.float32)
        with torch.no_grad():
            target_q = self.target(successors)
            if self.algorithm == "DDQN":
                selected = self.online(successors).argmax(dim=-1)
                next_value = target_q.gather(1, selected[:, None]).squeeze(1)
            else:
                next_value = target_q.max(dim=-1).values
            next_value = torch.where(bootstrap, next_value, torch.zeros_like(next_value))
            result = rewards + torch.pow(torch.full_like(rewards, self.gamma), durations) * next_value
        if not torch.isfinite(result).all():
            raise ArithmeticError("nonfinite Q target")
        return result

    def update(self):
        """One gradient step on a minibatch drawn without replacement."""
        count = self.settings["batch_size"]
        indices = np.asarray(self.replay_rng.choice(self.replay_size, size=count, replace=False), dtype=np.int64)
        self.replay_sample_presentations += count
        batch = {key: np.ascontiguousarray(value[indices]) for key, value in self.replay.items()}
        observations = torch.as_tensor(batch["observation"], dtype=torch.float32)
        actions = torch.as_tensor(batch["action"], dtype=torch.int64)
        target = self._targets(batch)
        current = self.online(observations).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(current, target, beta=self.settings["huber_beta"], reduction="mean")
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite Q loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.settings["max_gradient_norm"],
                                        error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer_updates += 1
        self.last_metrics = {"loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                             "mean_abs_td_error": float((target - current).detach().abs().mean()),
                             "mean_target": float(target.mean())}
        return dict(self.last_metrics)

    def learn_due(self, calendar_before, calendar_after, total_calendar_steps):
        """The gradient steps and then the target copies due between two calendar steps."""
        before = _integer(calendar_before, "calendar before")
        after = _integer(calendar_after, "calendar after")
        if before != self.calendar_steps or after < before or after - before != self.pending_calendar_steps:
            raise ValueError("the calendar must advance by exactly the durations inserted since the last call")
        total = self._budget(total_calendar_steps)
        interval = self.settings["gradient_calendar_interval"]
        performed, skipped, metrics = [], [], []
        for tick in range((before // interval + 1) * interval, after + 1, interval):
            if tick <= self.settings["warmup_calendar_steps"]:
                continue
            self.gradient_ticks_processed += 1
            if self.replay_size < self.settings["batch_size"]:
                self.gradient_ticks_skipped += 1
                skipped.append(tick)
                continue
            rate = self.settings["learning_rate"]
            if self.settings["learning_rate_schedule"] == "linear_to_10_percent":
                rate *= 1. - .9 * min(tick / total, 1.)
            for group in self.optimizer.param_groups:
                group["lr"] = rate
            for _ in range(self.settings["gradient_steps_per_tick"]):
                metrics.append(self.update())
            performed.append(tick)
        copy_interval = self.settings["target_copy_calendar_interval"]
        copies = list(range((before // copy_interval + 1) * copy_interval, after + 1, copy_interval))
        for _ in copies:
            self.target.load_state_dict(self.online.state_dict(), strict=True)
            self.target_copies += 1
        self.calendar_steps = after
        self.pending_calendar_steps = 0
        return {"gradient_ticks": performed, "skipped_gradient_ticks": skipped, "target_copy_ticks": copies,
                "updates_this_call": len(metrics), "optimizer_updates": self.optimizer_updates,
                "target_copies": self.target_copies, "replay_size": self.replay_size,
                "replay_sample_presentations": self.replay_sample_presentations,
                "calendar_steps": self.calendar_steps, "epsilon": self.epsilon(after, total),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "metrics": {key: math.fsum(item[key] for item in metrics) / len(metrics)
                            for key in metrics[0]} if metrics else {}}

    def policy_state(self):
        return {key: value.detach().clone() for key, value in self.online.state_dict().items()}

    def snapshot(self):
        """Weights, optimizer, the filled part of the replay, counters and random streams."""
        counters = ("replay_size", "replay_cursor", "observed_transitions", "pending_calendar_steps",
                    "calendar_steps", "phase_budget", "optimizer_updates", "target_copies",
                    "gradient_ticks_processed", "gradient_ticks_skipped", "replay_sample_presentations")
        return {"algorithm": self.algorithm, "seed": self.seed, "roles": dict(self.roles),
                "online": self.policy_state(),
                "target": {key: value.detach().clone() for key, value in self.target.state_dict().items()},
                "optimizer": deepcopy(self.optimizer.state_dict()),
                "replay": {key: torch.from_numpy(value[:self.replay_size].copy()) for key, value in self.replay.items()},
                "counters": {name: getattr(self, name) for name in counters},
                "action_rng": deepcopy(self.action_rng.bit_generator.state),
                "replay_rng": deepcopy(self.replay_rng.bit_generator.state),
                "last_metrics": dict(self.last_metrics)}

    def restore(self, state):
        if state["algorithm"] != self.algorithm or state["seed"] != self.seed or state["roles"] != self.roles:
            raise ValueError("the Q snapshot belongs to another learner")
        self.online.load_state_dict(state["online"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        size = int(state["counters"]["replay_size"])
        for key, destination in self.replay.items():
            saved = state["replay"][key].numpy()
            if saved.shape != (size, *destination.shape[1:]) or saved.dtype != destination.dtype:
                raise ValueError(f"the Q snapshot's replay column {key} has another shape or type")
            destination[:size] = saved
        for name, value in state["counters"].items():
            setattr(self, name, value)
        self.action_rng.bit_generator.state = deepcopy(state["action_rng"])
        self.replay_rng.bit_generator.state = deepcopy(state["replay_rng"])
        self.last_metrics = dict(state["last_metrics"])


__all__ = ["QLearner", "QNetwork"]
