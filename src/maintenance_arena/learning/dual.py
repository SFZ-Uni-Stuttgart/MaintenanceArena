"""PPO on both causal estimates of a noisy measurement: consecutive differences and the line fit.

One candidate design of the article's noise study gives the policy both state
estimates at once. The observation has eight coordinates: the consecutive-
difference state (health, rate, flag, deadline) followed by the line-fit state
reordered to (rate, deadline, health, flag). ``DualPPOAgent`` is the core PPO
recipe with eight inputs; fine-tuning copies a four-input policy into its first
four columns and sets the other four to zero, so before the first update it acts
exactly like the four-input policy on the consecutive-difference state.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from ..banks import ScenarioBankEnv
from ..observations import Reconstructor
from .collector import BankTrainer, PROFILE, load_checkpoint
from .ppo import PPOAgent

BASE_ROUTE, SECOND_ROUTE = "RAW_CAUSAL", "FILTERED_CAUSAL_EWLS8"
# Line-fit state (health, rate, flag, deadline) -> coordinates 4 to 7 (rate, deadline, health, flag).
LINE_FIT_INDEX = (1, 3, 0, 2)
DUAL_TO_LINE_FIT = (6, 4, 7, 5)
FIRST_LAYERS = ("policy_body.0.weight", "value_body.0.weight")


def dual_states(raw_states: np.ndarray, line_states: np.ndarray) -> np.ndarray:
    """Eight-coordinate states from equal consecutive-difference and line-fit state arrays."""
    raw = np.asarray(raw_states, dtype=np.float32)
    line = np.asarray(line_states, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4 or line.shape != raw.shape:
        raise ValueError("dual states need equal (n, 4) consecutive-difference and line-fit arrays")
    return np.ascontiguousarray(np.concatenate([raw, line[:, list(LINE_FIT_INDEX)]], axis=1), dtype=np.float32)


def split_dual(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The consecutive-difference and line-fit parts of eight-coordinate states."""
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("expected (n, 8) dual states")
    return values[:, :4], values[:, list(DUAL_TO_LINE_FIT)]


def _check(array: np.ndarray) -> None:
    if array.ndim != 2 or array.shape[1] != 8 or not np.isfinite(array).all():
        raise ValueError("expected finite (n, 8) policy inputs")
    if (np.any(array[:, :4] < np.asarray([0., 0., 0., -1.], dtype=np.float32)) or np.any(array[:, :4] > 1.)
            or np.any(array[:, 4] < 0.) or np.any(np.abs(array[:, 5:7]) > 1.)
            or not np.logical_or(array[:, 7] == 0., array[:, 7] == 1.).all()):
        raise ValueError("an eight-coordinate input violates the observation bounds")


def _one(observation: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(observation, dtype=np.float32))
    if array.shape != (8,):
        raise ValueError("expected one finite eight-coordinate observation")
    _check(array.reshape(1, 8))
    return array.copy()


class DualPPOAgent(PPOAgent):
    """The core PPO recipe with eight inputs: the first layers of actor and critic take both estimates."""

    def __init__(self, seed: int, *, role_seeds: Mapping[str, int] | None = None):
        super().__init__(PROFILE, seed, role_seeds=role_seeds)
        # Start from exactly the four-input actor and critic; the new columns are zero.
        for body in (self.network.policy_body, self.network.value_body):
            old = body[0]
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Linear(8, old.out_features).cpu()
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :4].copy_(old.weight)
                expanded.bias.copy_(old.bias)
            body[0] = expanded
        optimizer = self.config["optimizer"]
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"],
                                          betas=tuple(optimizer["betas"]), eps=optimizer["epsilon"],
                                          weight_decay=optimizer["weight_decay"])

    def initialize_four(self, policy_state: Mapping[str, torch.Tensor]) -> None:
        """Copy a four-input policy into columns 0 to 3 of the first layers; columns 4 to 7 become zero."""
        expanded = self.policy_state()
        if set(policy_state) != set(expanded):
            raise ValueError("source policy tensor names differ")
        for name, tensor in policy_state.items():
            target = expanded[name]
            if name in FIRST_LAYERS:
                if tuple(tensor.shape) != (64, 4):
                    raise ValueError("initialization requires a four-input policy")
                target.zero_()
                target[:, :4].copy_(tensor)
            else:
                if tensor.shape != target.shape:
                    raise ValueError("source policy tensor geometry differs")
                target.copy_(tensor)
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
                raise ValueError("source tensors must be finite float32")
        self.network.load_state_dict(expanded, strict=True)

    def batch_output(self, observations):
        array = np.ascontiguousarray(np.asarray(observations, dtype=np.float32))
        _check(array)
        with torch.no_grad():
            logits, values = self.network(torch.from_numpy(array))
            probabilities = torch.softmax(logits, dim=-1)
        return tuple(x.detach().cpu().numpy().astype(np.float32, copy=True) for x in (logits, probabilities, values))

    def probabilities(self, observation):
        return self.batch_output(_one(observation).reshape(1, 8))[1][0]

    def predict(self, observation):
        return int(np.argmax(self.probabilities(observation)))

    deterministic_action = predict

    def value(self, observation):
        return float(self.batch_output(_one(observation).reshape(1, 8))[2][0])

    def sample_action(self, observation):
        _, probabilities, values = self.batch_output(_one(observation).reshape(1, 8))
        action = 0 if float(self.action_rng.random()) < float(probabilities[0, 0]) else 1
        log = math.log(max(float(probabilities[0, action]), float(np.finfo(np.float32).tiny)))
        return action, log, float(values[0])


class DualRouteBankEnv(ScenarioBankEnv):
    """A scenario bank with both estimates: the consecutive-difference route plus a line-fit reconstructor.

The accounting, diagnostics and internal observation are those of the
consecutive-difference environment; ``reset`` and ``step`` return the eight
coordinates. The preparation time stays 12. The environment cannot be restored.
"""

    def __init__(self, bank: Mapping[str, Any], config: Mapping[str, Any] | None = None,
                 render_mode: str | None = None):
        options = dict(config or {})
        if options.get("observation_route", BASE_ROUTE) != BASE_ROUTE:
            raise ValueError("the dual-route environment uses the RAW_CAUSAL base route")
        if options.get("mobilization_lead", 12) != 12:
            raise ValueError("the dual-route environment keeps mobilization_lead=12")
        super().__init__(bank, {**options, "observation_route": BASE_ROUTE}, render_mode=render_mode)
        # Public attributes, so the inherited snapshot does not copy them.
        self.dual_reconstructor: Reconstructor | None = None
        self.dual_state: np.ndarray | None = None
        self.observation_space = spaces.Box(low=np.asarray([0., 0., 0., -1., 0., -1., 0., 0.], dtype=np.float32),
                                            high=np.ones(8, dtype=np.float32), dtype=np.float32)

    def _observe_current(self) -> None:
        super()._observe_current()
        if self._reconstructor.age == 0:
            self.dual_reconstructor = Reconstructor(SECOND_ROUTE)
        available = self._active["available"][self._age]
        value = float(self._active["measured_health"][self._age]) if available else None
        self.dual_state = np.asarray(self.dual_reconstructor.step(value), dtype=np.float32)
        if self.dual_reconstructor.age != self._reconstructor.age:
            raise AssertionError("the two estimators lost their common age")

    def dual_observation(self) -> np.ndarray:
        return np.ascontiguousarray(np.concatenate([np.asarray(self._observation, dtype=np.float32),
                                                    self.dual_state[list(LINE_FIT_INDEX)]]), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        return self.dual_observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return self.dual_observation(), reward, terminated, truncated, info

    def snapshot(self) -> dict[str, Any]:
        state = super().snapshot()
        state["dual_route"] = {"route": SECOND_ROUTE, "base_route": BASE_ROUTE,
                               "line_fit_index": list(LINE_FIT_INDEX),
                               "reconstructor": json.loads(json.dumps(self.dual_reconstructor.snapshot())),
                               "state": [float(v) for v in self.dual_state]}
        return state

    def restore(self, state):
        raise RuntimeError("the dual-route environment cannot be restored")


class DualBankTrainer(BankTrainer):
    """``BankTrainer`` with eight inputs; ``initialize`` copies a four-input policy (fine-tuning both estimates)."""

    def __init__(self, bank: Mapping[str, Any], seed: int, *, env_config: Mapping[str, Any] | None = None,
                 initialize: Any = None):
        super().__init__(bank, seed, env_config=env_config, initialize=initialize)

    def _make_agent(self) -> DualPPOAgent:
        return DualPPOAgent(self.seed, role_seeds=self.roles)

    def _make_env(self, bank, env_config) -> DualRouteBankEnv:
        return DualRouteBankEnv(bank, env_config)

    def _load_weights(self, state: Mapping[str, Any]) -> None:
        if tuple(state["policy_body.0.weight"].shape) == (64, 4):
            self.agent.initialize_four(state)
        else:
            self.agent.network.load_state_dict(state, strict=True)


__all__ = ["DUAL_TO_LINE_FIT", "DualBankTrainer", "DualPPOAgent", "DualRouteBankEnv", "LINE_FIT_INDEX", "dual_states",
           "load_checkpoint", "split_dual"]
