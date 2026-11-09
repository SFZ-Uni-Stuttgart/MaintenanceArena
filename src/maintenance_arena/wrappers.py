"""Gymnasium wrappers used by the benchmark tracks; they never change the wrapped environment.

``ZeroDeadline`` sets the core's deadline coordinate (observation index 3,
``causal_intervention_slack``) to zero. This is a matched control: the input
keeps its size, so a network keeps its capacity, but the coordinate carries
nothing. Health, wear rate and the validity
flag are unchanged, and the coordinate is a function of them, so the control
removes a representation, not information.

``InputSet`` selects one of the three input sets of the article's ablation:
Full inputs (``full``), Deadline only (``deadline_only``: health, rate and flag
set to zero) and Health + rate (``health_rate``: the deadline coordinate set to
zero). Unused inputs are multiplied by zero, so all three sets keep the network's
four inputs. Health + rate differs from ``ZeroDeadline`` only in the sign of a
zero (a negative coordinate times zero is -0.0), which no policy output depends on.

``CalendarBudget`` truncates an episode once its elapsed calendar steps reach a
budget. Gymnasium's ``TimeLimit`` counts decisions, but a decision in these tasks
lasts one or more calendar steps (``info["duration_calendar_steps"]``). The
wrapper sums those durations and truncates after the transition that reaches the
budget, exactly where the benchmark's evaluation loop stops. The transition
completes first, so the elapsed time can exceed the budget by the rest of that
transition, as in the evaluation.

The wrappers act on what the agent sees or on when its episode ends. The wrapped
environment, its configuration, its random streams and its ``info`` are untouched.
"""
from __future__ import annotations

from numbers import Integral

import gymnasium as gym
import numpy as np


DEADLINE_INDEX = 3
DEADLINE_NAME = "causal_intervention_slack"
# Input sets of the article's ablation: 1 keeps a coordinate, 0 sets it to zero.
INPUT_SETS = {"full": (1., 1., 1., 1.), "deadline_only": (0., 0., 0., 1.), "health_rate": (1., 1., 1., 0.)}
INPUT_SET_NAMES = {"full": "Full inputs", "deadline_only": "Deadline only", "health_rate": "Health + rate"}


def input_set(observation, name: str) -> np.ndarray:
    """A float32 core observation (or batch) with the inputs outside the named input set multiplied by zero."""
    if name not in INPUT_SETS:
        raise ValueError(f"input set must be one of {sorted(INPUT_SETS)}")
    value = np.asarray(observation, dtype=np.float32)
    if value.ndim == 0 or value.shape[-1] != 4 or not np.isfinite(value).all():
        raise ValueError("input sets apply to finite core observations with four coordinates in the last dimension")
    return value * np.asarray(INPUT_SETS[name], dtype=np.float32)


def zero_deadline(observation) -> np.ndarray:
    """A float32 copy of a core observation (or a batch of them) with the deadline coordinate set to zero."""
    value = np.array(observation, dtype=np.float32, copy=True)
    if value.ndim == 0 or value.shape[-1] != 4:
        raise ValueError("zero_deadline expects core observations with four coordinates in the last dimension")
    value[..., DEADLINE_INDEX] = 0.0
    return value


class ZeroDeadline(gym.ObservationWrapper):
    """Core observation with the deadline coordinate (index 3) set to zero; the space is unchanged."""

    def __init__(self, env: gym.Env):
        names = getattr(env.unwrapped, "observation_names", ())
        if len(names) != 4 or names[DEADLINE_INDEX] != DEADLINE_NAME:
            raise ValueError("ZeroDeadline applies to the core task (maintenanceArena/CoreCycle-v0), whose "
                             f"observation index {DEADLINE_INDEX} is {DEADLINE_NAME}")
        super().__init__(env)

    def observation(self, observation):
        return zero_deadline(observation)


class InputSet(gym.ObservationWrapper):
    """Core observation restricted to one input set of the article's ablation; the space is unchanged."""

    def __init__(self, env: gym.Env, name: str):
        names = getattr(env.unwrapped, "observation_names", ())
        if len(names) != 4 or names[DEADLINE_INDEX] != DEADLINE_NAME:
            raise ValueError("InputSet applies to the core task (maintenanceArena/CoreCycle-v0), whose "
                             f"observation index {DEADLINE_INDEX} is {DEADLINE_NAME}")
        if name not in INPUT_SETS:
            raise ValueError(f"input set must be one of {sorted(INPUT_SETS)}")
        super().__init__(env)
        self.input_set = name

    def observation(self, observation):
        return input_set(observation, self.input_set)


class CalendarBudget(gym.Wrapper):
    """Truncate an episode once its elapsed calendar steps reach ``calendar_steps``.

    The environment's own ``terminated`` takes precedence: a transition that ends the
    episode is reported as terminated, not truncated. After the episode has ended,
    ``step`` raises ``RuntimeError`` until ``reset``.
    """

    def __init__(self, env: gym.Env, calendar_steps: int):
        if isinstance(calendar_steps, (bool, np.bool_)) or not isinstance(calendar_steps, Integral) or calendar_steps < 1:
            raise ValueError("calendar_steps must be an integer >= 1")
        super().__init__(env)
        self.calendar_budget = int(calendar_steps)
        self._elapsed = 0
        self._ended = True

    @property
    def elapsed_calendar_steps(self) -> int:
        """Calendar steps elapsed since the last reset."""
        return self._elapsed

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._elapsed, self._ended = 0, False
        return observation, info

    def step(self, action):
        if self._ended:
            raise RuntimeError("CalendarBudget: the episode has ended; call reset() before step()")
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "duration_calendar_steps" not in info:
            raise ValueError("CalendarBudget needs info['duration_calendar_steps'], which every MaintenanceArena "
                             "environment reports")
        duration = int(info["duration_calendar_steps"])
        if duration <= 0:
            raise ArithmeticError("The environment reported a nonpositive transition duration")
        self._elapsed += duration
        truncated = bool(truncated or (not terminated and self._elapsed >= self.calendar_budget))
        self._ended = bool(terminated or truncated)
        return observation, reward, terminated, truncated, info


__all__ = ["CalendarBudget", "DEADLINE_INDEX", "INPUT_SETS", "INPUT_SET_NAMES", "InputSet", "ZeroDeadline",
           "input_set", "zero_deadline"]
