"""Gymnasium replay of recorded bearing lives with duration-aware WAIT/ORDER transitions.

The environment needs prepared policy inputs, so it is not registered as a Gymnasium ID;
construct ``BearingReplayEnv`` directly.
"""
from __future__ import annotations

from collections.abc import Sequence
import copy
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .economics import ORDER, WAIT, ReplayEconomics, ReplayOutcome
from .features import OBSERVATION_BOUNDS


class BearingReplayEnv(gym.Env[np.ndarray, int]):
    """One recorded bearing life per episode, decided minute by minute.

    The observation at decision minute t is row t - 1 of the bearing's input matrix, whose
    row count is the last productive minute T. Actions are 0 WAIT and 1 ORDER. A WAIT
    before T lasts one calendar minute; an ORDER, or a WAIT at T, runs through preparation,
    exchange and downtime in one terminal transition, as in ``CoreCycleEnv``, and the
    returned observation then repeats the last decision's input. The reward is the
    discounted cash of the transition divided by ``reward_scale``. The reset books the
    first recorded minute, which elapses before the first decision and enters no reward,
    so the reset and step ledgers of an episode add up to ``ReplayEconomics.outcome``.
    Each reset draws a bearing uniformly with the environment's generator unless
    ``options={"bearing": i}`` names one. Observations and ``info`` never contain T;
    ``diagnostics`` is a separate evaluator interface.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, observations: Sequence[np.ndarray], *, names: Sequence[str] | None = None,
                 observation_names: Sequence[str] | None = None, economics: ReplayEconomics | None = None,
                 gamma: float = 0.999, reward_scale: float = 1000.0, render_mode: str | None = None):
        super().__init__()
        if isinstance(observations, np.ndarray) or len(observations) == 0:
            raise ValueError("observations must be a nonempty sequence of (T, n) input matrices")
        banks = [np.ascontiguousarray(np.asarray(x, dtype=np.float32)) for x in observations]
        width = banks[0].shape[1] if banks[0].ndim == 2 else 0
        for bank in banks:
            if bank.ndim != 2 or len(bank) == 0 or bank.shape[1] != width or width == 0 or not np.isfinite(bank).all():
                raise ValueError("every input matrix must be finite, nonempty and of equal width")
        self.names = tuple(str(i) for i in range(len(banks))) if names is None else tuple(str(n) for n in names)
        if len(self.names) != len(banks) or len(set(self.names)) != len(banks):
            raise ValueError("names must be unique, one per input matrix")
        self.observation_names = None if observation_names is None else tuple(observation_names)
        if self.observation_names is None:
            low, high = np.full(width, -np.inf, dtype=np.float32), np.full(width, np.inf, dtype=np.float32)
        else:
            if len(self.observation_names) != width or set(self.observation_names) - set(OBSERVATION_BOUNDS):
                raise ValueError(f"observation_names must name the {width} input columns")
            low = np.asarray([OBSERVATION_BOUNDS[n][0] for n in self.observation_names], dtype=np.float32)
            high = np.asarray([OBSERVATION_BOUNDS[n][1] for n in self.observation_names], dtype=np.float32)
            if any(np.any(bank < low) or np.any(bank > high) for bank in banks):
                raise ValueError("an input matrix lies outside the bounds of its named columns")
        if not 0 < gamma <= 1 or not reward_scale > 0:
            raise ValueError("gamma must lie in (0, 1] and reward_scale must be positive")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or ansi")
        self.economics = economics or ReplayEconomics()
        self.gamma, self.reward_scale, self.render_mode = float(gamma), float(reward_scale), render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._banks = banks
        self._needs_reset = True
        self._initialized = False

    @property
    def lives(self) -> tuple[int, ...]:
        """Last productive minute of every history; evaluator information, never a policy input."""
        return tuple(len(bank) for bank in self._banks)

    def _index(self, value) -> int:
        if isinstance(value, str):
            if value not in self.names:
                raise ValueError(f"unknown bearing {value!r}")
            return self.names.index(value)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not 0 <= value < len(self._banks)):
            raise ValueError("bearing must be a name or an index of the input matrices")
        return int(value)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        requested = dict(options or {})
        if set(requested) - {"bearing"}:
            raise ValueError("reset options only support bearing")
        self._bearing = (self._index(requested["bearing"]) if "bearing" in requested
                         else int(self.np_random.integers(len(self._banks))))
        self._life = len(self._banks[self._bearing])
        self._minute = self._decision_minute = 1
        self._decisions = 0
        self._discounted_return = 0.0
        self._failures = self._planned = self._corrective = 0
        self._first_order: int | None = None
        self._order_pending = self._corrective_pending = self._renewed = False
        self._mobilization_remaining = self._downtime_remaining = 0
        revenue = self.economics.revenue
        first = {"calendar_step": 1, "event": "operate", "order_placed": False, "revenue": revenue,
                 "replacement_cost": 0.0, "downtime_cost": 0.0, "failure_cost": 0.0, "raw_profit": revenue,
                 "planned_replacement": False, "corrective_replacement": False, "failure": False}
        self._calendar, self._raw_profit = 1, revenue
        self._trace: list[dict[str, Any]] = [dict(first)]
        self._needs_reset, self._initialized = False, True
        return self._banks[self._bearing][0].copy(), self._info([first])

    def _exchange(self, corrective: bool) -> None:
        self._corrective += int(corrective)
        self._planned += int(not corrective)
        self._order_pending = self._corrective_pending = False
        self._mobilization_remaining = 0
        self._downtime_remaining = self.economics.downtime_steps
        self._renewed = True

    def _primitive(self, action: int) -> dict[str, Any]:
        """One calendar minute, in the event order of ``CoreCycleEnv``."""
        econ = self.economics
        event, order = "operate", False
        revenue = replacement = downtime = failure_cost = 0.0
        failure = planned = corrective = False
        if self._downtime_remaining > 0:
            self._downtime_remaining -= 1
            downtime, event = econ.nonproductive_cost, "downtime"
        elif self._order_pending and self._mobilization_remaining == 0:
            corrective = self._corrective_pending
            planned = not corrective
            replacement, downtime = econ.replacement_cost, econ.nonproductive_cost
            event = "corrective_exchange" if corrective else "planned_exchange"
            self._exchange(corrective)
        elif self._corrective_pending:
            self._mobilization_remaining -= 1
            downtime, event = econ.nonproductive_cost, "corrective_mobilize"
        else:
            if not self._order_pending and action == ORDER:
                self._order_pending, order = True, True
                self._mobilization_remaining = econ.preparation
                if self._first_order is None:
                    self._first_order = self._minute
            if self._minute >= self._life:
                failure = True
                downtime, failure_cost = econ.nonproductive_cost, econ.failure_cost
                self._failures += 1
                if econ.recovery_contract == "preserve_preparation_v1":
                    event = "failure_during_mobilization" if self._order_pending else "failure"
                    if not self._order_pending:
                        self._order_pending = True
                        self._mobilization_remaining = econ.preparation
                    self._corrective_pending = True
                    # Detection consumes the first remaining preparation step.
                    self._mobilization_remaining -= 1
                else:
                    event = "failure_during_mobilization" if self._order_pending else "corrective_exchange"
                    corrective, replacement = True, econ.replacement_cost
                    self._exchange(True)
            else:
                self._minute += 1
                revenue = econ.revenue
                if self._order_pending:
                    self._mobilization_remaining -= 1
                    event = "mobilize"
        self._calendar += 1
        return {"calendar_step": self._calendar, "event": event, "order_placed": order, "revenue": revenue,
                "replacement_cost": replacement, "downtime_cost": downtime, "failure_cost": failure_cost,
                "raw_profit": revenue - replacement - downtime - failure_cost,
                "planned_replacement": planned, "corrective_replacement": corrective, "failure": failure}

    def _info(self, ledger: list[dict[str, Any]]) -> dict[str, Any]:
        duration = len(ledger)
        return {"schema_version": "maintenance_arena_bearing_replay_v1", "objective": "one_life",
                "duration_calendar_steps": duration, "gamma_pow_duration": self.gamma ** duration,
                "raw_profit_sum": float(math.fsum(row["raw_profit"] for row in ledger)),
                "events": [{"calendar_step": row["calendar_step"], "event": row["event"],
                            "order_placed": row["order_placed"]} for row in ledger],
                "ledger": ledger, "calendar_steps": self._calendar, "decision_steps": self._decisions,
                "episode_raw_profit": self._raw_profit, "episode_discounted_return": self._discounted_return,
                "failures": self._failures, "planned_replacements": self._planned,
                "corrective_replacements": self._corrective}

    def step(self, action: int):
        if self._needs_reset:
            raise RuntimeError("the bearing replay requires reset before step")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"invalid bearing replay action {action!r}")
        start = self._calendar
        self._decision_minute = self._minute
        ledger = [self._primitive(int(action))]
        while self._order_pending or self._downtime_remaining > 0:
            ledger.append(self._primitive(WAIT))
        reward = math.fsum(self.gamma ** i * (row["raw_profit"] / self.reward_scale) for i, row in enumerate(ledger))
        self._decisions += 1
        self._raw_profit += math.fsum(row["raw_profit"] for row in ledger)
        # Rewards are discounted to the first decision, which follows the first minute.
        self._discounted_return += self.gamma ** (start - 1) * reward
        self._trace.extend(dict(row) for row in ledger)
        terminated = self._renewed
        self._needs_reset = terminated
        row = self._decision_minute if terminated else self._minute
        return self._banks[self._bearing][row - 1].copy(), float(reward), terminated, False, self._info(ledger)

    def policy_state(self) -> dict[str, Any]:
        """Current decision minute, the causal clock that the age input encodes."""
        if not self._initialized:
            raise RuntimeError("the bearing replay has not been reset")
        return {"minute": self._minute}

    def episode_outcome(self) -> ReplayOutcome:
        """Closed-form outcome of the finished episode; evaluator information that reveals T."""
        if not self._initialized or not self._renewed:
            raise RuntimeError("the episode has not ended")
        return self.economics.outcome(self._life, self._first_order)

    def diagnostics(self) -> dict[str, Any]:
        """Evaluator sidecar with the bearing and its last productive minute; never a policy input."""
        if not self._initialized:
            raise RuntimeError("the bearing replay has not been reset")
        outcome = self.episode_outcome() if self._renewed else None
        return {"bearing": self.names[self._bearing], "index": self._bearing,
                "last_productive_minute": self._life, "last_safe_order": self._life - self.economics.preparation,
                "minute": self._minute, "order_minute": self._first_order, "terminated": self._renewed,
                "calendar_steps": self._calendar, "raw_profit": self._raw_profit, "failures": self._failures,
                "planned_replacements": self._planned, "corrective_replacements": self._corrective,
                "outcome": None if outcome is None else outcome.as_dict(), "trace": copy.deepcopy(self._trace)}

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        if not self._initialized:
            return "Bearing replay: reset required"
        return f"minute={self._minute} calendar={self._calendar} profit={self._raw_profit:.2f}"
