"""Gymnasium adapter for the exact, continuing gearbox process.

One step completes a decision macro. Its reward discounts each primitive cash
flow in calendar time; ``info['gamma_pow_duration']`` discounts the successor.
Applying an ordinary fixed-discount learner to these macros changes the objective.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from . import gearbox_model as model_api
from .gearbox_filter import build_filter

Action = model_api.Action
OBSERVATION_NAMES = ("health", "reference_damage_rate", "opportunity", "intervention_slack")


@lru_cache(maxsize=1)
def exact_model():
    """The immutable exact transition kernel, shared without constructing a learner."""
    model = model_api.build_exact_model()
    for value in (model.reward_raw, model.duration, model.renews,
                  model.next_damage, model.immediate_metrics):
        value.flags.writeable = False
    return model


def primitive_rewards(model, damage: int, opportunity: int, action: int) -> tuple[int, ...]:
    """Exact integer-cent cash flow of each primitive step within one decision macro."""
    if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
            or action not in (0, 1, 2) or not 0 <= damage < model.spec.damage_capacity
            or opportunity not in (0, 1)):
        raise ValueError("Invalid gearbox state/action")
    spec = model.spec
    index = (damage, opportunity, action)
    metric = model.immediate_metrics[index]
    if action in (0, 1) and not model.renews[index]:
        revenue = (spec.normal_high_revenue_cents if action == 0 and opportunity == 1
                   else spec.normal_low_revenue_cents if action == 0 else spec.derate_revenue_cents)
        rows = (revenue,)
    elif action in (0, 1):
        rows = (-(spec.failure_cost_cents + spec.replacement_cost_cents + spec.downtime_cost_cents),
                *(-spec.downtime_cost_cents for _ in range(spec.post_event_downtime_rows)))
    elif metric[model_api.METRIC_INDEX["planned_renewals"]] == 1:
        rows = (*(spec.derate_revenue_cents for _ in range(spec.mobilization_epochs)),
                -(spec.replacement_cost_cents + spec.downtime_cost_cents),
                *(-spec.downtime_cost_cents for _ in range(spec.post_event_downtime_rows)))
    else:
        productive = int(metric[model_api.METRIC_INDEX["productive_epochs"]])
        rows = (*(spec.derate_revenue_cents for _ in range(productive)),
                -(spec.failure_cost_cents + spec.replacement_cost_cents + spec.downtime_cost_cents),
                *(-spec.downtime_cost_cents for _ in range(spec.post_event_downtime_rows)))
    if len(rows) != model.duration[index] or math.fsum(rows) != metric[0]:
        raise ArithmeticError("Primitive gearbox schedule differs from exact kernel")
    return tuple(int(x) for x in rows)


class GearboxEnv(gym.Env):
    """Unfiltered continuing NORMAL/DERATE/REPLACE task with a real renewal successor."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    observation_names = OBSERVATION_NAMES
    environment_id = "maintenanceArena/Gearbox-v0"

    def __init__(self, config: Mapping[str, Any] | None = None, render_mode=None):
        super().__init__()
        values = dict(config or {})
        unknown = set(values) - {"horizon", "gamma", "reward_scale"}
        if unknown:
            raise ValueError(f"Unknown gearbox settings: {sorted(unknown)}")
        self.config = {"horizon": None, "gamma": .999, "reward_scale": 1000., **values}
        horizon = self.config["horizon"]
        if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1):
            raise ValueError("horizon must be a positive integer or None")
        if self.config["gamma"] != .999 or self.config["reward_scale"] != 1000.:
            raise ValueError("The published gearbox task fixes gamma=.999 and reward_scale=1000")
        if render_mode not in (None, "ansi"):
            raise ValueError("Gearbox supports only ansi rendering")
        self.render_mode = render_mode
        self.model = exact_model()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.asarray([.2, 0., 0., -1.], dtype=np.float32),
                                            np.asarray([1., 1., 1., 1.], dtype=np.float32))
        self._ready = False
        self._done = False

    def _observation(self):
        return model_api.observation(self.model.spec, self._damage, self._opportunity)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if set(options) - {"damage", "opportunity"}:
            raise ValueError("Only explicit damage/opportunity start diagnostics are supported")
        damage = options.get("damage", 0)
        if (isinstance(damage, (bool, np.bool_)) or not isinstance(damage, (int, np.integer))
                or not 0 <= damage < self.model.spec.damage_capacity):
            raise ValueError("Invalid initial damage")
        if "opportunity" in options:
            opportunity = options["opportunity"]
            if (isinstance(opportunity, (bool, np.bool_)) or not isinstance(opportunity, (int, np.integer))
                    or opportunity not in (0, 1)):
                raise ValueError("Invalid initial opportunity")
        super().reset(seed=seed)
        self._damage = int(damage)
        self._opportunity = int(self.np_random.integers(0, 2))
        if "opportunity" in options:
            self._opportunity = int(options["opportunity"])
        self._calendar = self._decisions = self._renewals = self._failures = 0
        self._raw_profit = 0.
        self._cumulative_metrics = np.zeros(len(model_api.METRIC_NAMES), dtype=np.float64)
        self._ready, self._done = True, False
        return self._observation(), self._info(0, 0., [], {}, False)

    def policy_state(self):
        if not self._ready:
            raise RuntimeError("reset must precede reading state")
        obs = self._observation()
        return {"damage": int(self._damage), "remaining_damage": self.model.spec.damage_capacity - int(self._damage),
                "opportunity": int(self._opportunity), **dict(zip(OBSERVATION_NAMES, map(float, obs)))}

    def diagnostics(self):
        if not self._ready:
            raise RuntimeError("reset must precede diagnostics")
        return {"calendar_steps": self._calendar, "decisions": self._decisions,
                "renewals": self._renewals, "failures": self._failures,
                "raw_profit": self._raw_profit, "state": self.policy_state(),
                "cumulative_metrics": dict(zip(model_api.METRIC_NAMES, self._cumulative_metrics.tolist()))}

    def _info(self, duration, raw, events, ledger, renewal):
        return {"schema_version": "maintenance_arena_gearbox_v1", "duration_calendar_steps": duration,
                "gamma_pow_duration": self.config["gamma"] ** duration, "raw_profit_sum": raw,
                "events": events, "ledger": ledger, "renewal": renewal,
                "calendar_steps": self._calendar, "metrics": self.diagnostics(),
                "calendar_budget": self.config["horizon"],
                "calendar_overshoot": max(0, self._calendar - (self.config["horizon"] or self._calendar))}

    def step(self, action):
        if not self._ready or self._done:
            raise RuntimeError("reset is required before stepping or after the accounting cutoff")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("Action must be NORMAL=0, DERATE=1 or REPLACE=2")
        action = int(action)
        damage, opportunity, start = self._damage, self._opportunity, self._calendar
        index = (damage, opportunity, action)
        cents = primitive_rewards(self.model, damage, opportunity, action)
        duration = len(cents)
        metric = self.model.immediate_metrics[index]
        get = lambda name: float(metric[model_api.METRIC_INDEX[name]])
        renewal = bool(self.model.renews[index])
        raw = float(math.fsum(value / 100. for value in cents))
        reward = float(math.fsum(self.config["gamma"] ** offset * (value / 100.) / self.config["reward_scale"]
                                 for offset, value in enumerate(cents)))
        ledger = {"revenue": sum(get(name) for name in ("revenue_low_cents", "revenue_high_cents", "revenue_mobilization_cents")) / 100.,
                  "replacement_cost": get("replacement_count") * self.model.spec.replacement_cost_cents / 100.,
                  "downtime_cost": get("downtime_rows") * self.model.spec.downtime_cost_cents / 100.,
                  "failure_cost": get("failures") * self.model.spec.failure_cost_cents / 100., "net_profit": raw}
        productive = int(get("productive_epochs"))
        events = [{"type": "decision", "calendar_step": start, "action": action, "damage": damage, "opportunity": opportunity}]
        if renewal:
            events.append({"type": "failure" if get("failures") else "exchange", "calendar_step": start + productive,
                           "installed_damage": 0, "planned": bool(get("planned_renewals"))})
            events.append({"type": "renewal_ready", "calendar_step": start + duration})
        self._damage = 0 if renewal else int(self.model.next_damage[index])
        # Exactly one actual successor draw, including renewal. No placeholder
        # is evaluated or re-drawn by a collector.
        self._opportunity = int(self.np_random.integers(0, 2))
        self._calendar += duration
        self._decisions += 1
        self._renewals += int(renewal)
        self._failures += int(get("failures"))
        self._raw_profit += raw
        self._cumulative_metrics += metric
        horizon = self.config["horizon"]
        self._done = horizon is not None and self._calendar >= horizon
        info = self._info(duration, raw, events, ledger, renewal)
        info.update(primitive_reward_cents=list(cents), planned_renewals=int(get("planned_renewals")),
                    corrective_renewals=int(get("corrective_renewals")), failures=int(get("failures")),
                    calendar_step_start=start, calendar_step_end=self._calendar)
        return self._observation(), reward, False, self._done, info

    def snapshot(self):
        if not self._ready:
            raise RuntimeError("Cannot snapshot before reset")
        return {"schema_version": "maintenance_arena_gearbox_snapshot_v1", "config": deepcopy(self.config),
                "rng_state": deepcopy(self.np_random.bit_generator.state), "damage": self._damage,
                "opportunity": self._opportunity, "calendar": self._calendar, "decisions": self._decisions,
                "renewals": self._renewals, "failures": self._failures, "raw_profit": self._raw_profit,
                "cumulative_metrics": self._cumulative_metrics.tolist(), "done": self._done}

    def restore(self, state):
        if state.get("schema_version") != "maintenance_arena_gearbox_snapshot_v1" or state.get("config") != self.config:
            raise ValueError("Snapshot contract/configuration mismatch")
        model_api.observation(self.model.spec, state["damage"], state["opportunity"])
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = deepcopy(state["rng_state"])
        metrics = np.asarray(state["cumulative_metrics"], dtype=np.float64)
        if metrics.shape != (len(model_api.METRIC_NAMES),) or not np.isfinite(metrics).all():
            raise ValueError("Invalid snapshot metrics")
        self._np_random = rng
        for name in ("damage", "opportunity", "calendar", "decisions", "renewals", "failures", "raw_profit", "done"):
            setattr(self, "_" + name, deepcopy(state[name]))
        self._cumulative_metrics = metrics.copy()
        self._ready = True
        return self._observation()

    def render(self):
        if self.render_mode != "ansi":
            return None
        if not self._ready:
            return "Gearbox: reset required"
        state = self.policy_state()
        return (f"calendar={self._calendar}, remaining_damage={state['remaining_damage']}, "
                f"opportunity={int(state['opportunity'])}, renewals={self._renewals}, profit={self._raw_profit:.2f} CU")


@lru_cache(maxsize=1)
def _dp_policy():
    return model_api.solve_exact_dp(exact_model()).policy


@lru_cache(maxsize=1)
def _filter_map():
    return build_filter(exact_model())[0]


def filtered_action(env: GearboxEnv, proposed_action: int) -> int:
    """Correct the already-selected proposal; report intervention outside this helper."""
    if isinstance(proposed_action, (bool, np.bool_)) or not env.action_space.contains(proposed_action):
        raise ValueError("Invalid proposed gearbox action")
    state = env.policy_state()
    return int(_filter_map()[state["damage"], int(state["opportunity"]), int(proposed_action)])


def policy_action(name: str, env: GearboxEnv, observation=None, **kwargs) -> int:
    """Frozen exact-state references; filtering is an explicit execution choice."""
    state = env.policy_state()
    q, opportunity = int(state["remaining_damage"]), int(state["opportunity"])
    if name in ("dp", "average_profit_optimal"):
        proposed = int(_dp_policy()[state["damage"], opportunity])
    elif name in ("reactive", "reactive_normal"):
        proposed = 0
    elif name == "reactive_derate":
        proposed = 1
    elif name in ("normal_q96", "normal_threshold"):
        proposed = 2 if q <= 96 else 0
    elif name in ("derate_q89", "derate_threshold"):
        proposed = 2 if q <= 89 else 1
    elif name in ("separable", "separable_q96", "projection"):
        proposed = 2 if q <= 96 else (0 if opportunity else 1)
    else:
        raise ValueError(f"Unknown gearbox reference: {name}")
    if kwargs.get("filtered", False):
        return int(_filter_map()[state["damage"], opportunity, proposed])
    return proposed
