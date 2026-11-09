"""Resource task variant with lifetime spread and public operating-age records.

The base resource environment, its observation vector and its random streams
are unchanged. This subclass adds independent lifetime-quantile and shape
streams. True residual life is exported only as analysis diagnostics at PM starts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from gymnasium import spaces

from .curves import _normalized_curve
from .resource import COST_FIELDS, LEDGER_FIELDS, FAILURE_HEALTH, ResourceMaintenanceEnv
from .resource_evaluation import CUMULATIVE_METRICS, _snapshot
from .resource_policies import _expected_lead, _next_shift_opportunity, _local_failure_time
from .resource_planning import PlanningPolicyConfig, plan_resource_jobs


FAMILIES = (
    ("normalized_linear", {}),
    ("normalized_power", {"beta_range": (0.65, 0.9)}),
    ("normalized_power", {"beta_range": (1.5, 2.8)}),
    ("normalized_softplus_knee", {"linear_weight_range": (0.1, 0.3),
     "knee_fraction_range": (0.6, 0.85), "softness_range": (0.03, 0.08)}),
)


class ResourceLifetimeEnv(ResourceMaintenanceEnv):
    """Task variant with observed operating age; recorded age is shared policy information."""

    def __init__(self, config=None, render_mode=None):
        self._recorded_age = []
        self._lifecycle_metadata = {}
        self._last_analysis_pm_starts = []
        super().__init__(config, render_mode=render_mode)
        # Append coordinates, leaving the existing prefix order unchanged.
        self.observation_names = (*self.observation_names,
            *(f"asset_{i}.operating_age" for i in range(self.config.n_assets)))
        self.observation_space = spaces.Box(0, np.inf, shape=(len(self.observation_names),), dtype=np.float32)

    def _make_curve(self, asset_id, renewal):
        quantile = float(self._keyed_rng(310, asset_id, renewal).random())
        low, high = self.config.lifetime_range
        lifetime = low + int(math.floor(quantile * (high - low + 1)))
        shape_rng = self._keyed_rng(311, asset_id, renewal)
        family = int(shape_rng.integers(len(FAMILIES)))
        law, parameters = FAMILIES[family]
        damage = _normalized_curve(law, np.arange(lifetime + 1, dtype=float) / lifetime, parameters, shape_rng)
        health = np.exp(-math.log(5.0) * damage)
        health[0], health[-1] = 1.0, FAILURE_HEALTH
        self._lifecycle_metadata[(asset_id, renewal)] = {
            "lifetime_quantile": quantile, "lifetime": lifetime, "family": family,
            "shape_rng_state_after_curve": shape_rng.bit_generator.state,
        }
        return health

    def lifecycle_metadata(self, asset_id, renewal):
        """Analysis/test API; never supplied to a policy or flat observation."""
        return dict(self._lifecycle_metadata[(asset_id, renewal)])

    def _reset_extra(self):
        self._recorded_age = list(self.config.initial_ages)
        self._lifecycle_metadata = {}
        self._last_analysis_pm_starts = []

    def policy_state(self):
        state = super().policy_state()
        for asset, age in zip(state["assets"], self._recorded_age, strict=True):
            asset["operating_age"] = age
        return state

    def _observation(self):
        # The base vector is reconstructed without dispatching to the extended
        # asset dictionaries, so age is appended exactly once at the end.
        state = super().policy_state()
        values = [state[k] for k in ("phase", "on_shift", "hours_to_next_shift", "stock", "inventory_position", "free_crews")]
        for asset in state["assets"]:
            values.extend(asset.values())
        for i in range(self.config.inventory_capacity):
            values.extend((1, state["pipeline"][i]["age"], state["pipeline"][i]["mode"])
                          if i < len(state["pipeline"]) else (0, 0, 0))
        values.extend(state["config"].values())
        values.extend(self._recorded_age)
        return np.asarray(values, dtype=np.float32)

    def action_masks(self):
        # Equivalent 3x(n+1) joint feasibility; avoid repeated decode/space
        # validation in the many-million-hour conventional-policy comparison.
        if not self._has_reset:
            raise RuntimeError("reset must precede action_masks")
        can_start = self.on_shift and self.free_crews > 0 and self._stock > 0
        jobs = [True, *(can_start and a.repair_work_remaining == 0 for a in self._assets)]
        position = self._stock + len(self._pipeline)
        orders = [feasible and position - int(i > 0) < self.config.inventory_capacity
                  for i, feasible in enumerate(jobs)]
        return np.asarray([*jobs, *orders, *orders], dtype=bool)

    def step(self, action):
        before = super().policy_state()
        _, job = self.decode_action(action)
        if not self.action_masks()[int(action)]:
            job = 0
        pm = []
        if job and not before["assets"][job - 1]["failed"]:
            asset = self._assets[job - 1]
            pm.append({"asset": job - 1, "operating_age": self._recorded_age[job - 1],
                       "true_lifetime": len(asset.curve) - 1,
                       "unused_life_hours": len(asset.curve) - 1 - asset.age})
        operating = [not a["failed"] and a["repair_work_remaining"] == 0 and i + 1 != job
                     for i, a in enumerate(before["assets"])]
        _, reward, terminated, truncated, info = super().step(action)
        self._recorded_age = [age + int(ran) for age, ran in zip(self._recorded_age, operating, strict=True)]
        for event in info["events"]:
            if event["type"] == "service_complete":
                self._recorded_age[event["asset"]] = 0
        self._last_analysis_pm_starts = pm
        info["schema_version"] = "resource_lifetime_spread_v1"
        return self._observation(), reward, terminated, truncated, info

    def diagnostics(self):
        return dict(super().diagnostics(),
                    analysis_pm_starts=copy.deepcopy(self._last_analysis_pm_starts))

    def _snapshot_extra(self):
        return {"recorded_age": list(self._recorded_age),
                "lifecycle_metadata": [[list(key), value] for key, value in self._lifecycle_metadata.items()],
                "last_analysis_pm_starts": self._last_analysis_pm_starts}

    def _restore_extra(self, state):
        self._recorded_age = list(state["recorded_age"])
        self._lifecycle_metadata = {tuple(key): value for key, value in state["lifecycle_metadata"]}
        self._last_analysis_pm_starts = state["last_analysis_pm_starts"]


@dataclass(frozen=True, slots=True)
class LifetimePolicyConfig:
    method: str = "age"
    procurement: str = "normal"
    base_stock: int = 2
    age_interval: int = 144
    health_threshold: float = 0.3
    safety_margin_hours: float = 24.0
    planning_horizon: int = 48

    def __post_init__(self):
        if self.method not in ("reactive", "age", "health", "rolling"):
            raise ValueError("Unknown maintenance policy")
        if self.procurement not in ("normal", "deadline_express"):
            raise ValueError("Unknown procurement rule")
        if self.base_stock not in (1, 2, 3) or self.age_interval < 1:
            raise ValueError("Invalid stock or age interval")
        PlanningPolicyConfig(base_stock=self.base_stock, health_threshold=self.health_threshold,
                             safety_margin_hours=self.safety_margin_hours, planning_horizon=self.planning_horizon)


def _demand_time(asset, state, policy):
    if asset["failed"]:
        return 0.0
    if policy.method == "reactive":
        return math.inf
    if policy.method == "age":
        return max(0.0, policy.age_interval - asset["operating_age"])
    if policy.method == "health":
        rate = asset["log_damage_rate"]
        return max(0.0, math.log(max(asset["health"], policy.health_threshold) / policy.health_threshold) / rate) if rate > 0 else math.inf
    return max(0.0, _local_failure_time(asset, state["config"]["failure_health"]) - policy.safety_margin_hours)


def _procurement(state, policy, job):
    if state["inventory_position"] - int(job > 0) >= policy.base_stock:
        return 0
    if policy.procurement == "normal":
        return 1
    lead = _expected_lead(state["config"], 1)
    events = [(_expected_lead(state["config"], int(o["mode"]), float(o["age"])), 0, 1)
              for o in state["pipeline"]]
    for i, asset in enumerate(state["assets"]):
        if i + 1 != job and asset["repair_work_remaining"] == 0:
            demand = _demand_time(asset, state, policy)
            # Age schedules may start before an off-shift gap. Conservatively
            # include the known longest gap in their procurement deadline.
            if policy.method == "age" and not asset["failed"]:
                demand = max(0.0, demand - (state["config"]["shift_period"] - state["config"]["shift_hours"] + state["config"]["pm_work_hours"]))
            if demand < lead:
                events.append((demand, 1, -1))
    balance = state["stock"] - int(job > 0)
    for at, _, change in sorted(events):
        if at >= lead:
            break
        balance += change
        if balance < 0:
            return 2
    return 1


def choose_lifetime_action(env, policy):
    state, mask = env.policy_state(), env.action_masks()
    available = [i for i, a in enumerate(state["assets"]) if a["repair_work_remaining"] == 0]
    failed = sorted((i for i in available if state["assets"][i]["failed"]),
                    key=lambda i: (-state["assets"][i]["failure_wait_hours"], -state["assets"][i]["contribution_per_hour"], i))
    healthy = [i for i in available if not state["assets"][i]["failed"]]
    due = []
    if policy.method == "age":
        # Prefer the last PM start that can finish within the current shift
        # before the age deadline would force waiting through a shift gap.
        # If a crew/part arrives later, an already-due job may still cross it.
        next_slot = _next_shift_opportunity(state)
        context = state["config"]
        if (context["pm_work_hours"] <= context["shift_hours"]
                and state["on_shift"]
                and state["phase"] + 1 + context["pm_work_hours"] > context["shift_hours"]):
            next_slot = context["shift_period"] - state["phase"]
        due = [i for i in healthy if state["assets"][i]["operating_age"] >= policy.age_interval
               or state["assets"][i]["operating_age"] + next_slot > policy.age_interval]
        due.sort(key=lambda i: (-state["assets"][i]["operating_age"], -state["assets"][i]["contribution_per_hour"], i))
    elif policy.method == "health":
        due = [i for i in healthy if state["assets"][i]["health"] <= policy.health_threshold]
        due.sort(key=lambda i: (state["assets"][i]["health"], -state["assets"][i]["contribution_per_hour"], i))
    elif policy.method == "rolling" and state["on_shift"] and state["stock"] > 0 and state["free_crews"] > 0:
        schedule = plan_resource_jobs(state, PlanningPolicyConfig(base_stock=policy.base_stock,
            safety_margin_hours=policy.safety_margin_hours, planning_horizon=policy.planning_horizon))
        # The existing planner decides whether immediate corrective or PM work
        # is economical; preserve that scheduling behavior without forced jobs.
        failed = []
        due = [r["asset"] for r in schedule if r["start"] == 0]
    for job in [*(i + 1 for i in failed + due), 0]:
        if mask[env.encode_action(0, job)]:
            mode = _procurement(state, policy, job)
            action = env.encode_action(mode, job)
            if mask[action]:
                return action
            return env.encode_action(0, job)
    raise AssertionError("WAIT must be feasible")


def run_episode(configuration, policy, seed, windows, trace_path=None, *, action_fn=None):
    """One continuing trajectory; cash and consumed-part accounting as before."""
    started = perf_counter()
    horizon = max(end for _, end in windows)
    env = ResourceLifetimeEnv(dict(configuration, horizon=horizon))
    observation, info = env.reset(seed=seed)
    boundaries = {0, *(point for window in windows for point in window)}
    prefixes = {}
    reward_sum = position_hours = on_hand_hours = decision_seconds = 0.0
    pm_lost = pm_lifetime = pm_age = pm_count = 0
    state = env.policy_state()
    trace = Path(trace_path).open("x", encoding="utf-8") if trace_path else None
    trace_rows = 0

    def capture(t):
        prefixes[t] = dict(ledger=dict(info["cumulative_ledger"]),
            metrics={k: int(info["metrics"][k]) for k in CUMULATIVE_METRICS},
            state=_snapshot(state), reward=reward_sum, position=position_hours, stock=on_hand_hours,
            pm_lost=pm_lost, pm_lifetime=pm_lifetime, pm_age=pm_age, pm_count=pm_count)

    capture(0)
    try:
        for hour in range(horizon):
            before = state
            t0 = perf_counter()
            action = (choose_lifetime_action if action_fn is None else action_fn)(env, policy)
            decision_seconds += perf_counter() - t0
            mode, job = env.decode_action(action)
            on_hand_hours += state["stock"] - int(job > 0)
            position_hours += state["inventory_position"] - int(job > 0) + int(mode > 0)
            observation, reward, terminated, truncated, info = env.step(action)
            if info["invalid_action"] or info["metrics"]["inventory_conservation_residual"] != 0:
                raise AssertionError("Action or stock conservation violated")
            if terminated or truncated != (hour + 1 == horizon) or not np.isfinite(observation).all() or not np.isfinite(reward):
                raise AssertionError("Invalid temporal/finite-value contract")
            reward_sum += reward
            for pm in env.diagnostics()["analysis_pm_starts"]:
                pm_count += 1
                pm_lost += pm["unused_life_hours"]
                pm_lifetime += pm["true_lifetime"]
                pm_age += pm["operating_age"]
            state = env.policy_state()
            if trace and (action or info["events"]):
                trace.write(json.dumps(dict(t=hour, state=before, action=action,
                    ledger=info["ledger"], events=info["events"], analysis_pm_starts=env.diagnostics()["analysis_pm_starts"]), allow_nan=False) + "\n")
                trace_rows += 1
            if hour + 1 in boundaries:
                capture(hour + 1)
        results = []
        for start, end in windows:
            a, b = prefixes[start], prefixes[end]
            hours = end - start
            ledger = {k: b["ledger"][k] - a["ledger"][k] for k in LEDGER_FIELDS}
            counters = {k: b["metrics"][k] - a["metrics"][k] for k in CUMULATIVE_METRICS}
            cash = ledger["net_profit"]
            if not np.isclose(cash, ledger["contribution"] - sum(ledger[k] for k in COST_FIELDS), atol=1e-6, rtol=1e-10):
                raise AssertionError("Cash ledger does not close")
            if not np.isclose((b["reward"] - a["reward"]) * env.config.reward_scale, cash, atol=1e-6, rtol=1e-10):
                raise AssertionError("Reward ledger does not close")
            if counters["operating_asset_hours"] + counters["down_asset_hours"] != env.config.n_assets * hours:
                raise AssertionError("Asset-hour ledger does not close")
            consumed = cash + ledger["initial_stock_cost"] + ledger["parts_cost"] - env.config.part_cost * counters["parts_consumed"]
            book = cash + ledger["initial_stock_cost"] + env.config.part_cost * (b["state"]["inventory_position"] - a["state"]["inventory_position"])
            if not np.isclose(consumed, book, atol=1e-6, rtol=1e-10):
                raise AssertionError("Consumed-part and book-value corrections differ")
            paused = counters["down_asset_hours"] - counters["crew_work_hours"] - counters["failed_waiting_asset_hours"]
            count, lost = b["pm_count"] - a["pm_count"], b["pm_lost"] - a["pm_lost"]
            if paused < 0 or lost < 0 or count != counters["planned_starts"]:
                raise AssertionError("PM or downtime analysis does not close")
            results.append(dict(start=start, end=end, hours=hours, cash_ledger=ledger,
                cumulative_metrics=counters, cash_profit_per_hour=cash/hours,
                consumption_profit=consumed, consumption_profit_per_hour=consumed/hours,
                availability=counters["operating_asset_hours"]/(env.config.n_assets*hours),
                failures_per_1000_hours=1000*counters["failures"]/hours,
                paused_repair_asset_hours=paused, inventory_on_hand_hours=b["stock"]-a["stock"],
                inventory_position_hours=b["position"]-a["position"],
                average_inventory_on_hand=(b["stock"]-a["stock"])/hours,
                average_inventory_position=(b["position"]-a["position"])/hours,
                opening_state=a["state"], closing_state=b["state"],
                analysis_pm=dict(count=count, unused_life_hours=lost,
                    installed_life_hours=b["pm_lifetime"]-a["pm_lifetime"],
                    operating_age_hours=b["pm_age"]-a["pm_age"],
                    mean_unused_life_hours=lost/count if count else None)))
        return dict(seed=int(seed), policy=asdict(policy), windows=results,
                    elapsed_seconds=perf_counter()-started, decision_seconds=decision_seconds,
                    total_decisions=horizon, all_actions_feasible=True,
                    trace=dict(path=str(Path(trace_path).resolve()) if trace_path else None, rows=trace_rows,
                               sampling="nonzero action or event hours"))
    finally:
        if trace:
            trace.close()
        env.close()


LifetimeResourceEnv = ResourceLifetimeEnv


def policy_action(name, env, observation=None, **kwargs):
    if name not in ("reactive", "health", "rolling", "age"):
        raise ValueError(f"unsupported resource-lifetime policy: {name}")
    supplied = dict(kwargs)
    if supplied.pop("method", name) != name:
        raise ValueError("policy name conflicts with its configured lifetime method")
    return choose_lifetime_action(env, LifetimePolicyConfig(method=name, **supplied))
