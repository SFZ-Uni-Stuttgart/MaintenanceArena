"""Transparent maintenance rules using the same information as the RL vector.

The resource rule extrapolates the last observed log-damage slope locally. This
is an urgency proxy, not a learned prognostic model or access to true lifetime.
Its expected-delivery calculation uses the configured uniform lead-time bounds
and the observable age of outstanding orders, never their realized due dates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .resource import ResourceMaintenanceEnv


@dataclass(frozen=True, slots=True)
class ResourcePolicyConfig:
    kind: str = "reactive"
    base_stock: int = 2
    health_threshold: float = 0.4
    safety_margin_hours: float = 8.0

    def __post_init__(self) -> None:
        if self.kind not in ("reactive", "health_threshold", "resource_aware"):
            raise ValueError("unknown resource policy kind")
        if isinstance(self.base_stock, bool) or not isinstance(self.base_stock, int) or self.base_stock < 1:
            raise ValueError("base_stock must be a positive integer")
        if not math.isfinite(self.health_threshold) or not 0 < self.health_threshold < 1:
            raise ValueError("health_threshold must be finite and between zero and one")
        if not math.isfinite(self.safety_margin_hours) or self.safety_margin_hours < 0:
            raise ValueError("safety_margin_hours must be finite and nonnegative")


def _local_failure_time(asset: dict[str, Any], failure_health: float) -> float:
    """Operating hours to a known failure threshold under constant local rate."""
    if asset["failed"]:
        return 0.0
    rate = float(asset["log_damage_rate"])
    if rate <= 0:
        return math.inf
    return max(0.0, math.log(max(float(asset["health"]), failure_health) / failure_health) / rate)


def _expected_lead(context: dict[str, float], mode: int, age: float = 0.0) -> float:
    prefix = "normal" if mode == 1 else "express"
    low = context[f"{prefix}_lead_range_min"]
    high = context[f"{prefix}_lead_range_max"]
    # An order still in the pipeline at integer age a has lead strictly > a.
    conditional_low = max(low, math.floor(age) + 1)
    return max(0.0, (min(conditional_low, high) + high) / 2.0 - age)


def _elapsed_work(work: float, phase: float, shift: float, period: float) -> float:
    """Calendar time to finish a known workload, pausing outside the shift."""
    delay = 0.0
    if phase >= shift:
        delay, phase = period - phase, 0.0
    first_shift = shift - phase
    if work <= first_shift:
        return delay + work
    work -= first_shift
    delay += first_shift + period - shift
    full_cycles = max(0, math.ceil(work / shift) - 1)
    return delay + full_cycles * period + work - full_cycles * shift


def _crew_start_delay(state: dict[str, Any]) -> float:
    if state["free_crews"] > 0:
        return float(state["hours_to_next_shift"])
    context = state["config"]
    phase, shift, period = float(state["phase"]), context["shift_hours"], context["shift_period"]
    work = min(float(a["repair_work_remaining"]) for a in state["assets"] if a["repair_work_remaining"] > 0)
    elapsed = _elapsed_work(work, phase, shift, period)
    completion_phase = (phase + elapsed) % period
    return elapsed + (period - completion_phase if completion_phase >= shift else 0.0)


def _next_shift_opportunity(state: dict[str, Any]) -> float:
    """Delay to the next decision on shift if the current hour is skipped."""
    context = state["config"]
    next_phase = (float(state["phase"]) + 1.0) % context["shift_period"]
    return 1.0 + (context["shift_period"] - next_phase if next_phase >= context["shift_hours"] else 0.0)


def _job_priorities(state: dict[str, Any], policy: ResourcePolicyConfig) -> list[int]:
    assets = state["assets"]
    candidates = [i for i, a in enumerate(assets) if a["repair_work_remaining"] == 0]
    failed = sorted(
        (i for i in candidates if assets[i]["failed"]),
        key=lambda i: (-assets[i]["failure_wait_hours"], -assets[i]["contribution_per_hour"], i),
    )
    if policy.kind == "reactive":
        return [i + 1 for i in failed]
    healthy = [i for i in candidates if not assets[i]["failed"]]
    if policy.kind == "health_threshold":
        due = [i for i in healthy if assets[i]["health"] <= policy.health_threshold]
        due.sort(key=lambda i: (assets[i]["health"], -assets[i]["contribution_per_hour"], i))
    else:
        context = state["config"]
        if state["stock"] > 0:
            part_delay = 0.0
        else:
            part_delay = min(
                (_expected_lead(context, int(order["mode"]), float(order["age"])) for order in state["pipeline"]),
                default=_expected_lead(context, 1),
            )
        # Parts and crew become available in parallel. The repair's own
        # duration is not added: degradation stops when the job starts.
        budget = max(_crew_start_delay(state), part_delay, _next_shift_opportunity(state)) + policy.safety_margin_hours
        times = {i: _local_failure_time(assets[i], context["failure_health"]) for i in healthy}
        due = [i for i in healthy if times[i] <= budget]
        due.sort(key=lambda i: (times[i], -assets[i]["contribution_per_hour"], i))
    return [i + 1 for i in failed + due]


def _order_mode(state: dict[str, Any], policy: ResourcePolicyConfig, job: int) -> int:
    context = state["config"]
    # The selected job consumes an existing spare in this same joint action.
    target = min(policy.base_stock, int(context["inventory_capacity"]))
    if state["inventory_position"] - int(job > 0) >= target:
        return 0
    if policy.kind != "resource_aware":
        return 1
    normal_lead = _expected_lead(context, 1)
    if _expected_lead(context, 2) >= normal_lead:
        return 1
    # Forecast a stock shortfall before a new normal order would arrive.
    # Existing deliveries at the same expected time precede spare demand.
    events: list[tuple[float, int, int]] = [
        (_expected_lead(context, int(order["mode"]), float(order["age"])), 0, 1)
        for order in state["pipeline"]
    ]
    for i, asset in enumerate(state["assets"]):
        if i + 1 == job or asset["repair_work_remaining"] > 0:
            continue
        time = _local_failure_time(asset, context["failure_health"])
        demand_time = max(0.0, time - policy.safety_margin_hours)
        if demand_time < normal_lead:
            events.append((demand_time, 1, -1))
    balance = int(state["stock"]) - int(job > 0)
    for time, _, change in sorted(events):
        if time >= normal_lead:
            break
        balance += change
        if balance < 0:
            return 2
    return 1


def choose_resource_action(env: ResourceMaintenanceEnv, config: ResourcePolicyConfig) -> int:
    """Choose a feasible joint order/job action, with WAIT as the fallback.

    All three rules share base-stock replenishment. Reactive and threshold
    policies use normal orders; the resource rule expedites a locally predicted
    stock shortfall. Jobs always take priority over placing a simultaneous order.
    No policy reads a lifecycle, sampled delivery date, or episode time limit.
    """
    state = env.policy_state()
    mask = env.action_masks()
    for job in [*_job_priorities(state, config), 0]:
        if not mask[env.encode_action(0, job)]:
            continue
        desired = _order_mode(state, config, job)
        # Express may be unavailable in future extensions; preserve the job
        # and use a normal order before deferring replenishment entirely.
        modes = (2, 1, 0) if desired == 2 else (desired, 0)
        for mode in modes:
            action = env.encode_action(mode, job)
            if mask[action]:
                return action
    raise RuntimeError("resource environment mask must permit WAIT")
