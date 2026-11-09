"""Observable rolling schedules and independently selectable procurement rules.

The planner enumerates the next intervention of each current bearing. Its local
log-rate forecast is a surrogate, not simulator truth. With local full-life proxy
L=log(1/failure_health)/rate and buffered deadline F=max(1,ceil(R-margin)), a
healthy bearing left at H has renewal liability K*clip((L-F+H)/L,0,1), where
K=part+PM setup+(contribution+extra-down)*PM work hours. A renewed bearing has
liability K*clip((H-completion)/L,0,1). This proxy prices discarded residual life.
A failed bearing left at H instead carries a corrective debt: one part, CM
setup, and full waiting/repair downtime from H with shift pauses. This terminal
continuation optimistically relaxes unknown future stock and crew competition.
Scheduled repair downtime is charged through completion, including beyond H.

Known jobs retain their crews until completion. Supply consists only of current
stock and observed outstanding orders at conditional mean arrival times. The
planner does not invent future orders or subsequent failures/renewals. Fixed
shift wages cancel from scheduling; parts are valued at replacement opportunity
cost even when already purchased. These are planning costs, not the cash ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TYPE_CHECKING

from .resource_policies import (
    ResourcePolicyConfig, _elapsed_work, _expected_lead, _job_priorities,
    _local_failure_time, _order_mode,
)

if TYPE_CHECKING:
    from .resource import ResourceMaintenanceEnv


@dataclass(frozen=True, slots=True)
class PlanningPolicyConfig:
    trigger: str = "rolling_plan"
    procurement: str = "normal"
    base_stock: int = 2
    health_threshold: float = 0.3
    safety_margin_hours: float = 24.0
    planning_horizon: int = 48
    procurement_margin_hours: float = 24.0

    def __post_init__(self) -> None:
        if self.trigger not in ("reactive", "health_threshold", "local_trend", "rolling_plan"):
            raise ValueError("unknown planning trigger")
        if self.procurement not in ("normal", "risk_express"):
            raise ValueError("unknown procurement rule")
        ResourcePolicyConfig(
            base_stock=self.base_stock, health_threshold=self.health_threshold,
            safety_margin_hours=self.safety_margin_hours,
        )
        if (isinstance(self.planning_horizon, bool)
                or not isinstance(self.planning_horizon, int) or self.planning_horizon < 1):
            raise ValueError("planning_horizon must be a positive integer")
        if not math.isfinite(self.procurement_margin_hours) or self.procurement_margin_hours < 0:
            raise ValueError("procurement_margin_hours must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class _Job:
    asset: int
    start: int
    finish: int
    saving: float


def _completion(start: int, work: float, state: dict[str, Any]) -> int:
    context = state["config"]
    phase = (state["phase"] + start) % context["shift_period"]
    return start + int(_elapsed_work(work, phase, context["shift_hours"], context["shift_period"]))


def _job_candidates(state: dict[str, Any], config: PlanningPolicyConfig) -> list[list[_Job]]:
    """Positive savings versus no intervention, with an explicit terminal debt."""
    context, horizon = state["config"], config.planning_horizon
    shift, period = context["shift_hours"], context["shift_period"]
    starts = [t for t in range(horizon) if (state["phase"] + t) % period < shift]
    terminal_cm_finish = _completion(horizon, context["cm_work_hours"], state)
    result: list[list[_Job]] = []
    for index, asset in enumerate(state["assets"]):
        if asset["repair_work_remaining"] > 0:
            continue
        rate = float(asset["log_damage_rate"])
        full_life = math.log(1 / context["failure_health"]) / rate if rate > 0 else math.inf
        remaining = _local_failure_time(asset, context["failure_health"])
        deadline = (0 if asset["failed"] else
                    max(1, math.ceil(remaining - config.safety_margin_hours)) if math.isfinite(remaining)
                    else math.inf)
        loss = asset["contribution_per_hour"] + context["extra_down_cost_per_asset_hour"]
        cycle_cost = context["part_cost"] + context["pm_setup_cost"] + loss * context["pm_work_hours"]

        def liability(age: float) -> float:
            return cycle_cost * max(0.0, min(1.0, age / full_life)) if math.isfinite(full_life) else 0.0

        failure_charge = 0.0 if asset["failed"] else context["failure_event_cost"]
        if deadline <= horizon:
            no_job_cost = (failure_charge + loss * (terminal_cm_finish - deadline)
                           + context["part_cost"] + context["cm_setup_cost"])
        else:
            no_job_cost = liability(full_life - deadline + horizon) if math.isfinite(full_life) else 0.0
        candidates = []
        for start in starts:
            corrective = deadline <= start
            work = context["cm_work_hours"] if corrective else context["pm_work_hours"]
            finish = _completion(start, work, state)
            downtime = finish - start + (start - deadline if corrective else 0)
            cost = (context["part_cost"]
                    + context["cm_setup_cost" if corrective else "pm_setup_cost"]
                    + loss * downtime + (failure_charge if corrective else 0.0)
                    + liability(max(0, horizon - finish)))
            saving = no_job_cost - cost
            if saving > 1e-9:
                candidates.append(_Job(index, start, finish, saving))
        # High-benefit schedules provide an early branch-and-bound incumbent.
        candidates.sort(key=lambda job: (-job.saving, -job.start, job.asset))
        if candidates:
            result.append(candidates)
    result.sort(key=lambda candidates: (-candidates[0].saving, candidates[0].asset))
    return result


def plan_resource_jobs(state: dict[str, Any], config: PlanningPolicyConfig) -> list[dict[str, Any]]:
    """Return a deterministic joint schedule without modifying observable state.

    Start slots cover every on-shift integer hour below the fixed window. The
    small-fleet enumeration is exact for this one-intervention surrogate; it is
    intentionally bounded to six current assets. It is not a global maintenance
    optimum or a forecast of actual sampled deliveries/degradation.
    """
    if len(state["assets"]) > 6:
        raise ValueError("rolling enumeration supports at most six assets")
    context = state["config"]
    supply = sorted([0] * int(state["stock"]) + [
        math.ceil(_expected_lead(context, int(order["mode"]), float(order["age"])))
        for order in state["pipeline"]
    ])
    if not supply:
        return []
    busy = [(0, _completion(0, a["repair_work_remaining"], state))
            for a in state["assets"] if a["repair_work_remaining"] > 0]
    candidates = _job_candidates(state, config)
    upper = [0.0] * (len(candidates) + 1)
    for index in range(len(candidates) - 1, -1, -1):
        upper[index] = upper[index + 1] + candidates[index][0].saving
    best: list[_Job] = []
    best_value = 0.0
    best_tie: tuple[Any, ...] = ()

    def feasible(job: _Job, chosen: list[_Job]) -> bool:
        starts = sorted([job.start, *(j.start for j in chosen)])
        if (len(starts) > len(supply) or len(set(starts)) != len(starts)
                or any(time < supply[k] for k, time in enumerate(starts))):
            return False
        intervals = [*busy, *((j.start, j.finish) for j in chosen)]
        overlapping = [(max(start, job.start), min(finish, job.finish))
                       for start, finish in intervals if start < job.finish and finish > job.start]
        if int(context["crew_count"]) == 1:
            return not overlapping
        events = [(start, 1) for start, _ in overlapping] + [(finish, -1) for _, finish in overlapping]
        occupied = 1
        for _, change in sorted(events):  # Release before a same-hour start.
            occupied += change
            if occupied > context["crew_count"]:
                return False
        return True

    def search(index: int, chosen: list[_Job], value: float) -> None:
        nonlocal best, best_value, best_tie
        if value + upper[index] < best_value - 1e-9:
            return
        if index == len(candidates):
            # On equal costs postpone the earliest start, then the next; use
            # fewer interventions and asset index as deterministic final ties.
            chronological = sorted(chosen, key=lambda job: (job.start, job.asset))
            tie = (tuple(job.start for job in chronological), -len(chosen),
                   tuple(-job.asset for job in chronological))
            if value > best_value + 1e-9 or (abs(value - best_value) <= 1e-9 and tie > best_tie):
                best, best_value, best_tie = list(chosen), value, tie
            return
        for job in candidates[index]:
            if feasible(job, chosen):
                chosen.append(job)
                search(index + 1, chosen, value + job.saving)
                chosen.pop()
        search(index + 1, chosen, value)

    search(0, [], 0.0)
    return [{"asset": job.asset, "start": job.start, "finish": job.finish,
             "surrogate_saving": job.saving} for job in sorted(best, key=lambda job: (job.start, job.asset))]


def choose_planning_action(env: ResourceMaintenanceEnv, config: PlanningPolicyConfig) -> int:
    """Return a feasible joint action through the three public observable APIs."""
    state, mask = env.policy_state(), env.action_masks()
    if config.trigger == "rolling_plan":
        # No present work can start off shift or without a part/free crew.
        # Tomorrow's schedule cannot change today's common replenishment rule.
        if state["on_shift"] and state["stock"] > 0 and state["free_crews"] > 0:
            schedule = plan_resource_jobs(state, config)
            jobs = [job["asset"] + 1 for job in schedule if job["start"] == 0]
        else:
            jobs = []
    else:
        trigger = "resource_aware" if config.trigger == "local_trend" else config.trigger
        jobs = _job_priorities(state, ResourcePolicyConfig(
            kind=trigger, base_stock=config.base_stock, health_threshold=config.health_threshold,
            safety_margin_hours=config.safety_margin_hours,
        ))
    supplier = ResourcePolicyConfig(
        kind="resource_aware" if config.procurement == "risk_express" else "reactive",
        base_stock=config.base_stock, health_threshold=config.health_threshold,
        safety_margin_hours=config.procurement_margin_hours,
    )
    for job in [*jobs, 0]:
        if not mask[env.encode_action(0, job)]:
            continue
        desired = _order_mode(state, supplier, job)
        modes = (2, 1, 0) if desired == 2 else (desired, 0)
        for mode in modes:
            action = env.encode_action(mode, job)
            if mask[action]:
                return action
    raise RuntimeError("resource environment mask must permit WAIT")
