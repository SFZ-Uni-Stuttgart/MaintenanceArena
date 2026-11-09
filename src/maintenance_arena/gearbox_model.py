"""Exact finite gearbox operation-and-maintenance reference.

The model is intentionally small and learner free.  All damage and monetary
quantities are integers; floating point is used only for expectations under the
two equiprobable opportunity states and for the average-reward recursion.
This module owns no paths, random-number generators, solvers, or publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "maintenance_compact_exact_state_gearbox_bridge_runtime_v1"


class GearboxIntegrityError(RuntimeError):
    """Raised when the frozen mechanics or exact solution become inconsistent."""


class Action(IntEnum):
    NORMAL = 0
    DERATE = 1
    REPLACE = 2


ACTION_NAMES = tuple(action.name for action in Action)
TIE_PRIORITY = (Action.REPLACE, Action.DERATE, Action.NORMAL)

METRIC_NAMES = (
    "reward_cents",
    "duration",
    "revenue_low_cents",
    "revenue_high_cents",
    "revenue_mobilization_cents",
    "productive_epochs",
    "planned_renewals",
    "corrective_renewals",
    "failures",
    "downtime_rows",
    "replacement_count",
    "successful_normal_decisions",
    "successful_derate_decisions",
    "normal_selections",
    "derate_selections",
    "replace_selections",
    "all_decisions",
    "discarded_damage",
)
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}


@dataclass(frozen=True)
class ModelSpec:
    damage_capacity: int = 1440
    normal_damage: int = 12
    derate_damage: int = 7
    mobilization_epochs: int = 12
    post_event_downtime_rows: int = 6
    normal_low_revenue_cents: int = 24225
    normal_high_revenue_cents: int = 25500
    derate_revenue_cents: int = 24225
    replacement_cost_cents: int = 52000
    downtime_cost_cents: int = 2500
    failure_cost_cents: int = 700000
    load_cycles_per_reference_increment: int = 37625
    tie_tolerance_raw: float = 1.0e-10
    regenerative_tolerance_raw: float = 1.0e-8
    bellman_tolerance_raw: float = 1.0e-8
    independent_relative_tolerance: float = 1.0e-9
    transition_probability_tolerance: float = 1.0e-15
    max_dp_iterations: int = 200
    initial_average_reward_raw: float = 0.0

    @property
    def downtime_rows(self) -> int:
        return 1 + self.post_event_downtime_rows

    @property
    def mobilization_damage(self) -> int:
        return self.mobilization_epochs * self.derate_damage

    @property
    def last_safe_remaining_damage(self) -> int:
        return self.mobilization_damage + 1

    @property
    def successful_replace_reward_cents(self) -> int:
        return (
            self.mobilization_epochs * self.derate_revenue_cents
            - self.replacement_cost_cents
            - self.downtime_rows * self.downtime_cost_cents
        )

    @property
    def operating_failure_reward_cents(self) -> int:
        return -(
            self.failure_cost_cents
            + self.replacement_cost_cents
            + self.downtime_rows * self.downtime_cost_cents
        )


@dataclass(frozen=True)
class ExactModel:
    spec: ModelSpec
    reward_raw: np.ndarray
    duration: np.ndarray
    renews: np.ndarray
    next_damage: np.ndarray
    immediate_metrics: np.ndarray
    transition_probability_residual_max: float
    raw_accounting_residual_cents_max: float


@dataclass(frozen=True)
class PolicyEvaluation:
    expected: np.ndarray
    visitation: np.ndarray
    renewal_probability: float
    method: str


@dataclass(frozen=True)
class ExactSolution:
    average_reward_raw: float
    policy: np.ndarray
    bias: np.ndarray
    action_values: np.ndarray
    advantages: np.ndarray
    forward: PolicyEvaluation
    backward: PolicyEvaluation
    iteration_log: tuple[Mapping[str, Any], ...]
    iteration_count: int
    tie_count: int
    regenerative_residual_raw: float
    bellman_residual_raw: float
    selected_action_residual_raw: float
    independent_relative_difference: float


def _metric_vector() -> np.ndarray:
    return np.zeros(len(METRIC_NAMES), dtype=np.float64)


def _set(vector: np.ndarray, name: str, value: float) -> None:
    vector[METRIC_INDEX[name]] = float(value)


def _add(vector: np.ndarray, name: str, value: float) -> None:
    vector[METRIC_INDEX[name]] += float(value)


def _replacement_cost_components(spec: ModelSpec, vector: np.ndarray) -> None:
    _set(vector, "downtime_rows", spec.downtime_rows)
    _set(vector, "replacement_count", 1)


def _action_mechanics(
    spec: ModelSpec, damage: int, opportunity: int, action: Action
) -> tuple[np.ndarray, bool, int]:
    """Return the exact immediate metric vector, renewal flag, and next damage."""

    if not 0 <= damage < spec.damage_capacity or opportunity not in (0, 1):
        raise GearboxIntegrityError("state is outside the finite nonfailed grid")
    metrics = _metric_vector()
    _set(metrics, "all_decisions", 1)

    if action in (Action.NORMAL, Action.DERATE):
        increment = (
            spec.normal_damage if action is Action.NORMAL else spec.derate_damage
        )
        next_damage = damage + increment
        selection = (
            "normal_selections" if action is Action.NORMAL else "derate_selections"
        )
        _set(metrics, selection, 1)
        if next_damage < spec.damage_capacity:
            if action is Action.NORMAL:
                revenue = (
                    spec.normal_high_revenue_cents
                    if opportunity == 1
                    else spec.normal_low_revenue_cents
                )
                _set(metrics, "successful_normal_decisions", 1)
            else:
                revenue = spec.derate_revenue_cents
                _set(metrics, "successful_derate_decisions", 1)
            _set(metrics, "duration", 1)
            _set(metrics, "productive_epochs", 1)
            _set(
                metrics,
                "revenue_high_cents" if opportunity == 1 else "revenue_low_cents",
                revenue,
            )
            _set(metrics, "reward_cents", revenue)
            return metrics, False, next_damage

        _set(metrics, "duration", spec.downtime_rows)
        _set(metrics, "corrective_renewals", 1)
        _set(metrics, "failures", 1)
        _replacement_cost_components(spec, metrics)
        _set(metrics, "reward_cents", spec.operating_failure_reward_cents)
        return metrics, True, -1

    if action is not Action.REPLACE:
        raise GearboxIntegrityError(f"unsupported action: {action}")

    _set(metrics, "replace_selections", 1)
    _replacement_cost_components(spec, metrics)
    if damage + spec.mobilization_damage < spec.damage_capacity:
        revenue = spec.mobilization_epochs * spec.derate_revenue_cents
        _set(metrics, "duration", spec.mobilization_epochs + spec.downtime_rows)
        _set(metrics, "productive_epochs", spec.mobilization_epochs)
        _set(metrics, "planned_renewals", 1)
        _set(metrics, "revenue_mobilization_cents", revenue)
        _set(metrics, "reward_cents", spec.successful_replace_reward_cents)
        _set(
            metrics,
            "discarded_damage",
            spec.damage_capacity - (damage + spec.mobilization_damage),
        )
        return metrics, True, -1

    remaining = spec.damage_capacity - damage
    failing_epoch = math.ceil(remaining / spec.derate_damage)
    if not 1 <= failing_epoch <= spec.mobilization_epochs:
        raise GearboxIntegrityError("failure-before-exchange epoch is invalid")
    productive_epochs = failing_epoch - 1
    revenue = productive_epochs * spec.derate_revenue_cents
    reward = (
        revenue
        - spec.failure_cost_cents
        - spec.replacement_cost_cents
        - spec.downtime_rows * spec.downtime_cost_cents
    )
    _set(metrics, "duration", failing_epoch + spec.post_event_downtime_rows)
    _set(metrics, "productive_epochs", productive_epochs)
    _set(metrics, "corrective_renewals", 1)
    _set(metrics, "failures", 1)
    _set(metrics, "revenue_mobilization_cents", revenue)
    _set(metrics, "reward_cents", reward)
    return metrics, True, -1


def _accounting_reward_cents(spec: ModelSpec, metrics: np.ndarray) -> float:
    revenue = math.fsum(
        float(metrics[METRIC_INDEX[name]])
        for name in (
            "revenue_low_cents",
            "revenue_high_cents",
            "revenue_mobilization_cents",
        )
    )
    return (
        revenue
        - metrics[METRIC_INDEX["replacement_count"]]
        * spec.replacement_cost_cents
        - metrics[METRIC_INDEX["downtime_rows"]] * spec.downtime_cost_cents
        - metrics[METRIC_INDEX["failures"]] * spec.failure_cost_cents
    )


def build_exact_model(spec: ModelSpec | None = None) -> ExactModel:
    """Enumerate the frozen finite kernel and audit every accounting row."""

    spec = spec or ModelSpec()
    if (
        spec.damage_capacity != 1440
        or spec.normal_damage != 12
        or spec.derate_damage != 7
        or spec.mobilization_epochs != 12
        or spec.downtime_rows != 7
        or spec.successful_replace_reward_cents != 221200
        or spec.operating_failure_reward_cents != -769500
        or spec.last_safe_remaining_damage != 85
    ):
        raise GearboxIntegrityError("frozen integer mechanics drifted")

    shape = (spec.damage_capacity, 2, len(Action))
    reward_raw = np.empty(shape, dtype=np.float64)
    duration = np.empty(shape, dtype=np.float64)
    renews = np.empty(shape, dtype=np.bool_)
    next_damage = np.empty(shape, dtype=np.int32)
    immediate = np.empty(shape + (len(METRIC_NAMES),), dtype=np.float64)
    accounting_residuals: list[float] = []
    transition_residuals: list[float] = []

    for damage in range(spec.damage_capacity):
        for opportunity in (0, 1):
            for action in Action:
                metrics, renewal, successor = _action_mechanics(
                    spec, damage, opportunity, action
                )
                index = (damage, opportunity, int(action))
                reward_raw[index] = metrics[METRIC_INDEX["reward_cents"]] / 100.0
                duration[index] = metrics[METRIC_INDEX["duration"]]
                renews[index] = renewal
                next_damage[index] = successor
                immediate[index] = metrics
                accounting_residuals.append(
                    abs(
                        metrics[METRIC_INDEX["reward_cents"]]
                        - _accounting_reward_cents(spec, metrics)
                    )
                )
                row_probability = 1.0 if renewal else math.fsum((0.5, 0.5))
                transition_residuals.append(abs(row_probability - 1.0))
                if not renewal and not damage < successor < spec.damage_capacity:
                    raise GearboxIntegrityError("nonrenewal transition is not acyclic")

    accounting_max = max(accounting_residuals, default=math.inf)
    transition_max = max(transition_residuals, default=math.inf)
    if accounting_max != 0.0:
        raise GearboxIntegrityError("integer-cent raw accounting does not close")
    if transition_max > spec.transition_probability_tolerance:
        raise GearboxIntegrityError("transition probability row does not close")
    return ExactModel(
        spec=spec,
        reward_raw=reward_raw,
        duration=duration,
        renews=renews,
        next_damage=next_damage,
        immediate_metrics=immediate,
        transition_probability_residual_max=transition_max,
        raw_accounting_residual_cents_max=accounting_max,
    )


def observation(spec: ModelSpec, damage: int, opportunity: int) -> np.ndarray:
    if not 0 <= damage < spec.damage_capacity or opportunity not in (0, 1):
        raise GearboxIntegrityError("observation state is invalid")
    dcrit = -math.log(0.2)
    health = math.exp(-dcrit * damage / spec.damage_capacity)
    rate = spec.mobilization_epochs * spec.normal_damage / spec.damage_capacity
    remaining = spec.damage_capacity - damage
    slack = min(
        1.0,
        max(-1.0, (spec.last_safe_remaining_damage - remaining) / 85.0),
    )
    value = np.asarray((health, rate, float(opportunity), slack), dtype=np.float32)
    if value.shape != (4,) or not bool(np.isfinite(value).all()):
        raise GearboxIntegrityError("observation is not finite Float32 shape-(4,)")
    return value


def evaluate_policy_forward(model: ExactModel, policy: np.ndarray) -> PolicyEvaluation:
    """Evaluate a deterministic policy using forward regenerative occupancies."""

    spec = model.spec
    policy = np.asarray(policy, dtype=np.int8)
    if policy.shape != (spec.damage_capacity, 2):
        raise GearboxIntegrityError("policy has the wrong finite-grid shape")
    if bool(np.any(policy < 0) or np.any(policy >= len(Action))):
        raise GearboxIntegrityError("policy contains an invalid action")

    visitation = np.zeros((spec.damage_capacity, 2), dtype=np.float64)
    visitation[0, :] = 0.5
    expected = _metric_vector()
    renewal_probability = 0.0
    for damage in range(spec.damage_capacity):
        for opportunity in (0, 1):
            mass = float(visitation[damage, opportunity])
            if mass == 0.0:
                continue
            action = int(policy[damage, opportunity])
            index = (damage, opportunity, action)
            expected += mass * model.immediate_metrics[index]
            if bool(model.renews[index]):
                renewal_probability += mass
            else:
                successor = int(model.next_damage[index])
                visitation[successor, 0] += 0.5 * mass
                visitation[successor, 1] += 0.5 * mass
    if abs(renewal_probability - 1.0) > 1.0e-12:
        raise GearboxIntegrityError(
            f"regenerative forward mass does not close: {renewal_probability}"
        )
    return PolicyEvaluation(
        expected=expected,
        visitation=visitation,
        renewal_probability=renewal_probability,
        method="forward_stationary_occupancy",
    )


def evaluate_policy_backward(model: ExactModel, policy: np.ndarray) -> PolicyEvaluation:
    """Independently evaluate a policy by acyclic backward recursion."""

    spec = model.spec
    policy = np.asarray(policy, dtype=np.int8)
    values = np.zeros(
        (spec.damage_capacity, 2, len(METRIC_NAMES)), dtype=np.float64
    )
    for damage in range(spec.damage_capacity - 1, -1, -1):
        for opportunity in (0, 1):
            action = int(policy[damage, opportunity])
            if not 0 <= action < len(Action):
                raise GearboxIntegrityError("policy contains an invalid action")
            index = (damage, opportunity, action)
            values[damage, opportunity] = model.immediate_metrics[index]
            if not bool(model.renews[index]):
                successor = int(model.next_damage[index])
                values[damage, opportunity] += 0.5 * (
                    values[successor, 0] + values[successor, 1]
                )
    expected = 0.5 * (values[0, 0] + values[0, 1])
    return PolicyEvaluation(
        expected=expected,
        visitation=np.empty((0, 0), dtype=np.float64),
        renewal_probability=1.0,
        method="backward_regenerative_recursion",
    )


def _optimal_bias(
    model: ExactModel, average_reward_raw: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    spec = model.spec
    bias = np.zeros((spec.damage_capacity, 2), dtype=np.float64)
    policy = np.zeros((spec.damage_capacity, 2), dtype=np.int8)
    action_values = np.zeros((spec.damage_capacity, 2, len(Action)), dtype=np.float64)
    tie_count = 0
    for damage in range(spec.damage_capacity - 1, -1, -1):
        for opportunity in (0, 1):
            for action in Action:
                index = (damage, opportunity, int(action))
                value = (
                    model.reward_raw[index]
                    - average_reward_raw * model.duration[index]
                )
                if not bool(model.renews[index]):
                    successor = int(model.next_damage[index])
                    value += 0.5 * (
                        bias[successor, 0] + bias[successor, 1]
                    )
                action_values[index] = value
            maximum = float(np.max(action_values[damage, opportunity]))
            tied = tuple(
                action
                for action in TIE_PRIORITY
                if maximum - action_values[damage, opportunity, int(action)]
                <= spec.tie_tolerance_raw
            )
            if len(tied) > 1:
                tie_count += 1
            selected = tied[0]
            policy[damage, opportunity] = int(selected)
            bias[damage, opportunity] = action_values[
                damage, opportunity, int(selected)
            ]
    return bias, policy, action_values, tie_count


def _rate_raw(evaluation: PolicyEvaluation) -> float:
    reward_raw = evaluation.expected[METRIC_INDEX["reward_cents"]] / 100.0
    duration = evaluation.expected[METRIC_INDEX["duration"]]
    if not math.isfinite(reward_raw) or not math.isfinite(duration) or duration <= 0:
        raise GearboxIntegrityError("policy evaluation is nonfinite or has zero duration")
    return float(reward_raw / duration)


def solve_exact_dp(model: ExactModel) -> ExactSolution:
    """Solve the finite average-reward SMDP by regenerative policy iteration."""

    spec = model.spec
    average = float(spec.initial_average_reward_raw)
    previous_policy: np.ndarray | None = None
    log: list[Mapping[str, Any]] = []
    converged = False
    for iteration in range(1, spec.max_dp_iterations + 1):
        bias, policy, action_values, tie_count = _optimal_bias(model, average)
        forward = evaluate_policy_forward(model, policy)
        updated = _rate_raw(forward)
        reward_raw = forward.expected[METRIC_INDEX["reward_cents"]] / 100.0
        duration = forward.expected[METRIC_INDEX["duration"]]
        residual = reward_raw - average * duration
        stable = previous_policy is not None and bool(
            np.array_equal(policy, previous_policy)
        )
        log.append(
            {
                "iteration": iteration,
                "candidate_average_reward_raw": average,
                "selected_policy_average_reward_raw": updated,
                "regenerative_residual_raw": residual,
                "policy_stable": stable,
                "tie_state_count": tie_count,
            }
        )
        if stable and abs(residual) <= spec.regenerative_tolerance_raw:
            converged = True
            average = updated
            break
        previous_policy = policy.copy()
        average = updated
    if not converged:
        raise GearboxIntegrityError("regenerative policy iteration did not converge")

    bias, policy, action_values, tie_count = _optimal_bias(model, average)
    forward = evaluate_policy_forward(model, policy)
    backward = evaluate_policy_backward(model, policy)
    verified_average = _rate_raw(forward)
    verification_policy = _optimal_bias(model, verified_average)[1]
    if not np.array_equal(policy, verification_policy):
        raise GearboxIntegrityError("final policy is not stable at its own average reward")

    reward_raw = forward.expected[METRIC_INDEX["reward_cents"]] / 100.0
    duration = forward.expected[METRIC_INDEX["duration"]]
    regenerative_residual = abs(reward_raw - verified_average * duration)
    maximum = np.max(action_values, axis=2)
    bellman_residual = float(np.max(np.abs(maximum - bias)))
    selected_values = np.take_along_axis(
        action_values, policy[:, :, None].astype(np.int64), axis=2
    )[:, :, 0]
    selected_residual = float(np.max(np.abs(maximum - selected_values)))
    _rate_raw(backward)  # Finite positive-duration guard for the independent route.
    component_scale = np.maximum(np.abs(forward.expected), 1.0)
    independent_relative = float(
        np.max(np.abs(forward.expected - backward.expected) / component_scale)
    )
    if regenerative_residual > spec.regenerative_tolerance_raw:
        raise GearboxIntegrityError("regenerative residual exceeds the frozen limit")
    if bellman_residual > spec.bellman_tolerance_raw:
        raise GearboxIntegrityError("Bellman residual exceeds the frozen limit")
    if selected_residual > spec.bellman_tolerance_raw:
        raise GearboxIntegrityError("selected policy is not Bellman optimal")
    if independent_relative > spec.independent_relative_tolerance:
        raise GearboxIntegrityError("independent policy evaluations disagree")

    advantages = action_values - maximum[:, :, None]
    return ExactSolution(
        average_reward_raw=verified_average,
        policy=policy,
        bias=bias,
        action_values=action_values,
        advantages=advantages,
        forward=forward,
        backward=backward,
        iteration_log=tuple(log),
        iteration_count=len(log),
        tie_count=tie_count,
        regenerative_residual_raw=regenerative_residual,
        bellman_residual_raw=bellman_residual,
        selected_action_residual_raw=selected_residual,
        independent_relative_difference=independent_relative,
    )


def policy_summary(
    model: ExactModel,
    evaluation: PolicyEvaluation,
    *,
    policy_id: str,
    theta: int | None = None,
) -> dict[str, Any]:
    expected = evaluation.expected
    get = lambda name: float(expected[METRIC_INDEX[name]])
    direct_reward_cents = get("reward_cents")
    # The scientific accounting identity is defined algebraically from the
    # integer-cent components.  Keep the tiny accumulation-order discrepancy
    # as a separate numerical diagnostic rather than turning it into economics.
    reward_cents = _accounting_reward_cents(model.spec, expected)
    duration = get("duration")
    planned = get("planned_renewals")
    corrective = get("corrective_renewals")
    renewals = planned + corrective
    successful_normal = get("successful_normal_decisions")
    successful_derate = get("successful_derate_decisions")
    productive_decisions = successful_normal + successful_derate
    all_decisions = get("all_decisions")
    planned_share = planned / renewals if renewals > 0 else 0.0
    failures_per_100 = 100.0 * get("failures") / renewals if renewals > 0 else math.inf
    accumulation_difference = direct_reward_cents - reward_cents
    return {
        "policy_id": policy_id,
        "theta_remaining_damage": theta,
        "expected_cycle_reward_cents": reward_cents,
        "direct_accumulated_cycle_reward_cents": direct_reward_cents,
        "expected_cycle_reward_raw": reward_cents / 100.0,
        "expected_cycle_duration": duration,
        "raw_profit_per_1000": reward_cents / duration * 10.0,
        "revenue_low_cents": get("revenue_low_cents"),
        "revenue_high_cents": get("revenue_high_cents"),
        "revenue_mobilization_cents": get("revenue_mobilization_cents"),
        "productive_epochs": get("productive_epochs"),
        "availability": get("productive_epochs") / duration,
        "planned_renewals": planned,
        "corrective_renewals": corrective,
        "planned_renewal_share": planned_share,
        "failures": get("failures"),
        "failures_per_100_renewals": failures_per_100,
        "downtime_rows": get("downtime_rows"),
        "replacement_count": get("replacement_count"),
        "successful_normal_decisions": successful_normal,
        "successful_derate_decisions": successful_derate,
        "normal_productive_decision_share": (
            successful_normal / productive_decisions if productive_decisions > 0 else 0.0
        ),
        "derate_productive_decision_share": (
            successful_derate / productive_decisions if productive_decisions > 0 else 0.0
        ),
        "normal_selections": get("normal_selections"),
        "derate_selections": get("derate_selections"),
        "replace_selections": get("replace_selections"),
        "all_decisions": all_decisions,
        "replace_total_decision_share": (
            get("replace_selections") / all_decisions if all_decisions > 0 else 0.0
        ),
        "discarded_damage_total": get("discarded_damage"),
        "discarded_damage_per_planned_exchange": (
            get("discarded_damage") / planned if planned > 0 else None
        ),
        "discarded_normal_equivalent_epochs_per_planned_exchange": (
            get("discarded_damage") / planned / model.spec.normal_damage
            if planned > 0
            else None
        ),
        "discarded_load_cycles_per_planned_exchange": (
            get("discarded_damage")
            / planned
            / model.spec.normal_damage
            * model.spec.load_cycles_per_reference_increment
            if planned > 0
            else None
        ),
        "raw_accounting_residual_cents": 0.0,
        "floating_accumulation_difference_cents": accumulation_difference,
        "renewal_probability": evaluation.renewal_probability,
        "evaluation_method": evaluation.method,
    }


def _threshold_policy(
    spec: ModelSpec, theta: int, operating_actions: Sequence[Action]
) -> np.ndarray:
    if not 1 <= theta <= spec.damage_capacity or len(operating_actions) != 2:
        raise GearboxIntegrityError("threshold comparator specification is invalid")
    policy = np.empty((spec.damage_capacity, 2), dtype=np.int8)
    for damage in range(spec.damage_capacity):
        remaining = spec.damage_capacity - damage
        for opportunity in (0, 1):
            policy[damage, opportunity] = int(
                Action.REPLACE
                if remaining <= theta
                else operating_actions[opportunity]
            )
    return policy


def _best_threshold(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise GearboxIntegrityError("threshold scan is empty")
    return min(
        rows,
        key=lambda row: (
            -float(row["raw_profit_per_1000"]),
            float(row["failures_per_100_renewals"]),
            int(row["theta_remaining_damage"]),
            str(row["policy_id"]),
        ),
    )


def enumerate_comparators(
    model: ExactModel,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Exhaustively evaluate the four frozen conventional policy families."""

    spec = model.spec
    reactive = np.full((spec.damage_capacity, 2), int(Action.NORMAL), dtype=np.int8)
    summaries: list[dict[str, Any]] = [
        policy_summary(
            model,
            evaluate_policy_forward(model, reactive),
            policy_id="reactive_normal",
        )
    ]
    scans: list[dict[str, Any]] = []
    best_policies: dict[str, np.ndarray] = {"reactive_normal": reactive}
    families = (
        ("best_normal_threshold", (Action.NORMAL, Action.NORMAL)),
        ("best_derate_threshold", (Action.DERATE, Action.DERATE)),
        (
            "best_demand_responsive_separable_threshold",
            (Action.DERATE, Action.NORMAL),
        ),
    )
    for policy_id, operating in families:
        family_rows: list[dict[str, Any]] = []
        for theta in range(1, spec.damage_capacity + 1):
            policy = _threshold_policy(spec, theta, operating)
            summary = policy_summary(
                model,
                evaluate_policy_forward(model, policy),
                policy_id=policy_id,
                theta=theta,
            )
            scans.append(summary)
            family_rows.append(summary)
        best = dict(_best_threshold(family_rows))
        summaries.append(best)
        best_policies[policy_id] = _threshold_policy(
            spec, int(best["theta_remaining_damage"]), operating
        )
    return summaries, scans, best_policies


def classify_structural_outcome(
    model: ExactModel,
    solution: ExactSolution,
    dp_summary: Mapping[str, Any],
    best_normal_summary: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    spec = model.spec
    ratio = float(dp_summary["raw_profit_per_1000"]) / float(
        best_normal_summary["raw_profit_per_1000"]
    )
    failure_delta = float(dp_summary["failures_per_100_renewals"]) - float(
        best_normal_summary["failures_per_100_renewals"]
    )
    checks = (
        (
            "transition_probability_integrity",
            model.transition_probability_residual_max
            <= spec.transition_probability_tolerance,
            model.transition_probability_residual_max,
            spec.transition_probability_tolerance,
            "maximum",
        ),
        (
            "integer_cent_raw_accounting_integrity",
            model.raw_accounting_residual_cents_max == 0.0
            and float(dp_summary["raw_accounting_residual_cents"]) == 0.0,
            max(
                model.raw_accounting_residual_cents_max,
                float(dp_summary["raw_accounting_residual_cents"]),
            ),
            0.0,
            "exact",
        ),
        (
            "regenerative_residual_integrity",
            solution.regenerative_residual_raw <= spec.regenerative_tolerance_raw,
            solution.regenerative_residual_raw,
            spec.regenerative_tolerance_raw,
            "maximum",
        ),
        (
            "bellman_residual_integrity",
            solution.bellman_residual_raw <= spec.bellman_tolerance_raw,
            solution.bellman_residual_raw,
            spec.bellman_tolerance_raw,
            "maximum",
        ),
        (
            "independent_evaluation_integrity",
            solution.independent_relative_difference
            <= spec.independent_relative_tolerance,
            solution.independent_relative_difference,
            spec.independent_relative_tolerance,
            "maximum",
        ),
        (
            "dp_failures_per_100_renewals",
            float(dp_summary["failures_per_100_renewals"]) <= 0.5,
            float(dp_summary["failures_per_100_renewals"]),
            0.5,
            "maximum",
        ),
        (
            "normal_productive_decision_share",
            float(dp_summary["normal_productive_decision_share"]) >= 0.1,
            float(dp_summary["normal_productive_decision_share"]),
            0.1,
            "minimum",
        ),
        (
            "derate_productive_decision_share",
            float(dp_summary["derate_productive_decision_share"]) >= 0.1,
            float(dp_summary["derate_productive_decision_share"]),
            0.1,
            "minimum",
        ),
        (
            "replace_total_decision_share",
            float(dp_summary["replace_total_decision_share"]) >= 0.003,
            float(dp_summary["replace_total_decision_share"]),
            0.003,
            "minimum",
        ),
        (
            "dp_over_best_normal_profit_ratio",
            ratio >= 1.005,
            ratio,
            1.005,
            "minimum",
        ),
        (
            "dp_minus_best_normal_failures_per_100",
            failure_delta <= 0.25,
            failure_delta,
            0.25,
            "maximum",
        ),
    )
    rows = [
        {
            "gate": name,
            "passed": bool(passed),
            "observed": observed,
            "threshold": threshold,
            "direction": direction,
            "kind": "integrity" if index < 5 else "scientific_admission",
        }
        for index, (name, passed, observed, threshold, direction) in enumerate(checks)
    ]
    integrity_pass = all(row["passed"] for row in rows if row["kind"] == "integrity")
    if not integrity_pass:
        return "INVALID_TECHNICAL_LEAF", rows
    supported = all(row["passed"] for row in rows)
    return (
        "COUPLED_HEADROOM_SUPPORTED"
        if supported
        else "VALID_NO_MATERIAL_HEADROOM",
        rows,
    )


def policy_rows(model: ExactModel, solution: ExactSolution) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spec = model.spec
    visitation = solution.forward.visitation
    reachable_damage = np.zeros(spec.damage_capacity, dtype=np.bool_)
    reachable_damage[0] = True
    for damage in range(spec.damage_capacity):
        if not bool(reachable_damage[damage]):
            continue
        for increment in (spec.normal_damage, spec.derate_damage):
            successor = damage + increment
            if successor < spec.damage_capacity:
                reachable_damage[successor] = True
    for damage in range(spec.damage_capacity):
        remaining = spec.damage_capacity - damage
        for opportunity in (0, 1):
            selected = Action(int(solution.policy[damage, opportunity]))
            obs = observation(spec, damage, opportunity)
            row: dict[str, Any] = {
                "damage": damage,
                "remaining_damage": remaining,
                "opportunity": "HIGH" if opportunity else "LOW",
                "opportunity_index": opportunity,
                "exact_health": float(obs[0]),
                "lead_scaled_normal_damage_rate": float(obs[1]),
                "causal_replacement_slack": float(obs[3]),
                "selected_action": selected.name,
                "selected_action_index": int(selected),
                "reachable_visitation_mass_per_cycle": float(
                    visitation[damage, opportunity]
                ),
                "positive_mass_under_dp_policy": bool(
                    visitation[damage, opportunity] > 0.0
                ),
                "reachable_under_any_operating_action_sequence": bool(
                    reachable_damage[damage]
                ),
                "bias_raw": float(solution.bias[damage, opportunity]),
            }
            for action in Action:
                row[f"q_{action.name.lower()}_raw"] = float(
                    solution.action_values[damage, opportunity, int(action)]
                )
                row[f"advantage_{action.name.lower()}_raw"] = float(
                    solution.advantages[damage, opportunity, int(action)]
                )
            rows.append(row)
    return rows


def conventional_policy_rows(
    model: ExactModel,
    policies: Mapping[str, np.ndarray],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Export every selected conventional policy on the complete state grid."""

    theta_by_id = {
        str(row["policy_id"]): row.get("theta_remaining_damage") for row in summaries
    }
    rows: list[dict[str, Any]] = []
    for policy_id in (
        "reactive_normal",
        "best_normal_threshold",
        "best_derate_threshold",
        "best_demand_responsive_separable_threshold",
    ):
        policy = np.asarray(policies[policy_id], dtype=np.int8)
        if policy.shape != (model.spec.damage_capacity, 2):
            raise GearboxIntegrityError("conventional policy map shape drifted")
        for damage in range(model.spec.damage_capacity):
            for opportunity in (0, 1):
                action = Action(int(policy[damage, opportunity]))
                rows.append(
                    {
                        "policy_id": policy_id,
                        "theta_remaining_damage": theta_by_id[policy_id],
                        "damage": damage,
                        "remaining_damage": model.spec.damage_capacity - damage,
                        "opportunity": "HIGH" if opportunity else "LOW",
                        "opportunity_index": opportunity,
                        "selected_action": action.name,
                        "selected_action_index": int(action),
                    }
                )
    return rows


def representative_lifecycle(
    model: ExactModel, policy: np.ndarray
) -> list[dict[str, Any]]:
    """Build one deterministic alternating-opportunity illustration.

    The trace is explicitly illustrative and is not used for any exact metric.
    """

    spec = model.spec
    damage = 0
    decision_index = 0
    calendar = 0
    rows: list[dict[str, Any]] = []
    while damage < spec.damage_capacity and decision_index < spec.damage_capacity:
        opportunity = decision_index % 2
        action = Action(int(policy[damage, opportunity]))
        before = damage
        if action in (Action.NORMAL, Action.DERATE):
            increment = spec.normal_damage if action is Action.NORMAL else spec.derate_damage
            after = damage + increment
            success = after < spec.damage_capacity
            revenue = (
                (
                    spec.normal_high_revenue_cents
                    if opportunity == 1
                    else spec.normal_low_revenue_cents
                )
                if action is Action.NORMAL
                else spec.derate_revenue_cents
            ) if success else 0
            rows.append(
                {
                    "calendar_step": calendar,
                    "decision_index": decision_index,
                    "phase": "standalone_operation" if success else "failure_event",
                    "opportunity": "HIGH" if opportunity else "LOW",
                    "action": action.name,
                    "damage_before": before,
                    "damage_after": min(after, spec.damage_capacity),
                    "reward_cents": (
                        revenue if success else spec.operating_failure_reward_cents
                    ),
                    "productive_revenue_cents": revenue,
                    "replacement_cost_cents": 0 if success else spec.replacement_cost_cents,
                    "downtime_cost_cents": 0 if success else spec.downtime_rows * spec.downtime_cost_cents,
                    "failure_cost_cents": 0 if success else spec.failure_cost_cents,
                    "illustrative_not_scientific_metric": True,
                }
            )
            if not success:
                break
            damage = after
            calendar += 1
            decision_index += 1
            continue

        # Expand the irreversible macro for reader-facing timing transparency.
        for lead_epoch in range(1, spec.mobilization_epochs + 1):
            before = damage
            after = damage + spec.derate_damage
            success = after < spec.damage_capacity
            rows.append(
                {
                    "calendar_step": calendar,
                    "decision_index": decision_index,
                    "phase": "replacement_mobilization" if success else "failure_event",
                    "opportunity": "COMMITTED",
                    "action": "DERATE_MOBILIZATION",
                    "damage_before": before,
                    "damage_after": min(after, spec.damage_capacity),
                    "reward_cents": spec.derate_revenue_cents if success else -spec.failure_cost_cents,
                    "productive_revenue_cents": spec.derate_revenue_cents if success else 0,
                    "replacement_cost_cents": 0,
                    "downtime_cost_cents": 0,
                    "failure_cost_cents": 0 if success else spec.failure_cost_cents,
                    "illustrative_not_scientific_metric": True,
                }
            )
            calendar += 1
            if not success:
                damage = spec.damage_capacity
                break
            damage = after
        planned = damage < spec.damage_capacity
        event_phase = "planned_exchange_event" if planned else "post_failure_downtime"
        # In a failed macro the failure row above is already the charged event
        # row.  It must also carry corrective replacement and the first
        # downtime charge, leaving exactly six subsequent rows (duration j+6).
        if not planned:
            failed = rows[-1]
            failed["reward_cents"] -= (
                spec.replacement_cost_cents + spec.downtime_cost_cents
            )
            failed["replacement_cost_cents"] = spec.replacement_cost_cents
            failed["downtime_cost_cents"] = spec.downtime_cost_cents
        downtime_count = spec.downtime_rows if planned else spec.post_event_downtime_rows
        for downtime_index in range(downtime_count):
            rows.append(
                {
                    "calendar_step": calendar,
                    "decision_index": decision_index,
                    "phase": event_phase if downtime_index == 0 else "post_event_downtime",
                    "opportunity": "NONE",
                    "action": "EXCHANGE" if planned and downtime_index == 0 else "DOWNTIME",
                    "damage_before": damage,
                    "damage_after": damage,
                    "reward_cents": -spec.downtime_cost_cents - (spec.replacement_cost_cents if planned and downtime_index == 0 else 0),
                    "productive_revenue_cents": 0,
                    "replacement_cost_cents": spec.replacement_cost_cents if planned and downtime_index == 0 else 0,
                    "downtime_cost_cents": spec.downtime_cost_cents,
                    "failure_cost_cents": 0,
                    "illustrative_not_scientific_metric": True,
                }
            )
            calendar += 1
        break
    return rows
