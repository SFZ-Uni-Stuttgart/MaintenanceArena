"""Fixed mechanical action correction for the known finite gearbox model.

The learned proposal is preserved unless productive operation leaves the set from
which a preventive replacement can still complete safely. This is a model-based
execution layer, not a learned safety guarantee or a cost-optimal fallback.
"""
from __future__ import annotations

import numpy as np

from . import gearbox_model as gearbox


def build_filter(model):
    """Return proposal-to-execution map and exhaustive finite invariance checks."""
    n = model.spec.damage_capacity
    boundary = model.spec.last_safe_remaining_damage
    remaining = np.broadcast_to(np.arange(n, 0, -1)[:, None], (n, 2))
    safe = remaining >= boundary
    actions = np.broadcast_to(np.arange(3, dtype=np.int8), (n, 2, 3)).copy()
    productive = np.broadcast_to(np.arange(3) < 2, actions.shape)
    crossing = (safe[..., None] & productive & ~model.renews
                & ((n - model.next_damage) < boundary))
    actions[crossing] = int(gearbox.Action.REPLACE)
    failures = model.immediate_metrics[..., gearbox.METRIC_INDEX["failures"]]
    chosen_renews = np.take_along_axis(model.renews, actions, axis=2)
    chosen_next = np.take_along_axis(model.next_damage, actions, axis=2)
    chosen_failures = np.take_along_axis(failures, actions, axis=2)
    safe_after = chosen_renews | ((n - chosen_next) >= boundary)
    identity = np.broadcast_to(np.arange(3), actions.shape)
    checks = {
        "new_state_in_safe_set": bool(safe[0].all()),
        "nonrenewal_damage_strictly_increases": bool(np.all(
            model.next_damage[~model.renews] > np.broadcast_to(
                np.arange(n)[:, None, None], actions.shape)[~model.renews])),
        "operating_actions_do_not_renew_inside_safe_set": bool(
            not np.any(model.renews[safe, :2])),
        "replace_always_renews": bool(model.renews[..., 2].all()),
        "replace_safe_inside_safe_set": bool(np.all(failures[..., 2][safe] == 0)),
        "renewal_outside_safe_set_fails": bool(np.all(
            failures[(~safe[..., None]) & model.renews] == 1)),
        "all_filtered_successors_remain_safe_or_renew": bool(np.all(safe_after[safe])),
        "all_filtered_macros_inside_safe_set_failure_free": bool(
            np.all(chosen_failures[safe] == 0)),
        "admissible_proposals_preserved": bool(np.array_equal(actions[~crossing], identity[~crossing])),
        "outside_safe_set_unchanged": bool(np.array_equal(actions[~safe], identity[~safe])),
    }
    if not 1 < boundary < n or not all(checks.values()):
        raise ValueError(f"Model does not support the declared action-filter contract: {checks}")
    actions.flags.writeable = False
    return actions, {
        "safe_remaining_damage_minimum": int(boundary),
        "derivation": "mobilization_epochs * derate_damage + 1",
        "state_count": n * 2, "safe_state_count": int(safe.sum()),
        "outside_safe_state_count": int((~safe).sum()),
        "corrected_state_action_proposals": int(crossing.sum()),
        "checks": checks,
        "guarantee": "From any q >= boundary, every filtered proposal sequence stays preventively recoverable and every renewal is failure-free under the exact model.",
        "limits": "No rescue guarantee from q < boundary; no model uncertainty, physical validation, learned safety or cost-optimal correction claim.",
    }


def apply_policy(policy, action_map):
    """Map each greedy proposal; do not argmax a merged sampled distribution."""
    policy = np.asarray(policy)
    if (policy.shape != action_map.shape[:-1] or not np.issubdtype(policy.dtype, np.integer)
            or np.any(policy < 0) or np.any(policy > 2)):
        raise ValueError("Expected an integer greedy proposal at every state")
    return np.take_along_axis(action_map, policy[..., None], axis=2)[..., 0]


def _probabilities(model_shape, probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.shape != model_shape or not np.isfinite(values).all() or np.any(values < 0)
            or np.max(np.abs(values.sum(axis=2) - 1)) > 1e-12):
        raise ValueError("Expected finite nonnegative actual action probabilities summing to one")
    return values


def pushforward(probabilities, action_map):
    """Transfer proposal mass to its executed action, without resampling or scaling."""
    proposed = _probabilities(action_map.shape, probabilities)
    executed = np.zeros_like(proposed)
    for source in range(3):
        for destination in range(3):
            executed[..., destination] += proposed[..., source] * (action_map[..., source] == destination)
    if np.max(np.abs(executed.sum(axis=2) - proposed.sum(axis=2))) > 1e-14:
        raise ArithmeticError("Action correction lost probability mass")
    return executed


def backward_metrics(model, probabilities):
    """Independent backward cycle-ledger recursion from every possible start state."""
    probabilities = _probabilities(model.reward_raw.shape, probabilities)
    n = model.spec.damage_capacity
    values = np.zeros((n, 2, len(gearbox.METRIC_NAMES)), dtype=np.float64)
    for damage in range(n - 1, -1, -1):
        for opportunity in (0, 1):
            for action in range(3):
                index = (damage, opportunity, action)
                target = model.immediate_metrics[index]
                if not model.renews[index]:
                    nxt = int(model.next_damage[index])
                    if nxt <= damage:
                        raise ValueError("Backward recursion requires strictly increasing damage")
                    target = target + values[nxt].mean(axis=0)
                values[damage, opportunity] += probabilities[index] * target
    return values


def intervention_summary(model, evaluation, proposals, action_map, enabled):
    """Actual proposal corrections weighted by the executed policy's own visits."""
    proposals = _probabilities(action_map.shape, proposals)
    veto = action_map != np.arange(3)
    proposal_mass = evaluation.visitation[..., None] * proposals * veto
    required = float(proposal_mass.sum())
    actual = required if enabled else 0.0
    expected = evaluation.expected
    get = lambda name: float(expected[gearbox.METRIC_INDEX[name]])
    duration, decisions = get("duration"), get("all_decisions")
    renewals = get("planned_renewals") + get("corrective_renewals")
    time_scale = 1000.0 / duration
    costs = {
        "replacement_cost_per_1000": get("replacement_count") * model.spec.replacement_cost_cents / 100 * time_scale,
        "downtime_cost_per_1000": get("downtime_rows") * model.spec.downtime_cost_cents / 100 * time_scale,
        "failure_cost_per_1000": get("failures") * model.spec.failure_cost_cents / 100 * time_scale,
    }
    revenue = sum(get(name) for name in (
        "revenue_low_cents", "revenue_high_cents", "revenue_mobilization_cents")) / 100 * time_scale
    return {
        "unsafe_proposal_count_per_cycle": required,
        "interventions_per_cycle": actual,
        "interventions_per_100_renewals": 100 * actual / renewals,
        "interventions_per_1000_calendar_steps": actual * time_scale,
        "intervention_decision_share": actual / decisions,
        "normal_interventions_per_cycle": float(proposal_mass[..., 0].sum()) if enabled else 0.0,
        "derate_interventions_per_cycle": float(proposal_mass[..., 1].sum()) if enabled else 0.0,
        "revenue_per_1000": revenue, **costs,
        "accounted_profit_per_1000": revenue - sum(costs.values()),
    }
