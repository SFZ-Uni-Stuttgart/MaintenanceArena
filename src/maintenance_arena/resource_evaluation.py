"""Continuing-run resource economics with explicit observation-window boundaries.

Windows are slices of one uninterrupted episode. Spare book-value correction
charges consumed parts instead of purchased parts; it does not value installed
bearing life, future failures, or unfinished service. Inventory integrals use
the stock after decisions and before end-of-hour delivery, matching the actual
exposure during each hourly interval. Traces contain only action/event hours.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from .resource import COST_FIELDS, LEDGER_FIELDS, ResourceMaintenanceEnv

if TYPE_CHECKING:
    from .resource_planning import PlanningPolicyConfig


# State snapshots in env.info['metrics'] must not be differenced as counters.
CUMULATIVE_METRICS = (
    "operating_asset_hours", "down_asset_hours", "crew_work_hours", "paid_crew_hours",
    "failures", "planned_starts", "corrective_starts", "completed_replacements",
    "failed_waiting_asset_hours", "no_part_wait_hours", "no_crew_wait_hours",
    "off_shift_wait_hours", "parts_ordered", "parts_delivered", "parts_consumed",
    "normal_orders", "express_orders", "invalid_actions",
)


def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Keep observable boundary state, including unvalued outstanding work."""
    assets = [dict(asset) for asset in state["assets"]]
    return {
        "phase": state["phase"],
        "stock": state["stock"],
        "inventory_position": state["inventory_position"],
        "pipeline": [dict(order) for order in state["pipeline"]],
        "repair_jobs_open": sum(a["repair_work_remaining"] > 0 for a in assets),
        "repair_work_remaining": sum(a["repair_work_remaining"] for a in assets),
        "failed_assets": sum(a["failed"] for a in assets),
        "failed_backlog_assets": sum(a["failed"] and a["repair_work_remaining"] == 0 for a in assets),
        "health": [a["health"] for a in assets],
        "assets": assets,
    }


def run_episode(
    configuration: dict[str, Any],
    policy: PlanningPolicyConfig,
    seed: int,
    windows: list[tuple[int, int]],
    trace_path: Path | None = None,
    *,
    action_fn: Callable[[ResourceMaintenanceEnv, Any], int] | None = None,
) -> dict[str, Any]:
    """Evaluate one seeded fleet without resets or horizon cues at window edges.

    ``action_fn`` supports focused mechanical tests. Production uses the
    observable-only planning policy. Annual capital charges can be calculated
    from ``inventory_position_hours`` without modifying the raw cash ledger.
    """
    started = perf_counter()
    if not windows or any(
        len(window) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) for x in window)
        or not 0 <= window[0] < window[1]
        for window in windows
    ):
        raise ValueError("windows must contain nonempty integer intervals [start, end)")
    if len(set(tuple(window) for window in windows)) != len(windows):
        raise ValueError("duplicate evaluation windows are not supported")
    if action_fn is None:
        from .resource_planning import choose_planning_action
        action_fn = choose_planning_action

    horizon = max(end for _, end in windows)
    cfg = dict(configuration, horizon=horizon)
    env = ResourceMaintenanceEnv(cfg)
    observation, info = env.reset(seed=seed)
    if not np.isfinite(observation).all():
        raise AssertionError("Nonfinite initial observation")
    boundaries = {0, *(point for window in windows for point in window)}
    prefixes: dict[int, dict[str, Any]] = {}
    reward_sum, position_hours, on_hand_hours = 0.0, 0.0, 0.0
    decision_seconds, trace_rows = 0.0, 0
    state = env.policy_state()

    def capture(hour: int) -> None:
        prefixes[hour] = {
            "ledger": dict(info["cumulative_ledger"]),
            "metrics": {name: int(info["metrics"][name]) for name in CUMULATIVE_METRICS},
            "state": _snapshot(state),
            "reward_sum": reward_sum,
            "inventory_position_hours": position_hours,
            "inventory_on_hand_hours": on_hand_hours,
        }

    capture(0)
    trace_path = Path(trace_path) if trace_path is not None else None
    trace = None
    try:
        if trace_path is not None:
            trace = trace_path.open("x", encoding="utf-8")
        for hour in range(horizon):
            before_state = state
            mask = env.action_masks()
            choice_started = perf_counter()
            action = action_fn(env, policy)
            decision_seconds += perf_counter() - choice_started
            if not env.action_space.contains(action) or not mask[int(action)]:
                raise AssertionError("Policy proposed an infeasible action")
            order_mode, job_choice = env.decode_action(action)
            # Orders are paid and enter inventory position immediately; a
            # starting repair consumes one part. Delivery does not alter position.
            on_hand_hours += before_state["stock"] - int(job_choice > 0)
            position_hours += before_state["inventory_position"] - int(job_choice > 0) + int(order_mode > 0)
            observation, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise AssertionError("Nonfinite observation or reward")
            if info["metrics"]["inventory_conservation_residual"] != 0:
                raise AssertionError("Inventory conservation failed")
            if info["invalid_action"]:
                raise AssertionError("Environment rejected the masked policy action")
            if terminated or bool(truncated) != (hour + 1 == horizon):
                raise AssertionError("Unexpected continuing-task termination contract")
            reward_sum += float(reward)
            state = env.policy_state()
            if trace is not None and (int(action) != 0 or info["events"]):
                trace.write(json.dumps({
                    "t": hour, "state": before_state, "action": int(action),
                    "ledger": info["ledger"], "events": info["events"],
                }, allow_nan=False) + "\n")
                trace_rows += 1
            if hour + 1 in boundaries:
                capture(hour + 1)

        result_windows = []
        for start, end in windows:
            opening, closing = prefixes[start], prefixes[end]
            hours = end - start
            ledger = {name: closing["ledger"][name] - opening["ledger"][name] for name in LEDGER_FIELDS}
            metrics = {name: closing["metrics"][name] - opening["metrics"][name] for name in CUMULATIVE_METRICS}
            raw_net = ledger["net_profit"]
            if not np.isclose(raw_net, ledger["contribution"] - sum(ledger[key] for key in COST_FIELDS), atol=1e-6, rtol=1e-10):
                raise AssertionError("Window cash ledger does not reconcile")
            if not np.isclose((closing["reward_sum"] - opening["reward_sum"]) * env.config.reward_scale, raw_net, atol=1e-6, rtol=1e-10):
                raise AssertionError("Window reward does not equal scaled cash ledger")
            if metrics["operating_asset_hours"] + metrics["down_asset_hours"] != env.config.n_assets * hours:
                raise AssertionError("Window asset-hour partition failed")
            paused = metrics["down_asset_hours"] - metrics["crew_work_hours"] - metrics["failed_waiting_asset_hours"]
            if paused < 0:
                raise AssertionError("Downtime decomposition produced negative paused repair hours")
            consumed_net = raw_net + ledger["initial_stock_cost"] + ledger["parts_cost"] - env.config.part_cost * metrics["parts_consumed"]
            book_value_net = raw_net + ledger["initial_stock_cost"] + env.config.part_cost * (
                closing["state"]["inventory_position"] - opening["state"]["inventory_position"]
            )
            if not np.isclose(consumed_net, book_value_net, atol=1e-6, rtol=1e-10):
                raise AssertionError("Consumed-part and inventory-position corrections disagree")
            window_position_hours = closing["inventory_position_hours"] - opening["inventory_position_hours"]
            window_on_hand_hours = closing["inventory_on_hand_hours"] - opening["inventory_on_hand_hours"]
            result_windows.append({
                "start": start, "end": end, "hours": hours,
                "cash_ledger": ledger, "cumulative_metrics": metrics,
                "cash_profit_per_hour": raw_net / hours,
                "consumption_profit": consumed_net,
                "consumption_profit_per_hour": consumed_net / hours,
                "availability": metrics["operating_asset_hours"] / (env.config.n_assets * hours),
                "failures_per_1000_hours": 1000.0 * metrics["failures"] / hours,
                "paused_repair_asset_hours": paused,
                "inventory_on_hand_hours": window_on_hand_hours,
                "inventory_position_hours": window_position_hours,
                "average_inventory_on_hand": window_on_hand_hours / hours,
                "average_inventory_position": window_position_hours / hours,
                "opening_state": opening["state"], "closing_state": closing["state"],
            })
        return {
            "seed": int(seed), "policy": asdict(policy), "windows": result_windows,
            "elapsed_seconds": perf_counter() - started,
            "decision_seconds": decision_seconds,
            "total_decisions": horizon, "all_actions_feasible": True,
            "trace": {
                "path": str(trace_path.resolve()) if trace_path is not None else None,
                "rows": trace_rows,
                "sampling": "nonzero-action or event hours; no quiet WAIT rows",
            },
        }
    finally:
        if trace is not None:
            trace.close()
        env.close()
