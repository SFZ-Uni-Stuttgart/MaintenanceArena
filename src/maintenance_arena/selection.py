"""Screen statistics and the selection rules of the noise design in the article (Section 4.3).

A screen replays every candidate design (each with its training runs), the Rule
with each state estimate and the last-safe reference on the same validation
lives. The selection reads three numbers per design in the noisy cell, each the
mean over its runs:

* ``R``: retained profit, the equal-law calendar profit rate (each law's total
  cash over its total calendar time, averaged over the laws) in percent of the
  reference's;
* ``F``: the share of failed lives in percent, averaged over the laws;
* ``family_F``: the failure share of each law.

``select_rule_estimator`` chooses the Rule's state estimate: among the Rules with
``F`` at most 1% and every law at most 2%, the highest ``R``; within 0.10
percentage points the lower ``F``, then the configured order. Without a passing
Rule it takes the lowest ``F``. ``select_design`` then gates the designs at
``F`` at most min(1%, the chosen Rule's ``F`` + 0.25 points) and every law at most
2%, and among them takes the highest ``R``; within 0.10 points the lower ``F``,
the smaller training budget and the configured order. No passing design is a
valid result. ``matching_rule`` names the Rule a selected design is compared with
(the Rule on the design's own estimate), and ``outcome_arrays`` reads the input of
``screen_values`` from outcome rows.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .curves import FAMILY_IDS

LAW_ORDER = tuple(sorted(FAMILY_IDS))
DEFAULT_RULES = {"rule_gate": {"failed_pct": 1.0, "family_pct": 2.0},
                 "design_gate": {"absolute_pct": 1.0, "relative_pp": 0.25, "family_pct": 2.0},
                 "retention_tie_pp": 0.1, "failure_tie": 1e-9}


def screen_values(outcomes: Mapping[str, Mapping[str, np.ndarray]], families: Sequence[str], *,
                  reference: str) -> dict[str, dict[str, Any]]:
    """``R``, ``F`` and ``family_F`` of every arm; ``outcomes[arm]`` holds (runs, lives) ``cash``, ``calendar``, ``failed``.

Every arm is replayed on the same lives, in the order of ``families``. The
reference arm has one run.
"""
    labels = np.asarray(families)
    laws = [law for law in LAW_ORDER if law in set(labels.tolist())] + sorted(set(labels.tolist()) - set(LAW_ORDER))
    index = [np.flatnonzero(labels == law) for law in laws]
    keys = [(arm, run) for arm, data in outcomes.items() for run in range(np.atleast_2d(data["cash"]).shape[0])]
    stack = {name: np.stack([np.atleast_2d(np.asarray(outcomes[arm][name], dtype=np.float64))[run] for arm, run in keys])
             for name in ("cash", "calendar", "failed")}
    rates, fails = [], []
    for rows in index:
        weights = np.ones((1, len(rows)))
        rates.append((weights @ stack["cash"][:, rows].T) / (weights @ stack["calendar"][:, rows].T))
        fails.append(100. * (weights @ stack["failed"][:, rows].T) / weights.sum(axis=1)[:, None])
    rates, fails = np.stack(rates, 1), np.stack(fails, 1)
    mean_rate = rates.mean(1)
    position = keys.index((reference, 0))
    retention, failure = 100. * mean_rate / mean_rate[:, [position]], fails.mean(1)
    result = {}
    for arm in outcomes:
        columns = [i for i, key in enumerate(keys) if key[0] == arm]
        result[arm] = {"R": float(retention[:, columns].mean(1)[0]), "F": float(failure[:, columns].mean(1)[0]),
                       "family_F": {law: float(fails[:, k, columns].mean(1)[0]) for k, law in enumerate(laws)},
                       "per_run_F": [float(v) for v in failure[0, columns]],
                       "per_run_R": [float(v) for v in retention[0, columns]]}
    return result


def outcome_arrays(rows: Iterable[Mapping[str, Any]], law: Callable[[Mapping[str, Any]], str] | None = None
                   ) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """The input of ``screen_values`` from outcome rows (``maintenance_arena.metrics`` columns).

Returns ``{method: {"cash", "calendar", "failed"}}`` with (runs, lives) arrays,
runs in the order they first appear and lives in ``unit`` order, and the law of
every life. ``law`` reads a row's law (default: its ``group``).
"""
    law = law or (lambda row: str(row["group"]))
    data: dict[str, dict[str, dict[int, tuple[float, float, float]]]] = defaultdict(dict)
    laws: dict[int, str] = {}
    for row in rows:
        unit, name = int(row["unit"]), law(row)
        if laws.setdefault(unit, name) != name:
            raise ValueError(f"life {unit} carries two laws")
        failed = row["failed"] if isinstance(row["failed"], bool) else str(row["failed"]) in ("True", "true", "1")
        runs = data[str(row["method"])].setdefault(str(row["run"]), {})
        if unit in runs:
            raise ValueError(f"duplicate outcome of life {unit}")
        runs[unit] = (float(row["cash"]), float(row["calendar_steps"]), float(failed))
    units = sorted(laws)
    arrays = {}
    for method, runs in data.items():
        if any(set(values) != set(units) for values in runs.values()):
            raise ValueError(f"every run of {method} must cover the same lives")
        cube = np.asarray([[values[unit] for unit in units] for values in runs.values()], dtype=np.float64)
        arrays[method] = {"cash": cube[:, :, 0], "calendar": cube[:, :, 1], "failed": cube[:, :, 2]}
    return arrays, [laws[unit] for unit in units]


def matching_rule(values: Mapping[str, Mapping[str, Any]], estimate: str, rules_by_estimate: Mapping[str, Sequence[str]],
                  rules: Mapping[str, Any] = DEFAULT_RULES) -> str:
    """The Rule a selected design is compared with: the Rule on the design's own estimate, chosen like the Rule step."""
    return select_rule_estimator(values, list(rules_by_estimate[estimate]), rules)["rule"]


def _gate(values: Mapping[str, Any], failed: float, family: float) -> bool:
    return values["F"] <= failed and max(values["family_F"].values()) <= family


def select_rule_estimator(values: Mapping[str, Mapping[str, Any]], order: Sequence[str],
                          rules: Mapping[str, Any] = DEFAULT_RULES) -> dict[str, Any]:
    """The Rule (state estimate) of the screen: gate, highest retention, ties by failures and order."""
    gate = rules["rule_gate"]
    passed = {name: _gate(values[name], gate["failed_pct"], gate["family_pct"]) for name in order}
    passing = [name for name in order if passed[name]]
    if passing:
        best = max(values[name]["R"] for name in passing)
        tied = [name for name in passing if values[name]["R"] >= best - rules["retention_tie_pp"]]
        lowest = min(values[name]["F"] for name in tied)
        chosen = [name for name in tied if values[name]["F"] <= lowest + rules["failure_tie"]][0]
        branch = "gate passed"
    else:
        lowest = min(values[name]["F"] for name in order)
        tied = [name for name in order if values[name]["F"] <= lowest + rules["failure_tie"]]
        chosen = min(tied, key=lambda name: (-values[name]["R"], order.index(name)))
        branch = "no Rule passed the gate; the lowest failure share"
    return {"rule": chosen, "branch": branch, "passing": passing, "gate": passed,
            "values": {name: {"R": values[name]["R"], "F": values[name]["F"],
                              "max_family_F": max(values[name]["family_F"].values())} for name in order},
            "F_rule": values[chosen]["F"], "R_rule": values[chosen]["R"]}


def select_design(values: Mapping[str, Mapping[str, Any]], order: Sequence[str], budgets: Mapping[str, int],
                  rule_failure_pct: float, rules: Mapping[str, Any] = DEFAULT_RULES) -> dict[str, Any]:
    """The design of the screen, or ``None`` when no design passes the failure bounds."""
    gate = rules["design_gate"]
    bound = min(gate["absolute_pct"], rule_failure_pct + gate["relative_pp"])
    passed = {name: _gate(values[name], bound, gate["family_pct"]) for name in order}
    passing = [name for name in order if passed[name]]
    result = {"bound_pct": bound, "family_cap_pct": gate["family_pct"], "passing": passing, "gate": passed,
              "values": {name: {"R": values[name]["R"], "F": values[name]["F"],
                                "max_family_F": max(values[name]["family_F"].values()), "budget": budgets[name]}
                         for name in order}}
    if not passing:
        return {**result, "design": None, "ties": {}}
    best = max(values[name]["R"] for name in passing)
    zone = [name for name in passing if values[name]["R"] >= best - rules["retention_tie_pp"]]
    lowest = min(values[name]["F"] for name in zone)
    after_failure = [name for name in zone if values[name]["F"] <= lowest + rules["failure_tie"]]
    smallest = min(budgets[name] for name in after_failure)
    after_budget = [name for name in after_failure if budgets[name] == smallest]
    return {**result, "design": after_budget[0],
            "ties": {"retention_zone": zone, "after_failure_tie": after_failure, "after_budget_tie": after_budget}}


__all__ = ["DEFAULT_RULES", "LAW_ORDER", "matching_rule", "outcome_arrays", "screen_values", "select_design",
           "select_rule_estimator"]
