"""Batch replay of first orders on scenario banks, and the article's rule calibrations.

Until its first order a policy only waits, so the causal states of an
unmaintained life fix the first order of every threshold rule and of every
policy that acts on these states alone. The cash and calendar time of a life
then follow from its first order and its last productive step ``T = eol - 1``:
an order at step ``t <= T - L`` completes before failure, any later order or
no order ends in a failure. ``replay`` applies the core's event accounting to
whole arrays of first orders; ``ScenarioBankEnv`` gives the same results step
by step.

Rules are calibrated by maximizing the calendar profit (cash per elapsed step)
averaged with equal weight over the laws of the calibration lives. Candidates
within 1e-9 of the best rate are tied; ties go to the lower failure share, then
to a rule-specific tie-break (for example the distance to an anchor threshold),
then to the lower candidate index. ``select_failure_budget`` instead minimizes
unused life subject to a maximum failure share.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .banks import component_measurements
from .observations import reconstruct

RATE_TIE = 1e-9
RULE_FAMILIES = ("health", "projection")
TIE_ANCHORS = {"health": .36, "projection": 15.}
# Order cuts on the learned ORDER probability: twelve small values, k/100 for
# k = 1..99 and the complements of the small values; index 61 is 0.5.
_SMALL_CUTS = (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)
CUT_GRID = (*_SMALL_CUTS, *[k / 100 for k in range(1, 100)], *[1. - x for x in reversed(_SMALL_CUTS)])
CUT_ANCHOR_INDEX = 61


def grid(start: float, stop: float, step: float, digits: int) -> list[float]:
    """An inclusive, rounded arithmetic grid, for example ``grid(.2, .6, .0025, 4)``."""
    count = int(round((stop - start) / step)) + 1
    return [round(start + step * k, digits) for k in range(count)]


@dataclass(frozen=True)
class Economics:
    """Prices and event clock of the core; ``replay`` applies them to first orders.

The defaults are the article's synthetic setting: ``L = 12``, the core prices
and preserved preparation after a failure.
"""
    mobilization_lead: int = 12
    downtime_steps: int = 6
    revenue: float = 255.
    replacement_cost: float = 520.
    downtime_cost: float = 25.
    failure_cost: float = 7000.
    recovery_contract: str = "preserve_preparation_v1"

    def __post_init__(self) -> None:
        if self.recovery_contract not in ("immediate_exchange_v1", "preserve_preparation_v1"):
            raise ValueError("unknown recovery contract")
        if self.mobilization_lead < 1 or self.downtime_steps < 0:
            raise ValueError("invalid event clock")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Economics":
        """The economics of a validated core configuration."""
        return cls(int(config["mobilization_lead"]), int(config["downtime_steps"]),
                   float(config["revenue_per_operating_step"]), float(config["replacement_cost"]),
                   float(config["downtime_cost_per_step"]), float(config["failure_event_cost"]),
                   str(config.get("recovery_contract", "immediate_exchange_v1")))


def replay(last_productive: Sequence[int] | np.ndarray, orders: np.ndarray,
           economics: Economics = Economics()) -> dict[str, np.ndarray]:
    """Outcome of first orders (``-1`` for none) on lives with last productive step ``T``.

``orders`` may carry leading axes (candidates or runs) before the life axis.
Returns float64 ``cash`` and ``calendar``, boolean ``failed``, the unused steps
before the last safe order, ``unused_fraction`` (unused steps over ``T - L``)
and the ``margin`` (last safe order minus the order; NaN without an order).
"""
    lives = np.asarray(last_productive, dtype=np.int64)
    first = np.asarray(orders, dtype=np.int64)
    lead, downtime = economics.mobilization_lead, economics.downtime_steps
    if lives.ndim != 1 or first.shape[-1:] != lives.shape or np.any(lives <= lead):
        raise ValueError("expected a life axis with T > L and matching orders")
    if np.any((first < -1) | (first > lives)):
        raise ValueError("an order must be -1 or a decision step in [0, T]")
    present = first >= 0
    last_safe = lives - lead
    failed = (~present) | (first > last_safe)
    unused = np.where(failed, 0, last_safe - first)
    if economics.recovery_contract == "preserve_preparation_v1":
        recovery = np.where(failed, np.where(present, lead - (lives - first), lead), 0)
    else:
        recovery = np.zeros(first.shape, dtype=np.int64)
    operated = np.where(failed, lives, first + lead)
    calendar = operated + 1 + downtime + recovery
    cash = (economics.revenue * operated - economics.replacement_cost
            - economics.downtime_cost * (1 + downtime + recovery) - economics.failure_cost * failed)
    return {"cash": cash.astype(np.float64), "calendar": calendar.astype(np.float64), "failed": failed,
            "unused": unused, "unused_fraction": unused / last_safe,
            "margin": np.where(present, last_safe - first, np.nan)}


def reference_orders(last_productive: Sequence[int] | np.ndarray, mobilization_lead: int = 12) -> np.ndarray:
    """First orders of the last-safe reference (oracle): exactly at ``T - L``."""
    return np.asarray(last_productive, dtype=np.int64) - int(mobilization_lead)


# --- Causal decision states of whole banks ---

@dataclass(frozen=True, eq=False)
class DecisionSeries:
    """Causal states at every decision of every active life of a bank, without an order.

``states[a:b]`` holds the decisions of life ``i`` for ``(a, b) = slices[i]``,
one per operating age from 0 to ``T``; ``measured_health`` holds the
measurements behind them (the health itself for exact banks).
"""
    states: np.ndarray
    slices: tuple[tuple[int, int], ...]
    families: tuple[str, ...]
    last_productive: np.ndarray
    pair_index: np.ndarray
    mobilization_lead: int = 12
    observation_route: str = "RAW_CAUSAL"
    sensor_level: str = "exact"
    measured_health: np.ndarray | None = field(default=None, compare=False)

    @property
    def lives(self) -> int:
        return len(self.slices)

    @property
    def health(self) -> np.ndarray:
        return self.states[:, 0].astype(np.float64)

    def life_states(self, index: int) -> np.ndarray:
        start, stop = self.slices[index]
        return self.states[start:stop]


def decision_series(bank: Mapping[str, Any], *, mobilization_lead: int = 12, observation_route: str = "RAW_CAUSAL",
                    replay_seed: int | None = None) -> DecisionSeries:
    """Reconstruct the causal states of every active life of ``bank`` under WAIT.

Noisy banks draw their measurements from the component streams with the bank's
measurement seed (or ``replay_seed``), exactly as the first reset of a
``ScenarioBankEnv`` without a seed does. Exact measurements equal the health.
"""
    replay_seed = bank["measurement_seed"] if replay_seed is None else replay_seed
    exact = bank["sensor_level"] == "exact"
    values, measured_values, slices, families, lives, pairs = [], [], [], [], [], []
    cursor = 0
    for pair in bank["pairs"]:
        part = pair["active"]
        if exact:
            measured, available = np.asarray(part["health"], dtype=np.float32), None
        else:
            paths = component_measurements(part, bank["sensor_level"], replay_seed)
            measured, available = paths["measured_health"], paths["available"]
        states = reconstruct(measured, available, observation_route, mobilization_lead)["states"][:-1]
        if states.shape != (part["eol_operating_age"], 4) or not np.isfinite(states).all():
            raise ValueError("malformed causal state sequence")
        values.append(states)
        measured_values.append(np.asarray(measured, dtype=np.float32)[:-1])
        slices.append((cursor, cursor + len(states)))
        cursor += len(states)
        families.append(str(part["family_id"]))
        lives.append(int(part["eol_operating_age"]) - 1)
        pairs.append(int(pair["pair_index"]))
    return DecisionSeries(np.vstack(values), tuple(slices), tuple(families), np.asarray(lives, dtype=np.int64),
                          np.asarray(pairs, dtype=np.int64), int(mobilization_lead), observation_route,
                          bank["sensor_level"], np.concatenate(measured_values))


# --- First orders ---

def first_orders(actions: np.ndarray, slices: Sequence[tuple[int, int]]) -> np.ndarray:
    """First index of a true action within each life, ``-1`` without one."""
    values = np.asarray(actions, dtype=bool)
    result = np.full(len(slices), -1, dtype=np.int64)
    for life, (start, stop) in enumerate(slices):
        hit = np.flatnonzero(values[start:stop])
        if len(hit):
            result[life] = int(hit[0])
    return result


def first_at_or_below(values: np.ndarray, slices: Sequence[tuple[int, int]], thresholds: Sequence[float]) -> np.ndarray:
    """First index with ``value <= threshold`` for every threshold (rows) and life (columns); -1 if none."""
    negative = -np.asarray(thresholds, dtype=np.float64)
    result = np.full((len(negative), len(slices)), -1, dtype=np.int64)
    for life, (start, stop) in enumerate(slices):
        running = -np.minimum.accumulate(np.asarray(values[start:stop], dtype=np.float64))
        index = np.searchsorted(running, negative, side="left")
        result[:, life] = np.where(index < stop - start, index, -1)
    return result


def first_at_or_above(values: np.ndarray, slices: Sequence[tuple[int, int]], thresholds: Sequence[float]) -> np.ndarray:
    """First index with ``value >= threshold`` for every threshold (rows) and life (columns); -1 if none."""
    cuts = np.asarray(thresholds, dtype=np.float64)
    result = np.full((len(cuts), len(slices)), -1, dtype=np.int64)
    for life, (start, stop) in enumerate(slices):
        running = np.maximum.accumulate(np.asarray(values[start:stop], dtype=np.float64))
        index = np.searchsorted(running, cuts, side="left")
        result[:, life] = np.where(index < stop - start, index, -1)
    return result


def projected_time(states: np.ndarray, mobilization_lead: int = 12) -> np.ndarray:
    """Projected remaining time ``(1 - d) / max(rate / L, 1e-12)`` per state; infinite without a valid rate.

The scalar arithmetic is that of the core's projection rule.
"""
    lead = int(mobilization_lead)
    return np.asarray([math.inf if s[2] < .5 else 0. if s[0] <= 0 else
                       (1. - (-math.log(float(s[0])) / math.log(5.))) / max(float(s[1]) / lead, 1.e-12)
                       for s in np.asarray(states, dtype=np.float32)], dtype=np.float64)


def threshold_candidates(grids: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    """Single-indicator candidates in grid order: ``{"family", "threshold", "candidate_id"}``."""
    unknown = set(grids) - set(RULE_FAMILIES)
    if unknown:
        raise ValueError(f"rule families must be among {RULE_FAMILIES}")
    candidates = []
    for family, values in grids.items():
        for value in values:
            candidates.append({"family": family, "threshold": float(value), "candidate_id": len(candidates)})
    return candidates


def threshold_orders(series: DecisionSeries, candidates: Sequence[Mapping[str, Any]],
                     projection: np.ndarray | None = None) -> np.ndarray:
    """First orders of single-indicator rules: health or projected time at or below the threshold."""
    if any(candidate["family"] not in RULE_FAMILIES for candidate in candidates):
        raise ValueError(f"rule families must be among {RULE_FAMILIES}")
    result = np.full((len(candidates), series.lives), -1, dtype=np.int64)
    rows = {family: [i for i, candidate in enumerate(candidates) if candidate["family"] == family]
            for family in RULE_FAMILIES}
    if rows["projection"] and projection is None:
        projection = projected_time(series.states, series.mobilization_lead)
    for family, values in (("health", series.health), ("projection", projection)):
        if rows[family]:
            result[rows[family]] = first_at_or_below(values, series.slices,
                                                     [candidates[i]["threshold"] for i in rows[family]])
    return result


def two_indicator_orders(series: DecisionSeries, split: Sequence[float], a_values: Sequence[float],
                         b_values: Sequence[float], projection: np.ndarray | None = None) -> np.ndarray:
    """First orders of ``(health > s and T <= a) or (health <= s and T <= b)``; shape ``(s, a, b, lives)``."""
    projection = projected_time(series.states, series.mobilization_lead) if projection is None else projection
    health = series.health
    a_negative, b_negative = -np.asarray(a_values, dtype=np.float64), -np.asarray(b_values, dtype=np.float64)
    result = np.full((len(split), len(a_values), len(b_values), series.lives), -1, dtype=np.int16)
    for life, (start, stop) in enumerate(series.slices):
        h, t, count = health[start:stop], projection[start:stop], stop - start
        for k, level in enumerate(split):
            above = h > level
            p = np.searchsorted(-np.minimum.accumulate(np.where(above, t, np.inf)), a_negative, side="left")
            q = np.searchsorted(-np.minimum.accumulate(np.where(above, np.inf, t)), b_negative, side="left")
            first = np.minimum(p[:, None], q[None, :])
            result[k, :, :, life] = np.where(first < count, first, -1)
    return result


def age_orders(series: DecisionSeries, ages: Sequence[int]) -> np.ndarray:
    """Age replacement: the order falls at age ``a`` when the life still has a decision there, else -1."""
    lengths = np.asarray([stop - start for start, stop in series.slices])
    values = np.asarray(ages, dtype=np.int64)[:, None]
    return np.where(values < lengths[None, :], values, -1).astype(np.int64)


def policy_probabilities(policy: Any, states: np.ndarray) -> np.ndarray:
    """Float32 action probabilities of a PPO agent or policy for a whole state array in one batch.

Accepts an object with ``batch_output`` (``PPOAgent``), ``probabilities_batch``
(a loaded ``Policy``) or a callable returning ``(n, 2)`` probabilities.
"""
    values = np.ascontiguousarray(np.asarray(states, dtype=np.float32))
    if hasattr(policy, "batch_output"):
        probabilities = policy.batch_output(values)[1]
    elif hasattr(policy, "probabilities_batch"):
        probabilities = policy.probabilities_batch(values)
    else:
        probabilities = policy(values)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (len(values), 2) or not np.isfinite(probabilities).all():
        raise ValueError("expected finite (n, 2) action probabilities")
    return probabilities


def greedy_orders(probabilities: np.ndarray, slices: Sequence[tuple[int, int]]) -> np.ndarray:
    """First orders of the standard decision: ORDER when float32 P(ORDER) > P(WAIT)."""
    values = np.asarray(probabilities, dtype=np.float32)
    return first_orders(values[:, 1] > values[:, 0], slices)


def cut_orders(probabilities: np.ndarray, slices: Sequence[tuple[int, int]],
               cuts: Sequence[float] = CUT_GRID) -> np.ndarray:
    """First orders of ORDER when float64(P(ORDER)) >= cut, for every cut (rows) and life (columns)."""
    values = np.asarray(probabilities, dtype=np.float32)
    return first_at_or_above(values[:, 1].astype(np.float64), slices, cuts)


# --- Equal-law scores and selection ---

@dataclass(frozen=True, eq=False)
class Scores:
    """Equal-law calibration scores of candidates (one entry per candidate)."""
    rate: np.ndarray
    failure_share: np.ndarray
    unused_share: np.ndarray
    laws: tuple[str, ...]
    law_rate: np.ndarray
    law_failure_share: np.ndarray

    def candidate(self, index: int) -> dict[str, Any]:
        return {"profit_rate": float(self.rate[index]), "failure_share": float(self.failure_share[index]),
                "unused_share": float(self.unused_share[index]),
                "laws": {law: {"profit_rate": float(self.law_rate[index, k]),
                               "failure_share": float(self.law_failure_share[index, k])}
                         for k, law in enumerate(self.laws)}}


def law_groups(families: Sequence[str]) -> tuple[tuple[str, ...], list[np.ndarray]]:
    """Laws in sorted order and the life indices of each."""
    labels = np.asarray(families)
    laws = tuple(sorted(set(labels.tolist())))
    return laws, [np.flatnonzero(labels == law) for law in laws]


def score_orders(series: DecisionSeries, orders: np.ndarray, economics: Economics, *, chunk: int = 20000) -> Scores:
    """Equal-law calendar profit, failure share and unused share of every row of first orders.

The profit rate of a law is its total cash over its total calendar time; the
score is the mean over the laws. The failure and unused shares are means over
each law's lives, then over the laws.
"""
    matrix = np.atleast_2d(np.asarray(orders))
    laws, index = law_groups(series.families)
    counts = np.asarray([len(rows) for rows in index], dtype=np.float64)
    parts = {key: [] for key in ("cash", "calendar", "failed", "unused_fraction")}
    for start in range(0, len(matrix), chunk):
        outcome = replay(series.last_productive, matrix[start:start + chunk].astype(np.int64), economics)
        for key in parts:
            values = np.asarray(outcome[key], dtype=np.float64)
            parts[key].append(np.stack([values[..., rows].sum(axis=-1) for rows in index], axis=-1))
    sums = {key: np.concatenate(value) for key, value in parts.items()}
    law_rate = sums["cash"] / sums["calendar"]
    law_failure = sums["failed"] / counts
    return Scores(law_rate.mean(axis=-1), law_failure.mean(axis=-1), (sums["unused_fraction"] / counts).mean(axis=-1),
                  laws, law_rate, law_failure)


def select_maximum(rates: Sequence[float], failure_shares: Sequence[float], tiebreak: Sequence[float] | None = None,
                   tolerance: float = RATE_TIE) -> int:
    """Index of the highest rate; within ``tolerance`` the lower failure share, the lower tie-break, the lower index."""
    rate, failure = np.asarray(rates, dtype=np.float64), np.asarray(failure_shares, dtype=np.float64)
    order = np.zeros(len(rate)) if tiebreak is None else np.asarray(tiebreak, dtype=np.float64)
    if rate.ndim != 1 or not len(rate) or failure.shape != rate.shape or order.shape != rate.shape:
        raise ValueError("expected equally long nonempty candidate arrays")
    tied = np.flatnonzero(rate >= rate.max() - tolerance)
    return int(min(tied.tolist(), key=lambda i: (failure[i], order[i], i)))


def select_failure_budget(failure_shares: Sequence[float], unused_shares: Sequence[float],
                          limit: float = .01) -> int | None:
    """Index of the least unused share among candidates with a failure share at most ``limit`` (None if none)."""
    failure, unused = np.asarray(failure_shares, dtype=np.float64), np.asarray(unused_shares, dtype=np.float64)
    allowed = np.flatnonzero(failure <= limit)
    return None if not len(allowed) else int(min(allowed.tolist(), key=lambda i: (unused[i], failure[i], i)))


def anchor_distance(candidates: Sequence[Mapping[str, Any]], anchors: Mapping[str, float] = TIE_ANCHORS) -> np.ndarray:
    """Tie-break of single-indicator rules: the distance of each threshold to its family's anchor."""
    return np.asarray([abs(candidate["threshold"] - anchors[candidate["family"]]) for candidate in candidates])


def calibrate_rule(series: DecisionSeries, grids: Mapping[str, Sequence[float]], *, economics: Economics,
                   anchors: Mapping[str, float] = TIE_ANCHORS, projection: np.ndarray | None = None) -> dict[str, Any]:
    """Rule calibration: the health or projected-time threshold of the highest equal-law profit rate.

Ties go to the lower failure share, the threshold closer to its anchor
(health 0.36 and projected time 15, or ``L + 3`` in the preparation-time study)
and the lower candidate index.
"""
    candidates = threshold_candidates(grids)
    scores = score_orders(series, threshold_orders(series, candidates, projection), economics)
    chosen = select_maximum(scores.rate, scores.failure_share, anchor_distance(candidates, anchors))
    return {"selected": {**candidates[chosen], **scores.candidate(chosen)},
            "candidates": [{**candidate, **scores.candidate(i)} for i, candidate in enumerate(candidates)]}


def calibrate_per_law(series: DecisionSeries, grids: Mapping[str, Sequence[float]], *, economics: Economics,
                      anchors: Mapping[str, float] = TIE_ANCHORS, projection: np.ndarray | None = None
                      ) -> dict[str, dict[str, Any]]:
    """Privileged per-law rule: one single-indicator threshold per law, each calibrated on that law's lives."""
    candidates = threshold_candidates(grids)
    scores = score_orders(series, threshold_orders(series, candidates, projection), economics)
    distance = anchor_distance(candidates, anchors)
    result = {}
    for k, law in enumerate(scores.laws):
        chosen = select_maximum(scores.law_rate[:, k], scores.law_failure_share[:, k], distance)
        result[law] = {**candidates[chosen], "profit_rate": float(scores.law_rate[chosen, k]),
                       "failure_share": float(scores.law_failure_share[chosen, k])}
    return result


def per_law_orders(series: DecisionSeries, selection: Mapping[str, Mapping[str, Any]],
                   projection: np.ndarray | None = None) -> np.ndarray:
    """First orders of the per-law rule: each life uses its own law's threshold."""
    result = np.full(series.lives, -1, dtype=np.int64)
    laws, index = law_groups(series.families)
    for law, rows in zip(laws, index, strict=True):
        result[rows] = threshold_orders(series, [selection[law]], projection)[0, rows]
    return result


def _indicator(series: DecisionSeries, candidate: Mapping[str, Any], projection: np.ndarray) -> np.ndarray:
    values = series.health if candidate["family"] == "health" else projection
    return values <= candidate["threshold"]


def dual_threshold_orders(raw: DecisionSeries, line: DecisionSeries, candidates: Sequence[Mapping[str, Any]],
                          combine: str, projections: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """First orders of a threshold rule on two state estimates of the same lives.

``combine="both"`` orders where both estimates meet the threshold, ``"either"``
where at least one does (consecutive differences ``raw`` and the line fit ``line``).
"""
    if combine not in ("both", "either"):
        raise ValueError("combine must be both or either")
    if raw.slices != line.slices or not np.array_equal(raw.pair_index, line.pair_index):
        raise ValueError("both estimates must describe the same lives")
    if projections is None:
        projections = (projected_time(raw.states, raw.mobilization_lead), projected_time(line.states, line.mobilization_lead))
    result = np.full((len(candidates), raw.lives), -1, dtype=np.int64)
    for row, candidate in enumerate(candidates):
        a, b = _indicator(raw, candidate, projections[0]), _indicator(line, candidate, projections[1])
        result[row] = first_orders(a & b if combine == "both" else a | b, raw.slices)
    return result


def calibrate_dual_rule(raw: DecisionSeries, line: DecisionSeries, grids: Mapping[str, Sequence[float]], combine: str,
                        *, economics: Economics, anchors: Mapping[str, float] = TIE_ANCHORS) -> dict[str, Any]:
    """``calibrate_rule`` for a rule on two estimates (``both`` or ``either``), with the same objective and ties."""
    candidates = threshold_candidates(grids)
    projections = (projected_time(raw.states, raw.mobilization_lead), projected_time(line.states, line.mobilization_lead))
    scores = score_orders(raw, dual_threshold_orders(raw, line, candidates, combine, projections), economics)
    chosen = select_maximum(scores.rate, scores.failure_share, anchor_distance(candidates, anchors))
    return {"selected": {**candidates[chosen], "combine": combine, **scores.candidate(chosen)},
            "candidates": [{**candidate, **scores.candidate(i)} for i, candidate in enumerate(candidates)]}


def calibrate_age(series: DecisionSeries, ages: Sequence[int], *, economics: Economics) -> dict[str, Any]:
    """Age replacement: the operating age of the highest equal-law profit rate (ties: failures, lower age index)."""
    values = [int(age) for age in ages]
    scores = score_orders(series, age_orders(series, values), economics)
    chosen = select_maximum(scores.rate, scores.failure_share)
    return {"selected": {"age": values[chosen], "candidate_id": chosen, **scores.candidate(chosen)}}


def two_indicator_labels(single: Sequence[Mapping[str, Any]], blocks: Sequence[Mapping[str, Sequence[float]]]
                         ) -> list[dict[str, Any]]:
    """Candidate descriptions of the two-indicator rule in candidate order: single rules, then each grid block."""
    labels = [{"kind": "single", "family": c["family"], "threshold": c["threshold"]} for c in single]
    for number, block in enumerate(blocks):
        labels += [{"kind": f"block_{number}", "health_split": s, "a_above_split": a, "b_at_or_below_split": b}
                   for s in block["health_split"] for a in block["a"] for b in block["b"]]
    return labels


def two_indicator_candidate_orders(series: DecisionSeries, single: Sequence[Mapping[str, Any]],
                                   blocks: Sequence[Mapping[str, Sequence[float]]],
                                   projection: np.ndarray | None = None) -> np.ndarray:
    """First orders of all two-indicator candidates (int16): the single rules, then each ``(s, a, b)`` block."""
    projection = projected_time(series.states, series.mobilization_lead) if projection is None else projection
    rows = [threshold_orders(series, single, projection).astype(np.int16)]
    for block in blocks:
        rows.append(two_indicator_orders(series, block["health_split"], block["a"], block["b"], projection)
                    .reshape(-1, series.lives))
    return np.concatenate(rows)


def calibrate_two_indicator(series: DecisionSeries, single: Sequence[Mapping[str, Any]],
                            blocks: Sequence[Mapping[str, Sequence[float]]], *, economics: Economics,
                            projection: np.ndarray | None = None) -> dict[str, Any]:
    """Two-indicator rule: order when ``T <= a`` above the health level ``s`` and when ``T <= b`` at or below it.

The candidates include every single-indicator rule, so the selection cannot do
worse than those on the calibration lives. Ties go to the lower failure share,
then the lower candidate index.
"""
    orders = two_indicator_candidate_orders(series, single, blocks, projection)
    scores = score_orders(series, orders, economics)
    chosen = select_maximum(scores.rate, scores.failure_share)
    labels = two_indicator_labels(single, blocks)
    return {"selected": {**labels[chosen], "candidate_id": chosen, **scores.candidate(chosen)},
            "candidates": len(labels)}


def two_indicator_rule_orders(series: DecisionSeries, label: Mapping[str, Any],
                              projection: np.ndarray | None = None) -> np.ndarray:
    """First orders of one selected two-indicator (or single-indicator) rule."""
    if label.get("kind") == "single" or "family" in label:
        return threshold_orders(series, [label], projection)[0]
    return two_indicator_orders(series, [label["health_split"]], [label["a_above_split"]],
                                [label["b_at_or_below_split"]], projection)[0, 0, 0].astype(np.int64)


def calibrate_cut(series: DecisionSeries, probabilities: np.ndarray, *, economics: Economics,
                  cuts: Sequence[float] = CUT_GRID, anchor_index: int = CUT_ANCHOR_INDEX,
                  failure_budget: float | None = None) -> dict[str, Any]:
    """Calibrated ORDER cut of one network: ORDER when P(ORDER) >= cut, at the highest equal-law profit rate.

Ties go to the lower failure share, the cut index closer to ``anchor_index``
(0.5 on ``CUT_GRID``) and the lower index. With ``failure_budget`` the cut
minimizing unused life within that failure share is reported as well.
"""
    scores = score_orders(series, cut_orders(probabilities, series.slices, cuts), economics)
    chosen = select_maximum(scores.rate, scores.failure_share, np.abs(np.arange(len(cuts)) - anchor_index))
    result = {"selected": {"cut": float(cuts[chosen]), "cut_index": chosen, **scores.candidate(chosen)}}
    if failure_budget is not None:
        budget = select_failure_budget(scores.failure_share, scores.unused_share, failure_budget)
        result["failure_budget"] = None if budget is None else {"cut": float(cuts[budget]), "cut_index": budget,
                                                                 **scores.candidate(budget)}
    return result


def outcome_rows(method: str, series: DecisionSeries, orders: np.ndarray, economics: Economics, *,
                 run: int | str | None = None, group: str | Callable[[int], str] | None = None,
                 units: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    """One outcome row per life with the columns of the experiment outcome tables.

``group`` defaults to each life's law; ``units`` default to the bank's pair indices.
"""
    order = np.asarray(orders, dtype=np.int64)
    outcome = replay(series.last_productive, order, economics)
    rows = []
    for life in range(series.lives):
        label = series.families[life] if group is None else (group(life) if callable(group) else group)
        rows.append({"method": method, "group": label, "run": "" if run is None else run,
                     "unit": int(series.pair_index[life]) if units is None else units[life],
                     "last_productive_step": int(series.last_productive[life]),
                     "order_step": "" if order[life] < 0 else int(order[life]),
                     "cash": float(outcome["cash"][life]), "calendar_steps": int(outcome["calendar"][life]),
                     "failed": bool(outcome["failed"][life]),
                     "unused_fraction": float(outcome["unused_fraction"][life])})
    return rows


__all__ = ["CUT_ANCHOR_INDEX", "CUT_GRID", "DecisionSeries", "Economics", "RATE_TIE", "Scores", "TIE_ANCHORS",
           "age_orders", "anchor_distance", "calibrate_age", "calibrate_cut", "calibrate_dual_rule", "calibrate_per_law",
           "calibrate_rule", "calibrate_two_indicator", "cut_orders", "decision_series", "dual_threshold_orders",
           "first_at_or_above", "first_at_or_below",
           "first_orders", "greedy_orders", "grid", "law_groups", "outcome_rows", "per_law_orders",
           "policy_probabilities", "projected_time", "reference_orders", "replay", "score_orders",
           "select_failure_budget", "select_maximum", "threshold_candidates", "threshold_orders",
           "two_indicator_candidate_orders", "two_indicator_labels", "two_indicator_orders",
           "two_indicator_rule_orders"]
