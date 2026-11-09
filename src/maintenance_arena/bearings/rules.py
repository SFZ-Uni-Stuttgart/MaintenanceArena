"""Threshold rules on bearing decision series: the Indicator rule and forecast rules.

The forecast rule is the Time rule of the article on the bearing prediction.
A rule orders at the first minute whose score is at most its threshold. The Indicator
rule searches five families: age (score -minute), smoothed RMS level and smoothed peak
level (score minus the level, so the threshold is minus the level that triggers), the
forecast r and its trend r + k slope(r). Calibration maximizes pooled training profit,
total cash over total calendar time, and the first candidate wins ties. A forecast rule
orders when the float32 forecast coordinate of the policy input, r/125, reaches its cutoff.

The broader search (``search_rule_families``) adds a guard during running-in and two
combinations with age to these conditions, sixteen families in all, and chooses a family
by inner cross-validation on the training bearings.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

from .economics import ReplayEconomics, pooled_rate
from .features import RULE_SMOOTHING

FAMILIES = ("age", "level", "peak", "forecast", "trend")
GUARD_MINUTE = 10


@dataclass(frozen=True)
class RuleGrid:
    """Candidate thresholds of the Indicator rule.

    ``age`` is (start, stop, count) of a geometric grid, rounded to whole minutes and
    de-duplicated; ``level``, ``peak``, ``forecast`` and ``trend`` are (start, stop, step)
    grids including stop. ``smoothing`` lists the EWM rates of the level and peak families
    and ``trend_k`` the trend weights.
    """
    age: tuple[float, float, int] = (5, 3000, 240)
    smoothing: tuple[float, ...] = RULE_SMOOTHING
    level: tuple[float, float, float] = (0.0, 4.0, 0.05)
    peak: tuple[float, float, float] = (0.0, 4.0, 0.05)
    forecast: tuple[float, float, float] = (0.0, 90.0, 0.5)
    trend_k: tuple[float, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    trend: tuple[float, float, float] = (0.0, 90.0, 1.0)

    def __post_init__(self):
        for name in ("age", "smoothing", "level", "peak", "forecast", "trend_k", "trend"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.age) != 3 or any(len(getattr(self, n)) != 3 for n in ("level", "peak", "forecast", "trend")):
            raise ValueError("grid ranges need three values")
        if not self.smoothing or not self.trend_k:
            raise ValueError("smoothing rates and trend weights cannot be empty")
        if any(getattr(self, n)[2] <= 0 for n in ("level", "peak", "forecast", "trend")):
            raise ValueError("grid steps must be positive")


@dataclass(frozen=True)
class RuleCandidate:
    """One rule: family, its parameter (smoothing rate or trend weight, else 0) and threshold."""
    family: str
    parameter: float
    threshold: float


@dataclass(frozen=True)
class RuleSelection:
    candidate: RuleCandidate
    training_profit_rate: float
    training_failures: int


@dataclass(frozen=True)
class ForecastRuleSelection:
    cutoff: float
    training_profit_rate: float
    training_failures: int
    candidates: int


def grid_values(start: float, stop: float, step: float) -> np.ndarray:
    """start, start + step, ... up to and including stop."""
    return np.arange(start, stop + 1e-9, step)


def indicator_candidates(grid: RuleGrid | None = None) -> list[RuleCandidate]:
    """All candidates in search order: age, level, peak, forecast, trend."""
    grid = grid or RuleGrid()
    ages = np.unique(np.round(np.geomspace(*grid.age)))
    result = [RuleCandidate("age", 0.0, -float(a)) for a in ages]
    for family in ("level", "peak"):
        result += [RuleCandidate(family, float(a), -float(x)) for a in grid.smoothing
                   for x in grid_values(*getattr(grid, family))]
    result += [RuleCandidate("forecast", 0.0, float(x)) for x in grid_values(*grid.forecast)]
    result += [RuleCandidate("trend", float(k), float(x)) for k in grid.trend_k for x in grid_values(*grid.trend)]
    return result


def first_trigger(score: np.ndarray, threshold: float) -> int | None:
    """First 1-based minute with score <= threshold, or None."""
    hits = np.flatnonzero(np.asarray(score) <= threshold)
    return None if hits.size == 0 else int(hits[0] + 1)


def rule_score(candidate: RuleCandidate, series: Mapping[str, np.ndarray]) -> np.ndarray:
    family, parameter = candidate.family, candidate.parameter
    try:
        if family == "age":
            return -series["minute"]
        if family in ("level", "peak"):
            return -series[f"{family}_a{parameter:g}"]
        if family == "forecast":
            return series["forecast"]
        if family == "trend":
            return series["forecast"] + parameter * series["forecast_trend"]
    except KeyError as error:
        raise ValueError(f"the decision series lacks {error.args[0]!r} for the {family} rule") from None
    raise ValueError(f"unknown rule family {family!r}; expected one of {FAMILIES}")


def apply_rule(candidate: RuleCandidate, series: Mapping[str, np.ndarray]) -> int | None:
    """First order minute of a rule on one decision series."""
    return first_trigger(rule_score(candidate, series), candidate.threshold)


def calibrate_indicator_rule(series: Sequence[Mapping[str, np.ndarray]], economics: ReplayEconomics | None = None,
                             grid: RuleGrid | None = None) -> dict[str, RuleSelection]:
    """Best candidate of every family on training series and the overall best under ``"best"``.

    Series must contain forecasts. Each candidate is scored by the pooled profit rate of
    its first orders on all training series.
    """
    economics = economics or ReplayEconomics()
    if not series:
        raise ValueError("calibration needs at least one training series")
    best: dict[str, RuleSelection] = {}
    for candidate in indicator_candidates(grid):
        outcomes = [economics.outcome(len(s["minute"]), apply_rule(candidate, s)) for s in series]
        rate = pooled_rate(outcomes)
        current = best.get(candidate.family)
        if current is None or rate > current.training_profit_rate:
            best[candidate.family] = RuleSelection(candidate, rate, sum(o.failed for o in outcomes))
    best["best"] = max(best.values(), key=lambda selection: selection.training_profit_rate)
    return best


def forecast_rule_order(forecast_input: np.ndarray, cutoff: float) -> int | None:
    """First minute whose forecast coordinate r/125 is at most float32(cutoff/125)."""
    values = np.asarray(forecast_input)
    if values.dtype != np.float32 or values.ndim != 1:
        raise ValueError("the forecast input is the float32 first column of a policy input matrix")
    return first_trigger(values, np.float32(cutoff / 125.0))


def calibrate_forecast_rule(forecast_inputs: Sequence[np.ndarray], cutoffs: Sequence[float],
                            economics: ReplayEconomics | None = None) -> ForecastRuleSelection:
    """Cutoff in minutes with the highest pooled training profit; the smallest wins ties."""
    economics = economics or ReplayEconomics()
    if not forecast_inputs or len(cutoffs) == 0:
        raise ValueError("calibration needs training inputs and at least one cutoff")
    selected = None
    for cutoff in cutoffs:
        outcomes = [economics.outcome(len(x), forecast_rule_order(x, float(cutoff))) for x in forecast_inputs]
        rate = pooled_rate(outcomes)
        if selected is None or rate > selected[1]:
            selected = (float(cutoff), rate, sum(o.failed for o in outcomes))
    return ForecastRuleSelection(*selected, len(cutoffs))


@dataclass(frozen=True)
class RuleFamily:
    """A family of the broader rule search, built on one indicator condition.

    ``kind`` is ``single`` (an indicator Rule family), ``guarded`` (no order before minute 10,
    when the running-in median is complete), ``after_age`` (the condition once a minimum age is
    reached) or ``or_age`` (from minute 10, the condition or an age limit). The ages of the two
    combinations are the age-rule grid values of at least 10 minutes.
    """
    name: str
    base: str
    kind: str


_LABELS = {"age": "age", "level": "level", "peak": "peak", "forecast": "forecast", "trend": "forecast_trend"}
RULE_FAMILIES = (*(RuleFamily(_LABELS[b], b, "single") for b in FAMILIES),
                 *(RuleFamily("guarded_" + _LABELS[b], b, "guarded") for b in FAMILIES),
                 *(RuleFamily(_LABELS[b] + "_after_age", b, "after_age") for b in ("level", "peak", "forecast")),
                 *(RuleFamily(_LABELS[b] + "_or_age", b, "or_age") for b in ("level", "peak", "forecast")))
_NO_ORDER = 10 ** 6


@lru_cache(maxsize=8)
def _family_grid(grid: RuleGrid):
    candidates = indicator_candidates(grid)
    conditions = {base: tuple(c for c in candidates if c.family == base) for base in FAMILIES}
    groups = {}
    for base in FAMILIES:
        grouped: list[tuple[float, list[float]]] = []
        for c in conditions[base]:
            if not grouped or grouped[-1][0] != c.parameter:
                grouped.append((c.parameter, []))
            grouped[-1][1].append(c.threshold)
        groups[base] = tuple((k, np.asarray(t, dtype=np.float64)) for k, t in grouped)
    ages = sorted({-c.threshold for c in conditions["age"]})
    return groups, conditions, np.asarray([int(a) for a in ages if a >= GUARD_MINUTE], dtype=np.int64)


def family_size(family: RuleFamily, grid: RuleGrid | None = None) -> int:
    """Number of candidates of a family: conditions, times ages for the age combinations."""
    _, conditions, ages = _family_grid(grid or RuleGrid())
    return len(conditions[family.base]) * (len(ages) if family.kind in ("after_age", "or_age") else 1)


def family_parameters(family: RuleFamily, index: int, grid: RuleGrid | None = None) -> dict:
    """Parameters of candidate ``index`` of a family, in the search order of ``family_orders``."""
    _, conditions, ages = _family_grid(grid or RuleGrid())
    if not 0 <= index < family_size(family, grid):
        raise ValueError(f"{family.name} has no candidate {index}")
    two = family.kind in ("after_age", "or_age")
    c = conditions[family.base][index // len(ages) if two else index]
    result = {"family": family.name, "condition": c.family, "parameter": c.parameter, "threshold": c.threshold}
    if family.kind in ("guarded", "or_age"):
        result["guard_minute"] = GUARD_MINUTE
    if two:
        result["minimum_age" if family.kind == "after_age" else "age_limit"] = int(ages[index % len(ages)])
    return result


def family_orders(family: RuleFamily, series: Mapping[str, np.ndarray], grid: RuleGrid | None = None) -> np.ndarray:
    """First order minute of every candidate of a family on one decision series; 0 means no order.

    Candidates follow the indicator-candidate order of the family's condition and, for the age
    combinations, all ages of each condition.
    """
    groups, _, ages = _family_grid(grid or RuleGrid())
    if family.kind not in ("single", "guarded", "after_age", "or_age"):
        raise ValueError(f"unknown family kind {family.kind!r}")
    life = len(series["minute"])
    minute = np.arange(1, life + 1)
    out = []
    for parameter, thresholds in groups[family.base]:
        hits = rule_score(RuleCandidate(family.base, parameter, 0.0), series)[None, :] <= thresholds[:, None]
        if family.kind in ("guarded", "or_age"):
            hits[:, :GUARD_MINUTE - 1] = False
        if family.kind in ("single", "guarded"):
            out.append(np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0))
            continue
        following = np.minimum.accumulate(np.where(hits, minute[None, :], _NO_ORDER)[:, ::-1], axis=1)[:, ::-1]
        if family.kind == "after_age":
            orders = np.full((len(thresholds), len(ages)), _NO_ORDER, dtype=np.int64)
            inside = ages <= life
            orders[:, inside] = following[:, ages[inside] - 1]
        else:
            first = following[:, GUARD_MINUTE - 1] if life >= GUARD_MINUTE else np.full(len(thresholds), _NO_ORDER)
            orders = np.minimum(first[:, None], np.where(ages <= life, ages, _NO_ORDER)[None, :])
        out.append(np.where(orders >= _NO_ORDER, 0, orders).reshape(-1))
    return np.concatenate(out).astype(np.int32)


class _OutcomeTable:
    """Cash and calendar minutes of every possible first order (0 = none) of every life."""

    def __init__(self, lives: Sequence[int], economics: ReplayEconomics):
        self.cash, self.calendar = [], []
        for life in lives:
            outcomes = [economics.outcome(life, None)] + [economics.outcome(life, o) for o in range(1, life + 1)]
            self.cash.append(np.asarray([o.cash for o in outcomes], dtype=np.float64))
            self.calendar.append(np.asarray([o.calendar for o in outcomes], dtype=np.float64))

    def rates(self, orders: Sequence[np.ndarray], members: Sequence[int]) -> np.ndarray:
        cash = sum(self.cash[i][orders[i]] for i in members)
        calendar = sum(self.calendar[i][orders[i]] for i in members)
        return cash / calendar


def search_rule_families(training: Mapping[str, Mapping[str, np.ndarray]], inner_folds: Sequence[Sequence[str]],
                         evaluation: Mapping[str, Mapping[str, np.ndarray]] | None = None, *,
                         economics: ReplayEconomics | None = None, grid: RuleGrid | None = None,
                         families: Sequence[RuleFamily] = RULE_FAMILIES) -> dict:
    """Refit every family on the training series and choose one family by inner cross-validation.

    Within a family, the candidate with the highest pooled training profit is selected, the
    first on ties. The inner cross-validation score of a family is the exact pooled profit, on
    each inner fold's series, of the candidates selected without that fold; the highest score
    wins, the first family on ties. ``training`` and ``evaluation`` map bearing names to
    decision series with forecasts; ``inner_folds`` partitions the training names. Returns per
    family the refit candidate, its training rate, the score and the evaluation orders, and the
    chosen family.
    """
    economics = economics or ReplayEconomics()
    grid = grid or RuleGrid()
    names = list(training)
    flat = [name for group in inner_folds for name in group]
    if not names or len(set(flat)) != len(flat) or set(flat) != set(names):
        raise ValueError("inner folds must partition the training bearings")
    evaluation = dict(evaluation or {})
    table = _OutcomeTable([len(training[n]["minute"]) for n in names], economics)
    position = {name: i for i, name in enumerate(names)}
    references = [economics.outcome(len(training[n]["minute"]), economics.oracle_order(len(training[n]["minute"])))
                  for n in names]
    reference = Fraction(sum(Fraction(o.cash) for o in references)) / sum(Fraction(o.calendar) for o in references)
    result, best, best_score = {}, None, None
    for rank, family in enumerate(families, start=1):
        orders = [family_orders(family, training[n], grid) for n in names]
        cash = calendar = Fraction(0)
        choices = []
        for k, group in enumerate(inner_folds):
            held = [position[n] for n in group]
            if not held:
                continue
            members = [i for i in range(len(names)) if i not in held]
            rates = table.rates(orders, members)
            index = int(np.argmax(rates))
            for i in held:
                order = int(orders[i][index])
                cash += Fraction(table.cash[i][order])
                calendar += Fraction(table.calendar[i][order])
            choices.append({"inner_fold": k, "evaluated": list(group), "candidate": family_parameters(family, index, grid),
                            "calibration_rate": float(rates[index])})
        score = cash / calendar
        rates = table.rates(orders, range(len(names)))
        index = int(np.argmax(rates))
        evaluation_orders = {name: int(family_orders(family, series, grid)[index]) or None
                             for name, series in evaluation.items()}
        result[family.name] = {"rank": rank, "kind": family.kind, "candidates": family_size(family, grid),
                               "inner_cv_rate": float(score), "inner_cv_retained_profit_pct": float(100 * score / reference),
                               "inner_choices": choices, "refit": family_parameters(family, index, grid),
                               "refit_index": index, "refit_training_rate": float(rates[index]),
                               "evaluation_orders": evaluation_orders, "_score": score}
        if best_score is None or score > best_score:
            best, best_score = family.name, score
    tied = [name for name, entry in result.items() if entry["_score"] == best_score]
    for entry in result.values():
        del entry["_score"]
    return {"chosen": best, "tied": tied, "training_reference_rate": float(reference), "families": result}
