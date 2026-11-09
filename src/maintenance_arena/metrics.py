"""Equal-life decision metrics of the accompanying article and their paired bootstrap.

Every physical life has equal weight. When a method has several runs (training
seeds) on the same lives, each life first averages its runs: the failure
indicator, the unused fraction, the cash and the calendar time. Then

* failure share: the mean failure indicator, in percent;
* unused life: the unused steps before the last safe order divided by the time
  from installation to that order, ``T - L``, in percent (a failed life has
  zero unused life, so failures and unused life are read together);
* retained profit: the life's mean cash over its mean calendar time, divided by
  the profit rate of the last-safe reference (an order exactly at ``T - L``) on
  the same life, in percent.

Each is averaged over lives. ``paired_bootstrap`` resamples whole lives together
with all their runs, the same lives for every method, and reports percentile
intervals for each method and for differences between methods.

The input rows have the columns of the experiment outcome tables: ``method``,
``group``, ``run``, ``unit``, ``last_productive_step``, ``order_step``, ``cash``,
``calendar_steps``, ``failed`` and ``unused_fraction``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .replay import Economics, replay

OUTCOME_COLUMNS = ("method", "group", "run", "unit", "last_productive_step", "order_step", "cash",
                   "calendar_steps", "failed", "unused_fraction")
METRICS = ("failure_pct", "unused_life_pct", "retained_profit_pct")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return math.fsum(items) / len(items)


@dataclass(frozen=True, eq=False)
class LifeTable:
    """Per-life values of one method, each averaged over the method's runs on that life."""
    units: tuple[Any, ...]
    runs: int
    last_productive: np.ndarray
    failure_pct: np.ndarray
    unused_life_pct: np.ndarray
    cash: np.ndarray
    calendar: np.ndarray
    retained_profit_pct: np.ndarray

    def metric(self, name: str) -> np.ndarray:
        if name not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        return getattr(self, name)

    def aligned(self, units: Sequence[Any]) -> "LifeTable":
        """The same table in the order of ``units`` (which must be the same lives)."""
        position = {unit: i for i, unit in enumerate(self.units)}
        if set(position) != set(units) or len(units) != len(position):
            raise ValueError("the methods were not evaluated on the same lives")
        index = np.asarray([position[unit] for unit in units], dtype=np.int64)
        return LifeTable(tuple(units), self.runs, self.last_productive[index], self.failure_pct[index],
                         self.unused_life_pct[index], self.cash[index], self.calendar[index],
                         self.retained_profit_pct[index])


def _value(row: Mapping[str, Any], name: str) -> Any:
    value = row[name]
    if name == "failed":
        return value if isinstance(value, bool) else str(value) in ("True", "true", "1")
    if name in ("cash", "unused_fraction"):
        return float(value)
    if name in ("last_productive_step", "calendar_steps"):
        return int(value)
    return value


def life_table(rows: Iterable[Mapping[str, Any]], *, economics: Economics = Economics(),
               key: Sequence[str] = ("group", "unit")) -> LifeTable:
    """Average the outcome rows of one method over its runs, life by life.

A life is identified by the ``key`` columns. Every life must carry the same
number of distinct runs and one last productive step.
"""
    lives: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        lives[tuple(str(row[name]) for name in key)].append(row)
    if not lives:
        raise ValueError("no outcome rows")
    counts = {len(group) for group in lives.values()}
    if len(counts) != 1:
        raise ValueError("every life must carry the same number of runs")
    units, last, failure, unused, cash, calendar = [], [], [], [], [], []
    for unit, group in sorted(lives.items()):
        if len({str(row["run"]) for row in group}) != len(group):
            raise ValueError(f"duplicate run on life {unit}")
        steps = {_value(row, "last_productive_step") for row in group}
        if len(steps) != 1:
            raise ValueError(f"the runs of life {unit} disagree on its last productive step")
        units.append(unit)
        last.append(steps.pop())
        failure.append(100 * _mean(float(_value(row, "failed")) for row in group))
        unused.append(100 * _mean(_value(row, "unused_fraction") for row in group))
        cash.append(_mean(_value(row, "cash") for row in group))
        calendar.append(_mean(float(_value(row, "calendar_steps")) for row in group))
    last_productive = np.asarray(last, dtype=np.int64)
    reference = replay(last_productive, last_productive - economics.mobilization_lead, economics)
    reference_rate = reference["cash"] / reference["calendar"]
    retained = [100 * (c / t) / r for c, t, r in zip(cash, calendar, reference_rate.tolist(), strict=True)]
    return LifeTable(tuple(units), counts.pop(), last_productive, np.asarray(failure), np.asarray(unused),
                     np.asarray(cash), np.asarray(calendar), np.asarray(retained))


def summarize(table: LifeTable) -> dict[str, Any]:
    """Equal-life failure share, unused life and retained profit, in percent."""
    return {"lives": len(table.units), "runs": table.runs,
            **{name: _mean(table.metric(name).tolist()) for name in METRICS}}


def _strata(units: Sequence[Any], strata: Mapping[Any, Any] | Sequence[Any] | None) -> list[np.ndarray]:
    if strata is None:
        return [np.arange(len(units))]
    labels = [strata[unit] for unit in units] if isinstance(strata, Mapping) else list(strata)
    if len(labels) != len(units):
        raise ValueError("one stratum per life is required")
    values = np.asarray([str(label) for label in labels])
    return [np.flatnonzero(values == label) for label in sorted(set(values.tolist()))]


def paired_bootstrap(tables: Mapping[str, LifeTable], *, contrasts: Sequence[tuple[str, str]] = (),
                     replicates: int = 10_000, seed: int = 0, strata: Mapping[Any, Any] | Sequence[Any] | None = None,
                     chunk: int = 500) -> dict[str, Any]:
    """Percentile 95% intervals of the equal-life metrics, resampling whole lives.

Every replicate draws lives with replacement (within each stratum, if given,
for example the law of each life) and uses the same draw for all methods, so
the differences in ``contrasts`` (first minus second) are paired. Runs stay
attached to their life.
"""
    if not tables:
        raise ValueError("no methods to resample")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    names = list(tables)
    units = tables[names[0]].units
    aligned = {name: table.aligned(units) for name, table in tables.items()}
    for first, second in contrasts:
        if first not in aligned or second not in aligned:
            raise ValueError(f"unknown method in contrast {first!r} - {second!r}")
    groups = _strata(units, strata)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        index = np.concatenate([rows[rng.integers(0, len(rows), size=(count, len(rows)))] for rows in groups], axis=1)
        for metric in METRICS:
            means = {name: table.metric(metric)[index].mean(axis=1) for name, table in aligned.items()}
            for name in names:
                draws[(name, metric)].append(means[name])
            for first, second in contrasts:
                draws[(f"{first} - {second}", metric)].append(means[first] - means[second])
    point = {name: summarize(table) for name, table in aligned.items()}

    def interval(label: str, metric: str, estimate: float) -> dict[str, Any]:
        values = np.concatenate(draws[(label, metric)])
        return {"estimate": estimate, "ci95": [float(v) for v in np.quantile(values, [.025, .975])]}
    return {"replicates": replicates, "seed": seed, "lives": len(units), "strata": len(groups),
            "methods": {name: {metric: interval(name, metric, point[name][metric]) for metric in METRICS}
                        for name in names},
            "contrasts": {f"{first} - {second}": {metric: interval(f"{first} - {second}", metric,
                                                                   point[first][metric] - point[second][metric])
                                                  for metric in METRICS}
                          for first, second in contrasts}}


def write_outcomes(path: str | Path, rows: Iterable[Mapping[str, Any]], *, exclusive: bool = True) -> int:
    """Write outcome rows with the standard columns; never replaces an existing file when ``exclusive``."""
    count = 0
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(OUTCOME_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in OUTCOME_COLUMNS})
            count += 1
    return count


def read_outcomes(path: str | Path) -> list[dict[str, Any]]:
    """Read an outcome table written by ``write_outcomes``."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and tuple(rows[0]) != OUTCOME_COLUMNS:
        raise ValueError(f"expected the columns {OUTCOME_COLUMNS}")
    return rows


def summarize_outcomes(rows: Iterable[Mapping[str, Any]], *, by: Sequence[str] = ("method",),
                       economics: Economics = Economics()) -> list[dict[str, Any]]:
    """Equal-life summaries of every cohort defined by the ``by`` columns (one preparation time per call)."""
    cohorts: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cohorts[tuple(str(row[name]) for name in by)].append(row)
    return [{**dict(zip(by, cohort, strict=True)), **summarize(life_table(members, economics=economics))}
            for cohort, members in sorted(cohorts.items())]


__all__ = ["LifeTable", "METRICS", "OUTCOME_COLUMNS", "life_table", "paired_bootstrap", "read_outcomes",
           "summarize", "summarize_outcomes", "write_outcomes"]
