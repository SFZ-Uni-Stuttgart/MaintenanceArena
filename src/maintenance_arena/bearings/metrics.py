"""Outcome rows and equal-bearing and pooled metrics of replayed decisions.

Every physical bearing has equal weight. Failure indicators, unused-life fractions, cash
and calendar time are first averaged over the seeds evaluated on a bearing; seeds are
not independent bearings. Retained profit divides a bearing's mean cash per mean
calendar minute by the rate of the last-safe reference that orders at T - L. The three
percentages are then averaged over bearings. Paired bootstrap intervals resample whole
bearings together with their seeds.

The pooled metrics weight bearings by calendar time instead: total seed-averaged cash over
total seed-averaged calendar minutes. They select training budgets and inputs
(``rank_candidates``) and give pooled-profit contrasts (``pooled_bootstrap``).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np

from .economics import ReplayEconomics, ReplayOutcome, pooled_rate
from .features import input_design

OUTCOME_FIELDS = ("design", "dataset", "fold", "seed", "bearing", "last_productive_minute", "order_minute",
                  "cash", "calendar_minutes", "failed", "unused_fraction")
METRICS = ("failure", "unused_life", "retained_profit")


def outcome_row(design: str, dataset: str, bearing: str, outcome: ReplayOutcome, *,
                fold: int | None = None, seed: int | None = None) -> dict:
    """One outcome row; ``fold`` and ``seed`` are None where they do not apply."""
    return {"design": design, "dataset": dataset, "fold": fold, "seed": seed, "bearing": bearing,
            "last_productive_minute": outcome.life, "order_minute": outcome.order, "cash": outcome.cash,
            "calendar_minutes": outcome.calendar, "failed": int(outcome.failed),
            "unused_fraction": outcome.unused_fraction}


def write_outcomes(path: str | os.PathLike, rows: Iterable[Mapping]) -> None:
    """Write outcome rows as CSV with ``OUTCOME_FIELDS``; None becomes an empty cell. Never replaces a file."""
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTCOME_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in OUTCOME_FIELDS})


def read_outcomes(path: str | os.PathLike) -> list[dict]:
    """Read an outcome CSV written by ``write_outcomes``."""
    def integer(text):
        return None if text == "" else int(text)

    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != OUTCOME_FIELDS:
            raise ValueError(f"{path}: expected the columns {OUTCOME_FIELDS}")
        return [{"design": r["design"], "dataset": r["dataset"], "fold": integer(r["fold"]),
                 "seed": integer(r["seed"]), "bearing": r["bearing"],
                 "last_productive_minute": int(r["last_productive_minute"]), "order_minute": integer(r["order_minute"]),
                 "cash": float(r["cash"]), "calendar_minutes": int(r["calendar_minutes"]),
                 "failed": int(r["failed"]), "unused_fraction": float(r["unused_fraction"])} for r in reader]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return math.fsum(values) / len(values)


@dataclass(frozen=True)
class BearingMetrics:
    """Seed-averaged result of one design on one bearing, in percent."""
    design: str
    dataset: str
    bearing: str
    replays: int
    failure: float
    unused_life: float
    retained_profit: float
    cash: float
    calendar: float


def bearing_metrics(rows: Iterable[Mapping], economics: ReplayEconomics | None = None) -> list[BearingMetrics]:
    """Per (dataset, design, bearing): failure share, unused life and retained profit over its seeds."""
    economics = economics or ReplayEconomics()
    groups: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["design"], row["bearing"])].append(row)
    result = []
    for (dataset, design, bearing), group in sorted(groups.items()):
        lives = {row["last_productive_minute"] for row in group}
        if len(lives) != 1:
            raise ValueError(f"{dataset}/{design}/{bearing}: rows disagree on the last productive minute")
        life = lives.pop()
        reference = economics.outcome(life, economics.oracle_order(life))
        cash = _mean(row["cash"] for row in group)
        calendar = _mean(row["calendar_minutes"] for row in group)
        result.append(BearingMetrics(design, dataset, bearing, len(group),
                                     100 * _mean(row["failed"] for row in group),
                                     100 * _mean(row["unused_fraction"] for row in group),
                                     100 * (cash / calendar) / (reference.cash / reference.calendar), cash, calendar))
    return result


def design_summary(rows: Iterable[Mapping], economics: ReplayEconomics | None = None) -> list[dict]:
    """Equal-bearing failure, unused-life and retained-profit percentages per dataset and design."""
    groups: dict[tuple[str, str], list[BearingMetrics]] = defaultdict(list)
    for metrics in bearing_metrics(rows, economics):
        groups[(metrics.dataset, metrics.design)].append(metrics)
    return [{"dataset": dataset, "design": design, "bearings": len(group), "replays": sum(m.replays for m in group),
             **{f"{name}_pct": _mean(getattr(m, name) for m in group) for name in METRICS}}
            for (dataset, design), group in sorted(groups.items())]


def paired_bootstrap(rows: Iterable[Mapping], design: str, reference: str, *, dataset: str,
                     metric: str = "retained_profit", economics: ReplayEconomics | None = None,
                     replicates: int = 10_000, seed: int = 20260930) -> dict:
    """Equal-bearing difference ``design`` minus ``reference`` in percentage points, with a 95% interval.

    Bearings are sorted by name; every replicate draws n bearing indices with replacement
    from a PCG64 generator and averages the per-bearing differences.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    per_design: dict[str, dict[str, float]] = {design: {}, reference: {}}
    for metrics in bearing_metrics((row for row in rows if row["dataset"] == dataset
                                    and row["design"] in per_design), economics):
        per_design[metrics.design][metrics.bearing] = getattr(metrics, metric)
    bearings = sorted(per_design[design])
    if not bearings or bearings != sorted(per_design[reference]):
        raise ValueError(f"{design} and {reference} need outcomes on the same {dataset} bearings")
    differences = np.asarray([per_design[design][b] - per_design[reference][b] for b in bearings])
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(bearings)
    indices = np.stack([rng.integers(0, n, n) for _ in range(replicates)])
    draws = differences[indices].mean(1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"design": design, "reference": reference, "dataset": dataset, "metric": metric, "bearings": n,
            "point": float(differences.mean()), "ci95": [float(low), float(high)],
            "replicates": replicates, "seed": seed}


def _seed_means(rows: Iterable[Mapping], design: str, dataset: str, economics: ReplayEconomics) -> list[dict]:
    """Per bearing, sorted by name: NumPy means over seeds of cash, calendar, failure and unused minutes."""
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        if row["dataset"] == dataset and row["design"] == design:
            groups[row["bearing"]].append(row)
    result = []
    for bearing, group in sorted(groups.items()):
        lives = {row["last_productive_minute"] for row in group}
        if len(lives) != 1:
            raise ValueError(f"{dataset}/{design}/{bearing}: rows disagree on the last productive minute")
        life = lives.pop()
        result.append({"bearing": bearing, "life": life, "replays": len(group),
                       "cash": float(np.mean([row["cash"] for row in group])),
                       "calendar": float(np.mean([row["calendar_minutes"] for row in group])),
                       "failure": float(np.mean([row["failed"] for row in group])),
                       "unused": float(np.mean([economics.outcome(life, row["order_minute"]).unused for row in group])),
                       "failures": sum(int(row["failed"]) for row in group),
                       "no_orders": sum(row["order_minute"] is None for row in group)})
    if not result:
        raise ValueError(f"no {dataset} outcomes of {design}")
    return result


def pooled_summary(rows: Iterable[Mapping], design: str, *, dataset: str,
                   economics: ReplayEconomics | None = None) -> dict:
    """Pooled profit of a design: total seed-averaged cash over total seed-averaged calendar minutes.

    ``pooled_retained_profit_pct`` divides it by the pooled rate of the last-safe reference on the
    same bearings. Failures are counted over all replays and, per seed, as the sum of the
    per-bearing failure shares.
    """
    economics = economics or ReplayEconomics()
    averaged = _seed_means(rows, design, dataset, economics)
    reference = [economics.outcome(r["life"], economics.oracle_order(r["life"])) for r in averaged]
    profit = math.fsum(r["cash"] for r in averaged) / math.fsum(r["calendar"] for r in averaged)
    return {"design": design, "dataset": dataset, "bearings": len(averaged), "replays": sum(r["replays"] for r in averaged),
            "profit_per_calendar_minute": profit, "pooled_retained_profit_pct": 100 * (profit / pooled_rate(reference)),
            "failure_outcomes": sum(r["failures"] for r in averaged),
            "failures_mean_per_seed": sum(r["failure"] for r in averaged),
            "no_order_outcomes": sum(r["no_orders"] for r in averaged),
            "mean_unused_minutes": float(np.mean([r["unused"] for r in averaged]))}


def rank_candidates(rows: Iterable[Mapping], candidates: Mapping[str, Mapping], *, dataset: str,
                    economics: ReplayEconomics | None = None) -> dict:
    """Rank candidate designs by held-out pooled profit, as the article selected Full and its budget.

    ``candidates`` maps every design to its ``calendar_steps`` and ``inputs`` (an input-design
    key or the number of policy inputs). Exact ties prefer fewer failures, then the shorter
    training budget, then fewer policy inputs. Returns the ranking, the selected design and
    the best design of every input design.
    """
    rows = list(rows)
    summaries = {}
    for design, spec in candidates.items():
        inputs = spec["inputs"]
        width = len(input_design(inputs).observation_names) if isinstance(inputs, str) else int(inputs)
        summaries[design] = {**pooled_summary(rows, design, dataset=dataset, economics=economics),
                             "calendar_steps": int(spec["calendar_steps"]), "inputs": inputs, "policy_inputs": width}
    if not summaries:
        raise ValueError("ranking needs at least one candidate")
    order = sorted(summaries, key=lambda d: (-summaries[d]["profit_per_calendar_minute"],
                                             summaries[d]["failure_outcomes"], summaries[d]["calendar_steps"],
                                             summaries[d]["policy_inputs"]))
    best: dict = {}
    for design in order:
        best.setdefault(summaries[design]["inputs"], design)
    return {"dataset": dataset, "selected": order[0], "best_by_inputs": best, "ranking": [summaries[d] for d in order]}


def pooled_bootstrap(rows: Iterable[Mapping], design: str, reference: str, *, dataset: str,
                     economics: ReplayEconomics | None = None, replicates: int = 10_000,
                     seed: int = 20260930) -> dict:
    """Pooled-profit difference ``design`` minus ``reference`` relative to the pooled last-safe rate.

    Seeds are averaged within each bearing. Every replicate draws n bearings with replacement
    from a PCG64 generator and recomputes the three pooled rates, so the interval resamples
    whole bearings with their seeds. Values are fractions of the reference rate;
    ``share_positive`` is the share of replicates above zero.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    economics = economics or ReplayEconomics()
    rows = list(rows)
    a, b = (_seed_means(rows, name, dataset, economics) for name in (design, reference))
    if [(r["bearing"], r["life"]) for r in a] != [(r["bearing"], r["life"]) for r in b]:
        raise ValueError(f"{design} and {reference} need outcomes on the same {dataset} bearings")
    oracle = [economics.outcome(r["life"], economics.oracle_order(r["life"])) for r in a]
    ca, ka = (np.asarray([r[key] for r in a], dtype=np.float64) for key in ("cash", "calendar"))
    cb, kb = (np.asarray([r[key] for r in b], dtype=np.float64) for key in ("cash", "calendar"))
    co = np.asarray([o.cash for o in oracle], dtype=np.float64)
    ko = np.asarray([o.calendar for o in oracle], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(ca)
    draws = np.empty(replicates)
    for index in range(replicates):
        i = rng.integers(0, n, n)
        rate = co[i].sum() / ko[i].sum()
        draws[index] = (ca[i].sum() / ka[i].sum() - cb[i].sum() / kb[i].sum()) / rate
    point = (ca.sum() / ka.sum() - cb.sum() / kb.sum()) / (co.sum() / ko.sum())
    return {"design": design, "reference": reference, "dataset": dataset, "bearings": n, "point": float(point),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "share_positive": float(np.mean(draws > 0)), "replicates": replicates, "seed": seed}


def summarize_rows(summary: Sequence[Mapping], digits: int = 2) -> str:
    """Markdown table of a ``design_summary``."""
    lines = ["| Dataset | Design | Bearings | Replays | Failures (%) | Unused life (%) | Retained profit (%) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['design']} | {row['bearings']} | {row['replays']} | "
                     + " | ".join(f"{row[f'{name}_pct']:.{digits}f}" for name in METRICS) + " |")
    return "\n".join(lines)
