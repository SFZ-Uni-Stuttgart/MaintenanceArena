"""Analysis helpers behind the accompanying article's synthetic figures.

* ``health_paths``: complete unmaintained health paths of chosen lives of a bank,
  with the last safe order ``T - 1 - L`` and the health there.
* ``map_states``: the causal state at every decision age of every life of a bank,
  with the physical health, the damage rate (consecutive differences of the
  damage ``-ln h / ln 5``), the age, the remaining life and the law. These are the
  states of a policy map; they cover whole unmaintained lives, also after a
  policy would have ordered, so they describe the policy, not its visits.
* ``order_probabilities``: P(ORDER) of several policies on the same states.
* ``binned_mean`` and ``native_means``: the state-weighted mean of a value in
  each bin of a (health, damage rate) grid, and its mean at every distinct
  coordinate (the support of a 0.5 contour).
* ``failure_share_curve``: the share of failed training episodes in bins of
  training steps, from the cumulative counters of a training log.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .observations import observed_damage
from .replay import decision_series, policy_probabilities


def _float32_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def health_paths(bank: Mapping[str, Any], *, per_law: int | None = None, pairs: Sequence[int] | None = None,
                 mobilization_lead: int = 12) -> list[dict[str, Any]]:
    """Health paths of active components: the first ``per_law`` of each law in pair order, or the listed positions."""
    if (per_law is None) == (pairs is None):
        raise ValueError("give either per_law or pairs")
    if pairs is not None:
        chosen = [bank["pairs"][int(index)] for index in pairs]
    else:
        counts: Counter[str] = Counter()
        chosen = []
        for pair in bank["pairs"]:
            law = pair["active"]["family_id"]
            if counts[law] < per_law:
                chosen.append(pair)
                counts[law] += 1
    result = []
    for pair in chosen:
        part = pair["active"]
        health = np.asarray(part["health"], dtype=np.float32)
        eol = int(part["eol_operating_age"])
        last = eol - 1 - int(mobilization_lead)
        result.append({"pair_index": int(pair["pair_index"]), "law": str(part["family_id"]),
                       "stream_address": part["stream_address"], "end_of_life": eol,
                       "last_safe_order_age": last, "health_at_last_safe_order": float(health[last]),
                       "health": [float(value) for value in health], "health_float32_sha256": _float32_sha256(health)})
    return result


@dataclass(frozen=True, eq=False)
class MapStates:
    """Decision states of whole lives (one row per life and age) with their physical coordinates."""
    states: np.ndarray        # (n, 4) float32 observations
    damage_rate: np.ndarray   # (n,) float64 consecutive damage differences of the true health, 0 at age 0
    age: np.ndarray           # (n,) int32 operating age
    remaining: np.ndarray     # (n,) int32 steps to failure, end of life minus age
    life: np.ndarray          # (n,) int32 position of the life in the bank
    law: np.ndarray           # (n,) law of the life
    mobilization_lead: int

    @property
    def health(self) -> np.ndarray:
        """The observed health (float32)."""
        return self.states[:, 0]

    @property
    def valid(self) -> np.ndarray:
        """States whose rate channel holds a rate (the validity flag)."""
        return self.states[:, 2] >= .5

    @property
    def observed_rate(self) -> np.ndarray:
        """The rate channel divided by the preparation time (float32 division, as float64)."""
        return (self.states[:, 1] / self.mobilization_lead).astype(np.float64)

    def select(self, rows: np.ndarray) -> "MapStates":
        return MapStates(self.states[rows], self.damage_rate[rows], self.age[rows], self.remaining[rows],
                         self.life[rows], self.law[rows], self.mobilization_lead)


def first_lives_per_law(bank: Mapping[str, Any], count: int) -> dict[str, list[int]]:
    """Positions of the first ``count`` lives of each law in pair order."""
    chosen: dict[str, list[int]] = {}
    for position, pair in enumerate(bank["pairs"]):
        law = str(pair["active"]["family_id"])
        members = chosen.setdefault(law, [])
        if len(members) < count:
            members.append(position)
    if any(len(members) < count for members in chosen.values()):
        raise ValueError(f"some law has fewer than {count} lives")
    return chosen


def map_states(bank: Mapping[str, Any], *, mobilization_lead: int = 12, observation_route: str = "RAW_CAUSAL",
               lives: Iterable[int] | None = None, min_age: int = 0) -> MapStates:
    """States at every age from ``min_age`` to the end of life minus one of the chosen lives (positions; all by default)."""
    series = decision_series(bank, mobilization_lead=mobilization_lead, observation_route=observation_route)
    positions = np.arange(series.lives) if lives is None else np.asarray(sorted(set(int(i) for i in lives)))
    parts = {name: [] for name in ("states", "rate", "age", "remaining", "life", "law")}
    for position in positions:
        start, stop = series.slices[position]
        part = bank["pairs"][position]["active"]
        damage = [observed_damage(value) for value in part["health"][:-1]]
        rate = np.r_[0., np.diff(damage)]
        eol = int(part["eol_operating_age"])
        keep = slice(min_age, stop - start)
        parts["states"].append(series.states[start:stop][keep])
        parts["rate"].append(rate[keep])
        parts["age"].append(np.arange(stop - start, dtype=np.int32)[keep])
        parts["remaining"].append(np.arange(eol, 0, -1, dtype=np.int32)[keep])
        parts["life"].append(np.full(stop - start - min_age, position, dtype=np.int32))
        parts["law"].append(np.full(stop - start - min_age, series.families[position], dtype="U32"))
    return MapStates(np.ascontiguousarray(np.vstack(parts["states"]), dtype=np.float32),
                     np.concatenate(parts["rate"]).astype(np.float64), np.concatenate(parts["age"]),
                     np.concatenate(parts["remaining"]), np.concatenate(parts["life"]), np.concatenate(parts["law"]),
                     int(mobilization_lead))


def order_probabilities(policies: Sequence[Any], states: np.ndarray) -> np.ndarray:
    """(policies, n) float32 P(ORDER) of every policy on the same states."""
    return np.stack([policy_probabilities(policy, states)[:, 1] for policy in policies])


def ensemble_mean(probabilities: np.ndarray) -> np.ndarray:
    """The equal mean over policies (rows) at every state, in float64."""
    return np.asarray(probabilities, dtype=np.float32).mean(axis=0, dtype=np.float64)


def edges(start: float, stop: float, width: float) -> np.ndarray:
    """Bin edges from ``start`` to ``stop`` in steps of ``width`` (``numpy.linspace``)."""
    return np.linspace(start, stop, round((stop - start) / width) + 1)


def in_view(x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, *,
            where: np.ndarray | None = None) -> np.ndarray:
    """States inside the edges (right edges inclusive, as in ``numpy.histogram2d``) and inside ``where``."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    inside = (x >= x_edges[0]) & (x <= x_edges[-1]) & (y >= y_edges[0]) & (y <= y_edges[-1])
    return inside if where is None else inside & np.asarray(where, dtype=bool)


def binned_mean(x: np.ndarray, y: np.ndarray, values: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, *,
                where: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Counts and state-weighted means of ``values`` in the (x, y) bins (NaN where no state falls).

States outside the edges (or outside ``where``) are left out (``in_view``).
"""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    inside = in_view(x, y, x_edges, y_edges, where=where)
    counts = np.histogram2d(x[inside], y[inside], bins=(x_edges, y_edges))[0]
    totals = np.histogram2d(x[inside], y[inside], bins=(x_edges, y_edges), weights=np.asarray(values)[inside])[0]
    means = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    return counts, means


def native_means(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (x, y) coordinates and the mean of ``values`` at each."""
    coordinates, inverse, multiplicity = np.unique(np.column_stack((np.asarray(x, dtype=np.float64),
                                                                    np.asarray(y, dtype=np.float64))),
                                                   axis=0, return_inverse=True, return_counts=True)
    return coordinates, np.bincount(inverse.ravel(), weights=np.asarray(values, dtype=np.float64)) / multiplicity


def failure_share_curve(log: Sequence[Mapping[str, Any]], budget: int, width: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin edges, failed / completed training episodes (%) and steps per completed episode in each bin.

The cumulative calendar steps, completed episodes and failures after every update
of ``log`` start from (0, 0, 0) and are interpolated linearly onto the edges.
"""
    if budget % width:
        raise ValueError("the bins must end at the budget")
    grid = np.arange(0, budget + 1, width, dtype=float)
    steps = np.array([0] + [entry["actual_calendar_steps"] for entry in log], float)
    episodes = np.array([0] + [entry["episodes"] for entry in log], float)
    failures = np.array([0] + [entry["failures"] for entry in log], float)
    if not (np.all(np.diff(steps) > 0) and np.all(np.diff(episodes) >= 0) and np.all(np.diff(failures) >= 0)):
        raise ValueError("the training counters must be cumulative")
    if steps[-1] < grid[-1]:
        raise ValueError("the log ends before the budget")
    completed, failed = np.diff(np.interp(grid, steps, episodes)), np.diff(np.interp(grid, steps, failures))
    if np.any(completed <= 0):
        raise ValueError("a bin without completed episodes")
    return grid, 100. * failed / completed, np.diff(grid) / completed


__all__ = ["MapStates", "binned_mean", "edges", "ensemble_mean", "failure_share_curve", "first_lives_per_law",
           "health_paths", "in_view", "map_states", "native_means", "order_probabilities"]
