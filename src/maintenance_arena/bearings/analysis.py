"""Figure data of the bearing study: policy maps, transfer timing and the age clock.

The functions work on runner outputs: outcome rows (``read_outcomes``), fit records with the
first orders on the training histories (``fits.json``) and trained actors with their inputs.
They compute display values only and change no fitted quantity. A policy map is a
retrospective policy-function diagnostic: it includes states after an actor would already have
ordered, and its colour is an action probability, not a failure risk.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math

import numpy as np


def median_order(orders: Sequence[int | None]) -> int | None:
    """Median first order; a missing order ranks after every order. Of five orders, the third."""
    if not orders:
        raise ValueError("a median needs at least one order")
    ranked = sorted(orders, key=lambda order: math.inf if order is None else order)
    return ranked[(len(ranked) - 1) // 2]


def order_status(order: int | None, last_safe: int) -> str:
    """``timely`` up to the last safe order, ``late`` after it, ``no_order`` without an order."""
    return "no_order" if order is None else ("late" if order > last_safe else "timely")


def first_order(probabilities: np.ndarray) -> int | None:
    """First minute with P(ORDER) > P(WAIT) in a (T, 2) probability matrix, or None."""
    hits = np.flatnonzero(probabilities[:, 1] > probabilities[:, 0])
    return None if hits.size == 0 else int(hits[0] + 1)


def policy_responses(agents: Sequence, observations: np.ndarray) -> np.ndarray:
    """Action probabilities (actors, T, 2) of trained actors on one input matrix, one batch per actor."""
    if not agents:
        raise ValueError("policy responses need at least one actor")
    return np.stack([np.asarray(agent.batch_output(observations)[1], dtype=np.float32) for agent in agents])


def policy_map(bearings: Mapping[str, Mapping], *, preparation: int = 12) -> list[dict]:
    """Mean ORDER probability along every history, sorted by life and name.

    ``bearings`` maps a name to ``probabilities`` (actors, T, 2), as from ``policy_responses``, and
    optionally ``rule_order``. Each row holds the mean P(ORDER) over the actors at every minute,
    every actor's first order, the Rule order and the last safe order T - L.
    """
    rows = []
    for name, entry in bearings.items():
        probabilities = np.asarray(entry["probabilities"])
        if probabilities.ndim != 3 or probabilities.shape[2] != 2 or not np.isfinite(probabilities).all():
            raise ValueError(f"{name}: probabilities must be a finite (actors, T, 2) array")
        life = probabilities.shape[1]
        rows.append({"bearing": name, "life": life, "last_safe_order": life - preparation,
                     "mean_order_probability": probabilities[:, :, 1].mean(axis=0).tolist(),
                     "actor_orders": [first_order(p) for p in probabilities],
                     "rule_order": entry.get("rule_order")})
    return sorted(rows, key=lambda row: (row["life"], row["bearing"]))


def _orders_by_bearing(rows: Iterable[Mapping], design: str, dataset: str) -> dict[str, list[Mapping]]:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        if row["design"] == design and row["dataset"] == dataset:
            groups[row["bearing"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: -1 if row["seed"] is None else row["seed"])
    return groups


def transfer_timing(rows: Iterable[Mapping], designs: Sequence[str] = ("Full", "No age", "Indicator Rule"), *,
                    dataset: str = "XJTU-SY", preparation: int = 12) -> list[dict]:
    """Order timing per target bearing and design, sorted by life and name.

    A design's order is the median first order over its seeds, a missing order ranking last; a
    rule has one order. ``margin`` is the last safe order T - L minus that order, and a timely
    cell has the unused share ``100 margin / (T - L)``. ``seed_failures`` counts the failed seeds.
    """
    rows = list(rows)
    groups = {design: _orders_by_bearing(rows, design, dataset) for design in designs}
    bearings = set().union(*(set(g) for g in groups.values()))
    if not bearings or any(set(g) != bearings for g in groups.values()):
        raise ValueError(f"every design needs outcomes on the same {dataset} bearings")
    result = []
    for bearing in bearings:
        lives = {row["last_productive_minute"] for g in groups.values() for row in g[bearing]}
        if len(lives) != 1:
            raise ValueError(f"{bearing}: rows disagree on the last productive minute")
        life = lives.pop()
        last_safe = life - preparation
        cells = {}
        for design in designs:
            seeds = groups[design][bearing]
            order = median_order([row["order_minute"] for row in seeds])
            status = order_status(order, last_safe)
            margin = None if order is None else last_safe - order
            cells[design] = {"order": order, "status": status, "margin": margin,
                             "unused_life_pct": 100.0 * margin / last_safe if status == "timely" else None,
                             "signed_margin_pct": None if margin is None else 100.0 * margin / last_safe,
                             "seed_orders": [row["order_minute"] for row in seeds],
                             "seed_failures": sum(int(row["failed"]) for row in seeds)}
        result.append({"bearing": bearing, "life": life, "last_safe_order": last_safe, "designs": cells})
    return sorted(result, key=lambda row: (row["life"], row["bearing"]))


def age_clock(fits: Iterable[Mapping], rows: Iterable[Mapping],
              designs: Sequence[str] = ("Full", "No forecaster age", "No policy age", "No age"), *,
              target_dataset: str = "XJTU-SY", longest: int = 3, preparation: int = 12) -> dict:
    """First orders of every transfer actor on its training histories and on the longest target lives.

    ``fits`` are the runner's fit records of the transfer (``fold`` None) with ``training_orders``
    and ``training_lives``; ``rows`` are outcome rows on the target. The longest target bearings
    are listed by ascending life with their last safe orders T - L.
    """
    fits, rows = list(fits), list(rows)
    lives = {row["bearing"]: row["last_productive_minute"] for row in rows if row["dataset"] == target_dataset}
    if len(lives) < longest:
        raise ValueError(f"the age clock needs at least {longest} {target_dataset} bearings")
    long = sorted(lives, key=lambda name: (lives[name], name))[-longest:]
    result = {"long_bearings": [{"bearing": b, "life": lives[b], "last_safe_order": lives[b] - preparation}
                                for b in long], "designs": {}}
    training_lives = {}
    for design in designs:
        entries = sorted((fit for fit in fits if fit["design"] == design and fit["fold"] is None
                          and fit["dataset"] == target_dataset), key=lambda fit: fit["seed"])
        if not entries:
            raise ValueError(f"no transfer fits of {design}")
        outcomes = _orders_by_bearing(rows, design, target_dataset)
        seeds = []
        for fit in entries:
            if len(fit["training_orders"]) != len(fit["training_bearings"]):
                raise ValueError(f"{design} seed {fit['seed']}: one training order per training bearing expected")
            training_lives.update(zip(fit["training_bearings"], fit["training_lives"]))
            target = {}
            for bearing in long:
                match = [row for row in outcomes.get(bearing, []) if row["seed"] == fit["seed"]]
                if len(match) != 1:
                    raise ValueError(f"{design} seed {fit['seed']}: one outcome on {bearing} expected")
                target[bearing] = match[0]["order_minute"]
            seeds.append({"seed": fit["seed"],
                          "training_orders": dict(zip(fit["training_bearings"], fit["training_orders"])),
                          "target_orders": target})
        result["designs"][design] = seeds
    result["longest_training_life"] = max(training_lives.values())
    return result
