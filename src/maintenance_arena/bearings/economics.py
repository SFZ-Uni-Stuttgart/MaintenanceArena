"""Replay economics of one recorded bearing life under the core task's events and prices.

A decision at minute t = 1..T follows the observation of minute t, where T is the last
productive minute and failure follows immediately afterwards. WAIT earns one productive
minute, or fails at t = T. ORDER starts L productive preparation minutes and is timely if
t + L <= T; one exchange step and the downtime steps follow, so the last safe order is
T - L. With ``preserve_preparation_v1`` a failure stops production and keeps completed
preparation: the remaining preparation runs nonproductively, the failure detection
counting as its first step, before the exchange. With ``immediate_exchange_v1`` the
exchange follows the failure at once. Cash is 255 U - 520 - 25 N - 7000 F for U
productive minutes, N nonproductive steps and F failures.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import math

import numpy as np

WAIT, ORDER = 0, 1
RECOVERY_CONTRACTS = ("preserve_preparation_v1", "immediate_exchange_v1")


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class ReplayOutcome:
    """First order and economic result of one replayed life.

    ``life`` is the last productive minute T, ``order`` the first order minute or None,
    ``calendar`` the elapsed calendar minutes up to the end of the renewal and
    ``unused`` the productive minutes left unused, T - U.
    """
    life: int
    preparation: int
    order: int | None
    cash: float
    calendar: int
    failed: bool
    unused: int

    @property
    def margin(self) -> int | None:
        """Minutes by which the order precedes the last safe order T - L; negative if late."""
        return None if self.order is None else self.life - self.preparation - self.order

    @property
    def unused_fraction(self) -> float:
        """Unused usable life (T - U) / (T - L)."""
        if self.life <= self.preparation:
            raise ValueError("the unused fraction needs a life longer than the preparation time")
        return self.unused / (self.life - self.preparation)

    def as_dict(self) -> dict:
        return {**asdict(self), "margin": self.margin}


@dataclass(frozen=True)
class ReplayEconomics:
    """Preparation time, downtime and prices of the replay (the article's defaults)."""
    preparation: int = 12
    downtime_steps: int = 6
    revenue: float = 255.0
    replacement_cost: float = 520.0
    nonproductive_cost: float = 25.0
    failure_cost: float = 7000.0
    recovery_contract: str = "preserve_preparation_v1"

    def __post_init__(self):
        _integer(self.preparation, "preparation", 1)
        _integer(self.downtime_steps, "downtime_steps", 0)
        for name in ("revenue", "replacement_cost", "nonproductive_cost", "failure_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.recovery_contract not in RECOVERY_CONTRACTS:
            raise ValueError(f"recovery_contract must be one of {RECOVERY_CONTRACTS}")

    def _check(self, life, order=None) -> tuple[int, int | None]:
        life = _integer(life, "life", 1)
        if order is not None:
            order = _integer(order, "order minute", 1)
            if order > life:
                raise ValueError("an order minute must lie within 1..life")
        return life, order

    def outcome(self, life: int, order: int | None) -> ReplayOutcome:
        """Outcome of a life whose first order is placed after observing minute ``order``."""
        life, order = self._check(life, order)
        preparation, downtime = self.preparation, self.downtime_steps
        nonproductive = (1 + downtime) * self.nonproductive_cost
        if order is not None and order + preparation <= life:
            operated = order + preparation
            cash = self.revenue * operated - self.replacement_cost - nonproductive
            return ReplayOutcome(life, preparation, order, cash, operated + 1 + downtime, False, life - operated)
        recovery = 0
        if self.recovery_contract == "preserve_preparation_v1":
            recovery = preparation if order is None else preparation - (life - order)
        cash = (self.revenue * life - self.replacement_cost - self.failure_cost - nonproductive
                - recovery * self.nonproductive_cost)
        return ReplayOutcome(life, preparation, order, cash, life + recovery + 1 + downtime, True, 0)

    def oracle_order(self, life: int) -> int | None:
        """Order of the last-safe reference that knows T: minute T - L.

        A life of at most L minutes has no safe order; the reference then orders at minute 1
        if partial preparation earns more per calendar minute than reactive operation.
        """
        life, _ = self._check(life)
        if life - self.preparation >= 1:
            return life - self.preparation
        early, reactive = self.outcome(life, 1), self.outcome(life, None)
        return 1 if early.cash / early.calendar > reactive.cash / reactive.calendar else None

    def _failure_cash(self, preparation_remaining: int) -> list[float]:
        c, downtime = self.nonproductive_cost, self.downtime_steps
        if self.recovery_contract == "immediate_exchange_v1" or preparation_remaining == 0:
            return [-(self.replacement_cost + self.failure_cost + c)] + [-c] * downtime
        return ([-(self.failure_cost + c)] + [-c] * (preparation_remaining - 1)
                + [-(self.replacement_cost + c)] + [-c] * downtime)

    def macro(self, life: int, minute: int, action: int) -> tuple[list[float], bool]:
        """Cash of every calendar step of one decision at ``minute`` and whether it ends the life."""
        life, minute = self._check(life, minute)
        if isinstance(action, (bool, np.bool_)) or action not in (WAIT, ORDER):
            raise ValueError("action must be WAIT (0) or ORDER (1)")
        if action == WAIT:
            if minute + 1 <= life:
                return [self.revenue], False
            return self._failure_cash(self.preparation), True
        remaining = life - minute
        if remaining >= self.preparation:
            return ([self.revenue] * self.preparation + [-(self.replacement_cost + self.nonproductive_cost)]
                    + [-self.nonproductive_cost] * self.downtime_steps), True
        return [self.revenue] * remaining + self._failure_cash(self.preparation - remaining), True


def pooled_rate(outcomes: Iterable[ReplayOutcome]) -> float:
    """Total cash divided by total calendar minutes."""
    rows = list(outcomes)
    if not rows:
        raise ValueError("a pooled rate needs at least one outcome")
    return math.fsum(o.cash for o in rows) / math.fsum(o.calendar for o in rows)
