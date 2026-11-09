"""Remaining-minutes forecaster with nested cross-fitting over bearings.

A gradient-boosted tree regressor learns min(T - t, 125) at minute t from the causal
forecaster features of training histories. Policy training and rule calibration use
out-of-fold forecasts from inner folds; a refit on all training bearings forecasts the
evaluation bearings. Fitting requires scikit-learn.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib

import numpy as np

from .data import BearingHistory
from .features import FeatureSettings, forecaster_features


@dataclass(frozen=True)
class ForecasterSettings:
    """Target cap and ``HistGradientBoostingRegressor`` settings of the forecaster."""
    cap: float = 125.0
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    l2_regularization: float = 1.0
    random_state: int = 0

    def __post_init__(self):
        if not self.cap > 0 or not self.learning_rate > 0 or self.l2_regularization < 0:
            raise ValueError("cap and learning_rate must be positive and l2_regularization nonnegative")
        for name in ("max_iter", "max_leaf_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class Forecaster:
    """Gradient-boosted regressor of the remaining minutes, with or without the minute input.

    Early stopping is off. Predictions are clipped at zero.
    """

    def __init__(self, *, age: bool = True, settings: ForecasterSettings | None = None,
                 features: FeatureSettings | None = None):
        self.age = bool(age)
        self.settings = settings or ForecasterSettings()
        self.features = features or FeatureSettings()
        self.model = None
        self.training_bearings: tuple[str, ...] = ()

    def fit(self, histories: Sequence[BearingHistory]) -> Forecaster:
        """Fit on complete training histories, stacked in the given order."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        if not histories:
            raise ValueError("the forecaster needs at least one training history")
        s = self.settings
        x = np.vstack([forecaster_features(h, self.features, age=self.age) for h in histories])
        y = np.concatenate([np.minimum(h.life - np.arange(1, h.life + 1), s.cap) for h in histories]).astype(np.float64)
        self.model = HistGradientBoostingRegressor(
            max_iter=s.max_iter, learning_rate=s.learning_rate, max_leaf_nodes=s.max_leaf_nodes,
            l2_regularization=s.l2_regularization, early_stopping=False, random_state=s.random_state).fit(x, y)
        self.training_bearings = tuple(h.name for h in histories)
        return self

    def predict(self, history: BearingHistory) -> np.ndarray:
        """Forecast remaining minutes at every minute of a history, clipped at zero."""
        if self.model is None:
            raise RuntimeError("fit the forecaster before prediction")
        return np.maximum(self.model.predict(forecaster_features(history, self.features, age=self.age)), 0.0)


def hashed_folds(names: Sequence[str], salt: str, folds: int) -> list[list[str]]:
    """Outcome-free assignment of names to groups by salted SHA-256.

    Names are ordered by sha256(f"{salt}|{name}") and dealt round-robin to ``folds``
    groups; each group keeps the input order of its names.
    """
    names = [str(name) for name in names]
    if len(set(names)) != len(names):
        raise ValueError("fold assignment needs unique names")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 1:
        raise ValueError("folds must be a positive integer")
    keyed = sorted(names, key=lambda name: hashlib.sha256(f"{salt}|{name}".encode()).hexdigest())
    group = {name: index % folds for index, name in enumerate(keyed)}
    return [[name for name in names if group[name] == k] for k in range(folds)]


def cross_fit(training: Sequence[BearingHistory], inner_folds: Sequence[Sequence[str]],
              evaluation: Sequence[BearingHistory] = (), *, age: bool = True,
              settings: ForecasterSettings | None = None,
              features: FeatureSettings | None = None) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Out-of-fold forecasts for training bearings and refit forecasts for evaluation bearings.

    ``inner_folds`` partitions the training names. Each training bearing is forecast by
    a model fitted without its inner fold, and a model fitted on all training bearings
    forecasts every evaluation history. Fits stack histories in the order of ``training``.
    Returns the out-of-fold forecasts by name and the evaluation forecasts in order.
    """
    names = [h.name for h in training]
    if not names or len(set(names)) != len(names):
        raise ValueError("cross-fitting needs uniquely named training histories")
    groups = [list(group) for group in inner_folds]
    flat = [name for group in groups for name in group]
    if len(set(flat)) != len(flat) or set(flat) != set(names):
        raise ValueError("inner folds must partition the training bearings")
    options = {"age": age, "settings": settings, "features": features}
    out_of_fold: dict[str, np.ndarray] = {}
    for group in groups:
        held = set(group)
        if not held:
            continue
        model = Forecaster(**options).fit([h for h in training if h.name not in held])
        for history in training:
            if history.name in held:
                out_of_fold[history.name] = model.predict(history)
    forecasts = []
    if evaluation:
        model = Forecaster(**options).fit(list(training))
        forecasts = [model.predict(history) for history in evaluation]
    return {name: out_of_fold[name] for name in names}, forecasts
