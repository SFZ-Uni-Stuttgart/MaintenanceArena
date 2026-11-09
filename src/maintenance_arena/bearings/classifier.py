"""Supervised risk classifier on the policy inputs of a bearing design.

The classifier is the non-RL reference of the article's Table 5 that uses all inputs of a PPO
design at once. A gradient-boosted tree model learns to mark every minute at or after the last
safe order T - L, with weight 1/T per minute so that every training bearing counts equally. It
orders at the first minute whose score reaches a threshold. The threshold maximizes the pooled
calendar profit of the leave-one-bearing-out scores of the training bearings, and the smallest
threshold wins ties. The classifier is then refitted on all training bearings. Training bearings
are taken in the order of their names. The model needs scikit-learn.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .economics import ReplayEconomics, pooled_rate

CLASSIFIER_SETTINGS = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 20,
                       "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}
THRESHOLDS = tuple(round(0.01 * i, 2) for i in range(1, 100))


@dataclass(frozen=True)
class ClassifierSelection:
    threshold: float
    training_profit_rate: float
    training_failures: int
    candidates: int


def late_labels(life: int, preparation: int) -> np.ndarray:
    """1 for the minutes T - L to T, else 0; row i holds minute i + 1."""
    if life <= preparation:
        raise ValueError("a training bearing needs a life longer than the preparation time")
    labels = np.zeros(life)
    labels[life - preparation - 1:] = 1.0
    return labels


def fit_classifier(inputs: Sequence[np.ndarray], preparation: int, settings: Mapping | None = None):
    """Gradient-boosted classifier on the stacked inputs of the given bearings, in the given order."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    if not inputs:
        raise ValueError("the classifier needs at least one training bearing")
    x = np.vstack([np.asarray(item, dtype=np.float64) for item in inputs])
    y = np.concatenate([late_labels(len(item), preparation) for item in inputs])
    weights = np.concatenate([np.full(len(item), 1.0 / len(item)) for item in inputs])
    model = HistGradientBoostingClassifier(**dict(settings or CLASSIFIER_SETTINGS))
    return model.fit(x, y, sample_weight=weights)


def classifier_scores(model, inputs: np.ndarray) -> np.ndarray:
    """Score of every minute of one bearing: the probability of the late class."""
    return model.predict_proba(np.asarray(inputs, dtype=np.float64))[:, 1]


def classifier_order(scores: np.ndarray, threshold: float) -> int | None:
    """First 1-based minute whose score is at least the threshold, or None."""
    hits = np.flatnonzero(np.asarray(scores) >= threshold)
    return None if hits.size == 0 else int(hits[0] + 1)


def calibrate_risk_classifier(training: Mapping[str, np.ndarray], economics: ReplayEconomics | None = None,
                              thresholds: Sequence[float] = THRESHOLDS, settings: Mapping | None = None):
    """Refitted classifier and its threshold, chosen on leave-one-bearing-out training scores."""
    economics = economics or ReplayEconomics()
    names = sorted(training)
    if len(names) < 2 or len(thresholds) == 0:
        raise ValueError("calibration needs at least two training bearings and one threshold")
    scores = {name: classifier_scores(fit_classifier([training[other] for other in names if other != name],
                                                     economics.preparation, settings), training[name])
              for name in names}
    selected = None
    for threshold in thresholds:
        outcomes = [economics.outcome(len(training[name]), classifier_order(scores[name], threshold)) for name in names]
        rate = pooled_rate(outcomes)
        if selected is None or rate > selected[1]:
            selected = (float(threshold), rate, sum(o.failed for o in outcomes))
    model = fit_classifier([training[name] for name in names], economics.preparation, settings)
    return model, ClassifierSelection(*selected, len(thresholds))


def risk_classifier_orders(model, threshold: float, inputs: Sequence[np.ndarray]) -> list[int | None]:
    """First order minutes of a fitted classifier on evaluation bearings."""
    return [classifier_order(classifier_scores(model, item), threshold) for item in inputs]
