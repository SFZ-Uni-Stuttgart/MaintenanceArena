"""The supervised risk classifier on bearing policy inputs."""
import numpy as np
import pytest

from maintenance_arena.bearings import ReplayEconomics, calibrate_risk_classifier, classifier_order, risk_classifier_orders
from maintenance_arena.bearings.classifier import THRESHOLDS, late_labels


def test_late_labels_mark_the_minutes_from_the_last_safe_order():
    labels = late_labels(20, 12)
    assert labels.tolist() == [0.0] * 7 + [1.0] * 13  # minute 8 = T - L is the first late minute
    with pytest.raises(ValueError):
        late_labels(12, 12)


def test_order_is_the_first_minute_whose_score_reaches_the_threshold():
    scores = np.array([0.1, 0.5, 0.7])
    assert classifier_order(scores, 0.5) == 2
    assert classifier_order(scores, 0.9) is None
    assert THRESHOLDS[0] == 0.01 and THRESHOLDS[-1] == 0.99 and len(THRESHOLDS) == 99


def _bearing(life, seed):
    rng = np.random.default_rng(seed)
    remaining = (life - np.arange(1, life + 1)) / 100.0
    return np.column_stack([remaining + rng.normal(0, 0.01, life), rng.normal(0, 1, life)]).astype(np.float32)


def test_calibration_is_deterministic_and_orders_in_time_on_a_clear_signal():
    pytest.importorskip("sklearn")
    training = {f"B{i}": _bearing(life, i) for i, life in enumerate((90, 120, 150, 180, 210))}
    economics = ReplayEconomics()
    model, chosen = calibrate_risk_classifier(training, economics)
    again, repeated = calibrate_risk_classifier(dict(reversed(list(training.items()))), economics)
    assert chosen == repeated and chosen.threshold in THRESHOLDS and chosen.candidates == 99
    test = [_bearing(life, 10 + life) for life in (100, 160)]
    orders = risk_classifier_orders(model, chosen.threshold, test)
    assert orders == risk_classifier_orders(again, chosen.threshold, test)
    assert all(order is not None and order <= len(x) - economics.preparation for order, x in zip(orders, test))


def test_the_smallest_threshold_wins_a_tie():
    pytest.importorskip("sklearn")
    training = {f"B{i}": _bearing(life, i) for i, life in enumerate((90, 120, 150))}
    _, chosen = calibrate_risk_classifier(training, thresholds=(0.5, 0.5000001))
    assert chosen.threshold == 0.5
    with pytest.raises(ValueError):
        calibrate_risk_classifier({"B0": training["B0"]})
