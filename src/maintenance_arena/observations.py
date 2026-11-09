"""Causal simulated-sensor observation routes.

These implementations receive current measurements only. Simulator truth is
used by the sensor generator, never by a reconstructor.

Every route estimates health and the per-step damage rate from the
measurements alone. The preparation time ``L`` (``mobilization_lead``) enters
only afterwards: the rate channel is ``L`` times the rate, clipped to [0, 1],
and the deadline coordinate compares the projected remaining time with ``L + 1``.
``L = 12`` reproduces the published observation bit for bit.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral
import statistics
from typing import Mapping, Any

import numpy as np

DCRIT = math.log(5.)
ROUTE_IDS = ("RAW_CAUSAL", "FILTERED_CAUSAL_EWLS8", "ROBUST_MULTISCALE_234")
SENSOR_LEVELS = {"exact": (0., 0., 0.), "gaussian_005": (.005, 0., 0.), "light": (.0025, .00025, .01),
                 "moderate": (.005, .0005, .05), "stress": (.01, .001, .10)}
BIAS_RHO = .98
DEFAULT_LEAD = 12


def _lead(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError("mobilization_lead must be an integer >= 1")
    return int(value)


def observed_damage(measured_health: float) -> float:
    health = float(np.float32(np.clip(float(measured_health), 0., 1.)))
    return float(np.clip(-math.log(max(health, 1.e-6)) / DCRIT, 0., 1.))


def valid_state(estimated_health: float, lead_scaled_rate: float, mobilization_lead: int = DEFAULT_LEAD) -> np.ndarray:
    """Observation with a valid rate: health, ``L`` times the rate, flag 1 and the deadline coordinate.

The projected remaining time is ``(1 - d) / max(rate / L, 1e-12)`` with the clipped
float32 rate, and the coordinate is ``clip((L + 1 - T) / (L + 1), -1, 1)``.
"""
    lead = _lead(mobilization_lead)
    health = np.float32(np.clip(float(estimated_health), 0., 1.))
    rate = np.float32(np.clip(float(lead_scaled_rate), 0., 1.))
    remaining = (1. - observed_damage(float(health))) / max(float(rate) / lead, 1.e-12)
    slack = np.float32(np.clip((lead + 1. - remaining) / (lead + 1.), -1., 1.))
    return np.asarray((health, rate, np.float32(1.), slack), dtype=np.float32)


def invalid_state(carried_health: float) -> np.ndarray:
    return np.asarray((np.float32(carried_health), 0., 0., -1.), dtype=np.float32)


def measurement_path(health: np.ndarray, *, level: str, white_noise: np.ndarray,
                     initial_bias: float, bias_innovations: np.ndarray,
                     missing_uniforms: np.ndarray) -> dict[str, np.ndarray]:
    """Original clipped additive noise with stationary AR(1) bias initialization."""
    if level not in SENSOR_LEVELS:
        raise ValueError(f"unknown sensor level {level!r}")
    curve = np.ascontiguousarray(health, dtype=np.float32)
    white, innovations, uniforms = (np.asarray(x, dtype=np.float64) for x in
                                    (white_noise, bias_innovations, missing_uniforms))
    if curve.ndim != 1 or curve.size < 2 or any(x.shape != curve.shape for x in (white, innovations, uniforms)):
        raise ValueError("sensor arrays must share one nonempty lifecycle shape")
    if not all(np.isfinite(x).all() for x in (curve, white, innovations, uniforms)) or not math.isfinite(initial_bias):
        raise ValueError("sensor inputs must be finite")
    sigma_epsilon, sigma_eta, missing_probability = SENSOR_LEVELS[level]
    bias = np.empty(curve.size, dtype=np.float64)
    bias[0] = 0. if sigma_eta == 0. else sigma_eta / math.sqrt(1. - BIAS_RHO ** 2) * float(initial_bias)
    for age in range(1, curve.size):
        bias[age] = BIAS_RHO * bias[age - 1] + sigma_eta * float(innovations[age])
    measured = np.clip(curve.astype(np.float64) + bias + sigma_epsilon * white, 0., 1.).astype(np.float32)
    available = np.asarray(uniforms >= missing_probability, dtype=np.bool_)
    available[0] = True
    return {"measured_health": measured, "available": available, "bias": bias}


def sample_measurements(health: np.ndarray, level: str,
                        rngs: Mapping[str, np.random.Generator]) -> dict[str, np.ndarray]:
    size = len(health)
    innovations, uniforms = np.zeros(size), np.ones(size)
    innovations[1:] = rngs["bias"].standard_normal(size - 1)
    uniforms[1:] = rngs["missing"].random(size - 1)
    return measurement_path(health, level=level, white_noise=rngs["white"].standard_normal(size),
                            initial_bias=float(rngs["initial"].standard_normal()),
                            bias_innovations=innovations, missing_uniforms=uniforms)


class Reconstructor:
    """Renewal-local sample stream; a missing acquisition is represented by None.

``mobilization_lead`` scales the rate channel and the deadline coordinate. The
default 12 is a class attribute, so the published snapshot fields are unchanged;
another lead is stored with the snapshot and checked on restore.
"""
    mobilization_lead = DEFAULT_LEAD

    def __init__(self, route: str = "RAW_CAUSAL", mobilization_lead: int = DEFAULT_LEAD) -> None:
        if route not in ROUTE_IDS:
            raise ValueError(f"unknown observation route {route!r}")
        self.route = route
        lead = _lead(mobilization_lead)
        if lead != DEFAULT_LEAD:
            self.mobilization_lead = lead
        self.reset()

    def reset(self) -> None:
        self.age = -1
        self.last_observed_age = -1
        self.last_health: float | None = None
        self.previous_available = False
        self.previous_health: float | None = None
        self.history: list[tuple[int, float]] = []
        self.measurement_gap = 0
        self.residual: float | None = None
        self.reason: str | None = "age_zero"

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    def restore(self, state: Mapping[str, Any]) -> None:
        if (state.get("route") != self.route or set(state) != set(self.__dict__)
                or state.get("mobilization_lead", DEFAULT_LEAD) != self.mobilization_lead):
            raise ValueError("reconstructor snapshot route, lead or fields differ")
        self.__dict__.update(copy.deepcopy(dict(state)))
        self.history = [(int(age), float(value)) for age, value in self.history]

    def step(self, current_measurement: float | None) -> np.ndarray:
        if current_measurement is not None and not math.isfinite(float(current_measurement)):
            raise ValueError("measurement must be finite or None")
        if self.age == -1 and current_measurement is None:
            raise ValueError("the first component measurement must be available")
        self.age += 1
        available = current_measurement is not None
        current = None
        if available:
            current = float(np.float32(np.clip(float(current_measurement), 0., 1.)))
            self.last_health = current
            self.last_observed_age = self.age
            self.history.append((self.age, current))
        assert self.last_health is not None
        self.measurement_gap = self.age - self.last_observed_age
        self.residual = None
        fallback = invalid_state(self.last_health)
        lead = self.mobilization_lead
        if self.route == "RAW_CAUSAL":
            if self.age > 0 and available and self.previous_available and self.previous_health is not None:
                increment = observed_damage(self.last_health) - observed_damage(self.previous_health)
                rate = float(np.float32(np.clip(lead * max(increment, 0.), 0., 1.)))
                result = valid_state(self.last_health, rate, lead)
                self.reason = None
            else:
                result = fallback
                self.reason = "age_zero" if self.age == 0 else (
                    "current_measurement_missing" if not available else "previous_measurement_missing")
            self.previous_available, self.previous_health = available, current
            # The raw method needs no long history.
            self.history = self.history[-2:]
            return result
        if self.route == "FILTERED_CAUSAL_EWLS8":
            self.history = [row for row in self.history if row[0] >= self.age - 7]
            if len(self.history) < 2 or self.measurement_gap > 2:
                self.reason = "insufficient_observed_points" if len(self.history) < 2 else "measurement_gap_exceeds_two"
                return fallback
            indices = np.asarray([row[0] for row in self.history], dtype=np.int64)
            x = indices.astype(np.float64) - float(self.age)
            z = np.asarray([observed_damage(row[1]) for row in self.history], dtype=np.float64)
            weights = np.power(7. / 9., self.age - indices).astype(np.float64)
            x_bar, z_bar = float(np.average(x, weights=weights)), float(np.average(z, weights=weights))
            denominator = float(np.sum(weights * np.square(x - x_bar)))
            if denominator <= 0:
                raise ValueError("EWLS requires distinct acquisition positions")
            slope = max(0., float(np.sum(weights * (x - x_bar) * (z - z_bar)) / denominator))
            damage = float(np.clip(z_bar - slope * x_bar, 0., 1.))
            fitted = damage + slope * x
            self.residual = float(math.sqrt(max(0., float(np.average(np.square(z - fitted), weights=weights)))))
            self.reason = None
            return valid_state(float(5. ** (-damage)), float(np.clip(lead * slope, 0., 1.)), lead)
        by_age = {int(age): observed_damage(value) for age, value in self.history}
        endpoint = max(by_age)
        secants = []
        for horizon in (2, 3, 4):
            if endpoint - horizon in by_age:
                slope = (by_age[endpoint] - by_age[endpoint - horizon]) / float(horizon)
                secants.append((slope, by_age[endpoint] + slope * float(self.age - endpoint)))
        # Preserve the latest observed endpoint and its four-step backward support.
        self.history = [row for row in self.history if row[0] >= endpoint - 4]
        if len(secants) < 2:
            self.reason = "insufficient_secants"
            return fallback
        derivative = float(statistics.median([row[0] for row in secants]))
        projections = [row[1] for row in secants]
        damage = float(np.clip(statistics.median(projections), 0., 1.))
        self.residual = float(math.sqrt(statistics.fmean((value - damage) ** 2 for value in projections)))
        self.reason = None
        health = np.float32(np.clip(5. ** (-damage), 0., 1.))
        rate = np.float32(np.clip(lead * max(0., derivative), 0., 1.))
        return valid_state(float(health), float(rate), lead)


def reconstruct(measured_health: np.ndarray, available: np.ndarray | None = None,
                route: str = "RAW_CAUSAL", mobilization_lead: int = DEFAULT_LEAD) -> dict[str, Any]:
    """Reconstruct the causal observation at every age of one measured life."""
    measured = np.asarray(measured_health, dtype=np.float32)
    present = np.ones(measured.shape, dtype=bool) if available is None else np.asarray(available, dtype=bool)
    if measured.ndim != 1 or measured.size == 0 or present.shape != measured.shape:
        raise ValueError("measurements and availability must be equal one-dimensional arrays")
    estimator = Reconstructor(route, mobilization_lead)
    states, gaps, residuals, reasons = [], [], [], []
    for value, is_present in zip(measured, present, strict=True):
        states.append(estimator.step(float(value) if is_present else None))
        gaps.append(estimator.measurement_gap)
        residuals.append(float("nan") if estimator.residual is None else estimator.residual)
        reasons.append(estimator.reason)
    return {"states": np.stack(states), "measurement_gap": np.asarray(gaps, dtype=np.int32),
            "residual": np.asarray(residuals, dtype=np.float64), "reasons": reasons}
