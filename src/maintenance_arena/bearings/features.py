"""Causal condition features, forecaster inputs and policy observations of bearing histories.

Every value at minute t uses minutes 1..t only. RMS and peak amplitudes are referenced
to each bearing's own running-in, the median over its first ten minutes, and enter as
logarithms. The policy observation combines the forecast with condition indicators and,
depending on the input design, operating age.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .data import BearingHistory

RULE_SMOOTHING = (0.1, 0.3, 1.0)
OBSERVATION_NAMES = ("forecast", "deadline_coordinate", "forecast_trend", "rms_level", "rms_trend",
                     "peak_level", "age")
# Bounds implied by the clipping of each coordinate.
OBSERVATION_BOUNDS = {"forecast": (0.0, 1.2), "deadline_coordinate": (-1.0, 1.0), "forecast_trend": (-1.0, 0.6),
                      "rms_level": (-0.5, 2.0), "rms_trend": (-1.0, 1.0), "peak_level": (-0.5, 2.0),
                      "age": (0.0, 4.0)}


@dataclass(frozen=True)
class FeatureSettings:
    """Running-in reference, smoothing and trend windows of the bearing features."""
    baseline_minutes: int = 10
    ewm_alpha: float = 0.3
    slope_window: int = 10
    minute_scale: float = 200.0
    trend_window: int = 10

    def __post_init__(self):
        for name in ("baseline_minutes", "slope_window", "trend_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.ewm_alpha <= 1 or not self.minute_scale > 0:
            raise ValueError("ewm_alpha must lie in (0, 1] and minute_scale must be positive")


@dataclass(frozen=True)
class InputDesign:
    """Where operating age enters a learned decision: forecaster, policy, both or neither.

    ``forecaster_age`` is None for a design without forecaster.
    """
    name: str
    forecaster_age: bool | None
    observation_names: tuple[str, ...]


INPUT_DESIGNS = {
    "full": InputDesign("Full", True, OBSERVATION_NAMES),
    "no_age": InputDesign("No age", False, OBSERVATION_NAMES[:6]),
    "no_policy_age": InputDesign("No policy age", True, OBSERVATION_NAMES[:6]),
    "no_forecaster_age": InputDesign("No forecaster age", False, OBSERVATION_NAMES),
    "condition_only": InputDesign("Condition only", None, OBSERVATION_NAMES[3:6]),
}


def input_design(key: str) -> InputDesign:
    if key not in INPUT_DESIGNS:
        raise ValueError(f"unknown input design {key!r}; expected one of {sorted(INPUT_DESIGNS)}")
    return INPUT_DESIGNS[key]


def ewm(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted mean along the first axis, started at the first value."""
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


def trailing_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope per minute over the last min(t, window) values ending at t; 0 with fewer than 3."""
    x = np.asarray(values, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    out = np.zeros_like(x)
    for t in range(len(x)):
        n = min(t + 1, window)
        if n < 3:
            continue
        segment = x[t - n + 1:t + 1]
        s = np.arange(n, dtype=np.float64)
        s -= s.mean()
        out[t] = s @ (segment - segment.mean(axis=0)) / (s @ s)
    return out[:, 0] if squeeze else out


def running_in_reference(values: np.ndarray, minutes: int) -> np.ndarray:
    """Median of the first min(t, minutes) minutes, available at minute t."""
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    for t in range(len(x)):
        out[t] = np.median(x[:min(t + 1, minutes)], axis=0)
    return out


def relative_signals(history: BearingHistory, settings: FeatureSettings | None = None) -> dict[str, np.ndarray]:
    """Log RMS and peak relative to running-in (``lr``, ``lp``) and log kurtosis over 3 (``lk``)."""
    settings = settings or FeatureSettings()
    m = history.minutes
    if np.any(m[:, 0:4] <= 0):
        raise ValueError(f"{history.name}: RMS and peak statistics must be positive")
    base = running_in_reference(m[:, 0:4], settings.baseline_minutes)
    return {"lr": np.log(m[:, 0:2] / base[:, 0:2]), "lp": np.log(m[:, 2:4] / base[:, 2:4]),
            "lk": np.log(np.maximum(m[:, 4:6], 1e-6) / 3.0)}


def forecaster_features(history: BearingHistory, settings: FeatureSettings | None = None, *,
                        age: bool = True) -> np.ndarray:
    """Forecaster inputs: EWM of the six log statistics, their slopes, channel maxima of lr and lp, minute.

    Fifteen columns with ``age``; without it the final minute/200 column is omitted.
    """
    settings = settings or FeatureSettings()
    signals = relative_signals(history, settings)
    z = np.hstack([signals["lr"], signals["lp"], signals["lk"]])
    smooth = ewm(z, settings.ewm_alpha)
    columns = [smooth, trailing_slope(smooth, settings.slope_window),
               signals["lr"].max(axis=1, keepdims=True), signals["lp"].max(axis=1, keepdims=True)]
    if age:
        columns.append(np.arange(1, history.life + 1, dtype=np.float64)[:, None] / settings.minute_scale)
    return np.hstack(columns)


def decision_series(history: BearingHistory, forecast: np.ndarray | None = None,
                    settings: FeatureSettings | None = None, *,
                    smoothing: Sequence[float] = RULE_SMOOTHING) -> dict[str, np.ndarray]:
    """Causal decision series shared by the rules and the policy observations.

    ``level`` and ``peak`` are the EWM of the channel maxima of ``lr`` and ``lp``;
    ``level_a{a}`` and ``peak_a{a}`` smooth the same maxima at the rule rates. With a
    forecast (remaining minutes per minute) the series add ``forecast`` and its trend.
    """
    settings = settings or FeatureSettings()
    signals = relative_signals(history, settings)
    raw_level, raw_peak = signals["lr"].max(axis=1), signals["lp"].max(axis=1)
    level = ewm(raw_level, settings.ewm_alpha)
    series = {"minute": np.arange(1, history.life + 1, dtype=np.float64), "raw_level": raw_level,
              "raw_peak": raw_peak, "level": level, "peak": ewm(raw_peak, settings.ewm_alpha),
              "level_trend": trailing_slope(level, settings.trend_window)}
    for alpha in smoothing:
        series[f"level_a{alpha:g}"] = ewm(raw_level, alpha)
        series[f"peak_a{alpha:g}"] = ewm(raw_peak, alpha)
    if forecast is not None:
        values = np.asarray(forecast, dtype=np.float64)
        if values.shape != (history.life,) or not np.isfinite(values).all():
            raise ValueError(f"{history.name}: the forecast needs one finite value per minute")
        series["forecast"] = values
        series["forecast_trend"] = trailing_slope(values, settings.trend_window)
    return series


def _column(series: Mapping[str, np.ndarray], name: str, horizon: float) -> np.ndarray:
    if name == "forecast":
        return np.clip(series["forecast"] / 125.0, 0.0, 1.2)
    if name == "deadline_coordinate":
        return np.clip((horizon - (series["forecast"] + 1.0)) / horizon, -1.0, 1.0)
    if name == "forecast_trend":
        return np.clip(series["forecast_trend"], -5.0, 3.0) / 5.0
    if name == "rms_level":
        return np.clip(series["level"], -1.0, 4.0) / 2.0
    if name == "rms_trend":
        return np.clip(series["level_trend"] * 10.0, -1.0, 1.0)
    if name == "peak_level":
        return np.clip(series["peak"], -1.0, 4.0) / 2.0
    if name == "age":
        return np.clip(series["minute"] / 250.0, 0.0, 4.0)
    raise ValueError(f"unknown observation {name!r}; expected one of {OBSERVATION_NAMES}")


def policy_observations(series: Mapping[str, np.ndarray], names: Sequence[str] = OBSERVATION_NAMES, *,
                        preparation: int = 12) -> np.ndarray:
    """Float32 policy inputs, one row per minute, with the named columns.

    forecast: clip(r/125, 0, 1.2); deadline_coordinate: clip((L+1-(r+1))/(L+1), -1, 1);
    forecast_trend: clip(slope(r), -5, 3)/5; rms_level: clip(level, -1, 4)/2;
    rms_trend: clip(10 slope(level), -1, 1); peak_level: clip(peak, -1, 4)/2;
    age: clip(minute/250, 0, 4).
    """
    if not names:
        raise ValueError("at least one observation column is required")
    if any(name in ("forecast", "deadline_coordinate", "forecast_trend") for name in names) and "forecast" not in series:
        raise ValueError("forecast observations need a series built with a forecast")
    horizon = preparation + 1.0
    return np.column_stack([_column(series, name, horizon) for name in names]).astype(np.float32)


def design_observations(series: Mapping[str, np.ndarray], design: str | InputDesign, *,
                        preparation: int = 12) -> np.ndarray:
    """Policy inputs of an input design from series built with that design's forecast."""
    spec = input_design(design) if isinstance(design, str) else design
    return policy_observations(series, spec.observation_names, preparation=preparation)
