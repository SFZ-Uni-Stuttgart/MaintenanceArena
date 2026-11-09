"""Analytic degradation laws of the core and resource tasks.

Random sampling uses NumPy Generators owned by this package. A seed does not
select a scenario of the experiments reported in the accompanying article;
``maintenance_arena.banks`` rebuilds the article's finite scenario banks.

The core has four source laws (``FAMILY_IDS``) and four new laws
(``TARGET_FAMILY_IDS``: Paris-inspired, Two-knee, Gompertz and Power-law) that
change when damage accelerates while the lifetime distribution stays the same.
"""
from __future__ import annotations

import math
import hashlib
import json
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

FAMILY_IDS = ("linear_reference", "concave_running_in",
              "convex_accelerating_fatigue", "bearing_late_knee")
TARGET_FAMILY_IDS = ("paris_inspired", "two_knee", "gompertz", "power_law")
LAW_IDS = (*FAMILY_IDS, *TARGET_FAMILY_IDS)
# Shape ranges of the new laws, in the order in which they are drawn from a
# shape stream. Each parameter is uniform within its range (article Appendix B).
TARGET_SHAPE_RANGES: dict[str, tuple[tuple[str, tuple[float, float]], ...]] = {
    "paris_inspired": (("w", (.15, .45)), ("q", (.55, .8)), ("p", (.7, 1.4))),
    "two_knee": (("w", (.08, .22)), ("k1", (.3, .45)), ("k2", (.72, .86)), ("v", (.2, .4)),
                 ("s", (.025, .045))),
    "gompertz": (("A", (4., 8.)), ("i", (.4, .65))),
    "power_law": (("beta", (2.8, 3.4)),),
}
DCRIT = math.log(5.0)
START_OFFSETS = (None, 29, 21, 17, None, 15, 14, 13)


def _sample_range(rng: np.random.Generator, values: Sequence[float]) -> float:
    if len(values) != 2:
        raise ValueError("parameter ranges must contain two values")
    return float(rng.uniform(float(values[0]), float(values[1])))


def _softplus(values: np.ndarray | float, softness: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    scaled = np.clip(array / float(softness), -60.0, 60.0)
    return float(softness) * np.log1p(np.exp(scaled))


def _normalized_curve(law: str, u: np.ndarray, parameters: Mapping[str, Any],
                      rng: np.random.Generator) -> np.ndarray:
    """Normalized degradation laws, shared with the resource simulator."""
    values = np.asarray(u, dtype=np.float64)
    if law == "normalized_linear":
        curve = values
    elif law == "normalized_power":
        curve = np.power(values, _sample_range(rng, parameters["beta_range"]))
    elif law == "normalized_softplus_knee":
        weight = _sample_range(rng, parameters["linear_weight_range"])
        knee = _sample_range(rng, parameters["knee_fraction_range"])
        softness = _sample_range(rng, parameters["softness_range"])
        offset = _softplus(-knee, softness)
        raw = weight * values + (1.0 - weight) * (_softplus(values - knee, softness) - offset)
        denominator = weight + (1.0 - weight) * (_softplus(1.0 - knee, softness) - offset)
        curve = raw / float(denominator)
    elif law == "normalized_paris":
        q_value = _sample_range(rng, parameters["q_range"])
        power = _sample_range(rng, parameters["p_range"])
        curve = (np.power(1.0 - q_value * values, -power) - 1.0) / ((1.0 - q_value) ** (-power) - 1.0)
    elif law == "normalized_gompertz":
        a_value = _sample_range(rng, parameters["a_range"])
        inflection = _sample_range(rng, parameters["inflection_fraction_range"])
        b_value = math.log(a_value) / inflection
        raw = np.exp(-a_value * np.exp(-b_value * values))
        lower = math.exp(-a_value)
        upper = math.exp(-a_value * math.exp(-b_value))
        curve = (raw - lower) / (upper - lower)
    elif law == "normalized_avrami":
        kappa = _sample_range(rng, parameters["kappa_range"])
        exponent = _sample_range(rng, parameters["exponent_range"])
        curve = (1.0 - np.exp(-kappa * np.power(values, exponent))) / (1.0 - math.exp(-kappa))
    elif law == "normalized_linear_paris":
        weight = _sample_range(rng, parameters["linear_weight_range"])
        q_value = _sample_range(rng, parameters["q_range"])
        power = _sample_range(rng, parameters["p_range"])
        paris = (np.power(1.0 - q_value * values, -power) - 1.0) / ((1.0 - q_value) ** (-power) - 1.0)
        curve = weight * values + (1.0 - weight) * paris
    elif law == "normalized_bi_knee":
        linear = _sample_range(rng, parameters["linear_weight_range"])
        first = _sample_range(rng, parameters["first_knee_range"])
        second = _sample_range(rng, parameters["second_knee_range"])
        mix = _sample_range(rng, parameters["first_knee_weight_range"])
        softness = _sample_range(rng, parameters["softness_range"])
        first_term = _softplus(values - first, softness) - _softplus(-first, softness)
        second_term = _softplus(values - second, softness) - _softplus(-second, softness)
        raw = linear * values + (1.0 - linear) * (mix * first_term + (1.0 - mix) * second_term)
        denominator = linear + (1.0 - linear) * (
            mix * float(_softplus(1.0 - first, softness) - _softplus(-first, softness))
            + (1.0 - mix) * float(_softplus(1.0 - second, softness) - _softplus(-second, softness)))
        curve = raw / denominator
    else:
        raise ValueError(f"Unsupported analytic degradation law {law!r}")
    curve = np.maximum.accumulate(np.asarray(curve, dtype=np.float64))
    curve[0], curve[-1] = 0.0, 1.0
    if not np.isfinite(curve).all() or np.any(np.diff(curve) < -1e-12):
        raise RuntimeError(f"non-monotone or non-finite curve for {law}")
    return curve


def shape_parameters(family_id: str, probability: float = .5) -> tuple[tuple[str, float], ...]:
    """Continuous source shape at a quantile; .5 gives the fixed laws of the discrete source."""
    q = float(probability)
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("shape quantile must lie in [0,1]")
    if family_id == "linear_reference":
        return ()
    if family_id == "concave_running_in":
        return (("c", 3.0 + 2.0 * q),)
    if family_id == "convex_accelerating_fatigue":
        return (("beta", 1.6 + .8 * q),)
    if family_id == "bearing_late_knee":
        return (("k", .62 + .16 * q), ("s", .04), ("w", .15))
    if family_id in TARGET_FAMILY_IDS:
        raise ValueError(f"{family_id} has several independent shape parameters; draw them with sample_target_shape")
    raise ValueError(f"unknown core family {family_id!r}")


def sample_target_shape(family_id: str, rng: np.random.Generator) -> tuple[tuple[str, float], ...]:
    """Draw the shape of a new law uniformly within its ranges, in the order of ``TARGET_SHAPE_RANGES``."""
    if family_id not in TARGET_SHAPE_RANGES:
        raise ValueError(f"{family_id!r} is not one of the new laws {TARGET_FAMILY_IDS}")
    return tuple((name, float(rng.uniform(low, high))) for name, (low, high) in TARGET_SHAPE_RANGES[family_id])


def _target_damage(family_id: str, u: np.ndarray, shape: Mapping[str, float]) -> np.ndarray:
    """Damage g(u) of a new law (article Appendix B), before the endpoints are fixed at 0 and 1.

The operation order is part of the scenario definition: it fixes the float32
health of every life in the article's banks.
"""
    names = [name for name, _ in TARGET_SHAPE_RANGES[family_id]]
    if sorted(shape) != sorted(names):
        raise ValueError(f"{family_id} requires the shape parameters {names}")
    if family_id == "paris_inspired":
        w, q, p = shape["w"], shape["q"], shape["p"]
        paris = ((1. - q * u) ** (-p) - 1.) / ((1. - q) ** (-p) - 1.)
        return w * u + (1. - w) * paris
    if family_id == "two_knee":
        w, k1, k2, v, s = (shape[name] for name in ("w", "k1", "k2", "v", "s"))

        def knee(x: np.ndarray | float, k: float) -> np.ndarray:
            return s * (np.logaddexp(0., (x - k) / s) - np.logaddexp(0., -k / s))

        def raw(x: np.ndarray | float) -> np.ndarray:
            return w * x + (1. - w) * (v * knee(x, k1) + (1. - v) * knee(x, k2))
        return raw(u) / raw(1.)
    if family_id == "gompertz":
        a, inflection = shape["A"], shape["i"]
        b = math.log(a) / inflection
        lower, upper = math.exp(-a), math.exp(-a * math.exp(-b))
        return (np.exp(-a * np.exp(-b * u)) - lower) / (upper - lower)
    return u ** shape["beta"]


def normalized_damage_fraction(family_id: str, normalized_age: np.ndarray,
                               parameters: Sequence[tuple[str, float]]) -> np.ndarray:
    values = np.asarray(normalized_age, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("normalized ages must be a finite vector in [0,1]")
    shape = dict(parameters)
    if len(shape) != len(parameters) or not all(math.isfinite(float(v)) for v in shape.values()):
        raise ValueError("shape parameters must be unique and finite")
    if family_id == "linear_reference":
        curve = values
    elif family_id == "concave_running_in":
        c = shape["c"]
        curve = np.log1p(c * values) / math.log1p(c)
    elif family_id == "convex_accelerating_fatigue":
        curve = np.power(values, shape["beta"])
    elif family_id == "bearing_late_knee":
        k, s, w = shape["k"], shape["s"], shape["w"]
        def softplus(argument: np.ndarray | float) -> np.ndarray:
            return s * np.logaddexp(0., np.asarray(argument, dtype=np.float64) / s)
        offset = softplus(-k)
        raw = w * values + (1. - w) * (softplus(values - k) - offset)
        denominator = w + (1. - w) * (float(softplus(1. - k)) - float(offset))
        curve = raw / denominator
    elif family_id in TARGET_FAMILY_IDS:
        curve = _target_damage(family_id, values, shape)
    else:
        raise ValueError(f"unknown core family {family_id!r}")
    result = np.ascontiguousarray(np.asarray(curve, dtype=np.float64))
    result[0], result[-1] = 0., 1.
    if not np.isfinite(result).all() or np.any(np.diff(result) <= 0):
        raise ValueError("normalized damage must be strictly increasing")
    return result


def materialize_health(family_id: str, eol_operating_age: int,
                       parameters: Sequence[tuple[str, float]] | None = None) -> np.ndarray:
    eol = int(eol_operating_age)
    if isinstance(eol_operating_age, bool) or eol != eol_operating_age or eol < 2:
        raise ValueError("lifetime must be an integer of at least two operating steps")
    shape = shape_parameters(family_id) if parameters is None else parameters
    ages = np.arange(eol + 1, dtype=np.float64)
    damage = normalized_damage_fraction(family_id, ages / float(eol), shape)
    health = np.exp(-DCRIT * damage).astype(np.float32)
    health[0], health[-1] = np.float32(1.), np.float32(.2)
    if not np.isfinite(health).all() or np.any(np.diff(health.astype(np.float64)) >= 0):
        raise ValueError("materialized Float32 health must decrease")
    return np.ascontiguousarray(health)


def truncated_lognormal_inverse(probability: float) -> float:
    q = float(probability)
    if not math.isfinite(q) or not 0 < q < 1:
        raise ValueError("lifetime quantile must lie in (0,1)")
    normal = NormalDist()
    mu, sigma = math.log(128.), .12
    low = normal.cdf((math.log(95.5) - mu) / sigma)
    high = normal.cdf((math.log(160.5) - mu) / sigma)
    return math.exp(mu + sigma * normal.inv_cdf(low + q * (high - low)))


def sample_curve(rng: np.random.Generator, *, source: str = "continuous",
                 family: str = "random", eol_operating_age: int | None = None) -> dict[str, Any]:
    """Draw one example curve from the package's own random stream.

``family="random"`` draws one of the four source laws. A new law must be named
explicitly; its shape parameters are drawn after the lifetime, in the order of
``TARGET_SHAPE_RANGES``. The discrete source has fixed shapes for the source laws only.
"""
    if source not in ("continuous", "discrete"):
        raise ValueError("source must be continuous or discrete")
    family_id = str(rng.choice(FAMILY_IDS)) if family == "random" else family
    if family_id not in LAW_IDS:
        raise ValueError(f"unknown family {family_id!r}")
    if source == "discrete" and family_id in TARGET_FAMILY_IDS:
        raise ValueError("the discrete source covers the four source laws only")
    if source == "continuous":
        q = float(np.clip(rng.random(), np.nextafter(0., 1.), np.nextafter(1., 0.)))
        t_cont = truncated_lognormal_inverse(q)
        eol = int(math.floor(t_cont + .5)) if eol_operating_age is None else int(eol_operating_age)
        shape = (sample_target_shape(family_id, rng) if family_id in TARGET_FAMILY_IDS
                 else shape_parameters(family_id, float(rng.random())))
    else:
        eol = int(rng.choice((112, 128, 144))) if eol_operating_age is None else int(eol_operating_age)
        t_cont = float(eol)
        shape = shape_parameters(family_id)
    return {"family_id": family_id, "eol_operating_age": eol,
            "t_cont": t_cont, "shape_parameters": list(shape),
            "health": materialize_health(family_id, eol, shape).tolist()}


def training_stream_seed(seed: int, stream: str, family: str | None = None) -> int:
    """Seed of an addressed stream of the package's training bank."""
    text = f"maintenance_arena_training_bank_v1|{int(seed)}|{stream}"
    if family is not None:
        text += f"|{family}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def build_training_bank(seed: int, *, source: str = "continuous",
                        pair_count: int | None = None) -> dict[str, Any]:
    """Materialize the published curriculum using independently owned public seeds.

Continuous: per-family, independently permuted lifetime/shape Latin hypercubes;
each shuffled 32-pair block contains all four families at all eight start slots.
Successors use independent draws and a global permutation. Discrete: the fixed
96-cell slot/lifetime/family schedule, explicitly unrolled into a finite bank.
Exhaustion is handled by the environment; this builder does not wrap at runtime.
"""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    if source not in ("continuous", "discrete"):
        raise ValueError("unknown training source")
    count = (16384 if source == "continuous" else 16416) if pair_count is None else pair_count
    multiple = 32 if source == "continuous" else 96
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0 or count % multiple:
        raise ValueError(f"{source} training bank must contain complete {multiple}-pair blocks")
    count = int(count)

    def rng(stream, family=None):
        return np.random.Generator(np.random.PCG64(training_stream_seed(seed, stream, family)))

    def family_draws(side, family, n):
        def lhs(dimension):
            probabilities = (np.arange(n, dtype=np.float64) + rng(f"{side}_{dimension}_jitter", family).random(n)) / float(n)
            order = np.asarray(rng(f"{side}_{dimension}_permutation", family).permutation(n), dtype=np.int64)
            return probabilities[order], np.arange(n, dtype=np.int64)[order]
        lifetime_q, lifetime_strata = lhs("lifetime")
        shape_q, shape_strata = (None, None) if family == "linear_reference" else lhs("shape")
        rows = []
        for i in range(n):
            t_cont = truncated_lognormal_inverse(float(lifetime_q[i]))
            eol = math.floor(t_cont + .5)
            shape_quantile = None if shape_q is None else float(shape_q[i])
            shape = shape_parameters(family, .5 if shape_quantile is None else shape_quantile)
            rows.append({"family_id": family, "eol_operating_age": eol, "t_cont": t_cont,
                "shape_parameters": list(shape), "lifetime_probability": float(lifetime_q[i]),
                "lifetime_stratum_index": int(lifetime_strata[i]), "shape_probability": shape_quantile,
                "shape_stratum_index": None if shape_strata is None else int(shape_strata[i])})
        return rows

    if source == "continuous":
        per_family = count // 4
        active_family = {family: family_draws("active", family, per_family) for family in FAMILY_IDS}
        arranged = []
        block_rng = rng("training_block_row_order")
        for block in range(per_family // 8):
            emitted = [(active_family[family][block * 8 + slot], slot)
                       for slot in range(8) for family in FAMILY_IDS]
            arranged.extend(emitted[int(i)] for i in block_rng.permutation(32))
        successors = [row for family in FAMILY_IDS for row in family_draws("successor", family, per_family)]
        successors = [successors[int(i)] for i in rng("successor_row_order").permutation(count)]
    else:
        cells = [(family, eol, slot) for slot in range(8) for eol in (112, 128, 144) for family in FAMILY_IDS]
        arranged, successors = [], []
        for i in range(count):
            family, eol, slot = cells[i % 96]
            row = {"family_id": family, "eol_operating_age": eol, "t_cont": float(eol),
                   "shape_parameters": list(shape_parameters(family)), "lifetime_probability": None,
                   "lifetime_stratum_index": None, "shape_probability": None, "shape_stratum_index": None}
            arranged.append((dict(row), slot))
            successors.append(dict(row))
    parameter_pairs = []
    for i, ((active, slot), successor) in enumerate(zip(arranged, successors, strict=True)):
        offset = START_OFFSETS[slot]
        start_age = 0 if offset is None else active["eol_operating_age"] - offset
        parameter_pairs.append({"active": active, "successor": successor,
                                "start_slot_index": slot, "start_age": start_age, "pair_index": i})
    parameter_payload = {"schema_version": "maintenance_arena_training_bank_v1", "source": source,
                         "seed": int(seed), "pair_count": count, "pairs": parameter_pairs}
    digest = hashlib.sha256(json.dumps(parameter_payload, allow_nan=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    for pair in parameter_pairs:
        for side in ("active", "successor"):
            row = pair[side]
            row["health"] = materialize_health(row["family_id"], row["eol_operating_age"], row["shape_parameters"])
            row["health"].setflags(write=False)
    return {**parameter_payload, "sha256": digest}
