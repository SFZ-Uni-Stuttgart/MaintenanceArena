"""Finite scenario banks of the accompanying article and the environment that replays them.

A bank is a list of independent pairs: an active component and the successor
that its exchange installs. ``build_bank`` derives every life from addressed
random streams (law set, split, seed, pair and side), so a law set, split, seed
and pair count rebuild the same float32 health histories bit for bit, whatever
was built before. The article's training, calibration and test lives are such
banks; ``experiments/synthetic`` lists their seeds.

``ScenarioBankEnv`` runs the unchanged core physics on a bank. Each reset installs
the next pair; ``reset(seed=...)`` rewinds to the first pair. A bank is consumed,
never wrapped. Measurement noise is drawn from streams addressed by the component
and a replay seed, so the same component receives the same measurements whatever
the policy did on earlier pairs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from .core import CoreCycleEnv
from .curves import (FAMILY_IDS, TARGET_FAMILY_IDS, materialize_health, sample_target_shape, shape_parameters,
                     truncated_lognormal_inverse)
from .observations import SENSOR_LEVELS, sample_measurements

SCHEMA = "maintenance_arena_scenario_bank_v1"
SNAPSHOT_SCHEMA = "maintenance_arena_scenario_bank_snapshot_v1"
LAW_SETS = ("source", *TARGET_FAMILY_IDS)
SPLITS = ("source_train", "adaptation", "development", "final_evaluation")
# Namespace and domain labels of the addressed random streams. They are part of
# the definition of the article's scenario banks and must not change.
STREAM_NAMESPACE = "adaptive_maintenance_transfer_v1"
STREAM_DOMAINS = {"source": "source", "paris_inspired": "paris_mixture", "two_knee": "bi_knee",
                  "gompertz": "gompertz", "power_law": "power_extrapolation"}
MEASUREMENT_ROLES = ("white", "initial", "bias", "missing")


def canonical_hash(value: Any) -> str:
    """SHA-256 of compact JSON with sorted keys."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def stream_seed(law_set: str, split: str, seed: int, role: str) -> int:
    """Seed of one addressed bank stream: the first 64 bits of a SHA-256 of its address."""
    if law_set not in STREAM_DOMAINS or split not in SPLITS:
        raise ValueError(f"law set must be one of {LAW_SETS} and split one of {SPLITS}")
    payload = [STREAM_NAMESPACE, STREAM_DOMAINS[law_set], split, _integer(seed, "seed"), str(role)]
    return int(canonical_hash(payload)[:16], 16)


def _rng(law_set: str, split: str, seed: int, role: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stream_seed(law_set, split, seed, role)))


def _component(law_set: str, split: str, seed: int, index: int, side: str, family: str) -> dict[str, Any]:
    address = f"{side}:{index}"
    lifetime_rng = _rng(law_set, split, seed, f"{address}:lifetime")
    probability = float(np.clip(lifetime_rng.random(), np.nextafter(0., 1.), np.nextafter(1., 0.)))
    continuous_life = truncated_lognormal_inverse(probability)
    eol = int(math.floor(continuous_life + .5))
    shape_rng = _rng(law_set, split, seed, f"{address}:shape")
    if law_set == "source":
        family_id = family
        parameters = shape_parameters(family_id, float(shape_rng.random()))
    else:
        family_id = law_set
        parameters = sample_target_shape(family_id, shape_rng)
    health = materialize_health(family_id, eol, parameters)
    return {"family_id": family_id, "eol_operating_age": eol, "t_cont": continuous_life,
            "shape_parameters": [[name, value] for name, value in parameters], "health": health.tolist(),
            "stream_address": f"{STREAM_NAMESPACE}/{STREAM_DOMAINS[law_set]}/{split}/{seed}/{address}"}


def build_bank(law_set: str, *, split: str, seed: int, pair_count: int, sensor_level: str = "exact") -> dict[str, Any]:
    """Build a finite bank of independent active/successor pairs.

``law_set`` is ``"source"`` (the four source laws, balanced in blocks of four
pairs, with independent random orders for active and successor components) or
one new law. Lifetimes follow the shared lognormal distribution. Training lives
use the split ``source_train`` (source) or ``adaptation`` (new laws); evaluation
lives use ``development``. ``sensor_level`` selects the measurement noise of the
replaying environment; it does not change the lives.
"""
    if law_set not in LAW_SETS:
        raise ValueError(f"law set must be one of {LAW_SETS}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "source_train" and law_set != "source":
        raise ValueError("the source_train split holds source lives only")
    if split == "adaptation" and law_set == "source":
        raise ValueError("the adaptation split holds lives of a new law")
    if sensor_level not in SENSOR_LEVELS:
        raise ValueError(f"sensor level must be one of {sorted(SENSOR_LEVELS)}")
    seed, pair_count = _integer(seed, "seed"), _integer(pair_count, "pair_count", 1)
    families = {}
    for side in ("active", "successor"):
        order = [FAMILY_IDS[i % len(FAMILY_IDS)] for i in range(pair_count)]
        families[side] = list(_rng(law_set, split, seed, f"{side}:family_order").permutation(order))
    pairs = []
    for index in range(pair_count):
        pair: dict[str, Any] = {"pair_index": index, "start_age": 0}
        for side in ("active", "successor"):
            pair[side] = _component(law_set, split, seed, index, side, str(families[side][index]))
        pairs.append(pair)
    bank = {"schema_version": SCHEMA, "laws": law_set, "split": split, "seed": seed, "pair_count": pair_count,
            "sensor_level": sensor_level, "measurement_seed": stream_seed(law_set, split, seed, "measurements"),
            "pairs": pairs}
    return {**bank, "sha256": canonical_hash(bank)}


def validate_bank(bank: Mapping[str, Any]) -> None:
    """Check the schema, the hash and every component of a bank."""
    if bank.get("schema_version") != SCHEMA:
        raise ValueError("not a scenario bank of this package")
    if bank.get("sha256") != canonical_hash({key: value for key, value in bank.items() if key != "sha256"}):
        raise ValueError("scenario bank hash mismatch")
    pairs = bank.get("pairs")
    if not pairs or bank.get("pair_count") != len(pairs):
        raise ValueError("scenario bank has an inconsistent pair count")
    if bank.get("sensor_level") not in SENSOR_LEVELS:
        raise ValueError("unknown sensor level")
    for pair in pairs:
        for side in ("active", "successor"):
            part = pair[side]
            health = np.asarray(part["health"], dtype=np.float32)
            if (health.shape != (part["eol_operating_age"] + 1,) or health[0] != np.float32(1.)
                    or health[-1] != np.float32(.2) or not np.isfinite(health).all()
                    or np.any(np.diff(health.astype(np.float64)) >= 0) or not part.get("stream_address")):
                raise ValueError("malformed scenario component")
        if not 0 <= pair["start_age"] < pair["active"]["eol_operating_age"]:
            raise ValueError("invalid pair start age")


def _rebind(body: dict[str, Any]) -> dict[str, Any]:
    result = {**body, "sha256": canonical_hash(body)}
    validate_bank(result)
    return result


def subset_bank(bank: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    """A separately hashed bank with the selected pairs, for example to replay single lives."""
    validate_bank(bank)
    chosen = [_integer(index, "pair index") for index in indices]
    if not chosen or len(chosen) != len(set(chosen)) or max(chosen) >= bank["pair_count"]:
        raise ValueError("a subset needs unique pair indices within the bank")
    body = {key: value for key, value in bank.items() if key not in ("sha256", "pairs")}
    body.update(pairs=[copy.deepcopy(bank["pairs"][index]) for index in chosen], pair_count=len(chosen),
                parent_sha256=bank["sha256"], parent_indices=chosen)
    return _rebind(body)


def concatenate_banks(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Append the pairs of ``second`` to those of ``first`` (a longer training bank).

The result keeps the seed and measurement seed of ``first``; every component
keeps its own stream address, so its measurements do not change.
"""
    for bank in (first, second):
        validate_bank(bank)
    for key in ("laws", "split", "sensor_level"):
        if first[key] != second[key]:
            raise ValueError(f"banks differ in {key}")
    body = {key: value for key, value in first.items() if key not in ("sha256", "pairs", "segments")}
    body.update(pairs=[*first["pairs"], *second["pairs"]], pair_count=first["pair_count"] + second["pair_count"],
                segments=[*first.get("segments", [{"seed": first["seed"], "pairs": first["pair_count"],
                                                    "sha256": first["sha256"]}]),
                          {"seed": second["seed"], "pairs": second["pair_count"], "sha256": second["sha256"]}])
    addresses = [pair[side]["stream_address"] for pair in body["pairs"] for side in ("active", "successor")]
    if len(addresses) != len(set(addresses)):
        raise ValueError("concatenated banks share components")
    return _rebind(body)


def with_sensor_level(bank: Mapping[str, Any], sensor_level: str) -> dict[str, Any]:
    """The same lives and measurement seed under another sensor level (paired noisy and exact cells)."""
    if sensor_level not in SENSOR_LEVELS:
        raise ValueError(f"sensor level must be one of {sorted(SENSOR_LEVELS)}")
    validate_bank(bank)
    return _rebind({**{key: value for key, value in bank.items() if key != "sha256"}, "sensor_level": sensor_level})


def measurement_streams(stream_address: str, replay_seed: int) -> dict[str, np.random.Generator]:
    """The four measurement streams of one component under one replay seed."""
    replay = _integer(replay_seed, "replay seed")
    return {role: np.random.Generator(np.random.PCG64(int(canonical_hash(
        [STREAM_NAMESPACE, "component_measurements", str(stream_address), replay, role])[:16], 16)))
        for role in MEASUREMENT_ROLES}


def component_measurements(component: Mapping[str, Any], sensor_level: str, replay_seed: int) -> dict[str, np.ndarray]:
    """Measured health, availability and bias of one component, as the replaying environment draws them."""
    return sample_measurements(np.asarray(component["health"], dtype=np.float32), sensor_level,
                               measurement_streams(component["stream_address"], replay_seed))


class ScenarioBankEnv(CoreCycleEnv):
    """Unchanged core physics on a finite scenario bank; one reset installs one pair.

``reset()`` advances to the next pair and ``reset(seed=s)`` rewinds to the first
pair, with ``s`` as the measurement replay seed. The first reset without a seed
uses the bank's measurement seed. ``options={"start_age": n}`` is a diagnostic
start of the next pair (or of the first pair with a seed) at operating age ``n``.
The configuration may set the recovery contract, the preparation time, the
reconstruction route and the economics; the bank owns components, starts and
the sensor level. The bank is read, not copied: do not change it while an
environment uses it.
"""

    def __init__(self, bank: Mapping[str, Any], config: Mapping[str, Any] | None = None,
                 render_mode: str | None = None):
        validate_bank(bank)
        options = dict(config or {})
        if options.get("sensor_level", bank["sensor_level"]) != bank["sensor_level"]:
            raise ValueError("the sensor level differs from the scenario bank's")
        owned = {"source", "family", "eol_operating_age", "training_start_offsets"} & set(options)
        if owned:
            raise ValueError(f"the scenario bank owns {sorted(owned)}")
        super().__init__({**options, "sensor_level": bank["sensor_level"]}, render_mode=render_mode)
        self.bank = bank
        self._training_bank = bank
        self._bank_spec = None
        self._measurement_replay_seed: int | None = None

    @property
    def bank_cursor(self) -> int:
        """Number of pairs consumed since the last rewind."""
        return self._bank_cursor

    @property
    def remaining_pair_count(self) -> int:
        return self.bank["pair_count"] - self._bank_cursor

    def configure_training(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("a scenario bank environment replays its own bank")

    def _seed(self, seed: int | None) -> None:
        if seed is None:
            raise ValueError("a scenario bank needs an explicit measurement replay seed")
        super()._seed(seed)
        self._measurement_replay_seed = int(seed)

    def _bank_draw(self, row: Mapping[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(dict(row))
        paths = component_measurements(row, self.config["sensor_level"], self._measurement_replay_seed)
        result.update({name: value.tolist() for name, value in paths.items()})
        result["draw_index"] = self._source_draws
        self._source_draws += 1
        return result

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        requested = dict(options or {})
        if set(requested) - {"start_age"}:
            raise ValueError("scenario bank resets only support a diagnostic start_age")
        effective_seed = self.bank["measurement_seed"] if seed is None and not self._rngs else seed
        if "start_age" not in requested:
            return super().reset(seed=effective_seed)
        age = _integer(requested["start_age"], "start_age")
        index = 0 if effective_seed is not None else self._bank_cursor
        if index >= self.bank["pair_count"]:
            raise RuntimeError("finite core training bank exhausted; wrapping is forbidden")
        if age >= self.bank["pairs"][index]["active"]["eol_operating_age"]:
            raise ValueError("start_age must precede the component endpoint")
        # A temporary view gives the kernel its start age; the bank itself is unchanged.
        view = {**self.bank, "pairs": list(self.bank["pairs"])}
        view["pairs"][index] = {**view["pairs"][index], "start_age": age}
        self._training_bank = view
        try:
            return super().reset(seed=effective_seed)
        finally:
            self._training_bank = self.bank

    def diagnostics(self) -> dict[str, Any]:
        return {**super().diagnostics(), "bank_sha256": self.bank["sha256"], "bank_laws": self.bank["laws"],
                "bank_split": self.bank["split"], "pair_index": self.bank["pairs"][self._bank_cursor - 1]["pair_index"]}

    def snapshot(self) -> dict[str, Any]:
        return {"schema_version": SNAPSHOT_SCHEMA, "bank_sha256": self.bank["sha256"], "core": super().snapshot()}

    def restore(self, state: Mapping[str, Any]) -> np.ndarray:
        if state.get("schema_version") != SNAPSHOT_SCHEMA or state.get("bank_sha256") != self.bank["sha256"]:
            raise ValueError("the snapshot belongs to another scenario bank")
        body = state["core"]["state"]
        if body.get("_bank_spec") is not None or not 0 <= body["_bank_cursor"] <= self.bank["pair_count"]:
            raise ValueError("invalid scenario bank continuation state")
        observation = super().restore(state["core"])
        self._training_bank = self.bank
        return observation


__all__ = ["LAW_SETS", "SPLITS", "STREAM_NAMESPACE", "ScenarioBankEnv", "build_bank", "canonical_hash",
           "component_measurements", "concatenate_banks", "measurement_streams", "stream_seed", "subset_bank",
           "validate_bank", "with_sensor_level"]
