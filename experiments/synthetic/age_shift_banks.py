"""Lifetime twins, the operating-age input and the bootstrap of the age-shift study.

A lifetime twin keeps every component of a bank, its shape parameters, its
lifetime quantile and its stream address, and scales the continuous lifetime
before rounding. The damage path over relative age is therefore unchanged. A
wide lifetime keeps the quantile on a lognormal with a larger scale. Very long
twins can round two successive float32 health values to the same number at
installation. Such a value is lowered by one float32 step, so health stays
strictly decreasing.

The age input is ``min(t / 160, 4)`` at operating age ``t``. The age policy is
the core PPO recipe with a fifth input. It starts from exactly the four-input
network of its seed with a zero age column, so before the first update it acts
like the policy without age.

A scaled-noise series keeps the measurement streams of a noisy bank and only
changes the standard deviation of its white noise. Twins with the noise divided
by their lifetime factor keep the noise per damage increment of the source lives.

The intervals resample the training runs and the lives together, within each
law, and use the same draw for every method and cell of one draw group.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

import common as c
from maintenance_arena.banks import (SCHEMA, SPLITS, STREAM_DOMAINS, STREAM_NAMESPACE, ScenarioBankEnv, canonical_hash,
                                     stream_seed)
from maintenance_arena.curves import (DCRIT, FAMILY_IDS, materialize_health, normalized_damage_fraction,
                                      sample_target_shape, shape_parameters, truncated_lognormal_inverse)
from maintenance_arena import replay as rp

SOURCE_SCALE = .12
AGE_SCALE, AGE_CLIP = 160., 4.
METRICS = ("failures", "unused_life", "retained_profit")


# --- Lifetime twins ---

def continuous_lifetime(probability: float, lifetime: Mapping[str, Any]) -> float:
    """Continuous lifetime at a quantile: the source draw, ``factor`` times it, or a wider lognormal."""
    base = truncated_lognormal_inverse(probability)
    if lifetime["kind"] == "source":
        return base
    if lifetime["kind"] == "scale":
        return float(lifetime["factor"]) * base
    if lifetime["kind"] == "wide":
        return 128. * (base / 128.) ** (float(lifetime["scale"]) / SOURCE_SCALE)
    raise ValueError(f"unknown lifetime kind {lifetime['kind']!r}")


def strictly_decreasing(health: np.ndarray) -> tuple[np.ndarray, int]:
    """Lower a repeated float32 health value by one float32 step and count the changed steps."""
    values = np.array(health, dtype=np.float32)
    changed = 0
    for step in range(1, len(values)):
        if values[step] >= values[step - 1]:
            values[step] = np.nextafter(values[step - 1], np.float32(0.))
            changed += 1
    return values, changed


def _stream(law_set: str, split: str, seed: int, role: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stream_seed(law_set, split, seed, role)))


def _component(law_set: str, split: str, seed: int, index: int, side: str, family: str,
               lifetime: Mapping[str, Any]) -> dict[str, Any]:
    address = f"{side}:{index}"
    lifetime_rng = _stream(law_set, split, seed, f"{address}:lifetime")
    probability = float(np.clip(lifetime_rng.random(), np.nextafter(0., 1.), np.nextafter(1., 0.)))
    continuous_life = continuous_lifetime(probability, lifetime)
    eol = int(math.floor(continuous_life + .5))
    shape_rng = _stream(law_set, split, seed, f"{address}:shape")
    if law_set == "source":
        family_id = family
        parameters = shape_parameters(family_id, float(shape_rng.random()))
    else:
        family_id = law_set
        parameters = sample_target_shape(family_id, shape_rng)
    changed = 0
    if lifetime["kind"] == "source":
        health = materialize_health(family_id, eol, parameters)
    else:
        damage = normalized_damage_fraction(family_id, np.arange(eol + 1, dtype=np.float64) / float(eol), parameters)
        health = np.exp(-DCRIT * damage).astype(np.float32)
        health[0], health[-1] = np.float32(1.), np.float32(.2)
        health, changed = strictly_decreasing(health)
    component = {"family_id": family_id, "eol_operating_age": eol, "t_cont": continuous_life,
                 "shape_parameters": [[name, value] for name, value in parameters], "health": health.tolist(),
                 "stream_address": f"{STREAM_NAMESPACE}/{STREAM_DOMAINS[law_set]}/{split}/{seed}/{address}"}
    if changed:
        component["float32_steps_lowered"] = changed
    return component


def lifetime_bank(law_set: str, *, split: str, seed: int, pair_count: int, lifetime: Mapping[str, Any],
                  sensor_level: str = "exact") -> dict[str, Any]:
    """``maintenance_arena.banks.build_bank`` with another lifetime distribution.

With the source lifetime the bank equals ``build_bank`` bit for bit. Otherwise
the bank records its lifetime and keeps the components' stream addresses, so a
twin receives the same measurement streams as its source bank.
"""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    families = {}
    for side in ("active", "successor"):
        order = [FAMILY_IDS[i % len(FAMILY_IDS)] for i in range(pair_count)]
        families[side] = list(_stream(law_set, split, seed, f"{side}:family_order").permutation(order))
    pairs = []
    for index in range(pair_count):
        pair: dict[str, Any] = {"pair_index": index, "start_age": 0}
        for side in ("active", "successor"):
            pair[side] = _component(law_set, split, seed, index, side, str(families[side][index]), lifetime)
        pairs.append(pair)
    bank = {"schema_version": SCHEMA, "laws": law_set, "split": split, "seed": int(seed), "pair_count": int(pair_count),
            "sensor_level": sensor_level, "measurement_seed": stream_seed(law_set, split, seed, "measurements"),
            "pairs": pairs}
    if lifetime["kind"] != "source":
        bank["lifetime"] = dict(lifetime)
    return {**bank, "sha256": canonical_hash(bank)}


def bank_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """A bank from a resolved specification with an optional ``lifetime``."""
    return lifetime_bank(spec["laws"], split=spec["split"], seed=spec["seed"], pair_count=spec["pairs"],
                         lifetime=spec.get("lifetime", {"kind": "source"}), sensor_level=spec.get("sensor_level", "exact"))


# --- Noise scaled with the lifetime ---

def scaled_series(bank: Mapping[str, Any], *, observation_route: str, sigma: float,
                  mobilization_lead: int = 12) -> rp.DecisionSeries:
    """``replay.decision_series`` of a white-noise bank with the noise standard deviation ``sigma``.

The measurement streams and their draws are those of the bank, so only the size
of the white noise changes. With the standard deviation of the bank's sensor
level the series equals ``replay.decision_series`` bit for bit.
"""
    from maintenance_arena.banks import measurement_streams
    from maintenance_arena.observations import SENSOR_LEVELS, reconstruct
    if SENSOR_LEVELS[bank["sensor_level"]][1:] != (0., 0.):
        raise ValueError("scaled noise needs a white-noise sensor level without bias or missing values")
    replay_seed = bank["measurement_seed"]
    values, measured_values, slices, families, lives, pairs = [], [], [], [], [], []
    cursor = 0
    for pair in bank["pairs"]:
        part = pair["active"]
        curve = np.ascontiguousarray(np.asarray(part["health"], dtype=np.float32))
        white = measurement_streams(part["stream_address"], replay_seed)["white"].standard_normal(curve.size)
        bias = np.zeros(curve.size, dtype=np.float64)
        measured = np.clip(curve.astype(np.float64) + bias + float(sigma) * white, 0., 1.).astype(np.float32)
        states = reconstruct(measured, np.ones(curve.size, dtype=bool), observation_route, mobilization_lead)["states"][:-1]
        if states.shape != (part["eol_operating_age"], 4) or not np.isfinite(states).all():
            raise ValueError("malformed causal state sequence")
        values.append(states)
        measured_values.append(measured[:-1])
        slices.append((cursor, cursor + len(states)))
        cursor += len(states)
        families.append(str(part["family_id"]))
        lives.append(int(part["eol_operating_age"]) - 1)
        pairs.append(int(pair["pair_index"]))
    return rp.DecisionSeries(np.vstack(values), tuple(slices), tuple(families), np.asarray(lives, dtype=np.int64),
                             np.asarray(pairs, dtype=np.int64), int(mobilization_lead), observation_route,
                             bank["sensor_level"], np.concatenate(measured_values))


# --- Operating-age input ---

def age_column(ages: np.ndarray, scale: float = AGE_SCALE, clip: float = AGE_CLIP) -> np.ndarray:
    return np.minimum(np.asarray(ages, dtype=np.float64) / scale, clip).astype(np.float32)


def series_ages(series: rp.DecisionSeries) -> np.ndarray:
    """Operating age of every decision state of a decision series."""
    return np.concatenate([np.arange(stop - start) for start, stop in series.slices])


def age_states(series: rp.DecisionSeries, scale: float = AGE_SCALE, clip: float = AGE_CLIP) -> np.ndarray:
    """The four causal states of a series followed by the age input."""
    return np.ascontiguousarray(np.column_stack([series.states, age_column(series_ages(series), scale, clip)]),
                                dtype=np.float32)


class AgeScenarioBankEnv(ScenarioBankEnv):
    """A scenario bank whose observations end with the age input. Events, prices and accounting are unchanged."""

    def __init__(self, bank: Mapping[str, Any], config: Mapping[str, Any] | None = None,
                 render_mode: str | None = None, *, scale: float = AGE_SCALE, clip: float = AGE_CLIP):
        from gymnasium import spaces
        super().__init__(bank, config, render_mode=render_mode)
        self.age_scale, self.age_clip = float(scale), float(clip)
        self.observation_space = spaces.Box(low=np.asarray([0., 0., 0., -1., 0.], dtype=np.float32),
                                            high=np.asarray([1., 1., 1., 1., self.age_clip], dtype=np.float32),
                                            dtype=np.float32)

    def age_observation(self, observation: np.ndarray) -> np.ndarray:
        age = np.float32(min(self._age / self.age_scale, self.age_clip))
        return np.concatenate([np.asarray(observation, dtype=np.float32), np.asarray([age], dtype=np.float32)])

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        return self.age_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return self.age_observation(observation), reward, terminated, truncated, info


def _age_classes():
    import torch
    from torch import nn
    from maintenance_arena.learning.collector import PROFILE, BankTrainer
    from maintenance_arena.learning.ppo import PPOAgent

    low = np.asarray([0., 0., 0., -1., 0.], dtype=np.float32)

    def check(array: np.ndarray) -> None:
        if array.ndim != 2 or array.shape[1] != 5 or not np.isfinite(array).all():
            raise ValueError("expected finite (n, 5) policy inputs")
        if np.any(array < low) or np.any(array[:, :4] > 1.) or np.any(array[:, 4] > AGE_CLIP):
            raise ValueError("a five-coordinate input violates the observation bounds")

    def one(observation: Any) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
        check(array)
        return array

    class AgePPOAgent(PPOAgent):
        """The core PPO recipe with the age input as a fifth column of the first layers."""

        def __init__(self, seed: int, *, role_seeds: Mapping[str, int] | None = None):
            super().__init__(PROFILE, seed, role_seeds=role_seeds)
            for body in (self.network.policy_body, self.network.value_body):
                old = body[0]
                with torch.random.fork_rng(devices=[]):
                    expanded = nn.Linear(5, old.out_features).cpu()
                with torch.no_grad():
                    expanded.weight.zero_()
                    expanded.weight[:, :4].copy_(old.weight)
                    expanded.bias.copy_(old.bias)
                body[0] = expanded
            optimizer = self.config["optimizer"]
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"],
                                              betas=tuple(optimizer["betas"]), eps=optimizer["epsilon"],
                                              weight_decay=optimizer["weight_decay"])

        def batch_output(self, observations):
            array = np.ascontiguousarray(np.asarray(observations, dtype=np.float32))
            check(array)
            with torch.no_grad():
                logits, values = self.network(torch.from_numpy(array))
                probabilities = torch.softmax(logits, dim=-1)
            return tuple(x.detach().cpu().numpy().astype(np.float32, copy=True) for x in (logits, probabilities, values))

        def probabilities(self, observation):
            return self.batch_output(one(observation))[1][0]

        def predict(self, observation):
            return int(np.argmax(self.probabilities(observation)))

        deterministic_action = predict

        def value(self, observation):
            return float(self.batch_output(one(observation))[2][0])

        def sample_action(self, observation):
            _, probabilities, values = self.batch_output(one(observation))
            action = 0 if float(self.action_rng.random()) < float(probabilities[0, 0]) else 1
            log = math.log(max(float(probabilities[0, action]), float(np.finfo(np.float32).tiny)))
            return action, log, float(values[0])

    class AgeBankTrainer(BankTrainer):
        """``BankTrainer`` with the age input. Seeds, bank use and training loop are those of ``BankTrainer``."""

        def __init__(self, bank: Mapping[str, Any], seed: int, *, env_config: Mapping[str, Any] | None = None,
                     scale: float = AGE_SCALE, clip: float = AGE_CLIP):
            self.age_scale, self.age_clip = float(scale), float(clip)
            super().__init__(bank, seed, env_config=env_config)

        def _make_agent(self):
            return AgePPOAgent(self.seed, role_seeds=self.roles)

        def _make_env(self, bank, env_config):
            return AgeScenarioBankEnv(bank, env_config, scale=self.age_scale, clip=self.age_clip)

    return AgePPOAgent, AgeBankTrainer


def age_agent_class():
    return _age_classes()[0]


def age_trainer_class():
    return _age_classes()[1]


def policy(path: str | Path):
    """The agent of a trainer checkpoint, with or without the age input."""
    from maintenance_arena.learning.collector import load_checkpoint, policy_agent, role_seeds, single_thread
    payload = load_checkpoint(path)
    state = payload["agent"]["policy_state"]
    if tuple(state["policy_body.0.weight"].shape) != (64, 5):
        return policy_agent(payload)
    with single_thread():
        agent = age_agent_class()(int(payload["seed"]), role_seeds=role_seeds(int(payload["seed"])))
    agent.network.load_state_dict(state, strict=True)
    return agent


# --- Training ---

def fit(job: Mapping[str, Any]) -> dict[str, Any]:
    """Train one run through all milestones (process-pool job). ``age_input`` selects the age policy."""
    from maintenance_arena.learning.collector import BankTrainer
    started = time.perf_counter()
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    bank = bank_from_spec(job["bank"])
    age = job.get("age_input")
    if age:
        trainer = age_trainer_class()(bank, job["seed"], env_config=job["env_config"], scale=age["scale"],
                                      clip=age["clip"])
    else:
        trainer = BankTrainer(bank, job["seed"], env_config=job["env_config"])
    del bank
    checkpoints = []
    try:
        for milestone in job["milestones"]:
            progress = trainer.train_until(int(milestone))
            checkpoints.append({**trainer.save(directory / f"checkpoint_{milestone}.pt"), **progress})
    finally:
        trainer.close()
    record = {"label": job["label"], "seed": job["seed"], "bank": {**job["bank"], "sha256": trainer.bank_sha256},
              "env_config": job["env_config"], "age_input": age, "checkpoints": checkpoints,
              "elapsed_seconds": round(time.perf_counter() - started, 1)}
    c.write_json(directory / "training.json", {**record, "training_log": trainer.logs})
    return record


# --- Intervals over runs and lives ---

def stratum_key(law: str) -> str:
    """Strata are drawn in the order of the laws' stream names."""
    return STREAM_DOMAINS.get(law, law)


def life_arrays(rows: Sequence[Mapping[str, Any]], order: Sequence[tuple[Any, ...]], economics: rp.Economics
                ) -> dict[str, np.ndarray]:
    """Lives by runs arrays of one method and cell, rebuilt from the first orders.

``order`` lists the life keys ``(law, unit)`` in the cell's draw order. Runs are
sorted by their training seed.
"""
    position = {key: i for i, key in enumerate(order)}
    runs = sorted({str(row["run"]) for row in rows}, key=lambda run: (len(run), run))
    column = {run: j for j, run in enumerate(runs)}
    last = np.zeros(len(order), dtype=np.int64)
    first = np.full((len(order), len(runs)), -1, dtype=np.int64)
    filled = np.zeros(first.shape, dtype=bool)
    for row in rows:
        i, j = position[(str(row["group"]).split("/", 1)[1], int(row["unit"]))], column[str(row["run"])]
        last[i] = int(row["last_productive_step"])
        first[i, j] = -1 if row["order_step"] in ("", None) else int(row["order_step"])
        filled[i, j] = True
    if not filled.all():
        raise ValueError("every life needs one row per run")
    outcome = rp.replay(np.repeat(last[:, None], len(runs), axis=1).ravel(), first.ravel(), economics)
    shape = first.shape
    reference = rp.replay(last, rp.reference_orders(last, economics.mobilization_lead), economics)
    return {"failure": outcome["failed"].reshape(shape).astype(np.float64),
            "unused": outcome["unused_fraction"].reshape(shape).astype(np.float64),
            "cash": outcome["cash"].reshape(shape), "calendar": outcome["calendar"].reshape(shape),
            "reference_rate": reference["cash"] / reference["calendar"], "first": first, "last": last,
            "runs": runs}


def _weighted(values: np.ndarray, runs: np.ndarray | None, lives: np.ndarray) -> np.ndarray:
    if runs is None or values.shape[1] == 1:
        per_life = values.mean(axis=1)[None, :]
    else:
        per_life = (runs @ values.T) / runs.sum(axis=1, keepdims=True)
    return (lives * per_life).sum(axis=1) / lives.sum(axis=1)


def metric_values(arrays: Mapping[str, np.ndarray], runs: np.ndarray | None, lives: np.ndarray) -> dict[str, np.ndarray]:
    """Failure share, unused life and retained profit in percent under run and life weights."""
    if runs is None or arrays["cash"].shape[1] == 1:
        rate = (arrays["cash"].mean(axis=1) / arrays["calendar"].mean(axis=1))[None, :]
    else:
        rate = (runs @ arrays["cash"].T) / (runs @ arrays["calendar"].T)
    rate = rate / arrays["reference_rate"][None, :]
    return {"failures": 100. * _weighted(arrays["failure"], runs, lives),
            "unused_life": 100. * _weighted(arrays["unused"], runs, lives),
            "retained_profit": 100. * (lives * rate).sum(axis=1) / lives.sum(axis=1)}


def runs_and_lives(tables: Mapping[tuple[str, str], Mapping[str, np.ndarray]], groups: Sequence[Mapping[str, Any]],
                   runs: int, *, replicates: int, seed: int, chunk: int) -> dict[tuple[str, str, str], np.ndarray]:
    """Bootstrap draws of every metric for every (cell, method) table.

Each chunk first draws the run weights, shared by all methods and cells. Then
each group in turn draws its life weights within every law, shared by its cells.
"""
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        run_weights = rng.multinomial(runs, np.full(runs, 1. / runs), size=count).astype(float)
        for group in groups:
            laws = np.asarray(group["laws"])
            life_weights = np.zeros((count, len(laws)))
            for law in sorted(set(laws.tolist()), key=stratum_key):
                index = np.flatnonzero(laws == law)
                life_weights[:, index] = rng.multinomial(len(index), np.full(len(index), 1. / len(index)), size=count)
            for (cell, method), arrays in tables.items():
                if cell in group["cells"]:
                    for metric, values in metric_values(arrays, run_weights, life_weights).items():
                        samples[(cell, method, metric)].append(values)
    return {key: np.concatenate(values) for key, values in samples.items()}


def point_values(tables: Mapping[tuple[str, str], Mapping[str, np.ndarray]]) -> dict[tuple[str, str, str], float]:
    result = {}
    for (cell, method), arrays in tables.items():
        for metric, values in metric_values(arrays, None, np.ones((1, len(arrays["last"])))).items():
            result[(cell, method, metric)] = float(values[0])
    return result


def contrast(samples, points, terms: Sequence[tuple[int, str, str]], metric: str) -> dict[str, Any]:
    """A signed sum of (cell, method) values with its percentile 95% interval."""
    values = sum(sign * samples[(cell, method, metric)] for sign, cell, method in terms)
    estimate = sum(sign * points[(cell, method, metric)] for sign, cell, method in terms)
    return {"estimate": float(estimate),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def gap(samples, points, first: tuple[tuple[str, str], tuple[str, str]], second: tuple[tuple[str, str], tuple[str, str]],
        metric: str) -> dict[str, Any]:
    """The difference of two (cell, method) differences, ``(a - b) - (c - d)``, with its 95% interval."""
    (a, b), (c_, d) = first, second
    values = ((samples[(*a, metric)] - samples[(*b, metric)]) - (samples[(*c_, metric)] - samples[(*d, metric)]))
    estimate = (points[(*a, metric)] - points[(*b, metric)]) - (points[(*c_, metric)] - points[(*d, metric)])
    return {"estimate": float(estimate),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def deadline_following(first_a: np.ndarray, last_a: np.ndarray, first_b: np.ndarray, last_b: np.ndarray) -> dict[str, Any]:
    """Share of the deadline shift that the first order follows, over twin lives with orders in both cells."""
    finite = (first_a >= 0) & (first_b >= 0)
    shift = ((last_b - 12) - (last_a - 12))[:, None]
    fraction = ((first_b - first_a) / shift)[finite]
    if not fraction.size:
        return {"pairs": 0, "missing": int((~finite).sum()), "median": None, "quartiles": None}
    return {"pairs": int(finite.sum()), "missing": int((~finite).sum()), "median": float(np.median(fraction)),
            "quartiles": np.percentile(fraction, [25, 75]).tolist(), "share_below_half": float(np.mean(fraction < .5))}


__all__ = ["AGE_CLIP", "AGE_SCALE", "AgeScenarioBankEnv", "age_agent_class", "age_column", "age_states",
           "age_trainer_class", "bank_from_spec", "contrast", "continuous_lifetime", "deadline_following", "fit", "gap",
           "life_arrays", "lifetime_bank", "metric_values", "point_values", "policy", "runs_and_lives",
           "scaled_series", "series_ages", "strictly_decreasing"]
