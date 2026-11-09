"""Core cycle task: delayed WAIT/REPLACE decisions through one completed renewal.

The simulator physically installs a successor component at the exchange. The
learning task ends after the six downtime steps that follow; this episodic
objective is explicit.

The preparation time ``mobilization_lead`` (the article's ``L``) is 12 by
default and may be 6 or 24. It sets the number of preparation steps after an
order, the scaling of the rate channel and the deadline coordinate; the last
safe order lies ``L + 1`` steps before failure.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .curves import FAMILY_IDS, LAW_IDS, TARGET_FAMILY_IDS, sample_curve, build_training_bank
from .observations import Reconstructor, ROUTE_IDS, SENSOR_LEVELS, sample_measurements

WAIT, REPLACE = 0, 1
RECOVERY_CONTRACTS = ("immediate_exchange_v1", "preserve_preparation_v1")
# Supported preparation times. 12 is the published task; 6 and 24 are the
# preparation-time study of the accompanying article.
MOBILIZATION_LEADS = (6, 12, 24)
DEFAULT_CONFIG = {
    "source": "continuous", "family": "random", "eol_operating_age": None,
    "observation_route": "RAW_CAUSAL", "sensor_level": "exact",
    "gamma": .999, "reward_scale": 1000., "mobilization_lead": 12,
    "downtime_steps": 6, "revenue_per_operating_step": 255.,
    "replacement_cost": 520., "downtime_cost_per_step": 25.,
    "failure_event_cost": 7000., "calendar_horizon": None,
    "training_start_offsets": None,
}


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = {} if config is None else dict(config)
    recovery = supplied.pop("recovery_contract", "immediate_exchange_v1")
    if recovery not in RECOVERY_CONTRACTS:
        raise ValueError(f"recovery_contract must be one of {RECOVERY_CONTRACTS}")
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown core configuration keys: {sorted(unknown)}")
    result = copy.deepcopy(DEFAULT_CONFIG)
    result.update(copy.deepcopy(supplied))
    # Keep the published default configuration/checkpoint identity unchanged.
    if recovery != "immediate_exchange_v1":
        result["recovery_contract"] = recovery
    if result["source"] not in ("continuous", "discrete"):
        raise ValueError("source must be continuous or discrete")
    if result["family"] not in (*LAW_IDS, "random"):
        raise ValueError("unknown core degradation family")
    if result["source"] == "discrete" and result["family"] in TARGET_FAMILY_IDS:
        raise ValueError("the discrete source covers the four source laws only")
    if result["observation_route"] not in ROUTE_IDS or result["sensor_level"] not in SENSOR_LEVELS:
        raise ValueError("unknown sensor preset or reconstruction route")
    if result["eol_operating_age"] is not None:
        result["eol_operating_age"] = _integer(result["eol_operating_age"], "eol_operating_age", minimum=2)
    if _integer(result["mobilization_lead"], "mobilization_lead") not in MOBILIZATION_LEADS:
        raise ValueError(f"the core task supports mobilization_lead in {MOBILIZATION_LEADS}")
    if _integer(result["downtime_steps"], "downtime_steps") != 6:
        raise ValueError("the published core observation/task contract requires downtime_steps=6")
    for key in ("gamma", "reward_scale", "revenue_per_operating_step", "replacement_cost", "downtime_cost_per_step", "failure_event_cost"):
        value = float(result[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        result[key] = value
    if not 0 < result["gamma"] <= 1 or result["reward_scale"] <= 0:
        raise ValueError("gamma must lie in (0,1] and reward_scale must be positive")
    if result["calendar_horizon"] is not None:
        result["calendar_horizon"] = _integer(result["calendar_horizon"], "calendar_horizon", minimum=1)
    if result["training_start_offsets"] is not None:
        offsets = list(result["training_start_offsets"])
        if not offsets:
            raise ValueError("training_start_offsets cannot be empty")
        for offset in offsets:
            if offset is not None:
                _integer(offset, "training start RUL offset", minimum=1)
        result["training_start_offsets"] = offsets
    return result


class CoreCycleEnv(gym.Env[np.ndarray, int]):
    """One complete component cycle with duration-aware decision transitions.

Policy observations and public info exclude hidden lifetime, family and true
RUL. ``diagnostics`` is a separate evaluator interface. ``snapshot`` contains
simulator truth for resumption and must never be passed to a policy.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    observation_names = ("current_health", "lead_scaled_local_damage_rate",
                         "rate_valid", "causal_intervention_slack")

    def __init__(self, config: Mapping[str, Any] | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = validate_config(config)
        if render_mode not in (None, "ansi"):
            raise ValueError("core render_mode must be None or ansi")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.asarray([0., 0., 0., -1.], dtype=np.float32),
            high=np.ones(4, dtype=np.float32), dtype=np.float32)
        self._rngs: dict[str, np.random.Generator] = {}
        self._needs_reset = True
        self._initialized = False
        self._reset_count = 0
        self._total_calendar = 0
        self._source_draws = 0
        self._trace: list[dict[str, Any]] = []
        self._training_bank: dict[str, Any] | None = None
        self._bank_spec: dict[str, Any] | None = None
        self._bank_cursor = 0

    def configure_training(self, *, seed: int, profile: str | None = None,
                           pair_count: int | None = None) -> dict[str, Any]:
        """Attach the finite published curriculum under a fresh public seed.

Call before the first reset. This source contract is separate from env.config,
so native policy checkpoints also load into ordinary age-zero evaluation envs.
        """
        if self._initialized:
            raise RuntimeError("configure the training bank before the first reset")
        if self.config["family"] != "random" or self.config["eol_operating_age"] is not None:
            raise ValueError("the published training bank requires all families and lifetime cells")
        if self.config["training_start_offsets"] is not None:
            raise ValueError("the finite bank owns its start curriculum")
        if self.config["mobilization_lead"] != 12:
            raise ValueError("the published training curriculum is defined for mobilization_lead=12; "
                             "train other preparation times on a scenario bank (maintenance_arena.learning.collector)")
        source = self.config["source"]
        expected_profile = f"paper-core-{source}-v1"
        if profile is not None and profile != expected_profile:
            raise ValueError("training profile differs from the configured source")
        self._training_bank = build_training_bank(seed, source=source, pair_count=pair_count)
        self._bank_spec = {name: self._training_bank[name] for name in ("source", "seed", "pair_count", "sha256")}
        self._bank_cursor = 0
        return copy.deepcopy(self._bank_spec)

    def _seed(self, seed: int | None) -> None:
        sequence = np.random.SeedSequence(seed)
        names = ("curve", "white", "initial", "bias", "missing")
        children = sequence.spawn(len(names) + 1)
        self._rngs = {name: np.random.Generator(np.random.PCG64(child)) for name, child in zip(names, children)}
        self.action_space.seed(int(children[-1].generate_state(1)[0]))
        self._reset_count = self._total_calendar = self._source_draws = 0
        self._bank_cursor = 0

    def _draw(self) -> dict[str, Any]:
        result = sample_curve(self._rngs["curve"], source=self.config["source"],
                              family=self.config["family"], eol_operating_age=self.config["eol_operating_age"])
        paths = sample_measurements(np.asarray(result["health"], dtype=np.float32),
                                     self.config["sensor_level"], self._rngs)
        result.update({name: value.tolist() for name, value in paths.items()})
        result["draw_index"] = self._source_draws
        self._source_draws += 1
        return result

    def _bank_draw(self, row: Mapping[str, Any]) -> dict[str, Any]:
        result = {key: copy.deepcopy(value) for key, value in row.items() if key != "health"}
        health = np.asarray(row["health"], dtype=np.float32)
        result["health"] = health.tolist()
        paths = sample_measurements(health, self.config["sensor_level"], self._rngs)
        result.update({name: value.tolist() for name, value in paths.items()})
        result["draw_index"] = self._source_draws
        self._source_draws += 1
        return result

    def _observe_current(self) -> None:
        available = self._active["available"][self._age]
        value = float(self._active["measured_health"][self._age]) if available else None
        self._observation = self._reconstructor.step(value)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        requested = dict(options or {})
        if set(requested) - {"start_age"}:
            raise ValueError("core reset options only support an observed start_age")
        if self._training_bank is not None and requested:
            raise ValueError("finite-bank reset cannot alter its materialized curriculum")
        if "start_age" in requested:
            _integer(requested["start_age"], "start_age")
        # A requested age can exceed the sampled lifetime. Preserve the active
        # episode and its streams if that diagnostic start cannot be constructed.
        previous = self.__dict__.copy()
        rng_states = {name: copy.deepcopy(rng.bit_generator.state) for name, rng in self._rngs.items()}
        action_rng = self.action_space._np_random
        try:
            return self._reset_episode(seed=seed, requested=requested)
        except (ValueError, RuntimeError):
            self.__dict__.clear()
            self.__dict__.update(previous)
            for name, rng_state in rng_states.items():
                self._rngs[name].bit_generator.state = rng_state
            self.action_space._np_random = action_rng
            raise

    def _reset_episode(self, *, seed: int | None, requested: Mapping[str, Any]):
        super().reset(seed=seed)
        if seed is not None or not self._rngs:
            self._seed(seed)
        bank_start = None
        if self._training_bank is not None:
            if self._bank_cursor >= self._training_bank["pair_count"]:
                raise RuntimeError("finite core training bank exhausted; wrapping is forbidden")
            pair = self._training_bank["pairs"][self._bank_cursor]
            self._active, self._successor = self._bank_draw(pair["active"]), self._bank_draw(pair["successor"])
            bank_start = int(pair["start_age"])
            self._bank_cursor += 1
        else:
            self._active, self._successor = self._draw(), self._draw()
        self._initial = copy.deepcopy(self._active)
        offsets = self.config["training_start_offsets"]
        if bank_start is not None:
            start_age = bank_start
        elif "start_age" in requested:
            start_age = _integer(requested["start_age"], "start_age")
        elif offsets is not None:
            offset = offsets[self._reset_count % len(offsets)]
            start_age = 0 if offset is None else self._active["eol_operating_age"] - offset
        else:
            start_age = 0
        if not 0 <= start_age < self._active["eol_operating_age"]:
            raise ValueError("start_age must precede the sampled component endpoint")
        self._reset_count += 1
        self._reconstructor = Reconstructor(self.config["observation_route"], self.config["mobilization_lead"])
        for self._age in range(start_age + 1):
            self._observe_current()
        self._start_age = start_age
        self._calendar = self._decisions = 0
        self._raw_profit = self._discounted_return = 0.
        self._failures = self._planned = self._corrective = 0
        self._discarded_steps = 0
        self._first_order_age: int | None = None
        self._mobilization_remaining = 0
        self._order_pending = False
        self._corrective_pending = False
        self._downtime_remaining = 0
        self._renewed = False
        self._trace = []
        self._needs_reset = False
        self._initialized = True
        return self._observation.copy(), self._info([], [])

    def _exchange(self, failure: bool) -> None:
        self._discarded_steps = 0 if failure else max(0, self._active["eol_operating_age"] - self._age)
        self._failures += int(failure and not self._corrective_pending)
        self._corrective += int(failure)
        self._planned += int(not failure)
        self._active = self._successor
        self._age = 0
        self._reconstructor = Reconstructor(self.config["observation_route"], self.config["mobilization_lead"])
        self._observe_current()
        self._order_pending = False
        self._corrective_pending = False
        self._mobilization_remaining = 0
        self._downtime_remaining = self.config["downtime_steps"]
        self._renewed = True

    def _primitive(self, action: int) -> dict[str, Any]:
        """One calendar transition, preserving the source event ordering."""
        old_age, old_health = self._age, float(self._active["health"][self._age])
        old_draw = self._active["draw_index"]
        event = "operate"
        failure = planned = corrective = order = False
        revenue = replacement = downtime = failure_cost = 0.
        if self._downtime_remaining > 0:
            self._downtime_remaining -= 1
            downtime = self.config["downtime_cost_per_step"]
            event = "downtime"
        elif self._order_pending and self._mobilization_remaining == 0:
            corrective = self._corrective_pending
            planned = not corrective
            replacement, downtime = self.config["replacement_cost"], self.config["downtime_cost_per_step"]
            event = "corrective_exchange" if corrective else "planned_exchange"
            self._exchange(corrective)
        elif self._corrective_pending:
            self._mobilization_remaining -= 1
            downtime = self.config["downtime_cost_per_step"]
            event = "corrective_mobilize"
        else:
            if not self._order_pending and action == REPLACE:
                self._order_pending = True
                self._mobilization_remaining = self.config["mobilization_lead"]
                order = True
                if self._first_order_age is None:
                    self._first_order_age = self._age
            if self._age + 1 >= self._active["eol_operating_age"]:
                failure = True
                downtime, failure_cost = self.config["downtime_cost_per_step"], self.config["failure_event_cost"]
                if self.config.get("recovery_contract") == "preserve_preparation_v1":
                    event = "failure_during_mobilization" if self._order_pending else "failure"
                    if not self._order_pending:
                        self._order_pending = True
                        self._mobilization_remaining = self.config["mobilization_lead"]
                    self._corrective_pending = True
                    self._failures += 1
                    self._age += 1
                    self._observe_current()
                    # Detection consumes the first remaining preparation step.
                    self._mobilization_remaining -= 1
                else:
                    event = "failure_during_mobilization" if self._order_pending else "corrective_exchange"
                    corrective = True
                    replacement = self.config["replacement_cost"]
                    self._exchange(True)
            else:
                self._age += 1
                self._observe_current()
                revenue = self.config["revenue_per_operating_step"]
                if self._order_pending:
                    self._mobilization_remaining -= 1
                    event = "mobilize"
        profit = revenue - replacement - downtime - failure_cost
        self._calendar += 1
        self._total_calendar += 1
        row = {"calendar_step": self._calendar, "event": event, "order_placed": order,
               "revenue": revenue, "replacement_cost": replacement, "downtime_cost": downtime,
               "failure_cost": failure_cost, "raw_profit": profit,
               "planned_replacement": planned, "corrective_replacement": corrective,
               "failure": failure}
        self._trace.append({**row, "operating_age_before": old_age, "operating_age_after": self._age,
                            "true_health_before": old_health,
                            "true_health_after": float(self._active["health"][self._age]),
                            "component_before": old_draw, "component_after": self._active["draw_index"],
                            "observation_after": self._observation.tolist()})
        return row

    def _info(self, ledger: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
        duration = len(ledger)
        return {"schema_version": "maintenance_arena_core_cycle_v1", "objective": "one_cycle",
                "duration_calendar_steps": duration, "gamma_pow_duration": self.config["gamma"] ** duration,
                "raw_profit_sum": float(math.fsum(row["raw_profit"] for row in ledger)),
                "events": copy.deepcopy(events), "ledger": copy.deepcopy(ledger),
                "calendar_steps": self._calendar, "decision_steps": self._decisions,
                "episode_raw_profit": self._raw_profit, "episode_discounted_return": self._discounted_return,
                "failures": self._failures, "planned_replacements": self._planned,
                "corrective_replacements": self._corrective,
                "measurement_gap": self._reconstructor.measurement_gap,
                "observation_valid": bool(self._observation[2] > .5),
                "calendar_budget": self.config["calendar_horizon"],
                "calendar_overshoot": max(0, self._calendar - (self.config["calendar_horizon"] or self._calendar))}

    def step(self, action: int):
        if self._needs_reset:
            raise RuntimeError("core environment requires reset before step")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"invalid core action {action!r}")
        start = self._calendar
        ledger = [self._primitive(int(action))]
        while self._order_pending or self._downtime_remaining > 0:
            ledger.append(self._primitive(WAIT))
        duration = len(ledger)
        reward = math.fsum(self.config["gamma"] ** i * (row["raw_profit"] / self.config["reward_scale"])
                           for i, row in enumerate(ledger))
        self._decisions += 1
        self._raw_profit += math.fsum(row["raw_profit"] for row in ledger)
        self._discounted_return += self.config["gamma"] ** start * reward
        terminated = bool(self._renewed)
        horizon = self.config["calendar_horizon"]
        truncated = bool(not terminated and horizon is not None and self._calendar >= horizon)
        self._needs_reset = terminated or truncated
        events = [{"calendar_step": row["calendar_step"], "event": row["event"],
                   "order_placed": row["order_placed"]} for row in ledger]
        return self._observation.copy(), float(reward), terminated, truncated, self._info(ledger, events)

    def policy_state(self) -> dict[str, Any]:
        """Causally counted operating age for the explicitly age-informed rule."""
        if not self._initialized:
            raise RuntimeError("core environment has not been reset")
        return {"operating_age": self._age, "measurement_gap": self._reconstructor.measurement_gap}

    def diagnostics(self) -> dict[str, Any]:
        """Offline evaluator sidecar; never include this mapping in learner info."""
        if not self._initialized:
            raise RuntimeError("core environment has not been reset")
        eol = self._initial["eol_operating_age"]
        return {"family_id": self._initial["family_id"], "source_draw_index": self._initial["draw_index"],
                "eol_operating_age": eol, "start_age": self._start_age,
                "first_trigger_age": self._first_order_age,
                "first_trigger_true_rul": None if self._first_order_age is None else eol - self._first_order_age,
                "discarded_operating_steps": self._discarded_steps,
                "normalized_discarded_life": self._discarded_steps / eol,
                "calendar_steps": self._calendar, "raw_profit": self._raw_profit,
                "failures": self._failures, "planned_replacements": self._planned,
                "corrective_replacements": self._corrective, "trace": copy.deepcopy(self._trace)}

    def snapshot(self) -> dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("reset the environment before taking a snapshot")
        excluded = {"_rngs", "_reconstructor", "_np_random", "_np_random_seed", "_training_bank",
                    "action_space", "observation_space", "config", "render_mode"}
        state = {key: copy.deepcopy(value) for key, value in self.__dict__.items()
                 if key.startswith("_") and key not in excluded}
        state["_observation"] = self._observation.tolist()
        return {"schema_version": "maintenance_arena_core_snapshot_v1", "config": copy.deepcopy(self.config),
                "state": state, "reconstructor": self._reconstructor.snapshot(),
                "rngs": {key: copy.deepcopy(rng.bit_generator.state) for key, rng in self._rngs.items()},
                "gym_rng": copy.deepcopy(self.np_random.bit_generator.state),
                "action_rng": copy.deepcopy(self.action_space.np_random.bit_generator.state)}

    def restore(self, state: Mapping[str, Any]) -> np.ndarray:
        if state.get("schema_version") != "maintenance_arena_core_snapshot_v1" or state.get("config") != self.config:
            raise ValueError("core snapshot version or configuration differs")
        body = copy.deepcopy(dict(state["state"]))
        body.setdefault("_corrective_pending", False)
        for key in body:
            if not key.startswith("_") or key in {"_rngs", "_reconstructor", "_np_random"}:
                raise ValueError("invalid core snapshot state field")
        body["_observation"] = np.asarray(body["_observation"], dtype=np.float32)
        if not self.observation_space.contains(body["_observation"]):
            raise ValueError("snapshot observation is outside the core space")
        self.__dict__.update(body)
        self._training_bank = None
        if self._bank_spec is not None:
            spec = self._bank_spec
            self._training_bank = build_training_bank(spec["seed"], source=spec["source"], pair_count=spec["pair_count"])
            if self._training_bank["sha256"] != spec["sha256"]:
                raise ValueError("training bank could not be reproduced from the snapshot contract")
        self._rngs = {}
        for name, rng_state in state["rngs"].items():
            self._rngs[name] = np.random.default_rng()
            self._rngs[name].bit_generator.state = copy.deepcopy(rng_state)
        self._np_random = np.random.default_rng()
        self._np_random.bit_generator.state = copy.deepcopy(state["gym_rng"])
        self.action_space.np_random.bit_generator.state = copy.deepcopy(state["action_rng"])
        self._reconstructor = Reconstructor(self.config["observation_route"], self.config["mobilization_lead"])
        self._reconstructor.restore(state["reconstructor"])
        return self._observation.copy()

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        if not self._initialized:
            return "Core cycle: reset required"
        return f"calendar={self._calendar} observation={self._observation.tolist()} profit={self._raw_profit:.2f}"


def policy_action(name: str, env: CoreCycleEnv, observation: np.ndarray, **kwargs: Any) -> int:
    """Information-matched conventional rules; no rule reads true lifetime."""
    value = np.asarray(observation, dtype=np.float32)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("core rule requires one finite four-coordinate observation")
    if name in ("reactive", "wait"):
        return WAIT
    if name in ("always_replace", "replace", "immediate"):
        return REPLACE
    if name in ("age", "fixed_age"):
        return REPLACE if env.policy_state()["operating_age"] >= int(kwargs.get("age_interval", kwargs.get("trigger_age", kwargs.get("threshold", 83)))) else WAIT
    if name in ("health", "health_threshold"):
        return REPLACE if float(value[0]) <= float(kwargs.get("health_threshold", kwargs.get("threshold", .36))) else WAIT
    if name in ("projection", "local_projection"):
        lead = int(env.config["mobilization_lead"])
        threshold = kwargs.get("projection_threshold", kwargs.get("trigger", kwargs.get("threshold")))
        if threshold is None:
            if lead != 12:
                raise ValueError("the projection rule needs an explicit threshold when mobilization_lead is not 12")
            threshold = 14 if env.config["source"] == "discrete" else 15
        if value[2] < .5:
            return WAIT
        # The point rule retains the original scalar math.log calculation.
        health = float(value[0])
        if health <= 0:
            remaining = 0.
        else:
            damage = -math.log(health) / math.log(5.)
            remaining = (1. - damage) / max(float(value[1]) / lead, 1.e-12)
        return REPLACE if remaining <= float(threshold) else WAIT
    raise ValueError(f"unknown core policy {name!r}")
