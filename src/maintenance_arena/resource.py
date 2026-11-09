"""Hourly multi-machine maintenance with finite spares and shift-bound crews.

This controlled extension leaves the published single-machine environment intact.
Health is an exact simulator proxy, prices are illustrative, and the horizon is
an accounting cutoff of a continuing task. Delivery realizations and future wear
are deliberately absent from policy observations. See ``step`` for event order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import copy
import math
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .curves import _normalized_curve


SCHEMA_VERSION = "resource_maintenance_v1"
FAILURE_HEALTH = 0.20
COST_FIELDS = (
    "initial_stock_cost", "parts_cost", "order_cost", "holding_cost",
    "labor_cost", "service_cost", "failure_cost", "extra_down_cost",
)
LEDGER_FIELDS = ("contribution", *COST_FIELDS, "net_profit")


@dataclass(frozen=True, slots=True)
class ResourceMaintenanceConfig:
    n_assets: int = 3
    horizon: int = 720
    initial_stock: int = 1
    inventory_capacity: int = 3
    crew_count: int = 1
    shift_period: int = 24
    shift_hours: int = 8
    normal_lead_range: tuple[int, int] = (12, 36)
    express_lead_range: tuple[int, int] = (2, 6)
    pm_work_hours: int = 4
    cm_work_hours: int = 8
    part_cost: float = 600.0
    normal_order_cost: float = 40.0
    express_order_cost: float = 140.0
    holding_cost_per_part_hour: float = 0.05
    crew_wage_per_hour: float = 35.0
    pm_setup_cost: float = 100.0
    cm_setup_cost: float = 200.0
    failure_event_cost: float = 1500.0
    extra_down_cost_per_asset_hour: float = 0.0
    contribution_per_asset_hour: tuple[float, ...] = (100.0, 120.0, 80.0)
    lifetime_range: tuple[int, int] = (120, 200)
    initial_ages: tuple[int, ...] = (0, 24, 48)
    reward_scale: float = 100.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourceMaintenanceConfig":
        supplied = dict(value)
        unknown = set(supplied) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown resource configuration keys: {sorted(unknown)}")
        return cls(**supplied)

    def __post_init__(self) -> None:
        positive_ints = (
            "n_assets", "horizon", "inventory_capacity", "crew_count",
            "shift_period", "shift_hours", "pm_work_hours", "cm_work_hours",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.initial_stock, bool)
                or not isinstance(self.initial_stock, (int, np.integer))
                or not 0 <= self.initial_stock <= self.inventory_capacity):
            raise ValueError("initial_stock must be an integer within inventory capacity")
        if self.shift_hours > self.shift_period:
            raise ValueError("shift_hours must not exceed shift_period")
        for name in ("normal_lead_range", "express_lead_range", "lifetime_range"):
            value = tuple(getattr(self, name))
            if (len(value) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in value)
                    or value[0] < 1 or value[1] < value[0]):
                raise ValueError(f"{name} must contain two ordered positive integers")
            object.__setattr__(self, name, value)
        ages = tuple(self.initial_ages)
        if len(ages) != self.n_assets or any(
            isinstance(x, bool) or not isinstance(x, (int, np.integer))
            or not 0 <= x < self.lifetime_range[0] for x in ages
        ):
            raise ValueError("initial_ages must contain one nonnegative integer below minimum life per asset")
        object.__setattr__(self, "initial_ages", ages)
        contribution = tuple(self.contribution_per_asset_hour)
        if len(contribution) != self.n_assets or any(
            isinstance(x, bool) or not np.isfinite(x) or x < 0 for x in contribution
        ):
            raise ValueError("contribution_per_asset_hour must contain one finite nonnegative value per asset")
        object.__setattr__(self, "contribution_per_asset_hour", tuple(float(x) for x in contribution))
        numeric_costs = (
            "part_cost", "normal_order_cost", "express_order_cost", "holding_cost_per_part_hour",
            "crew_wage_per_hour", "pm_setup_cost", "cm_setup_cost", "failure_event_cost",
            "extra_down_cost_per_asset_hour", "reward_scale",
        )
        for name in numeric_costs:
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")


@dataclass(slots=True)
class _Asset:
    curve: np.ndarray
    age: int
    renewal: int = 0
    log_damage_rate: float = 0.0
    failed: bool = False
    repair_work_remaining: int = 0
    failure_wait_hours: int = 0

    @property
    def health(self) -> float:
        return float(self.curve[self.age])


@dataclass(frozen=True, slots=True)
class _Order:
    placed_at: int
    arrives_at: int
    mode: int


class ResourceEnv(gym.Env[np.ndarray, int]):
    """One action orders one spare and/or starts one feasible maintenance job.

    Action ``order_mode * (n_assets + 1) + job_choice`` has order modes
    0/1/2 = none/normal/express and job choices 0/1..n = none/asset.
    A physically invalid *joint* action becomes an atomic WAIT without a
    synthetic penalty. Out-of-space actions raise. Use ``action_masks`` for RL.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config: ResourceMaintenanceConfig | Mapping[str, Any] | None = None,
                 render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("resource render_mode must be None or 'ansi'")
        self.config = config if isinstance(config, ResourceMaintenanceConfig) else ResourceMaintenanceConfig.from_mapping(config or {})
        self.action_space = spaces.Discrete(3 * (self.config.n_assets + 1))
        self.render_mode = render_mode
        self.gamma = 0.995
        self._static_context = self._make_static_context()
        global_names = ("phase", "on_shift", "hours_to_next_shift", "stock", "inventory_position", "free_crews")
        asset_names = ("health", "log_damage_rate", "failed", "repair_work_remaining", "failure_wait_hours", "contribution_per_hour")
        self.observation_names = (
            *global_names,
            *(f"asset_{i}.{name}" for i in range(self.config.n_assets) for name in asset_names),
            *(f"pipeline_{i}.{name}" for i in range(self.config.inventory_capacity) for name in ("occupied", "age", "mode")),
            *(f"config.{name}" for name in self._static_context),
        )
        self.observation_space = spaces.Box(0.0, np.inf, shape=(len(self.observation_names),), dtype=np.float32)
        self._assets: list[_Asset] = []
        self._pipeline: list[_Order] = []
        self._time = 0
        self._stock = 0
        self._has_reset = False
        self._done = False

    def _make_static_context(self) -> dict[str, float]:
        # The horizon, sampled degradation law and its lifetime distribution
        # are scientific simulator settings, not available policy information.
        excluded = {"horizon", "lifetime_range", "initial_ages", "initial_stock", "contribution_per_asset_hour"}
        context: dict[str, float] = {}
        for field in fields(self.config):
            if field.name in excluded:
                continue
            value = getattr(self.config, field.name)
            if field.name in ("normal_lead_range", "express_lead_range"):
                context[f"{field.name}_min"] = float(value[0])
                context[f"{field.name}_max"] = float(value[1])
            else:
                context[field.name] = float(value)
        context["failure_health"] = FAILURE_HEALTH
        return context

    def _keyed_rng(self, *coordinates: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([*self._episode_key, *coordinates]))

    def _make_curve(self, asset_id: int, renewal: int) -> np.ndarray:
        rng = self._keyed_rng(100, asset_id, renewal)
        life = int(rng.integers(self.config.lifetime_range[0], self.config.lifetime_range[1] + 1))
        families = (
            ("normalized_linear", {}),
            ("normalized_power", {"beta_range": (0.65, 0.9)}),
            ("normalized_power", {"beta_range": (1.5, 2.8)}),
            ("normalized_softplus_knee", {"linear_weight_range": (0.1, 0.3), "knee_fraction_range": (0.6, 0.85), "softness_range": (0.03, 0.08)}),
        )
        law, parameters = families[int(rng.integers(len(families)))]
        damage = _normalized_curve(law, np.arange(life + 1, dtype=float) / life, parameters, rng)
        health = np.exp(-math.log(5.0) * damage)
        health[0], health[-1] = 1.0, FAILURE_HEALTH
        return health

    def _delivery_lead(self, placed_at: int, mode: int) -> int:
        bounds = self.config.normal_lead_range if mode == 1 else self.config.express_lead_range
        rng = self._keyed_rng(200, placed_at, mode)
        return int(rng.integers(bounds[0], bounds[1] + 1))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if options:
            raise ValueError("resource maintenance reset does not accept hidden-state overrides")
        super().reset(seed=seed)
        self._reset_extra()
        self._episode_key = tuple(int(x) for x in self.np_random.integers(0, 2**32, size=4, dtype=np.uint32))
        self._assets = [_Asset(self._make_curve(i, 0), int(age)) for i, age in enumerate(self.config.initial_ages)]
        self._pipeline = []
        self._time, self._stock = 0, self.config.initial_stock
        self._has_reset, self._done, self._initial_stock_charged = True, False, False
        self._cumulative = {name: 0.0 for name in LEDGER_FIELDS}
        self._metrics = {name: 0 for name in (
            "operating_asset_hours", "down_asset_hours", "crew_work_hours", "paid_crew_hours",
            "failures", "planned_starts", "corrective_starts", "completed_replacements",
            "failed_waiting_asset_hours", "no_part_wait_hours", "no_crew_wait_hours",
            "off_shift_wait_hours", "parts_ordered", "parts_delivered", "parts_consumed",
            "normal_orders", "express_orders", "invalid_actions",
        )}
        return self._observation(), self._info({name: 0.0 for name in LEDGER_FIELDS}, [], False)

    def _reset_extra(self) -> None:
        """Initialize variant records only after reset arguments are accepted."""

    @property
    def on_shift(self) -> bool:
        return self._time % self.config.shift_period < self.config.shift_hours

    @property
    def free_crews(self) -> int:
        return self.config.crew_count - sum(asset.repair_work_remaining > 0 for asset in self._assets)

    def encode_action(self, order_mode: int, job_choice: int) -> int:
        if (isinstance(order_mode, (bool, np.bool_)) or isinstance(job_choice, (bool, np.bool_))
                or not (isinstance(order_mode, (int, np.integer)) and isinstance(job_choice, (int, np.integer))
                        and 0 <= order_mode <= 2 and 0 <= job_choice <= self.config.n_assets)):
            raise ValueError("invalid order mode or job choice")
        return int(order_mode * (self.config.n_assets + 1) + job_choice)

    def decode_action(self, action: int) -> tuple[int, int]:
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the discrete action space")
        return divmod(int(action), self.config.n_assets + 1)

    def action_masks(self) -> np.ndarray:
        if not self._has_reset:
            raise RuntimeError("reset must precede action_masks")
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action in range(self.action_space.n):
            order, job = self.decode_action(action)
            feasible_job = job == 0 or (
                self.on_shift and self.free_crews > 0 and self._stock > 0
                and self._assets[job - 1].repair_work_remaining == 0
            )
            # The new job consumes one existing spare, freeing a reservation
            # for a simultaneous order. A same-hour order cannot supply a job.
            position_after_job = self._stock + len(self._pipeline) - int(job > 0 and feasible_job)
            mask[action] = feasible_job and (order == 0 or position_after_job < self.config.inventory_capacity)
        return mask

    def policy_state(self) -> dict[str, Any]:
        """Observable structured equivalent of the flat RL vector, as copies."""
        if not self._has_reset:
            raise RuntimeError("reset must precede policy_state")
        phase = self._time % self.config.shift_period
        return {
            "phase": phase,
            "on_shift": self.on_shift,
            "hours_to_next_shift": 0 if self.on_shift else self.config.shift_period - phase,
            "stock": self._stock,
            "inventory_position": self._stock + len(self._pipeline),
            "free_crews": self.free_crews,
            "assets": [{
                "health": asset.health, "log_damage_rate": asset.log_damage_rate,
                "failed": asset.failed, "repair_work_remaining": asset.repair_work_remaining,
                "failure_wait_hours": asset.failure_wait_hours,
                "contribution_per_hour": self.config.contribution_per_asset_hour[i],
            } for i, asset in enumerate(self._assets)],
            "pipeline": [{"age": self._time - order.placed_at, "mode": order.mode} for order in self._pipeline],
            "config": dict(self._static_context),
        }

    def _observation(self) -> np.ndarray:
        state = self.policy_state()
        values = [state[name] for name in ("phase", "on_shift", "hours_to_next_shift", "stock", "inventory_position", "free_crews")]
        for asset in state["assets"]:
            values.extend(asset.values())
        for i in range(self.config.inventory_capacity):
            values.extend((1, state["pipeline"][i]["age"], state["pipeline"][i]["mode"]) if i < len(state["pipeline"]) else (0, 0, 0))
        values.extend(state["config"].values())
        return np.asarray(values, dtype=np.float32)

    def _info(self, ledger: dict[str, float], events: list[dict[str, Any]], invalid: bool) -> dict[str, Any]:
        metrics = dict(self._metrics)
        metrics.update({
            "elapsed_hours": self._time,
            "inventory_on_hand": self._stock,
            "pipeline_units": len(self._pipeline),
            "repair_jobs_open": sum(a.repair_work_remaining > 0 for a in self._assets),
            "repair_work_remaining": sum(a.repair_work_remaining for a in self._assets),
            "failed_assets": sum(a.failed for a in self._assets),
            "failed_backlog_assets": sum(a.failed and a.repair_work_remaining == 0 for a in self._assets),
            "inventory_conservation_residual": self.config.initial_stock + self._metrics["parts_delivered"] - self._metrics["parts_consumed"] - self._stock,
        })
        return {
            "action_mask": self.action_masks(), "ledger": dict(ledger),
            "cumulative_ledger": dict(self._cumulative), "metrics": metrics,
            "events": events, "invalid_action": invalid, "schema_version": SCHEMA_VERSION,
            "duration_calendar_steps": int(self._time > 0),
            "gamma_pow_duration": self.gamma if self._time > 0 else 1.0,
            "raw_profit_sum": float(ledger["net_profit"]),
        }

    def _operating_contribution(self, asset_id: int) -> float:
        """Contribution of one operating hour; default resource mechanics."""
        return self.config.contribution_per_asset_hour[asset_id]

    def _advance_operating_wear(self, asset_id: int, asset: _Asset) -> None:
        """Advance wear before the native failure check; one hour by default."""
        asset.age += 1

    def step(self, action: int):
        """Decide; produce/wear and pay/work for [t,t+1); finish; deliver.

        Productive assets earn contribution before an end-of-hour failure.
        Jobs reserve a crew across off-shifts and consume their part at start.
        Shift wages include idle staff; down machines lose contribution, with
        an optional *additional* down cost kept separate. No automatic repair.
        """
        if not self._has_reset or self._done:
            raise RuntimeError("reset is required before stepping or after the accounting cutoff")
        order_mode, job_choice = self.decode_action(action)
        invalid = not bool(self.action_masks()[int(action)])
        if invalid:
            order_mode, job_choice = 0, 0
            self._metrics["invalid_actions"] += 1
        ledger = {name: 0.0 for name in LEDGER_FIELDS}
        events: list[dict[str, Any]] = []
        cfg = self.config
        if not self._initial_stock_charged:
            ledger["initial_stock_cost"] = cfg.initial_stock * cfg.part_cost
            self._initial_stock_charged = True
        if order_mode:
            self._pipeline.append(_Order(self._time, self._time + self._delivery_lead(self._time, order_mode), order_mode))
            ledger["parts_cost"] += cfg.part_cost
            ledger["order_cost"] += cfg.normal_order_cost if order_mode == 1 else cfg.express_order_cost
            self._metrics["parts_ordered"] += 1
            self._metrics["normal_orders" if order_mode == 1 else "express_orders"] += 1
            events.append({"type": "order", "time": self._time, "mode": order_mode})
        if job_choice:
            asset = self._assets[job_choice - 1]
            corrective = asset.failed
            asset.repair_work_remaining = cfg.cm_work_hours if corrective else cfg.pm_work_hours
            self._stock -= 1
            self._metrics["parts_consumed"] += 1
            self._metrics["corrective_starts" if corrective else "planned_starts"] += 1
            ledger["service_cost"] += cfg.cm_setup_cost if corrective else cfg.pm_setup_cost
            events.append({"type": "service_start", "time": self._time, "asset": job_choice - 1, "kind": "corrective" if corrective else "planned"})

        ledger["holding_cost"] = self._stock * cfg.holding_cost_per_part_hour
        if self.on_shift:
            self._metrics["paid_crew_hours"] += cfg.crew_count
            ledger["labor_cost"] = cfg.crew_count * cfg.crew_wage_per_hour
        free_crews_during_interval = self.free_crews
        # Waiting diagnostics overlap: a failed job may lack both parts and a
        # free crew while off shift. They are not additive downtime categories.
        for i, asset in enumerate(self._assets):
            busy = asset.repair_work_remaining > 0
            if not busy and not asset.failed:
                ledger["contribution"] += self._operating_contribution(i)
                self._metrics["operating_asset_hours"] += 1
                previous_health = asset.health
                self._advance_operating_wear(i, asset)
                asset.log_damage_rate = math.log(previous_health) - math.log(asset.health)
                if asset.health <= FAILURE_HEALTH:
                    asset.failed = True
                    self._metrics["failures"] += 1
                    ledger["failure_cost"] += cfg.failure_event_cost
                    events.append({"type": "failure", "time": self._time + 1, "asset": i})
            else:
                self._metrics["down_asset_hours"] += 1
                ledger["extra_down_cost"] += cfg.extra_down_cost_per_asset_hour
                if asset.failed and not busy:
                    asset.failure_wait_hours += 1
                    self._metrics["failed_waiting_asset_hours"] += 1
                    self._metrics["no_part_wait_hours"] += int(self._stock == 0)
                    self._metrics["no_crew_wait_hours"] += int(free_crews_during_interval == 0)
                    self._metrics["off_shift_wait_hours"] += int(not self.on_shift)
            if busy and self.on_shift:
                asset.repair_work_remaining -= 1
                self._metrics["crew_work_hours"] += 1
                if asset.repair_work_remaining == 0:
                    asset.renewal += 1
                    asset.curve = self._make_curve(i, asset.renewal)
                    asset.age, asset.log_damage_rate = 0, 0.0
                    asset.failed, asset.failure_wait_hours = False, 0
                    self._metrics["completed_replacements"] += 1
                    events.append({"type": "service_complete", "time": self._time + 1, "asset": i})

        ledger["net_profit"] = ledger["contribution"] - sum(ledger[name] for name in COST_FIELDS)
        for name, value in ledger.items():
            self._cumulative[name] += value
        self._time += 1
        outstanding = []
        for order in self._pipeline:
            if order.arrives_at <= self._time:
                self._stock += 1
                self._metrics["parts_delivered"] += 1
                events.append({"type": "delivery", "time": self._time, "mode": order.mode})
            else:
                outstanding.append(order)
        self._pipeline = outstanding
        self._done = self._time >= cfg.horizon
        return self._observation(), ledger["net_profit"] / cfg.reward_scale, False, self._done, self._info(ledger, events, invalid)

    def diagnostics(self) -> dict[str, Any]:
        """Privileged inspection, deliberately absent from policy state/info."""
        if not self._has_reset:
            raise RuntimeError("reset must precede diagnostics")
        return {"calendar_time": self._time, "assets": [
            {"asset": i, "operating_age": a.age, "renewal": a.renewal,
             "true_lifetime": len(a.curve) - 1,
             "true_remaining_life": len(a.curve) - 1 - a.age}
            for i, a in enumerate(self._assets)],
            "deliveries": [asdict(order) for order in self._pipeline]}

    def snapshot(self) -> dict[str, Any]:
        """Full simulator/RNG state for native continuation, not a policy input."""
        if not self._has_reset:
            raise RuntimeError("reset must precede snapshot")
        return copy.deepcopy({
            "schema": "maintenance_arena_resource_state_v1",
            "environment_class": type(self).__name__, "configuration": asdict(self.config),
            "episode_key": self._episode_key, "time": self._time, "stock": self._stock,
            "done": self._done, "initial_stock_charged": self._initial_stock_charged,
            "assets": [dict(asdict(a), curve=a.curve.tolist()) for a in self._assets],
            "pipeline": [asdict(order) for order in self._pipeline],
            "cumulative": self._cumulative, "metrics": self._metrics,
            "rng_state": self.np_random.bit_generator.state,
            "action_rng_state": self.action_space.np_random.bit_generator.state,
            "extra": self._snapshot_extra(),
        })

    def _snapshot_extra(self) -> dict[str, Any]:
        return {}

    def _restore_extra(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("unexpected resource snapshot extension")

    def restore(self, state: Mapping[str, Any]) -> np.ndarray:
        """Restore a snapshot only into the same environment/configuration."""
        state = copy.deepcopy(dict(state))
        if (state.get("schema") != "maintenance_arena_resource_state_v1"
                or state.get("environment_class") != type(self).__name__
                or ResourceMaintenanceConfig.from_mapping(state["configuration"]) != self.config):
            raise ValueError("resource snapshot environment/configuration mismatch")
        self._episode_key = tuple(state["episode_key"])
        self._time, self._stock = int(state["time"]), int(state["stock"])
        self._done, self._initial_stock_charged = state["done"], state["initial_stock_charged"]
        self._assets = [_Asset(**dict(row, curve=np.asarray(row["curve"], dtype=float)))
                        for row in state["assets"]]
        self._pipeline = [_Order(**row) for row in state["pipeline"]]
        self._cumulative, self._metrics = state["cumulative"], state["metrics"]
        self.np_random.bit_generator.state = state["rng_state"]
        self.action_space.np_random.bit_generator.state = state["action_rng_state"]
        self._restore_extra(state["extra"])
        self._has_reset = True
        return self._observation()

    def render(self):
        if self.render_mode == "ansi":
            if not self._has_reset:
                return "Resource maintenance: reset required"
            state = self.policy_state()
            return (f"hour={self._time} stock={state['stock']} "
                    f"free_crews={state['free_crews']} "
                    f"health={[round(a['health'], 4) for a in state['assets']]}")
        return None


# Alternative names for the resource environment and its configuration.
ResourceMaintenanceEnv = ResourceEnv
ResourceConfig = ResourceMaintenanceConfig


def policy_action(name: str, env: ResourceEnv, observation=None, **kwargs) -> int:
    """Dispatch declared conventional policies using only observable fields."""
    from .resource_planning import PlanningPolicyConfig, choose_planning_action
    names = {"reactive": "reactive", "health": "health_threshold", "rolling": "rolling_plan",
             "health_threshold": "health_threshold", "rolling_plan": "rolling_plan"}
    if name not in names:
        raise ValueError(f"unsupported resource policy: {name}")
    supplied = dict(kwargs)
    if supplied.pop("trigger", names[name]) != names[name]:
        raise ValueError("policy name conflicts with its configured resource trigger")
    policy = PlanningPolicyConfig(trigger=names[name], **supplied)
    return choose_planning_action(env, policy)
