# Environment contracts

MaintenanceArena has four tasks. They ask related questions but are not difficulty levels of one simulator, because each has its own clock, observation, action set and objective. Prices are illustrative currency units (CU), not calibrated industrial costs.

This page describes MaintenanceArena 1.0.0, and the [glossary](GLOSSARY.md) maps the notation of the accompanying article to these identifiers.

## Quick reference

| Gymnasium ID | Observation | Actions | One step | `terminated` | `truncated` | Length under `gymnasium.make` | Discount | Mask |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `maintenanceArena/CoreCycle-v0` | `Box(4,)` float32 in [0, 1], index 3 in [-1, 1] | `Discrete(2)`: WAIT, REPLACE | 1 calendar step, or 7 to 19 (`L + 7`) for the final transition | At the renewal | Only with `calendar_horizon` | One component cycle, at most 166 calendar steps and 160 decisions | 0.999 per calendar step (a fixed gamma is exact) | None |
| `maintenanceArena/Gearbox-v0` | `Box(4,)` float32, bounds [0.2, 1], [0, 1], [0, 1], [-1, 1] | `Discrete(3)`: NORMAL, DERATE, REPLACE | 1, 7 or up to 19 calendar steps | Always `False` (continuing) | Only with `horizon` | Unbounded unless `config={"horizon": N}` | 0.999 per calendar step, with `info["gamma_pow_duration"]` | None |
| `maintenanceArena/Resource-v0` | `Box(55,)` float32 in [0, inf) | `Discrete(12)`: order mode x job | 1 hour | Always `False` (continuing) | After `horizon` hours | 720 steps (720 hours) | 0.995 per hour (a fixed gamma is exact) | `info["action_mask"]` |
| `maintenanceArena/ResourceLifetime-v0` | `Box(58,)` float32 in [0, inf) | `Discrete(12)`: order mode x job | 1 hour | Always `False` (continuing) | After `horizon` hours | 720 steps (720 hours) | 0.995 per hour (a fixed gamma is exact) | `info["action_mask"]` |

Five profiles bind these environments to versioned task definitions:

| Profile | Environment |
| --- | --- |
| `paper-core-continuous-v1`, `paper-core-discrete-v1` | `CoreCycle-v0` |
| `paper-gearbox-v1` | `Gearbox-v0` |
| `paper-resource-v1` | `Resource-v0` |
| `paper-resource-lifetime-v1` | `ResourceLifetime-v0` |

The `paper-` profiles are the frozen published reference tasks. Profile and environment IDs carry independent version numbers ([glossary](GLOSSARY.md#tasks-profiles-and-environment-ids)).

## Using the environments with an RL library

Importing `maintenance_arena` registers the four IDs with Gymnasium. Tools that only take an ID string can pass `maintenance_arena:maintenanceArena/CoreCycle-v0`, which makes Gymnasium import the package. The IDs have no `max_episode_steps`, so `gymnasium.make` adds no `TimeLimit` wrapper. `maintenance_arena.wrappers.CalendarBudget(env, calendar_steps)` adds a cutoff in calendar time without changing the environment, and `maintenance_arena.tracks.make_track(name)` builds the environment of a [benchmark track](BENCHMARK.md) with its budget.

`gymnasium.make(env_id, config={...})` passes only an environment configuration. `maintenance_arena.make(profile_id, config=...)` resolves a named profile with the settings of its conventional policies and attaches it as `env.unwrapped.arena_profile`. Use it when you compare a learned policy with the built-in rules, because the rules read their thresholds from the profile.

A minimal Stable-Baselines3 loop for the core needs no package-specific code:

```python
import gymnasium as gym
import maintenance_arena  # registers the four IDs
from stable_baselines3 import PPO

env = gym.make("maintenanceArena/CoreCycle-v0")
model = PPO("MlpPolicy", env, gamma=0.999, seed=0, device="cpu")
model.learn(total_timesteps=100_000)
```

The resource tasks also need the action mask and normalized observations. MaskablePPO from `sb3-contrib` reads the mask through the environment's `action_masks()`:

```python
import gymnasium as gym
import maintenance_arena
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

venv = DummyVecEnv([lambda: gym.make("maintenanceArena/Resource-v0")])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.995)
model = MaskablePPO("MlpPolicy", venv, gamma=0.995, seed=0, device="cpu")
model.learn(total_timesteps=100_000)
```

Both snippets show the interface with untuned budgets. The package's own training recipes are in [Workflows](WORKFLOWS.md#train-and-continue-native-runs), and the `rl` extra installs PyTorch, Stable-Baselines3 and sb3-contrib.

To score a trained model on the same scenarios as the built-in references, pass `lambda observation, info: model.predict(observation, deterministic=True)[0]` to `maintenance_arena.benchmark.run_track` or `maintenance_arena.benchmark.score` ([Benchmark tracks and protocol](BENCHMARK.md), [Workflows](WORKFLOWS.md#score-your-own-agent-against-the-references)). A resource model also needs `action_masks=info["action_mask"]` in MaskablePPO's `predict` and its frozen training normalization. [examples/benchmark_your_agent.py](../examples/benchmark_your_agent.py) does this for the core and [examples/train_maskable_resource.py](../examples/train_maskable_resource.py) for `Resource-v0`.

## Units

All environments book cash in CU per primitive calendar step: one operating interval in the core and the gearbox, one hour in the resource tasks.

`simulate` and `evaluate` report `profit_per_1000_calendar_steps` per scenario, 1,000 times its cash profit over its elapsed calendar steps. The `evaluate` summary adds the mean of these rates (`mean_profit_per_1000_calendar_steps`) and the pooled rate, total cash over total calendar steps (`pooled_profit_per_1000_calendar_steps`), which `benchmark` reports. [Workflows](WORKFLOWS.md#compare-policies-on-paired-synthetic-units) lists the summary fields of evaluation schema v3, and [Benchmark tracks and protocol](BENCHMARK.md#metrics-and-pooled-aggregation) defines the `benchmark` metrics.

Divide these rates by 1,000 for CU per calendar step, which is CU per hour in the resource tasks. For example, 236,000 CU per 1,000 calendar steps is 236 CU per step.

In the resource tasks, `evaluate` reports cash profit, which charges the initial stock and every purchased part. Consumption-adjusted profit charges the consumed parts instead and answers a different accounting question. `maintenance_arena.resource_evaluation` implements it, and the resource PPO workflow uses it for its validation windows.

Rewards are learning coordinates, not cash. A reward unit is 1,000 CU in the core and the gearbox and 100 CU in the resource tasks. Core and gearbox rewards are also discounted within a transition, so report economics from `info["raw_profit_sum"]` and the ledger.

The `evaluate` summary field `failures_per_1000_lives` counts per ended life (planned intervention or failure), so dividing it by 10 gives failures per 100 lives. In the resource tasks, `failures_per_1000_calendar_steps` counts per 1,000 hours of the whole fleet.

## Shared public interface

All environments follow the [Gymnasium environment interface](https://gymnasium.farama.org/api/env/) with `reset(seed=...)` and `step(action)`, which returns `(observation, reward, terminated, truncated, info)`. Three more methods serve the built-in policies and evaluation:

- `policy_state()` is the structured public state that the conventional policies read, including the operating age for the core age rule.
- `diagnostics()` returns privileged evaluator information, such as the true remaining life.
- `snapshot()` and `restore(state)` cover the full simulator and random-generator state for native continuation.

A policy acts on the observation and, in the resource tasks, the action mask. Diagnostics and snapshots stay out of its input.

The common `info` fields are:

| Field | Meaning |
| --- | --- |
| `duration_calendar_steps` | Elapsed primitive calendar steps of the transition, 0 after reset. |
| `gamma_pow_duration` | Successor discount `gamma ** duration_calendar_steps`, 1.0 after reset. |
| `raw_profit_sum` | Undiscounted cash profit of the transition in CU. |
| `events` | Physical events of the transition, in order. |
| `ledger` | Economic breakdown, one row per calendar step in the core and one mapping per transition elsewhere. |
| `schema_version` | Identifier of the `info` layout. |

A terminated core episode has no successor value. A truncated episode only stops the accounting, so a learner should bootstrap from the final observation. Stable-Baselines3's on-policy algorithms do this through their vectorized environments, and the package's PPO implementations keep the distinction.

An explicit seed restarts the scenario streams, and `seed=None` continues them. A calendar cutoff completes the current transition and reports any overshoot, so it limits calendar time, not decisions. Call `reset` after termination or truncation. `render_mode="ansi"` renders a one-line text summary.

## Core: delayed replacement of one component

The core isolates replacement timing for one machine with one renewable component. Its profiles are `paper-core-continuous-v1` and `paper-core-discrete-v1`, both on `maintenanceArena/CoreCycle-v0`.

| Item | Contract |
| --- | --- |
| Class | `maintenance_arena.core.CoreCycleEnv`, configured by the keys of `maintenance_arena.core.DEFAULT_CONFIG` and the optional `recovery_contract`. `mobilization_lead` (6, 12 or 24, default 12) sets the [preparation time](#preparation-time), and `family` (one of eight laws, or `random` for the four source laws) the [degradation law](#degradation-laws). |
| Observation | `Box(4,)` float32 from `[0, 0, 0, -1]` to `[1, 1, 1, 1]`, unitless, named in `observation_names`: 0 `current_health`, 1 `lead_scaled_local_damage_rate` (normalized log-damage rate times `L`), 2 `rate_valid` (0/1), 3 `causal_intervention_slack`. |
| Actions | `Discrete(2)`: `0` WAIT, `1` REPLACE. Both are always valid, including an unsafe WAIT. Invalid codes raise `ValueError`. |
| Reward | `r = sum(0.999**i * c[i] / 1000 for i in range(tau))` over the cash `c[i]` in CU of the `tau` calendar steps of the transition (`gamma` and `reward_scale` in the configuration). A non-terminal transition has `tau = 1`. |
| `terminated` | `True` on the transition that completes a renewal: a planned exchange or a failure, followed by six downtime steps. |
| `truncated` | `True` only if `calendar_horizon` (default `None`) is set and reached before the renewal. |
| Length under `gymnasium.make` | One component cycle from installation: at most 166 calendar steps and 160 decisions for the continuous source, 150 and 144 for the discrete source. |
| Discount | 0.999 per calendar step. Only the final transition lasts more than one step, so a fixed `gamma=0.999` optimizes the same objective as the package's duration-aware PPO. |
| Action mask | Not needed. |
| Specific `info` keys | `calendar_steps`, `decision_steps`, `episode_raw_profit`, `episode_discounted_return`, `failures`, `planned_replacements`, `corrective_replacements`, `measurement_gap` (operating steps since the last available measurement), `observation_valid` (`rate_valid` as a Boolean), `calendar_budget`, `calendar_overshoot`, `objective` (`"one_cycle"`). |
| Events | `operate`, `mobilize`, `planned_exchange`, `downtime`, `failure_during_mobilization`, `corrective_exchange`. |
| Reset options | `options={"start_age": n}` starts at operating age `n` and reconstructs only the measurements up to that age. |
| Privileged diagnostics | Component family, sampled lifetime, true RUL at the first order and a trace with the true health. |
| Built-in policies | `reactive`, `age`, `health`, `projection`, `immediate`, and `ppo` with a checkpoint. The `oracle` of `maintenance-arena benchmark` is an evaluator-side reference, not a policy of the profile. |
| Units | CU per calendar step. One reward unit is 1,000 CU. |

The package's PPO recipe trains on a finite curriculum that starts some episodes shortly before the end of life (`CoreCycleEnv.configure_training`). A plain `gymnasium.make` environment starts at installation unless you pass `start_age`.

### Mechanics

Health starts at 1, and the component fails at 0.2. By default each new component draws one of the four source laws with equal probability: linear, concave running-in, convex accelerating or late-knee. Continuous-source lifetimes span 96 to 160 operating steps with varying shapes. The discrete recipe-screen source uses lifetimes 112, 128 and 144 with fixed shapes. A sampled component degrades deterministically.

| Item | Contract |
| --- | --- |
| Action 0 | WAIT for one productive operating step, unless the component fails. |
| Action 1 | Order a replacement. The transition runs the preparation, the exchange and the downtime to the end. |
| Planned replacement | `L` productive preparation steps, one nonproductive exchange step and six nonproductive downtime steps, so `L + 7` calendar steps (19 by default). |
| Failure, default | Immediate exchange, which cancels any remaining preparation, then the same six downtime steps. A failure while waiting takes 7 calendar steps. |
| Timing boundary | True RUL 13 (`L + 1`) is the last safe order, and an order at RUL 12 (`L`) fails during preparation. Since evaluation schema v3, `evaluate` records `order_margin` on the evaluator side: the true RUL at the first order minus `L + 1`. |
| Economics | 255 CU per productive step, 520 CU per replacement, 25 CU per nonproductive step and an extra 7,000 CU per failure. |

The simulator installs the successor at the exchange, but the episode deliberately ends at this renewal. A port to another library should keep this objective rather than adopt the continuing one of the gearbox.

The deadline coordinate (index 3) re-expresses health and rate. It adds no information and is not proven to be a sufficient Markov state. With normalized log damage `d` and rate channel `rho` (index 1), the projected remaining time is `T = (1 - d) / max(rho / L, 1e-12)`. The coordinate is `clip((L + 1 - T) / (L + 1), -1, 1)` with a valid rate and -1 without one.

`maintenance_arena.wrappers.ZeroDeadline` sets the coordinate to zero, as in the article's Health + rate input set, and the track `core-zero-deadline-v1` scores agents this way. `maintenance_arena.wrappers.InputSet(env, name)` selects the input set `full` (Full inputs), `deadline_only` (Deadline only) or `health_rate` (Health + rate) by multiplying the unused inputs by zero.

The rule settings selected on the continuous source are age 83, health 0.36 and projection 15. The discrete screen uses projection 14, selected on its own population. `maintenance-arena calibrate` selects the three thresholds again by grid search, for example for a changed configuration. The core benchmark tracks calibrate them on their own calibration lives ([Workflows](WORKFLOWS.md#calibrate-the-core-rules)).

#### Optional recovery with preserved preparation

`CoreCycleEnv({"recovery_contract": "preserve_preparation_v1"})` keeps the preparation already completed when a component fails. The accompanying article uses it in all synthetic studies, while the published profiles and benchmark tracks keep the default `immediate_exchange_v1`. Omitting the key or giving the default keeps the original configuration dictionary and checkpoint hash. The other value is recorded in configuration and snapshot records, so a checkpoint or snapshot cannot switch contracts unnoticed.

Planned orders are unchanged. A failure stops production and charges the failure surcharge once. With an order pending, only its uncompleted preparation remains. Without one, all `L` preparation steps remain, and the detection step counts as the first. An exchange then charges the replacement cost and installs the successor, and six downtime steps complete the renewal.

A reactive failure thus has `L + 7` nonproductive steps, 19 instead of seven at `L = 12`. An order one step after the last safe order leaves one nonproductive preparation step, eight nonproductive steps in all. Each costs 25 CU and earns nothing, and lost production is not charged again. The 7,000 CU surcharge sits on the failure row and the 520 CU replacement on the exchange row, so the ledger flags `failure` and `corrective_replacement` appear on different rows.

The event `failure` marks a detection without an order and `corrective_mobilize` each later preparation step. A detection during an order keeps `failure_during_mobilization`. The failed component stays installed at its endpoint until the exchange resets its age.

The last safe order stays at RUL 13, and the objective stays episodic. One `step` still covers the whole recovery. The longest episode from age zero grows by `L` steps, at `L = 12` to 178 calendar steps for the continuous and 162 for the discrete source.

### Simulated observation uncertainty

By default the core observes exact simulator health, a controlled starting point rather than a sensor-derived state. The reconstruction route (`observation_route`) and the sensor preset (`sensor_level`) are set independently.

| Route | Processing |
| --- | --- |
| `RAW_CAUSAL` | Consecutive difference of the current and the previous measurement (default). |
| `FILTERED_CAUSAL_EWLS8` | Exponentially weighted least-squares line through the measurements of the latest eight steps. |
| `ROBUST_MULTISCALE_234` | Median of the secant slopes over two, three and four steps. |

| Sensor preset | White-noise standard deviation | Bias-innovation standard deviation | Missing-measurement probability |
| --- | ---: | ---: | ---: |
| `exact` | 0 | 0 | 0 |
| `gaussian_005` | 0.005 | 0 | 0 |
| `light` | 0.0025 | 0.00025 | 0.01 |
| `moderate` | 0.005 | 0.0005 | 0.05 |
| `stress` | 0.01 | 0.001 | 0.10 |

The bias is an AR(1) process with coefficient 0.98 and stationary initialization. Measured health is clipped to [0, 1], the first measurement is always available, and reconstructors use only causal measurements, also when handling missing ones.

`gaussian_005` is the error model of the article's measurement-noise study, independent Gaussian error on the unit health scale. Combine `sensor_level: gaussian_005` with `RAW_CAUSAL` for consecutive measured differences or with `FILTERED_CAUSAL_EWLS8` for a fixed causal filter.

The rate and the deadline coordinate come from the same measurements. `RAW_CAUSAL` converts measured health to normalized log damage and clips negative increments to zero. The validity flag means that two consecutive measurements are available, not that the slope is positive, so a zero slope is valid and gives a distant deadline. The initial observation has an invalid rate.

Noise is drawn once per component age, from streams separate from the lifetime and curve draws, and reading an observation again draws no new error. The same seed thus pairs the latent component histories across noise settings, with identical degradation, failure times, rewards and costs for the same actions.

### Preparation time

`mobilization_lead` sets the preparation time `L` to 12 (the default and every published profile), 6 or 24, the values of the accompanying article. The step counts and boundaries above scale with `L`. The observation keeps its four coordinates and bounds, and `L` enters only the rate channel and the deadline coordinate.

Every reconstruction route supports every `L`, because the route estimates health and the per-step rate before this scaling. With `L = 12` the observations match the published ones bit for bit. `maintenance_arena.observations.Reconstructor(route, mobilization_lead)` and `reconstruct(..., mobilization_lead=L)` apply the same scaling. A reconstructor snapshot with a lead other than 12 restores only into a reconstructor with that lead.

The `projection` rule divides the rate channel by `L`. It needs a `projection_threshold` when `L` is not 12, because the profile defaults (15, or 14 for the discrete source) were selected for `L = 12`. `L` is part of the environment configuration, so a checkpoint trained for one `L` does not load into an environment with another.

The `train` command keeps its published curriculum, defined for `L = 12`, and refuses other values. Train those with `maintenance_arena.learning.collector.BankTrainer` ([Scenario banks](#scenario-banks-of-the-accompanying-article)). `calibrate` scores the projection rule with the configured `L`, on a default grid from `L - 4` to `L + 12` (8 to 24 at `L = 12`).

### Degradation laws

`family` selects the law of every new component, and the default `random` draws one of the four source laws with equal probability. The four new laws of the article's adaptation study are available by name with the continuous source. They keep the shared lifetime distribution, and their shape parameters are drawn after the lifetime, uniformly within `maintenance_arena.curves.TARGET_SHAPE_RANGES`:

| Law | Package identifier | Damage `g(u)` | Shape ranges |
| --- | --- | --- | --- |
| Paris-inspired | `paris_inspired` | `w u + (1-w) ((1-q u)^(-p) - 1)/((1-q)^(-p) - 1)` | `w` 0.15 to 0.45, `q` 0.55 to 0.8, `p` 0.7 to 1.4 |
| Two-knee | `two_knee` | normalized `w u + (1-w) (v S(u; k1, s) + (1-v) S(u; k2, s))` with the softplus knee `S` | `w` 0.08 to 0.22, `k1` 0.3 to 0.45, `k2` 0.72 to 0.86, `v` 0.2 to 0.4, `s` 0.025 to 0.045 |
| Gompertz | `gompertz` | normalized `exp(-A exp(-b u))` with `b = log(A)/i` | `A` 4 to 8, `i` 0.4 to 0.65 |
| Power-law | `power_law` | `u^beta` | `beta` 2.8 to 3.4 |

As for the source laws, health is `5^(-g(u))` at relative age `u = t/E` and falls from one to the failure level 0.2. The discrete source covers the four source laws only.

### Scenario banks of the accompanying article

The article's synthetic studies replay finite banks of independent lives, which the package rebuilds bit for bit. `maintenance_arena.banks.build_bank(law_set, split=..., seed=..., pair_count=...)` derives each active component and its successor from streams addressed by law set, split, seed, pair and side, independent of earlier builds. `law_set` is `source` (the four source laws in balanced blocks of four) or one new law.

`ScenarioBankEnv(bank, config)` runs the unchanged core physics on a bank. Each reset installs the next pair, `reset(seed=s)` rewinds to the first, and an exhausted bank raises `RuntimeError` instead of wrapping around. Measurement noise is addressed by component and replay seed, so it does not depend on earlier decisions. Snapshots are bound to the bank's hash.

A policy only waits until its first order, so the causal states of an unmaintained life fix the first order of every threshold rule and of every policy acting on these states. `maintenance_arena.replay` computes these states for whole banks (`decision_series`) and the first orders of rules, probability cuts and greedy policies. Its `replay` function applies the core's event accounting and matches step-by-step simulation.

The module also implements the calibrations of the article and of its additional tables: the Time rule and other threshold rules, age replacement, the two-indicator rule, the privileged per-law rule, ORDER cuts and failure-budget selection. A calibrated rule maximizes the calendar profit averaged with equal weight over the laws. Rates within 1e-9 tie, and ties go to the lower failure share, then the threshold nearer an anchor, then the earlier candidate.

`maintenance_arena.metrics` computes the equal-life failure share, unused life and retained profit, with a paired bootstrap over whole lives and all their runs. `maintenance_arena.learning.collector.BankTrainer` trains the core PPO recipe on a bank, with milestones, weights-only initialization (fine-tuning) and exact resumption. `maintenance_arena.learning.learners` trains the learners that the article compares with PPO, DQN and double DQN, on the same banks, and `greedy_first_orders` replays a saved policy of any of the three on a whole decision series.

For noisy measurements the article compares consecutive differences (`RAW_CAUSAL`) with the line fit over the latest eight observations (`FILTERED_CAUSAL_EWLS8`). `maintenance_arena.learning.dual` trains on both. `DualRouteBankEnv` replays a bank with both reconstructors. `DualPPOAgent` takes eight inputs, the consecutive-difference state followed by the line-fit state as rate, deadline, health and flag. Fine-tuning copies a four-input policy into the first four input columns and zeros the rest. `replay.dual_threshold_orders` and `calibrate_dual_rule` order when both estimates meet a threshold, or when either does.

`maintenance_arena.selection` implements the choice of the noise design in Section 4.3 of the article. `screen_values` computes the equal-law retained profit and failure shares of a screen. `select_rule_estimator` chooses the estimate of the Time rule under at most 1% failures overall and 2% per law. `select_design` chooses a design under at most min(1%, the Time rule's share + 0.25 points) and 2% per law. Both take the highest retained profit and, within 0.10 points, the lower failure share, and `select_design` then prefers the smaller training budget.

`maintenance_arena.analysis` computes the data of the article's synthetic figures. `health_paths` returns health paths with the last safe order, which Fig. 3 of the article draws as damage paths. `map_states` returns the causal states at every age of unmaintained lives, with health, damage rate, age and remaining life. `order_probabilities`, `binned_mean` and `native_means` turn them into maps of the state-weighted mean ORDER probability, in (health, damage rate) bins or at every distinct coordinate. `failure_share_curve` bins the share of failed training episodes over training steps. The maps cover all states of a life, including those after a policy would have ordered, so they are not the states a policy visits. `experiments/synthetic` holds the configurations and runners of the synthetic studies, the choice of the noise design and the figure data.

## Gearbox: operating choice with continuing renewal

The gearbox couples operation and maintenance, because the operator can slow wear by derating. Its profile is `paper-gearbox-v1` on `maintenanceArena/Gearbox-v0`.

| Item | Contract |
| --- | --- |
| Class | `maintenance_arena.gearbox.GearboxEnv`. The only configuration keys are `horizon`, `gamma` (fixed at 0.999) and `reward_scale` (fixed at 1,000). |
| Observation | `Box(4,)` float32 from `[0.2, 0, 0, -1]` to `[1, 1, 1, 1]`, unitless: 0 `health`, 1 `reference_damage_rate` (fixed at `12*12/1440`), 2 `opportunity` (LOW 0, HIGH 1), 3 `intervention_slack` (`clip((85 - q)/85, -1, 1)` for remaining capacity `q`). |
| Actions | `Discrete(3)`: `0` NORMAL, `1` DERATE, `2` REPLACE. Every action in the space is accepted, and invalid codes raise `ValueError`. |
| Reward | `r = sum(0.999**i * cents[i] / 100 / 1000 for i in range(tau))` over the integer-cent cash flows of the `tau` calendar steps, listed in `info["primitive_reward_cents"]`. |
| `terminated` | Always `False`. A renewal does not end the continuing objective, and `info["renewal"]` reports it. |
| `truncated` | `True` once `horizon` is set and the calendar steps reach it. The current transition completes first, and `info["calendar_overshoot"]` reports any excess. Stepping after truncation raises `RuntimeError` until `reset`. |
| Length under `gymnasium.make` | Unbounded, since `horizon` defaults to `None`. `config={"horizon": N}` or `maintenance_arena.wrappers.CalendarBudget(env, N)` cuts off after `N` calendar steps. A Gymnasium `TimeLimit` would count decisions instead. |
| Transition durations | NORMAL or DERATE without failure: 1 calendar step. Failure while operating: 7 steps. Planned REPLACE: 19 steps (12 forced DERATE preparation steps, exchange, 6 downtime steps). REPLACE that fails during preparation: the preparation steps up to and including the failure, plus 6 downtime steps (7 to 18 steps). |
| Discount | 0.999 per calendar step. Bootstrap with `info["gamma_pow_duration"]`, since one fixed gamma per decision would treat a 19-step replacement like a 1-step transition. Scoring always uses calendar profit. |
| Action mask | Not needed. The optional execution filter is a separate execution policy, not a mask. |
| Specific `info` keys | `renewal`, `metrics` (calendar steps, decisions, renewals, failures, raw profit, public state and cumulative exact-model metrics), `calendar_steps`, `calendar_budget`, `calendar_overshoot`. After a step also `primitive_reward_cents`, `planned_renewals`, `corrective_renewals`, `failures`, `calendar_step_start`, `calendar_step_end`. |
| Ledger | Mapping with `revenue`, `replacement_cost`, `downtime_cost`, `failure_cost` and `net_profit` in CU. |
| Events | `decision`, and on renewal also `exchange` or `failure`, then `renewal_ready`. |
| Reset options | `options={"damage": d, "opportunity": o}` for synthetic starts. Otherwise `reset` installs a new component and samples the opportunity. |
| Built-in policies | `reactive_normal`, `reactive_derate`, `normal_q96`, `derate_q89`, `separable_q96`, `dp`, and `ppo` with a checkpoint. `dp` is the exact average-profit-optimal policy of the finite model, solved by policy iteration. |
| Units | CU per calendar step. One reward unit is 1,000 CU. |

### Mechanics

The exact finite process has a damage capacity of 1,440 and an economic opportunity that is LOW or HIGH with equal probability. NORMAL consumes 12 damage units per operating step and earns 255 CU at HIGH and 242.25 CU at LOW opportunity. DERATE consumes 7 units and earns 242.25 CU. A replacement costs 520 CU, a nonproductive step 25 CU and a failure an extra 7,000 CU. The task deliberately grants the exact state and mechanics.

A planned REPLACE has the same twelve productive preparation steps as the core, with operation forced to DERATE. A step returns the complete transition and the sampled successor. The package's PPO recipe subtracts a constant economic value prior of 233.01899356282811 CU per elapsed calendar step from its training coordinates, which leaves the reward and the cash ledger unchanged.

The exact DP and the strong `separable_q96` rule serve as model references. With `--filter`, a run records the proposed and the executed actions. The filter guarantees the model-specific boundary only from a remaining capacity of `q >= 85`. Training stays unfiltered, so zero filtered failures show neither learned nor physical safety.

## Resource: maintenance jobs, inventory and staffed hours

The resource task schedules maintenance for a small fleet with finite spares and one shift crew. Its profile is `paper-resource-v1` on `maintenanceArena/Resource-v0`.

| Item | Contract |
| --- | --- |
| Class | `maintenance_arena.resource.ResourceEnv` (alias `ResourceMaintenanceEnv`), configured by the fields of `ResourceMaintenanceConfig`. |
| Observation | `Box(55,)` float32 in `[0, inf)`, with the expanded names in `observation_names`. Indices 33 to 54 are constant configuration context. |
| Actions | `Discrete(12)`: `order_mode * (n_assets + 1) + job_choice`, with order modes none, normal and express and job choices none or asset 1 to 3. `encode_action` and `decode_action` convert. |
| Reward | `r = net_profit / 100` for the hour, where `net_profit` is `info["ledger"]["net_profit"]` in CU. |
| `terminated` | Always `False` (continuing fleet). |
| `truncated` | `True` when the elapsed hours reach `horizon`. This is an accounting cutoff, not the end of the fleet. Stepping afterwards raises `RuntimeError` until `reset`. |
| Length under `gymnasium.make` | 720 steps, because the default `horizon` is 720 hours. The package's resource training uses 1,440 hours. |
| Discount | 0.995 per hour (the `gamma` attribute). Every step lasts one hour, so a fixed `gamma=0.995` is exact. Bootstrap at truncation. |
| Action mask | `info["action_mask"]` (Boolean, length 12) or `env.unwrapped.action_masks()`, from the parts, crew and shift at the decision. An infeasible action from the space becomes an atomic WAIT without a partial purchase or a penalty, and `info["invalid_action"]` is `True`. |
| Normalization | Needed for neural policies, for example with Stable-Baselines3's `VecNormalize`, because the space is unbounded and the fields keep raw magnitudes such as a part cost of 600 CU. The package's recipe uses `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=10, gamma=0.995)` and binds the normalizer to the saved model. |
| Checker warning | `gymnasium.utils.env_checker.check_env` warns "A Box observation space maximum value is infinity. This is probably too high." The open bound is intended, since counts, hours and costs have no natural maximum. `gymnasium.make` itself does not warn. |
| Specific `info` keys | `action_mask`, `invalid_action`, `cumulative_ledger`, `metrics` (failures, planned and corrective starts, completed replacements, operating and down asset hours, crew hours, waiting hours by cause, parts ordered, delivered and consumed, inventory and backlog state, `invalid_actions` and more). |
| Ledger | Mapping with `contribution`, `initial_stock_cost`, `parts_cost`, `order_cost`, `holding_cost`, `labor_cost`, `service_cost`, `failure_cost`, `extra_down_cost` and `net_profit` in CU. The first step charges the initial stock. |
| Events | `order`, `service_start`, `failure`, `service_complete`, `delivery`. |
| Reset options | None. `reset` rejects overrides of the hidden state. |
| Built-in policies | `reactive`, `health`, `rolling`, and `ppo` with a checkpoint. `rolling` is the 48-hour rolling planner selected on the source scenarios. The task has no oracle. |
| Units | CU per hour. One reward unit is 100 CU. |

### Mechanics

The default fleet has three independent components, one crew staffed eight hours per 24-hour day, room for three parts and one part in stock. The initial operating ages are 0, 24 and 48 hours. Each new component draws a life from 120 to 200 operating hours and, with equal probability, a linear, concave power (exponent 0.65 to 0.9), convex power (1.5 to 2.8) or softplus-knee damage law.

No action sets the operating load. A job consumes a part and reserves a crew when work starts, and the component stops wearing then. Preventive work takes four staffed hours and corrective repair eight. Jobs pause off shift and keep their crew, and the new component is installed at completion. Normal deliveries take 12 to 36 hours and express deliveries 2 to 6, arriving at the end of the hour. An order placed now cannot supply a job at the same decision.

The public vector omits the operating age, the lifetime and its distribution, the sampled degradation law, future delivery times and the horizon. Saved policies need masks and normalization in the default order below. All fields are float32, with integers and Booleans in numeric form.

| Indices | Ordered fields and units |
| --- | --- |
| 0-5 | `phase` (hour within the shift period), `on_shift` (0/1), `hours_to_next_shift` (h), `stock` (parts), `inventory_position` (parts), `free_crews` (crews). |
| 6-11, 12-17, 18-23 | Assets 0/1/2, each `health` (unitless), `log_damage_rate` (log-health decrease/h), `failed` (0/1), `repair_work_remaining` (staffed h), `failure_wait_hours` (h), `contribution_per_hour` (CU/h). |
| 24-26, 27-29, 30-32 | Pipeline slots 0/1/2, each `occupied` (0/1), `age` (h since order), `mode` (normal 1, express 2, empty 0). |
| 33-35 | `n_assets`, `inventory_capacity`, `crew_count` (counts). |
| 36-43 | `shift_period`, `shift_hours`, normal lead minimum/maximum, express lead minimum/maximum, `pm_work_hours`, `cm_work_hours` (h). |
| 44-46 | `part_cost`, `normal_order_cost`, `express_order_cost` (CU/event). |
| 47-48 | `holding_cost_per_part_hour` (CU/part/h), `crew_wage_per_hour` (CU/crew/h). |
| 49-52 | `pm_setup_cost`, `cm_setup_cost`, `failure_event_cost` (CU/event), `extra_down_cost_per_asset_hour` (CU/asset/h). |
| 53-54 | `reward_scale` (CU per reward unit), `failure_health` (unitless). |

`reset` creates the configured initial stock and operating ages, with no pending orders or jobs and with fresh keyed streams for components and deliveries. Another asset count changes the observation and the action space, and another inventory capacity changes the observation. Neither fits the published 55-input model.

A down asset earns no contribution, so lost contribution is not charged again, and shift wages include idle staff. The consumption-adjusted profit corrects spare inventory but does not value the remaining life of installed components or unfinished repairs. Report the opening and closing stock, the unfinished work and the failure backlog alongside profit. Each conventional policy combines a maintenance trigger with a procurement rule, both selected on the source scenarios.

## Resource lifetime: observable operating age

This variant adds public operating-age records and a configurable lifetime spread. Its profile is `paper-resource-lifetime-v1` on `maintenanceArena/ResourceLifetime-v0`.

| Item | Contract |
| --- | --- |
| Class | `maintenance_arena.resource_lifetime.ResourceLifetimeEnv`, a subclass of the resource environment with the same configuration fields. |
| Observation | `Box(58,)` float32 in `[0, inf)`: the 55 resource fields in the same order, then `asset_0.operating_age` to `asset_2.operating_age` in operating hours. |
| Actions, reward, `terminated`, `truncated`, discount | As in the resource task. |
| Length under `gymnasium.make` | 720 steps (720 hours). |
| Action mask, normalization, checker warning | As in the resource task. |
| Specific `info` keys | As in the resource task. After a step, `schema_version` is `"resource_lifetime_spread_v1"`. |
| Privileged diagnostics | `diagnostics()` also reports the true unused life at each preventive-job start (`analysis_pm_starts`). |
| Built-in policies | `reactive`, `age`, `health`, `rolling`, and `ppo` with a checkpoint. |
| Units | CU per hour. One reward unit is 100 CU. |

### Mechanics

The physical resource contract is unchanged. The recorded age starts from `initial_ages`, advances only while a component operates, pauses during failure and repair, and resets when a replacement is complete.

Separate streams draw each component's lifetime quantile and shape, so a component keeps both when the lifetime range changes. This pairs runs across ranges, such as the fixed-mean ranges `[160, 160]`, `[144, 176]`, `[120, 200]` and `[80, 240]`. The default profile uses settings selected for the nominal range `[120, 200]`, and overriding the range does not retune them.

The age policy includes shift gaps and the preventive workload when it chooses the last suitable start before its age deadline, so it is not a calendar block schedule. Each procurement rule uses only the information its policy may use. The true unused life at a preventive start is available only through `diagnostics()` and analysis output, not through `info` or the observations.

The package supports PPO training on this task. A saved 55-input resource policy cannot be loaded into this 58-input task.
