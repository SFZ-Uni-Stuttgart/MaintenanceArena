# Article-to-package glossary

The accompanying article, *Learning When to Act: What Reinforcement Learning Agents Learn about Maintenance Timing and When It Transfers*, describes the decision problem with mathematical symbols and descriptive names. This page maps them to the code identifiers of MaintenanceArena 1.0.0. The runners in [experiments/](../experiments/README.md) select the article's settings, and [paper/](../paper/README.md) holds the saved outcomes of its tables. Saved outcomes and some identifiers use their own names, for example Rule and Forecast rule for the article's Time rule. The tables below name them where they differ.

## Tasks, profiles and environment IDs

A Gymnasium environment ID such as `maintenanceArena/CoreCycle-v0` names an environment class. A profile ID such as `paper-core-continuous-v1` names a versioned task definition: an environment ID with a fixed environment configuration, the settings of the conventional policies and the PPO training recipe. The prefix `paper-` marks the published reference tasks, whose identities stay frozen. The two version numbers are independent. A profile changed through `--config` or a Python override gets the identifier `<profile>+<12 hexadecimal characters>` and does not inherit the published results.

| Article | Package |
| --- | --- |
| Simulated maintenance decision process of one component | `maintenanceArena/CoreCycle-v0` (profile `paper-core-continuous-v1`) with `recovery_contract: preserve_preparation_v1` |
| Training, calibration and test lives of the synthetic studies | `maintenance_arena.banks.build_bank` and `ScenarioBankEnv`, which runs the core physics on a bank |
| Replay of recorded bearing histories (FEMTO, XJTU-SY) | `maintenance_arena.bearings.BearingReplayEnv` and `maintenance_arena.bearings.ReplayEconomics` |

The package also contains tasks that the article does not use. [Environment contracts](ENVIRONMENTS.md) describes the discrete core source (`paper-core-discrete-v1`), the gearbox with an operating choice (`paper-gearbox-v1`) and the resource tasks with parts, crews and shifts (`paper-resource-v1`, `paper-resource-lifetime-v1`).

## Decision process and observations

| Article | Package | Notes |
| --- | --- | --- |
| WAIT | Action `0`, `WAIT` | One more productive step. |
| ORDER | Action `1`, `REPLACE` (core) or `ORDER` (bearing replay) | Starts preparation, exchange and downtime as one committed transition. |
| Preparation time `L` | `mobilization_lead` | 6, 12 or 24. The default 12 is the article's main setting. |
| One exchange step and six downtime steps | Task contract, `downtime_steps` | Fixed at 6. |
| Prices 255, 25, 520 and 7,000 CU | `revenue_per_operating_step`, `downtime_cost_per_step`, `replacement_cost`, `failure_event_cost` | Illustrative currency units. `experiments/synthetic/failure_cost.py` trains with a failure cost of 3,500 and 14,000 CU (Section 4.1 and Appendix C). |
| Failure stops production and keeps completed preparation | `recovery_contract: preserve_preparation_v1` | The default `immediate_exchange_v1` cancels the remaining preparation instead ([Environment contracts](ENVIRONMENTS.md#optional-recovery-with-preserved-preparation)). |
| Last safe order, `L + 1` steps before failure | Task contract | An order at true RUL `L + 1` is the last that completes before failure. |
| Damage `d_t` | Not exposed | `-log(h)/log(5)` from health, clipped to [0, 1]. |
| Health `h_t = 5^(-d_t)` | Index 0, `current_health` | Falls from 1 to the failure threshold 0.2. |
| Damage rate channel `L * delta_t` | Index 1, `lead_scaled_local_damage_rate` | Latest damage increment times `L`, clipped to [0, 1]. |
| Flag that a damage rate is available | Index 2, `rate_valid` | 1 once the reconstruction has a valid rate, with exact measurements from the second measurement on. |
| Deadline coordinate `z_t` | Index 3, `causal_intervention_slack` | `clip((L + 1 - T_hat)/(L + 1), -1, 1)` with a valid rate, otherwise -1. |
| Projected remaining time `T_hat_t = (1 - d_t) / max(delta_t, 1e-12)` | `maintenance_arena.replay.projected_time` and the `projection` rule | Not an observation field. |
| Input sets Full inputs, Deadline only, Health + rate | `maintenance_arena.wrappers.InputSet` with `full`, `deadline_only`, `health_rate` | Unused inputs are multiplied by zero. `ZeroDeadline` and the track `core-zero-deadline-v1` give Health + rate. |
| True RUL, failure age and degradation law | `diagnostics()` only | Not a policy input. |
| Discount `gamma = 0.999` per calendar step and continuation discount `gamma^m` after an action of `m` steps | `gamma` and `info["gamma_pow_duration"]` | See [Environment contracts](ENVIRONMENTS.md#core-delayed-replacement-of-one-component). |
| Reward scale 1/1000 | `reward_scale: 1000` | One reward unit is 1,000 CU. |

## Degradation laws

The failure age follows the article's lognormal distribution with log-location `log 128` and log-scale 0.12, conditioned to 95.5 to 160.5 and rounded. Shape parameters are uniform within the ranges of Appendix B. `family` selects one law, and `random` draws one of the four source laws with equal probability for each new component.

| Article | Package identifier | Shape range |
| --- | --- | --- |
| Linear | `linear_reference` | none |
| Concave | `concave_running_in` | `c` from 3 to 5 |
| Convex | `convex_accelerating_fatigue` | `beta` from 1.6 to 2.4 |
| Late-knee | `bearing_late_knee` | knee `k` from 0.62 to 0.78, softness 0.04, linear share 0.15 |
| Paris-inspired | `paris_inspired` | `w` 0.15 to 0.45, `q` 0.55 to 0.8, `p` 0.7 to 1.4 |
| Two-knee | `two_knee` | `w` 0.08 to 0.22, `v` 0.2 to 0.4, `k1` 0.3 to 0.45, `k2` 0.72 to 0.86, `s` 0.025 to 0.045 |
| Gompertz | `gompertz` | `A` 4 to 8, `i` 0.4 to 0.65 |
| Power-law | `power_law` | `beta` 2.8 to 3.4 |

`maintenance_arena.curves` lists the laws (`FAMILY_IDS`, `TARGET_FAMILY_IDS`, `LAW_IDS`) and the shape ranges of the new laws (`TARGET_SHAPE_RANGES`). The resource tasks use laws of their own ([Environment contracts](ENVIRONMENTS.md#resource-maintenance-jobs-inventory-and-staffed-hours)).

## Policies, references and calibration

| Article | Package | Notes |
| --- | --- | --- |
| Time rule, a calibrated threshold `tau` on the projected remaining time | `projection` with `projection_threshold`, `maintenance_arena.replay.calibrate_rule` | Highest calendar profit on the calibration lives among health and projected-time thresholds, with the article's grids and tie-breaks. `T_hat <= 15` on the source laws at `L = 12`, recalibrated for every `L` and every new law. Stored as Rule, and in Tables 3 and 4 as Recalibrated Rule and Source Rule. |
| Time rule without margin, `tau = L + 1` (13 in Table 2) | `projection` with `projection_threshold: L + 1` | Stored as Naive projection rule. |
| Reactive operation (replace after failure) | `reactive` | Additional tables of source controls and bearing comparison. |
| Age replacement | `age` with `age_interval`, `maintenance_arena.replay.calibrate_age` | Fitted on calibration lives. Additional table of source controls. |
| Health threshold | `health` with `health_threshold`, `maintenance_arena.replay.calibrate_rule` | Fitted on calibration lives. Additional table of source controls. |
| PPO | `maintenance_arena.learning.collector.BankTrainer`, and `ppo` with `--checkpoint` | Two hidden layers of 64 tanh units and the settings of Table A1. Orders when the ORDER probability exceeds the WAIT probability. |
| Fine-tuned and fresh policies | `BankTrainer(bank, seed, initialize=...)` and `BankTrainer(bank, seed)` | Fine-tuning starts from saved weights with a fresh optimizer. |
| DQN and double DQN (Appendix A) | `maintenance_arena.learning.learners` (`make_trainer`, `recipe`, `load_policy`) | One fixed recipe each, trained on the same banks, inputs, rewards and budget as PPO. They order when ORDER is the greedy action. |
| PPO + calibration, a calibrated threshold on the ORDER probability | `maintenance_arena.replay.calibrate_cut` | 123 candidate cuts, fitted on calibration lives. Stored as PPO + calibrated cut and, in Table 4, Fine-tuned PPO, calibrated cut. |
| Two-indicator rule, per-law rule, failure-budget variants | `calibrate_two_indicator`, `calibrate_per_law` and `select_failure_budget` in `maintenance_arena.replay` | Additional table of further calibrations. |
| Last-safe reference that knows the failure time (oracle) | `maintenance_arena.replay.reference_orders`, and the `oracle` of `maintenance_arena.benchmark` | Evaluator side, not a policy input. |
| Grid-search calibration of the package's benchmark rules | `calibrate` command, `maintenance_arena.calibration.calibrate` | 252 default candidates. The default criterion `zero_failure` admits only candidates without a failure, and `--criterion pooled` selects by pooled calendar profit ([Workflows](WORKFLOWS.md#calibrate-the-core-rules)). |
| Immediate replacement | `immediate` | A package baseline that the article does not report. |

## Observation noise

| Article | Package | Notes |
| --- | --- | --- |
| Independent Gaussian error with standard deviation 0.005 on normalized health | `sensor_level: gaussian_005` | No bias and no missing measurements. |
| Damage rate from consecutive noisy observations | `observation_route: RAW_CAUSAL` | |
| Causal line fit over the latest eight observations | `observation_route: FILTERED_CAUSAL_EWLS8` | Exponentially weighted least squares. |
| Noisy calibration and test lives | `maintenance_arena.banks.with_sensor_level` | Noise is addressed by component, so paired comparisons see the same measurements. |
| Policies trained on both estimates | `maintenance_arena.learning.dual` (`DualPPOAgent`, `DualBankTrainer`) | Eight inputs, the raw and the line-fit state, started from a four-input policy. |
| Time rule whose state estimate is chosen among four | `maintenance_arena.replay.calibrate_dual_rule`, `maintenance_arena.selection.select_rule_estimator` | Raw, line fit, both or either estimate. |
| Candidate designs of the noise study, failure bounds and the selection of a design | `maintenance_arena.selection` (`screen_values`, `select_design`), and `experiments/synthetic/measurement_noise.py --stage development` and `--stage selection` | Section 4.3 of the article, where Table 3 evaluates the selected design. |

[Environment contracts](ENVIRONMENTS.md#simulated-observation-uncertainty) lists all sensor presets and reconstruction routes.

## Bearing replay

| Article | Package | Notes |
| --- | --- | --- |
| One step is one recorded minute, and the last safe order lies twelve minutes before the recorded end | `maintenance_arena.bearings.ReplayEconomics` | The recorded end is the last productive minute `T`. |
| Remaining-time prediction model | `maintenance_arena.bearings.Forecaster`, `cross_fit` | Gradient-boosted regression on causal vibration features, with out-of-fold predictions for training and calibration. The code calls the model a forecaster. |
| Input designs Full, No age, No policy age, No prediction age, condition-only policy | `INPUT_DESIGNS` in `maintenance_arena.bearings` | Where operating age enters the prediction model and the policy. No prediction age is stored as No forecaster age (`no_forecaster_age`). |
| Bearing PPO with a running estimate of calendar profit | `maintenance_arena.bearings.ppo.train_policy` | Rewards subtract the running profit estimate, and reported profits use the actual cash. |
| Time rule on bearings | `calibrate_forecast_rule`, `forecast_rule_order` | One cutoff on the prediction supplied to the matching PPO design, stored as Forecast rule. The variant without margin orders at 12 minutes and is stored as Naive forecast rule. |
| Indicator rule | `calibrate_indicator_rule` | One threshold on age, vibration level or peak, the prediction or its trend. Additional bearing table, stored as Indicator Rule. |
| Cross-validated search over sixteen threshold forms | `RULE_FAMILIES`, `search_rule_families` | With inner cross-validation on the training bearings (Section 6.4 and Appendix D). |
| Risk classifier on the inputs of Full or No age | `calibrate_risk_classifier`, `risk_classifier_orders` | Gradient-boosted classifier of the minutes at or after the last safe order, with a threshold chosen on leave-one-bearing-out scores (Table 5). |
| Selection among six candidate designs | `rank_candidates`, `experiments/bearings/run_candidate_selection.py` | Held-out FEMTO pooled calendar profit (Appendix D). `run_nested_selection.py` repeats it inside the training bearings of every outer fold (Section 6.1). |

[docs/BEARINGS.md](BEARINGS.md) describes the data, the features and the replay in detail.

## Objectives, units and outputs

| Article | Package | Notes |
| --- | --- | --- |
| Order margin `M = RUL_order - (L + 1)` | `order_margin` | Recorded per evaluated core scenario, with `order_margin_mean`, `order_margin_sd` and `late_order_share` in the summary and in the `benchmark` table. |
| Share of lives that fail, in percent | `maintenance_arena.metrics.summarize`, or `failures_per_1000_lives / 10` in `evaluate` | Runs are averaged per life first. |
| Unused life: the positive margin divided by the time from installation to the last safe order | `unused_fraction` in the runners' outcomes, `maintenance_arena.metrics` | `evaluate` and `benchmark` report `normalized_discarded_life`, which divides by the lifetime instead. |
| Retained profit: a life's calendar profit as a percentage of the oracle's on the same life, averaged with equal weight per life | `maintenance_arena.metrics.summarize` | `percent_of_reference` in `benchmark.json` divides pooled profits over all scenarios instead. |
| Paired bootstrap intervals over lives or bearings | `maintenance_arena.metrics.paired_bootstrap`, `maintenance_arena.bearings.metrics.paired_bootstrap` and `pooled_bootstrap` | Whole lives or bearings are resampled with all their runs. |
| Calendar profit, cash per elapsed step | `profit_per_1000_calendar_steps / 1000` per scenario, `pooled_profit_per_1000_calendar_steps / 1000` over several scenarios | The pooled value divides total cash by total calendar time. |
| Episode ends at each renewal | The core's `terminated` signal | The only core objective in 1.0.0. |
| Policy maps: mean ORDER probability by health and damage rate | `maintenance_arena.analysis` (`map_states`, `order_probabilities`, `binned_mean`), `maintenance_arena.bearings.analysis.policy_map` | The `figures.py` runners in `experiments/` compute the article's maps. |
| Failure share during training | `maintenance_arena.analysis.failure_share_curve` | Binned over training steps, from the training logs. |

## What is not shipped

- The FEMTO and XJTU-SY recordings. [docs/BEARINGS.md](BEARINGS.md#data) explains how to download and place them.
- Trained policies. The runners in [experiments/](../experiments/README.md) train them again from the article's seeds.
