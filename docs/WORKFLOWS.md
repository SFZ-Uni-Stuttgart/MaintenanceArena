# Workflows and reproducibility

Run these commands from the project directory after installation. Any new directory can replace the output paths. Simulation and evaluation need only the base package, and training also needs the `rl` extra. [Benchmark tracks and protocol](BENCHMARK.md) describes the fixed tracks that `tracks` lists.

## Simulate each task

Each command writes the trajectory, the decisions, the events and the run identity to a new output directory.

```shell
python -m maintenance_arena simulate --profile paper-core-continuous-v1 --policy projection --seed 12001 --calendar-steps 240 --output outputs/core-example
python -m maintenance_arena simulate --profile paper-gearbox-v1 --policy separable_q96 --seed 12002 --calendar-steps 720 --output outputs/gearbox-example
python -m maintenance_arena simulate --profile paper-resource-v1 --policy rolling --seed 12003 --calendar-steps 720 --output outputs/resource-example
python -m maintenance_arena simulate --profile paper-resource-lifetime-v1 --policy age --seed 12004 --calendar-steps 720 --output outputs/resource-lifetime-example
```

With the `plots` extra (`.[plots]`), `--plot` also draws the trajectory. `python -m maintenance_arena list` prints every profile with its environment and policy names. The Python example runs every profile with a suitable conventional policy, or the profile and policy you choose:

```shell
python examples/run_environment.py
python examples/run_environment.py --profile paper-resource-lifetime-v1 --policy age --seed 12004 --calendar-steps 720
```

Core episodes end at the renewal. A gearbox transition can overrun the requested budget, because a chosen physical event always completes. Each resource action takes one hour. Read `actual_calendar_steps`, the termination and truncation flags and any overshoot instead of assuming the requested duration.

## Compare policies on paired synthetic units

```shell
python -m maintenance_arena evaluate --profile paper-core-continuous-v1 --policy reactive age health projection --seed 12100 --episodes 32 --calendar-steps 240 --output outputs/core-rules
python -m maintenance_arena evaluate --profile paper-gearbox-v1 --policy reactive_normal separable_q96 dp --seed 12101 --episodes 16 --calendar-steps 720 --output outputs/gearbox-rules
python -m maintenance_arena evaluate --profile paper-resource-v1 --policy reactive health rolling --seed 12102 --episodes 16 --calendar-steps 720 --output outputs/resource-rules
python -m maintenance_arena evaluate --profile paper-resource-lifetime-v1 --policy reactive age health rolling --seed 12103 --episodes 16 --calendar-steps 720 --output outputs/lifetime-rules
```

`fixture.json` binds distinct scenario seeds to the base profile, the environment configuration (including the observation route and the effective resource horizon) and the calendar budget. Replay it with `--fixture outputs/core-rules/fixture.json` to keep the units paired. Its seeds then define the scenarios, so `evaluate` ignores `--seed` and `--episodes`, and `evaluate(..., fixture=...)` in Python needs no `seed`. Thresholds, trained policies and gearbox filtering may change between runs on one fixture, but a new environment or budget needs a new fixture. Duplicate policy names and repeated seeds are rejected, because they would inflate the number of independent scenarios.

The schemas are `maintenance_arena_synthetic_fixture_v2` and `maintenance_arena_synthetic_evaluation_v3`. Older v1 fixtures lack the environment and budget bindings and are rejected with an explanation. Keep their results, and create a v2 fixture for a new comparison rather than relabeling an old record.

`evaluation.json` keeps the unit records, states its aggregation and defines each added metric in `metric_definitions`. `mean_profit_per_1000_calendar_steps` is the mean of the per-scenario rates in CU per 1,000 calendar steps, as in v2. `pooled_profit_per_1000_calendar_steps` is the total cash over the total calendar steps. Schema v3 keeps every v2 field and its meaning and adds:

- to every summary: `policy_kind`, the pooled profit, `ended_lives` (planned interventions plus failures), `failures_per_1000_lives` and `failures_per_1000_calendar_steps`
- to core summaries: `orders_placed`, `late_orders`, `late_order_share`, `order_margin_mean` and `order_margin_sd`
- to resource summaries: `invalid_actions`
- to every unit: `policy_kind`, `ended_lives`, `order_margin` and `invalid_actions`

With `--no-events`, units store `events` as `null`. New simulator seeds are not additional trained models, and neither aggregate is the equal-life aggregation of the accompanying article. [Environment contracts](ENVIRONMENTS.md#units) explains the units, and the [glossary](GLOSSARY.md#objectives-units-and-outputs) relates them to the article.

Report failures, renewals, timing, operating choices and filtered interventions together with the economics. Resource comparisons also need availability, inventories, unfinished work, backlog, and the difference between cash and consumed-part accounting.

## Score your own agent against the references

An agent is any callable `agent(observation, info) -> action` that returns one integer action. A NumPy integer or a one-element integer array, as Stable-Baselines3's `predict` returns, also works. The agent receives only copies of the observation and the public `info` (see [Rules for agents](BENCHMARK.md#rules-for-agents)). Resource agents find the action mask in `info["action_mask"]`. A masked action runs as no action and counts in `invalid_actions`.

`evaluate --agent module:callable` adds agents to a paired comparison of built-in policies. The module is imported with the current directory on the import path. The label is the callable's name unless `--agent-label` sets one, and labels that name a built-in policy or `oracle` are rejected. `run.json` records the module file and its hash.

```shell
python -m maintenance_arena evaluate --profile paper-core-continuous-v1 --policy projection --agent examples.benchmark_your_agent:deadline_agent --seed 12410 --episodes 32 --no-events --output outputs/agent-vs-projection
python -m maintenance_arena benchmark --track core-continuous-v1 --agent examples.benchmark_your_agent:deadline_agent --output outputs/workflow-track-core
python -m maintenance_arena benchmark --profile paper-core-continuous-v1 --agent examples.benchmark_your_agent:deadline_agent --seed 12400 --episodes 100 --output outputs/workflow-benchmark-core
python -m maintenance_arena benchmark --profile paper-gearbox-v1 --seed 12401 --episodes 100 --output outputs/benchmark-gearbox
```

`benchmark` runs the agents and the task's references on identical scenarios and prints a Markdown table. It writes `benchmark.md`, `benchmark.json` (schema `maintenance_arena_benchmark_v1`), `evaluation.json` without event lists, `fixture.json` and `run.json`. With `--track`, a fixed track sets the scenarios, references and rule calibration (see [Score an agent on a track](BENCHMARK.md#score-an-agent-on-a-track)). Without a track, the scenarios come from `--seed`, and the default references are the primary reference and every built-in policy of the profile except `ppo`. The primary reference is always included, and `--reference ppo` needs `--checkpoint`.

The primary references are `oracle` in the core, `dp` in the gearbox and `rolling` in the resource tasks (see [What each reference knows](BENCHMARK.md#what-each-reference-knows)). The oracle belongs to no profile, so `simulate` and `evaluate` reject it as a policy. It bounds the other policies only over complete cycles. With `--calendar-steps` below the longest cycle (166 calendar steps in `paper-core-continuous-v1`), a waiting policy is cut off before its renewal and can beat the oracle. The table then says so in a note.

[Metrics and pooled aggregation](BENCHMARK.md#metrics-and-pooled-aggregation) explains the two tables. Without a track, the rules `age`, `health` and `projection` use the profile's `policy_defaults`. If `--config` changes the environment of a core profile, `benchmark` first recalibrates them, as the next section describes. The table notes list each threshold and its source.

In Python, `maintenance_arena.benchmark.run_track(track, agents)` scores on a track, and `maintenance_arena.benchmark.score(agent, profile=..., seed=..., episodes=...)` scores on derived scenarios. Both take agents as a callable, a `(label, agent)` tuple, a mapping from labels to agents, or a list of such tuples. Both return a `BenchmarkResult`, whose `to_markdown()` gives the table and `to_dict()` the content of `benchmark.json`. `score` takes `seeds=` for a given seed list, and `recalibrate=`, `calibration=`, `calibration_lives=` and `criterion=` for the rule calibration. `run_track` also takes `criterion=`. `maintenance_arena.evaluation.evaluate(profile, policies=[...], agents={...}, ...)` is the lower-level paired evaluation, and `maintenance_arena.evaluation.make_fixture(profile, seeds, calendar_steps=...)` binds a seed list to a profile.

## Calibrate the core rules

The best threshold of a rule depends on the task. The core profiles fix thresholds selected for their published configuration: age 83, health 0.36 and projection 15 (14 in the discrete profile). `calibrate` selects the thresholds of `age`, `health` and `projection` again by grid search on calibration lives, for the published configuration or one changed with `--config`.

```shell
python -m maintenance_arena calibrate --profile paper-core-continuous-v1 --seed 20261002 --lives 800 --output outputs/calibration-core
python -m maintenance_arena calibrate --profile paper-core-continuous-v1 --config examples/noisy_core.yaml --seed 12500 --output outputs/calibration-noisy
python -m maintenance_arena benchmark --profile paper-core-continuous-v1 --config examples/noisy_core.yaml --calibration outputs/calibration-noisy/calibration.json --agent examples.benchmark_your_agent:deadline_agent --seed 12501 --episodes 100 --output outputs/benchmark-noisy
python -m maintenance_arena evaluate --profile paper-core-continuous-v1 --config outputs/calibration-noisy/overrides.json --policy age health projection --seed 12502 --episodes 32 --output outputs/noisy-calibrated-rules
```

The default grids hold 252 candidates: ages 1 to 159, health thresholds 0.20 to 0.95 in steps of 0.01, and projection thresholds 8 to 24. `--age-grid`, `--health-grid` and `--projection-grid` replace a grid with numbers and inclusive ranges, such as `60:120` or `0.30:0.40:0.02`. `--rule` restricts the calibration to some rules.

`--criterion` selects among the candidates. The default, `zero_failure`, proceeds in three steps:

1. A candidate is eligible only if it causes no failure on the calibration lives and orders at least one planned exchange there.
2. The eligible candidate with the highest family-equal profit wins. This is the mean of the pooled profits (1,000 times cash over calendar steps) of the four wear families `linear_reference`, `concave_running_in`, `convex_accelerating_fatigue` and `bearing_late_knee`. Each family has equal weight, however many lives it has.
3. Ties go to the lower mean normalized discarded life. This is the operating life left unused at a planned exchange divided by the lifetime, averaged over the planned exchanges. Remaining ties go to the later-acting threshold, the higher age or the lower health or projection threshold.

Only the evaluator reads the wear family and the lifetime. If no candidate of a rule is eligible, for example because all of them order too late, the command stops. Its error names the rule and the reason. Widen the grid or use `--criterion pooled`. If a small bank lacks one of the random wear families, the family-equal profit averages the families present and the command warns.

`--criterion pooled` ranks the candidates by pooled profit per 1,000 calendar steps over all calibration lives, the metric of the `benchmark` table. It accepts failures when they pay. Ties go to fewer failures, then to the later-acting threshold.

Calibration lives have their own seed stream. Life `i` of the root `--seed` uses `SeedSequence(seed, spawn_key=(1, i))`, while `evaluate` and `benchmark` derive test scenarios as `SeedSequence([seed, i])`. `--exclude-fixture` takes the `fixture.json` of a test run and can be repeated. Calibration seeds that are also test seeds are skipped, so the two sets are disjoint. `calibration.json` lists every seed and every skipped one.

The three rules wait until their condition first holds and then order. `calibrate` therefore runs each life once under WAIT and computes the cash of every threshold from its first order. It then reruns the selected thresholds through the full simulator on every calibration life and stops with an error if the results differ.

The command prints a summary table and writes `calibration.json`, `calibration.md`, `overrides.json` and `run.json`. `calibration.json` (schema `maintenance_arena_rule_calibration_v1`) holds:

- the seeds and the number of lives per wear family (`families`)
- the grids and the criterion (`selection_criterion`, defined in `criterion` and `tie_break`)
- the selected thresholds and one row per candidate
- the comparison with the profile's thresholds
- a `sha256` over the record

Each row gives the pooled and mean profit, failures, order-margin statistics, the zero-failure quantities (`family_profit_per_1000_calendar_steps`, `family_equal_profit_per_1000_calendar_steps`, `mean_normalized_discarded_life_over_planned`, `zero_failure_eligible`) and the rank under the selected criterion. One table thus serves both criteria. The comparison gives the paired difference in pooled profit between the selected and the profile's threshold, with a 95% percentile bootstrap interval over the calibration lives. It also gives the difference in family-equal profit. If a selected threshold is the smallest or largest value of its grid, the command prints and records a warning. The optimum may then lie outside the grid, and a rule stopped at the edge can flatter a learner.

The first two commands, and the same commands with `--criterion pooled`, select these thresholds on their 800 lives. Differences are in pooled profit per 1,000 calendar steps against the profile's threshold, with 95% intervals.

| Configuration | Criterion | Age | Health | Projection | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| published | `zero_failure` | 83 | 0.36 | 15 | The profile's thresholds. None of them fails. |
| published | `pooled` | 83 | 0.35 | 15 | Health 0.35 fails on 2 lives. It earns 221.0 more than 0.36, but the interval [-28.6, +390.5] includes zero. |
| `examples/noisy_core.yaml` | `zero_failure` | 83 | 0.37 | 18 | Under light noise with the robust reconstruction, projection 15 fails on 26 lives and health 0.36 on one. Projection 18 earns 1,190.2 more than 15 [+519.0, +1,863.5]. |
| `examples/noisy_core.yaml` | `pooled` | 86 | 0.34 | 17 | These fail on 4, 5 and 2 lives. |

`overrides.json` holds the changed `env_config` keys, any `training` section of the `--config` file and the selected thresholds as `policy_defaults`. Passed to `--config`, it runs the calibrated rules as ordinary built-in policies, as in the fourth command.

`benchmark` uses a calibration in one of three ways:

- `--calibration` applies a stored `calibration.json`. Its environment configuration and calendar budget must match the benchmark, and none of its lives may be a test scenario. A given `--criterion` must match the record.
- Without `--calibration`, a `--config` that changes the environment of a core profile makes `benchmark` recalibrate the rules among its references. It uses `--calibration-lives` lives (default 800) from `--seed` on the calibration stream, disjoint from the test scenarios, and the criterion of `--criterion` (default `zero_failure`). A rule whose threshold the configuration sets in `policy_defaults` keeps it.
- `--no-recalibrate` keeps the profile's `policy_defaults`. It excludes `--criterion` and `--calibration`.

An applied calibration is saved as `calibration.json` in the output directory, and `benchmark.json` records its hash and `selection_criterion`. The table notes name the criterion. Only the core rules are calibrated.

In Python, `maintenance_arena.calibration.calibrate(profile, criterion=..., seed=..., lives=..., exclude_seeds=..., grids=...)` returns a `CalibrationResult` with `selected`, `criterion`, `families`, `table`, `policy_defaults()`, `overrides()`, `to_dict()` and `to_markdown()`. `load_calibration(path)` reads a stored record. `CRITERIA` lists the criteria, and `CRITERION_TEXTS` and `TIE_BREAK_TEXTS` define them. Without an eligible candidate, a zero-failure calibration raises `NoEligibleThresholdError`, a `ValueError`. The warnings `GridEdgeWarning` and `FamilyCoverageWarning` are subclasses of `CalibrationWarning`. `calibration_seeds(seed, lives, exclude=...)` gives the seeds of a calibration bank, and `default_grids()` the 252 default candidates.

## Noisy core observations and explicit overrides

[noisy_core.yaml](../examples/noisy_core.yaml) selects light sensor noise and the robust causal reconstruction. The economics and the replacement event clock stay unchanged:

```shell
python -m maintenance_arena simulate --profile paper-core-continuous-v1 --config examples/noisy_core.yaml --policy projection --seed 12110 --calendar-steps 240 --output outputs/noisy-core
```

Overrides in JSON or YAML use the top-level sections `env_config`, `training` and `policy_defaults`. The run records the resolved configuration, its hash, the base profile and a modified-profile identity. An override defines a new configuration, not the one behind a published result, and the modified profile states that it inherits no published validation.

Rollouts record the effective environment configuration and separate the requested budget from an environment's own shorter cutoff. `budget_overshoot` refers to the requested budget and `environment_budget_overshoot` to the environment cutoff. A started transition completes under either cutoff, and termination stays distinct from truncation.

The core supports the reconstructions `RAW_CAUSAL`, `FILTERED_CAUSAL_EWLS8` and `ROBUST_MULTISCALE_234` and the sensor presets `exact`, `gaussian_005`, `light`, `moderate` and `stress`. `gaussian_005` is white measurement noise with standard deviation 0.005, without bias or missing readings, as in the article's measurement-noise study. [Environment contracts](ENVIRONMENTS.md#simulated-observation-uncertainty) describes the assumptions. A new observation route neither refits a checkpoint nor demonstrates sensor transfer.

## Train and continue native runs

Training needs the `rl` extra and runs on the CPU with one PyTorch thread per process. The commands below request two updates of the default recipes, enough to exercise collection, optimization and saving. They are software checks. The accompanying article lists its research budgets in Appendix A.

```shell
python -m maintenance_arena train --profile paper-core-continuous-v1 --seed 12200 --calendar-steps 2048 --output outputs/train-core
python -m maintenance_arena train --profile paper-gearbox-v1 --seed 12201 --calendar-steps 2048 --output outputs/train-gearbox
python -m maintenance_arena train --profile paper-resource-v1 --seed 12202 --calendar-steps 4096 --output outputs/train-resource
python -m maintenance_arena train --profile paper-resource-lifetime-v1 --seed 12203 --calendar-steps 4096 --output outputs/train-lifetime
```

Core and gearbox training use a custom PPO collector that discounts per actual calendar step. A chosen exchange transition always completes. The core keeps its episodic and the gearbox its continuing bootstrap contract. Gearbox training disables the execution filter and uses the fixed economic value prior.

Resource training uses MaskablePPO with eight synchronous environments, 256 steps per environment and rollout, and the paired training normalizer. Its calendar budget must be a multiple of `n_envs * n_steps`, which is 8 x 256 = 2,048 with the default recipe. Other values are rejected up front. The recipes sample four known contexts: nominal, slow supply, doubled part cost and continuous staffing. Lifetime training applies the same recipe to the age-augmented task.

The training result reports the native checkpoint as `checkpoint` (absolute path) and `checkpoint_relative` (relative to the output directory). Core and gearbox runs write `final.pt` and `training.json`, and resource runs write `checkpoint.json`. Resume from that checkpoint:

```shell
python -m maintenance_arena train --profile paper-resource-v1 --seed 12202 --calendar-steps 8192 --resume outputs/train-resource/checkpoint.json --output outputs/train-resource-continued
```

`--calendar-steps` is the cumulative target, not an additional budget, and the output must be a new directory. Keep the profile, seed, environment and training configuration, including any `--config`. Continuation rejects incompatible overrides and runtime contracts.

A native resource checkpoint holds:

- the model and the optimizer
- the last observations and episode-start flags
- the observation and reward statistics and the discounted-return accumulators
- the physical states of all vector environments
- the context and fleet random generators with their reset counters
- pending monitor rewards
- the Python, NumPy and PyTorch random states

It resumes only at completed optimizer updates. This state is serialized Python and can run code when loaded. Load only checkpoints that you created or that come from a trusted source (see [SECURITY.md](../SECURITY.md)).

### Explicitly initialize a new run from saved weights

`--initialize` starts a new run from saved weights, while `--resume` continues a native run, and the two exclude each other. An initialized run gets the new seed and a fresh optimizer, collector, exposure count, environment and random streams. Its calendar budget starts from zero. The run records the source, so a warm start cannot pass as training from scratch.

For resource policies, initialization takes the same paired manifest as inference. It copies the actor and critic weights with their observation and reward normalization statistics, so the inputs stay consistent. It resets pending discounted returns, enables normalization updates for the new data, and discards the optimizer state and the old exposure. This example starts a new run from the checkpoint above:

```shell
python -m maintenance_arena train --profile paper-resource-v1 --seed 12210 --calendar-steps 4096 --initialize outputs/train-resource/checkpoint.json --output outputs/resource-initialized
```

A bare `model.zip` or an unrelated normalizer is not enough. Profile, observation order and normalization contract must match, so a 55-input resource actor cannot initialize the 58-input resource-lifetime task. Core and gearbox initialization also needs compatible weights and keeps the declared critic-centering coordinates. Weight-only checkpoints lack the optimizer, rollout, environment and random state. They support initialization and frozen inference, but not an exact resumption.

### Train the comparison learners

Appendix A of the article compares PPO with DQN and double DQN. `maintenance_arena.learning.learners` trains all three on a scenario bank with the recipes of the article. The `train` command trains only PPO.

```python
from pathlib import Path

from maintenance_arena.banks import build_bank
from maintenance_arena.learning.learners import greedy_first_orders, load_policy, make_trainer
from maintenance_arena.replay import decision_series

Path("outputs/learners").mkdir(parents=True, exist_ok=True)
bank = build_bank("source", split="source_train", seed=1, pair_count=512)
trainer = make_trainer("DQN", bank, 7, total_calendar_steps=8192,
                       env_config={"recovery_contract": "preserve_preparation_v1"})
trainer.train_until(8192)
trainer.save("outputs/learners/dqn.pt")
trainer.close()

lives = decision_series(build_bank("source", split="development", seed=2, pair_count=64))
orders = greedy_first_orders(load_policy("outputs/learners/dqn.pt"), lives)  # -1 means no order
```

`make_trainer` takes `PPO`, `DQN` or `DDQN`, and `recipe(name)` returns the fixed settings of a learner. DQN and double DQN explore less and less over the first half of `total_calendar_steps`, so keep this budget when a run stops at intermediate milestones. A checkpoint holds the weights, the optimizer, the replay memory, the environment and the random streams, and `make_trainer(..., resume=path)` continues the run exactly. `greedy_first_orders` replays a saved policy on all lives at once. `experiments/synthetic/learner_comparison.py` runs the comparison of the article.

## Evaluate a saved policy

The package never downloads, selects or trains a PPO checkpoint by itself. Supply one, for example from the training section above:

```shell
python -m maintenance_arena evaluate --profile paper-resource-v1 --policy ppo rolling --seed 12300 --episodes 16 --calendar-steps 720 --checkpoint outputs/train-resource/checkpoint.json --output outputs/resource-trained-comparison
```

Resource checkpoints need a manifest that binds `model.zip` to the matching `normalizer.pkl`. The loader freezes the observation and reward normalization and checks the observation order and the action space. A normalizer from another seed or checkpoint cannot be substituted. Resource checkpoints have 55 inputs, and native resource-lifetime checkpoints 58. Core and gearbox weights support frozen inference when architecture, recipe and hash identity match.

For the gearbox, evaluate raw proposals and filtered execution separately by running the same fixture with and without `--filter`. The flag changes the executed system, not the actor or its training history. Correction counts and operating decisions stay visible. Freedom from failure under the filter holds only for the exact mechanical model.

## Run records and development checks

`run.json` records one execution: the command, the resolved profile and configuration, the source identity, the seed, the input hashes, the runtime, the nominal budget, the result or error, and the elapsed time. Keep it with the results.

The command line checks its arguments before it creates the output directory. It checks the names of profiles, policies and references, `--agent` imports and labels, calibration grids and stored calibrations, input paths, checkpoint directories, optional dependencies, environment overrides and the resource training multiple. A rejected command leaves no directory, so you can rerun it with the same `--output`. An existing `--output` is refused, because outputs are never overwritten. A run that fails after it has started keeps its `run.json`, and a retry needs a new directory.

JSON records are written atomically, so an interrupted update leaves the previous record intact. Ctrl+C returns exit status 130 and records `interrupted`, and completed checkpoints remain. Checkpoint sidecars and model files refuse overwrites. Artifact manifests use relative POSIX paths inside their directory, so the same identities hold on Windows and Linux. Loading or training resource policies restores the caller's Python, NumPy and PyTorch random states and the PyTorch thread setting, also after failures.

In the pinned development environment of [Contributing](../CONTRIBUTING.md), run the portable suite:

```shell
python -m pytest tests -q
```

It covers mechanical boundaries, accounting, causal observation access, saved-policy contracts and native continuation, with small synthetic training budgets. For a scientific experiment, fix the question, the allowed information, the baselines, the independent units, the selection rule and the compute budget in advance. Selected conventional policies remain valid competitors, even when they win.

[Extending MaintenanceArena](EXTENDING.md) covers configuration identities, the extension points and the changes that need a new environment or method contract.
