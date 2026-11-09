# Benchmark tracks and protocol

A benchmark track fixes everything a result depends on except the agent. It sets the profile, the calendar budget per scenario, the references, the calibrated rules, what agents observe, and the lists of test and calibration seeds. Results on the same track are therefore paired. [Workflows](WORKFLOWS.md#score-your-own-agent-against-the-references) covers the rest of the benchmark layer.

The definitions are package data in `src/maintenance_arena/data/benchmark_tracks.json`. Each track names an unchanged v1 profile and records its `configuration_sha256`, so a track fails to load if its profile has changed.

## The tracks

| Track | Profile and environment | Agents observe | Scenarios (test / calibration) | Calendar steps per scenario | References, primary first |
| --- | --- | --- | ---: | ---: | --- |
| `core-continuous-v1` (default) | `paper-core-continuous-v1`, `CoreCycle-v0` | observation and `info` | 1,000 / 800 | 720 | `oracle`, `reactive`, `age`, `health`, `projection`, `immediate` |
| `core-zero-deadline-v1` | `paper-core-continuous-v1`, `CoreCycle-v0` | observation with index 3 set to zero, and `info` | 1,000 / 800 | 720 | as `core-continuous-v1` |
| `gearbox-v1` | `paper-gearbox-v1`, `Gearbox-v0` | observation and `info` | 200 / 200 | 720 | `dp`, `reactive_normal`, `reactive_derate`, `normal_q96`, `derate_q89`, `separable_q96` |
| `resource-v1` | `paper-resource-v1`, `Resource-v0` | observation and `info` with `info["action_mask"]` | 200 / 200 | 720 (hours) | `rolling`, `reactive`, `health` |
| `resource-lifetime-v1` | `paper-resource-lifetime-v1`, `ResourceLifetime-v0` | observation and `info` with `info["action_mask"]` | 200 / 200 | 720 (hours) | `rolling`, `reactive`, `age`, `health` |

A core scenario is one component from installation to renewal. A cycle lasts at most 166 calendar steps, so the 720-step budget never binds. A gearbox or resource scenario is a 720-step window of a continuing task, starting from the reset state.

On the core tracks, each run first calibrates `age`, `health` and `projection` on the calibration scenarios. It uses the 252 default candidates and the zero-failure criterion of [rule calibration](WORKFLOWS.md#calibrate-the-core-rules). `--criterion pooled` (in Python, `run_track(..., criterion="pooled")`) selects by pooled profit instead. The gearbox and resource rules keep their profile settings.

`core-zero-deadline-v1` is a matched control, the Health + rate input set of the accompanying article. Agents see the deadline coordinate (index 3, `causal_intervention_slack`) as zero, and everything else equals `core-continuous-v1`. The coordinate is a function of health, wear rate and validity flag, so the control removes a ready-made representation of the lead time, not information. No reference uses index 3.

`maintenance-arena tracks` prints every track definition with its `sha256`, without the seed lists.

## Score an agent on a track

From the repository root, after the base install:

```shell
python -m maintenance_arena tracks
python -m maintenance_arena benchmark --track core-continuous-v1 --output outputs/track-core
python -m maintenance_arena benchmark --track core-continuous-v1 --agent examples.benchmark_your_agent:deadline_agent --output outputs/track-core-deadline-agent
python -m maintenance_arena benchmark --track gearbox-v1 --output outputs/track-gearbox
```

`--agent` takes `module:callable` and can be repeated. Because the track fixes the profile, scenarios, budget, references and calibration, the command rejects `--profile`, `--seed`, `--config`, `--calendar-steps`, `--episodes`, `--fixture`, `--reference`, `--checkpoint`, `--filter`, `--calibration`, `--no-recalibrate` and `--calibration-lives` next to `--track`. `--criterion` still selects the calibration criterion.

The command prints the table and writes:

- `benchmark.md`, the table
- `benchmark.json` (schema `maintenance_arena_benchmark_v1`), with the track summary under `track`
- `evaluation.json` with the per-scenario records
- `fixture.json` with the test seeds
- `calibration.json` on the core tracks
- `run.json` with the track name and its `sha256`

Scoring the references takes about 20 seconds on `gearbox-v1` and a little over a minute on a core track. These times were measured on an ordinary Windows PC with CPython 3.12. An agent adds its own inference time.

In Python, any callable `agent(observation, info) -> action` works:

```python
from maintenance_arena import benchmark


def my_agent(observation, info):
    health, wear_rate, rate_valid, deadline = observation
    return int(rate_valid > 0.5 and deadline >= -2 / 13)  # 0 = WAIT, 1 = REPLACE


result = benchmark.run_track("core-continuous-v1", {"my_agent": my_agent})
print(result.to_markdown())
```

`run_track(track, agents)` takes a track name, a tracks JSON file, a `Track` or its JSON form. Agents can be a callable, a `(label, agent)` tuple, a mapping from labels to agents, or a list of such tuples. It returns a `BenchmarkResult` (see [Workflows](WORKFLOWS.md#score-your-own-agent-against-the-references)) with `track` set.

To train, build the track's environment with `maintenance_arena.tracks.make_track(name)`. This is the `gymnasium.make` environment with the profile's configuration, wrapped in `CalendarBudget` and, on `core-zero-deadline-v1`, in `ZeroDeadline` (see [Wrappers](#wrappers)). Seed it with your own seeds.

Two examples train an agent and score it on a track. Training needs the `rl` extra and a few minutes on a CPU, and the budgets only check the workflow.

- [examples/benchmark_your_agent.py](../examples/benchmark_your_agent.py) trains stock Stable-Baselines3 PPO for 50,000 steps on the core and scores it on `core-continuous-v1`. `--training-seeds 0 1 2 3 4` trains five models, and `--track core-zero-deadline-v1` trains without the deadline coordinate. `--steps 0` scores only the hand-written `deadline_agent`, which needs no extra.
- [examples/train_maskable_resource.py](../examples/train_maskable_resource.py) trains sb3-contrib's MaskablePPO with `VecNormalize` for 20,480 steps on `Resource-v0`, with the PPO settings of `paper-resource-v1`. It scores the frozen model on `resource-v1`, with the frozen normalization and `info["action_mask"]` passed to `predict`.

## Reference table of the default track

The command

```shell
python -m maintenance_arena benchmark --track core-continuous-v1 --output outputs/track-core
```

writes the `benchmark.md` below, byte for byte on every run. With `MAINTENANCE_ARENA_FULL_TRACK=1`, `tests/test_tracks.py` recomputes it and compares it with this block. The README repeats the first table and the rule thresholds, and a test checks that they match. These are package results, not results of the accompanying article.

<!-- default-track-table:begin -->
Benchmark track `core-continuous-v1` (sha256 d16bdfc5d882) on `paper-core-continuous-v1`: 1,000 test scenarios, up to 720 calendar steps each.
Primary reference `oracle`: a privileged oracle that knows each component's true lifetime and orders at the last safe order (true RUL 13); an upper bound, not a feasible policy.

| Policy | Kind | Information | Profit per 1,000 calendar steps | % of oracle | Failures per 1,000 lives | Order margin mean (sd) | Late orders |
|---|---|---|---:|---:|---:|---:|---:|
| `oracle` | privileged reference | true lifetime (evaluator side) | 236,517.2 | 100.00 | 0.0 | 0.00 (0.00) | 0.0% (0/1000) |
| `projection` | built-in | observation | 236,336.4 | 99.92 | 0.0 | 1.30 (0.71) | 0.0% (0/1000) |
| `health` | built-in | observation | 232,014.9 | 98.10 | 0.0 | 26.28 (20.61) | 0.0% (0/1000) |
| `age` | built-in | operating age | 230,686.3 | 97.53 | 0.0 | 32.18 (13.87) | 0.0% (0/1000) |
| `reactive` | built-in | none | 184,348.1 | 77.94 | 1,000.0 | n/a | n/a |
| `immediate` | built-in | none | 124,473.7 | 52.63 | 0.0 | 115.18 (13.87) | 0.0% (0/1000) |

Paired differences on the same scenarios in pooled profit per 1,000 calendar steps, with 95% percentile bootstrap intervals over scenarios (2,000 resamples, seed 0):

| Policy | Against | Difference | 95% interval | Scenarios better / tied / worse |
|---|---|---:|---:|---:|
| `reactive` | `oracle` | -52,169.1 | [-52,496.3, -51,835.3] | 0 / 0 / 1000 |
| `age` | `oracle` | -5,831.0 | [-5,949.2, -5,715.0] | 0 / 1 / 999 |
| `health` | `oracle` | -4,502.3 | [-4,771.6, -4,238.6] | 0 / 5 / 995 |
| `projection` | `oracle` | -180.8 | [-187.7, -174.3] | 0 / 104 / 896 |
| `immediate` | `oracle` | -112,043.5 | [-112,161.8, -111,927.6] | 0 / 0 / 1000 |

- Rule thresholds: age 83 (calibrated), health 0.36 (calibrated), projection 15 (calibrated).
- Rules calibrated for track core-continuous-v1: selected by the zero-failure criterion (no failure, then the highest family-equal profit) on its 800 calibration lives, which are disjoint from the test scenarios.
- Intervals resample scenarios and are conditional on each agent as given; they do not include variation across independently trained models.
<!-- default-track-table:end -->

The default criterion selects the profile's own thresholds, and none of the three rules fails on the 1,000 test lives. `--criterion pooled` selects age 87, health 0.34 and projection 15, and the age and health rules then fail on 13 and 7 test lives. `calibration.json` in the output directory holds the full selection table with the quantities of both criteria.

The table pools all lives, while the default criterion weights the four degradation laws equally. Both agree closely except for the health rule, whose loss depends on the degradation law because it ignores the wear rate. The accompanying article weights every life equally.

## Metrics and pooled aggregation

The first table reports for every policy and agent:

- Profit per 1,000 calendar steps: 1,000 times the total cash over the total calendar steps of all test scenarios. This pooled rate is the long-run profit of running the policy scenario after scenario. Cash is `info["raw_profit_sum"]`, not the reward.
- Percentage of the primary reference: `oracle` in the core, `dp` in the gearbox and `rolling` in the resource tasks.
- Failures per 1,000 lives: failures over the lives that ended within the scenarios, at a planned intervention or a failure.
- Order margin (core): the true remaining life at the first order minus 13, the last safe order (12 steps of lead time plus the exchange step). A negative margin is late and ends in a failure, and a positive one discards useful life. The table gives its mean, its standard deviation and the share of late first orders. Agents never see the margin.
- Invalid actions (resource tasks): masked actions, which the environment executes as no action.

`evaluation.json` keeps the per-scenario units and defines every metric in `metric_definitions`. It adds failures per 1,000 calendar steps and the mean of the per-scenario rates (`mean_profit_per_1000_calendar_steps`). That mean weights short and long scenarios alike, so it is not a long-run rate and can rank policies differently.

The second table gives the paired difference in pooled profit of every agent against every reference. Each difference has a 95% percentile bootstrap interval over scenarios (2,000 resamples, seed 0). The table also counts the scenarios in which the agent did better, tied or did worse. Without agents, each reference is compared with the primary reference.

The interval holds the trained agent fixed. It measures how precisely the test scenarios score one model, not how much another training run would differ. That needs several models (see the protocol below).

## What each reference knows

| Task | Policy | Information at decision time |
| --- | --- | --- |
| Core | `oracle` | True lifetime, read by the evaluator through `diagnostics()`. Orders at the last safe order (true remaining life 13). An upper bound, not a feasible policy. |
| Core | `projection` | Health, wear rate and validity flag (observation indices 0 to 2). Orders once the projected remaining life reaches its threshold. |
| Core | `health` | Health (observation index 0). |
| Core | `age` | Operating age since installation, passed through `policy_state()`. An agent can count it from its own decisions. |
| Core | `reactive`, `immediate` | Nothing. They always wait or always order. |
| Gearbox | `dp` | Exact state and model. Dynamic programming for long-run average profit, so a rule can score slightly above it over a 720-step window. |
| Gearbox | `separable_q96`, `normal_q96`, `derate_q89` | Exact state: remaining damage capacity and revenue opportunity. |
| Gearbox | `reactive_normal`, `reactive_derate` | Nothing. They run to failure. |
| Resource tasks | `rolling` | Observable state, action mask and known configuration. Plans each component's next intervention over 48 hours with a surrogate wear forecast. |
| Resource tasks | `health`, `age`, `reactive` | Observable state and action mask. `age` reads the operating ages that `ResourceLifetime-v0` observes. |
| Every task | an agent | Observation and public `info` of each decision, plus its own memory. |

The Information column of the benchmark table gives this in short form. The protocol below names the rule to compare an agent with.

## Rules for agents

The harness calls `agent(observation, info)` with a copy of the observation and a deep copy of the public `info`, and nothing else. An agent must not obtain the environment, `diagnostics()`, `snapshot()` or `policy_state()` in other ways. The agent runs in the evaluator's Python process, so the harness cannot stop deliberate introspection such as walking the call stack. A result can therefore only be checked together with the agent's code. In particular, an agent must not use:

- the true remaining useful life, lifetime or wear family of a component
- the simulator state or private attributes of the environment
- future information, such as future degradation or the realized delivery times of the resource tasks
- the track's test seeds, for training, tuning or selection

An agent may keep memory within a scenario, such as its past observations. Each scenario starts with the `info` of `reset`, in which `duration_calendar_steps` is 0, so a stateful agent resets its memory there.

## Recommended protocol

1. Train at least five models per agent with independent training seeds and otherwise identical settings and budgets. Score and report all of them, including poor ones.
2. Tune hyperparameters and select checkpoints on the calibration scenarios or on your own seeds, never on the test seeds. The test seeds are listed for pairing, not for inspection.
3. Report each model's paired difference and interval against the strongest calibrated rule with the same information: `projection`, the Time rule of the article, on the core tracks, `rolling` on the resource tracks, and `separable_q96` and `dp` on the gearbox track. The score is always calendar profit, cash per calendar step. If you trained on another objective, such as a discounted, episodic or shaped return, say so.
4. Summarize over the models, for example with the mean and range of the paired difference. The package's intervals cover one fixed model. The package provides no summary across training seeds, such as a two-level bootstrap over models and scenarios.
5. Report privileged information used in training, even if the trained agent sees only observations, and keep such results separate. The package's own core PPO recipe (`maintenance-arena train`) is an example. Its training bank starts six of every eight episodes 29, 21, 17, 15, 14 or 13 steps before the true end of life (`START_OFFSETS` in `maintenance_arena.curves`). Starts set from the lifetime with `reset(options={"start_age": n})`, rewards shaped with the true remaining life, critics or teachers that read `diagnostics()` and oracle demonstrations count as well.
6. Report the training budget in environment and calendar steps, the hardware, the wall-clock time, the library versions and the agent's code. `run.json` records the package and library versions, the implementation hash and the hash of every `--agent` module.

## Wrappers

The tracks use two wrappers from `maintenance_arena.wrappers`. Neither changes the wrapped environment, its configuration, random streams, rewards or `info`, so profile identities and checkpoint bindings stay valid.

- `ZeroDeadline(env)` sets observation index 3 of `CoreCycle-v0` to zero and keeps the observation space. It rejects other environments. `zero_deadline(observation)` does the same for one observation or a batch.
- `CalendarBudget(env, calendar_steps)` truncates an episode once the summed `info["duration_calendar_steps"]` reaches the budget. Gymnasium's `TimeLimit` counts decisions instead, but one gearbox replacement decision lasts up to 19 calendar steps. The transition that reaches the budget completes first, as in the benchmark's evaluation loop, and the environment's own termination takes precedence. After the end, `step` raises an error until `reset`. `elapsed_calendar_steps` gives the time since the last reset.

`make_track` applies both as the track requires. The module also provides `InputSet(env, name)` with the article's input sets `full`, `deadline_only` and `health_rate`, which no track uses.

## Seeds and integrity

All five tracks use the root seed 271828. NumPy's `SeedSequence` derives both seed lists from it:

- Test scenario `i` (from 0 to `n - 1`) has the seed `SeedSequence([271828, i]).generate_state(1, dtype=uint32)[0]`. These are the scenarios that `evaluate` and `benchmark` derive from `--seed 271828 --episodes n`. The gearbox and resource tracks thus use the first 200 seeds of the core tracks.
- Calibration scenario `i` has the seed `SeedSequence(271828, spawn_key=(1, i)).generate_state(1, dtype=uint32)[0]`, the stream of `maintenance_arena.calibration.calibration_seeds`. Test seeds and repeated seeds are skipped, so the two lists are disjoint.

The track file stores both lists, so a track keeps its scenarios even if a later NumPy release derives other numbers. `python tools/make_benchmark_tracks.py --check` derives all lists and profile hashes again and compares them with the packaged file. In Python:

```python
from maintenance_arena.tracks import load_track, track_seeds

track = load_track("core-continuous-v1")
test, calibration = track_seeds(track.root_seed, track.test_scenarios, track.calibration_scenarios)
assert tuple(test) == track.test_seeds and tuple(calibration) == track.calibration_seeds
print(track.sha256)
```

A track's `sha256` covers its whole definition, including both seed lists. It appears in the table header, `benchmark.json` and `run.json`. Results with the same track hash were scored on the same scenarios and references.

## Scoring outside the tracks

`benchmark --profile ... --seed ...` and `benchmark.score(agent, profile=..., seed=..., episodes=...)` score agents on scenarios derived from a seed, with the same table. Such results are paired only with runs on the same fixture. A configuration (`--config`, or a profile resolved with overrides) that changes a core environment first triggers a recalibration of the core rules. In the core, keep `--calendar-steps` at 166 or more, the longest cycle in `paper-core-continuous-v1`, or the oracle is no longer an upper bound. [Workflows](WORKFLOWS.md#score-your-own-agent-against-the-references) gives the commands, and [Calibrate the core rules](WORKFLOWS.md#calibrate-the-core-rules) the calibration options.

## Custom tracks

`maintenance_arena.tracks.define_track(name, profile, root_seed=..., test_scenarios=..., calibration_scenarios=...)` defines a track on a packaged profile and derives both seed lists as above. The optional arguments are `calendar_steps`, `references`, `calibrated_rules`, `agent_observation` and `description`. `write_tracks(path, [track])` writes a tracks file for `benchmark --track path/to/tracks.json` or `run_track("path/to/tracks.json")`. A file with several tracks uses its `default_track`. A custom track is a different benchmark, so publish its file with the results.

## Submit results

There is no leaderboard yet, and the maintainers have not verified any submitted result. To share one, open an issue with the template "Benchmark result" (`.github/ISSUE_TEMPLATE/benchmark_result.md`). It asks for:

- the track and its `sha256`, the package version and the implementation hash
- the agent, its training and any privileged information used
- `benchmark.md` of every model and the paired differences against the strongest rule with the same information
- links to the code
