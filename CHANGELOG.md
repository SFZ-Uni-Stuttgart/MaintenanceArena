# Changelog

Notable changes to MaintenanceArena, newest first.

## 1.0.0 (2026-11-07)

First public release, together with the article *Learning When to Act: What Reinforcement
Learning Agents Learn about Maintenance Timing and When It Transfers*.

### Environments

- Four Gymnasium environments, registered on `import maintenance_arena`:
  `maintenanceArena/CoreCycle-v0` (WAIT or REPLACE for one component until its renewal),
  `maintenanceArena/Gearbox-v0` (NORMAL or DERATE operation with continuing renewal),
  `maintenanceArena/Resource-v0` (maintenance jobs with spare parts, crews and shifts for three
  assets) and `maintenanceArena/ResourceLifetime-v0` (the resource task with observed operating
  age).
- Five frozen profiles: `paper-core-continuous-v1`, `paper-core-discrete-v1`,
  `paper-gearbox-v1`, `paper-resource-v1` and `paper-resource-lifetime-v1`. Each binds a task
  configuration, the settings of its baselines and a PPO recipe under a configuration hash, which
  `tests/test_identity_freeze.py` pins.
- Preparation times of 6, 12 and 24 steps (`mobilization_lead`). The rate channel and the
  deadline coordinate scale with the preparation time.
- Eight degradation laws (`family`): the source laws Linear, Concave, Convex and Late-knee, and
  the new laws Paris-inspired, Two-knee, Gompertz and Power-law with the shape ranges of
  Appendix B of the article. `random` draws one of the four source laws.
- Two recoveries after a failure. The default `immediate_exchange_v1` exchanges at once.
  `preserve_preparation_v1`, used in the article, stops production and keeps the completed
  preparation. The rest of the preparation, the exchange and the six downtime steps follow.
  Profiles and benchmark tracks keep the default, and rule calibration uses the selected
  recovery.
- Simulated measurement noise with the sensor levels `exact`, `gaussian_005`, `light`,
  `moderate` and `stress`, and the causal reconstructions `RAW_CAUSAL`, `FILTERED_CAUSAL_EWLS8`
  and `ROBUST_MULTISCALE_234`.
- The input sets Full inputs, Deadline only and Health + rate (`wrappers.InputSet`), and the
  wrappers `ZeroDeadline` and `CalendarBudget`.
- Scenario banks (`maintenance_arena.banks`): training, calibration and test lives rebuilt from a
  law set, split, seed and size, and `ScenarioBankEnv`, which runs the core task on a bank.

### Bearing replay

- `maintenance_arena.bearings` reads the FEMTO and XJTU-SY run-to-failure recordings. Its layout
  check also compares the number of acquisitions of every bearing with the article, so an
  incomplete download is reported.
- It computes vibration statistics and causal features, fits the remaining-time prediction model
  with nested cross-fitting, and replays every bearing in `BearingReplayEnv` with the prices of
  the core.
- Bearing PPO with a running profit estimate, the Time rule on the same prediction, the
  Indicator rule with a search over sixteen rule families, input designs with and without
  operating age, a gradient-boosted risk classifier on the inputs of Full and No age
  (`calibrate_risk_classifier`, `risk_classifier_orders`), equal-weight metrics with bootstrap
  intervals, and the policy maps and order timing of the article.
- The optional `bearings` extra installs pandas and scikit-learn.

### Policies, learning and evaluation

- Conventional rules for every task, exact average-profit dynamic programming for the gearbox
  with an optional execution filter, a 48-hour rolling-horizon planner for the resource tasks,
  and an oracle for the core that only the evaluator uses.
- Duration-aware PPO for core and gearbox and MaskablePPO for the resource tasks, with exact
  resumption (`--resume`) and initialization from saved weights (`--initialize`).
  `learning.collector.BankTrainer` trains on scenario banks with the settings of Table A1 of the
  article, with milestones and fine-tuning. `learning.dual` trains policies on two state
  estimates at once.
- `learning.learners` adds the two learners that Appendix A of the article compares with PPO,
  DQN and double DQN, each with one fixed recipe. They train on scenario banks like PPO, with
  milestones and exact resumption, and `greedy_first_orders` replays all three.
- `maintenance_arena.replay` computes first orders on whole banks and the calibrations of the
  article and of the additional tables: the Time rule on projected remaining time, health rules,
  age and health cutoffs, a two-indicator rule, per-law rules, failure-budget selections and
  calibrated thresholds on the ORDER probability of PPO.
- `maintenance_arena.metrics` computes failure share, unused life and retained profit with equal
  weight per life, and paired bootstrap intervals that resample whole lives.
  `maintenance_arena.selection` holds the screening and selection of the noise study, and
  `maintenance_arena.analysis` the policy maps and the data of the figures.
- `evaluate` compares built-in policies and user agents on identical scenarios and writes a
  fixture that replays them (schemas `maintenance_arena_synthetic_fixture_v2` and
  `maintenance_arena_synthetic_evaluation_v3`).
- `benchmark` scores agents against the references of a task with paired bootstrap intervals,
  on five fixed tracks (`core-continuous-v1`, `core-zero-deadline-v1`, `gearbox-v1`,
  `resource-v1`, `resource-lifetime-v1`) or on scenarios derived from a seed.
- `calibrate` selects the thresholds of the core rules on calibration lives that are disjoint
  from the test scenarios. By default it chooses among thresholds without a failure
  (`zero_failure`), and `pooled` selects by pooled profit. It uses the configured preparation
  time, and so does the recalibration in `benchmark`.
- Every run writes a `run.json` with the command, the configuration and its hash, the
  implementation hash, seeds, input hashes, runtime and status. Outputs are not overwritten.
- The command line `maintenance-arena`, also `python -m maintenance_arena`, with the commands
  `list`, `tracks`, `simulate`, `evaluate`, `benchmark`, `calibrate` and `train`.

### The article

- `experiments/synthetic/`: runners of the source learning, the input sets, the preparation
  times, the adaptation to the new laws, the measurement-noise study with the selection of its
  design, the failure-cost study, the controlled test of the age input, the learner comparison,
  the additional calibration comparisons and the figure data.
- `experiments/bearings/`: runners of the FEMTO cross-validation, the transfer to XJTU-SY, the
  age designs, the Time rule, the risk classifier, the candidate selection with its nested check
  and the rule search, with summaries and figure data.
- `paper/`: the saved outcomes behind the five data tables of the article and four additional
  tables, their configurations, seeds, folds and selections, and `rebuild_tables.py`, which
  rebuilds the tables byte for byte. `order_margins.py` and `transfer_split.py` compute the order
  margins and the age effect after transfer that the text of the article quotes.
- `tools/readme_figures.py`: draws the data figures of the README, the degradation laws, the
  order margins, the adaptation to the new laws and the transfer to XJTU-SY, from the saved
  outcomes and the scenario banks.

### Known limitations

- The test suite and the reproduction of the article ran on Windows with CPython 3.12.9, NumPy
  2.4.2, PyTorch 2.8.0 (the CUDA 12.8 build, run on the CPU), scikit-learn 1.8.0 and pandas
  2.3.0. There every rerun matched the stored results of the article bit for bit, on the
  synthetic lives, on FEMTO and on XJTU-SY. Linux is covered by the CI workflow, while macOS and
  other Python versions are untested. Other platforms or library builds, including the CPU-only
  PyTorch wheel of the CI workflow, can change the last digits of a fit and then the trained
  policies.
- Recomputing the default-track table is an opt-in test (`MAINTENANCE_ARENA_FULL_TRACK=1`)
  outside the CI workflow.
- The package's own `train` curriculum and the benchmark tracks use a preparation time of 12
  steps. The runners of the article train every preparation time on scenario banks.
- No dataset and no trained model is shipped. The bearing replay needs the FEMTO and XJTU-SY
  recordings from their providers.
- The `zero_failure` criterion needs, for every rule, a threshold without a failure on the
  calibration lives. If every threshold of a grid fails on some life, `calibrate` and
  `benchmark --config` stop with an error, and a wider grid or `--criterion pooled` is needed.
  The family-equal profit averages the laws present in the calibration bank, and a warning names
  a sampled law that is missing.
- Rule calibration exists for the core rules only. Gearbox and resource rules keep the settings
  of their profiles.
- The test scenarios of the tracks are the ones that `evaluate` and `benchmark` derive from
  `--seed 271828`. Training or tuning with that seed touches the test lives.
- The bootstrap intervals resample scenarios for one fixed agent. Variation across independently
  trained models needs its own summary, such as a two-level bootstrap.
- Agents run in the Python process of the evaluator. The harness passes them only the
  observation and `info`, but it cannot prevent deliberate introspection, so a reported result
  can only be checked together with the code of the agent.
- Checkpoint files differ between repeated training runs, although the trained weights are
  identical. Native checkpoints store run-specific random-generator states, and
  Stable-Baselines3 archives store a start time.
- The resource observation spaces have infinite upper bounds, which triggers one Gymnasium
  warning in environment checks.
- Some record fields, such as `checkpoint` and the sources of resume and initialization, contain
  local absolute paths.
- Training tests use small budgets. They check the implementation, not convergence.
