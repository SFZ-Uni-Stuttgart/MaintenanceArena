# Synthetic experiments of the accompanying article

This folder holds the runners and configurations of the article's synthetic studies. The seeds
in the JSON configurations rebuild every training, calibration and test life, so no data are
shipped or downloaded.

The runners build on the scenario banks of `maintenance_arena.banks` and train with
`learning.collector.BankTrainer`, or with `learning.dual` on both noisy estimates. The learner
comparison also trains DQN and double DQN with `learning.learners`. `replay`
gives the first orders and rule calibrations, `metrics` the summaries, `selection` the noise
selection rules and `analysis` the figure data. The age test adds `age_shift_banks.py` next to
its runner, with the lifetime twins, the age input and its interval code.
[docs/ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md#scenario-banks-of-the-accompanying-article)
describes these modules.

## Requirements and commands

Install the package with the `rl` extra ([README](../../README.md#install)) and run the commands
from the repository root. `--output` must name a new directory.

```shell
python experiments/synthetic/source_learning.py --output outputs/synthetic/source --jobs 10
python experiments/synthetic/input_sets.py --output outputs/synthetic/input-sets --jobs 30
python experiments/synthetic/preparation_time.py --output outputs/synthetic/preparation --jobs 30
python experiments/synthetic/adaptation.py --source outputs/synthetic/source --output outputs/synthetic/adaptation --jobs 40
python experiments/synthetic/measurement_noise.py --source outputs/synthetic/source --output outputs/synthetic/noise --jobs 10
python experiments/synthetic/benchmarks.py --source outputs/synthetic/source --output outputs/synthetic/benchmarks
python experiments/synthetic/learner_comparison.py --output outputs/synthetic/learners --jobs 30
```

`adaptation.py`, `measurement_noise.py` and `benchmarks.py` need the final source policies, so
`--source` names the output of `source_learning.py`. The preparation-time study can also run in
two stages, 204,800 steps and then a continuation to 307,200:

```shell
python experiments/synthetic/preparation_time.py --stage first --output outputs/synthetic/preparation-first --jobs 30
python experiments/synthetic/preparation_time.py --stage continuation --continue-from outputs/synthetic/preparation-first --output outputs/synthetic/preparation-continued --jobs 30
```

The continuation restores weights, optimizer, random streams, bank position and counters, so it
gives the same policies and outcomes as the single run. It needs the Python, NumPy and PyTorch
versions of the first stage.

The measurement-noise runner has three stages. `--stage development` trains the candidate
designs of the noise study and screens them on separate lives, and `--stage selection`
recomputes the selection from the outcome tables of these screens. The default stage evaluates
the selected design on the test lives of Table 3. With `--development`, it reuses the line-fit
runs of a `--stage development` output, which are the same runs:

```shell
python experiments/synthetic/measurement_noise.py --stage development --source outputs/synthetic/source --output outputs/synthetic/noise-designs --jobs 24
python experiments/synthetic/measurement_noise.py --stage selection --development outputs/synthetic/noise-designs --output outputs/synthetic/noise-selection
python experiments/synthetic/measurement_noise.py --source outputs/synthetic/source --development outputs/synthetic/noise-designs --output outputs/synthetic/noise-from-designs
```

`figures.py` computes each figure whose input directories are given, and with `--plots` (the
`plots` extra) it also draws plain figures:

```shell
python experiments/synthetic/figures.py --output outputs/synthetic/figures --source outputs/synthetic/source --input-sets outputs/synthetic/input-sets --preparation outputs/synthetic/preparation --adaptation outputs/synthetic/adaptation --plots
```

Options shared by all runners:

| Option | Meaning |
| --- | --- |
| `--output DIR` | New output directory (required) |
| `--jobs N` | Training processes in parallel (default 1). Results do not depend on `N`. |
| `--budget STEPS` | Quick check with a smaller training budget (see below) |
| `--roles K` | Quick check with only the first `K` training seeds |
| `--config FILE` | Another configuration file (default: the runner's JSON) |

A quick check keeps the article's banks and seeds but does not reproduce its results. A budget
equal to a configured milestone keeps the milestones up to it, so the runs are exact beginnings
of the full runs. `--budget 51200 --roles 1` of `source_learning.py`, for example, gives the
first checkpoint of the first source run. Any other budget scales all milestones in proportion,
so every stage still runs, but the rollouts end at other steps. `--stage development` always
scales. Its budget replaces the 819,200 steps of the longest runs, and every run keeps all
milestones at the same fraction. The selection stage and `figures.py` accept neither option, and
`benchmarks.py` rejects `--budget`. In `learner_comparison.py`, `--roles K` keeps the first `K`
runs of every learner, and DQN and double DQN keep the exploration of the full budget when
`--budget` equals a milestone.

## What each runner reproduces

| Runner | Article | Training | Evaluation |
| --- | --- | --- | --- |
| `source_learning.py` | Section 4.1, Table 2, the additional table of source controls | 10 runs, 204,800 steps, milestones 51,200, 102,400 and 204,800 | Time rule fitted on 256 lives among 26 health or projected-time thresholds. 800 validation lives. The 2,000 test lives of Table 2 with the Time rule at projected time 13 and 15. The source controls add age replacement and health threshold (both fitted on 2,000 further lives), reactive operation and the last-safe reference |
| `input_sets.py` | Section 4.2 | 3 input sets x 10 runs, 204,800 steps, milestones every 51,200 | Every checkpoint on 800 diagnostic lives, the final policies on 800 test lives, Time rule fitted on 256 lives |
| `preparation_time.py` | Section 4.4 | 3 preparation times x 10 runs, 204,800 steps, continued to 307,200 | 800 test lives per `L`. Time rule fitted per `L` on 256 lives (health, or projected time `L - 2` to `L + 36`, ties toward `L + 3`) |
| `adaptation.py` | Section 5, Table 4 | 4 laws x 10 seeds x (fine-tuned, fresh), 51,200 steps, milestones 12,800 and 51,200. Fresh with the full budget: 204,800 steps on 10,781 training-life pairs, milestones 12,800, 51,200, 102,400 and 204,800 | 800 test lives per law: source PPO and the source Time rule unchanged, fine-tuned and fresh PPO, fresh PPO with the full budget, the Time rule refitted per law on 256 lives and fine-tuned PPO + calibration, with an ORDER cut per run fitted on the same 256 lives |
| `measurement_noise.py --stage development` | Section 4.3, choice of the noise design | 5 kinds of runs x 10 seeds on noisy lives (Gaussian error 0.005), see [Choice of the noise design](#choice-of-the-noise-design) | Four Time rule estimates (raw, line fit, both agreeing, either) and 120 ORDER cuts for the calibrated designs, fitted on 256 noisy calibration lives. Four screens on 800 further lives each. Time rule estimate and design selected by the failure bounds |
| `measurement_noise.py --stage selection` | Section 4.3 | none | The selection from the outcome tables of a `--stage development` output |
| `measurement_noise.py` | Section 4.3, Table 3 | 10 runs of the selected design (line-fit inputs), 819,200 steps, or those of a `--stage development` output | ORDER cuts of the noise-trained and source policies and the line-fit Time rule threshold, fitted on 256 noisy calibration lives. 2,000 test lives with noisy and exact measurements |
| `benchmarks.py` | The additional table of further calibrations | none | Fitted on 2,000 calibration lives: ORDER cuts of the source policies (123 cuts), the Time rule with 802 thresholds, the two-indicator rule (236,876 candidates), the privileged per-law rule and failure-budget variants (at most 1% failures). Replayed on the 2,000 test lives of Table 2 |
| `age_shift.py` | Section 6.5, Appendix D | 4 designs (exact) or 2 designs (noisy) x 10 runs, 204,800 steps, milestones 51,200, 102,400 and 204,800 | 2,000 source test lives and their twins with doubled and quadrupled lifetime, with exact inputs also 800 lives per new law and their doubled twins. Time rule fitted on 256 source lives of the setting |
| `age_shift.py --scaled-noise` | Section 6.5, Appendix D | none, replays the runs of a `filtered` or `raw` output | Cells A, B and B4 and the twins Bs and B4s of B and B4 with the Gaussian error divided by the lifetime factor (0.0025 and 0.00125). Time rule fitted again on the 256 source lives and checked against the stored one |
| `failure_cost.py` | Section 4.1, Appendix C | 2 failure costs (3,500 and 14,000 CU) x 10 runs, 204,800 steps, milestones 51,200, 102,400 and 204,800 | Time rule fitted per cost on 256 source lives. The 2,000 test lives of Table 2, priced with the failure cost of each arm |
| `learner_comparison.py` | Appendix A, Table A2, the additional learner table | 3 learners (PPO, DQN, double DQN) x 10 runs, 204,800 steps, milestones 51,200, 102,400 and 204,800. The three runs of one index share their training lives | Every checkpoint on 800 diagnostic lives. The final policies on 2,000 new test lives with the Time rule at projected time 15 (not refitted), reactive operation and the last-safe reference. The additional learner table adds the Time rule at 13 |
| `figures.py` | Figures 3 to 6 and the training failures of Figures C1 and C2 | none | Damage paths of the degradation laws. First orders of the ten source policies and the Time rule on one life per source law. Preparation-time policy maps (ten policies per `L` at 307,200 steps, 512 lives per law). Adaptation policy maps (source, fine-tuned and their difference on 512 lives per new law). Failed training episodes in bins of 6,400 or 3,200 steps |

The outcome files keep the stored names: Rule, Source Rule and Recalibrated Rule for the Time
rule, and Naive projection rule for the Time rule at projected time 13.

### Choice of the noise design

`--stage development` trains the candidate designs of the noise study and screens them on 800
lives per screen, separate from the calibration and test lives. The ten designs are
checkpoints of five kinds of runs, named by estimate, start and steps on noisy lives:

| Design | Run | Total budget |
| --- | --- | ---: |
| `raw_fresh_204800`, `raw_fresh_819200` | consecutive differences, from scratch | 204,800, 819,200 |
| `line_fit_fresh_204800`, `line_fit_fresh_819200` | line fit, from scratch | 204,800, 819,200 |
| `raw_tuned_51200`, `raw_tuned_204800` | source policies fine-tuned on consecutive differences | 256,000, 409,600 |
| `line_fit_tuned_51200`, `line_fit_tuned_204800` | source policies fine-tuned on the line fit | 256,000, 409,600 |
| `both_tuned_51200`, `both_tuned_204800` | source policies fine-tuned on both estimates (eight inputs) | 256,000, 409,600 |

The fresh runs train to 819,200 steps on each seed's source training bank followed by 32,337
further pairs. The fine-tuned runs train to 204,800 steps on the source training bank. The four
screens are stored as `round_one` to `round_four`:

- `round_one` replays `raw_fresh_204800`, the first 204,800 steps of the fresh raw runs, with
  greedy decisions (ORDER when P(ORDER) > P(WAIT)) on noisy and exact measurements.
- `round_two` and `round_three` replay their designs with greedy decisions.
- `round_four` trains nothing. Every network of the ten designs gets an ORDER cut on the
  123-value grid (ties toward 0.5), fitted on the calibration lives of the Time rule. The source
  policies get cuts as controls that cannot be selected.

Each screen reads the noisy cell and averages over the ten runs of a design. `R` is the
equal-law profit rate in percent of the last-safe reference, and `F` the failure share averaged
over the laws. The state estimate of the Time rule is the one with the highest `R` among those
with `F` at most 1% and at most 2% on every law. Within 0.10 points of `R`, the lower `F` wins,
then the configured order. A design must have `F` at most min(1%, `F` of that Time rule + 0.25
points) and at most 2% on every law. Among these, the highest `R` wins, within 0.10 points the
lower `F`, then the smaller budget.

The greedy screens select no design. Among the ten calibrated designs of `round_four`, the
selection picks `line_fit_fresh_819200` with the line-fit Time rule (projected time at most 18),
the procedure of Table 3. `selection.json` records every gate, tie and the final choice, and
`screen_<round>.json` the values behind them.

### Operating age under lifetime and degradation shift

```shell
python experiments/synthetic/age_shift.py --output outputs/synthetic/age-shift --jobs 20
python experiments/synthetic/age_shift.py --setting filtered --exact outputs/synthetic/age-shift --output outputs/synthetic/age-shift-filtered --jobs 20
python experiments/synthetic/age_shift.py --setting raw --exact outputs/synthetic/age-shift --output outputs/synthetic/age-shift-raw --jobs 20
python experiments/synthetic/age_shift.py --setting raw --scaled-noise --policies outputs/synthetic/age-shift-raw --exact outputs/synthetic/age-shift --output outputs/synthetic/age-shift-raw-scaled
```

`age_shift.py` tests operating age as a policy input (article Section 6.5 and Appendix D). Ten
runs per design train on the source laws for 204,800 steps with the seeds and training lives of
`source_learning.py`, so the design without age is the source cohort of Table 2. The age design
adds `min(t / 160, 4)` at operating age `t` as a fifth input. It starts from the same network
with a zero age column and therefore from the same function. The test cells pair the 2,000 test
lives of Table 2 (A) with twins whose lifetime is doubled (B) or quadrupled (B4). A twin keeps
the shape parameters, lifetime quantile and stream address of its life, so its damage path over
relative age is unchanged.

| Option | Meaning |
| --- | --- |
| `--setting exact` | Exact health and rate (default). Two further designs train on a wider lifetime distribution (lognormal scale 0.30). The cells add the 800 test lives per new law (C) and their doubled twins (D) |
| `--setting filtered` | Health with Gaussian error 0.005, rate from the eight-step line fit. Designs without and with age, cells A, B and B4 |
| `--setting raw` | As `filtered`, with the rate from consecutive differences |
| `--exact DIR` | Output of `--setting exact`, whose runs join the intervals of a noisy setting |
| `--scaled-noise --policies DIR` | No training. Replays the runs of `DIR`, an output of the same noisy setting, on A, B, B4, Bs and B4s |

In the twins Bs and B4s, the Gaussian error is divided by the lifetime factor (0.0025 and
0.00125), so the noise per damage increment equals that of the source lives. They keep the
measurement streams of B and B4, so only the size of the noise changes. The replay fits the
Time rule again and stops if it differs from the one stored in `DIR`.

The Time rule (stored as Rule) is fitted on the 256 source calibration lives of each setting. It
selects projected time 15 with exact inputs, 18 with the line fit and 12 with consecutive
differences. On the longest Power-law twins in cell D, two successive float32 health values can
round to the same number at installation. Such a value is lowered by one float32 step, on 522
steps in all.

The outputs follow the other runners, with group `<cell>/<law>` in `outcomes_test.csv` and
summaries per cell and per cell and law. Unlike the other runners, `intervals.json` resamples the
ten runs and the lives within each law together, with one draw of the lives shared by twin cells,
as in the article. With `--exact` it also holds the age effect under exact inputs and its
difference from the noisy setting. `descriptives.json` holds the share of the deadline shift that
the first orders follow, the first-order ages and the precision of the deadline coordinate at the
last safe order.

### Failure cost

```shell
python experiments/synthetic/failure_cost.py --output outputs/synthetic/failure-cost --jobs 20
```

`failure_cost.py` trains the source policies again with a failure cost of 3,500 or 14,000 CU
instead of 7,000 CU (article Section 4.1 and Appendix C). The ten runs per cost use the seeds and
training lives of `source_learning.py`, and only the failure cost of the training environment
changes. The Time rule is fitted per cost on the 256 source calibration lives and selects
projected time 15 at both costs. PPO and the Time rule are replayed on the 2,000 test lives of
Table 2 and priced with the cost of their arm. `outcomes_test.csv` stores the methods as
`PPO, 3500 CU` and `Rule, 3500 CU`, `rules.json` holds the Time rule of every cost and
`margins.json` the order margins (T − L) − o of every policy and life.

## Outputs

Every runner writes `configuration.json`, the configuration it used, and `run.json` with the
command, versions, start, end and elapsed time. Training runners keep one folder per run below
`fits/`, such as `fits/source/26101601/`, with a checkpoint at every milestone and
`training.json` (bank, checkpoints and training log). The learner comparison names its folders by
learner, such as `fits/DQN/26102601/`, and also records the recipe and the number of trainable
parameters in `training.json`. Its `run.json` counts the failure-free runs of every learner.
Runners that calibrate write `rules.json` with every threshold and cut. Each evaluation set gets
two tables:

- `outcomes_<set>.csv` with one row per method, run and life and the columns `method`, `group`,
  `run`, `unit`, `last_productive_step`, `order_step`, `cash`, `calendar_steps`, `failed` and
  `unused_fraction`. `group` is the law of the life or the condition (training steps,
  `L6`/`L12`/`L24`, `noisy`/`exact`). `run` is the training seed, empty for rules and
  references, and `unit` the life's pair index in its bank. `last_productive_step` is
  `T = eol - 1`, `order_step` the first order, empty without one, and `unused_fraction` the
  unused steps over `T - L`.
- `summary_<set>.csv` with the equal-life failure share, unused life and retained profit in
  percent, for all lives (`group` = `all`) and per group.

The summaries weight every life equally and average several runs on each life first, as in
[paper/README.md](../../paper/README.md#metrics). `intervals.json` holds 95% paired bootstrap
intervals for the configured contrasts on the test lives. They use 10,000 resamples of whole
lives with all their runs, drawn within each law when the outcome groups are laws.
`preparation_time.py` stores its intervals per `L` in `rules.json`.

`--stage development` writes `outcomes_<round>.csv` and `summary_<round>.csv` for `round_one` to
`round_four`, with the group `<cell>/<law>` (for example `noisy/linear_reference`). Its
summaries are per cell and per cell and law. It also writes `rules.json` with the four Time rule
estimates, the cut grid and the 120 cuts, `screen_<round>.json` and `selection.json`.

`figures.py` writes `curves.json`, `order_cases.json`, `preparation_maps.*` and
`adaptation_maps.*`. The `.csv` maps hold the state count and mean P(ORDER) of every occupied
bin, and the `.json` maps the edges and support. `preparation_native.csv` and
`adaptation_native.csv` hold the mean at every distinct (health, damage rate) coordinate, which
supports the 0.5 contours. `training_failures.csv` and `.json` hold the failures per run and bin
and the median of the runs. `--plots` adds the `fig_*.png` files.

## Run times

Measured on a Windows workstation with 48 logical CPUs (CPython 3.12.9, NumPy 2.4.2, PyTorch
2.8.0), one study at a time, with the `--jobs` of the commands above:

| Runner | `--jobs` | Wall time | One training run |
| --- | ---: | ---: | ---: |
| `source_learning.py` | 10 | 3.6 min | 2.5 min (204,800 steps) |
| `input_sets.py` | 30 | 4.6 min | 3.5 to 3.6 min (204,800 steps) |
| `preparation_time.py` | 30 | 6.1 min | 4.7 to 5.4 min (307,200 steps) |
| `adaptation.py` | 40 | 3.8 min without the full-budget runs | 0.8 to 1.5 min (51,200 steps), 2.2 min alone (204,800 steps) |
| `measurement_noise.py` | 10 | 14.0 min | 12.2 to 12.4 min (819,200 steps) |
| `measurement_noise.py --stage development` | 24 | 20.1 min | 12.2 to 15.0 min (819,200 steps), 2.7 to 4.3 min (204,800 fine-tuning steps) |
| `measurement_noise.py --stage selection` | - | 1 s | no training |
| `measurement_noise.py --development` | - | 1.4 min | reuses the runs of `--stage development` |
| `benchmarks.py` | - | 1.6 min | no training |
| `learner_comparison.py` | 30 | 4.8 min | 2.9 to 3.5 min (204,800 steps) |
| `age_shift.py --roles 1` | 4 | 4.5 min | 1.8 to 2.5 min alone (204,800 steps) |
| `age_shift.py --setting raw --exact DIR --roles 1` | 2 | 5.4 min | 1.8 to 2.5 min alone (204,800 steps) |
| `age_shift.py --setting raw --scaled-noise --policies DIR --exact DIR --roles 1` | - | 3.2 min | no training |
| `failure_cost.py` | 8 | 14.5 min | 3.4 to 5.9 min (204,800 steps) |
| `figures.py` | - | 2.0 min | no training |

The scaled-noise and failure-cost rows were measured while other jobs used the machine.
The seven studies take about 39 minutes one after another. The candidate designs of the noise
study, their selection, the evaluation that reuses their runs and the figure data take about 24
minutes more. Runs slow down in larger pools: a 204,800-step run takes 2.5 minutes in a pool of
10 and 3.6 minutes in a pool of 30. Quick checks such as `--budget 2048 --roles 1` take 1 to 2.5
minutes, mostly to build and replay the full calibration and test lives. `--stage development`
takes 3.5 minutes with `--budget 8192 --roles 2`. Training uses one CPU thread per process, and evaluation and
calibration run in the main process. The two-indicator search of `benchmarks.py` holds about
1 GB of first orders in memory.

## Reproduction of the article

The runners use the article's banks, run seeds, training procedure and replay. On the setup
above, they rebuild the article's training, calibration and test lives bit for bit, and every
checkpoint with its training log. The first orders, thresholds and outcome rows behind the
synthetic tables match, so every cell of those tables is reproduced. The same holds for the
choice of the noise design: the checkpoints of its 50 runs, the four Time rule thresholds and
the 120 cuts match, and so do the outcome rows of the four screens and the three selections.

For the controlled age test, 81 checkpoints, 80,000 first orders and every stored interval were
compared with the article's results, and all agree. With `--scaled-noise`, a replay of one seed
and the Time rule matches its 60,000 stored first orders, and the stored orders of all ten runs
give the 76 stored intervals. A retrained full-budget run and a retrained failure-cost run match
their stored checkpoints. The stored policies give all 32,000 first orders of fine-tuned PPO +
calibration and all 44,000 first orders of the failure-cost study. The figure data reproduce the
plotted lives and orders and the binned training failures. They also reproduce the policy-map
states and ORDER probabilities with their bin counts, means and 0.5 contours.
Other platforms, library builds or CPU instruction sets can change the floating-point details of
training and then the trained policies. The lives, the rules and the replay do not depend on
them.
