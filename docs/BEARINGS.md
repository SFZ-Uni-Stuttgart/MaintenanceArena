# Bearing replay

`maintenance_arena.bearings` replays recorded run-to-failure bearing histories in the decision
problem of MaintenanceArena. It implements the bearing study of the accompanying article
(Section 6 and Appendix D). No dataset is shipped. The runners in
[experiments/bearings](../experiments/bearings/README.md) repeat the article's experiments from
the raw files.

- [Install](#install)
- [Data](#data)
- [Processing pipeline](#processing-pipeline)
- [Replay economics and environment](#replay-economics-and-environment)
- [Policies, rules and metrics](#policies-rules-and-metrics)
- [Commands](#commands)
- [Measured run times](#measured-run-times)
- [Verification](#verification)
- [What the replay assumes](#what-the-replay-assumes)

## Install

Raw files need pandas, the prediction model scikit-learn and PPO PyTorch. The `bearings` and `rl`
extras install them:

```shell
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0+cpu"
python -m pip install -e ".[rl,bearings]"
```

Importing `maintenance_arena.bearings` loads none of the three. The article's fits used
scikit-learn 1.8.0, on which the predictions depend, pandas 2.3.0, NumPy 2.4.2 and PyTorch 2.8.0
on CPython 3.12. The runners limit numerical libraries to one thread, because multithreaded sums
can change the last digits of a fit.

## Data

Both datasets are run-to-failure tests of rolling bearings under three speed and load
conditions. A horizontal and a vertical accelerometer sample at 25.6 kHz.

| Dataset | Bearings | Acquisitions | Minutes per bearing | Provider |
| --- | --- | --- | --- | --- |
| FEMTO (PRONOSTIA, IEEE PHM 2012 Prognostic Challenge) | 17: 6 in `Learning_set`, 11 in `Full_Test_Set` | 2,560 samples every 10 s | 39 to 468 | FEMTO-ST Institute, Besançon |
| XJTU-SY | 15: 5 per condition | 32,768 samples every minute | 42 to 2,538 | Xi'an Jiaotong University and Changxing Sumyoung Technology |

### Sources

- FEMTO is entry 10, "FEMTO Bearing", of the NASA Prognostics Center of Excellence Data Set
  Repository, <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>.
  The archive is <https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip>. Cite
  P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Chebel-Morello, N. Zerhouni and
  C. Varnier, *PRONOSTIA: An experimental platform for bearings accelerated degradation tests*,
  IEEE International Conference on Prognostics and Health Management, Denver, 2012.
- XJTU-SY is distributed through <https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets>,
  maintained by Biao Wang. It lists the download mirrors (Google Drive, Dropbox, MediaFire,
  MEGA and Baidu Netdisk) and the website biaowang.tech. Cite B. Wang, Y. Lei, N. Li and N. Li,
  *A hybrid prognostics approach for estimating remaining useful life of rolling element
  bearings*, IEEE Transactions on Reliability 69(1), 401–412, 2020,
  doi:10.1109/TR.2018.2882682. The dataset authors describe the tests and their stop criterion
  in Y. Lei, T. Han, B. Wang, N. Li, T. Yan and J. Yang, *XJTU-SY rolling element bearing
  accelerated life test datasets: A tutorial*, Journal of Mechanical Engineering 55(16), 1–6,
  2019, doi:10.3901/JME.2019.16.001 (in Chinese).

### Layout

Only these folders below one data root are read:

```text
DATA/
  FEMTO/
    Learning_set/Bearing1_1/acc_00001.csv ... acc_02803.csv
    Learning_set/Bearing1_2/ ... Bearing3_2/          (6 bearings)
    Full_Test_Set/Bearing1_3/ ... Bearing3_3/         (11 bearings)
  XJTU-SY/
    35Hz12kN/Bearing1_1/1.csv 2.csv ... Bearing1_5/
    37.5Hz11kN/Bearing2_1/ ... Bearing2_5/
    40Hz10kN/Bearing3_1/ ... Bearing3_5/
```

`Full_Test_Set` holds the eleven test bearings recorded to the end of their tests. The
truncated test set of the challenge and incomplete downloads have other acquisition counts,
which the layout check reports.

FEMTO files have no header, `;` or `,` separators and six columns: hour, minute, second,
microsecond, horizontal and vertical acceleration. Other files, such as `temp_*.csv`, are
ignored. XJTU-SY files have two columns under the header
`Horizontal_vibration_signals,Vertical_vibration_signals`. Acquisitions must be numbered 1..N
without gaps. Bearing names are local to each dataset: FEMTO Bearing1_1 and XJTU-SY Bearing1_1
are different bearings.

### Steps

1. Download the FEMTO archive and the XJTU-SY parts from the sources above.
2. Extract them, including archives inside archives, until the folders `Learning_set`,
   `Full_Test_Set`, `35Hz12kN`, `37.5Hz11kN` and `40Hz10kN` appear with one folder per bearing.
3. Move these folders into `DATA/FEMTO` and `DATA/XJTU-SY`. Or leave them in place and pass
   their parent folders with `--femto-dir` and `--xjtu-sy-dir`, or `directory=` in Python.
4. Check the layout. Without opening a file, the check lists the bearings and acquisition
   counts and names every missing folder, bearing or file. It exits with status 1 if anything
   is missing or differs from the article's counts.

   ```shell
   python experiments/bearings/prepare_statistics.py --data-root DATA --check
   ```

   In Python, `maintenance_arena.bearings.check_layout(DATA, "femto")` returns the same report.

## Processing pipeline

1. `acquisition_statistics` reduces each acquisition to the uncentered RMS, the maximum
   absolute amplitude and the kurtosis of each channel
   (`rms_h, rms_v, peak_h, peak_v, kurt_h, kurt_v`). Kurtosis is the centered fourth moment
   over the squared variance, with the variance floored at 1e-12. `prepare(data_root, dataset, path, jobs=...)` reads the raw files
   with pandas and saves one (N, 6) array per bearing in a compressed `.npz` cache, which
   `load_statistics` reads back. `prepare(None, dataset, path, directory=...)` reads a dataset
   folder elsewhere. The bearing order, by folder and then by name, fixes the bearing indices
   of the folds and of PPO sampling.
2. `to_minutes` and `histories` aggregate six FEMTO acquisitions or one XJTU-SY acquisition per
   minute: RMS by quadratic mean, peak by maximum and kurtosis by mean. A partial last minute is
   kept. The number of minutes T is the last productive minute, so a FEMTO bearing with N
   acquisitions has T = ceil(N/6).
3. `relative_signals` divides RMS and peak by their median over the first min(t, 10) minutes
   and takes logarithms, `lr` and `lp`. Kurtosis gives `lk = log(max(kurtosis, 1e-6)/3)`.
4. `Forecaster` and `cross_fit` predict the remaining minutes. This is the remaining-time
   prediction model of the article. A `HistGradientBoostingRegressor` (300 iterations, learning
   rate 0.05, 15 leaves, L2 1.0, random state 0, no early stopping) learns min(T - t, 125). Its
   15 inputs are the EWM (alpha 0.3) of the six log signals, their least-squares slopes over the
   last ten minutes (zero before minute 3), the channel maxima of `lr` and `lp`, and minute/200.
   The model without age drops the last input. Predictions are clipped at zero. In `cross_fit`,
   a model fitted without its inner fold predicts each training bearing, and policy training and
   rule calibration use only these out-of-fold predictions. A model fitted on all training
   bearings predicts the evaluation bearings.
5. `decision_series`, `policy_observations` and `design_observations` build the float32 policy
   input of minute t. With r the prediction, `level` and `peak` the EWM (alpha 0.3) of the
   channel maxima of `lr` and `lp`, and slopes over ten minutes, its columns are:

   | Column | Value |
   | --- | --- |
   | `forecast` | clip(r/125, 0, 1.2) |
   | `deadline_coordinate` | clip((13 - (r + 1))/13, -1, 1) |
   | `forecast_trend` | clip(slope(r), -5, 3)/5 |
   | `rms_level` | clip(level, -1, 4)/2 |
   | `rms_trend` | clip(10 slope(level), -1, 1) |
   | `peak_level` | clip(peak, -1, 4)/2 |
   | `age` | clip(t/250, 0, 4) |

   Every value at minute t uses minutes 1..t only. Table D1 of the article lists the same steps.

The input designs (`INPUT_DESIGNS`) decide where operating age enters:

| Key | Article name | Prediction model | Policy columns |
| --- | --- | --- | --- |
| `full` | Full | with age | all seven |
| `no_age` | No age | without age | the first six |
| `no_policy_age` | No policy age | with age | the first six |
| `no_forecaster_age` | No prediction age | without age | all seven |
| `condition_only` | Condition only | none | `rms_level`, `rms_trend`, `peak_level` |

The runners store No prediction age as No forecaster age.

## Replay economics and environment

`ReplayEconomics` replays one life with the event order and prices of the core task: 255 per
productive minute, 520 per replacement, 25 per nonproductive step and 7,000 per failure. A
decision at minute t follows the observation of minute t. WAIT earns the next productive
minute, or fails at the recorded end t = T. ORDER starts twelve productive preparation minutes
and is timely if t + 12 <= T. One exchange step and six downtime steps follow, so the last safe
order is T - 12.

With `preserve_preparation_v1`, the article's setting and the default here, a failure stops
production and keeps the preparation already done. The rest of the preparation runs without
production before the exchange, and the failure detection counts as its first step. A reactive
failure thus takes 19 nonproductive steps. `immediate_exchange_v1`, the core package default,
is also available.

The cash of a life is 255 U - 520 - 25 N - 7,000 F for U productive minutes, N nonproductive
steps and F failures. `outcome(T, order)` returns cash, calendar minutes, failure and unused
minutes T - U. `macro(T, t, action)` gives the cash of every calendar step of one decision, and
`oracle_order(T)` the last safe order of the reference that knows T.

`BearingReplayEnv(observations, ...)` is a Gymnasium environment over prepared input matrices,
one row per minute. Each episode is one life, drawn uniformly by the environment's generator or
chosen with `reset(options={"bearing": i})`. Actions are `0` WAIT and `1` ORDER. A WAIT before
T lasts one calendar minute. As in `CoreCycleEnv`, an ORDER or a WAIT at T runs to the end of
the renewal in one terminal transition of up to 19 calendar minutes
(`info["duration_calendar_steps"]`). The reward is the discounted cash of the transition
divided by 1,000, with gamma 0.999 per calendar minute.

`info` has the keys of `CoreCycleEnv` (`ledger`, `events`, `raw_profit_sum`,
`duration_calendar_steps`, `gamma_pow_duration` and the episode totals) and the events
`operate`, `mobilize`, `planned_exchange`, `failure`, `failure_during_mobilization`,
`corrective_mobilize`, `corrective_exchange` and `downtime`. The reset books the first recorded
minute, which passes before the first decision and earns no reward, so the ledgers of an
episode add up to `outcome`. Observations and `info` do not contain T. `policy_state()` returns
the decision minute, and `diagnostics()` and `episode_outcome()` serve evaluators. The
environment needs data, so no Gymnasium ID is registered.

## Policies, rules and metrics

### PPO

`BearingPPOAgent` in `maintenance_arena.bearings.ppo` is the core agent of profile
`paper-core-continuous-v1`. It has 64-64 tanh actor and critic networks, Adam with learning
rate 3e-4, rollouts of 1,024 calendar steps, 10 epochs, minibatch 128, clip 0.2, gamma 0.999
and lambda 0.95. Seed roles follow the core recipe. Only the first actor and critic layers
change, to take the bearing inputs. Torch draws them after the core initialization, reseeded
with the network seed role, actor first.

`train_policy(observations, seed, steps)` trains on `BearingReplayEnv`. Each episode starts at
minute 1 of a training bearing drawn uniformly with replacement from the environment seed role.
The reward of decision k is sum_j gamma^j (c_kj - rho_k)/1000, with c_kj the cash of its
calendar step j. The rate rho starts at the pooled reactive rate of the training lives,
weighted as 100 mean lives. After each episode, A <- 0.99 A + cash, W <- 0.99 W + calendar and
rho = A/W. The synthetic studies use rho = 0. Advantages use generalized advantage estimation
with the decision discount gamma^m_k of a decision lasting m_k steps in place of gamma, and lambda
applied per decision. An episode ends after the exchange and downtime that follow an order or a
failure, and its last decision has no continuation value. Training ends with the first complete update at or beyond the budget: 409,600
calendar steps, or 819,200 for Condition only. `greedy_order_minute` orders at the first minute
whose ORDER probability exceeds the WAIT probability. `save_policy` and `load_policy` store the
weights with metadata.

### Time rule and Indicator rule

The Time rule of the article uses the prediction supplied to the matching PPO design.
`calibrate_forecast_rule` and `forecast_rule_order` order at the first minute whose float32
prediction input is at most float32(theta/125). The fitted rule takes the theta of highest
pooled training profit on the `forecast` grid of the table below, the smallest on ties. The
runners store it as Forecast rule. The variant without margin, stored as Naive forecast rule,
fixes theta = 12.

`calibrate_indicator_rule` fits the Indicator rule of the additional bearing table, one
threshold on one decision series:

| Family | Orders at the first minute with | Candidates |
| --- | --- | --- |
| `age` | t >= a | 240 geometric values from 5 to 3,000 minutes, rounded to 197 ages |
| `level`, `peak` | EWM of the `lr` or `lp` channel maximum >= x | EWM rate 0.1, 0.3 or 1, x from 0 to 4 in steps of 0.05 |
| `forecast` | r <= theta | theta from 0 to 90 in steps of 0.5 |
| `trend` | r + k slope(r) <= theta | twelve values of k from 0 to 20, theta from 0 to 90 in steps of 1 |

The calibration keeps the candidate with the highest pooled training profit, total cash over
total calendar time on the out-of-fold series, per family and overall. The first candidate wins
ties. The runners store this rule as Indicator Rule.

### Broader rule search

`RULE_FAMILIES`, `family_orders` and `search_rule_families` extend the Indicator rule to
sixteen families:

| Families | Order when |
| --- | --- |
| `age`, `level`, `peak`, `forecast`, `forecast_trend` | the Indicator rule condition holds |
| the same five names with the prefix `guarded_` | the condition holds at minute 10 or later, when the running-in median is complete |
| `level_after_age`, `peak_after_age`, `forecast_after_age` | the condition holds once a minimum age is reached |
| `level_or_age`, `peak_or_age`, `forecast_or_age` | from minute 10 on, the condition holds or an age limit is reached |

The minimum ages and age limits are the age-grid values of at least 10 minutes. Each family is
refitted with its candidate of highest pooled profit on all training bearings. Inner
cross-validation chooses the family. Each inner fold is evaluated with the candidate selected
without it, and the family with the highest exact pooled profit over the inner folds wins. The
first family wins ties. The runner also reports the family with the highest pooled held-out
profit, the criterion that selected Full. This choice uses the evaluation bearings and is a
hindsight comparison.

### Risk classifier

`calibrate_risk_classifier` and `risk_classifier_orders` fit the non-RL reference of Table 5
that uses all inputs of a PPO design at once. A gradient-boosted tree model (scikit-learn
`HistGradientBoostingClassifier` with `CLASSIFIER_SETTINGS`) receives the policy inputs of Full
or No age and learns to mark every minute at or after the last safe order T - 12. Every minute
has weight 1/T, so each training bearing counts equally. The classifier orders at the first
minute whose score reaches a threshold. The threshold runs from 0.01 to 0.99 in steps of 0.01
and is chosen by pooled training profit on leave-one-bearing-out scores of the training
bearings. The smallest threshold wins ties. The classifier is then refitted on all training
bearings, taken in the order of their names. The runner stores the two classifiers as Risk
classifier: Full and Risk classifier: No age.

### Metrics

The equal-bearing metrics (`outcome_row`, `design_summary`, `paired_bootstrap`) first average
failure, unused life (T - U)/(T - 12), cash and calendar time over the seeds on a bearing.
Retained profit is the bearing's mean cash per mean calendar minute, relative to the last-safe
reference. The three percentages are then averaged over bearings. `paired_bootstrap` resamples
whole bearings with their seeds and gives the difference of two designs with a 95% interval.
The article used 10,000 replicates with PCG64 seed 20260930.

Pooled profit (`pooled_summary`, `pooled_bootstrap`) is the total seed-averaged cash over the
total seed-averaged calendar minutes. It weights bearings by their calendar time, and the
selections of Appendix D use it. `pooled_bootstrap` gives the difference of two designs'
pooled rates relative to the pooled last-safe rate, with the same resampling and the share of
replicates above zero.

`rank_candidates` ranks designs by held-out FEMTO pooled profit. The article selected Full and
its budget this way among six candidates: Full, No age and Condition only at 409,600 or 819,200
training steps. Exact ties prefer fewer failed replays, then the shorter budget, then fewer
policy inputs. The function returns the ranking, the selected design and the best design of
every input design.

`run_nested_selection.py` repeats this choice inside the training bearings of every outer fold
(Section 6.1 and Appendix D of the article). Each of the five inner folds is held out once. The
prediction model is cross-fitted on the other inner folds, and each candidate trains with the
five seeds of the outer fold on the remaining bearings. `rank_candidates` then ranks the
candidates by the pooled profit over the held-out inner bearings. The check selects Full in four
of the five outer folds.

## Commands

Every runner computes missing statistics caches, so this first step is optional:

```shell
python experiments/bearings/prepare_statistics.py --data-root DATA --check
python experiments/bearings/prepare_statistics.py --data-root DATA --output outputs/bearing-statistics --jobs 8
```

Then run the experiments and summarize them:

```shell
python experiments/bearings/run_femto_cross_validation.py --data-root DATA --cache-dir outputs/bearing-statistics --output outputs/bearings-femto --jobs 12
python experiments/bearings/run_transfer.py --data-root DATA --cache-dir outputs/bearing-statistics --output outputs/bearings-transfer --jobs 12
python experiments/bearings/run_age_designs.py --data-root DATA --cache-dir outputs/bearing-statistics --output outputs/bearings-age --jobs 12
python experiments/bearings/run_forecast_rules.py --data-root DATA --cache-dir outputs/bearing-statistics --output outputs/bearings-forecast-rules
python experiments/bearings/run_candidate_selection.py --data-root DATA --cache-dir outputs/bearing-statistics --output outputs/bearings-candidates --jobs 12
python experiments/bearings/run_rule_search.py --data-root DATA --cache-dir outputs/bearing-statistics --output outputs/bearings-rule-search
python experiments/bearings/run_risk_classifier.py --data-root DATA --cache-dir outputs/bearing-statistics --output outputs/bearings-risk-classifier
python experiments/bearings/run_nested_selection.py --data-root DATA --cache-dir outputs/bearing-statistics --output outputs/bearings-nested --jobs 40
python experiments/bearings/summarize.py outputs/bearings-femto/outcomes.csv outputs/bearings-forecast-rules/outcomes.csv --reference "Forecast rule: Full" --output outputs/bearings-femto-summary
python experiments/bearings/summarize.py outputs/bearings-candidates/outcomes.csv --select candidate_selection --output outputs/bearings-selection
```

`--reference "Forecast rule: Full"` pairs every design with the Time rule on the Full
prediction, as Section 6.2 of the article does for Full PPO on FEMTO. Section 6.3 pairs No age
PPO with the Time rule on the No age prediction after transfer:

```shell
python experiments/bearings/summarize.py outputs/bearings-transfer/outcomes.csv outputs/bearings-forecast-rules/outcomes.csv --reference "Forecast rule: No age" --output outputs/bearings-transfer-summary
```

[experiments/bearings/README.md](../experiments/bearings/README.md#running) lists the options
for quick checks and the outputs. The candidate runner writes its ranking to `selection.json`,
and the nested runner the ranking of every outer fold. Three candidates repeat cross-validation designs with the same seeds: Full and No age at
409,600 steps and Condition only at 819,200. `summarize.py --select candidate_selection` ranks
only rows with candidate names such as `Full (409600 steps)`, so rename the cross-validation
designs if you want to reuse their rows.

The rule search writes rows for all sixteen families, the Cross-validated rule and the family
with the highest pooled held-out profit. Its inner cross-validation scores go to
`selections.json`. `summarize.py` writes `summary.md` and `summary.json` into a new directory.
`--reference` adds pooled-profit contrasts to the equal-weight ones.

### Figures

`maintenance_arena.bearings.analysis` computes the data of the three bearing figures from
runner outputs, and `experiments/bearings/figures.py` writes it as JSON. With `--plot` and the
`plots` extra, it also draws simple PNG and PDF versions. The JSON files keep the stored design
names, and the plots show the names of the article.

- The policy map of Fig. 7 (`policy_responses`, `policy_map`) holds the mean ORDER probability
  of the five Full actors of each FEMTO bearing's outer fold at every held-out minute. Rows are
  sorted by life and mark each actor's first order, the order of the Time rule on the same Full
  prediction and the last safe order. The Time rule orders come from the outcomes of
  `run_forecast_rules.py` (`--rule-outcomes`). The script rebuilds the held-out inputs from the
  run's statistics cache and queries the saved actors (`policy_map.json`,
  `policy_responses.npz`). The map includes states after an actor would already have ordered.
  Its colour is an action probability, not a failure risk, and its fraction-of-life axis is for
  display only.
- The transfer timing of Fig. 8 (`transfer_timing`, `transfer_timing.json`) gives the median
  first order of the five Full and the five No age seeds per XJTU-SY bearing, each next to the
  Time rule on the same prediction. A missing order ranks last. A median up to the last safe
  order T - 12 shows the unused share 100 (T - 12 - order)/(T - 12), and a later median is late.
  A timely median does not mean that every seed ordered in time, so the seed orders and failures
  are kept.
- The age clock of Fig. D1 (`age_clock`, `age_clock.json`) holds the first orders of the twenty
  transfer actors (Full, No prediction age, No policy age and No age, five seeds each). It
  covers their seventeen FEMTO training histories, from `fits.json`, and the three longest
  XJTU-SY bearings, with their last safe orders and the longest FEMTO life.

```shell
python experiments/bearings/figures.py policy-map --run outputs/bearings-femto --rule-outcomes outputs/bearings-forecast-rules/outcomes.csv --output outputs/figure-policy-map --plot
python experiments/bearings/figures.py transfer-timing --outcomes outputs/bearings-transfer/outcomes.csv outputs/bearings-forecast-rules/outcomes.csv --output outputs/figure-transfer-timing --plot
python experiments/bearings/figures.py age-clock --runs outputs/bearings-transfer outputs/bearings-age --output outputs/figure-age-clock --plot
```

For the age clock, the age-design runner needs only its transfer:
`run_age_designs.py --evaluations xjtu_sy_transfer`. The same steps in Python, for the first
outer fold, start from the FEMTO cache of `prepare_statistics.py`:

```python
import json
from maintenance_arena.bearings import (BearingReplayEnv, ReplayEconomics, cross_fit, decision_series,
                                        design_observations, histories, load_statistics)
from maintenance_arena.bearings.ppo import train_policy

folds = json.load(open("experiments/bearings/configs/folds.json", encoding="utf-8"))
femto = histories(load_statistics("outputs/bearing-statistics/femto_statistics.npz"), "femto")
test = set(folds["outer_folds"][0])
training = [h for h in femto if h.name not in test]
evaluation = [h for h in femto if h.name in test]
out_of_fold, forecasts = cross_fit(training, folds["inner_folds"][0], evaluation, age=True)
train = [design_observations(decision_series(h, out_of_fold[h.name]), "full") for h in training]
held_out = [design_observations(decision_series(h, f), "full") for h, f in zip(evaluation, forecasts)]
agent = train_policy(train, 2026101600, 409_600).agent
economics = ReplayEconomics()
for history, inputs in zip(evaluation, held_out):
    print(history.name, economics.outcome(history.life, agent.greedy_order_minute(inputs)))
env = BearingReplayEnv(held_out, names=[h.name for h in evaluation])  # for your own agent
```

## Measured run times

Measured on a 24-core Intel Xeon w7-2495X under Windows 11 with one thread per process. Other
jobs shared the machine, and fit times grow with the load.

| Step | Wall time |
| --- | --- |
| FEMTO statistics from the raw files, 24,889 acquisitions, 8 or 12 processes | 9 to 12 s |
| XJTU-SY statistics from the raw files, 9,216 acquisitions, 12 processes | 36 s |
| Predictions, rule calibration and inputs of one FEMTO outer fold, both prediction models | 5 to 9 s |
| One PPO fit of 409,600 calendar steps (Full, No age), 12 in parallel | 192 to 305 s, median 210 s |
| One PPO fit of 819,200 calendar steps (Condition only), 12 in parallel | 403 to 602 s, median 498 s |
| `run_femto_cross_validation.py` with Indicator Rule and Reactive only | 19 s |
| `run_femto_cross_validation.py` from the raw files, 75 fits, 12 processes | 37 min |
| `run_rule_search.py`, sixteen families, five folds | 28 s |
| `run_risk_classifier.py` from the raw files, both datasets, 12 processes | 2.9 min |
| `figures.py policy-map`, 25 Full actors, with rebuilt held-out inputs | 39 s |
| Candidate-selection fits of 819,200 and 409,600 steps, 3 processes | about 500 s and 260 s |
| `run_nested_selection.py` from the statistics cache, 750 fits, 38 processes | 2.8 h |
| `run_transfer.py` from the raw files, 15 fits, 15 processes | 10 min |
| `run_age_designs.py --evaluations xjtu_sy_transfer`, 10 fits, 10 processes, alongside the transfer | 6 min |
| `run_age_designs.py --evaluations femto_cross_validation`, 50 fits, 13 processes | 16 min |
| `run_forecast_rules.py`, both datasets | 91 s |
| `summarize.py` on all outcome files, with bootstrap contrasts | 6 s |

The candidate runner trains 150 actors, so its time follows from the fit times above.

## Verification

The public code was checked against the outputs behind the article by running the commands of
this page on both datasets. The setup was CPython 3.12.9, NumPy 2.4.2, pandas 2.3.0,
scikit-learn 1.8.0 and PyTorch 2.8.0 (the CUDA 12.8 build, run on the CPU) under Windows 11,
with one thread per process.

- The statistics of all 24,889 FEMTO and 9,216 XJTU-SY acquisitions equal the article's caches
  bit for bit, in the same bearing order. Predictions, decision series and policy inputs are
  identical for every FEMTO fold, the all-FEMTO fits and every XJTU-SY bearing, in all five
  input designs.
- The Time rule cutoffs, the Indicator rule of every fold and of the transfer and the
  sixteen-family search select the stored thresholds and families. On the article's stored
  candidate outcomes, the ranking selects Full at 409,600 steps.
- Both risk classifiers select the stored threshold in every fold and for the transfer, and
  their 64 first orders equal the rows of Table 5.
- Two retrained actors of the nested check have the stored network weights and the same
  held-out first orders. On the stored outcomes of all 750 nested fits, the ranking selects the
  stored design in every outer fold.
- All 153 retrained actors have the article's network weights. They are the 75 FEMTO
  cross-validation actors of Full, No age and Condition only, the 50 FEMTO actors of the mixed
  age designs, one actor of each of the three other candidate budgets and the 25 transfer
  actors. Their action probabilities in the policy map and on every XJTU-SY bearing are
  identical.
- Every first order behind Table 5 of the article and the additional bearing table is
  reproduced, on all 17 FEMTO and 15 XJTU-SY bearings. `summarize.py` gives both columns of the
  tables to the printed digits, and the equal-weight and pooled bootstrap intervals agree with
  the stored ones. The three bearing figures draw on these actors and first orders.

Other platforms or library builds, the CPU wheel of PyTorch among them, can change the last
digits of a fit and with them the trained policies.

## What the replay assumes

- The recorded test end is the last productive minute T, and failure follows at once. Both
  datasets stopped their tests by a vibration-amplitude criterion, so T partly depends on the
  amplitudes that the peak inputs describe.
- Replay measures when a policy would have ordered. An order does not change the remaining
  history, and the recordings do not show the physical effect of an intervention.
- Prices, preparation and downtime are those of the core task, not calibrated industrial costs.
  A high profit rate alone shows neither availability nor realistic operating costs.
- Seeds on the same bearing are not independent bearings, so the intervals resample bearings.
  The FEMTO folds are fixed lists, so repeated runs hold out the same bearings.
