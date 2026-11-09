# Bearing experiments

These scripts run the bearing experiments of the accompanying article with
`maintenance_arena.bearings`. [docs/BEARINGS.md](../../docs/BEARINGS.md) describes the data,
the processing pipeline, the replay and the measured run times.

The design column gives the names the runners store. The article calls Forecast rule the Time
rule and No forecaster age No prediction age.

| Script | Experiment | Designs |
| --- | --- | --- |
| `run_femto_cross_validation.py` | Five outer FEMTO folds, five PPO seeds per fold | Full, No age, Condition only, Indicator Rule, Reactive |
| `run_transfer.py` | Fits on all 17 FEMTO bearings, applied unchanged to the 15 XJTU-SY bearings, five seeds | Full, No age, Condition only, Indicator Rule, Reactive |
| `run_age_designs.py` | Both evaluations above | No policy age, No forecaster age |
| `run_forecast_rules.py` | Both evaluations above | Forecast rule and Naive forecast rule on the Full and No age predictions |
| `run_candidate_selection.py` | Five outer FEMTO folds, ranked by pooled held-out profit | Full, No age and Condition only at 409,600 and 819,200 steps |
| `run_nested_selection.py` | The same ranking inside the training bearings of every outer FEMTO fold, five inner folds | Full, No age and Condition only at 409,600 and 819,200 steps |
| `run_rule_search.py` | Five outer FEMTO folds | Sixteen rule families, Cross-validated rule, Rule family by held-out profit |
| `run_risk_classifier.py` | Five outer FEMTO folds and the transfer to XJTU-SY | Risk classifier: Full, Risk classifier: No age |
| `prepare_statistics.py` | Layout check (`--check`) and statistics caches for `--cache-dir` | |
| `summarize.py` | Equal-bearing table, equal-weight and pooled bootstrap contrasts, candidate ranking (`--select`) | |
| `figures.py` | Data of the policy map, the transfer timing and the age clock (Figs. 7, 8 and D1), drawn with `--plot` | |

## Data

Download FEMTO and XJTU-SY from the sources in
[docs/BEARINGS.md](../../docs/BEARINGS.md#sources), which also gives the citations. Extract all
archives, including nested ones, into one data root with the
[layout](../../docs/BEARINGS.md#layout) `DATA/FEMTO/{Learning_set,Full_Test_Set}/BearingC_R/`
and `DATA/XJTU-SY/{35Hz12kN,37.5Hz11kN,40Hz10kN}/BearingC_R/`. Then check the layout. The
check names every missing folder, bearing or acquisition file:

```shell
python experiments/bearings/prepare_statistics.py --data-root DATA --check
```

To use the data where they are, pass `--femto-dir` with the folder that holds `Learning_set`
and `Full_Test_Set`, and `--xjtu-sy-dir` with the folder of the three condition folders.

## Running

Run the scripts from the repository root after you install the package with the `rl` and
`bearings` extras:

```shell
python experiments/bearings/run_femto_cross_validation.py --data-root DATA --output outputs/bearings-femto --jobs 12
```

`--data-root` defaults to the environment variable `MAINTENANCE_ARENA_DATA`, and `--output` must
be a new directory. `--jobs` sets the processes that read raw files and fit PPO, each with one
thread. `--cache-dir` reuses `<dataset>_statistics.npz` caches and stores missing ones there
(default `OUTPUT/statistics`). `--config-dir` selects another folder of configurations
(default `experiments/bearings/configs`). `--folds`, `--seeds`, `--designs`, `--evaluations`
and `--calendar-steps` restrict a run. A quick check is, for example,
`--folds 0 --seeds 1 --calendar-steps 2048`. `run.json` then records `article_settings: false`.

## Outputs

Every runner writes into its output directory:

- `outcomes.csv` with one row per design, fold, seed and bearing and the columns `design`,
  `dataset`, `fold`, `seed`, `bearing`, `last_productive_minute`, `order_minute`, `cash`,
  `calendar_minutes`, `failed` and `unused_fraction`. `fold` is empty for the transfer, `seed`
  for rules and Reactive, and `order_minute` when no order was placed.
- `selections.json` with the fitted thresholds of the Indicator Rule, the Time rule and the risk
  classifiers and the rule-search scores.
- `fits.json` with one entry per PPO fit, including its first orders on the training
  histories.
- the trained actors below `policies/`.
- `run.json` with the command, configuration hashes, library versions and statistics caches.

The candidate runner adds `selection.json` with the candidate ranking. The nested runner writes
`selection.json` with the ranking of every outer fold instead of `selections.json`, and its
`outcomes.csv` adds the column `inner_fold`. `figures.py` reads these outputs.

## Reproduction

With these configurations, the runners reproduce the bearing results of the article on FEMTO
and XJTU-SY bit for bit. [docs/BEARINGS.md](../../docs/BEARINGS.md#verification) lists the
library versions and what was compared.

## Configurations

| File | Content |
| --- | --- |
| `configs/settings.json` | Economics, features, the prediction model (`forecaster`), grids of the Indicator Rule and the Time rule, PPO training and bootstrap |
| `configs/folds.json` | FEMTO bearing order, the five outer folds, the inner folds of every outer fold and of the all-FEMTO fits |
| `configs/experiments.json` | Seeds of every fold and of the transfer, and the designs, training budgets and selection criteria of every experiment |

`maintenance_arena.bearings.hashed_folds` reproduces the folds from the salts in `derivation`.
With a new salt, it assigns folds for new data.
