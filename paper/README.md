# Saved outcomes and table rebuild

This folder holds the saved decisions behind the five data tables of the article *Learning When
to Act: What Reinforcement Learning Agents Learn about Maintenance Timing and When It
Transfers* (Tables 2 to 5 and A2). The same decisions give four additional tables with further
comparisons on the same lives and bearings. For every evaluated life or bearing, method and
trained run, the files keep the step of the first order and the last productive step. These two
numbers determine cash, downtime, failure and unused life, so `rebuild_tables.py` can recompute
every table entry from single decisions. It reproduces the LaTeX table bodies of the article
byte for byte.

## Contents

| Path | Content |
| --- | --- |
| `rebuild_tables.py` | Rebuilds the nine tables from `outcomes/` as LaTeX table bodies and as Markdown, and checks them against `expected/` |
| `order_margins.py` | Median order margins and per-bearing age effects that the text of the article quotes |
| `transfer_split.py` | Age effect after transfer within and beyond the FEMTO lifetime range, with the bearing bootstrap of the article |
| `outcomes/` | One CSV file per study, one row per evaluated life or bearing, method and run (393,232 rows, 16 MB) |
| `expected/` | The nine table bodies: the five that the LaTeX source of the article includes and the four additional tables |
| `configs/` | Experiment settings, training seeds, fold assignments and the selected thresholds and cuts, one JSON file per study, and `economics.json` for the shared accounting |

## Rebuild the tables

The script needs only CPython 3.12 and its standard library, not `maintenance_arena`, and runs
from any working directory:

```shell
python paper/rebuild_tables.py                      # print the nine tables as Markdown
python paper/rebuild_tables.py --output DIR         # write nine .tex table bodies and tables.md into the new DIR
python paper/rebuild_tables.py --check              # compare the rebuilt .tex bodies with expected/
```

The expected files of the article are `table2_source_comparison.tex`,
`table3_measurement_noise.tex`, `table4_degradation_change.tex`, `table5_bearings.tex` and
`tableA2_learners.tex`. The additional tables are:

- `additional_source_controls.tex`: age replacement, a health threshold, reactive operation and
  the last-safe reference next to the methods of Table 2.
- `additional_calibration_comparisons.tex`: PPO + calibration, the Time rule recalibrated on a
  denser grid, a two-indicator rule, a per-law rule and selections with a failure budget, on the
  lives of Table 2.
- `additional_learner_comparison.tex`: the learners of Table A2 with the Time rule at projected
  time 13 and 15.
- `additional_bearing_comparison.tex`: the PPO designs and Time rules of Table 5 with the
  Indicator rule and reactive operation.

`--output` writes files of the same names and refuses an existing directory. `--check` prints
one line per table and exits with status 0 only if every rebuilt table equals its expected
file. A difference gives status 1 and a line diff. A missing or invalid outcome file gives
status 2, with a message that names the file, the line or the method. `--outcomes DIR` and
`--expected DIR` select other directories, for example a modified copy of the outcomes. A
rebuild takes a few seconds, and `tests/test_paper_tables.py` runs the same check.

Before computing anything, the script validates every file. It checks the columns, the integer
steps, that no order lies after the last productive step and the documented number of lives per
law or dataset. It also checks that every run appears once on every life and that all methods of
a study share the same lives and last productive steps.

## Values in the text

Two further scripts compute values that the article quotes in its text from the same files:

```shell
python paper/order_margins.py                       # median order margins and per-bearing age effects
python paper/transfer_split.py                      # age effect after transfer within and beyond the FEMTO range
```

The margin of an order is (T − o) − L, zero at the last safe order and negative for a late one.
`order_margins.py` gives its median on the source laws and on every new law, with missing orders
left out of the median and counted as late, and the age effect on every XJTU-SY bearing. Like
`rebuild_tables.py`, it needs only the standard library. `transfer_split.py` compares PPO: No age
with PPO: Full on the XJTU-SY bearings within and beyond the longest FEMTO test (468 minutes) and
without the longest bearings. It uses the paired bearing bootstrap of `maintenance_arena.bearings`,
so it needs the installed package, and it first checks that all fifteen bearings give the
interval of the article, −17.03 [−34.27, −0.92]. Both scripts print JSON, and `--output FILE`
also writes it to a new file. `tests/test_paper_text_values.py` checks the quoted values.

## Outcome files

| File | Tables | Rows | Lives | Stored methods |
| --- | --- | ---: | --- | --- |
| `source_comparison.csv` | 2, source controls | 78,000 | 2,000 test lives, 500 per source law | Naive projection rule; PPO; Rule; Age replacement; Health threshold; Reactive |
| | further calibrations | | the same lives | PPO + calibrated cut; Rule, recalibrated; Two-indicator rule; Per-law rule (privileged); PPO, failure-budget cut; Rule, failure-budget cut |
| `measurement_noise.csv` | 3 | 84,000 | 2,000 test lives other than those of Table 2, 500 per source law, each with noisy and with exact observations | Source PPO; Noise-trained PPO; Recalibrated Rule |
| `degradation_change.csv` | 4 | 166,400 | 800 test lives per new law | Source PPO; Source Rule; Fine-tuned PPO; Fine-tuned PPO, calibrated cut; Fresh PPO; Fresh PPO, full budget; Recalibrated Rule |
| `bearings.csv` | 5, bearing comparison | 832 | 17 held-out FEMTO and 15 transfer XJTU-SY bearings | PPO: Full; PPO: No policy age; PPO: No forecaster age; PPO: No age; Risk classifier: Full; Risk classifier: No age; Forecast rule: Full; Forecast rule: No age; Indicator Rule; Reactive |
| `learner_comparison.csv` | A2, learner comparison | 64,000 | 2,000 new lives other than those of Tables 2 and 3, 500 per source law | PPO; DQN; Double DQN; Naive projection rule; Rule |

The files keep the stored method names, and every table prints the names of the article. Rule
and Naive projection rule are the Time rule at projected time 15 and 13. Recalibrated Rule is the
Time rule of Table 3, and Source Rule and Recalibrated Rule are Time rule: source and Time rule:
adapted in Table 4. Fine-tuned PPO, calibrated cut is Fine-tuned PPO + calibration, and PPO +
calibrated cut is PPO + calibration. The forecast rules are Time rule: Full and Time rule: No
age, and PPO: No forecaster age is PPO: No prediction age.

Learned methods have ten runs in the synthetic studies and five per bearing. The rules, the risk
classifiers and reactive operation have one deterministic run. The additional bearing table keeps
eight bearing methods, without the risk classifiers. The files hold only the test rows that the
tables use, without calibration, validation or training outcomes. The script computes the
last-safe reference of the source controls from the lives.

The files are UTF-8 CSV with LF line endings and these columns:

| Column | Meaning |
| --- | --- |
| `method` | The stored method name, as listed above. Names with a comma are quoted, as CSV requires. |
| `condition` | `measurement_noise.csv` only: `noisy` or `exact` observations of the same life. |
| `group` | The degradation law of the life (source laws Linear, Concave, Convex and Late-knee, new laws Paris-inspired, Two-knee, Gompertz and Power-law) or the dataset (FEMTO, XJTU-SY). |
| `fold` | `bearings.csv` only: the outer cross-validation fold (0 to 4) that holds out a FEMTO bearing. Empty for XJTU-SY. |
| `run` | Index of the trained policy, 0 to 9 or 0 to 4. Empty for a method with one deterministic run. The configuration files map each index to its training seed. |
| `unit` | Index of the life in the test set (in its law for `degradation_change.csv`), or the bearing name. |
| `last_productive_step` | T, the last productive step. In the synthetic studies it is the failure age minus one, for the bearings the recorded test end in minutes. |
| `order_step` | o, the step of the first ORDER. Empty if the method did not order. |

The runners' columns `cash`, `calendar_steps`, `failed` and `unused_fraction` are not stored,
because they follow from T and o. `consequences(T, o)` in `rebuild_tables.py` returns the cash,
the calendar steps, the failure indicator and the unused productive steps, and
`unused_fraction` is the unused steps over T − L. The rows come from the stored evaluation
records of the article's experiments, and each was checked against the values stored there.

The script's functions also work on their own. This example counts the timely replays of Full
PPO on FEMTO that Section 6.2 of the article quotes:

```python
import sys
sys.path.insert(0, "paper")  # from the repository root
import rebuild_tables as rt

rows = [row for row in rt.read_study("bearings") if row.method == "PPO: Full" and row.group == "FEMTO"]
timely = sum(1 - rt.consequences(row.last_productive_step, row.order_step)[2] for row in rows)
print(timely, "of", len(rows))  # 62 of 85
```

## Metrics

All studies use the accounting of the article (Section 3.1 and Appendix A) with L = 12
preparation steps, priced in `configs/economics.json`. For one run on one life, with T the last
productive step and o the first order:

- An order is timely if o ≤ T − L, the last safe order. It produces U = o + L steps, followed by
  N = 7 nonproductive steps: one exchange step and six downtime steps.
- A late or missing order fails (F = 1) after U = T productive steps. The completed preparation
  is kept, and the missing preparation runs without production. N = 7 + L − (T − o) after a late
  order and 7 + L without an order.
- Cash is 255 U − 520 − 25 N − 7,000 F currency units. Calendar steps are D = U + N, and the
  unused productive steps are T − U, zero for a failure.

A table entry summarizes one method with all its runs on the n lives of a column group, such as
the noisy observations in Table 3 or the FEMTO bearings in Table 5 (Section 3.5 of the
article):

- Failures (%) = 100 · mean over lives of the mean over runs of F.
- Unused life (%) = 100 · mean over lives of the mean over runs of (T − U) / (T − L). This is
  the positive order margin divided by the time from installation to the last safe order.
- Retained profit (%) = 100 · mean over lives of (mean cash / mean D) / (cash_ref / D_ref).
  Cash and calendar steps are averaged over the runs of a life before their ratio is taken. The
  reference orders at the last safe order of the same life.

Every life or bearing has equal weight. The synthetic test sets are balanced, so the laws have
equal weight too, and Table 4 averages its 3,200 lives, 800 per new law. Means use `math.fsum`,
so they do not depend on the order of the rows. Cells show two decimals, rounded half up. 0 and
100 appear as integers, and values below 0.005 as `$<0.01$` in LaTeX and `<0.01` in Markdown.

Table A2 has two more columns. Failure-free runs counts the runs without a failed life, from the
same rows. Parameters is the number of trainable weights of the network, actor and critic for
PPO and the online network for DQN and double DQN. The script holds these three numbers as
constants, and `tests/test_learners.py` checks them against the package.

## Configurations

| File | Content |
| --- | --- |
| `economics.json` | Preparation, exchange and downtime steps, prices, the recovery after a failure and the accounting formulas |
| `source_comparison.json` | Table 2 and the additional source tables: test and calibration lives, the ten source policies and their training seeds, every rule threshold, the per-run PPO cuts, candidate grids and calibration objectives |
| `measurement_noise.json` | Table 3: noise level, the line-fit state estimate, the per-run cuts of both PPO procedures and the line-fit Time rule threshold |
| `degradation_change.json` | Table 4: the four new laws with their adapted Time rule thresholds, the calibrated cut of every fine-tuned run, training budgets including the full-budget fresh runs, training seeds and the derivation of the life and learner seeds |
| `bearings.json` | Table 5 and the additional bearing table: outer and inner FEMTO folds with their derivation, PPO designs, training seeds per fold, the risk classifiers with their settings and thresholds, the Indicator rule of each fold and of the transfer, and the Time rule cutoffs |
| `learner_comparison.json` | Table A2 and the additional learner table: test lives, training seeds and training lives, the recipes and parameter counts of the three learners and the two Time rule thresholds |

Each file starts with a short description. The seeds of the synthetic lives are those of the
scenario banks in `maintenance_arena.banks`, so the runners in
[experiments/](../experiments/README.md) rebuild the same lives.

## Limits

- The files reproduce the tables from stored decisions. They hold no trained policies,
  synthetic degradation paths, bearing recordings or predictions. The runners in
  [experiments/](../experiments/README.md) repeat the experiments, and the bearing runners need
  the FEMTO and XJTU-SY datasets ([docs/BEARINGS.md](../docs/BEARINGS.md#data)). Their outcome
  files record the same two numbers in their own column layout.
- Only these nine tables and the values of the two scripts above are rebuilt. Figures and the
  other paired bootstrap intervals behind statements that a difference is resolved or
  unresolved are not. Several counts in the text, such as the one above, follow from the same
  files.
- The controlled age study of the article (Section 6.5 and Appendix D) has no table. Its numbers
  are quoted in the text, and its runner in [experiments/](../experiments/README.md) repeats it.
- Thresholds, cuts and procedures were selected on calibration and validation lives before the
  test lives were evaluated. The configuration files record these selections but not their
  outcomes, so the selections cannot be repeated from this folder. Table A2 selects nothing. It
  keeps every final policy and the Time rule threshold of Table 2.
- The transfer to XJTU-SY covers one pair of bearing datasets. As the article states, it is
  exploratory, and a further bearing population would show how generally it holds.
