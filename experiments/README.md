# Experiments of the accompanying article

The runners in this folder repeat the experiments of *Learning When to Act: What Reinforcement
Learning Agents Learn about Maintenance Timing and When It Transfers* with the MaintenanceArena
package. Each runner reads its settings from JSON configurations next to it, writes into a new
`--output` directory and records the run in `run.json`. Install the package with the `rl` extra,
plus `bearings` for the bearing runners ([README](../README.md#install)), and run them from the
repository root.

| Folder | Experiments | Data | Article |
| --- | --- | --- | --- |
| [synthetic/](synthetic/README.md) | Source learning with the Time rule, input sets, preparation time, adaptation to four new laws, measurement noise, additional calibration comparisons, learner comparison, operating age under lifetime and degradation shift, failure cost | None, every life is rebuilt from seeds | Sections 4, 5 and 6.5, Tables 2 to 4 and A2, Appendices A to D, three additional tables |
| [bearings/](bearings/README.md) | FEMTO cross-validation, transfer to XJTU-SY, age designs, the Time rule, candidate selection and its nested check, rule search, risk classifier | FEMTO and XJTU-SY recordings from their providers ([docs/BEARINGS.md](../docs/BEARINGS.md#data)) | Section 6, Table 5, Appendix D, the additional bearing table |

Each folder's README lists its runners with their options, outputs and run times. Outcome
files have one row per method, run and life or bearing, with the first order and the last
productive step. `maintenance_arena.metrics` and `maintenance_arena.bearings.metrics` summarize
them.

To rebuild the article's tables without rerunning anything, use [paper/](../paper/README.md).
`python paper/rebuild_tables.py --check` recomputes the five data tables of the article and four
additional tables from the saved outcomes in seconds, and two further scripts there compute the
order margins and the transfer split quoted in its text. The outcome files keep the stored method
names, such as Rule and Forecast rule for the article's Time rule
([docs/GLOSSARY.md](../docs/GLOSSARY.md)).

The runners use one thread per process and seeded random streams. On the reference setup
(Windows, CPython 3.12.9, NumPy 2.4.2, PyTorch 2.8.0, scikit-learn 1.8.0, pandas 2.3.0), they
reproduce the article's training runs and outcome rows bit for bit. Another platform or library
build may change the last digits of a fit and then the trained policies.
