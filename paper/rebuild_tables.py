"""Rebuild the five data tables of the article and four additional tables from the saved first orders.

Usage from any working directory::

    python paper/rebuild_tables.py                  # print the tables as Markdown
    python paper/rebuild_tables.py --output DIR     # write the LaTeX table bodies and tables.md into a new DIR
    python paper/rebuild_tables.py --check          # compare the LaTeX bodies with paper/expected

Each outcome row is one evaluated life, or one bearing, of one method and run. It keeps the
last productive step T and the step of the first order. Cash, calendar steps, failure and
unused life follow from these two numbers and the accounting of the article
(``consequences``). Every table entry first averages the runs on each life and then weights
the lives equally (``summarize``). ``--check`` exits with status 1 if a rebuilt table
differs from its expected file in any byte, and every command exits with status 2 if an
input is missing or invalid. Only the Python standard library is needed.

The outcome files keep the stored method names. All tables print the article's names, for
example Time rule for the stored Rule and Forecast rule. The four additional tables hold further
comparisons on the same lives and bearings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
import csv
from decimal import ROUND_HALF_UP, Decimal
import difflib
import math
from pathlib import Path
import sys
from typing import NamedTuple

PAPER = Path(__file__).resolve().parent
OUTCOMES = PAPER / "outcomes"
EXPECTED = PAPER / "expected"

# Accounting of the article (Section 3.1 and Appendix A), in currency units (CU) and steps.
PREPARATION = 12            # L: steps from the order to the exchange
PLANNED_OUTAGE = 7          # one exchange step and six downtime steps
REVENUE = 255.0             # per productive step
NONPRODUCTIVE_COST = 25.0   # per nonproductive step
REPLACEMENT_COST = 520.0    # per replacement
FAILURE_SURCHARGE = 7000.0  # per failure, beyond downtime and lost production
REFERENCE = "Last-safe reference"
METRICS = ("failures", "unused_life", "retained_profit")


class Outcome(NamedTuple):
    """One outcome row: the first order of one method and run on one life (or bearing)."""
    method: str
    condition: str
    group: str
    fold: str
    run: str
    unit: str
    last_productive_step: int
    order_step: int | None


class Study(NamedTuple):
    """Layout of one outcome file: its columns, the lives per group and the runs per method."""
    file: str
    columns: tuple[str, ...]
    lives: dict[str, int]
    runs: dict[str, int]
    conditions: tuple[str, ...] = ("",)


COLUMNS = ("method", "group", "run", "unit", "last_productive_step", "order_step")
SOURCE_LAWS = {"Linear": 500, "Concave": 500, "Convex": 500, "Late-knee": 500}
STUDIES = {
    "source_comparison": Study("source_comparison.csv", COLUMNS, SOURCE_LAWS, {
        "Naive projection rule": 1, "PPO": 10, "Rule": 1, "Age replacement": 1, "Health threshold": 1,
        "Reactive": 1, "PPO + calibrated cut": 10, "Rule, recalibrated": 1, "Two-indicator rule": 1,
        "Per-law rule (privileged)": 1, "PPO, failure-budget cut": 10, "Rule, failure-budget cut": 1}),
    "measurement_noise": Study("measurement_noise.csv", ("method", "condition", *COLUMNS[1:]), SOURCE_LAWS,
                               {"Source PPO": 10, "Noise-trained PPO": 10, "Recalibrated Rule": 1},
                               ("noisy", "exact")),
    "degradation_change": Study("degradation_change.csv", COLUMNS,
                                {"Paris-inspired": 800, "Two-knee": 800, "Gompertz": 800, "Power-law": 800},
                                {"Source PPO": 10, "Source Rule": 1, "Fine-tuned PPO": 10,
                                 "Fine-tuned PPO, calibrated cut": 10, "Fresh PPO": 10,
                                 "Fresh PPO, full budget": 10, "Recalibrated Rule": 1}),
    "bearings": Study("bearings.csv", ("method", "group", "fold", *COLUMNS[2:]), {"FEMTO": 17, "XJTU-SY": 15}, {
        "PPO: Full": 5, "PPO: No policy age": 5, "PPO: No forecaster age": 5, "PPO: No age": 5,
        "Risk classifier: Full": 1, "Risk classifier: No age": 1,
        "Forecast rule: Full": 1, "Forecast rule: No age": 1, "Indicator Rule": 1, "Reactive": 1}),
    "learner_comparison": Study("learner_comparison.csv", COLUMNS, SOURCE_LAWS, {
        "PPO": 10, "DQN": 10, "Double DQN": 10, "Naive projection rule": 1, "Rule": 1}),
}
FEMTO_FOLDS = {"0", "1", "2", "3", "4"}
# Trainable parameters of the learners of Table A2: actor and critic, or the online Q-network.
# tests/test_learners.py checks them against maintenance_arena.learning.learners.
PARAMETERS = {"PPO": 9155, "DQN": 4610, "Double DQN": 4610}

THREE_METRICS = r"\shortstack{Failures\\(\%)} & \shortstack{Unused life\\(\%)} & \shortstack{Retained\\profit (\%)}"
HEADER = "Method & " + THREE_METRICS + r"\\"


class Table(NamedTuple):
    """One table: blocks of stored methods (rows), their printed labels and the column groups that split the lives."""
    stem: str
    name: str  # "Table 2" for a table of the article, "Additional table" for the others
    title: str
    study: str
    blocks: tuple[tuple[str, ...], ...]
    splits: tuple[tuple[str, str], ...] = ()  # (field, value) of each column group; () uses all lives
    metrics: tuple[str, ...] = METRICS
    size: str = r"\footnotesize"
    head: tuple[str, ...] = (HEADER,)
    markdown_head: tuple[str, ...] = ("Method", "Failures (%)", "Unused life (%)", "Retained profit (%)")
    extras: tuple[str, ...] = ()  # "failure_free_runs" and "parameters" after the metrics
    column_separation: str = "3pt"
    labels: tuple[tuple[str, str], ...] = ()  # (stored method, printed label); other methods print as stored


NOISE = dict(splits=(("condition", "noisy"), ("condition", "exact")), metrics=("failures", "unused_life"),
             size=r"\small",
             head=(r" & \multicolumn{2}{c}{Noisy inputs} & \multicolumn{2}{c}{Exact inputs}\\",
                   r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
                   r"Procedure & Failures & Unused life & Failures & Unused life\\"),
             markdown_head=("Procedure", "Failures, noisy (%)", "Unused life, noisy (%)", "Failures, exact (%)",
                            "Unused life, exact (%)"))
BEARINGS = dict(splits=(("group", "FEMTO"), ("group", "XJTU-SY")),
                head=(r" & \multicolumn{3}{c}{FEMTO held-out} & \multicolumn{3}{c}{XJTU-SY transfer}\\",
                      r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
                      "Method & " + THREE_METRICS + " & " + THREE_METRICS + r"\\"),
                markdown_head=("Method", "FEMTO failures (%)", "FEMTO unused life (%)", "FEMTO retained profit (%)",
                               "XJTU-SY failures (%)", "XJTU-SY unused life (%)", "XJTU-SY retained profit (%)"))
LEARNERS = dict(extras=("failure_free_runs", "parameters"), column_separation="2.5pt",
                head=(r"Method & \shortstack{Failures\\(\%)} & \shortstack{Unused\\life (\%)} & "
                      r"\shortstack{Retained\\profit (\%)} & \shortstack{Failure-\\free runs} & Parameters\\",),
                markdown_head=("Method", "Failures (%)", "Unused life (%)", "Retained profit (%)", "Failure-free runs",
                               "Parameters"))
# Printed names of stored methods that several tables share.
TIME_RULES = (("Naive projection rule", r"Time rule, $\tau=13$"), ("Rule", r"Time rule, $\tau=15$"))
BEARING_NAMES = (("PPO: No forecaster age", "PPO: No prediction age"), ("Forecast rule: Full", "Time rule: Full"),
                 ("Forecast rule: No age", "Time rule: No age"))
TABLES = (
    Table("table2_source_comparison", "Table 2", "Source decision quality on the same 2,000 test lives at L = 12",
          "source_comparison", (("Naive projection rule", "PPO", "Rule"),), labels=TIME_RULES),
    Table("table3_measurement_noise", "Table 3",
          "Filtered inputs and calibrated thresholds on the same 2,000 source-law test lives",
          "measurement_noise", (("Source PPO", "Noise-trained PPO", "Recalibrated Rule"),), **NOISE,
          labels=(("Source PPO", r"\shortstack[l]{Source PPO\\+ filter + calibration}"),
                  ("Recalibrated Rule", "Time rule"))),
    Table("table4_degradation_change", "Table 4",
          "The same 800 test lives per new law, averaged equally over the four laws", "degradation_change",
          (("Source PPO", "Source Rule"),
           ("Fine-tuned PPO", "Fine-tuned PPO, calibrated cut", "Fresh PPO", "Fresh PPO, full budget",
            "Recalibrated Rule")),
          labels=(("Source Rule", "Time rule: source"),
                  ("Fine-tuned PPO, calibrated cut", "Fine-tuned PPO + calibration"),
                  ("Recalibrated Rule", "Time rule: adapted"))),
    Table("table5_bearings", "Table 5", "Held-out FEMTO and transfer XJTU-SY bearings", "bearings",
          (("PPO: Full", "PPO: No policy age", "PPO: No forecaster age", "PPO: No age"),
           ("Risk classifier: Full", "Risk classifier: No age"),
           ("Forecast rule: Full", "Forecast rule: No age")), **BEARINGS, labels=BEARING_NAMES),
    Table("tableA2_learners", "Table A2",
          "Learners trained on the source laws, on 2,000 new lives shared by all methods",
          "learner_comparison", (("PPO", "DQN", "Double DQN"), ("Rule",)), **LEARNERS, labels=(("Rule", "Time rule"),)),
    # Additional tables: further comparisons on the same lives and bearings.
    Table("additional_source_controls", "Additional table", "Source controls on the same 2,000 test lives as Table 2",
          "source_comparison",
          (("Naive projection rule", "PPO", "Rule"), ("Age replacement", "Health threshold", "Reactive"),
           (REFERENCE,)), labels=TIME_RULES),
    Table("additional_calibration_comparisons", "Additional table",
          "Further calibrations on the same 2,000 test lives as Table 2", "source_comparison",
          (("PPO + calibrated cut", "Rule, recalibrated", "Two-indicator rule"),
           ("Per-law rule (privileged)", "PPO, failure-budget cut", "Rule, failure-budget cut")),
          labels=(("PPO + calibrated cut", "PPO + calibration"), ("Rule, recalibrated", "Time rule, recalibrated"),
                  ("PPO, failure-budget cut", "PPO + calibration, failure budget"),
                  ("Rule, failure-budget cut", "Time rule, failure budget"))),
    Table("additional_learner_comparison", "Additional table",
          "All learners and both Time rules on the lives of Table A2", "learner_comparison", (("PPO", "DQN", "Double DQN"), ("Naive projection rule", "Rule")), **LEARNERS,
          labels=TIME_RULES),
    Table("additional_bearing_comparison", "Additional table",
          "Bearing designs with the Indicator rule and reactive operation, on the bearings of Table 5", "bearings",
          (("PPO: Full", "PPO: No policy age", "PPO: No forecaster age", "PPO: No age"),
           ("Forecast rule: Full", "Forecast rule: No age"), ("Indicator Rule", "Reactive")), **BEARINGS,
          labels=(*BEARING_NAMES, ("Indicator Rule", "Indicator rule"))),
)
Blocks = list[list[tuple[str, list[str]]]]


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mean(values: Iterable[float]) -> float:
    """Exactly rounded mean, independent of the order of the values."""
    values = list(values)
    return math.fsum(values) / len(values)


def consequences(last_productive_step: int, order_step: int | None) -> tuple[float, float, int, float]:
    """Cash (CU), calendar steps, failure indicator and unused productive steps of one life.

    A timely order (order + L <= T) produces until its preparation ends, then has the planned
    outage of seven nonproductive steps. A late or missing order produces until T and fails; the
    preparation still missing then runs without production before the planned outage.
    """
    timely = order_step is not None and order_step + PREPARATION <= last_productive_step
    productive = order_step + PREPARATION if timely else last_productive_step
    missing = 0 if timely else PREPARATION if order_step is None else PREPARATION - (last_productive_step - order_step)
    nonproductive = PLANNED_OUTAGE + missing
    cash = (REVENUE * productive - REPLACEMENT_COST - NONPRODUCTIVE_COST * nonproductive
            - FAILURE_SURCHARGE * (not timely))
    return cash, float(productive + nonproductive), int(not timely), float(last_productive_step - productive)


def summarize(outcomes: Iterable[Outcome]) -> dict[str, float]:
    """Failure share, unused life and retained profit (all in percent) with equal life weights.

    On each life, failures and unused fractions are averaged over the runs; retained profit is
    the run-averaged cash over the run-averaged calendar steps, relative to the rate of the
    last-safe order on the same life. The lives are then averaged with equal weights.
    """
    lives: dict[tuple[str, str, str], list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        lives[outcome.condition, outcome.group, outcome.unit].append(outcome)
    require(lives, "No outcomes to summarize")
    failures, unused, retained = [], [], []
    for (_, group, unit), items in lives.items():
        last = items[0].last_productive_step
        require(all(item.last_productive_step == last for item in items),
                f"{group} unit {unit}: different last productive steps")
        results = [consequences(last, item.order_step) for item in items]
        reference_cash, reference_calendar, _, _ = consequences(last, last - PREPARATION)
        cash, calendar = mean(r[0] for r in results), mean(r[1] for r in results)
        failures.append(100 * mean(r[2] for r in results))
        unused.append(100 * mean(r[3] / (last - PREPARATION) for r in results))
        retained.append(100 * (cash / calendar) / (reference_cash / reference_calendar))
    return {"failures": mean(failures), "unused_life": mean(unused), "retained_profit": mean(retained)}


def _integer(text: str, label: str) -> int:
    require(text.isascii() and text.isdecimal(), f"{label} must be a nonnegative integer, not {text!r}")
    return int(text)


def _parse(study: Study, values: dict[str, str], where: str) -> Outcome:
    method, group, run, unit = values["method"], values["group"], values["run"], values["unit"]
    condition, fold = values.get("condition", ""), values.get("fold", "")
    require(method in study.runs, f"{where}: unknown method {method!r}")
    require(group in study.lives, f"{where}: unknown group {group!r}")
    require(condition in study.conditions, f"{where}: unknown condition {condition!r}")
    require(run == "" or (run.isascii() and run.isdecimal()), f"{where}: run must be empty or an integer, not {run!r}")
    require(unit != "", f"{where}: the unit is empty")
    last = _integer(values["last_productive_step"], f"{where}: last_productive_step")
    require(last > PREPARATION, f"{where}: last_productive_step {last} leaves no time for the {PREPARATION}-step "
                                "preparation")
    order = None if values["order_step"] == "" else _integer(values["order_step"], f"{where}: order_step")
    require(order is None or order <= last, f"{where}: order_step {order} lies after the last productive step {last}")
    return Outcome(method, condition, group, fold, run, unit, last, order)


def _validate(study: Study, outcomes: Sequence[Outcome]) -> None:
    """Every method covers the same lives with the documented counts, each life once per run."""
    cohorts: dict[tuple[str, str], dict[tuple[str, str], list[Outcome]]] = defaultdict(lambda: defaultdict(list))
    for outcome in outcomes:
        cohorts[outcome.method, outcome.condition][outcome.group, outcome.unit].append(outcome)
    shared: dict[str, dict[tuple[str, str], int]] = {}
    for method, runs in study.runs.items():
        roster = sorted(str(index) for index in range(runs)) if runs > 1 else [""]
        for condition in study.conditions:
            label = f"{study.file}: {method}" + (f" ({condition})" if condition else "")
            cohort = cohorts.get((method, condition))
            require(cohort, f"{label}: no rows")
            lives = {}
            for (group, unit), items in cohort.items():
                found = sorted(item.run for item in items)
                require(found == roster, f"{label}: {group} unit {unit} has runs {found}; expected {roster}")
                steps = {item.last_productive_step for item in items}
                require(len(steps) == 1, f"{label}: {group} unit {unit} has different last productive steps")
                lives[group, unit] = steps.pop()
            counts = Counter(group for group, _ in lives)
            require(counts == Counter(study.lives), f"{label}: lives per group {dict(counts)}; expected {study.lives}")
            require(shared.setdefault(condition, lives) == lives,
                    f"{label}: not evaluated on the same lives (unit and last productive step) as the other methods")
    if "fold" in study.columns:
        folds: dict[tuple[str, str], set[str]] = defaultdict(set)
        for outcome in outcomes:
            folds[outcome.group, outcome.unit].add(outcome.fold)
        for (group, unit), values in sorted(folds.items()):
            if group == "FEMTO":
                require(len(values) == 1 and values <= FEMTO_FOLDS,
                        f"{study.file}: FEMTO {unit} needs one outer fold from 0 to 4, found {sorted(values)}")
            else:
                require(values == {""}, f"{study.file}: {group} {unit} has no outer fold, found {sorted(values)}")


def read_study(name: str, directory: Path = OUTCOMES) -> list[Outcome]:
    """Read and validate one outcome file."""
    study = STUDIES[name]
    path = Path(directory) / study.file
    outcomes = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader, ()))
        require(header == study.columns,
                f"{study.file}: columns {', '.join(header) or '(none)'}; expected {', '.join(study.columns)}")
        for values in reader:
            where = f"{study.file} line {reader.line_num}"
            require(len(values) == len(header), f"{where}: {len(values)} fields; expected {len(header)}")
            outcomes.append(_parse(study, dict(zip(header, values)), where))
    _validate(study, outcomes)
    return outcomes


def load(directory: Path = OUTCOMES) -> dict[str, list[Outcome]]:
    """Read and validate the five outcome files."""
    return {name: read_study(name, directory) for name in STUDIES}


def reference(outcomes: Iterable[Outcome]) -> list[Outcome]:
    """The last-safe reference on the same lives: one order exactly L steps before the last productive step."""
    lives = {}
    for outcome in outcomes:
        lives[outcome.condition, outcome.group, outcome.unit] = outcome._replace(
            method=REFERENCE, run="", order_step=outcome.last_productive_step - PREPARATION)
    return list(lives.values())


def cell(value: float) -> str:
    """The article's rounding: 0 and 100 as integers, $<0.01$ below 0.005, else two decimals half up."""
    if value in (0, 100):
        return str(int(value))
    if 0 < value < 0.005:
        return r"$<0.01$"
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def extra(name: str, study: str, method: str, outcomes: Sequence[Outcome]) -> str:
    """Failure-free runs (runs without a failed life, of all runs) or trainable parameters; -- for a rule."""
    runs = STUDIES[study].runs[method]
    if runs == 1:
        return "--"
    if name == "parameters":
        return f"{PARAMETERS[method]:,}"
    failed = {item.run for item in outcomes if consequences(item.last_productive_step, item.order_step)[2]}
    return f"{runs - len(failed)}/{runs}"


def compute(studies: dict[str, list[Outcome]]) -> dict[str, Blocks]:
    """The rounded cells of every table as blocks of (row label, cells)."""
    rows: dict[tuple[str, str], list[Outcome]] = defaultdict(list)
    for name, outcomes in studies.items():
        for outcome in outcomes:
            rows[name, outcome.method].append(outcome)
        rows[name, REFERENCE] = reference(outcomes)
    summaries: dict[tuple[str, str, tuple[str, str] | None], dict[str, float]] = {}
    results = {}
    for table in TABLES:
        blocks: Blocks = []
        for block in table.blocks:
            lines = []
            for method in block:
                require((table.study, method) in rows, f"{table.name}: no outcomes of {method!r}")
                cells = []
                for split in table.splits or (None,):
                    key = (table.study, method, split)
                    if key not in summaries:
                        selected = rows[table.study, method]
                        if split is not None:
                            selected = [row for row in selected if getattr(row, split[0]) == split[1]]
                        summaries[key] = summarize(selected)
                    cells += [cell(summaries[key][metric]) for metric in table.metrics]
                cells += [extra(name, table.study, method, rows[table.study, method]) for name in table.extras]
                lines.append((dict(table.labels).get(method, method), cells))
            blocks.append(lines)
        results[table.stem] = blocks
    return results


def latex(table: Table, blocks: Blocks) -> str:
    """The LaTeX table body in the layout of the article's table files."""
    columns = "l" + "r" * (len(table.metrics) * max(1, len(table.splits)) + len(table.extras))
    lines = ["{" + table.size + r"\setlength{\tabcolsep}{" + table.column_separation
             + r"}\renewcommand{\arraystretch}{1.08}%",
             r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule", *table.head]
    for block in blocks:
        lines.append(r"\midrule")
        lines += [label + " & " + " & ".join(cells) + r"\\" for label, cells in block]
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}}", ""])


def plain(label: str) -> str:
    """A row label without LaTeX markup, for Markdown."""
    for old, new in ((r"\shortstack[l]{", ""), (r"\\", " "), ("}", ""), (r"$\tau=", "τ = "), ("$", "")):
        label = label.replace(old, new)
    return label


def markdown(results: dict[str, Blocks]) -> str:
    """All tables as one Markdown document."""
    lines = ["# Tables rebuilt from the saved outcomes", "",
             "Rebuilt by `paper/rebuild_tables.py` from `paper/outcomes`. All entries are percentages with equal "
             "weight per life or bearing; rows with several runs first average the runs on each life.", ""]
    for table in TABLES:
        lines += [f"## {table.name}: {table.title}", "", "| " + " | ".join(table.markdown_head) + " |",
                  "| --- |" + " ---: |" * (len(table.markdown_head) - 1)]
        for block in results[table.stem]:
            lines += ["| " + " | ".join([plain(label), *(c.replace("$<0.01$", "<0.01") for c in cells)]) + " |"
                      for label, cells in block]
        lines.append("")
    return "\n".join(lines)


def build(directory: Path = OUTCOMES) -> dict[str, str]:
    """The nine LaTeX table bodies, by file stem, rebuilt from the outcome files in ``directory``."""
    results = compute(load(directory))
    return {table.stem: latex(table, results[table.stem]) for table in TABLES}


def check(texts: dict[str, str], expected: Path = EXPECTED) -> list[str]:
    """Compare the rebuilt tables with the expected files byte for byte; return the stems that differ."""
    differing = []
    for table in TABLES:
        path = Path(expected) / f"{table.stem}.tex"
        if not path.is_file():
            print(f"{table.name}: {path} is missing", file=sys.stderr)
            differing.append(table.stem)
            continue
        recorded, rebuilt = path.read_bytes(), texts[table.stem]
        if recorded == rebuilt.encode("utf-8"):
            print(f"{table.name} ({path.name}): identical")
            continue
        differing.append(table.stem)
        old = recorded.decode("utf-8", errors="replace")
        if old.replace("\r\n", "\n") == rebuilt:
            print(f"{table.name} ({path.name}): differs only in line endings; the expected file must use LF",
                  file=sys.stderr)
            continue
        print(f"{table.name} ({path.name}): differs", file=sys.stderr)
        diff = difflib.unified_diff(old.splitlines(), rebuilt.splitlines(), f"expected/{path.name}", "rebuilt",
                                    lineterm="")
        sys.stderr.writelines(line + "\n" for line in diff)
    return differing


def write(directory: Path, texts: dict[str, str], document: str) -> None:
    """Write the LaTeX bodies and tables.md into a new directory; never overwrite an existing one."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite the existing output {directory}")
    directory.mkdir(parents=True)
    for stem, text in texts.items():
        (directory / f"{stem}.tex").write_bytes(text.encode("utf-8"))
    (directory / "tables.md").write_bytes(document.encode("utf-8"))
    print(f"Wrote {len(texts)} LaTeX tables and tables.md to {directory}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Compare the rebuilt LaTeX tables with the expected files; exit with status 1 on any "
                             "difference")
    parser.add_argument("--output", type=Path,
                        help="Write the LaTeX table bodies and tables.md into this new directory")
    parser.add_argument("--outcomes", type=Path, default=OUTCOMES,
                        help="Directory of the outcome files (default: outcomes next to this script)")
    parser.add_argument("--expected", type=Path, default=EXPECTED,
                        help="Directory of the expected tables for --check (default: expected next to this script)")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.exit(2, f"rebuild_tables: refusing to overwrite the existing output {args.output}\n")
    try:
        results = compute(load(args.outcomes))
        texts = {table.stem: latex(table, results[table.stem]) for table in TABLES}
        if args.output is not None:
            write(args.output, texts, markdown(results))
        differing = check(texts, args.expected) if args.check else []
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"rebuild_tables: {error}\n")
    if args.check:
        if differing:
            print(f"{len(differing)} of {len(TABLES)} tables differ from {args.expected}", file=sys.stderr)
            return 1
        print(f"All {len(TABLES)} tables are identical to the expected files")
    elif args.output is None:
        sys.stdout.write(markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
