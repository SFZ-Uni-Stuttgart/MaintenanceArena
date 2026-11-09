"""Operating age under lifetime and degradation shift: a controlled test of the age input.

    python experiments/synthetic/age_shift.py --output outputs/synthetic/age-shift --jobs 20
    python experiments/synthetic/age_shift.py --setting filtered --exact outputs/synthetic/age-shift --output outputs/synthetic/age-shift-filtered --jobs 20
    python experiments/synthetic/age_shift.py --setting raw --exact outputs/synthetic/age-shift --output outputs/synthetic/age-shift-raw --jobs 20
    python experiments/synthetic/age_shift.py --setting raw --scaled-noise --policies outputs/synthetic/age-shift-raw --exact outputs/synthetic/age-shift --output outputs/synthetic/age-shift-raw-scaled

Ten runs per design train on the source laws for 204,800 steps with the seeds and
training lives of ``source_learning.py``. The design without age is the source
cohort itself. The age design adds ``min(t / 160, 4)`` as a fifth input and starts
from the same network with a zero age column. The test cells pair the 2,000 source
test lives (A) with twins whose lifetime is doubled (B) or quadrupled (B4). Each
twin keeps its damage path over relative age, shape parameters and lifetime quantile.

``--setting exact`` (default) also trains both designs on a wider lifetime
distribution (lognormal scale 0.30) and tests the 800 lives per new law (C) and
their doubled twins (D). ``--setting filtered`` and ``--setting raw`` measure health
with Gaussian error 0.005 and estimate the rate by an eight-step line fit or by
consecutive differences. They train and test only the two source-lifetime
designs on cells A, B and B4. With ``--exact``, an exact-setting output joins the
intervals, so the age effect under noise is compared with the same effect under
exact inputs.

``--scaled-noise`` trains nothing. It replays the policies of a noisy-setting output
(``--policies``) on A, B and B4 and on the twins Bs and B4s, whose white noise is
divided by the lifetime factor (0.0025 and 0.00125). The noise per damage increment
then stays that of the source lives. The measurement streams are those of B and B4.

Outputs follow the other runners: ``outcomes_test.csv`` with group ``<cell>/<law>``,
``summary_test.csv`` per cell and per cell and law, ``rules.json`` with the Time rule
(stored as Rule) fitted on 256 source lives of the setting, ``intervals.json`` and
``descriptives.json``. The intervals resample the ten runs and the lives within each
law together, with the same draw for all methods of a draw group.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import age_shift_banks as ab
import common as c
from maintenance_arena import replay as rp
from maintenance_arena.metrics import read_outcomes
from maintenance_arena.observations import SENSOR_LEVELS

SETTINGS = ("exact", "filtered", "raw")
REFERENCE, REACTIVE = "Last-safe reference", "Reactive"
SHIFTS = (("B", "A"), ("C", "A"), ("D", "C"), ("D", "B"), ("B4", "A"))
PAIRS = (("age", "no_age"), ("age_wide", "age"), ("age_wide", "no_age_wide"), ("no_age_wide", "no_age"),
         ("no_age", "Rule"), ("age", "Rule"))
TWINS = (("A", "B"), ("A", "B4"), ("A", "Bs"), ("A", "B4s"), ("C", "D"))
SCALED = (("Bs", "B"), ("B4s", "B4"))


def environment(config, setting):
    return {**config["environment"], "observation_route": config["settings"][setting]["observation_route"]}


def plan(config, setting, scaled):
    """Cells, draw groups and bootstrap seed of a run."""
    spec = config["scaled_noise"] if scaled else config["settings"][setting]
    return spec["cells"], spec["draw_groups"], spec["bootstrap_seed"]


def train(config, setting, output, arguments):
    spec = config["settings"][setting]
    milestones = c.scaled(config["training"]["milestones"], arguments.budget)
    jobs = []
    for arm in spec["arms"]:
        design = config["arms"][arm]
        for role in config["roles"]:
            bank = c.resolve_bank(config, config["training"]["bank"], role=role, sensor_level=spec["sensor_level"],
                                  lifetime=config["lifetimes"][design["lifetime"]])
            job = {"label": f"{arm}/{role}", "seed": role, "bank": bank, "milestones": milestones,
                   "directory": str(output / "fits" / arm / str(role)), "env_config": environment(config, setting)}
            if design["age"]:
                job["age_input"] = config["age_input"]
            jobs.append(job)
    return c.run_jobs(ab.fit, jobs, arguments.jobs)


def noise_sigma(config, setting, cell):
    """Standard deviation of the white measurement noise in a test cell."""
    return SENSOR_LEVELS[config["settings"][setting]["sensor_level"]][0] * config["cells"][cell].get("noise_factor", 1.)


def cell_series(config, setting, cell):
    """(law, bank specification, decision series) of every bank of a test cell."""
    spec, layout = config["settings"][setting], config["cells"][cell]
    lifetime = config["lifetimes"][layout["lifetime"]]
    banks = ([("source", config["source_test_bank"])] if layout["laws"] == "source"
             else [(law, config["new_law_test_bank"]) for law in config["laws"]])
    result = []
    for law, bank in banks:
        resolved = c.resolve_bank(config, bank, law=law, sensor_level=spec["sensor_level"], lifetime=lifetime)
        if "noise_factor" in layout:
            series = ab.scaled_series(ab.bank_from_spec(resolved), observation_route=spec["observation_route"],
                                      sigma=noise_sigma(config, setting, cell))
        else:
            series = rp.decision_series(ab.bank_from_spec(resolved), observation_route=spec["observation_route"])
        result.append((law, resolved, series))
    return result


def deadline_precision(items):
    """Deadline coordinate at the true last safe order and false deadline signals more than five steps before it."""
    z_last, rate_last, health_last, early = [], [], [], []
    for _, _, series in items:
        for life, (start, stop) in enumerate(series.slices):
            safe = int(series.last_productive[life]) - series.mobilization_lead
            states = series.states[start:stop]
            z_last.append(states[safe, 3])
            rate_last.append(states[safe, 1])
            health_last.append(states[safe, 0])
            before = states[:max(safe - 4, 0)]
            early.append(before[before[:, 2] > .5, 3])
    z, signals = np.asarray(z_last), np.concatenate(early)
    return {"z_at_last_safe_order": dict(zip(("p5", "p25", "p50", "p75", "p95"),
                                             np.percentile(z, [5, 25, 50, 75, 95]).tolist())),
            "z_abs_mean_at_last_safe_order": float(np.mean(np.abs(z))),
            "false_deadline_share_more_than_5_steps_before": float(np.mean(signals >= 0.)),
            "scaled_rate_at_last_safe_order_p50": float(np.median(rate_last)),
            "health_at_last_safe_order_p50": float(np.median(health_last))}


def first_order_ages(table):
    finite = table["first"] >= 0
    safe = (table["last"] - 12)[:, None] + np.zeros_like(table["first"])
    return {"median_first_order_age": float(np.median(table["first"][finite])) if finite.any() else None,
            "median_last_safe_order_age": float(np.median(table["last"] - 12)),
            "orders_at_or_before_age_160_pct": 100. * float(np.mean(finite & (table["first"] <= 160))),
            "last_safe_order_after_age_160_pct": 100. * float(np.mean(safe > 160)),
            "missing_orders_pct": 100. * float(np.mean(~finite))}


def evaluate(config, setting, cells, fits, output, policies=None):
    """Test every run and the Time rule on the cells. With ``policies``, the runs and the rule come from that output."""
    spec = config["settings"][setting]
    prices = c.economics(config)
    rule_config = config["rule"]
    calibration = rp.decision_series(
        c.bank_from_spec(c.resolve_bank(config, rule_config["calibration_bank"], sensor_level=spec["sensor_level"])),
        observation_route=spec["observation_route"])
    rule = rp.calibrate_rule(calibration, rule_config["grid"], economics=prices, anchors=rule_config["tie_anchors"])
    selected = {key: rule["selected"][key] for key in ("family", "threshold")}
    if policies is not None:
        stored = json.loads((Path(policies) / "rules.json").read_text(encoding="utf-8"))["Rule"]
        if any(stored[key] != value for key, value in selected.items()):
            raise SystemExit(f"the Time rule differs from the one in {policies}")
    records = {record["label"]: record for record in fits}
    agents = {(arm, role): ab.policy(c.checkpoint_path(output if policies is None else policies,
                                                       records[f"{arm}/{role}"]))
              for arm in spec["arms"] for role in config["roles"]}
    age = config["age_input"]
    rows, orders, precision = [], {}, {}
    for cell in cells:
        items = cell_series(config, setting, cell)
        orders[cell] = [(series.families[i], int(series.pair_index[i])) for _, _, series in items
                        for i in range(series.lives)]
        precision[cell] = deadline_precision(items)
        for _, _, series in items:
            def group(life, cell=cell, series=series):
                return f"{cell}/{series.families[life]}"
            plain, with_age = series.states, ab.age_states(series, age["scale"], age["clip"])
            for arm in spec["arms"]:
                design = config["arms"][arm]
                for role in config["roles"]:
                    probabilities = c.probabilities(agents[(arm, role)], with_age if design["age"] else plain)
                    rows += rp.outcome_rows(design["method"], series, rp.greedy_orders(probabilities, series.slices),
                                            prices, run=role, group=group)
            rows += rp.outcome_rows("Rule", series, rp.threshold_orders(series, [selected])[0], prices, group=group)
            rows += rp.outcome_rows(REACTIVE, series, [-1] * series.lives, prices, group=group)
            rows += rp.outcome_rows(REFERENCE, series, rp.reference_orders(series.last_productive), prices, group=group)
    # Pooled per cell, a life is its law and pair index, since the new laws share pair indices.
    pooled = [{**row, "group": cell, "unit": f"{law}/{row['unit']}"}
              for row in rows for cell, law in [str(row["group"]).split("/", 1)]]
    summaries = c.summary_rows(pooled, prices, overall=None) + c.summary_rows(rows, prices, overall=None)
    c.write_results(output, "test", rows, summaries)
    c.write_json(output / "rules.json", {"Rule": rule["selected"], "rule_candidates": rule["candidates"]})
    return rows, orders, precision, selected, summaries


def tables_of(config, setting, rows, orders, prefix=""):
    """Lives by runs arrays per (cell, method) for the designs of a setting and the Rule."""
    prices = c.economics(config)
    methods = set([config["arms"][arm]["method"] for arm in config["settings"][setting]["arms"]] + ["Rule"])
    members = {}
    for row in rows:
        cell = str(row["group"]).split("/", 1)[0]
        if row["method"] in methods and cell in orders:
            members.setdefault((cell, row["method"]), []).append(row)
    return {(cell, prefix + method): ab.life_arrays(values, orders[cell], prices)
            for (cell, method), values in members.items()}


def exact_reference(config, directory, orders):
    """The exact-setting tables of the source-lifetime designs and the Rule, for the joint intervals."""
    stored = c.load_config(Path(directory) / "configuration.json")
    if stored["roles"] != config["roles"]:
        raise SystemExit(f"{directory} used other roles; give an exact-setting output with the same --roles")
    shared = {cell: order for cell, order in orders.items() if cell in config["settings"]["exact"]["cells"]}
    rows = [row for row in read_outcomes(Path(directory) / "outcomes_test.csv")
            if row["method"] in ("PPO, no age", "PPO, age", "Rule")]
    tables = tables_of(config, "exact", rows, shared, prefix="exact: ")
    if len(tables) != 3 * len(shared):
        raise SystemExit(f"{directory} lacks cells {sorted(shared)} of the exact setting")
    return tables


def intervals(config, setting, tables, orders, exact, scaled):
    _, draw_groups, seed = plan(config, setting, scaled)
    spec = config["settings"][setting]
    groups = [{"laws": [law for law, _ in orders[cells[0]]], "cells": cells}
              for cells in draw_groups if all(cell in orders for cell in cells)]
    samples = ab.runs_and_lives(tables, groups, len(config["roles"]), replicates=config["bootstrap"]["replicates"],
                                seed=seed, chunk=config["bootstrap"]["chunk"])
    points = ab.point_values(tables)
    name = {arm: config["arms"][arm]["method"] for arm in config["arms"]}
    name["Rule"] = "Rule"
    present = {method for _, method in tables}
    age, no_age = "PPO, age", "PPO, no age"
    result = {}
    for metric in ab.METRICS:
        values = result.setdefault(metric, {})
        for cell, method in sorted(tables):
            values[f"{cell}: {method}"] = ab.contrast(samples, points, [(1, cell, method)], metric)
        if setting == "exact":
            for method in [name[arm] for arm in spec["arms"]] + ["Rule"]:
                for later, earlier in SHIFTS:
                    if (later, method) in tables and (earlier, method) in tables:
                        values[f"{method}: {later} - {earlier}"] = ab.contrast(
                            samples, points, [(1, later, method), (-1, earlier, method)], metric)
            for cell in spec["cells"]:
                for left, right in PAIRS:
                    values[f"{cell}: {name[left]} - {name[right]}"] = ab.contrast(
                        samples, points, [(1, cell, name[left]), (-1, cell, name[right])], metric)
                values[f"{cell}: ({name['age_wide']} - {name['no_age_wide']}) - ({name['age']} - {name['no_age']})"] = ab.gap(
                    samples, points, ((cell, name["age_wide"]), (cell, name["no_age_wide"])),
                    ((cell, name["age"]), (cell, name["no_age"])), metric)
            continue
        with_exact = bool(exact) and f"exact: {age}" in present
        for cell in orders:
            values[f"{cell}: {age} - {no_age}"] = ab.contrast(samples, points, [(1, cell, age), (-1, cell, no_age)], metric)
            if with_exact and (cell, f"exact: {age}") in tables:
                values[f"{cell}: exact: {age} - exact: {no_age}"] = ab.contrast(
                    samples, points, [(1, cell, f"exact: {age}"), (-1, cell, f"exact: {no_age}")], metric)
                values[f"{cell}: ({age} - {no_age}) - (exact: {age} - exact: {no_age})"] = ab.contrast(
                    samples, points, [(1, cell, age), (-1, cell, no_age),
                                      (-1, cell, f"exact: {age}"), (1, cell, f"exact: {no_age}")], metric)
        for new, old in SCALED if scaled else ():
            values[f"({new}: {age} - {no_age}) - ({old}: {age} - {no_age})"] = ab.contrast(
                samples, points, [(1, new, age), (-1, new, no_age), (-1, old, age), (1, old, no_age)], metric)
            if with_exact:
                values[f"({new}: {age} - {no_age}) - ({old}: exact: {age} - exact: {no_age})"] = ab.contrast(
                    samples, points, [(1, new, age), (-1, new, no_age), (-1, old, f"exact: {age}"),
                                      (1, old, f"exact: {no_age}")], metric)
        for method in (no_age, age, "Rule"):
            for later in [cell for cell in orders if cell != "A"]:
                values[f"{method}: {later} - A"] = ab.contrast(samples, points, [(1, later, method), (-1, "A", method)],
                                                               metric)
    return {"replicates": config["bootstrap"]["replicates"], "seed": seed, "chunk": config["bootstrap"]["chunk"],
            "draw_groups": [group["cells"] for group in groups], "metrics": result}


def descriptives(config, setting, tables, precision):
    spec = config["settings"][setting]
    methods = [config["arms"][arm]["method"] for arm in spec["arms"]] + ["Rule"]
    following = {}
    for source, twin in TWINS:
        for method in methods:
            if (source, method) in tables and (twin, method) in tables:
                a, b = tables[(source, method)], tables[(twin, method)]
                following[f"{method}: {source} to {twin}"] = ab.deadline_following(a["first"], a["last"], b["first"], b["last"])
    ages = {f"{cell}: {method}": first_order_ages(table) for (cell, method), table in sorted(tables.items())}
    return {"deadline_following": following, "first_order_ages": ages, "deadline_precision": precision}


def check_arguments(arguments):
    """Refuse inconsistent options before an output directory is created. Returns the run record of ``--policies``."""
    if arguments.exact is not None:
        if arguments.setting == "exact":
            raise SystemExit("--exact applies to the noisy settings")
        if not (arguments.exact / "outcomes_test.csv").is_file():
            raise SystemExit(f"{arguments.exact} holds no outcomes_test.csv")
    if not arguments.scaled_noise:
        if arguments.policies is not None:
            raise SystemExit("--policies applies only with --scaled-noise")
        return None
    if arguments.setting == "exact":
        raise SystemExit("--scaled-noise applies to the noisy settings")
    if arguments.budget is not None:
        raise SystemExit("--scaled-noise trains nothing, so --budget does not apply")
    path = None if arguments.policies is None else arguments.policies / "run.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.is_file() else {}
    if record.get("setting") != arguments.setting or record.get("status") != "complete":
        raise SystemExit(f"--scaled-noise needs --policies, a complete output of --setting {arguments.setting}")
    return record


def main():
    parser = c.parser(__doc__.splitlines()[0])
    parser.add_argument("--setting", choices=SETTINGS, default="exact",
                        help="exact inputs (default) or noisy health with the line fit (filtered) or raw rate")
    parser.add_argument("--exact", type=Path, default=None,
                        help="output of --setting exact, joined into the intervals of a noisy setting")
    parser.add_argument("--scaled-noise", action="store_true",
                        help="replay the policies of --policies on twins with noise divided by the lifetime factor")
    parser.add_argument("--policies", type=Path, default=None, help="output of the same noisy setting (--scaled-noise)")
    arguments = parser.parse_args()
    trained = check_arguments(arguments)
    output, config, record = c.start(arguments, c.HERE / "age_shift.json")
    setting, scaled = arguments.setting, arguments.scaled_noise
    cells, _, _ = plan(config, setting, scaled)
    extra = {}
    if scaled:
        policies = arguments.policies.resolve()
        arms = config["settings"][setting]["arms"]
        fits = list(c.read_fits(policies, [f"{arm}/{role}" for arm in arms for role in config["roles"]]).values())
        record["quick_check"] = record["quick_check"] or bool(trained.get("quick_check"))
        extra = {"scaled_noise": True, "policies": str(policies),
                 "noise_sigma": {cell: noise_sigma(config, setting, cell) for cell in cells}}
        rows, orders, precision, rule, summaries = evaluate(config, setting, cells, fits, output, policies)
    else:
        fits = train(config, setting, output, arguments)
        rows, orders, precision, rule, summaries = evaluate(config, setting, cells, fits, output)
    tables = tables_of(config, setting, rows, orders)
    if arguments.exact is not None:
        tables.update(exact_reference(config, arguments.exact, orders))
    c.write_json(output / "intervals.json", intervals(config, setting, tables, orders, arguments.exact, scaled))
    c.write_json(output / "descriptives.json", descriptives(config, setting, tables, precision))
    c.finish(output, record, setting=setting, rule=rule, test=[row for row in summaries if "/" not in str(row["group"])],
             **extra)


if __name__ == "__main__":
    main()
