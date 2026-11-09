"""Measurement noise: candidate designs, their selection and the evaluation of Table 3 (article Section 4.3).

    python experiments/synthetic/measurement_noise.py --stage development --source outputs/source --output outputs/noise-designs --jobs 10
    python experiments/synthetic/measurement_noise.py --stage selection --development outputs/noise-designs --output outputs/noise-selection
    python experiments/synthetic/measurement_noise.py --source outputs/source --output outputs/noise --jobs 10

``--source`` is the output of ``source_learning.py``.

``--stage development`` trains five kinds of runs on noisy lives: from scratch on
consecutive differences (raw) or on the line fit, each to 819,200 steps on the source
training bank followed by 32,337 further pairs, and fine-tuned from the source policies
to 204,800 steps on the noisy source training bank with the raw estimate, the line fit
or both (eight inputs). Ten candidate designs are checkpoints of these runs. The Time
rule (stored as Recalibrated Rule) is recalibrated on the 256 noisy calibration lives
with each of four state estimates (raw, line fit, both agreeing, either). Four screens
replay further lives (``outcomes_<round>.csv``, group = ``<cell>/<law>``): ``round_one``
to ``round_three`` with greedy decisions, ``round_four`` with one ORDER cut per network
fitted on the calibration lives (``rules.json``). The screens ``round_two`` to
``round_four`` choose the estimate of the Time rule and a design by the failure bounds
of the selection (``selection.json``).

``--stage selection`` recomputes ``selection.json`` from the outcome tables of a
``--stage development`` output.

The default stage (``confirmation``, Table 3): ten runs of the selected design (line fit,
from scratch, 819,200 steps) and the source policies get ORDER cuts fitted on the
256 noisy calibration lives, and the line-fit Time rule is recalibrated on the same
states. The test replays 2,000 lives with noisy and with exact measurements
(``outcomes_test.csv``, group = ``noisy`` or ``exact``) with the same cuts and
threshold. With ``--development`` this stage reuses that output's line-fit runs instead
of training them again.
"""
from __future__ import annotations

import json
from pathlib import Path

import common as c
from maintenance_arena import replay as rp
from maintenance_arena import selection as sel
from maintenance_arena.learning.dual import dual_states
from maintenance_arena.metrics import read_outcomes

STAGES = ("confirmation", "development", "selection")
REFERENCE, REACTIVE = "Last-safe reference", "Reactive"


# --- Default stage: Table 3 ---

def train(config, output, arguments):
    milestones = c.scaled(config["training"]["milestones"], arguments.budget)
    jobs = []
    for role in config["roles"]:
        jobs.append({"label": f"noise/{role}", "seed": role, "bank": extended_bank(config, role),
                     "milestones": milestones, "directory": str(output / "fits" / "noise" / str(role)),
                     "env_config": config["environment"]})
    return c.run_jobs(c.fit, jobs, arguments.jobs)


def extended_bank(config, role):
    """The noisy source training bank of a role followed by its extension pairs."""
    sensor = config["sensor_level"]
    extension = c.resolve_bank(config, config["training"]["extension"], role=role, sensor_level=sensor)
    return c.resolve_bank(config, config["training"]["bank"], role=role, sensor_level=sensor, extensions=[extension])


def reuse(config, arguments):
    """The selected design's runs from a --stage development output; they must equal the runs of the default stage."""
    run = config["development"]["designs"][config["confirmed_design"]]["run"]
    records = c.read_fits(arguments.development, [f"{run}/{role}" for role in config["roles"]])
    milestones = c.scaled(config["training"]["milestones"], arguments.budget)
    for role, record in zip(config["roles"], records.values()):
        if (record["bank"] != {**extended_bank(config, role), "sha256": record["bank"]["sha256"]}
                or record["env_config"] != config["environment"] or record["seed"] != role
                or record["checkpoints"][-1]["nominal_calendar_steps"] != milestones[-1]):
            raise SystemExit(f"{record['label']} differs from the run of the default stage; train it here instead")
    return list(records.values())


def evaluate(config, output, fits, fits_output, source, sources):
    prices = c.economics(config)
    route = config["environment"]["observation_route"]
    calibration = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, config["calibration_bank"],
                                                                     sensor_level=config["sensor_level"])),
                                     observation_route=route)
    rule = rp.calibrate_rule(calibration, config["rule"]["grid"], economics=prices,
                             anchors=config["rule"]["tie_anchors"])["selected"]
    networks = {("Noise-trained PPO", record["seed"]): c.checkpoint_path(fits_output, record) for record in fits}
    networks.update({("Source PPO", role): c.checkpoint_path(source, sources[f"source/{role}"])
                     for role in config["roles"]})
    policies = c.agents(networks)
    cuts = {}
    for key, agent in policies.items():
        cuts[key] = rp.calibrate_cut(calibration, c.probabilities(agent, calibration.states), economics=prices)["selected"]
    c.write_json(output / "rules.json", {"Recalibrated Rule": rule,
                                         "cuts": [{"method": method, "run": run, **cut} for (method, run), cut in cuts.items()]})
    rows = []
    for cell, sensor in config["cells"].items():
        series = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, config["test_bank"], sensor_level=sensor)),
                                    observation_route=route)
        for (method, run), agent in policies.items():
            orders = rp.cut_orders(c.probabilities(agent, series.states), series.slices, [cuts[(method, run)]["cut"]])[0]
            rows += rp.outcome_rows(method, series, orders, prices, run=run, group=cell)
        rows += rp.outcome_rows("Recalibrated Rule", series, rp.threshold_orders(series, [rule])[0], prices, group=cell)
        rows += rp.outcome_rows(REACTIVE, series, [-1] * series.lives, prices, group=cell)
        rows += rp.outcome_rows(REFERENCE, series, rp.reference_orders(series.last_productive), prices, group=cell)
    summaries = c.summary_rows(rows, prices, overall=None)
    c.write_results(output, "test", rows, summaries)
    c.write_json(output / "intervals.json", {cell: c.intervals([row for row in rows if row["group"] == cell], prices,
                                                               config["contrasts"], config["bootstrap"])
                                             for cell in config["cells"]})
    return {"rule": rule["threshold"], "test": summaries}


# --- Candidate designs and their screens ---

def development_milestones(config, budget):
    """Milestones of every run; a smaller ``--budget`` scales all runs by the same factor and keeps every milestone."""
    runs = config["development"]["runs"]
    longest = max(spec["milestones"][-1] for spec in runs.values())
    if budget is None:
        return {name: [int(m) for m in spec["milestones"]] for name, spec in runs.items()}
    if not 1 <= budget <= longest:
        raise SystemExit(f"--budget must lie in 1..{longest}")
    result = {}
    for name, spec in runs.items():
        points = [max(1, round(m * budget / longest)) for m in spec["milestones"]]
        if any(a >= b for a, b in zip(points, points[1:])):
            raise SystemExit(f"--budget {budget} is too small for the milestones of {name}")
        result[name] = points
    return result


def design_milestone(config, milestones, design):
    spec = config["development"]["designs"][design]
    return milestones[spec["run"]][config["development"]["runs"][spec["run"]]["milestones"].index(spec["steps"])]


def development_jobs(config, output, arguments, sources, milestones):
    dev = config["development"]
    jobs = []
    # Longest runs first, so the parallel pool ends early.
    for name, spec in sorted(dev["runs"].items(), key=lambda item: -item[1]["milestones"][-1]):
        both = spec["estimate"] == "both"
        for role in config["roles"]:
            bank = (extended_bank(config, role) if spec["bank"] == "extended" else
                    c.resolve_bank(config, config["training"]["bank"], role=role, sensor_level=config["sensor_level"]))
            job = {"label": f"{name}/{role}", "seed": role, "bank": bank, "milestones": milestones[name],
                   "directory": str(output / "fits" / name / str(role)),
                   "env_config": {**config["environment"],
                                  "observation_route": dev["routes"]["raw" if both else spec["estimate"]]}}
            if both:
                job["both_estimates"] = True
            if spec["start"] == "source":
                job["initialize"] = str(c.checkpoint_path(arguments.source, sources[f"source/{role}"]))
            jobs.append(job)
    return jobs


def estimates(bank, routes):
    """Decision series of the raw and line-fit estimates and the states of every estimate (``both``: eight inputs)."""
    series = {name: rp.decision_series(bank, observation_route=route) for name, route in routes.items()}
    if series["raw"].slices != series["line_fit"].slices:
        raise AssertionError("the two estimates must describe the same decisions")
    states = {name: values.states for name, values in series.items()}
    states["both"] = dual_states(states["raw"], states["line_fit"])
    return series, states


def calibrate_rules(config, prices, series):
    grid, anchors = config["rule"]["grid"], config["rule"]["tie_anchors"]
    rules = {}
    for name, spec in config["development"]["rules"].items():
        kind = spec["estimate"]
        if kind in series:
            result = rp.calibrate_rule(series[kind], grid, economics=prices, anchors=anchors)
        else:
            result = rp.calibrate_dual_rule(series["raw"], series["line_fit"], grid, kind, economics=prices,
                                            anchors=anchors)
        rules[name] = {"estimate": kind, "selected": result["selected"]}
    return rules


def rule_orders(rule, series):
    if rule["estimate"] in series:
        return rp.threshold_orders(series[rule["estimate"]], [rule["selected"]])[0]
    return rp.dual_threshold_orders(series["raw"], series["line_fit"], [rule["selected"]], rule["estimate"])[0]


def development_networks(config, output, fits, milestones, source, sources):
    """(method, role) -> (checkpoint, estimate) for every design and control."""
    dev = config["development"]
    records = {record["label"]: record for record in fits}
    result = {}
    for design, spec in dev["designs"].items():
        milestone = design_milestone(config, milestones, design)
        for role in config["roles"]:
            result[(design, role)] = (c.checkpoint_path(output, records[f"{spec['run']}/{role}"], milestone),
                                      dev["runs"][spec["run"]]["estimate"])
    for control, spec in dev["controls"].items():
        for role in config["roles"]:
            result[(control, role)] = (c.checkpoint_path(source, sources[f"source/{role}"]), spec["estimate"])
    return result


def replay_round(config, spec, prices, policies, estimate_of, cuts, rules):
    rows = []
    for cell, sensor in spec["cells"].items():
        bank = c.bank_from_spec(c.resolve_bank(config, spec["bank"], sensor_level=sensor))
        series, states = estimates(bank, config["development"]["routes"])
        base = series["raw"]

        def group(life, cell=cell, base=base):
            return f"{cell}/{base.families[life]}"
        for method in [*spec["designs"], *spec["descriptive"], *spec["controls"]]:
            for role in config["roles"]:
                probabilities = c.probabilities(policies[(method, role)], states[estimate_of[(method, role)]])
                if spec["decision"] == "greedy":
                    orders = rp.greedy_orders(probabilities, base.slices)
                else:
                    orders = rp.cut_orders(probabilities, base.slices, [cuts[(method, role)]["cut"]])[0]
                rows += rp.outcome_rows(method, base, orders, prices, run=role, group=group)
        for name in spec["rules"]:
            rows += rp.outcome_rows(name, base, rule_orders(rules[name], series), prices, group=group)
        rows += rp.outcome_rows(REFERENCE, base, rp.reference_orders(base.last_productive), prices, group=group)
        rows += rp.outcome_rows(REACTIVE, base, [-1] * base.lives, prices, group=group)
    return rows


def select_round(config, spec, rows):
    """Screen values and the selection of one screen from its outcome rows (the noisy cell)."""
    dev = config["development"]
    cell = next(name for name, level in spec["cells"].items() if level == config["sensor_level"])
    noisy = [row for row in rows if str(row["group"]).split("/", 1)[0] == cell]
    arrays, laws = sel.outcome_arrays(noisy, law=lambda row: str(row["group"]).split("/", 1)[1])
    values = sel.screen_values(arrays, laws, reference=REFERENCE)
    rule = sel.select_rule_estimator(values, spec["rules"], dev["selection"])
    budgets = {name: dev["designs"][name][spec["budget"]] for name in spec["designs"]}
    design = sel.select_design(values, spec["designs"], budgets, rule["F_rule"], dev["selection"])
    result = {"rule": rule, "design": design, "matching_rule": None}
    if design["design"] is not None:
        estimate = dev["runs"][dev["designs"][design["design"]]["run"]]["estimate"]
        available = {kind: [name for name in names if name in spec["rules"]]
                     for kind, names in dev["matching_rules"].items()}
        result["matching_rule"] = sel.matching_rule(values, estimate, available, dev["selection"])
    return values, result


def selection_record(config, results):
    """Every screen's selection and the final one: the last screen that selected a design."""
    final = None
    for name, result in results.items():
        if result["design"]["design"] is not None:
            final = {"round": name, "design": result["design"]["design"], "rule_estimate": result["rule"]["rule"],
                     "matching_rule": result["matching_rule"],
                     "passing": result["design"]["passing"], "ties": result["design"]["ties"],
                     "bound_pct": result["design"]["bound_pct"]}
    agrees = final is not None and (final["design"], final["matching_rule"]) == (config["confirmed_design"],
                                                                               config["confirmed_rule"])
    return {"rounds": results, "selected": final, "confirmed_design": config["confirmed_design"],
            "confirmed_rule": config["confirmed_rule"], "selection_equals_confirmed_design": agrees,
            "cut_grid": {"values": len(rp.CUT_GRID), "anchor_index": rp.CUT_ANCHOR_INDEX,
                         "anchor": rp.CUT_GRID[rp.CUT_ANCHOR_INDEX]}}


def failure_overview(rows, prices):
    """Equal-life summaries per method and cell (pooled over the laws) and per method, cell and law."""
    pooled = [{**row, "group": str(row["group"]).split("/", 1)[0]} for row in rows]
    return c.summary_rows(pooled, prices, overall=None) + c.summary_rows(rows, prices, overall=None)


def development(config, output, arguments, sources):
    dev = config["development"]
    milestones = development_milestones(config, arguments.budget)
    fits = c.run_jobs(c.fit, development_jobs(config, output, arguments, sources, milestones), arguments.jobs)
    prices = c.economics(config)
    calibration, calibration_states = estimates(
        c.bank_from_spec(c.resolve_bank(config, config["calibration_bank"], sensor_level=config["sensor_level"])),
        dev["routes"])
    rules = calibrate_rules(config, prices, calibration)
    networks = development_networks(config, output, fits, milestones, arguments.source, sources)
    policies = c.agents({key: path for key, (path, _) in networks.items()})
    estimate_of = {key: estimate for key, (_, estimate) in networks.items()}
    cut_methods = {method for spec in dev["rounds"] if spec["decision"] == "cut"
                   for method in [*spec["designs"], *spec["descriptive"], *spec["controls"]]}
    cuts = {key: rp.calibrate_cut(calibration["raw"], c.probabilities(agent, calibration_states[estimate_of[key]]),
                                  economics=prices)["selected"]
            for key, agent in policies.items() if key[0] in cut_methods}
    c.write_json(output / "rules.json", {
        "rules": rules, "cut_grid": list(rp.CUT_GRID), "cut_anchor_index": rp.CUT_ANCHOR_INDEX,
        "cuts": [{"method": method, "run": run, **cut} for (method, run), cut in cuts.items()]})
    results, overview = {}, {}
    for spec in dev["rounds"]:
        rows = replay_round(config, spec, prices, policies, estimate_of, cuts, rules)
        summaries = failure_overview(rows, prices)
        c.write_results(output, spec["name"], rows, summaries)
        overview[spec["name"]] = [row for row in summaries if "/" not in str(row["group"])]
        if spec["select"]:
            values, results[spec["name"]] = select_round(config, spec, rows)
            c.write_json(output / f"screen_{spec['name']}.json", values)
    record = selection_record(config, results)
    c.write_json(output / "selection.json", record)
    return {"stage": "development", "rules": {name: rule["selected"]["threshold"] for name, rule in rules.items()},
            "selected": record["selected"], "rounds": overview}


# --- Selection from the stored screens ---

def selection(config, output, development_output):
    results = {}
    for spec in config["development"]["rounds"]:
        if spec["select"]:
            values, results[spec["name"]] = select_round(
                config, spec, read_outcomes(development_output / f"outcomes_{spec['name']}.csv"))
            c.write_json(output / f"screen_{spec['name']}.json", values)
    record = selection_record(config, results)
    stored = development_output / "selection.json"
    if stored.is_file():
        record["equals_development_selection"] = json.loads(json.dumps(record)) == json.loads(stored.read_text("utf-8"))
    c.write_json(output / "selection.json", record)
    return {"stage": "selection", "selected": record["selected"],
            "equals_development_selection": record.get("equals_development_selection")}


# --- Command line ---

def check_inputs(arguments):
    """Refuse missing inputs before an output directory is created."""
    if arguments.stage == "selection":
        if arguments.development is None:
            raise SystemExit("--stage selection needs --development")
        if arguments.budget is not None or arguments.roles is not None:
            raise SystemExit("--budget and --roles do not apply to the selection stage")
        for spec in c.load_config(arguments.config or arguments.development / "configuration.json")["development"]["rounds"]:
            if spec["select"] and not (arguments.development / f"outcomes_{spec['name']}.csv").is_file():
                raise SystemExit(f"{arguments.development} holds no outcomes_{spec['name']}.csv")
        return
    if arguments.source is None:
        raise SystemExit(f"--stage {arguments.stage} needs --source")
    if not (arguments.source / "fits" / "source").is_dir():
        raise SystemExit(f"{arguments.source} holds no source runs; run source_learning.py first")
    if arguments.development is not None and arguments.stage != "confirmation":
        raise SystemExit("--development applies to the selection and confirmation stages")
    if arguments.development is not None and not (arguments.development / "fits").is_dir():
        raise SystemExit(f"{arguments.development} holds no runs of --stage development")


def main():
    parser = c.parser(__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=STAGES, default="confirmation",
                        help="candidate designs and their screens, the selection from their outcomes, or Table 3 (default)")
    parser.add_argument("--source", type=Path, default=None, help="output directory of source_learning.py")
    parser.add_argument("--development", type=Path, default=None,
                        help="output of --stage development (selection; default stage: reuse its line-fit runs)")
    arguments = parser.parse_args()
    check_inputs(arguments)
    if arguments.stage == "selection":
        output, config, record = c.start(arguments, arguments.development / "configuration.json")
        c.finish(output, record, **selection(config, output, arguments.development))
        return
    output, config, record = c.start(arguments, c.HERE / "measurement_noise.json")
    sources = c.read_fits(arguments.source, [f"source/{role}" for role in config["roles"]])
    if arguments.stage == "development":
        c.finish(output, record, **development(config, output, arguments, sources))
        return
    fits, fits_output = ((reuse(config, arguments), arguments.development) if arguments.development is not None
                         else (train(config, output, arguments), output))
    c.finish(output, record, stage="confirmation", **evaluate(config, output, fits, fits_output, arguments.source,
                                                              sources))


if __name__ == "__main__":
    main()
