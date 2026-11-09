"""Input-set ablation: Full inputs, Deadline only and Health + rate (article Section 4.2).

    python experiments/synthetic/input_sets.py --output outputs/input-sets --jobs 30

Thirty runs (ten seeds per input set, paired across the sets) train to 204,800
steps with checkpoints every 51,200 steps. Each checkpoint is replayed on 800
diagnostic lives (``outcomes_diagnostic.csv``, group = training steps) and the
final policies on 800 test lives (``outcomes_test.csv``, group = law), next to the
Rule calibrated on 256 separate lives, reactive operation and the last-safe reference.
"""
from __future__ import annotations

import common as c
from maintenance_arena import replay as rp


def train(config, output, arguments):
    milestones = c.scaled(config["training"]["milestones"], arguments.budget)
    jobs = []
    for role in config["roles"]:
        bank = c.resolve_bank(config, config["training"]["bank"], role=role)
        for name in config["input_sets"]:
            jobs.append({"label": f"{name}/{role}", "seed": role, "input_set": name, "bank": bank,
                         "directory": str(output / "fits" / name / str(role)), "milestones": milestones,
                         "env_config": config["environment"]})
    return c.run_jobs(c.fit, jobs, arguments.jobs)


def evaluate(config, output, fits):
    prices = c.economics(config)
    rule_config = config["rule"]
    calibration = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, rule_config["calibration_bank"])))
    rule = rp.calibrate_rule(calibration, rule_config["grid"], economics=prices, anchors=rule_config["tie_anchors"])
    c.write_json(output / "rules.json", {"rules": {"Rule": rule["selected"]}, "rule_candidates": rule["candidates"]})
    reference_rows = {}
    results = {}
    for name, spec, final_only in (("diagnostic", config["diagnostic_bank"], False), ("test", config["test_bank"], True)):
        series = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, spec)))
        rows = []
        for record in fits:
            label = config["input_sets"][record["input_set"]]
            checkpoints = record["checkpoints"][-1:] if final_only else record["checkpoints"]
            for entry in checkpoints:
                agent = c.agents({0: c.checkpoint_path(output, record, entry["nominal_calendar_steps"])})[0]
                orders = rp.greedy_orders(c.probabilities(agent, series.states, record["input_set"]), series.slices)
                group = None if final_only else str(entry["nominal_calendar_steps"])
                rows += rp.outcome_rows(label, series, orders, prices, run=record["seed"], group=group)
        group = None if final_only else "all checkpoints"
        rows += rp.outcome_rows("Rule", series, rp.threshold_orders(series, [rule["selected"]])[0], prices, group=group)
        rows += rp.outcome_rows("Reactive", series, [-1] * series.lives, prices, group=group)
        rows += rp.outcome_rows("Last-safe reference", series, rp.reference_orders(series.last_productive), prices,
                                group=group)
        summaries = c.summary_rows(rows, prices, overall=None if not final_only else "all")
        c.write_results(output, name, rows, summaries)
        results[name] = summaries
        reference_rows[name] = rows
    c.write_json(output / "intervals.json", c.intervals(reference_rows["test"], prices, config["contrasts"],
                                                        config["bootstrap"], strata_by_group=True))
    return {"rule": rule["selected"]["threshold"],
            "test": [row for row in results["test"] if row["group"] == "all"]}


def main():
    arguments = c.parser(__doc__.splitlines()[0]).parse_args()
    output, config, record = c.start(arguments, c.HERE / "input_sets.json")
    fits = train(config, output, arguments)
    c.finish(output, record, **evaluate(config, output, fits))


if __name__ == "__main__":
    main()
