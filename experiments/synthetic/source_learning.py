"""Source learning at L = 12: ten PPO runs, the Time rule and further controls (article Section 4.1, Table 2).

    python experiments/synthetic/source_learning.py --output outputs/source --jobs 10

The runs train on their own source banks with preserved preparation to 204,800
calendar steps and save checkpoints at 51,200, 102,400 and 204,800 steps. The
Time rule threshold on the projected remaining time (stored as Rule) is
calibrated on 256 separate source lives. The final policies (ORDER when P(ORDER) > P(WAIT)) and the rules are then
replayed on 800 validation lives and on the 2,000 test lives of Table 2, whose
age and health cutoffs are calibrated on 2,000 further lives.
"""
from __future__ import annotations

import common as c
from maintenance_arena import replay as rp


def train(config, output, arguments):
    milestones = c.scaled(config["training"]["milestones"], arguments.budget)
    jobs = [{"label": f"source/{role}", "seed": role, "directory": str(output / "fits" / "source" / str(role)),
             "bank": c.resolve_bank(config, config["training"]["bank"], role=role), "milestones": milestones,
             "env_config": config["environment"]} for role in config["roles"]]
    return {record["label"]: record for record in c.run_jobs(c.fit, jobs, arguments.jobs)}


def evaluate(config, output, fits):
    prices = c.economics(config)
    rule_config = config["rule"]
    calibration = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, rule_config["calibration_bank"])))
    rule = rp.calibrate_rule(calibration, rule_config["grid"], economics=prices, anchors=rule_config["tie_anchors"])
    selected = {key: rule["selected"][key] for key in ("family", "threshold")}
    policies = c.agents({record["seed"]: c.checkpoint_path(output, record) for record in fits.values()})
    comparison = config["comparison"]
    calibration_2000 = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, comparison["calibration_bank"])))
    health = rp.grid(**comparison["health_grid"])
    health_rule = rp.calibrate_rule(calibration_2000, {"health": health}, economics=prices,
                                    anchors=rule_config["tie_anchors"])["selected"]
    low, high = comparison["ages"]
    age_rule = rp.calibrate_age(calibration_2000, range(low, high + 1), economics=prices)["selected"]
    rules = {"Rule": rule["selected"], "Health threshold": health_rule, "Age replacement": age_rule,
             "Naive projection rule": {"family": "projection", "threshold": float(comparison["naive_projection_threshold"])}}
    c.write_json(output / "rules.json", {"rules": rules, "rule_candidates": rule["candidates"]})
    results = {}
    for name, spec, extra in (("development_800", config["development_bank"], False),
                              ("test_2000", comparison["test_bank"], True)):
        series = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, spec)))
        projection = rp.projected_time(series.states)
        rows = []
        if extra:
            rows += rp.outcome_rows("Naive projection rule", series,
                                    rp.threshold_orders(series, [rules["Naive projection rule"]], projection)[0], prices)
        for role, agent in policies.items():
            rows += rp.outcome_rows("PPO", series, rp.greedy_orders(c.probabilities(agent, series.states), series.slices),
                                    prices, run=role)
        rows += rp.outcome_rows("Rule", series, rp.threshold_orders(series, [selected], projection)[0], prices)
        if extra:
            rows += rp.outcome_rows("Age replacement", series, rp.age_orders(series, [age_rule["age"]])[0], prices)
            rows += rp.outcome_rows("Health threshold", series, rp.threshold_orders(series, [health_rule], projection)[0],
                                    prices)
        rows += rp.outcome_rows("Reactive", series, [-1] * series.lives, prices)
        rows += rp.outcome_rows("Last-safe reference", series, rp.reference_orders(series.last_productive), prices)
        summaries = c.summary_rows(rows, prices)
        c.write_results(output, name, rows, summaries)
        results[name] = summaries
        if extra:
            c.write_json(output / "intervals.json", c.intervals(rows, prices, config["contrasts"], config["bootstrap"],
                                                                strata_by_group=True))
    return {"rules": {name: value.get("threshold", value.get("age")) for name, value in rules.items()},
            "test_2000": [row for row in results["test_2000"] if row["group"] == "all"]}


def main():
    arguments = c.parser(__doc__.splitlines()[0]).parse_args()
    output, config, record = c.start(arguments, c.HERE / "source_learning.json")
    fits = train(config, output, arguments)
    c.finish(output, record, **evaluate(config, output, fits))


if __name__ == "__main__":
    main()
