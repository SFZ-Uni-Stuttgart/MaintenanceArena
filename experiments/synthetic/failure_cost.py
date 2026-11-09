"""Failure cost of the source benchmark: PPO and the Time rule at 3,500 and 14,000 CU (article Section 4.1 and Appendix C).

    python experiments/synthetic/failure_cost.py --output outputs/synthetic/failure-cost --jobs 20

Ten runs per failure cost train like the source policies of ``source_learning.py``,
with the same seeds and training lives. Only the failure cost of the training
environment changes. The Time rule (stored as Rule) is calibrated per cost on the
256 source calibration lives. PPO and the Rule are replayed on the 2,000 test lives
of Table 2 and priced with the cost of their arm. The cost of 7,000 CU is the source
cohort of Table 2. ``outcomes_test.csv`` stores the methods as ``PPO, <cost> CU`` and
``Rule, <cost> CU``, ``rules.json`` holds the Rule of every cost and ``margins.json``
the order margins (T - L) - o of every policy and life.
"""
from __future__ import annotations

from dataclasses import replace
import statistics

import common as c
from maintenance_arena import replay as rp


def arm(cost) -> str:
    return f"cost{int(cost)}"


def prices_for(config, cost) -> rp.Economics:
    return replace(c.economics(config), failure_cost=float(cost))


def train(config, output, arguments):
    milestones = c.scaled(config["training"]["milestones"], arguments.budget)
    jobs = []
    for cost in config["failure_costs"]:
        environment = {**config["environment"], "failure_event_cost": float(cost)}
        for role in config["roles"]:
            jobs.append({"label": f"{arm(cost)}/{role}", "seed": role, "milestones": milestones,
                         "directory": str(output / "fits" / arm(cost) / str(role)),
                         "bank": c.resolve_bank(config, config["training"]["bank"], role=role),
                         "env_config": environment})
    return c.run_jobs(c.fit, jobs, arguments.jobs)


def margins(rows, lead):
    """Median, quartiles and late share of the order margins (T - L) - o; a missing order counts as late."""
    values = [row["last_productive_step"] - lead - row["order_step"] for row in rows
              if row["order_step"] is not None and row["order_step"] >= 0]
    missing = len(rows) - len(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "quartiles": [quartiles[0], quartiles[2]],
            "late_pct": 100.0 * (sum(v < 0 for v in values) + missing) / len(rows), "missing": missing}


def evaluate(config, fits_root, fits):
    """Rows, summaries, Rules and margins of every cost; ``fits_root`` holds the checkpoints of ``fits``."""
    rule_config = config["rule"]
    calibration = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, rule_config["calibration_bank"])))
    test = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, config["test_bank"])))
    projection = rp.projected_time(test.states)
    rows, summaries, rules, margin = [], [], {}, {}
    for cost in config["failure_costs"]:
        prices = prices_for(config, cost)
        rule = rp.calibrate_rule(calibration, rule_config["grid"], economics=prices,
                                 anchors=rule_config["tie_anchors"])["selected"]
        rules[arm(cost)] = rule
        ppo, threshold = f"PPO, {int(cost)} CU", f"Rule, {int(cost)} CU"
        cost_rows = []
        for record in fits:
            if record["label"].startswith(f"{arm(cost)}/"):
                agent = c.agents({0: c.checkpoint_path(fits_root, record)})[0]
                orders = rp.greedy_orders(c.probabilities(agent, test.states), test.slices)
                cost_rows += rp.outcome_rows(ppo, test, orders, prices, run=record["seed"])
        cost_rows += rp.outcome_rows(threshold, test, rp.threshold_orders(test, [rule], projection)[0], prices)
        rows += cost_rows
        summaries += c.summary_rows(cost_rows, prices, by_group=False)
        margin.update({name: margins([row for row in cost_rows if row["method"] == name], prices.mobilization_lead)
                       for name in (ppo, threshold)})
    return rows, summaries, rules, margin


def main():
    arguments = c.parser(__doc__.splitlines()[0]).parse_args()
    output, config, record = c.start(arguments, c.HERE / "failure_cost.json")
    fits = train(config, output, arguments)
    rows, summaries, rules, margin = evaluate(config, output, fits)
    c.write_results(output, "test", rows, summaries)
    c.write_json(output / "rules.json", rules)
    c.write_json(output / "margins.json", margin)
    c.finish(output, record, rules={name: rule["threshold"] for name, rule in rules.items()}, test=summaries)


if __name__ == "__main__":
    main()
