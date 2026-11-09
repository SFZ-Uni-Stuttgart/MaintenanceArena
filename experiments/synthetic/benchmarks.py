"""Further calibrations on the lives of Table 2: calibrated cuts, dense and two-indicator rules, failure budgets.

    python experiments/synthetic/benchmarks.py --source outputs/source --output outputs/benchmarks

``--source`` is the output of ``source_learning.py``; no policy is trained. On
2,000 calibration lives every final source policy gets the ORDER cut of the
highest equal-law profit rate among 123 cuts (and the cut of least unused life
within a 1% failure budget). The recalibrated Rule chooses among 802 health and
projected-time thresholds, the two-indicator rule among 236,876 candidates and
the privileged per-law rule one threshold per law. All are replayed on the 2,000
test lives (``outcomes_test.csv``, group = law). The two-indicator search holds
about 1 GB of first orders in memory.
"""
from __future__ import annotations

from pathlib import Path

import common as c
from maintenance_arena import replay as rp


def grids(config):
    single = {family: rp.grid(**spec) for family, spec in config["single_grid"].items()}
    blocks = [{key: rp.grid(**spec) for key, spec in block.items()} for block in config["two_indicator_blocks"]]
    return single, blocks


def evaluate(config, output, source, sources):
    prices = c.economics(config)
    single_grid, blocks = grids(config)
    candidates = rp.threshold_candidates(single_grid)
    calibration = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, config["calibration_bank"])))
    test = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, config["test_bank"])))
    policies = c.agents({role: c.checkpoint_path(source, sources[f"source/{role}"]) for role in config["roles"]})
    budget = config["failure_budget"]
    cuts = {role: rp.calibrate_cut(calibration, c.probabilities(agent, calibration.states), economics=prices,
                                   failure_budget=budget) for role, agent in policies.items()}
    dense = rp.calibrate_rule(calibration, single_grid, economics=prices, anchors=config["tie_anchors"])
    scores = rp.score_orders(calibration, rp.threshold_orders(calibration, candidates), prices)
    budget_index = rp.select_failure_budget(scores.failure_share, scores.unused_share, budget)
    per_law = rp.calibrate_per_law(calibration, single_grid, economics=prices, anchors=config["tie_anchors"])
    two = rp.calibrate_two_indicator(calibration, candidates, blocks, economics=prices)
    rules = {"Rule, recalibrated": dense["selected"], "Two-indicator rule": two["selected"],
             "Per-law rule (privileged)": per_law,
             "Rule, failure-budget cut": None if budget_index is None else {**candidates[budget_index],
                                                                             **scores.candidate(budget_index)},
             "cuts": {str(role): cut for role, cut in cuts.items()}}
    c.write_json(output / "rules.json", rules)
    projection = rp.projected_time(test.states)
    rows = []
    for role, agent in policies.items():
        p = c.probabilities(agent, test.states)
        rows += rp.outcome_rows("PPO + calibrated cut", test,
                                rp.cut_orders(p, test.slices, [cuts[role]["selected"]["cut"]])[0], prices, run=role)
        if cuts[role]["failure_budget"] is not None:
            rows += rp.outcome_rows("PPO, failure-budget cut", test,
                                    rp.cut_orders(p, test.slices, [cuts[role]["failure_budget"]["cut"]])[0], prices,
                                    run=role)
    rows += rp.outcome_rows("Rule, recalibrated", test, rp.threshold_orders(test, [dense["selected"]], projection)[0],
                            prices)
    rows += rp.outcome_rows("Two-indicator rule", test, rp.two_indicator_rule_orders(test, two["selected"], projection),
                            prices)
    rows += rp.outcome_rows("Per-law rule (privileged)", test, rp.per_law_orders(test, per_law, projection), prices)
    if budget_index is not None:
        rows += rp.outcome_rows("Rule, failure-budget cut", test,
                                rp.threshold_orders(test, [candidates[budget_index]], projection)[0], prices)
    rows += rp.outcome_rows("Last-safe reference", test, rp.reference_orders(test.last_productive), prices)
    summaries = c.summary_rows(rows, prices)
    c.write_results(output, "test", rows, summaries)
    c.write_json(output / "intervals.json", c.intervals(rows, prices, config["contrasts"], config["bootstrap"],
                                                        strata_by_group=True))
    return {"rules": {"Rule, recalibrated": dense["selected"]["threshold"],
                      "Two-indicator rule": {k: two["selected"][k] for k in ("health_split", "a_above_split",
                                                                            "b_at_or_below_split")
                                             if k in two["selected"]},
                      "Per-law rule (privileged)": {law: value["threshold"] for law, value in per_law.items()}},
            "test": [row for row in summaries if row["group"] == "all"]}


def main():
    parser = c.parser(__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="output directory of source_learning.py")
    arguments = parser.parse_args()
    if arguments.budget is not None:
        raise SystemExit("this study trains no policy; --budget does not apply")
    output, config, record = c.start(arguments, c.HERE / "benchmarks.json")
    sources = c.read_fits(arguments.source, [f"source/{role}" for role in config["roles"]])
    c.finish(output, record, **evaluate(config, output, arguments.source, sources))


if __name__ == "__main__":
    main()
