"""Adaptation to the four new laws: source policies unchanged, fine-tuned and fresh (article Section 5).

    python experiments/synthetic/adaptation.py --source outputs/source --output outputs/adaptation --jobs 40

``--source`` is the output of ``source_learning.py``. For each new law and seed,
fine-tuning starts from the weights of the final source policy (fresh optimizer,
random streams and bank) and trains 51,200 further steps; a fresh policy trains
the same budget from scratch, with the same seed and training lives. Both save
checkpoints at 12,800 and 51,200 steps. A second fresh policy with the same seed
trains the full source budget of 204,800 steps on a longer bank whose first lives
are those of the other runs, with checkpoints at 12,800, 51,200, 102,400 and
204,800 steps. On 800 test lives per law (``outcomes_test.csv``, group = law) the
final policies are compared with the unchanged source policies and source Rule,
the Rule recalibrated on 256 lives of the law, reactive operation and the
last-safe reference. Each fine-tuned policy also gets an ORDER cut fitted on the
same 256 lives with the same calendar-profit criterion as the Rule; its orders
are stored as Fine-tuned PPO, calibrated cut, and ``rules.json`` keeps the cuts.
"""
from __future__ import annotations

import json
from pathlib import Path

import common as c
from maintenance_arena import replay as rp

ARMS = ("fine-tuned", "fresh", "fresh-full")
METHODS = {"fine-tuned": "Fine-tuned PPO", "fresh": "Fresh PPO", "fresh-full": "Fresh PPO, full budget"}
CUT_METHOD = "Fine-tuned PPO, calibrated cut"


def calibrated_cut_orders(agent, calibration, series, probabilities, prices):
    """ORDER cut of one policy fitted on the calibration lives of its law, and its first orders on the test lives."""
    selected = rp.calibrate_cut(calibration, c.probabilities(agent, calibration.states), economics=prices)["selected"]
    return selected, rp.cut_orders(probabilities, series.slices, [float(selected["cut"])])[0]


def train(config, output, arguments, sources):
    full = config["full_budget"]
    milestones = c.scaled(config["training"]["milestones"], arguments.budget)
    full_milestones = c.scaled(full["milestones"], arguments.budget)
    jobs = []
    for law in config["laws"]:
        for role in config["roles"]:
            bank = c.resolve_bank(config, config["training"]["bank"], law=law, role=role)
            seed = c.seed_of(config, config["learner_seed_purpose"], law, role)
            for arm in ARMS:
                job = {"label": f"{law}/{role}/{arm}", "seed": seed, "bank": bank, "milestones": milestones,
                       "directory": str(output / "fits" / law / str(role) / arm), "env_config": config["environment"]}
                if arm == "fine-tuned":
                    job["initialize"] = str(c.checkpoint_path(arguments.source, sources[f"source/{role}"]))
                if arm == "fresh-full":
                    job["bank"] = c.resolve_bank(config, config["training"]["bank"], law=law, role=role,
                                                 pairs=full["pairs"])
                    job["milestones"] = full_milestones
                jobs.append(job)
    return c.run_jobs(c.fit, jobs, arguments.jobs)


def evaluate(config, output, fits, source, sources):
    prices = c.economics(config)
    source_rule = json.loads((Path(source) / "rules.json").read_text(encoding="utf-8"))["rules"]["Rule"]
    source_agents = c.agents({role: c.checkpoint_path(source, sources[f"source/{role}"]) for role in config["roles"]})
    rule_config = config["rule"]
    rows, rules, cuts = [], {}, {}
    for law in config["laws"]:
        calibration = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, rule_config["calibration_bank"], law=law)))
        rules[law] = rp.calibrate_rule(calibration, rule_config["grid"], economics=prices,
                                       anchors=rule_config["tie_anchors"])["selected"]
        series = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, config["test_bank"], law=law)))
        projection = rp.projected_time(series.states)
        for role, agent in source_agents.items():
            rows += rp.outcome_rows("Source PPO", series,
                                    rp.greedy_orders(c.probabilities(agent, series.states), series.slices), prices, run=role)
        rows += rp.outcome_rows("Source Rule", series, rp.threshold_orders(series, [source_rule], projection)[0], prices)
        for record in fits:
            fit_law, role, arm = record["label"].split("/")
            if fit_law != law:
                continue
            agent = c.agents({0: c.checkpoint_path(output, record)})[0]
            probabilities = c.probabilities(agent, series.states)
            rows += rp.outcome_rows(METHODS[arm], series, rp.greedy_orders(probabilities, series.slices), prices,
                                    run=int(role))
            if arm == "fine-tuned":
                selected, orders = calibrated_cut_orders(agent, calibration, series, probabilities, prices)
                cuts.setdefault(law, {})[role] = selected
                rows += rp.outcome_rows(CUT_METHOD, series, orders, prices, run=int(role))
        rows += rp.outcome_rows("Recalibrated Rule", series, rp.threshold_orders(series, [rules[law]], projection)[0],
                                prices)
        rows += rp.outcome_rows("Reactive", series, [-1] * series.lives, prices)
        rows += rp.outcome_rows("Last-safe reference", series, rp.reference_orders(series.last_productive), prices)
    summaries = c.summary_rows(rows, prices)
    c.write_results(output, "test", rows, summaries)
    c.write_json(output / "rules.json", {"source_rule": source_rule, "recalibrated_rules": rules,
                                         "calibrated_cuts": cuts})
    c.write_json(output / "intervals.json", c.intervals(rows, prices, config["contrasts"], config["bootstrap"],
                                                        strata_by_group=True))
    return {"rules": {law: rule["threshold"] for law, rule in rules.items()},
            "test": [row for row in summaries if row["group"] == "all"]}


def main():
    parser = c.parser(__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="output directory of source_learning.py")
    arguments = parser.parse_args()
    output, config, record = c.start(arguments, c.HERE / "adaptation.json")
    sources = c.read_fits(arguments.source, [f"source/{role}" for role in config["roles"]])
    fits = train(config, output, arguments, sources)
    c.finish(output, record, **evaluate(config, output, fits, arguments.source, sources))


if __name__ == "__main__":
    main()
