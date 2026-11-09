"""Preparation time: separate runs for L = 6, 12 and 24 (article Section 4.4).

    python experiments/synthetic/preparation_time.py --output outputs/preparation --jobs 30
    python experiments/synthetic/preparation_time.py --stage first --output outputs/preparation-first --jobs 30
    python experiments/synthetic/preparation_time.py --stage continuation --continue-from outputs/preparation-first --output outputs/preparation-continued --jobs 30

For each L, ten runs train with the same initial weights and training lives
(one bank per seed, shared across L). The first stage trains to 204,800 steps
with checkpoints at 51,200, 102,400 and 204,800; the continuation resumes each
run exactly from its 204,800-step checkpoint and trains to 307,200 steps. The
default stage ``all`` does both in one run, with identical results. The last
trained policies are replayed on 800 test lives (group = ``L6``, ``L12``,
``L24``) next to the Rule calibrated per L on 256 separate lives (health 0.22 to
0.70 or projected time L - 2 to L + 36), reactive operation and the last-safe reference.
"""
from __future__ import annotations

from pathlib import Path

import common as c
from maintenance_arena import replay as rp


def stages(config, arguments):
    configured = config["training"]["milestones"]
    milestones = c.scaled(configured, arguments.budget)
    position = configured.index(config["training"]["continuation_from"])
    if arguments.stage != "all" and position >= len(milestones) - 1:
        raise SystemExit("a staged run needs a budget beyond the continuation milestone")
    cut = milestones[min(position, len(milestones) - 1)]
    if arguments.stage == "first":
        return [m for m in milestones if m <= cut], cut
    if arguments.stage == "continuation":
        return [m for m in milestones if m > cut], cut
    return milestones, cut


def train(config, output, arguments):
    milestones, cut = stages(config, arguments)
    earlier = None
    if arguments.stage == "continuation":
        if arguments.continue_from is None:
            raise SystemExit("--stage continuation needs --continue-from")
        labels = [f"L{lead}/{role}" for lead in config["preparation_times"] for role in config["roles"]]
        earlier = c.read_fits(arguments.continue_from, labels)
    jobs = []
    for role in config["roles"]:
        bank = c.resolve_bank(config, config["training"]["bank"], role=role)
        for lead in config["preparation_times"]:
            label = f"L{lead}/{role}"
            job = {"label": label, "seed": role, "bank": bank, "directory": str(output / "fits" / f"L{lead}" / str(role)),
                   "milestones": milestones, "env_config": {**config["environment"], "mobilization_lead": lead}}
            if earlier is not None:
                job["resume"] = str(c.checkpoint_path(Path(arguments.continue_from), earlier[label], cut))
            jobs.append(job)
    return c.run_jobs(c.fit, jobs, arguments.jobs)


def evaluate(config, output, fits):
    rule_config = config["rule"]
    calibration_bank = c.bank_from_spec(c.resolve_bank(config, rule_config["calibration_bank"]))
    test_bank = c.bank_from_spec(c.resolve_bank(config, config["test_bank"]))
    rows, summaries, rules = [], [], {}
    for lead in config["preparation_times"]:
        prices = c.economics(config, lead)
        anchors = {"health": rule_config["tie_anchors"]["health"],
                   "projection": lead + rule_config["tie_anchors"]["projection_offset"]}
        grids = {"health": rule_config["health_grid"],
                 "projection": [lead + offset for offset in rule_config["projection_offsets"]]}
        calibration = rp.decision_series(calibration_bank, mobilization_lead=lead)
        rule = rp.calibrate_rule(calibration, grids, economics=prices, anchors=anchors)
        rules[f"L{lead}"] = rule["selected"]
        series = rp.decision_series(test_bank, mobilization_lead=lead)
        group = f"L{lead}"
        lead_rows = []
        for record in (r for r in fits if r["env_config"]["mobilization_lead"] == lead):
            agent = c.agents({0: c.checkpoint_path(output, record)})[0]
            orders = rp.greedy_orders(c.probabilities(agent, series.states), series.slices)
            lead_rows += rp.outcome_rows("PPO", series, orders, prices, run=record["seed"], group=group)
        lead_rows += rp.outcome_rows("Rule", series, rp.threshold_orders(series, [rule["selected"]])[0], prices,
                                     group=group)
        lead_rows += rp.outcome_rows("Reactive", series, [-1] * series.lives, prices, group=group)
        lead_rows += rp.outcome_rows("Last-safe reference", series, rp.reference_orders(series.last_productive, lead),
                                     prices, group=group)
        summaries += c.summary_rows(lead_rows, prices, by_group=False, overall=group)
        rules[f"L{lead}"]["intervals"] = c.intervals(lead_rows, prices, config["contrasts"], config["bootstrap"])
        rows += lead_rows
    c.write_results(output, "test", rows, summaries)
    c.write_json(output / "rules.json", {"rules": rules})
    return {"rules": {key: value["threshold"] for key, value in rules.items()}, "test": summaries}


def main():
    parser = c.parser(__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("all", "first", "continuation"), default="all")
    parser.add_argument("--continue-from", type=Path, default=None,
                        help="output of an earlier first stage (for --stage continuation)")
    arguments = parser.parse_args()
    stages(c.load_config(arguments.config or c.HERE / "preparation_time.json"), arguments)  # check before writing
    output, config, record = c.start(arguments, c.HERE / "preparation_time.json")
    fits = train(config, output, arguments)
    c.finish(output, record, stage=arguments.stage, **evaluate(config, output, fits))


if __name__ == "__main__":
    main()
