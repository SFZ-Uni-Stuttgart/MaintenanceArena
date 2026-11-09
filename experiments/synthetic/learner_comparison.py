"""Learner comparison: ten runs each of PPO, DQN and double DQN (article Appendix A, Table A2).

    python experiments/synthetic/learner_comparison.py --output outputs/learners --jobs 30

Each run trains on the source bank of its index with preserved preparation to
204,800 calendar steps and saves checkpoints at 51,200, 102,400 and 204,800
steps. The three learners of the same index start on the same lives. Every
checkpoint is replayed greedily on 800 diagnostic lives (``outcomes_diagnostic.csv``,
group = training steps), and the final policies on 2,000 new test lives
(``outcomes_test.csv``, group = law) next to the Time rule at projected time 13
(stored as Naive projection rule) and 15 (stored as Rule, the threshold of the
source study), reactive operation and the last-safe reference.
"""
from __future__ import annotations

from pathlib import Path
import time

import common as c
from maintenance_arena import replay as rp


def fit(job):
    """Train one run through all milestones and save a checkpoint at each (process-pool job)."""
    from maintenance_arena.learning.learners import make_trainer, parameter_count, recipe
    started = time.perf_counter()
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    bank = c.bank_from_spec(job["bank"])
    trainer = make_trainer(job["algorithm"], bank, job["seed"], total_calendar_steps=job["total_calendar_steps"],
                           env_config=job["env_config"])
    del bank
    checkpoints = []
    try:
        for milestone in job["milestones"]:
            progress = trainer.train_until(int(milestone))
            checkpoints.append({**trainer.save(directory / f"checkpoint_{milestone}.pt"), **progress})
    finally:
        trainer.close()
    record = {"label": job["label"], "algorithm": job["algorithm"], "seed": job["seed"], "run": job["run"],
              "bank": {**job["bank"], "sha256": trainer.bank_sha256}, "env_config": job["env_config"],
              "recipe": recipe(job["algorithm"]), "parameters": parameter_count(job["algorithm"]),
              "total_calendar_steps": job["total_calendar_steps"], "checkpoints": checkpoints,
              "elapsed_seconds": round(time.perf_counter() - started, 1)}
    c.write_json(directory / "training.json", {**record, "training_log": trainer.logs})
    return record


def train(config, output, arguments):
    configured = config["training"]["milestones"]
    milestones = c.scaled(configured, arguments.budget)
    # Exploration of DQN and DDQN depends on the budget. A quick check that stops at a configured
    # milestone keeps the full budget, so its runs are exact beginnings of the full runs.
    total = config["training"]["steps"] if milestones == configured[:len(milestones)] else milestones[-1]
    jobs = []
    for algorithm in config["algorithms"]:
        for run, seed in enumerate(config["roles"]):
            jobs.append({"label": f"{algorithm}/{seed}", "algorithm": algorithm, "seed": seed, "run": run,
                         "directory": str(output / "fits" / algorithm / str(seed)),
                         "bank": c.resolve_bank(config, config["training"]["bank"], role=run),
                         "milestones": milestones, "total_calendar_steps": total, "env_config": config["environment"]})
    return c.run_jobs(fit, jobs, arguments.jobs)


def evaluate(config, output, fits):
    from maintenance_arena.learning.learners import greedy_first_orders, load_policy
    prices = c.economics(config)
    labels = config["algorithms"]
    results = {}
    for name, spec, final_only in (("diagnostic", config["diagnostic_bank"], False), ("test", config["test_bank"], True)):
        series = rp.decision_series(c.bank_from_spec(c.resolve_bank(config, spec)))
        rows = []
        for record in fits:
            for entry in record["checkpoints"][-1:] if final_only else record["checkpoints"]:
                policy = load_policy(c.checkpoint_path(output, record, entry["nominal_calendar_steps"]))
                rows += rp.outcome_rows(labels[record["algorithm"]], series, greedy_first_orders(policy, series),
                                        prices, run=record["seed"],
                                        group=None if final_only else str(entry["nominal_calendar_steps"]))
        if final_only:
            projection = rp.projected_time(series.states)
            for method, threshold in config["rules"].items():
                rule = {"family": "projection", "threshold": float(threshold)}
                rows += rp.outcome_rows(method, series, rp.threshold_orders(series, [rule], projection)[0], prices)
            rows += rp.outcome_rows("Reactive", series, [-1] * series.lives, prices)
            rows += rp.outcome_rows("Last-safe reference", series, rp.reference_orders(series.last_productive), prices)
        summaries = c.summary_rows(rows, prices, overall="all" if final_only else None)
        c.write_results(output, name, rows, summaries)
        results[name] = (rows, summaries)
    rows, summaries = results["test"]
    learners = {}
    for record in fits:
        label = labels[record["algorithm"]]
        failed = any(row["failed"] for row in rows if row["method"] == label and row["run"] == record["seed"])
        entry = learners.setdefault(label, {"parameters": record["parameters"], "failure_free_runs": 0,
                                            "failures_during_training": 0})
        entry["failure_free_runs"] += not failed
        entry["failures_during_training"] += record["checkpoints"][-1]["failures_during_training"]
    return {"learners": learners, "test": [row for row in summaries if row["group"] == "all"]}


def main():
    arguments = c.parser(__doc__.splitlines()[0]).parse_args()
    output, config, record = c.start(arguments, c.HERE / "learner_comparison.json")
    fits = train(config, output, arguments)
    c.finish(output, record, **evaluate(config, output, fits))


if __name__ == "__main__":
    main()
