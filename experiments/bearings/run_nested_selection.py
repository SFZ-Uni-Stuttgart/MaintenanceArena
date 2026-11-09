"""Nested choice of the PPO input design and budget on FEMTO, inside the training bearings of every outer fold.

    python experiments/bearings/run_nested_selection.py --data-root DATA --output outputs/bearings-nested --jobs 40

For every outer fold, the six candidates of run_candidate_selection.py train and are compared only
inside the fold's training bearings. Each inner fold of the outer fold is held out once. The
prediction model is cross-fitted on the other inner folds, and each candidate trains with the five
seeds of the outer fold on the remaining bearings. The candidates are then ranked by pooled
held-out profit over the inner folds, with the rule of the candidate selection. selection.json
holds the ranking of every outer fold, outcomes.csv the inner held-out rows with an inner_fold
column, and fits.json one entry per fit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import common as bc
from maintenance_arena.bearings import (cross_fit, decision_series, design_observations, input_design, outcome_row,
                                        rank_candidates)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    bc.add_data_options(parser)
    parser.add_argument("--cache-dir", type=Path,
                        help="statistics cache femto_statistics.npz, reused if present (default: OUTPUT/statistics)")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; never overwritten")
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes for raw files and PPO fits")
    parser.add_argument("--folds", type=int, nargs="+", help="subset of the five outer folds")
    parser.add_argument("--inner-folds", type=int, nargs="+", help="subset of the inner folds of every outer fold")
    parser.add_argument("--seeds", type=int, help="use only the first N seeds of every outer fold")
    parser.add_argument("--designs", nargs="+", help="subset of the six candidates")
    parser.add_argument("--calendar-steps", type=int, help="override every PPO budget; for quick checks only")
    parser.add_argument("--config-dir", type=Path, default=bc.CONFIGS, help="folder with the JSON configurations")
    args = parser.parse_args(argv)
    if args.jobs < 1 or (args.seeds is not None and args.seeds < 1) or (args.calendar_steps is not None
                                                                       and args.calendar_steps < 1):
        parser.error("--jobs, --seeds and --calendar-steps must be positive")
    return args


def inner_jobs(args, settings, folds, candidates, seeds, femto, outer):
    """PPO jobs and their slots for every inner fold of one outer fold."""
    held_out = set(folds["outer_folds"][outer])
    training = [h for h in femto if h.name not in held_out]
    inner = [list(group) for group in folds["inner_folds"][outer]]
    preparation = settings.economics.preparation
    jobs, slots = [], []
    for j, group in enumerate(inner):
        if args.inner_folds and j not in args.inner_folds:
            continue
        sub_train = [h for h in training if h.name not in set(group)]
        sub_test = [h for h in training if h.name in set(group)]
        rest = [g for i, g in enumerate(inner) if i != j]

        def series(history, forecast=None):
            return decision_series(history, forecast, settings.features, smoothing=settings.grid.smoothing)

        by_age = {None: ([series(h) for h in sub_train], [series(h) for h in sub_test])}
        for age in (True, False):
            out_of_fold, refit = cross_fit(sub_train, rest, sub_test, age=age, settings=settings.forecaster,
                                           features=settings.features)
            by_age[age] = ([series(h, out_of_fold[h.name]) for h in sub_train],
                           [series(h, f) for h, f in zip(sub_test, refit)])
        bc.log(f"outer {outer} inner {j}: {len(sub_train)} training, {len(sub_test)} held-out bearings")
        for name, spec in candidates.items():
            train_series, test_series = by_age[input_design(spec["inputs"]).forecaster_age]
            train = [design_observations(s, spec["inputs"], preparation=preparation) for s in train_series]
            test = [design_observations(s, spec["inputs"], preparation=preparation) for s in test_series]
            for seed in seeds[outer][:args.seeds]:
                jobs.append({"design": name, "dataset": "FEMTO", "fold": outer, "seed": seed,
                             "calendar_steps": args.calendar_steps or spec["calendar_steps"], "train": train,
                             "test": test, "training_bearings": [h.name for h in sub_train],
                             "observation_names": list(input_design(spec["inputs"]).observation_names),
                             "economics": settings.raw["economics"], "ppo": settings.raw["ppo"],
                             "path": str(args.output / "policies" / f"outer{outer}" / f"inner{j}"
                                         / f"{bc.slug(name)}_seed{seed}.pt")})
                slots.append({"outer": outer, "inner": j, "candidate": name, "seed": seed,
                              "bearings": [h.name for h in sub_test], "lives": [h.life for h in sub_test]})
    return jobs, slots


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {"schema": "maintenance_arena_bearing_experiment_v1", "experiment": "nested_selection", "status": "RUNNING",
              "command": [sys.executable, *sys.argv], "started_utc": datetime.now(timezone.utc).isoformat(),
              "options": {"folds": args.folds, "inner_folds": args.inner_folds, "seeds": args.seeds,
                          "designs": args.designs, "calendar_steps": args.calendar_steps, "jobs": args.jobs},
              "configs": {name: bc.file_hash(args.config_dir / name)
                          for name in ("settings.json", "folds.json", "experiments.json")},
              "versions": bc._versions(), "statistics": {}}
    bc.write_json(args.output / "run.json", record, exclusive=False)
    try:
        settings = bc.load_settings(args.config_dir)
        folds = json.loads((args.config_dir / "folds.json").read_text(encoding="utf-8"))
        config = json.loads((args.config_dir / "experiments.json").read_text(encoding="utf-8"))
        candidates = config["experiments"]["candidate_selection"]["designs"]
        if args.designs:
            unknown = set(args.designs) - set(candidates)
            if unknown:
                raise ValueError(f"unknown candidates {sorted(unknown)}; expected some of {list(candidates)}")
            candidates = {name: spec for name, spec in candidates.items() if name in args.designs}
        seeds = config["seeds"]["femto_cross_validation"]
        outer_folds = args.folds or list(range(len(folds["outer_folds"])))
        record["article_settings"] = all(value is None for value in (args.folds, args.inner_folds, args.seeds,
                                                                     args.designs, args.calendar_steps))
        femto = bc.load_histories(args, "femto", record)
        if [h.name for h in femto] != folds["bearings"]:
            raise ValueError("the FEMTO bearings differ from folds.json in names or order")
        bc.write_json(args.output / "run.json", record, exclusive=False)
        jobs, slots = [], []
        for outer in outer_folds:
            more_jobs, more_slots = inner_jobs(args, settings, folds, candidates, seeds, femto, outer)
            jobs += more_jobs
            slots += more_slots
        order = sorted(range(len(jobs)), key=lambda i: -jobs[i]["calendar_steps"])  # long fits first
        bc.log(f"{len(jobs)} PPO fits with {args.jobs} processes")
        results = bc.run_jobs([jobs[i] for i in order], args.jobs)
        economics, rows, fits = settings.economics, [], []
        for position, i in enumerate(order):
            slot, result = slots[i], results[position]
            for bearing, life, minute in zip(slot["bearings"], slot["lives"], result["orders"]):
                row = outcome_row(slot["candidate"], "FEMTO", bearing, economics.outcome(life, minute),
                                  fold=slot["outer"], seed=slot["seed"])
                rows.append({**row, "inner_fold": slot["inner"]})
            fits.append({**{key: slot[key] for key in ("outer", "inner", "candidate", "seed")},
                         **{key: result[key] for key in ("seconds", "episodes", "training_failures",
                                                         "actual_calendar_steps", "policy_sha256")}})
        names = list(candidates)
        rows.sort(key=lambda r: (r["fold"], names.index(r["design"]), r["inner_fold"], r["seed"], r["bearing"]))
        with (args.output / "outcomes.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: ("" if value is None else value) for key, value in row.items()} for row in rows)
        spec = {name: {"inputs": c["inputs"], "calendar_steps": args.calendar_steps or c["calendar_steps"]}
                for name, c in candidates.items()}
        selection = {}
        for outer in outer_folds:
            ranked = rank_candidates([r for r in rows if r["fold"] == outer], spec, dataset="FEMTO",
                                     economics=economics)
            selection[str(outer)] = {"selected": ranked["selected"], "ranking": ranked["ranking"]}
            bc.log(f"outer {outer}: selected {ranked['selected']}")
        bc.write_json(args.output / "selection.json", selection)
        bc.write_json(args.output / "fits.json", fits)
        products = ("outcomes.csv", "selection.json", "fits.json")
        record.update(status="COMPLETE", fits=len(fits), rows=len(rows),
                      selected={k: v["selected"] for k, v in selection.items()},
                      outputs={name: bc.file_hash(args.output / name) for name in products})
        return 0
    except BaseException as error:
        record.update(status="FAILED", error={"type": type(error).__name__, "message": str(error),
                                              "traceback": traceback.format_exc()})
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started)
        bc.write_json(args.output / "run.json", record, exclusive=False)


if __name__ == "__main__":
    raise SystemExit(main())
