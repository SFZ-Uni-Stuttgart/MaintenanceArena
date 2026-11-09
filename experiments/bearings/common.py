"""Shared steps of the bearing experiment runners: options, data, inputs, fits and records.

Every runner writes into a new ``--output`` directory: ``outcomes.csv`` (one row per
design, fold, seed and bearing), ``selections.json`` (fitted rule thresholds), ``fits.json``
(one entry per PPO fit), the saved actors below ``policies/`` and ``run.json``.
"""
from __future__ import annotations

import os

# One thread per process keeps forecaster fits and PPO updates reproducible.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

import maintenance_arena
from maintenance_arena.bearings import (RULE_FAMILIES, BearingHistory, FeatureSettings, ForecasterSettings,
                                        ReplayEconomics, RuleGrid, apply_rule, calibrate_forecast_rule,
                                        calibrate_indicator_rule, calibrate_risk_classifier, cross_fit,
                                        dataset_directory, decision_series, design_observations, forecast_rule_order,
                                        histories, input_design, load_statistics, outcome_row, pooled_summary, prepare,
                                        rank_candidates, risk_classifier_orders, search_rule_families, write_outcomes)
from maintenance_arena.bearings.rules import grid_values

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
DATASET_LABELS = {"femto": "FEMTO", "xjtu_sy": "XJTU-SY"}
EVALUATIONS = ("femto_cross_validation", "xjtu_sy_transfer")
KINDS = ("ppo", "indicator_rule", "forecast_rule", "naive_forecast_rule", "reactive", "rule_search", "risk_classifier")


def log(message: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value, *, exclusive: bool = True) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


@dataclass
class Settings:
    economics: ReplayEconomics
    features: FeatureSettings
    forecaster: ForecasterSettings
    grid: RuleGrid
    cutoffs: np.ndarray
    naive_cutoff: float
    raw: dict


def load_settings(directory: Path) -> Settings:
    raw = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    if raw["ppo_recipe"] != "paper-core-continuous-v1":
        raise ValueError("the bearing PPO recipe is paper-core-continuous-v1")
    return Settings(ReplayEconomics(**raw["economics"]), FeatureSettings(**raw["features"]),
                    ForecasterSettings(**raw["forecaster"]), RuleGrid(**raw["indicator_rule"]),
                    grid_values(*raw["forecast_rule"]["cutoffs"]), float(raw["forecast_rule"]["naive_cutoff"]), raw)


@dataclass
class Split:
    """Training and evaluation histories of one outer fold or of the transfer."""
    evaluation: str
    dataset: str
    fold: int | None
    training: list[BearingHistory]
    test: list[BearingHistory]
    inner_folds: list[list[str]]
    series: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "transfer" if self.fold is None else f"fold{self.fold}"


def add_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=os.environ.get("MAINTENANCE_ARENA_DATA"),
                        help="folder holding FEMTO/ and XJTU-SY/ (default: $MAINTENANCE_ARENA_DATA)")
    parser.add_argument("--femto-dir", type=Path, help="FEMTO folder with Learning_set and Full_Test_Set, "
                                                       "instead of DATA_ROOT/FEMTO")
    parser.add_argument("--xjtu-sy-dir", type=Path, help="XJTU-SY folder with 35Hz12kN, 37.5Hz11kN and 40Hz10kN, "
                                                         "instead of DATA_ROOT/XJTU-SY")


def dataset_option(args, dataset: str) -> Path | None:
    return args.femto_dir if dataset == "femto" else args.xjtu_sy_dir


def parse_args(experiment: str, argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Bearing experiment {experiment!r}; see experiments/bearings/README.md.")
    add_data_options(parser)
    parser.add_argument("--cache-dir", type=Path,
                        help="statistics caches <dataset>_statistics.npz, reused if present (default: OUTPUT/statistics)")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; never overwritten")
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes for raw files and PPO fits")
    parser.add_argument("--folds", type=int, nargs="+", help="subset of the five FEMTO outer folds")
    parser.add_argument("--seeds", type=int, help="use only the first N seeds of every fold and of the transfer")
    parser.add_argument("--designs", nargs="+", help="subset of the experiment's designs (article names)")
    parser.add_argument("--calendar-steps", type=int, help="override every PPO budget; for quick checks only")
    parser.add_argument("--evaluations", nargs="+", choices=EVALUATIONS,
                        help="subset of the experiment's evaluations, FEMTO cross-validation or transfer")
    parser.add_argument("--config-dir", type=Path, default=CONFIGS, help="folder with the JSON configurations")
    args = parser.parse_args(argv)
    if args.jobs < 1 or (args.seeds is not None and args.seeds < 1):
        parser.error("--jobs and --seeds must be positive")
    if args.calendar_steps is not None and args.calendar_steps < 1:
        parser.error("--calendar-steps must be positive")
    return args


def load_histories(args, dataset: str, record: dict) -> list[BearingHistory]:
    directory = args.cache_dir or args.output / "statistics"
    path = directory / f"{dataset}_statistics.npz"
    if path.exists():
        statistics, source = load_statistics(path), "loaded cache"
    else:
        explicit = dataset_option(args, dataset)
        if args.data_root is None and explicit is None:
            raise ValueError(f"{path} does not exist; pass --data-root or the dataset directory to compute it")
        raw = dataset_directory(args.data_root, dataset, directory=explicit)
        log(f"computing {dataset} statistics from {raw}")
        started = time.perf_counter()
        statistics = prepare(None, dataset, path, jobs=args.jobs, directory=raw)
        source = f"computed from raw files in {time.perf_counter() - started:.1f} s"
    record["statistics"][dataset] = {"path": str(path.resolve()), "sha256": file_hash(path),
                                     "bearings": len(statistics), "source": source}
    return histories(statistics, dataset)


def splits(evaluation: str, femto: list[BearingHistory], target: list[BearingHistory] | None, folds: dict,
           selected: list[int]) -> list[Split]:
    if evaluation == "femto_cross_validation":
        result = []
        for fold in selected:
            test = set(folds["outer_folds"][fold])
            result.append(Split(evaluation, DATASET_LABELS["femto"], fold, [h for h in femto if h.name not in test],
                                [h for h in femto if h.name in test], folds["inner_folds"][fold]))
        return result
    return [Split(evaluation, DATASET_LABELS["xjtu_sy"], None, list(femto), list(target), folds["all_bearings_inner_folds"])]


def needed_series(designs: dict) -> set:
    """Forecaster settings the designs need: True with age, False without, None for no forecast."""
    return {True if spec["kind"] in ("indicator_rule", "rule_search") else input_design(spec["inputs"]).forecaster_age
            for spec in designs.values() if spec["kind"] != "reactive"}


def prepare_series(split: Split, ages: set, settings: Settings) -> None:
    """Decision series of training (out-of-fold forecasts) and evaluation histories (refit forecasts)."""
    for age in sorted(ages, key=lambda a: (a is None, a is False)):
        def series(history, forecast=None):
            return decision_series(history, forecast, settings.features, smoothing=settings.grid.smoothing)

        if age is None:
            split.series[None] = ([series(h) for h in split.training], [series(h) for h in split.test])
            continue
        out_of_fold, refit = cross_fit(split.training, split.inner_folds, split.test, age=age,
                                       settings=settings.forecaster, features=settings.features)
        split.series[age] = ([series(h, out_of_fold[h.name]) for h in split.training],
                             [series(h, f) for h, f in zip(split.test, refit)])


def observations(split: Split, inputs: str, settings: Settings) -> tuple[list[np.ndarray], list[np.ndarray]]:
    training, test = split.series[input_design(inputs).forecaster_age]
    preparation = settings.economics.preparation
    return ([design_observations(s, inputs, preparation=preparation) for s in training],
            [design_observations(s, inputs, preparation=preparation) for s in test])


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def fit_job(job: dict) -> dict:
    """Train one actor, save it and return its greedy first orders on the evaluation and training histories."""
    from maintenance_arena.bearings.ppo import TrainingSettings, save_policy, train_policy

    result = train_policy(job["train"], job["seed"], job["calendar_steps"],
                          economics=ReplayEconomics(**job["economics"]), settings=TrainingSettings(**job["ppo"]))
    orders = [result.agent.greedy_order_minute(x) for x in job["test"]]
    training_orders = [result.agent.greedy_order_minute(x) for x in job["train"]]
    save_policy(job["path"], result.agent, design=job["design"], dataset=job["dataset"], fold=job["fold"],
                calendar_steps=job["calendar_steps"], actual_calendar_steps=result.calendar_steps,
                training_bearings=job["training_bearings"], observation_names=job["observation_names"])
    return {"orders": orders, "training_orders": training_orders, "seconds": result.seconds, "episodes": result.episodes,
            "training_failures": result.failures, "optimizer_updates": result.agent.optimizer_updates,
            "actual_calendar_steps": result.calendar_steps, "logs": result.logs,
            "policy_sha256": file_hash(Path(job["path"]))}


def run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    results: list[dict | None] = [None] * len(jobs)
    if not jobs:
        return []
    describe = lambda job: f"{job['design']} {job['dataset']} fold={job['fold']} seed={job['seed']}"
    if workers == 1:
        for index, job in enumerate(jobs):
            results[index] = fit_job(job)
            log(f"fit {index + 1}/{len(jobs)} {describe(job)} {results[index]['seconds']:.0f} s")
        return results
    with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
        futures = {pool.submit(fit_job, job): index for index, job in enumerate(jobs)}
        for count, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            results[index] = future.result()
            log(f"fit {count}/{len(jobs)} {describe(jobs[index])} {results[index]['seconds']:.0f} s")
    return results


def _versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                "maintenance_arena": maintenance_arena.__version__}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "gymnasium"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main(experiment: str, argv=None) -> int:
    args = parse_args(experiment, argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {"schema": "maintenance_arena_bearing_experiment_v1", "experiment": experiment, "status": "RUNNING",
              "command": [sys.executable, *sys.argv], "started_utc": datetime.now(timezone.utc).isoformat(),
              "options": {"folds": args.folds, "seeds": args.seeds, "designs": args.designs,
                          "evaluations": args.evaluations, "calendar_steps": args.calendar_steps, "jobs": args.jobs},
              "configs": {name: file_hash(args.config_dir / name)
                          for name in ("settings.json", "folds.json", "experiments.json")},
              "versions": _versions(), "statistics": {}}
    write_json(args.output / "run.json", record, exclusive=False)
    try:
        settings = load_settings(args.config_dir)
        folds = json.loads((args.config_dir / "folds.json").read_text(encoding="utf-8"))
        config = json.loads((args.config_dir / "experiments.json").read_text(encoding="utf-8"))
        plan = config["experiments"][experiment]
        designs = plan["designs"]
        if args.designs:
            unknown = set(args.designs) - set(designs)
            if unknown:
                raise ValueError(f"unknown designs {sorted(unknown)}; expected some of {list(designs)}")
            designs = {name: spec for name, spec in designs.items() if name in args.designs}
        if any(spec["kind"] not in KINDS for spec in designs.values()):
            raise ValueError(f"design kinds must be among {KINDS}")
        selected_folds = args.folds or list(range(len(folds["outer_folds"])))
        if any(not 0 <= f < len(folds["outer_folds"]) for f in selected_folds):
            raise ValueError("--folds must name outer folds 0..4")
        record["article_settings"] = (args.calendar_steps is None and args.seeds is None and args.folds is None
                                      and args.designs is None and args.evaluations is None)
        femto = load_histories(args, "femto", record)
        if [h.name for h in femto] != folds["bearings"]:
            raise ValueError("the FEMTO bearings differ from folds.json in names or order")
        evaluations = [e for e in plan["evaluations"] if not args.evaluations or e in args.evaluations]
        if not evaluations:
            raise ValueError(f"the experiment has the evaluations {plan['evaluations']}")
        target = load_histories(args, "xjtu_sy", record) if "xjtu_sy_transfer" in evaluations else None
        write_json(args.output / "run.json", record, exclusive=False)
        economics = settings.economics
        rows, selections, jobs = [], {}, []
        for evaluation in evaluations:
            seeds = config["seeds"][evaluation]
            for split in splits(evaluation, femto, target, folds, selected_folds):
                prepare_series(split, needed_series(designs), settings)
                log(f"{split.dataset} {split.label}: {len(split.training)} training, {len(split.test)} evaluation bearings")
                split_seeds = (seeds if split.fold is None else seeds[split.fold])[:args.seeds]
                for name, spec in designs.items():
                    kind = spec["kind"]
                    if kind == "reactive":
                        orders = [None] * len(split.test)
                    elif kind == "indicator_rule":
                        training, test = split.series[True]
                        chosen = calibrate_indicator_rule(training, economics, settings.grid)
                        selections.setdefault(name, {})[f"{split.dataset}/{split.label}"] = {
                            family: {**asdict(s.candidate), "training_profit_rate": s.training_profit_rate,
                                     "training_failures": s.training_failures} for family, s in chosen.items()}
                        orders = [apply_rule(chosen["best"].candidate, s) for s in test]
                    elif kind in ("forecast_rule", "naive_forecast_rule"):
                        training, test = observations(split, spec["inputs"], settings)
                        if kind == "forecast_rule":
                            chosen = calibrate_forecast_rule([x[:, 0] for x in training], settings.cutoffs, economics)
                            selections.setdefault(name, {})[f"{split.dataset}/{split.label}"] = asdict(chosen)
                            cutoff = chosen.cutoff
                        else:
                            cutoff = settings.naive_cutoff
                        orders = [forecast_rule_order(x[:, 0], cutoff) for x in test]
                    elif kind == "risk_classifier":
                        training, test = observations(split, spec["inputs"], settings)
                        model, chosen = calibrate_risk_classifier(
                            dict(zip((h.name for h in split.training), training)), economics)
                        selections.setdefault(name, {})[f"{split.dataset}/{split.label}"] = asdict(chosen)
                        orders = risk_classifier_orders(model, chosen.threshold, test)
                    elif kind == "rule_search":
                        training, test = split.series[True]
                        search = search_rule_families(dict(zip((h.name for h in split.training), training)),
                                                      split.inner_folds, dict(zip((h.name for h in split.test), test)),
                                                      economics=economics, grid=settings.grid)
                        selections.setdefault(name, {})[f"{split.dataset}/{split.label}"] = search
                        for family, entry in search["families"].items():
                            rows += [outcome_row(spec["family_prefix"] + family, split.dataset, h.name,
                                                 economics.outcome(h.life, entry["evaluation_orders"][h.name]),
                                                 fold=split.fold) for h in split.test]
                        chosen = search["families"][search["chosen"]]
                        orders = [chosen["evaluation_orders"][h.name] for h in split.test]
                    else:
                        training, test = observations(split, spec["inputs"], settings)
                        for seed in split_seeds:
                            jobs.append({"design": name, "dataset": split.dataset, "fold": split.fold, "seed": seed,
                                         "calendar_steps": args.calendar_steps or spec["calendar_steps"],
                                         "train": training, "test": test,
                                         "training_bearings": [h.name for h in split.training],
                                         "test_bearings": [h.name for h in split.test],
                                         "lives": [h.life for h in split.test],
                                         "observation_names": list(input_design(spec["inputs"]).observation_names),
                                         "economics": settings.raw["economics"], "ppo": settings.raw["ppo"],
                                         "path": str(args.output / "policies" / slug(name)
                                                     / f"{split.dataset.lower()}_{split.label}_seed{seed}.pt")})
                        continue
                    rows += [outcome_row(name, split.dataset, h.name, economics.outcome(h.life, order), fold=split.fold)
                             for h, order in zip(split.test, orders)]
        log(f"{len(jobs)} PPO fits with {args.jobs} processes")
        results = run_jobs(jobs, args.jobs)
        fits = []
        for job, result in zip(jobs, results):
            rows += [outcome_row(job["design"], job["dataset"], bearing, economics.outcome(life, order),
                                 fold=job["fold"], seed=job["seed"])
                     for bearing, life, order in zip(job["test_bearings"], job["lives"], result["orders"])]
            fits.append({key: job[key] for key in ("design", "dataset", "fold", "seed", "calendar_steps",
                                                   "training_bearings", "observation_names")}
                        | {"training_lives": [len(x) for x in job["train"]]}
                        | {"policy": Path(job["path"]).relative_to(args.output).as_posix()}
                        | {key: value for key, value in result.items() if key != "orders"})
        names = list(designs)
        for name, spec in designs.items():
            if spec["kind"] == "rule_search":
                families = [spec["family_prefix"] + family.name for family in RULE_FAMILIES]
                names += families
                if spec.get("held_out_choice"):
                    # The family with the highest pooled held-out profit, the criterion that selected Full.
                    names.append(spec["held_out_choice"])
                    for dataset in sorted({r["dataset"] for r in rows if r["design"] in families}):
                        profit = {f: pooled_summary(rows, f, dataset=dataset, economics=economics)
                                  ["profit_per_calendar_minute"] for f in families}
                        best = max(families, key=lambda f: profit[f])
                        selections.setdefault(spec["held_out_choice"], {})[dataset] = {
                            "family": best.removeprefix(spec["family_prefix"]), "pooled_profit": profit}
                        rows += [{**r, "design": spec["held_out_choice"]} for r in rows
                                 if r["design"] == best and r["dataset"] == dataset]
        datasets = list(DATASET_LABELS.values())
        rows.sort(key=lambda r: (names.index(r["design"]), datasets.index(r["dataset"]),
                                 -1 if r["fold"] is None else r["fold"], -1 if r["seed"] is None else r["seed"]))
        products = ["outcomes.csv", "selections.json", "fits.json"]
        write_outcomes(args.output / "outcomes.csv", rows)
        write_json(args.output / "selections.json", selections)
        write_json(args.output / "fits.json", fits)
        if "selection" in plan:
            candidates = {name: {"inputs": spec["inputs"], "calendar_steps": args.calendar_steps or spec["calendar_steps"]}
                          for name, spec in designs.items() if spec["kind"] == "ppo"}
            ranking = {dataset: rank_candidates(rows, candidates, dataset=dataset, economics=economics)
                       for dataset in sorted({r["dataset"] for r in rows if r["design"] in candidates})}
            write_json(args.output / "selection.json", {"rule": plan["selection"], **ranking})
            products.append("selection.json")
            for dataset, result in ranking.items():
                log(f"{dataset}: selected {result['selected']}; best per input design {result['best_by_inputs']}")
        record.update(status="COMPLETE", rows=len(rows), fits=len(fits),
                      outputs={name: file_hash(args.output / name) for name in products})
        log(f"wrote {len(rows)} outcome rows to {args.output / 'outcomes.csv'}")
        return 0
    except BaseException as error:
        record.update(status="FAILED", error={"type": type(error).__name__, "message": str(error),
                                              "traceback": traceback.format_exc()})
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      seconds=time.perf_counter() - started)
        write_json(args.output / "run.json", record, exclusive=False)
