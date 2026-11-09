"""Shared helpers of the synthetic experiment runners: options, seeds, banks, fits and outputs."""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import platform
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from maintenance_arena import __version__
from maintenance_arena.banks import LAW_SETS, build_bank, concatenate_banks
from maintenance_arena.metrics import life_table, paired_bootstrap, summarize, write_outcomes
from maintenance_arena.replay import Economics

HERE = Path(__file__).resolve().parent
SUMMARY_COLUMNS = ("method", "group", "lives", "runs", "failure_pct", "unused_life_pct", "retained_profit_pct")


# --- Options and records ---

def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--output", type=Path, required=True, help="new output directory (never overwritten)")
    result.add_argument("--config", type=Path, default=None, help="configuration file (default: the runner's JSON)")
    result.add_argument("--jobs", type=int, default=1, help="parallel training processes")
    result.add_argument("--budget", type=int, default=None,
                        help="smaller training budget for quick checks; milestones scale proportionally")
    result.add_argument("--roles", type=int, default=None, help="use only the first ROLES training seeds")
    return result


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def start(arguments: argparse.Namespace, default_config: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create the output directory, read the configuration, apply quick-check options and write the run record."""
    if arguments.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    output = arguments.output.resolve()
    if output.exists():
        raise SystemExit(f"{output} exists; choose a new output directory")
    config = load_config(arguments.config or default_config)
    if arguments.roles is not None:
        if not 1 <= arguments.roles <= len(config["roles"]):
            raise SystemExit(f"--roles must lie in 1..{len(config['roles'])}")
        config["roles"] = config["roles"][:arguments.roles]
    output.mkdir(parents=True)
    record = {"runner": Path(sys.argv[0]).name, "arguments": sys.argv[1:], "package_version": __version__,
              "python": platform.python_version(), "numpy": np.__version__, "started_utc": now(),
              "quick_check": arguments.budget is not None or arguments.roles is not None, "status": "running"}
    try:
        import torch
        record["torch"] = torch.__version__
    except ImportError:
        pass
    write_json(output / "configuration.json", config)
    write_json(output / "run.json", record, exclusive=False)
    record["_started"] = time.perf_counter()
    return output, config, record


def finish(output: Path, record: dict[str, Any], **result: Any) -> None:
    elapsed = time.perf_counter() - record.pop("_started")
    record.update(status="complete", finished_utc=now(), elapsed_seconds=round(elapsed, 1), **result)
    write_json(output / "run.json", record, exclusive=False)
    print(json.dumps({"status": "complete", "output": str(output), "elapsed_seconds": round(elapsed, 1)}), flush=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --- Seeds, milestones and banks ---

def derived_seed(*entropy: int) -> int:
    """``SeedSequence(entropy).generate_state(1)[0]``, the seed derivation of the article's banks and runs."""
    return int(np.random.SeedSequence([int(value) for value in entropy]).generate_state(1)[0])


def seed_of(config: Mapping[str, Any], purpose: int, law: str = "source", role: int = 0) -> int:
    """Seed of a bank or run from the study's namespace, the purpose code, the law and the role."""
    values = {"namespace": config["seed_namespace"], "purpose": purpose, "law_index": LAW_SETS.index(law), "role": role}
    return derived_seed(*(values[name] for name in config["seed_entropy"]))


def bank_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build the bank of a resolved specification (``laws``, ``split``, ``seed``, ``pairs``, ``sensor_level``)."""
    bank = build_bank(spec["laws"], split=spec["split"], seed=spec["seed"], pair_count=spec["pairs"],
                      sensor_level=spec.get("sensor_level", "exact"))
    for extension in spec.get("extensions", ()):
        bank = concatenate_banks(bank, bank_from_spec(extension))
    return bank


def resolve_bank(config: Mapping[str, Any], spec: Mapping[str, Any], law: str = "source", role: int = 0,
                 **changes: Any) -> dict[str, Any]:
    """A bank specification with its seed: explicit ``seed``, or derived from ``purpose``, law and role."""
    result = {"laws": law, **{k: v for k, v in spec.items() if k not in ("purpose", "namespace", "lives", "per_role")},
              **changes}
    if "lives" in spec:
        result["pairs"] = spec["lives"]
    if "seed" not in spec:
        scope = {**config, **({"seed_namespace": spec["namespace"]} if "namespace" in spec else {})}
        result["seed"] = seed_of(scope, spec["purpose"], law, role if spec.get("per_role", False) else 0)
    return result


def scaled(milestones: Sequence[int], budget: int | None) -> list[int]:
    """Milestones of a quick check with a smaller budget.

A budget that equals a configured milestone keeps the milestones up to it, so the
runs are exact beginnings of the full runs. Any other budget scales every
milestone in proportion (the last one becomes the budget), which cuts the
rollouts at other steps than the full runs.
"""
    points = [int(m) for m in milestones]
    if budget is None or budget == points[-1]:
        return points
    if budget < 1 or budget > points[-1]:
        raise SystemExit(f"--budget must lie in 1..{points[-1]}")
    if budget in points:
        return points[:points.index(budget) + 1]
    result = [max(1, round(m * budget / points[-1])) for m in points[:-1]] + [budget]
    if any(a >= b for a, b in zip(result, result[1:])):
        raise SystemExit(f"--budget {budget} is too small for {len(points)} distinct milestones")
    return result


# --- Training ---

def fit(job: Mapping[str, Any]) -> dict[str, Any]:
    """Train one run on its bank through all milestones and save a checkpoint at each (process-pool job).

``both_estimates`` trains on both causal estimates of a noisy measurement
(eight inputs, ``maintenance_arena.learning.dual``).
"""
    from maintenance_arena.learning.collector import BankTrainer
    started = time.perf_counter()
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    bank = bank_from_spec(job["bank"])
    if job.get("both_estimates"):
        from maintenance_arena.learning.dual import DualBankTrainer
        if job.get("input_set") is not None or job.get("resume") is not None:
            raise ValueError("training on both estimates supports neither input sets nor resumption")
        trainer = DualBankTrainer(bank, job["seed"], env_config=job["env_config"], initialize=job.get("initialize"))
    else:
        trainer = BankTrainer(bank, job["seed"], env_config=job["env_config"], input_set=job.get("input_set"),
                              initialize=job.get("initialize"), resume=job.get("resume"))
    del bank
    checkpoints = []
    try:
        for milestone in job["milestones"]:
            if milestone <= trainer.calendar:
                continue
            progress = trainer.train_until(int(milestone))
            checkpoints.append({**trainer.save(directory / f"checkpoint_{milestone}.pt"), **progress})
    finally:
        trainer.close()
    record = {"label": job["label"], "seed": job["seed"], "bank": {**job["bank"], "sha256": trainer.bank_sha256},
              "env_config": job["env_config"], "input_set": job.get("input_set"),
              "both_estimates": bool(job.get("both_estimates", False)),
              "initialized_from": trainer.initialized_from, "resumed": job.get("resume") is not None,
              "checkpoints": checkpoints, "elapsed_seconds": round(time.perf_counter() - started, 1)}
    write_json(directory / "training.json", {**record, "training_log": trainer.logs})
    return record


def run_jobs(function: Callable[[Mapping[str, Any]], dict[str, Any]], jobs: Sequence[Mapping[str, Any]],
             workers: int) -> list[dict[str, Any]]:
    """Run the jobs (in parallel processes when ``workers > 1``); results keep the job order."""
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    if workers <= 1 or len(jobs) <= 1:
        for index, job in enumerate(jobs):
            results[index] = function(job)
            _progress(results[index], index, len(jobs))
        return results  # type: ignore[return-value]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {pool.submit(function, job): index for index, job in enumerate(jobs)}
        done = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            done += 1
            _progress(results[index], done - 1, len(jobs))
    return results  # type: ignore[return-value]


def _progress(result: Mapping[str, Any], index: int, total: int) -> None:
    print(json.dumps({"fit": result.get("label"), "done": index + 1, "of": total,
                      "seconds": result.get("elapsed_seconds")}), flush=True)


def checkpoint_path(output: Path, record: Mapping[str, Any], milestone: int | None = None) -> Path:
    """Path of a run's checkpoint at ``milestone`` (default: its last checkpoint)."""
    entries = record["checkpoints"]
    entry = entries[-1] if milestone is None else next(c for c in entries if c["nominal_calendar_steps"] == milestone)
    return Path(output) / "fits" / record["label"] / entry["path"]


def read_fits(directory: Path, labels: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Training records of earlier runs, for example the source runs that adaptation starts from."""
    result = {}
    for label in labels:
        path = Path(directory) / "fits" / label / "training.json"
        if not path.is_file():
            raise SystemExit(f"{path} is missing; run the earlier study first")
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("training_log", None)
        result[label] = record
    return result


# --- Evaluation ---

def agents(paths: Mapping[Any, Path]) -> dict[Any, Any]:
    from maintenance_arena.learning.collector import policy_agent
    return {key: policy_agent(path) for key, path in paths.items()}


def probabilities(agent: Any, states: np.ndarray, input_set: str | None = None) -> np.ndarray:
    """Float32 action probabilities of a whole state array in one batch on one thread."""
    from maintenance_arena.learning.collector import single_thread
    from maintenance_arena.replay import policy_probabilities
    from maintenance_arena.wrappers import input_set as mask
    values = states if input_set is None else mask(states, input_set)
    with single_thread():
        return policy_probabilities(agent, values)


def economics(config: Mapping[str, Any], lead: int | None = None) -> Economics:
    env = config["environment"]
    return Economics(mobilization_lead=lead or env.get("mobilization_lead", 12),
                     recovery_contract=env.get("recovery_contract", "immediate_exchange_v1"))


def summary_rows(rows: Sequence[Mapping[str, Any]], prices: Economics, *, by_group: bool = True,
                 overall: str | None = "all") -> list[dict[str, Any]]:
    """Equal-life summaries per method (all lives) and, with ``by_group``, per method and group."""
    result = []
    methods = list(dict.fromkeys(row["method"] for row in rows))
    for method in methods:
        members = [row for row in rows if row["method"] == method]
        if overall is not None:
            result.append({"method": method, "group": overall, **summarize(life_table(members, economics=prices))})
        if by_group:
            for group in dict.fromkeys(row["group"] for row in members):
                subset = [row for row in members if row["group"] == group]
                result.append({"method": method, "group": group, **summarize(life_table(subset, economics=prices))})
    return result


def intervals(rows: Sequence[Mapping[str, Any]], prices: Economics, contrasts: Sequence[Sequence[str]],
              bootstrap: Mapping[str, Any], *, strata_by_group: bool = False) -> dict[str, Any]:
    """Paired bootstrap of the equal-life metrics for the configured method contrasts."""
    names = sorted({name for pair in contrasts for name in pair})
    tables = {name: life_table([row for row in rows if row["method"] == name], economics=prices) for name in names}
    strata = None
    if strata_by_group:
        first = next(iter(tables.values()))
        strata = [unit[0] for unit in first.units]
    return paired_bootstrap(tables, contrasts=[tuple(pair) for pair in contrasts], replicates=bootstrap["replicates"],
                            seed=bootstrap["seed"], strata=strata)


def write_results(output: Path, name: str, rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> None:
    write_outcomes(output / f"outcomes_{name}.csv", rows)
    write_csv(output / f"summary_{name}.csv", summaries, SUMMARY_COLUMNS)
