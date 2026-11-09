"""Figure data of the bearing study from runner outputs, and simple figures with the plots extra.

    python experiments/bearings/figures.py policy-map --run outputs/bearings-femto --rule-outcomes outputs/bearings-forecast-rules/outcomes.csv --output outputs/figure-policy-map --plot
    python experiments/bearings/figures.py transfer-timing --outcomes outputs/bearings-transfer/outcomes.csv outputs/bearings-forecast-rules/outcomes.csv --output outputs/figure-timing --plot
    python experiments/bearings/figures.py age-clock --runs outputs/bearings-transfer outputs/bearings-age --output outputs/figure-age-clock --plot

policy-map (Fig. 7): mean ORDER probability of the five Full actors of each bearing's outer
fold along every held-out FEMTO history, with the first order of the Time rule on the same
Full prediction; the held-out inputs are rebuilt from the run's statistics cache and the saved
actors are queried (``policy_map.json``, ``policy_responses.npz``). transfer-timing (Fig. 8):
median first order of every design per XJTU-SY bearing relative to the last safe order, PPO
next to the Time rule on the same prediction (``transfer_timing.json``). age-clock (Fig. D1):
first orders of every transfer actor on its FEMTO training histories and on the three longest
XJTU-SY histories (``age_clock.json``). Every output directory is new. ``--plot`` draws a PNG
and a PDF with Matplotlib.

The runners store the Time rule as "Forecast rule" and No prediction age as "No forecaster
age". The data files keep these names, and the plots show the article's names.
"""
import common

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from maintenance_arena.bearings import (age_clock, histories, load_statistics, policy_map, policy_responses,
                                        read_outcomes, transfer_timing)

DEFAULT_TIMING = ("Full", "Forecast rule: Full", "No age", "Forecast rule: No age")
DEFAULT_CLOCK = ("Full", "No forecaster age", "No policy age", "No age")
DEFAULT_RULE = "Forecast rule: Full"
# Stored design names and the names of the article.
ARTICLE_NAMES = {"Forecast rule: Full": "Time rule: Full", "Forecast rule: No age": "Time rule: No age",
                 "No forecaster age": "No prediction age"}


def article_name(design: str) -> str:
    return ARTICLE_NAMES.get(design, design)


def _write(output: Path, name: str, value) -> None:
    common.write_json(output / name, value)


def build_policy_map(run: Path, design: str, config_dir: Path, cache_dir: Path | None,
                     rule_outcomes: list[Path] | None = None, rule_design: str = DEFAULT_RULE) -> tuple[list[dict], dict]:
    """Query the saved actors of a cross-validation run on rebuilt held-out inputs.

    The rule orders come from ``rule_outcomes``, by default the run's own outcomes.
    """
    import torch
    from maintenance_arena.bearings.ppo import load_policy

    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    fits = [fit for fit in json.loads((run / "fits.json").read_text(encoding="utf-8"))
            if fit["design"] == design and fit["dataset"] == "FEMTO"]
    if not fits:
        raise ValueError(f"{run} holds no FEMTO fits of {design}")
    plan = json.loads((config_dir / "experiments.json").read_text(encoding="utf-8"))["experiments"][record["experiment"]]
    inputs = plan["designs"][design]["inputs"]
    settings = common.load_settings(config_dir)
    folds = json.loads((config_dir / "folds.json").read_text(encoding="utf-8"))
    cache = (cache_dir / "femto_statistics.npz") if cache_dir else Path(record["statistics"]["femto"]["path"])
    femto = histories(load_statistics(cache), "femto")
    rule = {row["bearing"]: row["order_minute"] for path in (rule_outcomes or [run / "outcomes.csv"])
            for row in read_outcomes(path) if row["design"] == rule_design and row["dataset"] == "FEMTO"}
    if not rule:
        raise ValueError(f"no FEMTO orders of {rule_design!r}; pass the outcomes of run_forecast_rules.py with "
                         "--rule-outcomes, or name another --rule-design")
    torch.set_num_threads(1)
    entries, arrays, fold_of = {}, {}, {}
    for fold in sorted({fit["fold"] for fit in fits}):
        split, = common.splits("femto_cross_validation", femto, None, folds, [fold])
        common.prepare_series(split, {common.input_design(inputs).forecaster_age}, settings)
        _, test = common.observations(split, inputs, settings)
        actors = [load_policy(run / fit["policy"])[0] for fit in sorted((f for f in fits if f["fold"] == fold),
                                                                         key=lambda f: f["seed"])]
        seeds = sorted(fit["seed"] for fit in fits if fit["fold"] == fold)
        for history, observations in zip(split.test, test):
            probabilities = policy_responses(actors, observations)
            entries[history.name] = {"probabilities": probabilities, "rule_order": rule.get(history.name)}
            fold_of[history.name] = fold
            arrays.update({f"{history.name}__observations": observations, f"{history.name}__probabilities": probabilities,
                           f"{history.name}__seeds": np.asarray(seeds)})
    rows = policy_map(entries, preparation=settings.economics.preparation)
    for row in rows:
        row["fold"] = fold_of[row["bearing"]]
    return rows, arrays


def plot_policy_map(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 0.25 * len(rows) + 1.2))
    for i, row in enumerate(rows):
        life = row["life"]
        x = np.arange(1, life + 1) / life
        edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
        mesh = ax.pcolormesh(edges, [i - .43, i + .43], np.asarray(row["mean_order_probability"])[None, :],
                             cmap="RdBu_r", vmin=0, vmax=1, shading="flat")
    ax.set(yticks=range(len(rows)), yticklabels=[f"{r['bearing']} ({r['life']})" for r in rows], xlim=(0, 1),
           ylim=(len(rows) - .5, -.5), xlabel="Fraction of recorded life (display only)")
    fig.colorbar(mesh, ax=ax, label="Mean P(ORDER)")
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=200)
    plt.close(fig)


def plot_transfer_timing(rows: list[dict], designs, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(3.6, 0.22 * len(rows) + 1.0))
    cmap = plt.get_cmap("Blues_r")
    for i, row in enumerate(rows):
        for j, design in enumerate(designs):
            cell = row["designs"][design]
            if cell["status"] == "timely":
                colour, label = cmap(0.15 + 0.7 * cell["unused_life_pct"] / 100), f"{cell['unused_life_pct']:.1f}"
            else:
                colour, label = ("#8c1d40", "late") if cell["status"] == "late" else ("#bbbbbb", "none")
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=colour, edgecolor="white"))
            ax.text(j, i, label, ha="center", va="center", fontsize=7)
    ax.set(xlim=(-.5, len(designs) - .5), ylim=(len(rows) - .5, -.5), xticks=range(len(designs)),
           xticklabels=[article_name(d) for d in designs], yticks=range(len(rows)),
           yticklabels=[f"{r['bearing']} ({r['life']})" for r in rows])
    ax.set_title("Unused life (%), 0 = last safe order", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=200)
    plt.close(fig)


def plot_age_clock(clock: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(design, seed) for design, seeds in clock["designs"].items() for seed in seeds]
    fig, ax = plt.subplots(figsize=(7.0, 0.22 * len(rows) + 1.2))
    ax.axvspan(5, clock["longest_training_life"], color="#eeeeee", zorder=0)
    for bearing in clock["long_bearings"]:
        ax.axvline(bearing["last_safe_order"], color="#555555", lw=.6)
    for i, (design, seed) in enumerate(rows):
        training = [o for o in seed["training_orders"].values() if o is not None]
        ax.plot(training, [i] * len(training), "o", mfc="none", mec="#888888", ms=3)
        target = [o for o in seed["target_orders"].values() if o is not None]
        ax.plot(target, [i] * len(target), "o", color="#1f5fa8", ms=3.5)
    ax.set(xscale="log", xlim=(5, 3000), ylim=(len(rows) - .5, -.5), yticks=range(len(rows)),
           yticklabels=[f"{article_name(d)} {s['seed']}" for d, s in rows], xlabel="First order (minute)")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="figure", required=True)
    policy = commands.add_parser("policy-map", help="mean ORDER probability along every held-out FEMTO history")
    policy.add_argument("--run", type=Path, required=True, help="output directory of run_femto_cross_validation.py")
    policy.add_argument("--design", default="Full")
    policy.add_argument("--rule-outcomes", type=Path, nargs="+",
                        help="outcome files with the rule orders, usually those of run_forecast_rules.py "
                             "(default: the run's own outcomes)")
    policy.add_argument("--rule-design", default=DEFAULT_RULE,
                        help=f"stored name of the rule whose orders are marked (default: {DEFAULT_RULE!r}, "
                             "the Time rule on the Full prediction)")
    policy.add_argument("--cache-dir", type=Path, help="statistics caches, if the run's cache has moved")
    timing = commands.add_parser("transfer-timing", help="median order timing per XJTU-SY bearing")
    timing.add_argument("--outcomes", type=Path, nargs="+", required=True)
    timing.add_argument("--designs", nargs="+", default=list(DEFAULT_TIMING))
    clock = commands.add_parser("age-clock", help="first orders on FEMTO training and the longest XJTU-SY histories")
    clock.add_argument("--runs", type=Path, nargs="+", required=True, help="transfer and age-design run directories")
    clock.add_argument("--designs", nargs="+", default=list(DEFAULT_CLOCK))
    for sub in (policy, timing, clock):
        sub.add_argument("--output", type=Path, required=True, help="new output directory")
        sub.add_argument("--plot", action="store_true", help="draw the figure (plots extra)")
        sub.add_argument("--config-dir", type=Path, default=common.CONFIGS)
    args = parser.parse_args(argv)
    if args.figure == "policy-map":
        rows, arrays = build_policy_map(args.run, args.design, args.config_dir, args.cache_dir,
                                        args.rule_outcomes, args.rule_design)
        args.output.mkdir(parents=True, exist_ok=False)
        _write(args.output, "policy_map.json", {"design": args.design, "rule_design": args.rule_design,
                                                "run": str(args.run), "bearings": rows})
        np.savez_compressed(args.output / "policy_responses.npz", **arrays)
        if args.plot:
            plot_policy_map(rows, args.output / "policy_map")
    elif args.figure == "transfer-timing":
        rows = [row for path in args.outcomes for row in read_outcomes(path)]
        timing_rows = transfer_timing(rows, args.designs)
        args.output.mkdir(parents=True, exist_ok=False)
        _write(args.output, "transfer_timing.json", {"designs": args.designs, "bearings": timing_rows})
        if args.plot:
            plot_transfer_timing(timing_rows, args.designs, args.output / "transfer_timing")
    else:
        fits = [fit for run in args.runs for fit in json.loads((run / "fits.json").read_text(encoding="utf-8"))]
        rows = [row for run in args.runs for row in read_outcomes(run / "outcomes.csv")]
        data = age_clock(fits, rows, args.designs)
        args.output.mkdir(parents=True, exist_ok=False)
        _write(args.output, "age_clock.json", data)
        if args.plot:
            plot_age_clock(data, args.output / "age_clock")
    common.log(f"wrote {args.figure} data to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
