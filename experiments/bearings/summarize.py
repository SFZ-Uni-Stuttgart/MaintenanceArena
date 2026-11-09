"""Equal-bearing summary, paired bootstrap contrasts and candidate ranking of one or more outcome files.

    python experiments/bearings/summarize.py outputs/bearings-femto/outcomes.csv outputs/bearings-forecast-rules/outcomes.csv --reference "Forecast rule: Full" --output outputs/bearings-femto-summary
    python experiments/bearings/summarize.py outputs/bearings-candidates/outcomes.csv --select candidate_selection --output outputs/bearings-selection

Contrasts are given with equal bearing weights for the three metrics and as pooled-profit
differences relative to the pooled last-safe rate. ``--select`` ranks the designs of an
experiment's configuration that occur in the files, by pooled held-out profit. Writes
summary.json and summary.md into a new output directory and prints the tables.
"""
import argparse
import json
from pathlib import Path

from maintenance_arena.bearings import (ReplayEconomics, design_summary, paired_bootstrap, pooled_bootstrap,
                                        rank_candidates, read_outcomes)
from maintenance_arena.bearings.metrics import METRICS, summarize_rows

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outcomes", type=Path, nargs="+", help="outcome CSV files of the runners")
    parser.add_argument("--reference", help="design compared with every other design of the same dataset")
    parser.add_argument("--select", help="experiment whose PPO designs are ranked by pooled held-out profit")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--config-dir", type=Path, default=HERE / "configs")
    args = parser.parse_args(argv)
    settings = json.loads((args.config_dir / "settings.json").read_text(encoding="utf-8"))
    economics = ReplayEconomics(**settings["economics"])
    rows = [row for path in args.outcomes for row in read_outcomes(path)]
    summary = design_summary(rows, economics)
    contrasts, pooled, selection = [], [], {}
    if args.reference:
        for dataset in sorted({row["dataset"] for row in summary}):
            designs = [row["design"] for row in summary if row["dataset"] == dataset]
            if args.reference not in designs:
                continue
            for design in (d for d in designs if d != args.reference):
                contrasts += [paired_bootstrap(rows, design, args.reference, dataset=dataset, metric=metric,
                                               economics=economics, **settings["bootstrap"]) for metric in METRICS]
                pooled.append(pooled_bootstrap(rows, design, args.reference, dataset=dataset, economics=economics,
                                               **settings["bootstrap"]))
    if args.select:
        experiments = json.loads((args.config_dir / "experiments.json").read_text(encoding="utf-8"))["experiments"]
        designs = experiments[args.select]["designs"]
        present = {(row["dataset"], row["design"]) for row in rows}
        for dataset in sorted({dataset for dataset, _ in present}):
            candidates = {name: spec for name, spec in designs.items()
                          if spec["kind"] == "ppo" and (dataset, name) in present}
            if candidates:
                selection[dataset] = rank_candidates(rows, candidates, dataset=dataset, economics=economics)
    args.output.mkdir(parents=True, exist_ok=False)
    lines = [summarize_rows(summary)]
    if contrasts:
        lines += ["", f"Differences from {args.reference}, equal bearing weights, percentage points, 95% paired "
                      "bootstrap intervals:", "", "| Dataset | Design | Metric | Difference | Interval |",
                  "|---|---|---|---:|---|"]
        lines += [f"| {c['dataset']} | {c['design']} | {c['metric']} | {c['point']:.2f} | "
                  f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}] |" for c in contrasts]
        lines += ["", f"Pooled-profit differences from {args.reference}, percent of the pooled last-safe rate:", "",
                  "| Dataset | Design | Difference | Interval | Share positive |", "|---|---|---:|---|---:|"]
        lines += [f"| {c['dataset']} | {c['design']} | {100 * c['point']:.2f} | "
                  f"[{100 * c['ci95'][0]:.2f}, {100 * c['ci95'][1]:.2f}] | {c['share_positive']:.4f} |" for c in pooled]
    for dataset, result in selection.items():
        lines += ["", f"{dataset} candidates by pooled held-out profit (selected: {result['selected']}):", "",
                  "| Design | Profit per minute | Pooled retained profit (%) | Failed replays | Steps | Inputs |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {r['design']} | {r['profit_per_calendar_minute']:.4f} | {r['pooled_retained_profit_pct']:.2f} | "
                  f"{r['failure_outcomes']} | {r['calendar_steps']} | {r['policy_inputs']} |" for r in result["ranking"]]
    text = "\n".join(lines) + "\n"
    (args.output / "summary.md").write_text(text, encoding="utf-8", newline="\n")
    (args.output / "summary.json").write_text(json.dumps({"inputs": [str(p) for p in args.outcomes], "summary": summary,
                                                          "contrasts": contrasts, "pooled_contrasts": pooled,
                                                          "selection": selection}, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
