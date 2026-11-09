"""Age effect after transfer on the XJTU-SY bearings within and beyond the FEMTO lifetime range (article Section 6.3).

Usage from the repository root, with the package installed::

    python paper/transfer_split.py                  # print the values as JSON
    python paper/transfer_split.py --output FILE    # also write them to a new JSON file

The age effect is the unused life of PPO: No age minus that of PPO: Full, in percentage points,
from the saved first orders in outcomes/bearings.csv. The paired bearing bootstrap of the article
(maintenance_arena.bearings.paired_bootstrap, 10,000 replicates, PCG64 seed 20260930) gives the
intervals for all fifteen bearings, for the bearings within and beyond the longest FEMTO test
(468 minutes), without each of the three longest bearings and without all three. The script
first checks that all fifteen bearings give the article's interval, -17.03 [-34.27, -0.92].
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from maintenance_arena.bearings import ReplayEconomics, paired_bootstrap

OUTCOMES = Path(__file__).resolve().parent / "outcomes" / "bearings.csv"
FEMTO_MAX = 468
LONGEST = ("Bearing3_1", "Bearing3_2", "Bearing3_4")
DESIGNS = ("PPO: Full", "PPO: No age")


def rows(path: Path = OUTCOMES) -> list[dict]:
    economics = ReplayEconomics()
    result = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["group"] != "XJTU-SY" or r["method"] not in DESIGNS:
                continue
            life, order = int(r["last_productive_step"]), (None if r["order_step"] == "" else int(r["order_step"]))
            o = economics.outcome(life, order)
            result.append({"design": r["method"], "dataset": "XJTU-SY", "bearing": r["unit"], "seed": r["run"],
                           "last_productive_minute": life, "order_minute": order, "cash": o.cash,
                           "calendar_minutes": o.calendar, "failed": int(o.failed),
                           "unused_fraction": o.unused_fraction})
    return result


def contrast(subset: list[dict], metric: str = "unused_life") -> dict:
    out = paired_bootstrap(subset, "PPO: No age", "PPO: Full", dataset="XJTU-SY", metric=metric)
    return {"bearings": out["bearings"], "point": round(out["point"], 4), "ci95": [round(v, 4) for v in out["ci95"]]}


def compute(path: Path = OUTCOMES) -> dict:
    data = rows(path)
    lives = {r["bearing"]: r["last_productive_minute"] for r in data}
    full = contrast(data)
    if f"{full['point']:.2f}" != "-17.03" or [f"{v:.2f}" for v in full["ci95"]] != ["-34.27", "-0.92"]:
        raise ValueError(f"all fifteen bearings give {full}, not the article's -17.03 [-34.27, -0.92]")
    return {"metric": "unused life, PPO: No age minus PPO: Full, percentage points",
            "all_15": full,
            "within_femto_range": contrast([r for r in data if lives[r["bearing"]] <= FEMTO_MAX]),
            "beyond_femto_range": contrast([r for r in data if lives[r["bearing"]] > FEMTO_MAX]),
            "leave_one_out": {b: contrast([r for r in data if r["bearing"] != b]) for b in LONGEST},
            "without_three_longest": contrast([r for r in data if r["bearing"] not in LONGEST]),
            "groups": {"within_femto_range": sorted(b for b in lives if lives[b] <= FEMTO_MAX),
                       "beyond_femto_range": sorted(b for b in lives if lives[b] > FEMTO_MAX)},
            "failures_all_15": contrast(data, "failure"),
            "retained_profit_all_15": contrast(data, "retained_profit")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outcomes", type=Path, default=OUTCOMES, help="the bearing outcome file")
    parser.add_argument("--output", type=Path, help="new JSON file for the values")
    args = parser.parse_args(argv)
    try:
        values = compute(args.outcomes)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(values, indent=1) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
