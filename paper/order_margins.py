"""Median order margins and per-bearing age effects that the article quotes in its text.

Usage from any working directory::

    python paper/order_margins.py                  # print the values as JSON
    python paper/order_margins.py --output FILE    # also write them to a new JSON file

The margin of one order is (T - o) - L with L = 12, as in Eq. (1) of the article: zero at the last
safe order, positive for an earlier order and negative for a late one. The median skips missing
orders, and the late share counts them as late. On the XJTU-SY bearings, the unused life of an
order is its positive margin over T - L and zero for a late or missing order, averaged over the
seeds of a bearing. The age effect of a bearing is the unused life of PPO: No age minus that of
PPO: Full, in percentage points. Like rebuild_tables.py, the script needs only the standard library.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import rebuild_tables as rt

NEW_LAWS = ("Paris-inspired", "Two-knee", "Gompertz", "Power-law")


def margin(outcome: rt.Outcome) -> int | None:
    if outcome.order_step is None:
        return None
    return outcome.last_productive_step - outcome.order_step - rt.PREPARATION


def summary(outcomes) -> dict:
    values = [margin(o) for o in outcomes]
    finite = [v for v in values if v is not None]
    late = sum(1 for v in values if v is None or v < 0)
    return {"median_margin": float(statistics.median(finite)), "late_pct": round(100 * (late / len(values)), 2),
            "rows": len(values)}


def per_bearing(outcomes) -> dict:
    """XJTU-SY age effect of every bearing, sorted by its last productive minute."""
    unused = defaultdict(list)
    lives = {}
    for o in outcomes:
        value = margin(o)
        usable = o.last_productive_step - rt.PREPARATION
        unused[(o.unit, o.method)].append(0.0 if value is None or value < 0 else value / usable)
        lives[o.unit] = o.last_productive_step
    return {unit: {"last_productive_minute": lives[unit],
                   "no_age_minus_full_pp": round(100 * (statistics.fmean(unused[(unit, "PPO: No age")])
                                                        - statistics.fmean(unused[(unit, "PPO: Full")])), 2)}
            for unit in sorted(lives, key=lambda unit: (lives[unit], unit))}


def compute(directory: Path = rt.OUTCOMES) -> dict:
    source = rt.read_study("source_comparison", directory)
    change = rt.read_study("degradation_change", directory)
    bearings = rt.read_study("bearings", directory)
    femto_max = max(o.last_productive_step for o in bearings if o.group == "FEMTO")
    xjtu = per_bearing([o for o in bearings if o.group == "XJTU-SY" and o.method in ("PPO: Full", "PPO: No age")])
    rows = [{"bearing": unit, **values, "beyond_femto_range": values["last_productive_minute"] > femto_max}
            for unit, values in xjtu.items()]
    beyond = [row["no_age_minus_full_pp"] for row in rows if row["beyond_femto_range"]]
    return {
        "source_laws": {m: summary([o for o in source if o.method == m]) for m in ("PPO", "Rule", "Naive projection rule")},
        "new_laws": {m: {law: summary([o for o in change if o.method == m and o.group == law]) for law in NEW_LAWS}
                     for m in dict.fromkeys(o.method for o in change)},
        "xjtu_per_bearing": {"femto_longest_last_productive_minute": femto_max, "rows": rows,
                             "beyond_range_all_negative": all(v < 0 for v in beyond),
                             "beyond_range_span_pp": [max(beyond), min(beyond)]},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outcomes", type=Path, default=rt.OUTCOMES, help="directory of the outcome files")
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
