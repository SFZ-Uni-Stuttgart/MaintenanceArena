"""Check the raw-data layout, or compute the per-acquisition statistics caches once for reuse by the runners.

    python experiments/bearings/prepare_statistics.py --data-root DATA --check
    python experiments/bearings/prepare_statistics.py --data-root DATA --output outputs/bearing-statistics --jobs 8

The check lists the bearings and acquisition counts found and names every missing folder,
bearing or file; it opens no file. Pass the cache directory as ``--cache-dir`` to the
runners. Existing caches are never replaced.
"""
import common

import argparse
from pathlib import Path
import sys
import time

from maintenance_arena.bearings import DATASETS, check_layout, prepare


def print_report(report: dict) -> None:
    print(f"{report['dataset']}: {report['directory']}")
    for folder, bearings in report["folders"].items():
        found = ", ".join(f"{name} ({count})" for name, count in bearings.items()) or "no bearings"
        print(f"  {folder}: {found}")
    for kind in ("missing", "problems", "notes"):
        for line in report[kind]:
            print(f"  {kind[:-1] if kind != 'missing' else 'missing'}: {line}")
    print(f"  {'complete' if report['ok'] else 'incomplete'}: {sum(map(len, report['folders'].values()))} bearings")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_data_options(parser)
    parser.add_argument("--check", action="store_true", help="check the layout only; exit 1 if anything is missing")
    parser.add_argument("--output", type=Path, help="cache directory for <dataset>_statistics.npz")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["femto", "xjtu_sy"])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.data_root is None and any(common.dataset_option(args, d) is None for d in args.datasets):
        parser.error("pass --data-root (or set MAINTENANCE_ARENA_DATA) or the directory of every dataset")
    if args.check:
        reports = [check_layout(args.data_root, d, directory=common.dataset_option(args, d)) for d in args.datasets]
        for report in reports:
            print_report(report)
        return 0 if all(report["ok"] for report in reports) else 1
    if args.output is None:
        parser.error("--output is required unless --check is given")
    for dataset in args.datasets:
        started = time.perf_counter()
        statistics = prepare(args.data_root, dataset, args.output / f"{dataset}_statistics.npz", jobs=args.jobs,
                             directory=common.dataset_option(args, dataset))
        acquisitions = sum(len(values) for values in statistics.values())
        seconds = time.perf_counter() - started
        common.log(f"{dataset}: {len(statistics)} bearings, {acquisitions} acquisitions in {seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
