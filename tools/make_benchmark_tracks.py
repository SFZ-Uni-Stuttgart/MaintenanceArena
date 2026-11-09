"""Re-derive the benchmark tracks file from the definition fields of its tracks.

Usage from the project root::

    python tools/make_benchmark_tracks.py --check
    python tools/make_benchmark_tracks.py --write

Each track's definition fields (name, description, profile, agent_observation,
calendar_steps, root_seed, test_scenarios, calibration_scenarios, references,
calibrated_rules) are kept. The derived fields (profile_configuration_sha256,
environment_id, task and both seed lists) are computed again with
``maintenance_arena.tracks.define_track``: the seeds from the documented root seed via
NumPy's SeedSequence. ``--check`` compares the result with the file byte for byte and
exits with status 1 on any difference; ``--write`` rewrites the file. To add a track,
append an entry with the definition fields only and run ``--write``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT / "src" / "maintenance_arena" / "data" / "benchmark_tracks.json"
DEFINITION = ("name", "description", "profile", "agent_observation", "calendar_steps", "root_seed",
              "test_scenarios", "calibration_scenarios", "references", "calibrated_rules")


def rebuild(text: str) -> str:
    """The canonical text of a tracks file, re-derived from its definition fields."""
    if str(PROJECT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT / "src"))
    from maintenance_arena.tracks import define_track, dumps_tracks, tracks_document
    document = json.loads(text)
    tracks = []
    for entry in document["tracks"]:
        missing = [name for name in DEFINITION if name not in entry]
        if missing:
            raise ValueError(f"Track {entry.get('name')!r} lacks the definition fields {missing}")
        values = {name: entry[name] for name in DEFINITION}
        tracks.append(define_track(values.pop("name"), values.pop("profile"), **values))
    return dumps_tracks(tracks_document(tracks, document.get("default_track")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="Tracks file (default: the packaged file)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Exit with status 1 if the file differs from its rebuild")
    mode.add_argument("--write", action="store_true", help="Rewrite the file with the rebuilt content")
    args = parser.parse_args(argv)
    try:
        current = args.path.read_bytes()
        expected = rebuild(current.decode("utf-8")).encode("utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"maintenance-arena tracks: {error}\n")
    if args.check:
        if current != expected:
            print(f"{args.path} differs from its rebuild; run with --write and review the change", file=sys.stderr)
            return 1
        print(f"{args.path}: up to date")
        return 0
    args.path.write_bytes(expected)
    print(f"{args.path}: written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
