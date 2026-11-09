"""Refresh the implementation inventory in provenance.json.

Usage from the project root after editing runtime or profile files::

    python tools/record_implementation.py --refresh-current --project .

Only ``current_implementation`` is rewritten: the project version and the SHA-256
of every runtime file below ``src/maintenance_arena``. tools/check_release.py
rejects a missing or stale inventory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

from export_project import collect


SCHEMA = "maintenance_arena_provenance_v2"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def current_implementation(project):
    """Inventory the runtime files of the source project."""
    sources = collect(project)
    specification = tomllib.loads(sources["pyproject.toml"].read_text(encoding="utf-8"))["project"]
    return {
        "schema": "maintenance_arena_implementation_v1",
        "project_version": specification["version"],
        "files": {name: sha(path) for name, path in sources.items() if name.startswith("src/maintenance_arena/")},
    }


def refresh_current(project):
    project = Path(project).resolve()
    output = project / "provenance.json"
    raw = output.read_bytes()
    result = json.loads(raw.decode("utf-8"))
    if result.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported provenance index (expected schema {SCHEMA}); the file was not changed")
    # The inventory is the only changed field; the other fields describe the file itself.
    result["current_implementation"] = current_implementation(project)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    # Keep the file's existing line endings on every platform, so that a refresh
    # changes nothing but the inventory.
    newline = "\r\n" if b"\r\n" in raw else "\n"
    output.write_bytes(encoded.replace("\n", newline).encode("utf-8"))
    return {"output": str(output), "current_files": len(result["current_implementation"]["files"]),
            "project_version": result["current_implementation"]["project_version"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--refresh-current", action="store_true", required=True,
                        help="Update the implementation inventory in provenance.json")
    args = parser.parse_args(argv)
    try:
        result = refresh_current(args.project)
    except (OSError, ValueError) as error:
        parser.exit(2, f"maintenance-arena provenance: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
