"""Export the source project without data, models or run artifacts.

Usage from any working directory::

    python tools/export_project.py --project . --output /new/path/maintenance-arena.zip

The ZIP has a flat project root and a deterministic SHA-256 inventory. The CI
workflow and the other repository settings come from this project's own .github
directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tomllib
import zipfile


WORKFLOW = Path(".github/workflows/maintenance-arena.yml")
# Top-level files. MANIFEST.in includes the same names. Files below .github
# (workflows, dependabot, issue and pull-request templates) and below docs
# (including docs/figures/*.svg) are covered by TREE_SUFFIXES.
METADATA = (
    "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    ".gitignore", ".gitattributes", "CITATION.cff", "MANIFEST.in", "provenance.json", "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md", "SECURITY.md",
)
REQUIRED = METADATA
# Package data that the runtime reads: the frozen profiles and the benchmark tracks.
# The tree rules below export them; an export without them is refused.
RUNTIME_DATA = ("src/maintenance_arena/data/profiles.json", "src/maintenance_arena/data/benchmark_tracks.json")
TREE_SUFFIXES = {
    ".github": {".yml", ".yaml", ".md"},
    "src/maintenance_arena": {".py", ".json", ".yaml", ".yml", ".typed"},
    "docs": {".md", ".json", ".svg"},
    "tests": {".py", ".json", ".yaml", ".yml"},
    "examples": {".py", ".json", ".yaml", ".yml", ".md"},
    "experiments": {".py", ".json", ".yaml", ".yml", ".md"},
    "paper": {".py", ".json", ".md", ".tex", ".csv"},
    "tools": {".py"},
    "constraints": {".txt", ".md"},
}
# Skipped at any depth inside the trees above: caches, run outputs, models and data.
# MANIFEST.in prunes the same directories from the sdist.
EXCLUDED_DIRECTORIES = {
    "__pycache__", "outputs", "models", "checkpoints", "artifacts", "cache",
    "datasets", "research", "archive", "build", "dist", "tmp", "venv",
}


def _ordinary(path: Path) -> None:
    """Never follow a symlink or Windows junction while collecting the export."""
    if path.is_symlink() or path.is_junction():
        raise ValueError(f"Export sources must be ordinary files/directories: {path}")


def collect(project: Path) -> dict[str, Path]:
    """Return only allowlisted source files, with portable collision checks."""
    project = Path(project).absolute()
    _ordinary(project)
    project = project.resolve(strict=True)
    for name in REQUIRED:
        _ordinary(project / name)
        if not (project / name).is_file():
            raise FileNotFoundError(f"Required project metadata missing: {name}")
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    if metadata.get("project", {}).get("name") != "maintenance-arena":
        raise ValueError("--project must be the maintenance-arena standalone source project")
    if not (project / "src/maintenance_arena/__init__.py").is_file():
        raise FileNotFoundError("The maintenance_arena source package is missing")
    files: dict[str, Path] = {}
    identities: set[str] = set()

    def add(name: str, path: Path) -> None:
        _ordinary(path)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"Export source is not a regular file: {path}")
        folded = name.casefold()
        if folded in identities:
            raise ValueError(f"Case-insensitive export path collision: {name}")
        identities.add(folded)
        files[name] = path

    for name in METADATA:
        path = project / name
        if path.is_file():
            add(name, path)
    for root, suffixes in TREE_SUFFIXES.items():
        source = project / root
        if not source.exists():
            continue
        for parent in [source, *source.parents]:
            _ordinary(parent)
            if parent == project:
                break
        for directory, names, filenames in os.walk(source, followlinks=False):
            names[:] = sorted(name for name in names
                              if not name.startswith(".")
                              and name.casefold() not in EXCLUDED_DIRECTORIES
                              and not name.endswith(".egg-info"))
            for name in names:
                _ordinary(Path(directory) / name)
            for name in sorted(filenames):
                path = Path(directory) / name
                relative = path.relative_to(project).as_posix()
                if not name.startswith(".") and path.suffix.lower() in suffixes:
                    add(relative, path)
    if WORKFLOW.as_posix() not in files:
        raise FileNotFoundError(f"CI workflow missing: {WORKFLOW.as_posix()}")
    for name in RUNTIME_DATA:
        if name not in files:
            raise FileNotFoundError(f"Runtime package data missing: {name}")
    return dict(sorted(files.items()))


def export_project(project: Path, output: Path) -> dict:
    sources = collect(project)
    output = Path(output).absolute()
    if output.suffix.lower() != ".zip":
        raise ValueError("--output must name a new .zip file")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite export: {output}")
    # Read before creating the ZIP: source errors cannot leave a partial export,
    # and the bytes bound by each hash are exactly the bytes written below.
    contents = {name: path.read_bytes() for name, path in sources.items()}
    manifest = {
        "schema_version": "maintenance_arena_source_export_v1",
        "project": "maintenance-arena",
        "files": [{"path": name, "bytes": len(data),
                   "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in contents.items()],
        "scope": "standalone source project; no scientific data, models or run outputs",
    }
    contents["EXPORT_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data, compresslevel=9)
    return {"path": str(output.resolve()), "files": len(contents),
            "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = export_project(args.project, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"maintenance-arena export: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
