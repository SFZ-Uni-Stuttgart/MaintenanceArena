"""Check built wheel/sdist contents against the allowlisted source project.

Run after ``python -m build`` (isolated; CI uses ``--no-isolation`` with the
pinned constraints) from the project root or from a source export::

    python tools/check_release.py --dist dist

This offline check neither installs artifacts nor uploads them. It verifies
metadata, README bytes, runtime/source completeness and the distribution boundary.
It does not replace hosted package-index rendering or scientific validation.
"""
from __future__ import annotations

import argparse
import ast
import configparser
from email.parser import BytesParser
from email.policy import default
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile

from packaging.requirements import Requirement
from packaging.markers import Marker

from export_project import collect
from record_implementation import current_implementation


def _portable_members(names):
    seen = set()
    for name in names:
        path = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or path.is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError(f"Non-portable distribution path: {name}")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate distribution path: {name}")
        seen.add(name.casefold())


def _metadata(raw, project_metadata, readme):
    metadata = BytesParser(policy=default).parsebytes(raw)
    expected = {
        "Name": project_metadata["name"],
        "Version": project_metadata["version"],
        "Requires-Python": project_metadata["requires-python"],
        "License-Expression": project_metadata["license"],
        "Description-Content-Type": "text/markdown",
    }
    for field, value in expected.items():
        if metadata[field] != value:
            raise ValueError(f"Metadata mismatch for {field}: {metadata[field]!r} != {value!r}")
    if set(metadata.get_all("Provides-Extra", [])) != set(project_metadata["optional-dependencies"]):
        raise ValueError("Optional dependency extras are missing or unexpected")
    if set(metadata.get_all("License-File", [])) != set(project_metadata["license-files"]):
        raise ValueError("License file metadata differs from the declared files")
    base = {Requirement(value) for value in metadata.get_all("Requires-Dist", []) if ";" not in value}
    if base != {Requirement(value) for value in project_metadata["dependencies"]}:
        raise ValueError("Base dependencies differ from the project metadata")
    expected_requirements = set(base)
    for extra, values in project_metadata["optional-dependencies"].items():
        for value in values:
            requirement = Requirement(value)
            marker = f'({requirement.marker}) and ' if requirement.marker else ""
            requirement.marker = Marker(marker + f'extra == "{extra}"')
            expected_requirements.add(requirement)
    if {Requirement(value) for value in metadata.get_all("Requires-Dist", [])} != expected_requirements:
        raise ValueError("Optional dependency requirements differ from the project metadata")
    # METADATA is UTF-8 text. get_payload() without decode=True replaces every
    # non-ASCII byte with U+FFFD, so any README with non-ASCII text would differ.
    description = metadata.get_payload(decode=True).decode("utf-8")
    if description.replace("\r\n", "\n").strip() != readme.replace("\r\n", "\n").strip():
        raise ValueError("Distribution README differs from the project README")
    return {key: str(metadata[key]) for key in expected}


def check_release(project: Path, dist: Path):
    project, dist = Path(project).resolve(), Path(dist).resolve()
    sources = collect(project)
    provenance = json.loads(sources["provenance.json"].read_text(encoding="utf-8"))
    if provenance.get("current_implementation") != current_implementation(project):
        raise ValueError("Current implementation provenance is missing or stale; run tools/record_implementation.py --refresh-current --project .")
    specification = tomllib.loads(sources["pyproject.toml"].read_text(encoding="utf-8"))["project"]
    # Parse the version without importing the package or optional dependencies.
    tree = ast.parse(sources["src/maintenance_arena/__init__.py"].read_text(encoding="utf-8"))
    versions = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__"
                        for target in node.targets) and isinstance(node.value, ast.Constant)]
    if versions != [specification["version"]]:
        raise ValueError("Runtime version and distribution version differ")
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Use a distribution directory containing exactly one wheel and one sdist")
    wheel, sdist = wheels[0], sdists[0]
    readme = sources["README.md"].read_text(encoding="utf-8")
    with zipfile.ZipFile(wheel) as archive:
        _portable_members(archive.namelist())
        contents = {name: archive.read(name) for name in archive.namelist()}
    metadata_paths = [name for name in contents if name.endswith(".dist-info/METADATA")]
    if len(metadata_paths) != 1:
        raise ValueError("The wheel must have exactly one package metadata directory")
    info = metadata_paths[0].rsplit("/", 1)[0]
    wheel_metadata = _metadata(contents[metadata_paths[0]], specification, readme)
    runtime = {name.removeprefix("src/"): path for name, path in sources.items() if name.startswith("src/")}
    allowed_info = {f"{info}/{name}" for name in
                    ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt")}
    allowed_info.update(f"{info}/licenses/{name}" for name in specification["license-files"])
    if set(contents) != set(runtime) | allowed_info:
        raise ValueError(f"Wheel content mismatch: {sorted(set(contents) ^ (set(runtime) | allowed_info))}")
    for name, path in runtime.items():
        if contents[name] != path.read_bytes():
            raise ValueError(f"Wheel source bytes differ: {name}")
    for name in specification["license-files"]:
        if contents[f"{info}/licenses/{name}"] != sources[name].read_bytes():
            raise ValueError(f"Wheel license bytes differ: {name}")
    entry_points = configparser.ConfigParser()
    entry_points.read_string(contents[f"{info}/entry_points.txt"].decode("utf-8"))
    if dict(entry_points["console_scripts"]) != specification["scripts"]:
        raise ValueError("Console entry points differ from the public interface")
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        _portable_members(member.name for member in members)
        if any(not (member.isfile() or member.isdir()) for member in members):
            raise ValueError("The sdist contains a link or special file")
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        if len(roots) != 1:
            raise ValueError("The sdist must have one project root")
        source_contents = {member.name.split("/", 1)[1]: archive.extractfile(member).read()
                           for member in members if member.isfile()}
    for name, path in sources.items():
        if source_contents.get(name) != path.read_bytes():
            raise ValueError(f"Sdist source missing or changed: {name}")
    extras = set(source_contents) - set(sources)
    allowed_generated = {"PKG-INFO", "setup.cfg"}
    allowed_generated.update(f"src/maintenance_arena.egg-info/{name}" for name in
                            ("PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
                             "requires.txt", "top_level.txt"))
    if extras - allowed_generated:
        raise ValueError(f"Unexpected sdist contents: {sorted(extras - allowed_generated)}")
    _metadata(source_contents["PKG-INFO"], specification, readme)
    return {
        "schema": "maintenance_arena_release_check_v1", "status": "passed",
        "metadata": wheel_metadata, "wheel_runtime_files": len(runtime),
        "source_files": len(sources),
        "current_implementation_files": len(provenance["current_implementation"]["files"]),
        "artifacts": [{"name": path.name, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in (wheel, sdist)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = check_release(args.project, args.dist)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, tarfile.TarError) as error:
        parser.exit(2, f"maintenance-arena release check: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
