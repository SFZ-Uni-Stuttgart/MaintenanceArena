"""Portable distribution boundaries, preservation and reproducible source export."""
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import tomllib
import zipfile

import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "tools"))
    return importlib.import_module("export_project"), importlib.import_module("check_release")


@pytest.fixture
def standalone(tmp_path, release_tools):
    export, _ = release_tools
    project = tmp_path / "standalone"
    for name, source in export.collect(PROJECT).items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return project


def test_source_export_is_deterministic_and_inventory_binds_every_member(tmp_path, standalone, release_tools):
    export, _ = release_tools
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    a, b = export.export_project(standalone, first), export.export_project(standalone, second)
    assert a["sha256"] == b["sha256"]
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read("EXPORT_MANIFEST.json"))
        assert set(archive.namelist()) == {row["path"] for row in manifest["files"]} | {"EXPORT_MANIFEST.json"}
        for row in manifest["files"]:
            data = archive.read(row["path"])
            assert row["bytes"] == len(data)
            assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert archive.read(".github/workflows/maintenance-arena.yml") == (standalone / export.WORKFLOW).read_bytes()
        assert ".github/dependabot.yml" in archive.namelist()
        assert "CONTRIBUTING.md" in archive.namelist()
        assert "docs/RELEASING.md" in archive.namelist()


def test_export_leaves_research_like_storage_untouched_and_excluded(tmp_path, standalone, release_tools):
    export, _ = release_tools
    excluded = ["outputs/results.json", "docs/outputs/receipt.json", "src/maintenance_arena/checkpoints/model.json",
                "examples/datasets/measurements.yaml", "tests/.cache/fixture.json", "src/maintenance_arena/secret.pkl"]
    for name in excluded:
        path = standalone / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"retained original bytes\r\n")
    result = export.collect(standalone)
    assert not set(excluded) & set(result)
    export.export_project(standalone, tmp_path / "project.zip")
    for name in excluded:
        assert (standalone / name).read_bytes() == b"retained original bytes\r\n"


def test_export_refuses_existing_file_without_overwriting(tmp_path, standalone, release_tools):
    export, _ = release_tools
    output = tmp_path / "previous.zip"
    output.write_bytes(b"previous release")
    with pytest.raises(FileExistsError, match="overwrite"):
        export.export_project(standalone, output)
    assert output.read_bytes() == b"previous release"


def test_export_requires_release_metadata(standalone, release_tools):
    export, _ = release_tools
    (standalone / "provenance.json").unlink()
    with pytest.raises(FileNotFoundError, match="provenance"):
        export.collect(standalone)


def test_export_ships_the_benchmark_tracks_and_requires_the_runtime_data(standalone, release_tools):
    export, _ = release_tools
    collected = export.collect(PROJECT)
    for name in ("src/maintenance_arena/data/benchmark_tracks.json", "src/maintenance_arena/tracks.py",
                 "src/maintenance_arena/wrappers.py", "docs/BENCHMARK.md", "examples/benchmark_your_agent.py",
                 "examples/train_maskable_resource.py", "tools/make_benchmark_tracks.py",
                 ".github/ISSUE_TEMPLATE/benchmark_result.md", "tests/test_tracks.py", "tests/test_wrappers.py"):
        assert name in collected, name
    assert set(export.RUNTIME_DATA) <= set(collected)
    (standalone / "src/maintenance_arena/data/benchmark_tracks.json").unlink()
    with pytest.raises(FileNotFoundError, match="benchmark_tracks.json"):
        export.collect(standalone)


def test_export_reads_the_workflow_only_from_the_project(tmp_path, standalone, release_tools):
    export, _ = release_tools
    # A workflow in an enclosing directory is never used, even in a code/ layout.
    nested = tmp_path / "enclosing" / "code" / "maintenance_arena"
    shutil.copytree(standalone, nested)
    outer = tmp_path / "enclosing" / export.WORKFLOW
    outer.parent.mkdir(parents=True)
    shutil.move(nested / export.WORKFLOW, outer)
    with pytest.raises(FileNotFoundError, match="workflow"):
        export.collect(nested)
    (standalone / export.WORKFLOW).unlink()
    with pytest.raises(FileNotFoundError, match="workflow"):
        export.collect(standalone)


def test_export_ships_the_release_tools_and_the_citation(release_tools):
    export, _ = release_tools
    collected = export.collect(PROJECT)
    assert {"tools/export_project.py", "tools/check_release.py", "tools/record_implementation.py",
            "tools/make_benchmark_tracks.py", "CITATION.cff"} <= set(collected)


def test_manifest_template_selects_exactly_the_exported_files(standalone, release_tools, monkeypatch):
    # check_release requires the sdist to equal the export. Plant every kind of
    # file the exporter skips and apply MANIFEST.in the way setuptools does.
    egg_info = pytest.importorskip("setuptools.command.egg_info")
    export, _ = release_tools
    planted = ["outputs/results.json", "docs/outputs/receipt.json", "docs/archive/old.md",
               "examples/datasets/measurements.yaml", "tests/.cache/fixture.json", "tests/.hidden.py",
               "constraints/venv/pinned.txt", "tests/__pycache__/cached.py",
               ".github/.private/notes.md", "src/maintenance_arena/checkpoints/model.json"]
    for name in planted:
        path = standalone / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# skipped by the exporter\n", encoding="utf-8")
    monkeypatch.chdir(standalone)
    selection = egg_info.FileList()
    selection.findall()
    for line in (standalone / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            selection.process_template_line(line)
    selected = {Path(name).as_posix() for name in selection.files}
    # Setuptools itself adds MANIFEST.in, pyproject.toml and the package modules and data.
    expected = {name for name in export.collect(standalone)
                if name not in {"MANIFEST.in", "pyproject.toml"} and not name.startswith("src/")}
    assert selected == expected


def test_package_data_ships_every_exported_runtime_suffix(release_tools):
    # The wheel must hold every exported runtime file; check_release compares them.
    export, _ = release_tools
    configuration = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = set(configuration["tool"]["setuptools"]["package-data"]["maintenance_arena"])
    suffixes = export.TREE_SUFFIXES["src/maintenance_arena"] - {".py"}
    assert patterns == {f"**/*{suffix}" for suffix in suffixes}


def test_only_gitattributes_preserved_files_contain_carriage_returns(release_tools):
    # A carriage return anywhere else would be normalized away by Git on commit,
    # so the committed bytes would differ from the recorded implementation hashes.
    export, _ = release_tools
    sources = export.collect(PROJECT)
    assert ".gitattributes" in sources
    lines = sources[".gitattributes"].read_text(encoding="utf-8").splitlines()
    assert "* text=auto eol=lf" in lines
    preserved = {line.split()[0].removeprefix("/") for line in lines
                 if line and not line.startswith("#") and line.split()[1:] == ["-text"]}
    with_carriage_returns = {name for name, path in sources.items() if b"\r" in path.read_bytes()}
    assert with_carriage_returns == preserved


@pytest.mark.parametrize("names", [["../escape.py"], ["/absolute.py"], ["C:/absolute.py"],
                                    ["src\\module.py"], ["src/../module.py"],
                                    ["src/module.py", "SRC/Module.py"], ["src//module.py"]])
def test_distribution_members_reject_nonportable_or_conflicting_names(names, release_tools):
    _, check = release_tools
    with pytest.raises(ValueError, match="path"):
        check._portable_members(names)


def test_export_rejects_links_when_supported(tmp_path, standalone, release_tools):
    export, _ = release_tools
    source = tmp_path / "outside.py"
    source.write_text("# outside the project", encoding="utf-8")
    link = standalone / "examples/linked.py"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this host")
    with pytest.raises(ValueError, match="ordinary"):
        export.collect(standalone)


def test_refresh_current_rewrites_only_the_implementation_inventory(standalone, release_tools):
    import record_implementation

    index = standalone / "provenance.json"
    original = {"schema": "maintenance_arena_provenance_v2", "meaning": "kept as it is"}
    index.write_text(json.dumps(original), encoding="utf-8")
    first = record_implementation.refresh_current(standalone)
    recorded = json.loads(index.read_text(encoding="utf-8"))
    assert {key: value for key, value in recorded.items() if key != "current_implementation"} == original
    current = recorded["current_implementation"]
    assert first["current_files"] == len(current["files"])
    assert "src/maintenance_arena/data/profiles.json" in current["files"]
    assert "src/maintenance_arena/learning/ppo.py" in current["files"]
    module = standalone / "src/maintenance_arena/cli.py"
    module.write_bytes(module.read_bytes() + b"\n# maintenance edit\n")
    assert record_implementation.current_implementation(standalone) != current
    record_implementation.refresh_current(standalone)
    refreshed = json.loads(index.read_text(encoding="utf-8"))
    assert {key: value for key, value in refreshed.items() if key != "current_implementation"} == original
    assert refreshed["current_implementation"] == record_implementation.current_implementation(standalone)


def test_refresh_current_rejects_an_unknown_index_without_changing_it(standalone, release_tools):
    import record_implementation

    index = standalone / "provenance.json"
    index.write_text(json.dumps({"schema": "unknown"}), encoding="utf-8")
    before = index.read_bytes()
    with pytest.raises(ValueError, match="Unsupported provenance index"):
        record_implementation.refresh_current(standalone)
    assert index.read_bytes() == before


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
def test_refresh_current_keeps_the_index_line_endings(standalone, release_tools, newline):
    # A refresh on any platform must not translate the line endings of the file.
    import record_implementation

    index = standalone / "provenance.json"
    recorded = json.loads(index.read_bytes())
    index.write_bytes((json.dumps(recorded, indent=2, sort_keys=True) + "\n").replace("\n", newline).encode("utf-8"))
    record_implementation.refresh_current(standalone)
    data = index.read_bytes()
    expected = (json.dumps(json.loads(data), indent=2, sort_keys=True) + "\n").replace("\n", newline)
    assert data == expected.encode("utf-8")
    record_implementation.refresh_current(standalone)
    assert index.read_bytes() == data


def test_release_check_rejects_stale_current_provenance_before_artifact_read(standalone, release_tools):
    import record_implementation

    _, check = release_tools
    record_implementation.refresh_current(standalone)
    profile = standalone / "src/maintenance_arena/data/profiles.json"
    profile.write_bytes(profile.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="provenance.*stale"):
        check.check_release(standalone, standalone / "dist")


def test_release_check_compares_a_non_ascii_readme_exactly(release_tools):
    # Wheel METADATA carries the README as UTF-8 text. Reading it without
    # decoding would replace every non-ASCII character and reject a valid README.
    _, check = release_tools
    readme = "# Title\n\nVersion 0.1.0 · CPython 3.12, order × job, clip(−1, 1)\n"
    project = {"name": "maintenance-arena", "version": "0.1.0", "requires-python": ">=3.12",
               "license": "MIT", "license-files": ["LICENSE"], "dependencies": ["numpy>=2.4.2,<3"],
               "optional-dependencies": {}}
    raw = ("Metadata-Version: 2.4\nName: maintenance-arena\nVersion: 0.1.0\n"
           "Requires-Python: >=3.12\nLicense-Expression: MIT\nLicense-File: LICENSE\n"
           "Requires-Dist: numpy<3,>=2.4.2\nDescription-Content-Type: text/markdown\n\n"
           + readme).encode("utf-8")
    assert check._metadata(raw, project, readme)["Name"] == "maintenance-arena"
    with pytest.raises(ValueError, match="README differs"):
        check._metadata(raw, project, readme.replace("×", "x"))


# One version everywhere: the distribution metadata, the runtime, the records and the documents.

def _project_version():
    return tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def test_package_version_is_the_same_in_metadata_runtime_and_provenance():
    import maintenance_arena

    version = _project_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert maintenance_arena.__version__ == version
    provenance = json.loads((PROJECT / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["current_implementation"]["project_version"] == version


def test_documents_name_the_package_version():
    version = _project_version()
    documents = {name: PROJECT / name for name in ("README.md", "CHANGELOG.md", "SECURITY.md", "docs/RELEASING.md",
                                                   "CITATION.cff")}
    if not all(path.is_file() for path in documents.values()):
        pytest.skip("documents are not available next to the tests")
    text = {name: path.read_text(encoding="utf-8") for name, path in documents.items()}
    newest = re.search(r"^## (\d+\.\d+\.\d+)\b", text["CHANGELOG.md"], flags=re.M)
    assert newest and newest.group(1) == version, "the newest CHANGELOG.md heading must name the version"
    assert f"\nVersion {version} · " in text["README.md"], "README status line"
    assert f"## Scope of {version} and roadmap" in text["README.md"]
    major_minor = version.rsplit(".", 1)[0]
    assert f"| {major_minor}.x | Yes |" in text["SECURITY.md"]
    assert f"currently {version}." in text["SECURITY.md"]
    assert f"\nversion: {version}\n" in text["CITATION.cff"], "CITATION.cff names another version"
    artifacts = re.findall(r"maintenance_arena-(\d+\.\d+\.\d+)", text["docs/RELEASING.md"])
    assert artifacts and set(artifacts) == {version}, "release commands name other artifact versions"


def test_cli_reports_the_package_version(capsys):
    from maintenance_arena.cli import main

    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out == f"MaintenanceArena {_project_version()}\n"
