"""Benchmark tracks: the packaged definitions, their seed lists, run_track and the track options of the CLI."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pytest

from maintenance_arena import benchmark, calibration, evaluation, tracks
from maintenance_arena.cli import main
from maintenance_arena.profiles import all_profiles, resolve_profile


PROJECT = Path(__file__).resolve().parents[1]
PACKAGED = Path(tracks.__file__).resolve().parent / "data" / "benchmark_tracks.json"
CORE = "paper-core-continuous-v1"
# The tiny test tracks calibrate on a few lives, which need not include every wear family; the
# warning that reports it is tested in tests/test_calibration.py.
pytestmark = pytest.mark.filterwarnings("ignore::maintenance_arena.calibration.FamilyCoverageWarning")


def _tiny(name="tiny-core", profile=CORE, **overrides):
    values = dict(root_seed=97, test_scenarios=5, calibration_scenarios=12, description="a tiny test track")
    values.update(overrides)
    return tracks.define_track(name, profile, **values)


def never(observation, info):
    return 0


class Recorder:
    """Agent that records what the harness passes to it, then waits."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return 0


# The packaged tracks.

def test_packaged_tracks_cover_every_environment_on_unchanged_v1_profiles():
    packaged = tracks.load_tracks()
    assert tracks.track_names() == tuple(packaged) == ("core-continuous-v1", "core-zero-deadline-v1", "gearbox-v1",
                                                       "resource-v1", "resource-lifetime-v1")
    document = json.loads(PACKAGED.read_text(encoding="utf-8"))
    assert document["schema"] == tracks.SCHEMA and document["default_track"] == tracks.DEFAULT_TRACK
    profiles = all_profiles()
    assert {track.environment_id for track in packaged.values()} == {profile["environment_id"] for profile in profiles.values()}
    for track in packaged.values():
        profile = resolve_profile(track.profile)
        assert not profile["modified"] and track.profile in profiles
        assert track.profile_configuration_sha256 == profile["configuration_sha256"]
        assert track.references[0] == benchmark.PRIMARY_REFERENCES[track.task]
        assert set(track.references) == set(benchmark.default_references(profile))
        assert track.calibrated_rules == (("age", "health", "projection") if track.task == "core" else ())
        assert track.calendar_steps == 720
    assert "oracle" not in json.dumps([profile["policies"] for profile in profiles.values()])


@pytest.mark.parametrize("name", tracks.track_names())
def test_track_seed_lists_follow_the_documented_root_seed_derivation(name):
    track = tracks.load_track(name)
    test, calibration_list = tracks.track_seeds(track.root_seed, track.test_scenarios, track.calibration_scenarios)
    assert tuple(test) == track.test_seeds and tuple(calibration_list) == track.calibration_seeds
    # The test scenarios are those that evaluate and benchmark derive from the root seed ...
    _, derived = evaluation._scenario_seeds(track.resolved_profile(), seed=track.root_seed,
                                            episodes=track.test_scenarios, calendar_steps=track.calendar_steps,
                                            fixture=None)
    assert tuple(derived) == track.test_seeds
    # ... and the calibration scenarios come from the calibration stream, without test seeds.
    assert track.calibration_seeds == tuple(calibration.calibration_seeds(
        track.root_seed, track.calibration_scenarios, exclude=track.test_seeds))
    assert track.calibration_seeds[0] == int(np.random.SeedSequence(track.root_seed, spawn_key=(1, 0))
                                             .generate_state(1, dtype=np.uint32)[0])


def test_calibration_and_test_seeds_are_disjoint_within_and_across_tracks():
    packaged = list(tracks.load_tracks().values())
    for track in packaged:
        assert len(set(track.test_seeds)) == track.test_scenarios
        assert len(set(track.calibration_seeds)) == track.calibration_scenarios
        assert not set(track.test_seeds) & set(track.calibration_seeds)
        for other in packaged:
            if other.profile == track.profile:
                assert not set(track.calibration_seeds) & set(other.test_seeds), (track.name, other.name)


def test_the_packaged_file_is_its_own_canonical_rebuild():
    spec = importlib.util.spec_from_file_location("make_benchmark_tracks", PROJECT / "tools" / "make_benchmark_tracks.py")
    if spec is None or not (PROJECT / "tools" / "make_benchmark_tracks.py").is_file():
        pytest.skip("tools are not available next to the tests")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    data = PACKAGED.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert tool.rebuild(data.decode("utf-8")).encode("utf-8") == data
    document = json.loads(data)
    assert tracks.dumps_tracks(document) == data.decode("utf-8")
    assert json.loads(tracks.dumps_tracks(document)) == document


def test_track_identity_and_round_trip():
    track = tracks.load_track()
    assert track.name == tracks.DEFAULT_TRACK
    assert tracks.Track.from_dict(track.to_dict()) == track
    assert list(track.to_dict()) == list(tracks.FIELDS)
    summary = track.summary()
    assert "test_seeds" not in summary and "calibration_seeds" not in summary
    assert summary["sha256"] == track.sha256 == tracks.load_track(track.name).sha256
    assert track.fixture() == evaluation.make_fixture(resolve_profile(CORE), list(track.test_seeds), calendar_steps=720)
    changed = track.to_dict()
    changed["test_seeds"] = [changed["test_seeds"][1], changed["test_seeds"][0], *changed["test_seeds"][2:]]
    assert tracks.Track.from_dict(changed).sha256 != track.sha256


def _mutated(change):
    entry = _tiny().to_dict()
    change(entry)
    return entry


@pytest.mark.parametrize("change,message", [
    (lambda e: e.update(profile="paper-core-imaginary-v1"), "unknown profile"),
    (lambda e: e.update(profile_configuration_sha256="0" * 64), "another configuration"),
    (lambda e: e.update(environment_id="maintenanceArena/Gearbox-v0"), "uses maintenanceArena/CoreCycle-v0"),
    (lambda e: e.update(task="gearbox"), "the task of"),
    (lambda e: e.update(agent_observation="raw"), "agent_observation must be one of"),
    (lambda e: e.update(references=["reactive", "projection"]), "primary reference 'oracle'"),
    (lambda e: e.update(references=["oracle", "ppo"]), "ppo needs a trained model"),
    (lambda e: e.update(references=["oracle", "imaginary"]), "Unknown reference"),
    (lambda e: e.update(references=["oracle", "oracle"]), "twice"),
    (lambda e: e.update(calibrated_rules=["immediate"]), "calibrated_rules must name core rules"),
    (lambda e: e.update(references=["oracle", "projection"], calibrated_rules=["age"]), "calibrated_rules must name"),
    (lambda e: e.update(test_scenarios=6), "holds 5 seeds, but the track declares 6"),
    (lambda e: e["test_seeds"].__setitem__(1, e["test_seeds"][0]), "seed twice"),
    (lambda e: e["calibration_seeds"].__setitem__(0, e["test_seeds"][0]), "both test and calibration seeds"),
    (lambda e: e["test_seeds"].__setitem__(0, -1), "integer >= 0"),
    (lambda e: e.pop("root_seed"), "missing \\['root_seed'\\]"),
    (lambda e: e.update(leaderboard=True), "unknown \\['leaderboard'\\]"),
    (lambda e: e.update(name="my/track.json"), "file path"),
    (lambda e: e.update(calendar_steps=0), "integer >= 1"),
])
def test_invalid_track_definitions_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        tracks.Track.from_dict(_mutated(change))


def test_zero_deadline_tracks_and_track_profiles_are_restricted():
    with pytest.raises(ValueError, match="core task only"):
        _tiny("tiny-gearbox", "paper-gearbox-v1", agent_observation="zero_deadline")
    with pytest.raises(ValueError, match="packaged profile ID"):
        tracks.define_track("modified", resolve_profile(CORE, {"env_config": {"replacement_cost": 600.0}}),
                            root_seed=1, test_scenarios=2, calibration_scenarios=2)
    assert _tiny("tiny-gearbox", "paper-gearbox-v1").calibrated_rules == ()


def test_tracks_files_are_written_once_and_loaded_by_path(tmp_path):
    first, second = _tiny("first"), _tiny("second", root_seed=98)
    path = tracks.write_tracks(tmp_path / "custom.json", [first, second], default_track="second")
    assert path.read_bytes().count(b"\r") == 0
    with pytest.raises(FileExistsError):
        tracks.write_tracks(path, [first])
    assert tracks.load_tracks(path) == {"first": first, "second": second}
    assert tracks.load_track(str(path)) == second == tracks.load_track(path)
    single = tracks.write_tracks(tmp_path / "single.json", [first])
    assert tracks.load_track(single) == first
    assert tracks.load_track(first.to_dict()) == first and tracks.load_track(first) is first
    with pytest.raises(ValueError, match="Unknown track 'nope'; choose from core-continuous-v1"):
        tracks.load_track("nope")
    with pytest.raises(ValueError, match="does not exist"):
        tracks.load_track(tmp_path / "missing.json")


@pytest.mark.parametrize("change,message", [
    (lambda d: d.update(schema="other"), "needs schema"),
    (lambda d: d.update(default_track="absent"), "is not among"),
    (lambda d: d["tracks"].append(copy.deepcopy(d["tracks"][0])), "twice"),
    (lambda d: d.update(tracks=[]), "at least one track"),
    (lambda d: d.update(extra=1), "exactly schema"),
])
def test_invalid_tracks_files_are_rejected(tmp_path, change, message):
    document = tracks.tracks_document([_tiny()])
    change(document)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        tracks.load_tracks(path)


# Scoring on a track.

def test_run_track_scores_on_the_fixed_scenarios_with_rules_calibrated_on_the_calibration_seeds():
    track = _tiny()
    result = benchmark.run_track(track, {"mine": never})
    assert result.track == track.summary() and result.calibration_origin == "track"
    assert result.scenario_seeds == track.test_seeds
    assert result.evaluation["fixture"] == track.fixture()
    assert tuple(result.calibration["seeds"]) == track.calibration_seeds
    assert result.calibration["rules"] == ["age", "health", "projection"]
    assert result.calibration["excluded_seed_count"] == track.test_scenarios
    selected = result.calibration["selected"]
    for rule in ("age", "health", "projection"):
        assert result.row(rule)["threshold"] == selected[rule]
    record = result.to_dict()
    assert record["track"] == track.summary() and record["calibration"]["origin"] == "track"
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    table = result.to_markdown()
    assert table.startswith(f"Benchmark track `tiny-core` (sha256 {track.sha256[:12]}) on `{CORE}`: "
                            "5 test scenarios, up to 720 calendar steps each.\n")
    assert ("Rules calibrated for track tiny-core: selected by the zero-failure criterion (no failure, then the highest "
            "family-equal profit) on its 12 calibration lives") in table
    assert result.calibration["selection_criterion"] == record["calibration"]["selection_criterion"] == "zero_failure"
    assert [(c["policy"], c["reference"]) for c in result.comparisons] == [("mine", name) for name in track.references]
    # The same scores as a plain benchmark on the test seeds with the track's calibration applied.
    stored = calibration.CalibrationResult.from_dict(result.calibration)
    plain = benchmark.score({"mine": never}, profile=CORE, seeds=list(track.test_seeds),
                            references=list(track.references), calibration=stored)
    assert plain.rows == result.rows and plain.comparisons == result.comparisons
    # The pooled criterion on the same calibration scenarios.
    pooled = benchmark.run_track(track, {"mine": never}, criterion="pooled")
    assert pooled.calibration["selection_criterion"] == "pooled"
    assert pooled.calibration["seeds"] == result.calibration["seeds"]
    assert "selected by pooled profit on its 12 calibration lives" in pooled.to_markdown()
    with pytest.raises(ValueError, match="criterion must be one of zero_failure, pooled"):
        benchmark.run_track(track, {"mine": never}, criterion="any")


def test_run_track_is_deterministic_and_renders_the_same_table():
    track = _tiny(test_scenarios=4, calibration_scenarios=8)
    first, second = benchmark.run_track(track, never), benchmark.run_track(track.to_dict(), never)
    assert first.to_markdown() == second.to_markdown() == str(first)
    assert first.to_dict() == second.to_dict() and first.evaluation == second.evaluation


@pytest.mark.parametrize("observation", ["full", "zero_deadline"])
def test_agents_receive_the_observation_that_the_track_declares(observation):
    track = _tiny(agent_observation=observation, test_scenarios=2, calibration_scenarios=3, references=["oracle"])
    recorder = Recorder()
    result = benchmark.run_track(track, ("recorder", recorder))
    assert recorder.calls and result.calibration is None and result.calibration_origin is None
    assert result.row("recorder")["information"] == tracks.AGENT_OBSERVATIONS[observation]
    deadlines = []
    for args, kwargs in recorder.calls:
        assert kwargs == {} and len(args) == 2 and isinstance(args[1], dict)
        assert args[0].shape == (4,) and args[0].dtype == np.float32
        deadlines.append(float(args[0][3]))
    if observation == "zero_deadline":
        assert set(deadlines) == {0.0}
        assert "deadline coordinate (index 3) set to zero" in result.to_markdown().splitlines()[2]
    else:
        assert len(set(deadlines)) > 10


def test_zero_deadline_tracks_keep_the_scenarios_and_references_of_the_full_track():
    full = _tiny(test_scenarios=3, calibration_scenarios=6)
    zero = _tiny("tiny-zero", test_scenarios=3, calibration_scenarios=6, agent_observation="zero_deadline")
    assert full.test_seeds == zero.test_seeds and full.calibration_seeds == zero.calibration_seeds
    left, right = benchmark.run_track(full, never), benchmark.run_track(zero, never)
    assert [row for row in left.rows if row["policy"] != "never"] == [row for row in right.rows if row["policy"] != "never"]
    # An agent that ignores index 3 scores identically on both tracks.
    assert left.row("never")["pooled_profit_per_1000_calendar_steps"] == right.row("never")["pooled_profit_per_1000_calendar_steps"]


def test_tracks_without_rule_calibration_say_so():
    track = _tiny("tiny-gearbox", "paper-gearbox-v1", test_scenarios=2, calibration_scenarios=2, calendar_steps=96)
    result = benchmark.run_track(track)
    assert result.reference == "dp" and result.calibration is None
    assert "Rule calibration is defined for the core rules only; the gearbox rules keep the profile's settings." in result.notes
    assert "does not optimize the profit of a finite window" in result.reference_role
    with pytest.raises(ValueError, match="label applies to a single callable agent"):
        benchmark.run_track(track, {"a": never}, label="b")


# Command line.

@pytest.fixture
def tiny_tracks(tmp_path):
    path = tmp_path / "tiny_tracks.json"
    tracks.write_tracks(path, [_tiny(test_scenarios=4, calibration_scenarios=10)])
    return path


AGENT_MODULE = '''
def act(observation, info):
    health, rate, valid, deadline = (float(value) for value in observation)
    return int(valid > 0.5 and deadline >= -2 / 13)
'''


@pytest.fixture
def agent_module(tmp_path, monkeypatch):
    name = f"track_agents_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(AGENT_MODULE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield name
    sys.modules.pop(name, None)


def test_cli_benchmark_on_a_track_reproduces_its_table_byte_for_byte(tmp_path, tiny_tracks, agent_module, capsys):
    track = tracks.load_track(tiny_tracks)
    tables = []
    for run in ("first", "second"):
        output = tmp_path / run
        assert main(["benchmark", "--track", str(tiny_tracks), "--agent", f"{agent_module}:act",
                     "--output", str(output)]) == 0
        printed = capsys.readouterr().out
        table = (output / "benchmark.md").read_bytes()
        assert printed == table.decode("utf-8") + f"\nResults: {output.resolve()}\n"
        tables.append(table)
    assert tables[0] == tables[1] and b"\r" not in tables[0]
    output = tmp_path / "first"
    record = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert record["track"] == track.summary() and record["scenarios"] == 4
    assert {row["policy"] for row in record["rows"]} == {*track.references, "act"}
    fixture = json.loads((output / "fixture.json").read_text(encoding="utf-8"))
    assert fixture == track.fixture()
    stored = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    assert tuple(stored["seeds"]) == track.calibration_seeds and stored["selection_criterion"] == "zero_failure"
    run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_record["status"] == "complete" and run_record["seed"] == track.root_seed
    assert run_record["inputs"]["track"]["name"] == "tiny-core"
    assert run_record["inputs"]["track"]["track_sha256"] == track.sha256
    assert len(run_record["inputs"]["track"]["sha256"]) == 64
    assert run_record["result"]["track"]["name"] == "tiny-core"
    # The fixture replays the test scenarios in evaluate.
    replay = tmp_path / "replay"
    assert main(["evaluate", "--profile", CORE, "--seed", "0", "--policy", "projection", "--fixture",
                 str(output / "fixture.json"), "--output", str(replay)]) == 0
    replayed = json.loads((replay / "evaluation.json").read_text(encoding="utf-8"))
    assert replayed["fixture"]["seeds"] == list(track.test_seeds)


def test_cli_benchmark_on_a_track_takes_the_calibration_criterion(tmp_path, tiny_tracks, capsys):
    output = tmp_path / "pooled"
    assert main(["benchmark", "--track", str(tiny_tracks), "--criterion", "pooled", "--output", str(output)]) == 0
    assert "selected by pooled profit on its 10 calibration lives" in capsys.readouterr().out
    record = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert record["calibration"]["selection_criterion"] == "pooled"
    stored = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    assert stored["selection_criterion"] == "pooled" and stored["sha256"] == record["calibration"]["sha256"]
    run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run_record["arguments"]["criterion"] == "pooled"


def test_cli_tracks_lists_the_packaged_tracks_without_seed_lists(capsys):
    assert main(["tracks"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["default_track"] == tracks.DEFAULT_TRACK
    assert listing["tracks"] == {name: track.summary() for name, track in tracks.load_tracks().items()}


@pytest.mark.parametrize("arguments,message", [
    (["--track", "core-continuous-v1", "--seed", "0"], "remove --seed"),
    (["--track", "core-continuous-v1", "--profile", CORE], "remove --profile"),
    (["--track", "core-continuous-v1", "--episodes", "5", "--calendar-steps", "100"],
     "remove --calendar-steps, --episodes"),
    (["--track", "core-continuous-v1", "--reference", "oracle"], "remove --reference"),
    (["--track", "core-continuous-v1", "--no-recalibrate", "--calibration-lives", "10"],
     "remove --no-recalibrate, --calibration-lives"),
    (["--track", "gearbox-v1", "--filter"], "remove --filter"),
    (["--track", "core-continuous-v1", "--config", "{config}"], "remove --config"),
    (["--track", "imaginary-v1"], "Unknown track 'imaginary-v1'"),
    (["--track", "{missing}"], "Tracks file does not exist"),
    (["--track", "{broken}"], "needs schema"),
    (["--track", "core-continuous-v1", "--agent", "module_that_does_not_exist_12345:act"], "cannot import module"),
    (["--track", "core-continuous-v1", "--agent", "maintenance_arena.wrappers:zero_deadline", "--agent-label",
      "oracle"], "names a built-in policy or reference"),
    (["--profile", CORE], "needs --track, or --profile and --seed; missing --seed"),
    ([], "missing --profile and --seed"),
])
def test_cli_track_options_are_validated_before_creating_the_output(tmp_path, capsys, arguments, message):
    (tmp_path / "config.yaml").write_text("env_config: {}\n", encoding="utf-8")
    (tmp_path / "broken.json").write_text("{}", encoding="utf-8")
    paths = {"{config}": tmp_path / "config.yaml", "{missing}": tmp_path / "missing.json",
             "{broken}": tmp_path / "broken.json"}
    output = tmp_path / "rejected"
    assert main(["benchmark", "--output", str(output)] + [str(paths.get(value, value)) for value in arguments]) == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and "Traceback" not in error
    assert re.search(message, error), error
    assert not output.exists()


# The reference table of the default track in docs/BENCHMARK.md (about a minute of computing).

BEGIN, END = "<!-- default-track-table:begin -->", "<!-- default-track-table:end -->"


@pytest.mark.skipif(os.environ.get("MAINTENANCE_ARENA_FULL_TRACK") != "1",
                    reason="Set MAINTENANCE_ARENA_FULL_TRACK=1 to recompute the default track table of docs/BENCHMARK.md")
def test_the_documented_default_track_table_is_reproduced_byte_for_byte(tmp_path, capsys):
    document = PROJECT / "docs" / "BENCHMARK.md"
    if not document.is_file():
        pytest.skip("docs are not available next to the tests")
    text = document.read_text(encoding="utf-8")
    documented = text.split(BEGIN + "\n", 1)[1].split(END, 1)[0]
    output = tmp_path / "track-core"
    assert main(["benchmark", "--track", tracks.DEFAULT_TRACK, "--output", str(output)]) == 0
    capsys.readouterr()
    computed = (output / "benchmark.md").read_text(encoding="utf-8")
    assert computed == documented
    assert _first_table_and_thresholds(computed) == _readme_summary_chunks()


# The README shows the first table and the rule thresholds of the documented default-track table.

SUMMARY_BEGIN, SUMMARY_END = "<!-- default-track-summary:begin -->", "<!-- default-track-summary:end -->"
DEFAULT_TRACK_COMMAND = "python -m maintenance_arena benchmark --track core-continuous-v1 --output outputs/track-core"


def _readme_summary_chunks():
    readme = PROJECT / "README.md"
    if not readme.is_file():
        pytest.skip("README.md is not available next to the tests")
    text = readme.read_text(encoding="utf-8")
    assert text.count(SUMMARY_BEGIN) == 1 and text.count(SUMMARY_END) == 1
    assert DEFAULT_TRACK_COMMAND in text, "the README must state the command that regenerates its table"
    block = text.split(SUMMARY_BEGIN + "\n", 1)[1].split(SUMMARY_END, 1)[0]
    return [chunk + "\n" for chunk in block.strip("\n").split("\n\n")]


def _first_table_and_thresholds(benchmark_md):
    lines = benchmark_md.splitlines(keepends=True)
    start = next(index for index, line in enumerate(lines) if line.startswith("| Policy |"))
    stop = next(index for index in range(start, len(lines)) if not lines[index].startswith("|"))
    thresholds = [line for line in lines if line.startswith("- Rule thresholds:")]
    assert len(thresholds) == 1
    return ["".join(lines[start:stop]), thresholds[0]]


def test_the_readme_baseline_table_is_the_documented_default_track_table():
    document = PROJECT / "docs" / "BENCHMARK.md"
    if not document.is_file():
        pytest.skip("docs are not available next to the tests")
    text = document.read_text(encoding="utf-8")
    assert DEFAULT_TRACK_COMMAND in text
    documented = text.split(BEGIN + "\n", 1)[1].split(END, 1)[0]
    chunks = _readme_summary_chunks()
    assert chunks == _first_table_and_thresholds(documented)
    references = [line.split("|")[1].strip().strip("`") for line in chunks[0].splitlines()[2:]]
    assert references == ["oracle", "projection", "health", "age", "reactive", "immediate"]
    assert sorted(references) == sorted(tracks.load_track(tracks.DEFAULT_TRACK).references)
