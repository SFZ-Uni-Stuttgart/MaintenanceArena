"""Real public workflows using synthetic scenarios and lightweight dependencies."""
import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import warnings

import gymnasium as gym
import numpy as np
import pytest

import maintenance_arena
from maintenance_arena import records
from maintenance_arena.cli import main, parser
from maintenance_arena.evaluation import evaluate, rollout
from maintenance_arena.profiles import all_profiles, resolve_profile


PUBLIC_PROFILES = (
    "paper-core-continuous-v1", "paper-gearbox-v1",
    "paper-resource-v1", "paper-resource-lifetime-v1",
)


@pytest.mark.parametrize("profile_id", PUBLIC_PROFILES)
def test_registered_public_environments_can_reset_and_step(profile_id):
    profile = resolve_profile(profile_id)
    assert profile["environment_id"] in gym.envs.registry
    env = maintenance_arena.make(profile_id)
    try:
        observation, initial = env.reset(seed=720019)
        assert env.observation_space.contains(observation)
        observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward) and np.isfinite(info["raw_profit_sum"])
        assert info["duration_calendar_steps"] > 0
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "events" in info and "ledger" in info
    finally:
        env.close()


def test_profile_overrides_have_stable_distinct_identity_and_do_not_mutate_defaults():
    original = resolve_profile("paper-core-continuous-v1")
    overrides = {"env_config": {"sensor_level": "light", "observation_route": "ROBUST_MULTISCALE_234"}}
    modified = resolve_profile("paper-core-continuous-v1", overrides)
    repeated = resolve_profile("paper-core-continuous-v1", overrides)
    assert modified["modified"] and not original["modified"]
    assert modified["configuration_sha256"] == repeated["configuration_sha256"]
    assert modified["id"] != original["id"]
    assert modified["configuration_sha256"] != original["configuration_sha256"]
    assert resolve_profile("paper-core-continuous-v1") == original
    overrides["env_config"]["sensor_level"] = "stress"
    assert modified["env_config"]["sensor_level"] == "light"


@pytest.mark.parametrize("profile_id", PUBLIC_PROFILES)
def test_unknown_environment_keys_are_rejected_as_user_configuration_errors(profile_id):
    with pytest.raises(ValueError, match="[Uu]nknown|[Uu]nexpected"):
        maintenance_arena.make(profile_id, config={"env_config": {"misspelled_setting": 1}})


def test_unknown_profile_sections_and_policies_are_rejected():
    with pytest.raises(ValueError):
        resolve_profile("paper-core-continuous-v1", {"observation": {"sensor": "light"}})
    with pytest.raises(ValueError, match="Policy"):
        rollout(resolve_profile("paper-core-continuous-v1"), policy="imaginary_policy",
                seed=700, calendar_steps=20)
    with pytest.raises(ValueError, match="checkpoint"):
        rollout(resolve_profile("paper-core-continuous-v1"), policy="ppo",
                seed=700, calendar_steps=20)


def test_cli_never_overwrites_an_existing_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "run.json"
    sentinel.write_bytes(b"original research or user output\r\n")
    before = sentinel.read_bytes()
    status = main(["simulate", "--profile", "paper-core-continuous-v1", "--policy", "immediate",
                   "--seed", "17", "--calendar-steps", "20", "--output", str(output)])
    assert status == 2
    assert sentinel.read_bytes() == before
    assert {path.name for path in output.iterdir()} == {"run.json"}


def test_cli_rejects_a_bad_policy_before_creating_output_so_a_corrected_retry_works(tmp_path, capsys):
    output = tmp_path / "retry"
    command = ["simulate", "--profile", "paper-core-continuous-v1", "--seed", "18",
               "--calendar-steps", "20", "--output", str(output)]
    assert main(command + ["--policy", "imaginary_policy"]) == 2
    error = capsys.readouterr().err
    assert "Policy 'imaginary_policy' is not supported by paper-core-continuous-v1" in error
    assert "choose from reactive, age, health, projection, immediate, ppo" in error
    assert "Traceback" not in error
    assert not output.exists()
    assert main(command + ["--policy", "immediate"]) == 0
    assert json.loads((output / "run.json").read_text(encoding="utf-8"))["status"] == "complete"
    capsys.readouterr()
    assert main(command + ["--policy", "immediate"]) == 2
    error = capsys.readouterr().err
    assert error == (f"maintenance-arena: output directory already exists: {output}. "
                     "Outputs are never overwritten; choose a new --output.\n")


@pytest.mark.parametrize("arguments,message", [
    (["evaluate", "--profile", "paper-gearbox-v1", "--seed", "1", "--policy", "dp", "dp"], "Duplicate policies"),
    (["evaluate", "--profile", "paper-gearbox-v1", "--seed", "1", "--policy", "dp", "ppo"], "requires --checkpoint"),
    (["evaluate", "--profile", "paper-gearbox-v1", "--seed", "1", "--policy", "ppo", "--checkpoint", "{missing}"],
     "--checkpoint path does not exist"),
    (["evaluate", "--profile", "paper-gearbox-v1", "--seed", "1", "--policy", "ppo", "--checkpoint", "{empty}"],
     "--checkpoint directory contains no final.pt"),
    (["evaluate", "--profile", "paper-resource-v1", "--seed", "1", "--policy", "ppo", "--checkpoint", "{empty}"],
     "--checkpoint directory contains no checkpoint.json"),
    (["simulate", "--profile", "paper-core-continuous-v1", "--seed", "1", "--policy", "projection", "--filter"],
     "execution filter is defined only for the frozen gearbox"),
    (["evaluate", "--profile", "paper-resource-v1", "--seed", "1", "--policy", "rolling", "--episodes", "0"],
     "episodes must be an integer >= 1"),
    (["evaluate", "--profile", "paper-resource-v1", "--seed", "1", "--policy", "rolling", "--plot"],
     r"defined for core profiles|maintenance-arena\[plots\]"),
    (["evaluate", "--profile", "paper-core-continuous-v1", "--seed", "1", "--policy", "immediate", "--fixture", "{missing}"],
     "--fixture path does not exist"),
    (["evaluate", "--profile", "paper-core-continuous-v1", "--policy", "immediate"],
     "evaluate needs --seed, or --fixture to replay the scenarios of an earlier run"),
    (["evaluate", "--profile", "paper-core-continuous-v1", "--policy", "immediate", "--seed", "-1"],
     "Seed must be nonnegative"),
    (["simulate", "--profile", "paper-resource-v1", "--seed", "1", "--policy", "rolling", "--config", "{bad_config}"],
     "unknown resource configuration keys"),
    (["train", "--profile", "paper-core-continuous-v1", "--seed", "1", "--calendar-steps", "1024",
      "--resume", "{empty}", "--initialize", "{empty}"], "--resume and --initialize are mutually exclusive"),
])
def test_cli_validates_arguments_before_creating_the_output(tmp_path, capsys, arguments, message):
    (tmp_path / "empty").mkdir()
    (tmp_path / "bad.json").write_text(json.dumps({"env_config": {"misspelled_setting": 1}}), encoding="utf-8")
    paths = {"{missing}": tmp_path / "missing", "{empty}": tmp_path / "empty", "{bad_config}": tmp_path / "bad.json"}
    output = tmp_path / "output"
    assert main([str(paths.get(value, value)) for value in arguments] + ["--output", str(output)]) == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and "Traceback" not in error
    assert re.search(message, error), error
    assert not output.exists()


def test_cli_reports_a_missing_optional_dependency_before_creating_the_output(tmp_path, capsys, monkeypatch):
    model = tmp_path / "model.pt"
    model.write_bytes(b"not loaded")
    available = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == "torch" else available(name, *args))
    output = tmp_path / "output"
    assert main(["evaluate", "--profile", "paper-core-continuous-v1", "--seed", "1", "--policy", "ppo",
                 "--checkpoint", str(model), "--output", str(output)]) == 2
    assert capsys.readouterr().err == "Missing optional dependency 'torch'; install maintenance-arena[rl].\n"
    assert not output.exists()


def test_cli_resource_training_budget_error_names_the_required_multiple(tmp_path, capsys):
    pytest.importorskip("sb3_contrib")
    output = tmp_path / "train"
    assert main(["train", "--profile", "paper-resource-v1", "--seed", "1", "--calendar-steps", "512",
                 "--output", str(output)]) == 2
    error = capsys.readouterr().err
    assert "calendar_steps must be an exact multiple of n_envs * n_steps = 8 * 256 = 2048" in error
    assert "got 512, use for example 2048" in error
    assert not output.exists()


def test_cli_help_documents_every_command_and_argument():
    root = parser()
    commands = next(action for action in root._actions if isinstance(action, argparse._SubParsersAction))
    summaries = {action.dest: action.help for action in commands._choices_actions}
    assert set(summaries) == set(commands.choices) == {"list", "tracks", "simulate", "evaluate", "benchmark",
                                                         "calibrate", "train"}
    assert all(summaries.values())
    for name, command in commands.choices.items():
        for action in command._actions:
            if not isinstance(action, argparse._HelpAction):
                assert action.help, (name, action.dest)
    for name, dest in (("simulate", "policy"), ("evaluate", "policy"), ("benchmark", "reference")):
        policy = next(action for action in commands.choices[name]._actions if action.dest == dest)
        for profile_id, profile in all_profiles().items():
            assert f"{profile_id}: {', '.join(profile['policies'])}" in policy.help
    assert "maintenance-arena list" in root.format_help()


@pytest.mark.parametrize("profile_id", PUBLIC_PROFILES)
def test_ansi_rendering_declares_a_frame_rate_without_checker_warnings(profile_id):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = maintenance_arena.make(profile_id, render_mode="ansi")
        try:
            env.reset(seed=3)
            assert isinstance(env.render(), str)
        finally:
            env.close()
    assert env.unwrapped.metadata["render_fps"] == 4
    assert not [warning for warning in caught if "render fps" in str(warning.message)]


def test_python_example_runs_every_profile_without_warnings():
    example = Path(__file__).resolve().parents[1] / "examples" / "run_environment.py"
    if not example.is_file():
        pytest.skip("examples are not available next to the tests")
    package_root = Path(maintenance_arena.__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(package_root), os.environ.get("PYTHONPATH")])))
    completed = subprocess.run([sys.executable, str(example), "--calendar-steps", "60"], capture_output=True,
                               text=True, env=environment, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "WARN" not in completed.stderr and "Warning" not in completed.stderr, completed.stderr
    for profile_id in all_profiles():
        assert completed.stdout.count(f'"profile": "{profile_id}"') == 1


def test_cli_simulation_writes_complete_nineteen_step_exchange_trace(tmp_path):
    output = tmp_path / "immediate_core"
    assert main(["simulate", "--profile", "paper-core-continuous-v1", "--policy", "immediate",
                 "--seed", "18", "--calendar-steps", "20", "--output", str(output)]) == 0
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    result = json.loads((output / "trajectory.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    assert result["actual_calendar_steps"] == 19
    assert result["decisions"] == 1
    assert result["terminated"] and not result["truncated"]
    assert result["failures"] == 0 and result["planned_interventions"] == 1
    ledger = result["trace"][0]["ledger"]
    assert [row["event"] for row in ledger] == ["mobilize"] * 12 + ["planned_exchange"] + ["downtime"] * 6
    assert [row["raw_profit"] for row in ledger] == [255.] * 12 + [-545.] + [-25.] * 6
    assert result["raw_profit"] == sum(row["raw_profit"] for row in ledger)
    assert (output / "decisions.csv").is_file()


@pytest.mark.parametrize("profile_id", ["paper-resource-v1", "paper-resource-lifetime-v1"])
def test_cli_resource_calendar_budget_overrides_a_short_profile_horizon(tmp_path, profile_id):
    override = tmp_path / "short_horizon.json"
    override.write_text(json.dumps({"env_config": {"horizon": 72}}), encoding="utf-8")
    output = tmp_path / profile_id
    assert main(["simulate", "--profile", profile_id, "--policy", "reactive",
                 "--seed", "19", "--calendar-steps", "73", "--config", str(override),
                 "--output", str(output)]) == 0
    result = json.loads((output / "trajectory.json").read_text(encoding="utf-8"))
    assert result["actual_calendar_steps"] == 73
    assert result["nominal_calendar_steps"] == 73
    assert result["effective_accounting_horizon"] == 73
    assert result["truncated"] and not result["terminated"]
    assert result["trace"][-1]["calendar_end"] == 73


def test_same_fixture_replays_paired_policy_scenarios_and_failure_totals():
    profile = resolve_profile("paper-core-continuous-v1")
    first = evaluate(profile, policies=["reactive", "immediate"], seed=7219,
                     episodes=3, calendar_steps=300)
    replay = evaluate(profile, policies=["reactive", "immediate"], seed=999999,
                      episodes=99, calendar_steps=300, fixture=first["fixture"])
    assert replay == first
    rows = first["units"]
    for unit in range(3):
        paired = [row for row in rows if row["paired_unit"] == unit]
        assert len(paired) == 2 and paired[0]["seed"] == paired[1]["seed"]
        left, right = (row["analysis_diagnostics"] for row in paired)
        assert left["family_id"] == right["family_id"]
        assert left["eol_operating_age"] == right["eol_operating_age"]
    summaries = {row["policy"]: row for row in first["summary"]}
    assert summaries["reactive"]["failures"] == 3
    assert summaries["reactive"]["corrective_interventions"] == 3
    assert summaries["immediate"]["failures"] == 0
    assert summaries["immediate"]["planned_interventions"] == 3
    for policy, summary in summaries.items():
        selected = [row for row in rows if row["policy"] == policy]
        assert summary["failures"] == sum(row["failures"] for row in selected)
        assert summary["raw_profit"] == sum(row["raw_profit"] for row in selected)
        assert summary["actual_calendar_steps"] == sum(row["actual_calendar_steps"] for row in selected)


@pytest.mark.parametrize("profile_id", ["paper-resource-v1", "paper-resource-lifetime-v1"])
def test_resource_failure_totals_count_failed_assets_not_only_episodes(profile_id):
    profile = resolve_profile(profile_id, {"env_config": {
        "lifetime_range": [2, 2], "initial_ages": [0, 0, 0]}})
    result = evaluate(profile, policies=["reactive"], seed=7251, episodes=2, calendar_steps=2)
    assert [row["failures"] for row in result["units"]] == [3, 3]
    assert result["summary"][0]["failures"] == 6
    assert result["summary"][0]["independent_synthetic_scenarios"] == 2


def test_a_fixture_replays_its_scenarios_without_a_seed(tmp_path, capsys):
    profile = resolve_profile("paper-core-continuous-v1")
    first = evaluate(profile, policies=["reactive", "projection"], seed=7219, episodes=3, calendar_steps=300)
    assert evaluate(profile, policies=["reactive", "projection"], episodes=1, calendar_steps=300,
                    fixture=first["fixture"]) == first
    with pytest.raises(ValueError, match="seed is required unless a fixture supplies the scenario seeds"):
        evaluate(profile, policies=["reactive"], episodes=1, calendar_steps=300)
    # The command needs no --seed with --fixture, and its run record keeps the seed empty.
    seeded = tmp_path / "seeded"
    assert main(["evaluate", "--profile", "paper-core-continuous-v1", "--policy", "reactive", "projection",
                 "--seed", "7219", "--episodes", "3", "--calendar-steps", "300", "--output", str(seeded)]) == 0
    replay = tmp_path / "replay"
    assert main(["evaluate", "--profile", "paper-core-continuous-v1", "--policy", "reactive", "projection",
                 "--calendar-steps", "300", "--fixture", str(seeded / "fixture.json"), "--output", str(replay)]) == 0
    capsys.readouterr()
    replayed = json.loads((replay / "evaluation.json").read_text(encoding="utf-8"))
    assert replayed == json.loads((seeded / "evaluation.json").read_text(encoding="utf-8"))
    assert replayed["fixture"]["seeds"] == first["fixture"]["seeds"]
    run = json.loads((replay / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete" and run["seed"] is None and run["inputs"]["fixture"]["sha256"]


def test_fixture_binds_environment_and_budget_but_allows_policy_tuning():
    profile = resolve_profile("paper-core-continuous-v1")
    result = evaluate(profile, policies=["immediate"], seed=72, episodes=2, calendar_steps=24)
    fixture = result["fixture"]
    assert fixture["schema"] == "maintenance_arena_synthetic_fixture_v2"
    altered = resolve_profile("paper-core-continuous-v1", {"env_config": {"failure_event_cost": 8000}})
    with pytest.raises(ValueError, match="incompatible"):
        evaluate(altered, policies=["immediate"], seed=72, episodes=2, calendar_steps=24, fixture=fixture)
    with pytest.raises(ValueError, match="incompatible"):
        evaluate(profile, policies=["immediate"], seed=72, episodes=2, calendar_steps=25, fixture=fixture)
    policy_tuned = resolve_profile("paper-core-continuous-v1", {"policy_defaults": {"projection": {"projection_threshold": 17}}})
    paired = evaluate(policy_tuned, policies=["projection"], seed=1, episodes=99, calendar_steps=24, fixture=fixture)
    assert paired["fixture"] == fixture


def test_repeated_policies_and_scenario_seeds_do_not_inflate_independent_counts():
    profile = resolve_profile("paper-core-continuous-v1")
    with pytest.raises(ValueError, match="Duplicate policies"):
        evaluate(profile, policies=["immediate", "immediate"], seed=1, episodes=2, calendar_steps=20)
    fixture = evaluate(profile, policies=["immediate"], seed=1, episodes=1, calendar_steps=20)["fixture"]
    fixture["seeds"] *= 2
    with pytest.raises(ValueError, match="Duplicate scenario"):
        evaluate(profile, policies=["immediate"], seed=1, episodes=2, calendar_steps=20, fixture=fixture)
    fixture["schema"] = "maintenance_arena_synthetic_fixture_v1"
    with pytest.raises(ValueError, match="regenerate unbound v1"):
        evaluate(profile, policies=["immediate"], seed=1, episodes=2, calendar_steps=20, fixture=fixture)


@pytest.mark.parametrize("argument,value", [("seed", -1), ("seed", True), ("seed", 1.5),
    ("episodes", 0), ("episodes", 1.5), ("calendar_steps", True), ("calendar_steps", 20.5)])
def test_public_evaluation_rejects_invalid_integer_contracts(argument, value):
    kwargs = dict(seed=1, episodes=1, calendar_steps=20)
    kwargs[argument] = value
    with pytest.raises(ValueError, match=argument):
        evaluate(resolve_profile("paper-core-continuous-v1"), policies=["immediate"], **kwargs)


def test_modified_profile_does_not_inherit_a_published_validation_claim():
    profile = resolve_profile("paper-core-continuous-v1", {"env_config": {"sensor_level": "light"}})
    assert "validation is not inherited" in profile["evidence_scope"]
    assert profile["base_evidence_scope"] == resolve_profile("paper-core-continuous-v1")["evidence_scope"]


def test_atomic_json_failure_preserves_previous_record_and_exclusive_write(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    records.write_json(path, {"status": "running"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        records.write_json(path, {"status": "different"}, exclusive=True)
    assert path.read_bytes() == original
    def cannot_publish(*args):
        raise OSError("simulated atomic publication failure")
    monkeypatch.setattr(records.os, "replace", cannot_publish)
    with pytest.raises(OSError, match="publication failure"):
        records.write_json(path, {"status": "complete"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_keyboard_interrupt_is_recorded_without_erasing_the_run(tmp_path, monkeypatch):
    import maintenance_arena.evaluation as evaluation
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(evaluation, "rollout", interrupted)
    output = tmp_path / "interrupted"
    status = main(["simulate", "--profile", "paper-core-continuous-v1", "--policy", "immediate",
                   "--seed", "17", "--output", str(output)])
    assert status == 130
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "interrupted"
    assert record["error"]["type"] == "KeyboardInterrupt"
    assert record["finished_utc"] and record["elapsed_seconds"] >= 0


def test_environment_cutoff_is_distinct_from_requested_rollout_budget():
    profile = resolve_profile("paper-core-continuous-v1", {"env_config": {"calendar_horizon": 1}})
    result = rollout(profile, policy="immediate", seed=17, calendar_steps=24)
    assert result["actual_calendar_steps"] == 19
    assert result["budget_overshoot"] == 0
    assert result["environment_calendar_budget"] == result["effective_accounting_horizon"] == 1
    assert result["environment_budget_overshoot"] == 18
    assert result["terminated"] and not result["truncated"]


@pytest.mark.parametrize("name", ["../model.pt", "models/../../model.pt", "C:/models/model.pt", "models\\model.pt", "/model.pt", "models//model.pt"])
def test_artifact_paths_are_portable_and_confined(tmp_path, name):
    from maintenance_arena.policies import contained_file
    with pytest.raises(ValueError, match="portable relative"):
        contained_file(tmp_path, {"path": name, "sha256": "0" * 64})
