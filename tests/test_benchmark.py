"""User agents in the paired evaluation, evaluation schema v3, the evaluator-side core oracle,
benchmark scoring and the agent options of the command line."""
import json
import math
import sys

import gymnasium as gym
import numpy as np
import pytest

from maintenance_arena import benchmark
from maintenance_arena.cli import main
from maintenance_arena.evaluation import (AGENT, BUILT_IN, EVALUATION_SCHEMA, METRIC_DEFINITIONS,
                                          PRIVILEGED_REFERENCE, evaluate, make_fixture, rollout)
from maintenance_arena.policies import Policy
from maintenance_arena.profiles import all_profiles, resolve_profile


CORE = "paper-core-continuous-v1"
PROFILES = ("paper-core-continuous-v1", "paper-gearbox-v1", "paper-resource-v1", "paper-resource-lifetime-v1")
# Simulator truth that only the evaluator may read.
PRIVILEGED_KEYS = {"eol_operating_age", "first_trigger_true_rul", "true_lifetime", "true_remaining_life",
                   "family_id", "trace", "analysis_pm_starts", "unused_life_hours"}
V2_SUMMARY_FIELDS = ("policy", "independent_synthetic_scenarios", "mean_profit_per_1000_calendar_steps",
                     "raw_profit", "actual_calendar_steps", "failures", "planned_interventions",
                     "corrective_interventions", "mean_normalized_discarded_life", "corrected_proposals")
V3_SUMMARY_FIELDS = ("policy_kind", "pooled_profit_per_1000_calendar_steps", "ended_lives",
                     "failures_per_1000_lives", "failures_per_1000_calendar_steps", "orders_placed",
                     "late_orders", "late_order_share", "order_margin_mean", "order_margin_sd", "invalid_actions")


def projection_agent(observation, info):
    """The built-in projection rule (threshold 15) computed from the observation alone."""
    health, rate, valid, _ = (float(value) for value in observation)
    if valid < .5:
        return 0
    remaining = 0. if health <= 0 else (1. + math.log(health) / math.log(5.)) / max(rate / 12., 1.e-12)
    return int(remaining <= 15)


def never(observation, info):
    return 0


class Recorder:
    """Agent that records exactly what the harness passes to it, then waits."""

    def __init__(self, action=0, mutate=False):
        self.calls, self.action, self.mutate = [], action, mutate

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.mutate:
            args[0][...] = -7
            args[1].clear()
        return self.action


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def _units(result, policy):
    return sorted((row for row in result["units"] if row["policy"] == policy), key=lambda row: row["paired_unit"])


def test_an_agent_reproduces_the_built_in_rule_it_implements_on_every_unit():
    profile = resolve_profile(CORE)
    result = evaluate(profile, policies=["projection"], agents={"my_projection": projection_agent},
                      seed=9101, episodes=12, calendar_steps=720)
    kinds = {summary["policy"]: summary["policy_kind"] for summary in result["summary"]}
    assert kinds == {"projection": BUILT_IN, "my_projection": AGENT}
    fields = ("seed", "raw_profit", "actual_calendar_steps", "decisions", "failures", "first_trigger_true_rul",
              "order_margin", "events")
    for built_in, agent in zip(_units(result, "projection"), _units(result, "my_projection"), strict=True):
        assert {field: built_in[field] for field in fields} == {field: agent[field] for field in fields}
    left, right = result["summary"]
    assert {key: value for key, value in left.items() if key not in ("policy", "policy_kind")} == {
        key: value for key, value in right.items() if key not in ("policy", "policy_kind")}


def test_agents_can_be_given_as_pairs_in_policies_or_as_a_mapping_and_keep_their_order():
    profile = resolve_profile(CORE)
    pairs = evaluate(profile, policies=["reactive", ("mine", projection_agent)], agents=[("idle", never)],
                     seed=3, episodes=2, calendar_steps=720)
    mapping = evaluate(profile, policies=["reactive"], agents={"mine": projection_agent, "idle": never},
                       seed=3, episodes=2, calendar_steps=720)
    assert [s["policy"] for s in pairs["summary"]] == ["reactive", "mine", "idle"]
    assert pairs == mapping
    single = rollout(profile, policy=("mine", projection_agent), seed=pairs["fixture"]["seeds"][0], calendar_steps=720)
    assert single["policy"] == "mine" and single["policy_kind"] == AGENT
    assert single["raw_profit"] == _units(pairs, "mine")[0]["raw_profit"]


@pytest.mark.parametrize("profile_id", PROFILES)
def test_agents_receive_only_copies_of_the_observation_and_public_info(profile_id):
    profile = resolve_profile(profile_id)
    recorder = Recorder()
    result = evaluate(profile, policies=[], agents={"recorder": recorder}, seed=17, episodes=2, calendar_steps=48)
    assert recorder.calls and len(recorder.calls) == sum(row["decisions"] for row in result["units"])
    env = gym.make(profile["environment_id"], config=profile["env_config"])
    space = env.observation_space
    env.close()
    for args, kwargs in recorder.calls:
        assert kwargs == {} and len(args) == 2
        observation, info = args
        assert isinstance(observation, np.ndarray) and space.contains(observation)
        assert isinstance(info, dict)
        for item in _walk(info):
            assert not isinstance(item, gym.Env) and not callable(item)
            assert not (isinstance(item, str) and item in PRIVILEGED_KEYS), item
    assert recorder.calls[0][0][1] is not recorder.calls[1][0][1]
    # Changing what the agent received cannot change the scenario.
    mutating = evaluate(profile, policies=[], agents={"recorder": Recorder(mutate=True)}, seed=17, episodes=2,
                        calendar_steps=48)
    assert mutating == result


@pytest.mark.parametrize("value", [np.int64(1), np.array(1), np.array([1]), 1])
def test_integer_like_agent_actions_are_accepted(value):
    row = rollout(resolve_profile(CORE), policy=("constant", lambda observation, info: value), seed=5, calendar_steps=40)
    assert row["planned_interventions"] == 1 and row["decisions"] == 1


@pytest.mark.parametrize("value", [1.0, True, np.bool_(True), [0, 1], "1", None])
def test_non_integer_agent_actions_are_rejected(value):
    with pytest.raises(ValueError, match="must return one integer action"):
        rollout(resolve_profile(CORE), policy=("bad", lambda observation, info: value), seed=5, calendar_steps=40)


@pytest.mark.parametrize("policies,agents,message", [
    ([], None, "nonempty"),
    ("projection", None, "sequence"),
    ([], {"oracle": never}, "names a built-in policy or reference"),
    ([], {"projection": never}, "names a built-in policy or reference"),
    ([], {" padded": never}, "without surrounding whitespace"),
    ([], {"broken": 3}, "must be callable"),
    ([("twice", never)], [("twice", never)], "Duplicate policies"),
    ([], [never], "tuples"),
    ([], "never", "mapping"),
])
def test_invalid_agent_specifications_are_rejected(policies, agents, message):
    with pytest.raises(ValueError, match=message):
        evaluate(resolve_profile(CORE), policies=policies, agents=agents, seed=1, episodes=1, calendar_steps=20)


@pytest.mark.parametrize("profile_id", PROFILES)
def test_schema_v3_adds_fields_next_to_unchanged_v2_fields(profile_id):
    profile = resolve_profile(profile_id)
    policies = [name for name in profile["policies"] if name != "ppo"]
    result = evaluate(profile, policies=policies, seed=4040, episodes=3, calendar_steps=240)
    assert result["schema"] == EVALUATION_SCHEMA == "maintenance_arena_synthetic_evaluation_v3"
    assert result["fixture"]["schema"] == "maintenance_arena_synthetic_fixture_v2"
    assert result["metric_definitions"] == METRIC_DEFINITIONS and result["unit_events"] == "recorded"
    assert "pooled_profit_per_1000_calendar_steps" in result["aggregation"]
    assert "accompanying article" in result["scope"]
    core = "CoreCycle" in profile["environment_id"]
    resource = "Resource" in profile["environment_id"]
    for summary in result["summary"]:
        assert tuple(summary) == V2_SUMMARY_FIELDS + V3_SUMMARY_FIELDS
        units = _units(result, summary["policy"])
        # The v2 fields keep their v2 definitions.
        assert summary["mean_profit_per_1000_calendar_steps"] == float(np.mean([u["profit_per_1000_calendar_steps"] for u in units]))
        assert summary["raw_profit"] == sum(u["raw_profit"] for u in units)
        assert summary["actual_calendar_steps"] == sum(u["actual_calendar_steps"] for u in units)
        assert summary["failures"] == sum(u["failures"] for u in units)
        # The v3 fields.
        assert summary["policy_kind"] == BUILT_IN
        assert summary["pooled_profit_per_1000_calendar_steps"] == 1000 * summary["raw_profit"] / summary["actual_calendar_steps"]
        assert summary["ended_lives"] == sum(u["ended_lives"] for u in units)
        assert summary["ended_lives"] == round(summary["planned_interventions"]) + summary["failures"]
        assert summary["failures_per_1000_calendar_steps"] == 1000 * summary["failures"] / summary["actual_calendar_steps"]
        if summary["ended_lives"]:
            assert summary["failures_per_1000_lives"] == 1000 * summary["failures"] / summary["ended_lives"]
        else:
            assert summary["failures_per_1000_lives"] is None
        assert (summary["invalid_actions"] == 0) if resource else (summary["invalid_actions"] is None)
        if not core:
            assert all(summary[name] is None for name in ("orders_placed", "late_orders", "late_order_share",
                                                          "order_margin_mean", "order_margin_sd"))
            assert all(u["order_margin"] is None for u in units)
    if not resource:
        # Core and gearbox lives end in a renewal: every failure is a corrective renewal.
        for unit in result["units"]:
            assert unit["failures"] == unit["corrective_interventions"]
            assert unit["ended_lives"] == unit["planned_interventions"] + unit["corrective_interventions"]


def test_core_order_margin_statistics_follow_the_first_order():
    result = evaluate(resolve_profile(CORE), policies=["reactive", "health", "projection"], seed=611,
                      episodes=6, calendar_steps=720)
    summaries = {summary["policy"]: summary for summary in result["summary"]}
    for policy, summary in summaries.items():
        units = _units(result, policy)
        margins = [u["order_margin"] for u in units if u["order_margin"] is not None]
        for unit in units:
            rul = unit["first_trigger_true_rul"]
            assert unit["order_margin"] == (None if rul is None else rul - 13)
        assert summary["orders_placed"] == len(margins)
        assert summary["late_orders"] == sum(margin < 0 for margin in margins)
        if margins:
            assert summary["order_margin_mean"] == pytest.approx(np.mean(margins))
            assert summary["order_margin_sd"] == pytest.approx(np.std(margins, ddof=1))
            assert summary["late_order_share"] == summary["late_orders"] / len(margins)
    reactive = summaries["reactive"]
    assert reactive["orders_placed"] == 0 and reactive["order_margin_mean"] is None and reactive["late_order_share"] is None
    assert reactive["failures_per_1000_lives"] == 1000.0 and reactive["ended_lives"] == 6


def test_pooled_profit_weights_scenarios_by_calendar_time():
    result = evaluate(resolve_profile(CORE), policies=["reactive"], seed=12, episodes=8, calendar_steps=720)
    summary = result["summary"][0]
    units = result["units"]
    assert len({unit["actual_calendar_steps"] for unit in units}) > 1
    pooled = 1000 * sum(u["raw_profit"] for u in units) / sum(u["actual_calendar_steps"] for u in units)
    assert summary["pooled_profit_per_1000_calendar_steps"] == pytest.approx(pooled, rel=1e-12)
    assert summary["pooled_profit_per_1000_calendar_steps"] != summary["mean_profit_per_1000_calendar_steps"]


def test_event_logging_can_be_omitted_without_changing_any_other_field():
    profile = resolve_profile("paper-gearbox-v1")
    recorded = evaluate(profile, policies=["dp", "separable_q96"], seed=8, episodes=2, calendar_steps=200)
    omitted = evaluate(profile, policies=["dp", "separable_q96"], seed=8, episodes=2, calendar_steps=200, events=False)
    assert omitted["unit_events"] == "omitted" and recorded["unit_events"] == "recorded"
    assert all(unit["events"] is None for unit in omitted["units"])
    assert all(unit["events"] for unit in recorded["units"])
    for left, right in zip(recorded["units"], omitted["units"], strict=True):
        assert {k: v for k, v in left.items() if k != "events"} == {k: v for k, v in right.items() if k != "events"}
    assert recorded["summary"] == omitted["summary"]


def test_evaluation_with_agents_is_deterministic():
    profile = resolve_profile("paper-resource-v1")
    kwargs = dict(policies=["rolling"], agents={"idle": never}, seed=99, episodes=2, calendar_steps=96)
    assert evaluate(profile, **kwargs) == evaluate(profile, **kwargs)


def test_explicit_seed_fixture_replays_like_an_evaluate_fixture():
    profile = resolve_profile(CORE)
    derived = evaluate(profile, policies=["immediate"], seed=31, episodes=3, calendar_steps=100)
    fixture = make_fixture(profile, derived["fixture"]["seeds"], calendar_steps=100)
    assert fixture == derived["fixture"]
    assert evaluate(profile, policies=["immediate"], seed=0, episodes=1, calendar_steps=100, fixture=fixture) == derived
    with pytest.raises(ValueError, match="Duplicate scenario"):
        make_fixture(profile, [1, 1], calendar_steps=100)
    with pytest.raises(ValueError, match="at least one"):
        make_fixture(profile, [], calendar_steps=100)


# Benchmark scoring and the oracle.

def test_core_oracle_orders_exactly_at_the_last_safe_order_and_is_never_late():
    result = benchmark.score(profile=CORE, seeds=range(700, 712), references=("oracle",))
    units = _units(result.evaluation, "oracle")
    assert len(units) == 12 and {u["policy_kind"] for u in units} == {PRIVILEGED_REFERENCE}
    for unit in units:
        lifetime = unit["analysis_diagnostics"]["eol_operating_age"]
        assert unit["first_trigger_true_rul"] == 13 and unit["order_margin"] == 0
        assert unit["failures"] == 0 and unit["planned_interventions"] == 1
        assert unit["discarded_operating_steps"] == 1
        assert unit["normalized_discarded_life"] == pytest.approx(1 / lifetime)
    row = result.row("oracle")
    assert row["late_orders"] == 0 and row["order_margin_mean"] == 0 and row["order_margin_sd"] == 0
    assert row["percent_of_reference"] == 100 and row["kind"] == PRIVILEGED_REFERENCE
    assert row["information"].startswith("true lifetime")


def test_no_built_in_rule_beats_the_oracle_on_any_core_scenario():
    result = benchmark.score(profile=CORE, seed=2718, episodes=40)
    assert result.reference == "oracle" and result.rows[0]["policy"] == "oracle"
    oracle = result.row("oracle")["pooled_profit_per_1000_calendar_steps"]
    for row in result.rows:
        assert row["percent_of_reference"] == 100 * (row["pooled_profit_per_1000_calendar_steps"] / oracle)
    assert {(c["policy"], c["reference"]) for c in result.comparisons} == {
        (name, "oracle") for name in ("reactive", "age", "health", "projection", "immediate")}
    for comparison in result.comparisons:
        assert comparison["scenarios_better"] == 0
        assert comparison["scenarios_better"] + comparison["scenarios_tied"] + comparison["scenarios_worse"] == 40


def test_a_calendar_budget_that_cuts_core_cycles_is_flagged_because_the_oracle_bound_fails():
    # A budget shorter than a cycle stops waiting policies before their renewal: such a run earns
    # 255 per productive step, above the oracle's complete cycle, so the oracle no longer bounds them.
    cut = benchmark.score(profile=CORE, seed=4, episodes=20, calendar_steps=110, recalibrate=False,
                          bootstrap_samples=50)
    assert cut.comparison("reactive", "oracle")["scenarios_better"] > 0
    runs = [unit for unit in cut.evaluation["units"] if not unit["terminated"]]
    assert runs and all(unit["accounting_cutoff"] for unit in runs)
    note = next(note for note in cut.notes if "upper bound only for complete component cycles" in note)
    assert note.startswith(f"{len(runs):,} of {len(cut.evaluation['units']):,} scenario runs ended at the calendar budget")
    assert note in cut.to_markdown() and note in cut.to_dict()["notes"]
    complete = benchmark.score(profile=CORE, seed=4, episodes=20, calendar_steps=166, recalibrate=False,
                               bootstrap_samples=50)
    assert all(unit["terminated"] for unit in complete.evaluation["units"])
    assert not [note for note in complete.notes if "complete component cycles" in note]
    assert all(c["scenarios_better"] == 0 for c in complete.comparisons)


def test_the_oracle_is_not_reachable_from_the_agent_side(tmp_path, capsys):
    assert all("oracle" not in profile["policies"] for profile in all_profiles().values())
    profile = resolve_profile(CORE)
    env = gym.make(profile["environment_id"], config=profile["env_config"])
    try:
        env.reset(seed=1)
        with pytest.raises(ValueError, match="not supported"):
            Policy("oracle", profile, env)
    finally:
        env.close()
    with pytest.raises(ValueError, match="not supported"):
        evaluate(profile, policies=["oracle"], seed=1, episodes=1, calendar_steps=20)
    with pytest.raises(ValueError, match="names a built-in policy or reference"):
        evaluate(profile, agents={"oracle": never}, seed=1, episodes=1, calendar_steps=20)
    assert main(["evaluate", "--profile", CORE, "--seed", "1", "--policy", "oracle",
                 "--output", str(tmp_path / "out")]) == 2
    assert "not supported" in capsys.readouterr().err and not (tmp_path / "out").exists()
    # Next to the oracle, an agent still sees only the observation and public info.
    recorder = Recorder()
    benchmark.score(recorder, profile=CORE, seeds=[5, 6], references=("oracle",))
    assert recorder.calls
    for (observation, info), _ in recorder.calls:
        assert isinstance(observation, np.ndarray) and observation.shape == (4,)
        assert not {item for item in _walk(info) if isinstance(item, str)} & PRIVILEGED_KEYS
    assert "oracle" not in [name for name in dir(benchmark) if not name.startswith("_") and callable(getattr(benchmark, name))]


@pytest.mark.parametrize("profile_id,primary", [("paper-gearbox-v1", "dp"), ("paper-resource-v1", "rolling"),
                                                 ("paper-resource-lifetime-v1", "rolling")])
def test_tasks_without_an_oracle_name_their_strongest_reference(profile_id, primary):
    with pytest.raises(ValueError, match="No oracle exists"):
        benchmark.score(profile=profile_id, seeds=[1], references=("oracle",))
    result = benchmark.score(never, profile=profile_id, seeds=[3, 4], calendar_steps=96, references=())
    assert result.reference == primary and {row["policy"] for row in result.rows} == {primary, "never"}
    assert result.row(primary)["percent_of_reference"] == 100
    if primary == "rolling":
        assert "strongest feasible reference" in result.reference_role and "no oracle" in result.reference_role
    else:
        assert "exact dynamic programming" in result.reference_role
    assert [(c["policy"], c["reference"]) for c in result.comparisons] == [("never", primary)]


def test_benchmark_compares_each_agent_with_each_reference_on_paired_scenarios():
    result = benchmark.score({"mine": projection_agent, "idle": never}, profile=CORE, seeds=range(40, 52),
                             references=("projection", "reactive"), bootstrap_samples=500)
    assert [entry for entry in result.bootstrap] == ["method", "samples", "seed", "confidence"]
    pairs = [(c["policy"], c["reference"]) for c in result.comparisons]
    assert pairs == [(agent, reference) for agent in ("mine", "idle") for reference in ("oracle", "projection", "reactive")]
    mine_vs_projection = result.comparison("mine", "projection")
    assert mine_vs_projection["difference_per_1000_calendar_steps"] == 0
    assert mine_vs_projection["interval_low"] == mine_vs_projection["interval_high"] == 0
    assert mine_vs_projection["scenarios_tied"] == 12
    idle = result.comparison("idle", "reactive")
    assert idle["scenarios_tied"] == 12 and idle["difference_per_1000_calendar_steps"] == 0
    oracle = result.comparison("mine", "oracle")
    assert oracle["difference_per_1000_calendar_steps"] < 0
    assert oracle["interval_low"] <= oracle["interval_high"] <= 0 and oracle["scenarios_better"] == 0
    assert result.row("mine")["information"] == "observation and info" and result.row("mine")["kind"] == AGENT


def test_benchmark_results_are_deterministic_and_render_one_table():
    first = benchmark.score(projection_agent, profile=CORE, seed=77, episodes=10)
    second = benchmark.score(projection_agent, profile=CORE, seed=77, episodes=10)
    assert first.to_dict() == second.to_dict() and first.evaluation == second.evaluation
    assert first.to_markdown() == second.to_markdown() == str(first)
    assert first.evaluation["unit_events"] == "omitted"
    record = first.to_dict()
    assert record["schema"] == "maintenance_arena_benchmark_v1" and record["evaluation_schema"] == EVALUATION_SCHEMA
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    table = first.to_markdown()
    assert "| Policy | Kind | Information | Profit per 1,000 calendar steps | % of oracle |" in table
    assert table.count("\n| `") == len(first.rows) + len(first.comparisons)
    assert [row["pooled_profit_per_1000_calendar_steps"] for row in first.rows] == sorted(
        (row["pooled_profit_per_1000_calendar_steps"] for row in first.rows), reverse=True)


def test_benchmark_labels_validation_and_modified_profiles():
    assert benchmark.agent_label(projection_agent) == "projection_agent"
    assert benchmark.agent_label(lambda observation, info: 0) == "agent"
    assert benchmark.agent_label(Recorder()) == "Recorder"
    assert benchmark.default_references(CORE) == ("oracle", "reactive", "age", "health", "projection", "immediate")
    assert benchmark.default_references("paper-gearbox-v1")[0] == "dp"
    with pytest.raises(ValueError, match="Unknown reference"):
        benchmark.score(profile=CORE, seeds=[1], references=("imaginary",))
    with pytest.raises(ValueError, match="requires a checkpoint"):
        benchmark.score(profile=CORE, seeds=[1], references=("ppo",))
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.score([("a", never), ("a", never)], profile=CORE, seeds=[1])
    with pytest.raises(ValueError, match="either seeds or fixture"):
        benchmark.score(profile=CORE, seeds=[1], fixture={})
    modified = resolve_profile(CORE, {"env_config": {"failure_event_cost": 8000.0}})
    # Without recalibration the modified profile keeps its policy_defaults and says so; the default
    # (automatic recalibration) is covered in test_calibration.py.
    result = benchmark.score(profile=modified, seeds=[1, 2], references=("projection",), recalibrate=False)
    assert result.modified and result.profile == modified["id"]
    assert any("not recalibrated" in note for note in result.notes)


def _run_example(name, tmp_path, track, *arguments):
    """Run an example script on a tiny custom track; return the completed process."""
    import os
    import subprocess
    from pathlib import Path
    import maintenance_arena
    from maintenance_arena import tracks
    example = Path(__file__).resolve().parents[1] / "examples" / name
    if not example.is_file():
        pytest.skip("examples are not available next to the tests")
    path = tracks.write_tracks(tmp_path / "tiny_tracks.json", [track])
    package_root = Path(maintenance_arena.__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(package_root), os.environ.get("PYTHONPATH")])))
    return subprocess.run([sys.executable, str(example), "--track", str(path), *arguments], capture_output=True,
                          text=True, env=environment, check=False, cwd=tmp_path)


def _tiny_track(name, profile, **values):
    from maintenance_arena import tracks
    return tracks.define_track(name, profile, root_seed=31, **values)


def test_benchmark_example_runs_on_the_base_install(tmp_path):
    track = _tiny_track("tiny-example", CORE, test_scenarios=4, calibration_scenarios=6)
    completed = _run_example("benchmark_your_agent.py", tmp_path, track, "--steps", "0")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("Benchmark track `tiny-example`")
    assert "| `deadline_agent` | agent | observation and info |" in completed.stdout
    assert "| `deadline_agent` | `oracle` |" in completed.stdout and not completed.stderr
    assert "sb3_ppo" not in completed.stdout
    other = tmp_path / "gearbox"
    other.mkdir()
    gearbox = _tiny_track("tiny-gearbox", "paper-gearbox-v1", test_scenarios=2, calibration_scenarios=2)
    rejected = _run_example("benchmark_your_agent.py", other, gearbox, "--steps", "0")
    assert rejected.returncode == 2 and "trains on the core task" in rejected.stderr


@pytest.mark.training
def test_the_sb3_example_trains_and_scores_a_model_on_a_zero_deadline_track(tmp_path):
    pytest.importorskip("stable_baselines3")
    track = _tiny_track("tiny-zero", CORE, test_scenarios=3, calibration_scenarios=4, agent_observation="zero_deadline")
    completed = _run_example("benchmark_your_agent.py", tmp_path, track, "--steps", "256", "--training-seeds", "3")
    assert completed.returncode == 0, completed.stderr
    assert ("| `sb3_ppo_seed3` | agent | observation with the deadline coordinate (index 3) set to zero, and info |"
            in completed.stdout)
    assert "| `sb3_ppo_seed3` | `projection` |" in completed.stdout


@pytest.mark.training
def test_the_maskable_resource_example_trains_and_scores_a_masked_agent(tmp_path):
    pytest.importorskip("sb3_contrib")
    track = _tiny_track("tiny-resource", "paper-resource-v1", test_scenarios=2, calibration_scenarios=2,
                        calendar_steps=48)
    completed = _run_example("train_maskable_resource.py", tmp_path, track, "--steps", "2048")
    assert completed.returncode == 0, completed.stderr
    row = next(line for line in completed.stdout.splitlines() if line.startswith("| `maskable_ppo_seed0` | agent"))
    assert row.endswith("| 0 |")  # the agent passes the action mask, so no masked action is executed
    assert "| `maskable_ppo_seed0` | `rolling` |" in completed.stdout


@pytest.mark.training
def test_a_stable_baselines3_model_is_scored_through_the_agent_interface():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", gym.make("maintenanceArena/CoreCycle-v0"), gamma=0.999, n_steps=512, batch_size=64,
                seed=0, device="cpu")
    model.learn(total_timesteps=2048)
    result = benchmark.score(("sb3_ppo", lambda observation, info: model.predict(observation, deterministic=True)[0]),
                             profile=CORE, seeds=range(4), references=("projection",))
    row = result.row("sb3_ppo")
    assert row["scenarios"] == 4 and row["kind"] == AGENT and row["percent_of_reference"] > 0


# Command line.

AGENT_MODULE = '''
import math


def act(observation, info):
    health, rate, valid, _ = (float(value) for value in observation)
    if valid < 0.5:
        return 0
    remaining = 0.0 if health <= 0 else (1.0 + math.log(health) / math.log(5.0)) / max(rate / 12.0, 1e-12)
    return int(remaining <= 15)


def projection(observation, info):
    return 0
'''


@pytest.fixture
def agent_module(tmp_path, monkeypatch):
    """An agent module in the working directory, as a user would write it."""
    name = f"byo_agents_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(AGENT_MODULE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield name
    sys.modules.pop(name, None)


def test_cli_evaluate_scores_an_importable_agent_next_to_built_in_policies(tmp_path, agent_module, capsys):
    output = tmp_path / "evaluated"
    assert main(["evaluate", "--profile", CORE, "--seed", "21", "--episodes", "3", "--policy", "projection",
                 "--agent", f"{agent_module}:act", "--no-events", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [s["policy"] for s in printed["result"]["summary"]] == ["projection", "act"]
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["schema"] == EVALUATION_SCHEMA and evaluation["unit_events"] == "omitted"
    summaries = {s["policy"]: s for s in evaluation["summary"]}
    assert summaries["act"]["policy_kind"] == AGENT
    assert summaries["act"]["pooled_profit_per_1000_calendar_steps"] == summaries["projection"]["pooled_profit_per_1000_calendar_steps"]
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    [agent] = run["inputs"]["agents"]
    assert agent["label"] == "act" and agent["spec"] == f"{agent_module}:act"
    assert len(agent["sha256"]) == 64 and agent["module_file"].endswith(f"{agent_module}.py")


@pytest.mark.parametrize("arguments,message", [
    (["--agent", "{module}:act", "--agent-label", "a", "--agent-label", "b"], "one --agent-label per --agent"),
    (["--agent", "no_colon"], "must have the form module:callable"),
    (["--agent", "{module}:missing"], "has no attribute"),
    (["--agent", "module_that_does_not_exist_12345:act"], "cannot import module"),
    (["--agent", "{module}:projection"], "names a built-in policy or reference"),
    (["--agent", "{module}:act", "--agent-label", "oracle"], "names a built-in policy or reference"),
    ([], "needs at least one --policy or --agent"),
])
def test_cli_rejects_bad_agent_options_before_creating_the_output(tmp_path, agent_module, capsys, arguments, message):
    output = tmp_path / "rejected"
    command = ["evaluate", "--profile", CORE, "--seed", "1", "--output", str(output)]
    assert main(command + [value.replace("{module}", agent_module) for value in arguments]) == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and "Traceback" not in error
    assert message in error, error
    assert not output.exists()


def test_cli_benchmark_prints_and_writes_the_comparison_table(tmp_path, agent_module, capsys):
    output = tmp_path / "bench"
    assert main(["benchmark", "--profile", CORE, "--seed", "5", "--episodes", "6", "--agent", f"{agent_module}:act",
                 "--agent-label", "deadline_rule", "--reference", "projection", "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    table = (output / "benchmark.md").read_text(encoding="utf-8")
    assert printed == table + f"\nResults: {output.resolve()}\n"
    record = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert record["reference"] == "oracle" and record["scenarios"] == 6
    assert {row["policy"] for row in record["rows"]} == {"oracle", "projection", "deadline_rule"}
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    fixture = json.loads((output / "fixture.json").read_text(encoding="utf-8"))
    assert evaluation["fixture"] == fixture == record["fixture"]
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete" and run["command"] == "benchmark"
    assert run["result"]["reference"] == "oracle"
    replay = tmp_path / "replay"
    assert main(["evaluate", "--profile", CORE, "--seed", "0", "--policy", "projection", "--fixture",
                 str(output / "fixture.json"), "--output", str(replay)]) == 0
    replayed = json.loads((replay / "evaluation.json").read_text(encoding="utf-8"))
    assert replayed["fixture"]["seeds"] == fixture["seeds"]
    projection = next(row for row in record["rows"] if row["policy"] == "projection")
    assert replayed["summary"][0]["pooled_profit_per_1000_calendar_steps"] == projection["pooled_profit_per_1000_calendar_steps"]


@pytest.mark.parametrize("arguments,message", [
    (["--profile", "paper-gearbox-v1", "--reference", "oracle"], "No oracle exists"),
    (["--profile", CORE, "--reference", "imaginary"], "Unknown reference"),
    (["--profile", CORE, "--reference", "ppo"], "requires a checkpoint"),
    (["--profile", CORE, "--episodes", "0"], "episodes must be an integer >= 1"),
    (["--profile", CORE, "--filter"], "execution filter is defined only for the frozen gearbox"),
])
def test_cli_benchmark_validates_references_before_creating_the_output(tmp_path, capsys, arguments, message):
    output = tmp_path / "rejected"
    assert main(["benchmark", "--seed", "1", "--output", str(output)] + arguments) == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and message in error, error
    assert not output.exists()
