"""Reproducible entry points; no implicit training, downloads or overwrites."""
from __future__ import annotations
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import warnings

from .profiles import all_profiles, resolve_profile
from .records import begin_run, finish_run, read_config, write_json, file_hash, json_value
from . import __version__


PATH_INPUTS = ("config", "checkpoint", "fixture", "resume", "initialize", "calibration")
OUTPUT_HELP = "New directory for the run record and results; it must not exist yet, because outputs are never overwritten"
AGENT_HELP = ("User agent as module:callable, imported with the current directory on the import path and called "
              "as agent(observation, info) -> action; it never receives the environment. Repeat for several agents")
AGENT_LABEL_HELP = ("Label of the agent in the results, one per --agent in the same order "
                    "(default: the callable's name)")


class OutputExistsError(FileExistsError):
    """The requested --output already exists; results are never overwritten."""


class _HelpFormatter(argparse.HelpFormatter):
    """Wrap help text without splitting hyphenated profile and policy names."""

    def _split_lines(self, text, width):
        import textwrap
        return textwrap.wrap(" ".join(text.split()), width, break_on_hyphens=False)


def parser():
    from .calibration import CRITERIA, DEFAULT_CRITERION, DEFAULT_LIVES, RULES
    profiles = all_profiles()
    policies = "; ".join(f"{name}: {', '.join(profile['policies'])}" for name, profile in profiles.items())
    root = argparse.ArgumentParser(
        prog="maintenance-arena", formatter_class=_HelpFormatter,
        description="Simulate, evaluate, benchmark and train policies on the MaintenanceArena tasks, and calibrate "
                    "the core rules. Nothing is trained, downloaded or overwritten implicitly.",
        epilog="Run 'maintenance-arena list' for the profiles and the policies each one supports, and "
               "'maintenance-arena tracks' for the benchmark tracks. Every command except list and tracks writes "
               "to a new --output directory.")
    root.add_argument("--version", action="version", version=f"MaintenanceArena {__version__}")
    commands = root.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("list", help="Print the profiles with their environments and policies as JSON",
                        description="Print the profiles with their environments and policies as JSON.",
                        formatter_class=_HelpFormatter)
    summary = ("Print the benchmark tracks as JSON: profile, calendar budget, scenario counts, references, root "
               "seed and hash of each track, without the seed lists")
    commands.add_parser("tracks", help=summary, description=summary + ".", formatter_class=_HelpFormatter)
    summaries = {"simulate": "Run one policy on one scenario and write its decision trace",
                 "evaluate": "Compare policies on identical synthetic scenarios",
                 "train": "Train the reference PPO agent of a profile (needs the rl extra)"}
    seeds = {"simulate": "Nonnegative seed of the scenario",
             "evaluate": ("Nonnegative root seed from which the scenario seeds are derived. Not needed with "
                          "--fixture, whose seeds define the scenarios"),
             "train": "Nonnegative seed of the training run"}
    for name in ("simulate", "evaluate", "train"):
        command = commands.add_parser(name, help=summaries[name], description=summaries[name] + ".",
                                      formatter_class=_HelpFormatter)
        command.add_argument("--profile", required=True, choices=list(profiles), metavar="PROFILE",
                             help=f"Versioned task profile: {', '.join(profiles)}; "
                                  "'maintenance-arena list' shows the details")
        command.add_argument("--seed", required=name != "evaluate", type=int, help=seeds[name])
        command.add_argument("--output", required=True, type=Path, help=OUTPUT_HELP)
        command.add_argument("--config", type=Path,
                             help="YAML/JSON file with profile overrides in the sections env_config, training and "
                                  "policy_defaults; the run then has a modified profile identity")
        command.add_argument("--calendar-steps", type=int, required=name == "train", default=None if name == "train" else 720,
                             help="Cumulative training budget in calendar steps; resource profiles need a multiple of "
                                  "n_envs * n_steps (8 * 256 = 2048 by default)" if name == "train" else
                                  "Calendar-step budget of each scenario (default: 720); a core episode can end "
                                  "earlier, at its renewal")
        if name == "train":
            command.add_argument("--resume", type=Path,
                                 help="Native checkpoint (a file or a training output directory) to continue exactly "
                                      "at an update boundary; use the same --seed, and --calendar-steps is the new "
                                      "cumulative total")
            command.add_argument("--initialize", type=Path,
                                 help="Compatible checkpoint whose weights start a new run with a fresh optimizer "
                                      "and fresh random streams")
        else:
            command.add_argument("--checkpoint", type=Path,
                                 help="Trained model for --policy ppo: a training output directory, its final.pt or "
                                      "checkpoint.json, or a model manifest")
            command.add_argument("--filter", action="store_true",
                                 help="Apply the exact gearbox execution filter (gearbox profile only)")
        if name == "simulate":
            command.add_argument("--policy", required=True, help=f"Policy to run. Valid names per profile: {policies}")
            command.add_argument("--plot", action="store_true", help="Also plot the trajectory (needs the plots extra)")
        if name == "evaluate":
            command.add_argument("--policy", nargs="+",
                                 help=f"One or more built-in policies, compared on identical scenario seeds; "
                                      f"--policy, --agent or both are required. Valid names per profile: {policies}")
            command.add_argument("--agent", action="append", metavar="MODULE:CALLABLE", help=AGENT_HELP)
            command.add_argument("--agent-label", action="append", metavar="LABEL", help=AGENT_LABEL_HELP)
            command.add_argument("--plot", action="store_true",
                                 help="Also plot the paired timing and economics comparison (core profiles; "
                                      "needs the plots extra)")
            command.add_argument("--episodes", type=int, default=32,
                                 help="Number of synthetic scenarios (default: 32); ignored when --fixture is given")
            command.add_argument("--fixture", type=Path,
                                 help="fixture.json of an earlier evaluate run; replays its scenario seeds "
                                      "(same environment configuration and --calendar-steps)")
            command.add_argument("--no-events", action="store_true",
                                 help="Omit the per-scenario event lists from evaluation.json, which keeps runs "
                                      "with many scenarios small")
    summary = ("Score agents against the task's references on identical scenarios and print the comparison "
               "table (oracle for the core, dp for the gearbox, rolling for the resource tasks), on a fixed "
               "benchmark track or on scenarios derived from --seed")
    bench = commands.add_parser("benchmark", help=summary, description=summary + ".", formatter_class=_HelpFormatter)
    bench.add_argument("--track", metavar="TRACK",
                       help="Benchmark track: a name from 'maintenance-arena tracks' or a tracks JSON file in the "
                            "same format. The track fixes the profile, the test and calibration scenarios, the "
                            "calendar budget, the references and the rule calibration, so it replaces --profile, "
                            "--seed, --config, --calendar-steps, --episodes, --fixture, --reference, --checkpoint, "
                            "--filter and the calibration options")
    bench.add_argument("--profile", choices=list(profiles), metavar="PROFILE",
                       help=f"Versioned task profile: {', '.join(profiles)}; 'maintenance-arena list' shows the "
                            "details. Required unless --track is given")
    bench.add_argument("--seed", type=int,
                       help="Nonnegative root seed; the scenario seeds are derived from it unless --fixture is given. "
                            "Required unless --track is given")
    bench.add_argument("--output", required=True, type=Path, help=OUTPUT_HELP)
    bench.add_argument("--config", type=Path,
                       help="YAML/JSON file with profile overrides in the sections env_config, training and "
                            "policy_defaults; the run then has a modified profile identity")
    bench.add_argument("--calendar-steps", type=int,
                       help="Calendar-step budget of each scenario (default: 720); a core episode can end earlier, "
                            "at its renewal")
    bench.add_argument("--episodes", type=int,
                       help="Number of synthetic scenarios (default: 100); ignored when --fixture is given")
    bench.add_argument("--fixture", type=Path,
                       help="fixture.json of an earlier evaluate or benchmark run; replays its scenario seeds")
    bench.add_argument("--agent", action="append", metavar="MODULE:CALLABLE", help=AGENT_HELP)
    bench.add_argument("--agent-label", action="append", metavar="LABEL", help=AGENT_LABEL_HELP)
    bench.add_argument("--reference", nargs="+",
                       help="References to compare against (default: the primary reference and every built-in "
                            "policy except ppo); the primary reference is always included. Valid names: oracle "
                            f"(core profiles only) and the profile's policies: {policies}")
    bench.add_argument("--checkpoint", type=Path,
                       help="Trained model for the reference ppo: a training output directory, its final.pt or "
                            "checkpoint.json, or a model manifest")
    bench.add_argument("--filter", action="store_true",
                       help="Apply the exact gearbox execution filter to every policy (gearbox profile only)")
    bench.add_argument("--calibration", type=Path,
                       help="calibration.json of a calibrate run for the same configuration and --calendar-steps; "
                            "its thresholds replace those of the rules age, health and projection (core profiles)")
    bench.add_argument("--no-recalibrate", action="store_true",
                       help="Keep the profile's policy_defaults even when --config changes the environment; by "
                            "default the core rules are then recalibrated on lives disjoint from the test scenarios")
    bench.add_argument("--calibration-lives", type=int,
                       help=f"Number of calibration lives when the core rules are recalibrated (default: "
                            f"{DEFAULT_LIVES}); they are derived from --seed on their own stream")
    bench.add_argument("--criterion", choices=CRITERIA,
                       help=f"Selection criterion when the core rules are calibrated, on a core track or for a "
                            f"changed --config: {' or '.join(CRITERIA)} (default: {DEFAULT_CRITERION}); see "
                            "'maintenance-arena calibrate --help'. With --calibration it must be the criterion of the "
                            "record; --no-recalibrate excludes it")
    core_profiles = [name for name, profile in profiles.items() if "CoreCycle" in profile["environment_id"]]
    summary = ("Select the thresholds of the core rules age, health and projection by grid search on calibration "
               "lives and write the full selection table")
    calib = commands.add_parser("calibrate", help=summary, description=summary + ".", formatter_class=_HelpFormatter)
    calib.add_argument("--profile", required=True, choices=list(profiles), metavar="PROFILE",
                       help=f"Core task profile: {', '.join(core_profiles)}; use --config to calibrate for a changed task")
    calib.add_argument("--seed", required=True, type=int,
                       help="Nonnegative root seed of the calibration lives. They come from their own stream, "
                            "SeedSequence(seed, spawn_key=(1, i)), so they differ from the scenarios that evaluate "
                            "and benchmark derive from the same seed")
    calib.add_argument("--output", required=True, type=Path, help=OUTPUT_HELP)
    calib.add_argument("--config", type=Path,
                       help="YAML/JSON file with profile overrides; the rules are calibrated for its env_config")
    calib.add_argument("--lives", type=int, default=DEFAULT_LIVES,
                       help=f"Number of calibration lives (default: {DEFAULT_LIVES})")
    calib.add_argument("--calendar-steps", type=int, default=720,
                       help="Calendar-step budget of each life (default: 720), as in evaluate and benchmark")
    calib.add_argument("--rule", nargs="+", choices=RULES,
                       help=f"Rules to calibrate (default: {' '.join(RULES)})")
    calib.add_argument("--criterion", choices=CRITERIA, default=DEFAULT_CRITERION,
                       help="Selection criterion. zero_failure (default): "
                            "only thresholds without a failure (and with a planned exchange) on the calibration lives "
                            "are eligible, and the one with the highest family-equal profit wins, the mean of the "
                            "pooled profits of the wear families with equal weight per family; the command stops with "
                            "an error when a rule has no eligible threshold. pooled: the highest pooled profit over all "
                            "lives, failures allowed")
    for rule, default in (("age", "1:159"), ("health", "0.20:0.95:0.01"), ("projection", "8:24")):
        calib.add_argument(f"--{rule}-grid", nargs="+", metavar="VALUES",
                           help=f"Candidate {rule} thresholds as numbers and inclusive ranges START:STOP[:STEP] "
                                f"(default: {default})")
    calib.add_argument("--exclude-fixture", action="append", type=Path, metavar="FIXTURE",
                       help="fixture.json of an evaluate or benchmark run; its scenario seeds are kept out of the "
                            "calibration lives. Repeat for several test sets")
    return root


def _require(module):
    if importlib.util.find_spec(module) is None:
        raise ModuleNotFoundError(f"No module named {module!r}", name=module)


def _output_exists(path):
    return OutputExistsError(f"output directory already exists: {path}. "
                             "Outputs are never overwritten; choose a new --output.")


def _check_paths(args):
    for name in PATH_INPUTS:
        value = getattr(args, name, None)
        if value is not None and not value.exists():
            raise ValueError(f"--{name} path does not exist: {value}")
    for value in getattr(args, "exclude_fixture", None) or ():
        if not value.is_file():
            raise ValueError(f"--exclude-fixture file does not exist: {value}")


def _fixture_seeds(path):
    """Scenario seeds of a fixture.json written by evaluate or benchmark."""
    from .evaluation import FIXTURE_SCHEMA, _integer
    fixture = read_config(path)
    seeds = fixture.get("seeds")
    if fixture.get("schema") != FIXTURE_SCHEMA or not isinstance(seeds, list):
        raise ValueError(f"--exclude-fixture {path} is not a fixture.json of evaluate or benchmark "
                         f"(schema {FIXTURE_SCHEMA})")
    return {_integer(seed, "fixture seed") for seed in seeds}


def load_agent(spec):
    """Import an agent given as 'module:callable'; return (agent, default label).

    The current directory is appended to the import path, so a module next to the
    command works with both the console script and ``python -m maintenance_arena``.
    """
    module_name, separator, attribute = str(spec).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"--agent must have the form module:callable, for example my_agents:act; got {spec!r}")
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    try:
        target = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"--agent {spec}: cannot import module {module_name!r}: {error}") from error
    for part in attribute.split("."):
        if not hasattr(target, part):
            raise ValueError(f"--agent {spec}: module {module_name!r} has no attribute {attribute!r}")
        target = getattr(target, part)
    if not callable(target):
        raise ValueError(f"--agent {spec}: {attribute!r} is not callable")
    from .benchmark import agent_label
    return target, agent_label(target, fallback=attribute.split(".")[-1])


def _load_agents(args):
    """Return [(label, agent, record)] for every --agent, validated before any output exists."""
    specs = list(getattr(args, "agent", None) or ())
    labels = list(getattr(args, "agent_label", None) or ())
    if labels and len(labels) != len(specs):
        raise ValueError("Give one --agent-label per --agent, in the same order")
    agents = []
    for index, spec in enumerate(specs):
        agent, default = load_agent(spec)
        module = sys.modules.get(spec.partition(":")[0])
        source = getattr(module, "__file__", None)
        record = {"spec": spec, "module_file": source,
                  "sha256": file_hash(source) if source and Path(source).is_file() else None}
        label = labels[index] if labels else default
        agents.append((label, agent, dict(record, label=label)))
    return agents


def _benchmark_track(args):
    """Resolve --track and fill in the track's profile, seed and budget, or complete the
    defaults of a benchmark on --profile and --seed. Return the track or None."""
    from .calibration import DEFAULT_LIVES
    if args.track is None:
        missing = [option for option, value in (("--profile", args.profile), ("--seed", args.seed)) if value is None]
        if missing:
            raise ValueError(f"benchmark needs --track, or --profile and --seed; missing {' and '.join(missing)}")
        for name, default in (("calendar_steps", 720), ("episodes", 100), ("calibration_lives", DEFAULT_LIVES)):
            if getattr(args, name) is None:
                setattr(args, name, default)
        return None
    fixed = [option for option, value in (
        ("--profile", args.profile), ("--seed", args.seed), ("--config", args.config),
        ("--calendar-steps", args.calendar_steps), ("--episodes", args.episodes), ("--fixture", args.fixture),
        ("--reference", args.reference), ("--checkpoint", args.checkpoint), ("--filter", args.filter),
        ("--calibration", args.calibration), ("--no-recalibrate", args.no_recalibrate),
        ("--calibration-lives", args.calibration_lives)) if value is not None and value is not False]
    if fixed:
        raise ValueError("--track fixes the profile, scenarios, calendar budget, references and rule calibration; "
                         f"remove {', '.join(fixed)}")
    from .tracks import load_track
    track = load_track(args.track)
    args.profile, args.seed, args.calendar_steps = track.profile, track.root_seed, track.calendar_steps
    args.episodes, args.calibration_lives = track.test_scenarios, track.calibration_scenarios
    return track


def _tracks_listing():
    from .tracks import DEFAULT_TRACK, load_tracks
    return {"default_track": DEFAULT_TRACK,
            "tracks": {name: track.summary() for name, track in load_tracks().items()}}


def _check_command(args, profile, track=None):
    """Validate a run before its output directory exists.

    Return the trainer for train and the loaded agents for evaluate and benchmark.
    A created run directory is never reused. Checking here lets a user correct a
    mistyped argument and repeat the command with the same --output.
    """
    if getattr(args, "plot", False):
        _require("matplotlib")
    if profile is None:
        return None
    resource = "Resource" in profile["environment_id"]
    if args.command == "train" and args.resume is not None and args.initialize is not None:
        raise ValueError("--resume and --initialize are mutually exclusive")
    for name in ("checkpoint", "resume", "initialize"):
        value = getattr(args, name, None)
        expected = "checkpoint.json" if resource else "final.pt"
        if value is not None and value.is_dir() and not (value / expected).is_file():
            raise ValueError(f"--{name} directory contains no {expected}: {value}")
    from .evaluation import make_resolved
    if args.command == "calibrate":
        return _check_calibrate(args, profile)
    if args.command == "train":
        if resource:
            from .learning.resource_ppo import check_calendar_budget, train
            check_calendar_budget(args.calendar_steps, profile["training"])
        else:
            from .learning.ppo import train
        make_resolved(profile).close()
        return train
    if args.command == "evaluate" and args.seed is None and args.fixture is None:
        raise ValueError("evaluate needs --seed, or --fixture to replay the scenarios of an earlier run")
    agents = _load_agents(args) if args.command in ("evaluate", "benchmark") else []
    pairs = [(label, agent) for label, agent, _ in agents]
    if args.command == "benchmark":
        from .benchmark import benchmark_entries
        references = args.reference if track is None else list(track.references)
        names = [entry.label for entry in benchmark_entries(profile, pairs or None, references=references,
                                                            checkpoint=args.checkpoint)]
    else:
        names = [args.policy] if args.command == "simulate" else list(args.policy or ())
        if args.command == "evaluate" and not names and not agents:
            raise ValueError("evaluate needs at least one --policy or --agent")
        for name in names:
            if name not in profile["policies"]:
                raise ValueError(f"Policy {name!r} is not supported by {profile['base_profile']}; "
                                 f"choose from {', '.join(profile['policies'])}")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate policies would repeat observations and inflate scenario counts")
        if agents:
            from .evaluation import _entries
            _entries(profile, names, pairs, checkpoint=args.checkpoint)
    if "ppo" in names:
        if args.checkpoint is None:
            raise ValueError("PPO evaluation requires --checkpoint; no model is downloaded or trained implicitly")
        _require("torch")
        if resource:
            _require("sb3_contrib")
    if args.filter and "Gearbox" not in profile["environment_id"]:
        raise ValueError("The execution filter is defined only for the frozen gearbox contract")
    if args.command in ("evaluate", "benchmark"):
        if args.episodes < 1:
            raise ValueError("episodes must be an integer >= 1")
        if getattr(args, "plot", False) and not profile["base_profile"].startswith("paper-core-"):
            raise ValueError("The timing/life comparison plot is defined for core profiles")
    if args.command == "benchmark" and track is None:
        _check_benchmark_calibration(args, profile)
    make_resolved(profile, args.calendar_steps).close()
    return agents


def _check_benchmark_calibration(args, profile):
    if args.calibration is not None and args.no_recalibrate:
        raise ValueError("--calibration and --no-recalibrate exclude each other")
    if args.criterion is not None and args.no_recalibrate:
        raise ValueError("--criterion and --no-recalibrate exclude each other: --no-recalibrate keeps the profile's "
                         "policy_defaults")
    if args.calibration_lives < 1:
        raise ValueError("calibration lives must be an integer >= 1")
    if args.calibration is not None:
        from .benchmark import check_calibration
        from .calibration import load_calibration
        from .evaluation import _scenario_seeds
        fixture = read_config(args.fixture) if args.fixture is not None else None
        _, seeds = _scenario_seeds(profile, seed=args.seed, episodes=args.episodes,
                                   calendar_steps=args.calendar_steps, fixture=fixture)
        check_calibration(profile, load_calibration(args.calibration), seeds, args.calendar_steps, args.criterion)


def _check_calibrate(args, profile):
    """Validate a calibrate run; return its rules, grids and excluded seeds."""
    from .calibration import RULES, _resolve, _rules, parse_grid
    from .evaluation import make_resolved
    _resolve(profile)
    rules = _rules(args.rule or RULES)
    grids = {}
    for rule in RULES:
        tokens = getattr(args, f"{rule}_grid")
        if tokens:
            if rule not in rules:
                raise ValueError(f"--{rule}-grid was given, but {rule} is not among the calibrated --rule")
            grids[rule] = parse_grid(rule, tokens)
    if args.lives < 1:
        raise ValueError("lives must be an integer >= 1")
    excluded = set()
    for path in args.exclude_fixture or ():
        excluded |= _fixture_seeds(path)
    make_resolved(profile, args.calendar_steps).close()
    return {"rules": rules, "grids": grids or None, "exclude": sorted(excluded)}


def run(args):
    if args.command == "list":
        from .observations import ROUTE_IDS, SENSOR_LEVELS
        data = {k: {"environment": v["environment_id"], "policies": v["policies"], "evidence_scope": v["evidence_scope"]} for k, v in all_profiles().items()}
        data["core_observation_routes"] = list(ROUTE_IDS)
        data["core_sensor_levels"] = list(SENSOR_LEVELS)
        print(json.dumps(data, indent=2))
        return 0
    if args.command == "tracks":
        print(json.dumps(_tracks_listing(), indent=2))
        return 0
    track = _benchmark_track(args) if args.command == "benchmark" else None
    seed = getattr(args, "seed", None)
    if (seed is not None and seed < 0) or getattr(args, "calendar_steps", 1) < 1:
        raise ValueError("Seed must be nonnegative and calendar budget positive")
    _check_paths(args)
    overrides = read_config(args.config) if getattr(args, "config", None) else None
    profile = resolve_profile(args.profile, overrides) if hasattr(args, "profile") else None
    if track is not None:
        profile = track.resolved_profile()
    prepared = _check_command(args, profile, track)
    trainer = prepared if args.command == "train" else None
    agents = prepared if args.command in ("evaluate", "benchmark") else []
    fixture = read_config(args.fixture) if getattr(args, "fixture", None) else None
    inputs = {}
    for name in PATH_INPUTS:
        value = getattr(args, name, None)
        if value is not None:
            inputs[name] = {"supplied_path": str(value), "sha256": file_hash(value) if value.is_file() else None}
    if agents:
        inputs["agents"] = [record for _, _, record in agents]
    if track is not None:
        inputs["track"] = {"name": track.name, "track_sha256": track.sha256}
        if Path(args.track).suffix.lower() == ".json":
            inputs["track"].update(supplied_path=str(args.track), sha256=file_hash(args.track))
        else:
            inputs["track"]["source"] = "packaged benchmark_tracks.json"
    if getattr(args, "exclude_fixture", None):
        inputs["exclude_fixtures"] = [{"supplied_path": str(path), "sha256": file_hash(path)}
                                      for path in args.exclude_fixture]
    if args.output.exists():
        raise _output_exists(args.output)
    try:
        directory, record = begin_run(args.output, command=args.command, profile=profile,
                                     seed=getattr(args, "seed", None), inputs=inputs,
                                     nominal_calendar_steps=getattr(args, "calendar_steps", None), arguments=vars(args))
    except FileExistsError as error:
        raise _output_exists(args.output) from error
    started = perf_counter()
    try:
        if args.command == "simulate":
            from .evaluation import rollout, write_trace
            result = rollout(profile, policy=args.policy, seed=args.seed, calendar_steps=args.calendar_steps,
                             checkpoint=args.checkpoint, apply_filter=args.filter, trace=True)
            write_trace(directory, result)
            if args.plot:
                from .plotting import plot_trajectory
                plot_trajectory(result, directory)
            result = {k: v for k, v in result.items() if k not in ("trace", "events", "analysis_diagnostics")}
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(profile, policies=list(args.policy or ()),
                              agents=[(label, agent) for label, agent, _ in agents] or None,
                              seed=args.seed, episodes=args.episodes,
                              calendar_steps=args.calendar_steps, checkpoint=args.checkpoint,
                              apply_filter=args.filter, fixture=fixture, events=not args.no_events)
            write_json(directory / "evaluation.json", result)
            write_json(directory / "fixture.json", result["fixture"])
            if args.plot:
                from .plotting import plot_comparison
                plot_comparison(result, directory)
            result = {"summary": result["summary"], "scope": result["scope"]}
        elif args.command == "benchmark":
            from .benchmark import run_track, score
            from .calibration import CalibrationWarning, load_calibration
            pairs = [(label, agent) for label, agent, _ in agents] or None
            with warnings.catch_warnings():
                # Warnings of a recalibration are listed in the table notes.
                warnings.simplefilter("ignore", CalibrationWarning)
                if track is not None:
                    outcome = run_track(track, pairs, criterion=args.criterion)
                else:
                    outcome = score(pairs, profile=profile,
                                    seed=args.seed, episodes=args.episodes, calendar_steps=args.calendar_steps,
                                    references=args.reference, fixture=fixture, checkpoint=args.checkpoint,
                                    apply_filter=args.filter, recalibrate=False if args.no_recalibrate else None,
                                    calibration=load_calibration(args.calibration) if args.calibration else None,
                                    calibration_lives=args.calibration_lives, criterion=args.criterion)
            table = outcome.to_markdown()
            summary = outcome.to_dict()
            write_json(directory / "benchmark.json", summary)
            (directory / "benchmark.md").write_text(table, encoding="utf-8", newline="\n")
            write_json(directory / "evaluation.json", outcome.evaluation)
            write_json(directory / "fixture.json", outcome.evaluation["fixture"])
            if outcome.calibration is not None:
                write_json(directory / "calibration.json", outcome.calibration)
            result = {"reference": outcome.reference, "rows": list(outcome.rows),
                      "comparisons": list(outcome.comparisons), "calibration": summary["calibration"],
                      "track": summary["track"]}
        elif args.command == "calibrate":
            from .calibration import CalibrationWarning, calibrate
            with warnings.catch_warnings():
                # Calibration warnings are printed below and recorded in calibration.json.
                warnings.simplefilter("ignore", CalibrationWarning)
                outcome = calibrate(profile, rules=prepared["rules"], grids=prepared["grids"],
                                    criterion=args.criterion, seed=args.seed, lives=args.lives,
                                    exclude_seeds=prepared["exclude"], calendar_steps=args.calendar_steps)
            record_data = outcome.to_dict()
            table = outcome.to_markdown()
            write_json(directory / "calibration.json", record_data)
            (directory / "calibration.md").write_text(table, encoding="utf-8", newline="\n")
            calibrated_overrides = outcome.overrides()
            if overrides and "training" in overrides:
                calibrated_overrides["training"] = overrides["training"]
            write_json(directory / "overrides.json", calibrated_overrides)
            result = {"policy_defaults": outcome.policy_defaults(), "warnings": list(outcome.warnings),
                      "lives": len(outcome.seeds), "sha256": record_data["sha256"],
                      "selection_criterion": outcome.criterion}
        else:
            from .evaluation import make_resolved
            result = trainer(lambda: make_resolved(profile).unwrapped, profile=profile["base_profile"],
                             seed=args.seed, calendar_steps=args.calendar_steps, output=directory,
                             resume=args.resume, initialize=args.initialize, config=profile["training"])
        finish_run(directory, record, elapsed_seconds=perf_counter() - started, result=result)
        if args.command in ("benchmark", "calibrate"):
            print(table + f"\nResults: {directory}")
            if args.command == "calibrate":
                for text in outcome.warnings:
                    print(f"maintenance-arena: warning: {text}", file=sys.stderr)
        else:
            print(json.dumps(json_value({"output": str(directory), "result": result}), indent=2, allow_nan=False))
        return 0
    except BaseException as error:
        finish_run(directory, record, elapsed_seconds=perf_counter() - started, error=error)
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("maintenance-arena: interrupted; run records and completed checkpoints are retained.", file=sys.stderr)
        return 130
    except ModuleNotFoundError as error:
        optional = "plots" if error.name and "matplotlib" in error.name else "rl"
        print(f"Missing optional dependency {error.name!r}; install maintenance-arena[{optional}].", file=sys.stderr)
        return 2
    except FileExistsError as error:
        # A run directory is never reused, so a repeated --output is a user error.
        print(f"maintenance-arena: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as error:
        print(f"maintenance-arena: {error}", file=sys.stderr)
        return 2
