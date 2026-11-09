"""Score agents against the reference policies of a task on identical scenarios.

``score`` runs one or more user agents together with the task's references on the
same sampled scenarios and returns a :class:`BenchmarkResult`. It reports pooled
calendar profit (cash per 1,000 calendar steps), its percentage of the task's
primary reference, failures per 1,000 ended component lives, the core order
margin, and paired differences with bootstrap intervals over scenarios.

An agent is a callable ``agent(observation, info) -> action``. The harness calls
it with copies of the observation and of the public ``info`` mapping only; it
never passes the environment, ``diagnostics()`` or ``snapshot()``.

Primary reference of each task:

* Core (``CoreCycle-v0`` profiles): the oracle. It knows each component's true
  lifetime and orders exactly at the last safe order, true RUL 13 (the 12-step
  lead time plus the exchange step). It is an upper bound for complete component
  cycles, not a feasible policy; a calendar budget that cuts cycles is noted.
  It exists only here, on the evaluator side: it reads the lifetime through the
  environment's ``diagnostics()``, it is not a profile policy, and no agent can
  reach it through observations, ``info`` or the agent call.
* Gearbox: ``dp``, exact dynamic programming on the exact model, the optimum of
  long-run average profit. It does not optimize the profit of a finite window, and
  over such a window a rule can score slightly above it.
* Resource tasks: ``rolling``, the 48-hour rolling-horizon planner, the strongest
  feasible reference. No oracle exists for these tasks.

Rule thresholds are the profile's ``policy_defaults`` unless the core rules
``age``, ``health`` and ``projection`` are recalibrated. When the environment
configuration differs from the base profile, ``score`` recalibrates them by
default with ``calibration.calibrate`` on calibration lives disjoint from the test
scenarios; a rule whose threshold the configuration sets explicitly keeps it.
``recalibrate=True`` or ``False`` forces the choice, and ``calibration=`` applies a
stored calibration of the same configuration. ``criterion=`` chooses the selection
criterion of a recalibration (``calibration.CRITERIA``; default ``zero_failure``:
no failure on the calibration lives, then the highest family-equal profit).

``run_track`` scores agents on a named track of ``maintenance_arena.tracks``: the
track's fixed test scenarios, its references, the core rules calibrated on its own
calibration scenarios, and the observation the track gives to agents.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from . import evaluation
from .calibration import (DEFAULT_CRITERION, DEFAULT_LIVES, PARAMETERS, RULES, CalibrationResult, _criterion, _value,
                          _with_thresholds, calibrate)
from .core import REPLACE, WAIT
from .policies import Policy
from .profiles import resolve_profile
from .records import canonical_hash
from .tracks import AGENT_OBSERVATIONS, load_track
from .wrappers import zero_deadline


SCHEMA = "maintenance_arena_benchmark_v1"
DEFAULT_PROFILE = "paper-core-continuous-v1"
ORACLE = "oracle"
PRIMARY_REFERENCES = {"core": ORACLE, "gearbox": "dp", "resource": "rolling"}
REFERENCE_ROLES = {
    "core": ("a privileged oracle that knows each component's true lifetime and orders at the last safe "
             "order (true RUL 13); an upper bound, not a feasible policy"),
    "gearbox": ("exact dynamic programming, the optimum of long-run average profit on the exact model; it does not "
                "optimize the profit of a finite window, over which a rule can score slightly above it"),
    "resource": ("the 48-hour rolling-horizon planner, the strongest feasible reference; no oracle exists "
                 "for this task"),
}
# What each reference reads to decide. Agents receive the observation and info.
INFORMATION = {
    "core": {ORACLE: "true lifetime (evaluator side)", "reactive": "none", "immediate": "none",
             "age": "operating age", "health": "observation", "projection": "observation",
             "ppo": "observation"},
    "gearbox": {"dp": "exact state", "reactive_normal": "none", "reactive_derate": "none",
                "normal_q96": "exact state", "derate_q89": "exact state", "separable_q96": "exact state",
                "ppo": "observation"},
    "resource": {"rolling": "observable state and mask", "reactive": "observable state and mask",
                 "age": "observable state and mask", "health": "observable state and mask",
                 "ppo": "observation and mask"},
}
AGENT_INFORMATION = "observation and info"
# How the table notes name the selection criterion of a rule calibration.
CRITERION_PHRASES = {"zero_failure": "the zero-failure criterion (no failure, then the highest family-equal profit)",
                     "pooled": "pooled profit"}
_KIND_TEXT = {evaluation.BUILT_IN: "built-in", evaluation.AGENT: "agent",
              evaluation.PRIVILEGED_REFERENCE: "privileged reference"}


class _CoreOracle:
    """Evaluator-side core oracle: REPLACE once the true RUL reaches the last safe order.

    Built by the evaluator with the environment, after the scenario's reset. It reads
    the sampled lifetime once through ``diagnostics()`` and the counted operating age
    through ``policy_state()``; neither is ever handed to an agent.
    """

    def __init__(self, env):
        self._env = env.unwrapped
        self._lifetime = int(self._env.diagnostics()["eol_operating_age"])
        self._last_safe_order = int(self._env.config["mobilization_lead"]) + 1

    def __call__(self, observation, info):
        age = int(self._env.policy_state()["operating_age"])
        return REPLACE if self._lifetime - age <= self._last_safe_order else WAIT

    def close(self):
        pass


def _resolve(profile):
    if isinstance(profile, str):
        return resolve_profile(profile)
    if isinstance(profile, Mapping) and "environment_id" in profile and "base_profile" in profile:
        return profile
    raise ValueError("profile must be a profile ID or a profile resolved with maintenance_arena.profiles.resolve_profile")


def task_of(profile) -> str:
    """Return 'core', 'gearbox' or 'resource' for a profile ID or resolved profile."""
    environment = _resolve(profile)["environment_id"]
    for marker, task in (("CoreCycle", "core"), ("Gearbox", "gearbox"), ("Resource", "resource")):
        if marker in environment:
            return task
    raise ValueError(f"No benchmark references are defined for {environment}")


def default_references(profile) -> tuple[str, ...]:
    """The primary reference followed by every built-in policy of the profile except ``ppo``."""
    resolved = _resolve(profile)
    primary = PRIMARY_REFERENCES[task_of(resolved)]
    return (primary, *(name for name in resolved["policies"] if name not in ("ppo", primary)))


def agent_label(agent, fallback="agent") -> str:
    """Label of an agent: its ``__name__``, or its class name for a callable object."""
    name = getattr(agent, "__name__", None)
    if not isinstance(name, str):
        name = type(agent).__name__
    return name if name.isidentifier() else fallback


def _agent_pairs(agent, label):
    if agent is None:
        if label is not None:
            raise ValueError("label names an agent, but no agent was given")
        return []
    if callable(agent) and not isinstance(agent, (tuple, list, Mapping)):
        return [(agent_label(agent) if label is None else label, agent)]
    if label is not None:
        raise ValueError("label applies to a single callable agent; give several agents as (label, agent) tuples or a mapping")
    if isinstance(agent, tuple) and len(agent) == 2 and isinstance(agent[0], str):
        return [agent]
    return evaluation._agent_items(agent)


def _reference_entries(profile, references, checkpoint):
    task = task_of(profile)
    primary = PRIMARY_REFERENCES[task]
    if references is None:
        names = default_references(profile)
    elif isinstance(references, (str, bytes)) or not isinstance(references, (list, tuple)):
        raise ValueError("references must be a sequence of reference names")
    else:
        names = tuple(references)
    if primary not in names:
        names = (primary, *names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate references would repeat observations and inflate scenario counts")
    valid = ((ORACLE,) if task == "core" else ()) + tuple(profile["policies"])
    entries = []
    for name in names:
        if name == ORACLE and task == "core":
            entries.append(evaluation._Entry(ORACLE, evaluation.PRIVILEGED_REFERENCE, _CoreOracle))
        elif name == ORACLE:
            raise ValueError(f"No oracle exists for {profile['base_profile']}; its primary reference is "
                             f"{primary!r} ({REFERENCE_ROLES[task]})")
        elif name in profile["policies"]:
            if name == "ppo" and checkpoint is None:
                raise ValueError("The ppo reference requires a checkpoint; no model is downloaded or trained implicitly")
            entries.append(evaluation._built_in_entry(profile, name, checkpoint))
        else:
            raise ValueError(f"Unknown reference {name!r} for {profile['base_profile']}; choose from {', '.join(valid)}")
    return entries


def benchmark_entries(profile, agent=None, *, references=None, label=None, checkpoint=None):
    """Validate references and agents and return the evaluation entries (no scenario is run)."""
    resolved = _resolve(profile)
    entries = _reference_entries(resolved, references, checkpoint)
    entries.extend(evaluation._agent_entry(resolved, pair) for pair in _agent_pairs(agent, label))
    labels = [entry.label for entry in entries]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate agent labels would repeat observations and inflate scenario counts")
    return entries


@dataclass(frozen=True)
class BenchmarkResult:
    """Table-like benchmark result; ``rows`` are sorted by pooled profit, best first."""

    profile: str
    base_profile: str
    configuration_sha256: str
    modified: bool
    environment_id: str
    task: str
    reference: str
    reference_role: str
    calendar_steps: int
    scenario_seeds: tuple[int, ...]
    rows: tuple[dict, ...]
    comparisons: tuple[dict, ...]
    bootstrap: dict
    notes: tuple[str, ...]
    evaluation: dict = field(repr=False)
    # The applied rule calibration (``CalibrationResult.to_dict()``) and how it was obtained:
    # "automatic", "requested", "supplied" or "track"; None when the rules keep their policy_defaults.
    calibration: dict | None = field(default=None, repr=False)
    calibration_origin: str | None = None
    # ``Track.summary()`` of the benchmark track (``run_track``); None for ``score``.
    track: dict | None = field(default=None, repr=False)

    def row(self, policy: str) -> dict:
        for row in self.rows:
            if row["policy"] == policy:
                return row
        raise KeyError(policy)

    def comparison(self, policy: str, reference: str) -> dict:
        for row in self.comparisons:
            if row["policy"] == policy and row["reference"] == reference:
                return row
        raise KeyError((policy, reference))

    def to_dict(self) -> dict:
        """JSON-ready record without the per-scenario units (those are in ``evaluation``)."""
        return {"schema": SCHEMA, "evaluation_schema": self.evaluation["schema"],
                "profile": self.profile, "base_profile": self.base_profile,
                "configuration_sha256": self.configuration_sha256, "modified": self.modified,
                "environment_id": self.environment_id, "task": self.task,
                "reference": self.reference, "reference_role": self.reference_role,
                "calendar_steps": self.calendar_steps, "scenarios": len(self.scenario_seeds),
                "fixture": self.evaluation["fixture"], "rows": [dict(row) for row in self.rows],
                "comparisons": [dict(row) for row in self.comparisons], "bootstrap": dict(self.bootstrap),
                "notes": list(self.notes), "metric_definitions": self.evaluation["metric_definitions"],
                "calibration": None if self.calibration is None else {
                    "origin": self.calibration_origin, "schema": self.calibration["schema"],
                    "sha256": self.calibration["sha256"], "lives": self.calibration["lives"],
                    "seed_root": self.calibration["seed_derivation"]["root"],
                    "selection_criterion": self.calibration["selection_criterion"],
                    "policy_defaults": self.calibration["policy_defaults"],
                    "warnings": self.calibration["warnings"],
                    "verified": bool(self.calibration["verification"].get("performed"))},
                "track": None if self.track is None else dict(self.track)}

    def to_markdown(self) -> str:
        core, resource = self.task == "core", self.task == "resource"
        if self.track is None:
            lines = [f"Benchmark on `{self.profile}`: {len(self.scenario_seeds)} scenarios, "
                     f"up to {self.calendar_steps:,} calendar steps each.",
                     f"Primary reference `{self.reference}`: {self.reference_role}."]
        else:
            lines = [f"Benchmark track `{self.track['name']}` (sha256 {self.track['sha256'][:12]}) on "
                     f"`{self.profile}`: {len(self.scenario_seeds):,} test scenarios, up to "
                     f"{self.calendar_steps:,} calendar steps each.",
                     f"Primary reference `{self.reference}`: {self.reference_role}."]
            if self.track["agent_observation"] != "full":
                lines.append(f"Agents receive the {AGENT_OBSERVATIONS[self.track['agent_observation']]}; "
                             "the references read the unchanged observation.")
        lines.append("")
        header = ["Policy", "Kind", "Information", "Profit per 1,000 calendar steps",
                  f"% of {self.reference}", "Failures per 1,000 lives"]
        if core:
            header += ["Order margin mean (sd)", "Late orders"]
        if resource:
            header += ["Invalid actions"]
        lines += ["| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|"]
        for row in self.rows:
            cells = [f"`{_cell(row['policy'])}`", _KIND_TEXT[row["kind"]], row["information"],
                     _number(row["pooled_profit_per_1000_calendar_steps"], 1),
                     _number(row["percent_of_reference"], 2), _number(row["failures_per_1000_lives"], 1)]
            if core:
                margin = "n/a" if row["order_margin_mean"] is None else (
                    f"{row['order_margin_mean']:.2f} ({_number(row['order_margin_sd'], 2)})")
                late = "n/a" if row["late_order_share"] is None else (
                    f"{100 * row['late_order_share']:.1f}% ({row['late_orders']}/{row['orders_placed']})")
                cells += [margin, late]
            if resource:
                cells += [str(row["invalid_actions"])]
            lines.append("| " + " | ".join(cells) + " |")
        if self.comparisons:
            level = f"{100 * self.bootstrap['confidence']:g}%"
            lines += ["", f"Paired differences on the same scenarios in pooled profit per 1,000 calendar steps, "
                          f"with {level} percentile bootstrap intervals over scenarios "
                          f"({self.bootstrap['samples']:,} resamples, seed {self.bootstrap['seed']}):", "",
                      f"| Policy | Against | Difference | {level} interval | Scenarios better / tied / worse |",
                      "|---|---|---:|---:|---:|"]
            for row in self.comparisons:
                lines.append(f"| `{_cell(row['policy'])}` | `{_cell(row['reference'])}` | "
                             f"{row['difference_per_1000_calendar_steps']:+,.1f} | "
                             f"[{row['interval_low']:+,.1f}, {row['interval_high']:+,.1f}] | "
                             f"{row['scenarios_better']} / {row['scenarios_tied']} / {row['scenarios_worse']} |")
        if self.notes:
            lines += ["", *[f"- {note}" for note in self.notes]]
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()


def _cell(text):
    return str(text).replace("|", "\\|")


def _number(value, digits):
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _pooled_differences(left, right, indices):
    """Bootstrap distribution of the pooled-profit difference for paired scenario resamples."""
    cash_a, calendar_a = left
    cash_b, calendar_b = right
    result = np.empty(len(indices))
    for start in range(0, len(indices), 256):
        block = indices[start:start + 256]
        result[start:start + 256] = 1000 * (cash_a[block].sum(axis=1) / calendar_a[block].sum(axis=1)
                                            - cash_b[block].sum(axis=1) / calendar_b[block].sum(axis=1))
    return result


def _result(profile, task, entries, record, *, bootstrap_samples, bootstrap_seed, confidence, thresholds=None,
            applied=None, origin=None, explicit=(), track=None, agent_information=AGENT_INFORMATION):
    thresholds = thresholds or {}
    primary = PRIMARY_REFERENCES[task]
    summaries = {summary["policy"]: summary for summary in record["summary"]}
    seeds = record["fixture"]["seeds"]
    units = {}
    for entry in entries:
        selected = sorted((row for row in record["units"] if row["policy"] == entry.label), key=lambda row: row["paired_unit"])
        units[entry.label] = (np.asarray([row["raw_profit"] for row in selected], dtype=float),
                              np.asarray([row["actual_calendar_steps"] for row in selected], dtype=float),
                              np.asarray([row["profit_per_1000_calendar_steps"] for row in selected], dtype=float))
    reference_rate = summaries[primary]["pooled_profit_per_1000_calendar_steps"]
    rows = []
    for entry in entries:
        summary = summaries[entry.label]
        rate = summary["pooled_profit_per_1000_calendar_steps"]
        rows.append({
            "policy": entry.label, "kind": entry.kind,
            "role": "primary reference" if entry.label == primary else "agent" if entry.kind == evaluation.AGENT else "reference",
            "information": agent_information if entry.kind == evaluation.AGENT else INFORMATION[task].get(entry.label, "observation"),
            "scenarios": summary["independent_synthetic_scenarios"],
            "pooled_profit_per_1000_calendar_steps": rate,
            "percent_of_reference": 100 * (rate / reference_rate) if reference_rate > 0 else None,
            "mean_profit_per_1000_calendar_steps": summary["mean_profit_per_1000_calendar_steps"],
            "failures": summary["failures"], "ended_lives": summary["ended_lives"],
            "failures_per_1000_lives": summary["failures_per_1000_lives"],
            "failures_per_1000_calendar_steps": summary["failures_per_1000_calendar_steps"],
            "orders_placed": summary["orders_placed"], "late_orders": summary["late_orders"],
            "late_order_share": summary["late_order_share"], "order_margin_mean": summary["order_margin_mean"],
            "order_margin_sd": summary["order_margin_sd"], "invalid_actions": summary["invalid_actions"],
            "threshold": (thresholds[entry.label][0] if entry.kind == evaluation.BUILT_IN and entry.label in thresholds
                          else None)})
    rows.sort(key=lambda row: (-row["pooled_profit_per_1000_calendar_steps"], row["policy"]))
    agents = [entry.label for entry in entries if entry.kind == evaluation.AGENT]
    references = [entry.label for entry in entries if entry.kind != evaluation.AGENT]
    pairs = ([(agent, reference) for agent in agents for reference in references] if agents
             else [(reference, primary) for reference in references if reference != primary])
    indices = np.random.default_rng(bootstrap_seed).integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    tail = (1 - confidence) / 2
    comparisons = []
    for left, right in pairs:
        draws = _pooled_differences(units[left][:2], units[right][:2], indices)
        low, high = np.quantile(draws, [tail, 1 - tail])
        per_scenario = units[left][2] - units[right][2]
        comparisons.append({
            "policy": left, "reference": right,
            "difference_per_1000_calendar_steps": (summaries[left]["pooled_profit_per_1000_calendar_steps"]
                                                   - summaries[right]["pooled_profit_per_1000_calendar_steps"]),
            "interval_low": float(low), "interval_high": float(high), "confidence": confidence,
            "paired_scenarios": len(seeds), "scenarios_better": int(np.sum(per_scenario > 0)),
            "scenarios_tied": int(np.sum(per_scenario == 0)), "scenarios_worse": int(np.sum(per_scenario < 0))})
    notes = []
    if reference_rate <= 0:
        notes.append("The primary reference's pooled profit is not positive, so percentages are omitted.")
    if task == "core":
        # A run cut by the calendar budget before its renewal can earn more per calendar step than the
        # oracle's complete cycle, so the oracle bounds the other policies only for complete cycles.
        cut = sum(1 for row in record["units"] if not row["terminated"])
        if cut:
            notes.append(f"{cut:,} of {len(record['units']):,} scenario runs ended at the calendar budget before their "
                         "renewal. The oracle is an upper bound only for complete component cycles; use a calendar "
                         "budget of at least the longest cycle (166 calendar steps in paper-core-continuous-v1).")
    kept =[rule for rule, (_, source) in thresholds.items() if source != "calibration"]
    if profile.get("modified") and (task != "core" or kept):
        notes.append("Modified profile: the rule thresholds are its policy_defaults and were not recalibrated."
                     + ("" if task == "core" else " Rule calibration is available for the core rules only."))
    if thresholds:
        notes.append("Rule thresholds: " + ", ".join(
            f"{rule} {_value(value)} ({'calibrated' if source == 'calibration' else 'policy_defaults'})"
            for rule, (value, source) in thresholds.items()) + ".")
    if applied is not None:
        selection = CRITERION_PHRASES[applied["selection_criterion"]]
        if origin == "track":
            # No record hash here: the table stays a function of rounded results only.
            notes.append(f"Rules calibrated for track {track['name']}: selected by {selection} on its "
                         f"{applied['lives']:,} calibration lives, which are disjoint from the test scenarios.")
        else:
            how = {"automatic": "because the environment configuration differs from " + profile["base_profile"],
                   "requested": "on request", "supplied": "from a supplied calibration record"}[origin]
            notes.append(f"Rules recalibrated {how}: selected by {selection} on {applied['lives']:,} calibration "
                         f"lives that are disjoint from the test scenarios (calibration sha256 {applied['sha256'][:12]}).")
        notes.extend(f"Calibration warning: {text}" for text in applied["warnings"])
    elif track is not None and task != "core":
        notes.append(f"Rule calibration is defined for the core rules only; the {task} rules keep the "
                     "profile's settings.")
    for rule in explicit:
        notes.append(f"The {rule} threshold is set explicitly by the configuration and was kept.")
    if any(row["invalid_actions"] for row in rows):
        notes.append("Masked actions were executed as no action; invalid_actions counts them.")
    notes.append("Intervals resample scenarios and are conditional on each agent as given; "
                 "they do not include variation across independently trained models.")
    return BenchmarkResult(
        profile=profile["id"], base_profile=profile["base_profile"],
        configuration_sha256=profile["configuration_sha256"], modified=bool(profile.get("modified")),
        environment_id=profile["environment_id"], task=task, reference=primary,
        reference_role=REFERENCE_ROLES[task], calendar_steps=record["fixture"]["scenario_contract"]["calendar_steps"],
        scenario_seeds=tuple(seeds), rows=tuple(rows), comparisons=tuple(comparisons),
        bootstrap={"method": "paired percentile bootstrap over scenarios of the pooled-profit difference",
                   "samples": bootstrap_samples, "seed": bootstrap_seed, "confidence": confidence},
        notes=tuple(notes), evaluation=record, calibration=applied, calibration_origin=origin, track=track)


def environment_differs(profile) -> bool:
    """True when the profile's ``env_config`` differs from that of its base profile."""
    resolved = _resolve(profile)
    base = resolve_profile(resolved["base_profile"])
    return canonical_hash(resolved["env_config"]) != canonical_hash(base["env_config"])


def check_calibration(profile, calibration, test_seeds, calendar_steps, criterion=None) -> CalibrationResult:
    """Validate a stored calibration for a benchmark run and return it as a ``CalibrationResult``.

    The calibration must belong to a core profile with the same environment configuration
    and calendar budget, none of its lives may be a test scenario, and when ``criterion`` is
    given it must have been selected with that criterion.
    """
    resolved = _resolve(profile)
    if criterion is not None:
        criterion = _criterion(criterion)
    if isinstance(calibration, Mapping):
        calibration = CalibrationResult.from_dict(calibration)
    elif not isinstance(calibration, CalibrationResult):
        raise ValueError("calibration must be a CalibrationResult or a calibration record (calibration.json)")
    if criterion is not None and calibration.criterion != criterion:
        raise ValueError(f"The calibration was selected with the criterion {calibration.criterion!r}, not "
                         f"{criterion!r}; calibrate again with {criterion!r} or omit the criterion")
    if task_of(resolved) != "core":
        raise ValueError(f"Rule calibration is defined for the core profiles only, not for {resolved['base_profile']}")
    configuration = evaluation._environment_configuration(resolved, calendar_steps)
    if (calibration.environment_id != resolved["environment_id"]
            or calibration.environment_configuration_sha256 != canonical_hash(configuration)):
        raise ValueError("The calibration was made for another environment configuration; calibrate again for this one")
    if calibration.calendar_steps != calendar_steps:
        raise ValueError(f"The calibration used {calibration.calendar_steps} calendar steps per life and the benchmark "
                         f"{calendar_steps}; use the same budget")
    overlap = sorted(set(calibration.seeds) & {int(seed) for seed in test_seeds})
    if overlap:
        raise ValueError(f"{len(overlap)} calibration lives are also test scenarios, e.g. seed {overlap[0]}; "
                         "calibration and test lives must be disjoint")
    return calibration


def _calibration_plan(profile, task, labels, *, recalibrate, calibration, test_seeds, calendar_steps, seed,
                      calibration_seed, calibration_lives, criterion=None):
    """Return (calibration result or None, origin, rules that keep an explicit threshold)."""
    if recalibrate not in (None, True, False):
        raise ValueError("recalibrate must be None (automatic), True or False")
    if criterion is not None:
        criterion = _criterion(criterion)
    rules = [rule for rule in RULES if rule in labels] if task == "core" else []
    if calibration is not None:
        if recalibrate is not None:
            raise ValueError("Pass either calibration or recalibrate, not both")
        return check_calibration(profile, calibration, test_seeds, calendar_steps, criterion), "supplied", []
    if recalibrate is False:
        if criterion is not None:
            raise ValueError("criterion selects how the rules are recalibrated, and recalibrate=False keeps the "
                             "profile's policy_defaults; pass only one of them")
        return None, None, []
    if task != "core":
        if recalibrate:
            raise ValueError(f"Rule calibration is defined for the core profiles only, not for {profile['base_profile']}")
        return None, None, []
    explicit = []
    if recalibrate is None:
        if not environment_differs(profile):
            return None, None, []
        base = resolve_profile(profile["base_profile"])["policy_defaults"]
        explicit = [rule for rule in rules if profile["policy_defaults"].get(rule) != base.get(rule)]
        origin = "automatic"
    else:
        origin = "requested"
    targets = [rule for rule in rules if rule not in explicit]
    if not targets:
        return None, None, explicit
    result = calibrate(profile, rules=targets, criterion=DEFAULT_CRITERION if criterion is None else criterion,
                       seed=seed if calibration_seed is None else calibration_seed, lives=calibration_lives,
                       exclude_seeds=test_seeds, calendar_steps=calendar_steps)
    return result, origin, explicit


def score(agent=None, *, profile=DEFAULT_PROFILE, seeds=None, seed=0, episodes=100, calendar_steps=720,
          references=None, label=None, fixture=None, checkpoint=None, apply_filter=False, events=False,
          bootstrap_samples=2000, bootstrap_seed=0, confidence=0.95, recalibrate=None, calibration=None,
          calibration_seed=None, calibration_lives=DEFAULT_LIVES, criterion=None) -> BenchmarkResult:
    """Score agents against the task's references on identical scenarios.

    agent: None (references only), a callable ``agent(observation, info) -> action``
        (labelled by ``label`` or its name), a ``(label, agent)`` tuple, a mapping
        from label to agent, or a sequence of ``(label, agent)`` tuples.
    profile: a profile ID or a profile resolved with ``resolve_profile`` (overrides).
    seeds: explicit scenario seeds. Otherwise ``fixture`` replays a saved fixture,
        or ``episodes`` seeds are derived from ``seed`` exactly as ``evaluate`` does.
    references: names to compare against; default ``default_references(profile)``.
        The primary reference (oracle, dp or rolling) is always included.
    events: record per-scenario event lists in ``result.evaluation`` (default off).
    recalibrate: None recalibrates the core rules among the references when the
        environment configuration differs from the base profile, except rules whose
        threshold the configuration sets explicitly; True recalibrates them always
        (core only); False keeps the profile's ``policy_defaults``.
    calibration: a ``CalibrationResult`` or its record (``calibration.json``) for the same
        configuration and calendar budget, applied instead of a new calibration.
    calibration_seed, calibration_lives: root seed (default: ``seed``) and size of the
        calibration bank; its lives are disjoint from the test scenarios.
    criterion: selection criterion of a recalibration, ``"zero_failure"`` (used when None) or
        ``"pooled"`` (see ``calibration.calibrate``). With ``calibration=`` it must name the
        criterion of the supplied record; with ``recalibrate=False`` it is rejected.
    """
    resolved = _resolve(profile)
    task = task_of(resolved)
    entries = benchmark_entries(resolved, agent, references=references, label=label, checkpoint=checkpoint)
    bootstrap_samples, bootstrap_seed, confidence = _bootstrap_settings(bootstrap_samples, bootstrap_seed, confidence)
    episodes = evaluation._integer(episodes, "episodes", 1)
    seed = evaluation._integer(seed, "seed")
    calendar_steps = evaluation._integer(calendar_steps, "calendar_steps", 1)
    if seeds is not None:
        if fixture is not None:
            raise ValueError("Pass either seeds or fixture, not both")
        fixture = evaluation.make_fixture(resolved, seeds, calendar_steps=calendar_steps)
    _, test_seeds = evaluation._scenario_seeds(resolved, seed=seed, episodes=episodes,
                                               calendar_steps=calendar_steps, fixture=fixture)
    built_in = [entry.label for entry in entries if entry.kind == evaluation.BUILT_IN]
    result, origin, explicit = _calibration_plan(
        resolved, task, built_in, recalibrate=recalibrate, calibration=calibration, test_seeds=test_seeds,
        calendar_steps=calendar_steps, seed=seed, calibration_seed=calibration_seed,
        calibration_lives=calibration_lives, criterion=criterion)
    return _score_entries(resolved, task, entries, test_seeds, calendar_steps=calendar_steps,
                          apply_filter=apply_filter, events=events, bootstrap_samples=bootstrap_samples,
                          bootstrap_seed=bootstrap_seed, confidence=confidence, calibration=result,
                          origin=origin, explicit=explicit)


def _bootstrap_settings(bootstrap_samples, bootstrap_seed, confidence):
    bootstrap_samples = evaluation._integer(bootstrap_samples, "bootstrap_samples", 1)
    bootstrap_seed = evaluation._integer(bootstrap_seed, "bootstrap_seed")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    return bootstrap_samples, bootstrap_seed, float(confidence)


def _score_entries(resolved, task, entries, test_seeds, *, calendar_steps, apply_filter=False, events=False,
                   bootstrap_samples, bootstrap_seed, confidence, calibration=None, origin=None, explicit=(),
                   track=None, agent_information=AGENT_INFORMATION):
    """Apply calibrated thresholds, evaluate every entry on the test seeds and build the result."""
    built_in = [entry.label for entry in entries if entry.kind == evaluation.BUILT_IN]
    thresholds = {}
    if task == "core":
        for rule in RULES:
            if rule in built_in:
                setting = resolved["policy_defaults"].get(rule, {})
                if calibration is not None and rule in calibration.rules:
                    thresholds[rule] = (calibration.selected[rule], "calibration")
                elif PARAMETERS[rule] in setting:
                    thresholds[rule] = (setting[PARAMETERS[rule]], "policy_defaults")
    if calibration is not None:
        calibrated = _with_thresholds(resolved, {rule: value for rule, (value, source) in thresholds.items()
                                                 if source == "calibration"})
        entries = [evaluation._Entry(entry.label, entry.kind, lambda env, name=entry.label: Policy(name, calibrated, env))
                   if entry.kind == evaluation.BUILT_IN and thresholds.get(entry.label, (None, None))[1] == "calibration"
                   else entry for entry in entries]
    # With a fixture, the evaluation reads its scenarios from the fixture alone.
    record = evaluation._evaluate_entries(resolved, entries, seed=0, episodes=1, calendar_steps=calendar_steps,
                                          apply_filter=apply_filter,
                                          fixture=evaluation.make_fixture(resolved, test_seeds, calendar_steps=calendar_steps),
                                          events=events)
    return _result(resolved, task, entries, record, bootstrap_samples=bootstrap_samples,
                   bootstrap_seed=bootstrap_seed, confidence=confidence, thresholds=thresholds,
                   applied=None if calibration is None else calibration.to_dict(), origin=origin, explicit=explicit,
                   track=track, agent_information=agent_information)


class _ZeroDeadlineAgent:
    """Hand an agent the observation with the deadline coordinate set to zero (zero-deadline tracks)."""

    def __init__(self, agent):
        self._agent = agent

    def __call__(self, observation, info):
        return self._agent(zero_deadline(observation), info)


def run_track(track=None, agents=None, *, label=None, events=False, bootstrap_samples=2000, bootstrap_seed=0,
              confidence=0.95, criterion=None) -> BenchmarkResult:
    """Score agents and the references of a benchmark track on its fixed test scenarios.

    track: None (the default track ``core-continuous-v1``), a packaged track name, a tracks
        JSON file, a ``tracks.Track`` or its JSON form.
    agents: None (references only), a callable ``agent(observation, info) -> action``
        (labelled by ``label`` or its name), a ``(label, agent)`` tuple, a mapping from
        label to agent, or a sequence of ``(label, agent)`` tuples.

    criterion: selection criterion of the rule calibration, ``"zero_failure"`` (used when None) or
        ``"pooled"`` (see ``calibration.calibrate``).

    The references and the profile are those of the track. The core rules listed in the
    track's ``calibrated_rules`` are calibrated on the track's calibration scenarios, which
    are disjoint from its test scenarios; every other rule keeps the profile's settings.
    On a zero-deadline track each agent receives the observation with index 3 set to zero.
    The result's ``track`` holds the track summary and its ``sha256``.
    """
    spec = load_track(track)
    resolved = spec.resolved_profile()
    task = task_of(resolved)
    pairs = _agent_pairs(agents, label)
    if spec.agent_observation == "zero_deadline":
        pairs = [(name, _ZeroDeadlineAgent(agent)) for name, agent in pairs]
    entries = benchmark_entries(resolved, pairs or None, references=list(spec.references))
    bootstrap_samples, bootstrap_seed, confidence = _bootstrap_settings(bootstrap_samples, bootstrap_seed, confidence)
    criterion = DEFAULT_CRITERION if criterion is None else _criterion(criterion)
    calibration = None
    if spec.calibrated_rules:
        calibration = calibrate(resolved, rules=spec.calibrated_rules, criterion=criterion,
                                seeds=list(spec.calibration_seeds), exclude_seeds=spec.test_seeds,
                                calendar_steps=spec.calendar_steps)
    return _score_entries(resolved, task, entries, list(spec.test_seeds), calendar_steps=spec.calendar_steps,
                          events=events, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
                          confidence=confidence, calibration=calibration,
                          origin=None if calibration is None else "track", track=spec.summary(),
                          agent_information=spec.agent_information)


__all__ = ["BenchmarkResult", "CRITERION_PHRASES", "DEFAULT_PROFILE", "ORACLE", "PRIMARY_REFERENCES", "REFERENCE_ROLES",
           "agent_label", "benchmark_entries", "check_calibration", "default_references", "environment_differs",
           "run_track", "score", "task_of"]
