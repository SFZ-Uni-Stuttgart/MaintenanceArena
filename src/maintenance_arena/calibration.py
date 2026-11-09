"""Grid-search calibration of the core rule thresholds on lives kept apart from the test scenarios.

``calibrate`` selects the thresholds of the core rules ``age`` (order at a counted
operating age), ``health`` (order once the observed health falls to a threshold)
and ``projection`` (order once the projected remaining life falls to a threshold)
on a bank of calibration lives. Use it whenever a configuration changes the task,
so that a learner is compared with rules selected for that task rather than with
the profile's frozen ``policy_defaults``. The profiles themselves are never changed.

Two selection criteria are available (``CRITERIA``):

* ``zero_failure`` (the default): only a candidate without a failure on the
  calibration lives (and with at least one planned exchange) is eligible, and
  among the eligible ones the highest family-equal profit wins. The family-equal
  profit is the mean, with equal weight per wear family, of each family's pooled
  profit per 1,000 calendar steps over that family's calibration lives. Ties go
  to the lower mean normalized discarded life over the planned exchanges, then
  to the later-acting threshold.
* ``pooled``: the highest pooled calendar profit, total cash over total calendar
  steps of all calibration lives, failures allowed; ties go to fewer failures,
  then to the later-acting threshold.

The calibration lives are derived from a root seed on their own stream,
``SeedSequence(root, spawn_key=(1, i))``, while ``evaluate`` and ``benchmark``
derive test scenarios as ``SeedSequence([root, i])``. Any calibration seed that
appears among ``exclude_seeds`` (or twice) is skipped, so the bank is disjoint
from the test scenarios it is checked against.

The three rules order at the first decision at which their condition holds. Until
that order every action is WAIT, so one WAIT pass through each life fixes the
first order of every threshold, and the cash of the life follows from the order
time and the core's fixed accounting. ``calibrate`` scores all candidates this
way and, by default, re-runs the selected thresholds through the full simulator
on every calibration life and requires identical results.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from numbers import Integral, Real
from pathlib import Path
import statistics
import warnings

import numpy as np

from . import evaluation
from .core import WAIT, validate_config
from .curves import FAMILY_IDS
from .policies import Policy
from .profiles import resolve_profile
from .records import canonical_hash


SCHEMA = "maintenance_arena_rule_calibration_v1"
DEFAULT_PROFILE = "paper-core-continuous-v1"
RULES = ("age", "health", "projection")
PARAMETERS = {"age": "age_interval", "health": "health_threshold", "projection": "projection_threshold"}
DEFAULT_LIVES = 800
# Calibration life i of root r has the scenario seed
# SeedSequence(r, spawn_key=(CALIBRATION_STREAM, i)).generate_state(1, uint32)[0].
CALIBRATION_STREAM = 1
SEED_DERIVATION = ("SeedSequence(root, spawn_key=(1, i)).generate_state(1, dtype=uint32)[0] for i = 0, 1, ...; "
                   "a seed already in the bank or among the excluded seeds is skipped")
# Selection criteria. Under "zero_failure" a candidate is eligible only with zero failures, finite
# metrics and at least one planned exchange on the calibration lives; the eligible candidate with
# the highest family-equal profit wins; ties go to the lower mean normalized discarded life over
# the planned exchanges and then to the later-acting threshold: the higher age, or the lower
# health or projection threshold. The scoring reproduces the simulator's cash ledger by
# construction, and the selected thresholds are verified in the full simulator.
CRITERIA = ("zero_failure", "pooled")
DEFAULT_CRITERION = "zero_failure"
CRITERION_TEXTS = {
    "zero_failure": ("Among the candidates without a failure and with at least one planned exchange on the calibration "
                   "lives, the highest family-equal profit: the mean, with equal weight per wear family, of each "
                   "family's pooled profit per 1,000 calendar steps, 1000 * sum(raw_profit) / "
                   "sum(actual_calendar_steps) over the calibration lives of that family, for the wear families "
                   "present in the bank"),
    "pooled": ("Highest pooled profit per 1,000 calendar steps, 1000 * sum(raw_profit) / sum(actual_calendar_steps), "
               "over all calibration lives"),
}
TIE_BREAK_TEXTS = {
    "zero_failure": ("Equal family-equal profit: the lower mean normalized discarded life over the lives with a planned "
                   "exchange, then the later-acting threshold (higher age, lower health threshold, lower projection "
                   "threshold)"),
    "pooled": ("Equal pooled profit: fewer failures, then the later-acting threshold (higher age, lower health "
               "threshold, lower projection threshold)"),
}
# Documentation strings of the default criterion.
CRITERION = CRITERION_TEXTS[DEFAULT_CRITERION]
TIE_BREAK = TIE_BREAK_TEXTS[DEFAULT_CRITERION]
# Default grids, 252 candidates: fixed ages 1..159 (159 candidates), health thresholds
# integer / 100 for integer 20..95 (76) and projection triggers 8..24 (17). With a
# preparation time L other than 12 the projection triggers shift with it, to
# L - 4 .. L + 12, so that they keep covering the last safe order L + 1.
_DEFAULT_GRIDS = {
    "age": tuple(range(1, 160)),
    "health": tuple(integer / 100.0 for integer in range(20, 96)),
    "projection": tuple(range(8, 25)),
}


class CalibrationWarning(UserWarning):
    """A calibration finished, but its selection deserves attention; the text is also recorded."""


class GridEdgeWarning(CalibrationWarning):
    """The selected threshold is the smallest or largest value of its grid."""


class FamilyCoverageWarning(CalibrationWarning):
    """A wear family that the configuration samples has no life in the calibration bank."""


def default_grids(mobilization_lead: int = 12) -> dict[str, tuple]:
    """The default candidate grids per rule (252 candidates in total) for a preparation time."""
    lead = evaluation._integer(mobilization_lead, "mobilization_lead", 1)
    grids = {rule: tuple(values) for rule, values in _DEFAULT_GRIDS.items()}
    if lead != 12:
        grids["projection"] = tuple(value + lead - 12 for value in _DEFAULT_GRIDS["projection"])
    return grids


def _criterion(criterion):
    if not isinstance(criterion, str) or criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {', '.join(CRITERIA)}; got {criterion!r}")
    return criterion


def _resolve(profile):
    if isinstance(profile, str):
        profile = resolve_profile(profile)
    if not isinstance(profile, Mapping) or "environment_id" not in profile or "base_profile" not in profile:
        raise ValueError("profile must be a profile ID or a profile resolved with maintenance_arena.profiles.resolve_profile")
    if "CoreCycle" not in profile["environment_id"]:
        raise ValueError(f"Rule calibration is defined for the core profiles (CoreCycle-v0) and their rules "
                         f"{', '.join(RULES)}; {profile['base_profile']} uses {profile['environment_id']}")
    return profile


def calibration_seeds(seed, lives=DEFAULT_LIVES, *, exclude=()) -> list[int]:
    """Scenario seeds of a calibration bank of ``lives`` lives derived from the root ``seed``.

    Life ``i`` gets ``SeedSequence(seed, spawn_key=(1, i)).generate_state(1, uint32)[0]``.
    A seed already in the bank or in ``exclude`` is skipped and the derivation
    continues with the next ``i``, so the bank always holds ``lives`` distinct seeds.
    """
    return _derive(seed, lives, _seed_set(exclude, "excluded seed"))[0]


def _derive(seed, lives, excluded):
    """Derived calibration seeds and the skipped derivations as (index, seed, reason) records."""
    root = evaluation._integer(seed, "calibration seed")
    lives = evaluation._integer(lives, "lives", 1)
    seeds, chosen, skipped, index = [], set(), [], 0
    while len(seeds) < lives:
        value = int(np.random.SeedSequence(root, spawn_key=(CALIBRATION_STREAM, index)).generate_state(1, dtype=np.uint32)[0])
        if value in excluded or value in chosen:
            skipped.append({"index": index, "seed": value,
                            "reason": "excluded test seed" if value in excluded else "duplicate"})
        else:
            seeds.append(value)
            chosen.add(value)
        index += 1
    return seeds, skipped


def _seed_set(values, name):
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValueError(f"{name}s must be a sequence of nonnegative integers")
    return {evaluation._integer(value, name) for value in values}


def _decimal_range(text, rule):
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"{rule} grid range must have the form START:STOP or START:STOP:STEP; got {text!r}")
    try:
        start, stop = Decimal(parts[0]), Decimal(parts[1])
        step = Decimal(parts[2]) if len(parts) == 3 else Decimal(1)
    except InvalidOperation as error:
        raise ValueError(f"{rule} grid range {text!r} contains a value that is not a number") from error
    if not all(value.is_finite() for value in (start, stop, step)) or step <= 0 or stop < start:
        raise ValueError(f"{rule} grid range {text!r} needs finite START <= STOP and a positive STEP")
    count = int((stop - start) / step) + 1
    if count > 100000:
        raise ValueError(f"{rule} grid range {text!r} has more than 100,000 values")
    return [start + index * step for index in range(count)]


def parse_grid(rule, tokens):
    """Read a command-line grid: numbers and inclusive ranges START:STOP[:STEP], e.g. ``10:20 30``."""
    values = []
    for token in tokens:
        for part in str(token).replace(",", " ").split():
            if ":" in part:
                values.extend(_decimal_range(part, rule))
            else:
                try:
                    values.append(Decimal(part))
                except InvalidOperation as error:
                    raise ValueError(f"{rule} grid value {part!r} is not a number") from error
    return _grid(rule, [int(value) if value == value.to_integral_value() else float(value) for value in values])


def _grid(rule, values):
    """Validate one grid; return its values sorted, ages as integers, integral projections as integers."""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValueError(f"The {rule} grid must be a sequence of numbers")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"The {rule} grid contains {value!r}; grid values must be finite numbers")
        if rule == "age":
            if not isinstance(value, Integral) and not float(value).is_integer():
                raise ValueError(f"The age grid contains {value!r}; ages are whole operating steps")
            value = int(value)
            if value < 0:
                raise ValueError("The age grid must contain nonnegative ages")
        elif rule == "health":
            value = float(value)
            if not 0 <= value <= 1:
                raise ValueError("The health grid must lie in [0, 1]")
        else:
            value = int(value) if float(value).is_integer() else float(value)
            if value < 0:
                raise ValueError("The projection grid must contain nonnegative thresholds")
        result.append(value)
    if not result:
        raise ValueError(f"The {rule} grid is empty")
    if len(set(result)) != len(result):
        raise ValueError(f"The {rule} grid contains duplicate values")
    return tuple(sorted(result))


def _rules(rules):
    if isinstance(rules, str) or not hasattr(rules, "__iter__"):
        raise ValueError(f"rules must be a sequence of rule names from {', '.join(RULES)}")
    requested = list(rules)
    unknown = [rule for rule in requested if rule not in RULES]
    if unknown or not requested:
        raise ValueError(f"Calibrated rules must be a nonempty selection of {', '.join(RULES)}; got {requested}")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate rules in the calibration request")
    return tuple(rule for rule in RULES if rule in requested)


def _grids(rules, grids, lead=12):
    supplied = {} if grids is None else grids
    if not isinstance(supplied, Mapping):
        raise ValueError("grids must be a mapping from rule name to a sequence of thresholds")
    extra = set(supplied) - set(rules)
    if extra:
        raise ValueError(f"Grids were given for rules that are not calibrated: {sorted(extra)}")
    defaults = default_grids(lead)
    return ({rule: _grid(rule, supplied[rule]) if rule in supplied else defaults[rule] for rule in rules},
            {rule: "user" if rule in supplied else "default" for rule in rules})


# --- Scoring all thresholds from one WAIT pass per life (evaluator side). ---

class _Life:
    """The decisions of one scenario under WAIT and the constants of its accounting."""

    __slots__ = ("seed", "start", "lifetime", "decisions", "budget", "health", "remaining", "family")

    def __init__(self, seed, start, lifetime, decisions, budget, health, remaining, family=None):
        self.seed, self.start, self.lifetime, self.decisions = seed, start, lifetime, decisions
        self.budget, self.health, self.remaining, self.family = budget, health, remaining, family


def _projected_remaining(observation, lead=12):
    """The projection rule's remaining-life estimate, computed exactly as ``core.policy_action``.

The rate channel is ``lead`` times the damage rate, so it is divided by the
configured preparation time (12 in every published profile).
"""
    value = np.asarray(observation, dtype=np.float32)
    if value[2] < .5:
        return math.inf
    health = float(value[0])
    if health <= 0:
        return 0.
    damage = -math.log(health) / math.log(5.)
    return (1. - damage) / max(float(value[1]) / lead, 1.e-12)


def _record_lives(profile, seeds, calendar_steps):
    """Run every life under WAIT up to its last decision; read the lifetime (for the accounting)
    and the wear family (for the family-equal profit) on the evaluator side only."""
    configuration = evaluation._environment_configuration(profile, calendar_steps)
    horizon = configuration.get("calendar_horizon")
    budget = calendar_steps if horizon is None else min(calendar_steps, int(horizon))
    lives, constants = [], None
    for seed in seeds:
        env = evaluation.make_resolved(profile, calendar_steps)
        try:
            observation, _ = env.reset(seed=int(seed))
            core = env.unwrapped
            diagnostics = core.diagnostics()
            lifetime = int(diagnostics["eol_operating_age"])
            family = str(diagnostics["family_id"])
            start = int(core.policy_state()["operating_age"])
            decisions = min(budget, lifetime - start)
            observations = [observation]
            for _ in range(decisions - 1):
                observation, _, terminated, truncated, _ = env.step(WAIT)
                if terminated or truncated:
                    raise RuntimeError("A WAIT pass ended before the last decision of its life")
                observations.append(observation)
            if core.policy_state()["operating_age"] != start + decisions - 1:
                raise RuntimeError("The WAIT pass did not advance one operating step per decision")
            config = core.config
            constants = constants or {name: config[name] for name in (
                "revenue_per_operating_step", "replacement_cost", "downtime_cost_per_step",
                "failure_event_cost", "mobilization_lead", "downtime_steps")}
            if "recovery_contract" in config:
                constants["recovery_contract"] = config["recovery_contract"]
        finally:
            env.close()
        stacked = np.asarray(observations, dtype=np.float32)
        lead = int(constants["mobilization_lead"])
        lives.append(_Life(int(seed), start, lifetime, decisions, budget,
                           stacked[:, 0].astype(np.float64),
                           np.asarray([_projected_remaining(row, lead) for row in stacked], dtype=np.float64), family))
    return lives, constants


def _macro_profits(constants):
    """Cash of the closing decision: a planned exchange, or a failure after m more operating steps."""
    revenue, replacement = constants["revenue_per_operating_step"], constants["replacement_cost"]
    downtime, failure = constants["downtime_cost_per_step"], constants["failure_event_cost"]
    lead, outage = int(constants["mobilization_lead"]), int(constants["downtime_steps"])
    # Each row is revenue - replacement - downtime - failure_cost, as in the core's ledger.
    operate = revenue - 0. - 0. - 0.
    outage_row = 0. - 0. - downtime - 0.
    planned_row = 0. - replacement - downtime - 0.
    failure_row = 0. - replacement - downtime - failure
    planned = math.fsum([operate] * lead + [planned_row] + [outage_row] * outage)
    if constants.get("recovery_contract") == "preserve_preparation_v1":
        detection_row = 0. - 0. - downtime - failure
        failed = [math.fsum([operate] * m + [detection_row] + [outage_row] * (lead - m - 1)
                           + [planned_row] + [outage_row] * outage) for m in range(lead)]
        failed_duration = np.full(lead, lead + 1 + outage, dtype=np.int64)
    else:
        failed = [math.fsum([operate] * m + [failure_row] + [outage_row] * outage) for m in range(lead)]
        failed_duration = np.arange(lead, dtype=np.int64) + 1 + outage
    return operate, planned, np.asarray(failed, dtype=np.float64), lead, outage, failed_duration


def _first_crossing(values, thresholds):
    """Index of the first value <= each threshold, or len(values) when there is none."""
    running = np.minimum.accumulate(values)
    return np.searchsorted(-running, -np.asarray(thresholds, dtype=np.float64), side="left")


def _trigger_indices(rule, life, grid):
    if rule == "age":
        return np.maximum(np.asarray(grid, dtype=np.int64) - life.start, 0)
    values = life.health if rule == "health" else life.remaining
    return _first_crossing(values, [float(value) for value in grid])


def _outcomes(life, indices, macro):
    """Per-threshold cash, calendar steps and events of one life, as the evaluation records them."""
    operate, planned_cash, failed_cash, lead, outage, failed_duration = macro
    count = life.decisions
    prefix = np.zeros(count + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(np.full(count, operate, dtype=np.float64))
    k = np.minimum(np.asarray(indices, dtype=np.int64), count)
    triggered = k < count
    age = life.start + k
    rul = life.lifetime - age
    planned = triggered & (rul >= lead + 1)
    late = triggered & ~planned
    reaches_end = life.lifetime - life.start <= life.budget
    m = np.clip(rul - 1, 0, lead - 1)
    raw = np.where(planned, prefix[k] + planned_cash, prefix[k] + failed_cash[m])
    elapsed = np.where(planned, k + lead + 1 + outage, k + failed_duration[m])
    failure = late.copy()
    if reaches_end:
        raw = np.where(triggered, raw, prefix[count - 1] + failed_cash[0])
        elapsed = np.where(triggered, elapsed, count - 1 + failed_duration[0])
        failure |= ~triggered
    else:
        raw = np.where(triggered, raw, prefix[count])
        elapsed = np.where(triggered, elapsed, count)
    discarded = np.where(planned, life.lifetime - (age + lead), 0)
    return {"raw": raw, "elapsed": elapsed.astype(np.int64), "failure": failure, "planned": planned,
            "rul": np.where(triggered, rul, -1), "triggered": triggered, "normalized_discarded": discarded / life.lifetime}


def _score_grid(rule, lives, grid, macro):
    """Score every threshold of one grid; returns per-life arrays of shape (lives, thresholds)."""
    rows = [_outcomes(life, _trigger_indices(rule, life, grid), macro) for life in lives]
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def _summaries(scored, lead):
    """Summary of every threshold, with the formulas and summation order of ``evaluation._summary``."""
    result = []
    for column in range(scored["raw"].shape[1]):
        raw = scored["raw"][:, column].tolist()
        elapsed = scored["elapsed"][:, column].tolist()
        profit_rates = [1000 * cash / steps for cash, steps in zip(raw, elapsed)]
        failures = int(scored["failure"][:, column].sum())
        planned = int(scored["planned"][:, column].sum())
        ended = failures + planned
        triggered = scored["triggered"][:, column]
        margins = np.asarray([rul - (lead + 1) for rul in scored["rul"][triggered, column].tolist()], dtype=float)
        total, calendar = sum(raw), sum(elapsed)
        result.append({
            "pooled_profit_per_1000_calendar_steps": 1000 * total / calendar,
            "mean_profit_per_1000_calendar_steps": float(np.mean(profit_rates)),
            "raw_profit": total, "actual_calendar_steps": calendar,
            "failures": failures, "planned_interventions": planned, "ended_lives": ended,
            "failures_per_1000_lives": 1000 * failures / ended if ended else None,
            "orders_placed": int(margins.size), "late_orders": int(np.sum(margins < 0)),
            "order_margin_mean": float(np.mean(margins)) if margins.size else None,
            "order_margin_sd": float(np.std(margins, ddof=1)) if margins.size > 1 else None,
            "mean_normalized_discarded_life": float(np.mean(scored["normalized_discarded"][:, column].tolist())),
        })
    return result


def _family_order(families):
    """The wear families that occur in ``families``, in the package's family order."""
    found = set(families)
    return [family for family in FAMILY_IDS if family in found] + sorted(found - set(FAMILY_IDS))


def _criterion_metrics(scored, families, summaries):
    """Per-threshold quantities of the zero-failure criterion.

    families: the wear family of every calibration life, in the row order of ``scored``.
    family_profit_per_1000_calendar_steps: per wear family present in the bank, 1000 times the
        exactly rounded sum of that family's cash over the sum of its calendar steps.
    family_equal_profit_per_1000_calendar_steps: the mean of those family rates, each family with
        equal weight however many lives it has in the bank.
    mean_normalized_discarded_life_over_planned: the mean, over the lives whose first order led to
        a planned exchange, of the operating steps discarded at the exchange divided by the
        component's lifetime; None without a planned exchange.
    zero_failure_eligible: no failure, finite pooled and family-equal profit, and at least one
        planned exchange.
    """
    labels = np.asarray(families)
    present = _family_order(families)
    masks = {family: labels == family for family in present}
    result = []
    for column, summary in enumerate(summaries):
        raw, elapsed = scored["raw"][:, column], scored["elapsed"][:, column]
        rates = {family: float(1000.0 * math.fsum(raw[masks[family]].tolist()) / int(elapsed[masks[family]].sum()))
                 for family in present}
        family_equal = float(statistics.fmean(rates[family] for family in present))
        discarded = scored["normalized_discarded"][scored["planned"][:, column], column].tolist()
        over_planned = float(statistics.fmean(discarded)) if discarded else None
        result.append({
            "family_profit_per_1000_calendar_steps": rates,
            "family_equal_profit_per_1000_calendar_steps": family_equal,
            "mean_normalized_discarded_life_over_planned": over_planned,
            "zero_failure_eligible": bool(summary["failures"] == 0 and over_planned is not None
                                        and math.isfinite(family_equal)
                                        and math.isfinite(summary["pooled_profit_per_1000_calendar_steps"])),
        })
    return result


def _later(rule, value):
    """Sort key that prefers the later-acting threshold: the higher age, the lower health or projection threshold."""
    return -value if rule == "age" else value


def _sort_key(rule, threshold, row, criterion):
    if criterion == "pooled":
        return (-row["pooled_profit_per_1000_calendar_steps"], row["failures"], _later(rule, threshold))
    discarded = row["mean_normalized_discarded_life_over_planned"]
    return (not row["zero_failure_eligible"], -row["family_equal_profit_per_1000_calendar_steps"],
            math.inf if discarded is None else discarded, _later(rule, threshold))


def _select(rule, grid, rows, criterion=DEFAULT_CRITERION):
    """Indices of ``grid`` in the selection order of ``criterion``; ``rows[j]`` holds the metrics of ``grid[j]``.

    ``pooled`` needs ``pooled_profit_per_1000_calendar_steps`` and ``failures``; ``zero_failure``
    needs ``zero_failure_eligible``, ``family_equal_profit_per_1000_calendar_steps`` and
    ``mean_normalized_discarded_life_over_planned``. Under ``zero_failure`` the ineligible
    candidates follow the eligible ones, in the same order among themselves.
    """
    criterion = _criterion(criterion)
    return sorted(range(len(grid)), key=lambda j: _sort_key(rule, grid[j], rows[j], criterion))


class NoEligibleThresholdError(ValueError):
    """The zero-failure criterion found no candidate without a failure (and with a planned exchange)."""


def _no_eligible_message(rule, grid, rows, lives):
    fewest = min(range(len(grid)), key=lambda j: (rows[j]["failures"], _later(rule, grid[j])))
    if rows[fewest]["failures"] == 0:
        reason = (f"the {rule} thresholds without a failure order no planned exchange within the calendar budget "
                  f"of the {lives:,} calibration lives")
    else:
        reason = (f"every {rule} threshold of the grid fails on at least one of the {lives:,} calibration lives "
                  f"(fewest failures: {rows[fewest]['failures']} at {_value(grid[fewest])})")
    return (f"No {rule} threshold is eligible under the zero-failure criterion, which selects only among candidates "
            f"without a failure and with a planned exchange on the calibration lives: {reason}. Widen the {rule} grid "
            f"({_value(grid[0])} to {_value(grid[-1])}), or select with the criterion 'pooled' (--criterion pooled), "
            "which accepts failures.")


def _paired_interval(left, right, *, samples, seed, confidence):
    from .benchmark import _pooled_differences
    indices = np.random.default_rng(seed).integers(0, len(left[0]), size=(samples, len(left[0])))
    draws = _pooled_differences(left, right, indices)
    tail = (1 - confidence) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return float(low), float(high)


def _with_thresholds(profile, thresholds):
    """A copy of ``profile`` whose rule settings use ``thresholds``; only for building policies."""
    result = copy.deepcopy(dict(profile))
    defaults = result.setdefault("policy_defaults", {})
    for rule, value in thresholds.items():
        defaults[rule] = dict(defaults.get(rule, {}), **{PARAMETERS[rule]: value})
    return result


def _verify(profile, rules, selected, seeds, calendar_steps, scored, columns):
    """Re-run the selected thresholds through the full simulator and require identical units."""
    calibrated = _with_thresholds(profile, {rule: selected[rule] for rule in rules})
    entries = [evaluation._Entry(rule, evaluation.BUILT_IN,
                                 lambda env, rule=rule: Policy(rule, calibrated, env)) for rule in rules]
    fixture = evaluation.make_fixture(profile, seeds, calendar_steps=calendar_steps)
    record = evaluation._evaluate_entries(profile, entries, seed=0, episodes=1, calendar_steps=calendar_steps,
                                          fixture=fixture, events=False)
    summaries = {summary["policy"]: summary for summary in record["summary"]}
    for rule in rules:
        units = sorted((row for row in record["units"] if row["policy"] == rule), key=lambda row: row["paired_unit"])
        column = columns[rule]
        expected = scored[rule]
        for index, unit in enumerate(units):
            rul = int(expected["rul"][index, column])
            observed = (unit["raw_profit"], unit["actual_calendar_steps"], unit["failures"],
                        unit["planned_interventions"], unit["first_trigger_true_rul"])
            wanted = (float(expected["raw"][index, column]), int(expected["elapsed"][index, column]),
                      int(expected["failure"][index, column]), int(expected["planned"][index, column]),
                      None if rul < 0 else rul)
            if observed != wanted:
                raise RuntimeError(f"Calibration scoring disagrees with the simulator for {rule}={selected[rule]} "
                                   f"on seed {unit['seed']}: {observed} != {wanted}")
    return {"performed": True, "identical": True, "scenarios": len(seeds), "policies": list(rules),
            "method": "The selected thresholds were re-run through the full simulator on every calibration life; "
                      "cash, calendar steps, failures, planned exchanges and the first order matched exactly.",
            "simulator_pooled_profit_per_1000_calendar_steps": {
                rule: summaries[rule]["pooled_profit_per_1000_calendar_steps"] for rule in rules}}


@dataclass(frozen=True)
class CalibrationResult:
    """Selected thresholds, the full selection table and the calibration bank of one calibration."""

    profile: str
    base_profile: str
    configuration_sha256: str
    environment_id: str
    environment_configuration: dict
    environment_configuration_sha256: str
    calendar_steps: int
    seed: int | None
    seeds: tuple[int, ...]
    skipped_seeds: tuple[dict, ...]
    excluded_seed_count: int
    rules: tuple[str, ...]
    grids: dict
    grid_sources: dict
    selected: dict
    table: dict
    defaults: dict
    warnings: tuple[str, ...]
    verification: dict
    bootstrap: dict
    # The selection criterion, one of CRITERIA, and the number of calibration lives per wear family.
    criterion: str
    families: dict

    def policy_defaults(self) -> dict:
        """The selected thresholds in the form of a profile's ``policy_defaults``."""
        return {rule: {PARAMETERS[rule]: self.selected[rule]} for rule in self.rules}

    def overrides(self) -> dict:
        """Profile overrides for ``--config``: the calibrated thresholds and the changed ``env_config`` keys."""
        base = resolve_profile(self.base_profile)["env_config"]
        changed = {name: copy.deepcopy(value) for name, value in self.environment_configuration.items()
                   if name not in base or base[name] != value}
        result = {"env_config": changed} if changed else {}
        result["policy_defaults"] = self.policy_defaults()
        return result

    def row(self, rule, threshold=None) -> dict:
        """The table row of a threshold, by default of the selected one."""
        wanted = self.selected[rule] if threshold is None else threshold
        for row in self.table[rule]:
            if row["threshold"] == wanted:
                return row
        raise KeyError((rule, wanted))

    def to_dict(self) -> dict:
        """JSON-ready record with the full selection table and an integrity hash (``sha256``)."""
        record = {
            "schema": SCHEMA, "profile": self.profile, "base_profile": self.base_profile,
            "configuration_sha256": self.configuration_sha256, "environment_id": self.environment_id,
            "environment_configuration": copy.deepcopy(self.environment_configuration),
            "environment_configuration_sha256": self.environment_configuration_sha256,
            "calendar_steps": self.calendar_steps, "selection_criterion": self.criterion,
            "criterion": CRITERION_TEXTS[self.criterion], "tie_break": TIE_BREAK_TEXTS[self.criterion],
            "families": dict(self.families),
            "seed_derivation": {"root": self.seed, "stream": CALIBRATION_STREAM,
                                "method": SEED_DERIVATION if self.seed is not None else "explicit seed list",
                                "skipped": [dict(row) for row in self.skipped_seeds]},
            "lives": len(self.seeds), "seeds": list(self.seeds), "excluded_seed_count": self.excluded_seed_count,
            "rules": list(self.rules), "grids": {rule: list(values) for rule, values in self.grids.items()},
            "grid_sources": dict(self.grid_sources), "selected": dict(self.selected),
            "policy_defaults": self.policy_defaults(),
            "table": {rule: [dict(row) for row in rows] for rule, rows in self.table.items()},
            "defaults": {rule: dict(row) for rule, row in self.defaults.items()},
            "warnings": list(self.warnings), "verification": dict(self.verification),
            "bootstrap": dict(self.bootstrap),
        }
        record["sha256"] = canonical_hash(record)
        return record

    @classmethod
    def from_dict(cls, data) -> "CalibrationResult":
        """Rebuild a result written by ``to_dict``; the schema and the integrity hash are checked."""
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
            raise ValueError(f"A calibration record needs schema {SCHEMA}")
        body = {key: value for key, value in data.items() if key != "sha256"}
        if data.get("sha256") != canonical_hash(body):
            raise ValueError("The calibration record's sha256 does not match its content; it was changed after calibration")
        if data.get("selection_criterion") not in CRITERIA or not isinstance(data.get("families"), Mapping):
            raise ValueError(f"The calibration record must name its selection_criterion ({', '.join(CRITERIA)}) and "
                             "its families; a record without them was written before the criterion was recorded, so "
                             "calibrate again")
        rules = tuple(data["rules"])
        selected = {rule: data["selected"][rule] for rule in rules}
        for rule in rules:
            if rule not in RULES or (_grid(rule, [selected[rule]])[0] != selected[rule]):
                raise ValueError(f"The calibration record has an invalid {rule} threshold")
        return cls(
            profile=data["profile"], base_profile=data["base_profile"],
            configuration_sha256=data["configuration_sha256"], environment_id=data["environment_id"],
            environment_configuration=dict(data["environment_configuration"]),
            environment_configuration_sha256=data["environment_configuration_sha256"],
            calendar_steps=int(data["calendar_steps"]), seed=data["seed_derivation"]["root"],
            seeds=tuple(int(seed) for seed in data["seeds"]),
            skipped_seeds=tuple(dict(row) for row in data["seed_derivation"]["skipped"]),
            excluded_seed_count=int(data["excluded_seed_count"]), rules=rules,
            grids={rule: tuple(values) for rule, values in data["grids"].items()},
            grid_sources=dict(data["grid_sources"]), selected=selected,
            table={rule: tuple(dict(row) for row in rows) for rule, rows in data["table"].items()},
            defaults={rule: dict(row) for rule, row in data["defaults"].items()},
            warnings=tuple(data["warnings"]), verification=dict(data["verification"]),
            bootstrap=dict(data["bootstrap"]), criterion=data["selection_criterion"], families=dict(data["families"]))

    def to_markdown(self) -> str:
        level = f"{100 * self.bootstrap['confidence']:g}%"
        origin = (f"root seed {self.seed}, calibration stream {CALIBRATION_STREAM}" if self.seed is not None
                  else "explicit seeds")
        present = sum(1 for count in self.families.values() if count)
        if self.criterion == "zero_failure":
            selection = ("Selection by the zero-failure criterion: among the candidates without a failure and with a "
                         "planned exchange on these lives, the highest family-equal profit, the mean of the per-family "
                         f"pooled profits over the {present} wear {'family' if present == 1 else 'families'} in the "
                         "bank; equal values go to the lower "
                         "mean discarded life of the planned exchanges, then to the later-acting threshold.")
        else:
            selection = ("Selection by pooled profit per 1,000 calendar steps; equal profit goes to fewer failures, "
                         "then to the later-acting threshold.")
        lines = [f"Rule calibration on `{self.profile}`: {len(self.seeds):,} calibration lives ({origin}), "
                 f"up to {self.calendar_steps:,} calendar steps each.", selection, "",
                 f"| Rule | Grid | Candidates | Without a failure | Selected | Family-equal profit per 1,000 calendar "
                 f"steps | Pooled profit per 1,000 calendar steps | Failures per 1,000 lives | Profile default "
                 f"| Selected minus default, pooled ({level} interval) |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for rule in self.rules:
            row = self.row(rule)
            grid = self.grids[rule]
            default = self.defaults.get(rule)
            if default is None:
                default_cell, difference = "n/a", "n/a"
            else:
                default_cell = _value(default["threshold"])
                difference = (f"{default['difference_per_1000_calendar_steps']:+,.1f} "
                              f"[{default['interval_low']:+,.1f}, {default['interval_high']:+,.1f}]")
            failures = "n/a" if row["failures_per_1000_lives"] is None else f"{row['failures_per_1000_lives']:,.1f}"
            without = sum(1 for candidate in self.table[rule] if candidate["failures"] == 0)
            lines.append(f"| `{rule}` | {_value(grid[0])} to {_value(grid[-1])} | {len(grid)} | {without} | "
                         f"{_value(self.selected[rule])} | {row['family_equal_profit_per_1000_calendar_steps']:,.1f} | "
                         f"{row['pooled_profit_per_1000_calendar_steps']:,.1f} | "
                         f"{failures} | {default_cell} | {difference} |")
        notes = [f"Warning: {text}" for text in self.warnings]
        if self.verification.get("performed"):
            notes.append(f"The selected thresholds were re-run through the full simulator on all "
                         f"{self.verification['scenarios']:,} calibration lives with identical results.")
        if self.excluded_seed_count:
            notes.append(f"{self.excluded_seed_count:,} excluded test seeds were checked; "
                         f"{len(self.skipped_seeds)} derived seeds were skipped.")
        if notes:
            lines += ["", *[f"- {note}" for note in notes]]
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()


def _value(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def load_calibration(path) -> CalibrationResult:
    """Read ``calibration.json`` written by ``calibrate`` or the ``calibrate`` command."""
    with Path(path).open(encoding="utf-8") as stream:
        try:
            data = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not a JSON calibration record: {error}") from error
    return CalibrationResult.from_dict(data)


def calibrate(profile=DEFAULT_PROFILE, *, rules=RULES, grids=None, criterion=DEFAULT_CRITERION, seed=0,
              lives=DEFAULT_LIVES, seeds=None, exclude_seeds=(), calendar_steps=720, verify=True,
              bootstrap_samples=2000, bootstrap_seed=0, confidence=0.95) -> CalibrationResult:
    """Select the core rule thresholds on calibration lives.

    profile: a core profile ID or a profile resolved with ``resolve_profile``; overrides of
        ``env_config`` define the task that the rules are calibrated for.
    rules: a selection of ``age``, ``health`` and ``projection``.
    grids: optional mapping from rule to candidate thresholds; other rules use ``default_grids(L)``
        for the configuration's preparation time ``L``.
    criterion: ``"zero_failure"`` (default): among the candidates without a failure and with a
        planned exchange, the highest family-equal profit; raises ``NoEligibleThresholdError``
        when a rule has no such candidate. ``"pooled"``: the highest pooled profit, failures
        allowed. ``CRITERION_TEXTS`` and ``TIE_BREAK_TEXTS`` define both.
    seed, lives: root seed and size of the calibration bank (see ``calibration_seeds``).
    seeds: an explicit list of calibration seeds instead of ``seed`` and ``lives``.
    exclude_seeds: test seeds that the calibration bank must not contain. Derived seeds are
        skipped; an explicit ``seeds`` list that contains one is rejected.
    verify: re-run the selected thresholds through the full simulator and require identical results.

    A selected threshold at the smallest or largest grid value raises a ``GridEdgeWarning``
    and is listed in ``warnings``; under the zero-failure criterion, a bank without a life of a
    wear family that the configuration samples raises a ``FamilyCoverageWarning``. The
    comparison with the profile's ``policy_defaults`` is a paired percentile bootstrap over
    the calibration lives of the pooled-profit difference.
    """
    resolved = _resolve(profile)
    rules = _rules(rules)
    criterion = _criterion(criterion)
    grids, grid_sources = _grids(rules, grids, validate_config(resolved["env_config"])["mobilization_lead"])
    for rule in rules:
        if rule not in resolved["policies"]:
            raise ValueError(f"{resolved['base_profile']} has no {rule} rule")
    calendar_steps = evaluation._integer(calendar_steps, "calendar_steps", 1)
    bootstrap_samples = evaluation._integer(bootstrap_samples, "bootstrap_samples", 1)
    bootstrap_seed = evaluation._integer(bootstrap_seed, "bootstrap_seed")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    excluded = _seed_set(exclude_seeds, "excluded seed")
    skipped = []
    if seeds is not None:
        bank = [evaluation._integer(value, "calibration seed") for value in _ordered(seeds)]
        if not bank:
            raise ValueError("seeds must contain at least one calibration seed")
        if len(set(bank)) != len(bank):
            raise ValueError("Duplicate calibration seeds cannot be counted as independent lives")
        overlap = sorted(set(bank) & excluded)
        if overlap:
            raise ValueError(f"{len(overlap)} calibration seeds are also excluded test seeds, e.g. {overlap[0]}; "
                             "calibration and test lives must be disjoint")
        root = None
    else:
        root = evaluation._integer(seed, "calibration seed")
        bank, skipped = _derive(root, lives, excluded)
    lives_data, constants = _record_lives(resolved, bank, calendar_steps)
    macro = _macro_profits(constants)
    lead = macro[3]
    families = [life.family for life in lives_data]
    counts = {family: families.count(family) for family in (*FAMILY_IDS, *_family_order(families))}
    scored, rows, columns, table, selected = {}, {}, {}, {}, {}
    for rule in rules:
        grid = grids[rule]
        scored[rule] = _score_grid(rule, lives_data, grid, macro)
        summaries = _summaries(scored[rule], lead)
        metrics = _criterion_metrics(scored[rule], families, summaries)
        rows[rule] = [{**summaries[j], **metrics[j]} for j in range(len(grid))]
        order = _select(rule, grid, rows[rule], criterion)
        if criterion == "zero_failure" and not rows[rule][order[0]]["zero_failure_eligible"]:
            raise NoEligibleThresholdError(_no_eligible_message(rule, grid, rows[rule], len(lives_data)))
        columns[rule] = order[0]
        selected[rule] = grid[order[0]]
        ranks = {column: rank + 1 for rank, column in enumerate(order)}
        table[rule] = tuple({"threshold": grid[j], **rows[rule][j], "rank": ranks[j], "selected": j == order[0]}
                            for j in range(len(grid)))
    notes = []
    for rule in rules:
        grid = grids[rule]
        if len(grid) == 1:
            notes.append((f"The {rule} grid has the single value {_value(grid[0])}; nothing was searched.",
                          GridEdgeWarning))
        elif selected[rule] in (grid[0], grid[-1]):
            side = "smallest" if selected[rule] == grid[0] else "largest"
            notes.append((f"The selected {rule} threshold {_value(selected[rule])} is the {side} value of its grid "
                          f"({_value(grid[0])} to {_value(grid[-1])}); the optimum may lie outside the grid. "
                          "Widen the grid and calibrate again.", GridEdgeWarning))
    missing = [family for family in FAMILY_IDS if not counts[family]]
    if criterion == "zero_failure" and missing and validate_config(resolved["env_config"])["family"] == "random":
        present = len(FAMILY_IDS) - len(missing)
        notes.append((f"The calibration lives contain no life of the wear "
                      f"{'family' if len(missing) == 1 else 'families'} {', '.join(missing)}, which the configuration "
                      f"samples; the family-equal profit averages the {present} "
                      f"{'family' if present == 1 else 'families'} present. Use more calibration lives.",
                      FamilyCoverageWarning))
    for text, category in notes:
        warnings.warn(text, category, stacklevel=2)
    notes = [text for text, _ in notes]
    defaults = {}
    for rule in rules:
        setting = resolved["policy_defaults"].get(rule, {})
        if PARAMETERS[rule] not in setting:
            continue
        threshold = setting[PARAMETERS[rule]]
        if rule == "age":
            threshold = int(threshold)
        grid = grids[rule]
        default_scored = _score_grid(rule, lives_data, (threshold,), macro)
        default_summary = _summaries(default_scored, lead)[0]
        default_metrics = _criterion_metrics(default_scored, families, [default_summary])[0]
        column = columns[rule]
        chosen = rows[rule][column]
        left = (scored[rule]["raw"][:, column], scored[rule]["elapsed"][:, column].astype(np.float64))
        right = (default_scored["raw"][:, 0], default_scored["elapsed"][:, 0].astype(np.float64))
        low, high = _paired_interval(left, right, samples=bootstrap_samples, seed=bootstrap_seed, confidence=confidence)
        on_grid = threshold in grid
        defaults[rule] = {
            "parameter": PARAMETERS[rule], "threshold": threshold, "on_grid": on_grid,
            "grid_steps_from_selected": abs(grid.index(threshold) - column) if on_grid else None,
            "pooled_profit_per_1000_calendar_steps": default_summary["pooled_profit_per_1000_calendar_steps"],
            "failures": default_summary["failures"],
            "difference_per_1000_calendar_steps": (chosen["pooled_profit_per_1000_calendar_steps"]
                                                   - default_summary["pooled_profit_per_1000_calendar_steps"]),
            "interval_low": low, "interval_high": high, "confidence": confidence,
            "family_equal_profit_per_1000_calendar_steps": default_metrics["family_equal_profit_per_1000_calendar_steps"],
            "family_equal_difference_per_1000_calendar_steps": (
                chosen["family_equal_profit_per_1000_calendar_steps"]
                - default_metrics["family_equal_profit_per_1000_calendar_steps"]),
            "zero_failure_eligible": default_metrics["zero_failure_eligible"]}
    verification = (_verify(resolved, rules, selected, bank, calendar_steps, scored, columns) if verify
                    else {"performed": False})
    configuration = evaluation._environment_configuration(resolved, calendar_steps)
    return CalibrationResult(
        profile=resolved["id"], base_profile=resolved["base_profile"],
        configuration_sha256=resolved["configuration_sha256"], environment_id=resolved["environment_id"],
        environment_configuration=copy.deepcopy(configuration),
        environment_configuration_sha256=canonical_hash(configuration), calendar_steps=calendar_steps,
        seed=root, seeds=tuple(bank), skipped_seeds=tuple(skipped), excluded_seed_count=len(excluded),
        rules=rules, grids=grids, grid_sources=grid_sources, selected=selected, table=table, defaults=defaults,
        warnings=tuple(notes), verification=verification,
        bootstrap={"method": "paired percentile bootstrap over calibration lives of the pooled-profit difference",
                   "samples": bootstrap_samples, "seed": bootstrap_seed, "confidence": confidence},
        criterion=criterion, families=counts)


def _ordered(values):
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValueError("seeds must be a sequence of nonnegative integers")
    return list(values)


__all__ = ["CALIBRATION_STREAM", "CRITERIA", "CRITERION_TEXTS", "CalibrationResult", "CalibrationWarning",
           "DEFAULT_CRITERION", "DEFAULT_LIVES", "FamilyCoverageWarning", "GridEdgeWarning", "NoEligibleThresholdError",
           "PARAMETERS", "RULES", "SCHEMA", "TIE_BREAK_TEXTS", "calibrate", "calibration_seeds", "default_grids",
           "load_calibration", "parse_grid"]
