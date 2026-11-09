"""Paired evaluation of policies on freshly generated synthetic scenarios.

Built-in policies are named by the profile. A user agent is a plain callable
``agent(observation, info) -> action``. The harness calls it with copies of the
observation and of the public ``info`` mapping and with nothing else: never the
environment, ``diagnostics()`` or ``snapshot()``.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping
import copy
import csv
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

from .policies import Policy
from .records import canonical_hash, json_value, write_json


EVALUATION_SCHEMA = "maintenance_arena_synthetic_evaluation_v3"
FIXTURE_SCHEMA = "maintenance_arena_synthetic_fixture_v2"
SCOPE = ("Synthetic evaluation on freshly sampled scenarios; it does not repeat the runs reported "
         "in the accompanying article.")
AGGREGATION = ("mean_profit_per_1000_calendar_steps is the equal-weight mean of the per-scenario rates; "
               "pooled_profit_per_1000_calendar_steps is total cash over total calendar steps. Scenarios "
               "are the units; independently trained model seeds are not inferred, and no family-equal "
               "aggregation is applied.")
METRIC_DEFINITIONS = {
    "pooled_profit_per_1000_calendar_steps": "1000 * sum(raw_profit) / sum(actual_calendar_steps) over the scenarios.",
    "mean_profit_per_1000_calendar_steps": "Mean over the scenarios of each scenario's profit_per_1000_calendar_steps.",
    "ended_lives": ("Component lives that ended within the evaluated calendar steps: planned (preventive) "
                    "interventions plus failures. Lives still running at the cutoff are not counted."),
    "failures_per_1000_lives": "1000 * failures / ended_lives; null when no life ended.",
    "failures_per_1000_calendar_steps": "1000 * failures / sum(actual_calendar_steps).",
    "order_margin": ("Core only, evaluator side: true RUL at the scenario's first order minus 13, the last safe "
                     "order (lead time 12 plus the exchange step). Negative margins are late; null without an order."),
    "orders_placed": "Core only: scenarios with at least one order.",
    "late_orders": "Core only: scenarios whose first order has a negative order margin.",
    "late_order_share": "Core only: late_orders / orders_placed; null without orders.",
    "order_margin_mean": "Core only: mean order margin over orders_placed.",
    "order_margin_sd": "Core only: sample standard deviation (ddof=1) of the order margin; null below two orders.",
    "invalid_actions": "Resource tasks only: masked (infeasible) actions that the environment executed as no action.",
    "policy_kind": ("built_in (named by the profile), agent (user callable that receives only observation and info) "
                    "or privileged_reference (evaluator-side reference with access to simulator truth)."),
}

# A user agent: called as agent(observation, info) and returning one integer action.
Agent = Callable[[np.ndarray, dict], Any]
BUILT_IN, AGENT, PRIVILEGED_REFERENCE = "built_in", "agent", "privileged_reference"
# Labels that belong to evaluator-side references and cannot name a user agent.
RESERVED_LABELS = frozenset({"oracle"})


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _environment_configuration(profile, calendar_steps=None):
    configuration = dict(profile["env_config"])
    if calendar_steps is not None and "Resource" in profile["environment_id"]:
        configuration["horizon"] = _integer(calendar_steps, "calendar_steps", 1)
    return configuration


def _fixture_contract(profile, calendar_steps):
    # Policies, filter use and optimizer settings may change in a paired
    # comparison. The environment, observation route and exposure must agree.
    configuration = _environment_configuration(profile, calendar_steps)
    return {"environment_id": profile["environment_id"],
            "environment_configuration_sha256": canonical_hash(configuration),
            "calendar_steps": calendar_steps}


def make_fixture(profile, seeds, *, calendar_steps):
    """Bind explicit scenario seeds to a profile's environment and calendar budget.

    The result is the ``fixture`` that ``evaluate`` writes and accepts, so a fixed
    seed list can be replayed for any set of policies and agents.
    """
    calendar_steps = _integer(calendar_steps, "calendar_steps", 1)
    if isinstance(seeds, (str, bytes)) or not hasattr(seeds, "__iter__"):
        raise ValueError("seeds must be a sequence of nonnegative integers")
    values = [_integer(seed, "scenario seed") for seed in seeds]
    if not values:
        raise ValueError("seeds must contain at least one scenario seed")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate scenario seeds cannot be counted as independent synthetic scenarios")
    return {"schema": FIXTURE_SCHEMA, "profile": profile["base_profile"],
            "scenario_contract": _fixture_contract(profile, calendar_steps), "seeds": values}


def make_resolved(profile, calendar_steps=None):
    configuration = _environment_configuration(profile, calendar_steps)
    env = gym.make(profile["environment_id"], config=configuration)
    env.unwrapped.arena_profile = profile
    return env


class _Entry:
    """One evaluated policy: its label, kind and a builder called once per scenario."""

    __slots__ = ("label", "kind", "build")

    def __init__(self, label, kind, build):
        self.label, self.kind, self.build = label, kind, build


class _AgentChooser:
    """Call a user agent with copies of (observation, info) only."""

    def __init__(self, label, agent):
        self.label, self._agent = label, agent

    def __call__(self, observation, info):
        value = self._agent(np.array(observation, copy=True), copy.deepcopy(dict(info)))
        return _agent_action(value, self.label)

    def close(self):
        pass


def _agent_action(value, label="agent"):
    """Convert an agent's return value to one integer action, or explain why it is not one."""
    array = np.asarray(value)
    if isinstance(value, (bool, np.bool_)) or array.size != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"Agent {label!r} returned {value!r}; an agent must return one integer action")
    return int(array.reshape(()))


def _built_in_entry(profile, name, checkpoint):
    return _Entry(name, BUILT_IN, lambda env: Policy(name, profile, env, checkpoint))


def _agent_entry(profile, item):
    if not isinstance(item, tuple) or len(item) != 2:
        raise ValueError("Agents must be given as (label, agent) tuples or as a mapping from label to agent")
    label, agent = item
    if not isinstance(label, str) or not label.strip() or label != label.strip():
        raise ValueError("Agent labels must be nonempty strings without surrounding whitespace")
    if label in profile["policies"] or label in RESERVED_LABELS:
        raise ValueError(f"Agent label {label!r} names a built-in policy or reference of "
                         f"{profile['base_profile']}; choose another label")
    if not callable(agent):
        raise ValueError(f"Agent {label!r} must be callable as agent(observation, info) -> action")
    return _Entry(label, AGENT, lambda env: _AgentChooser(label, agent))


def _agent_items(agents):
    if agents is None:
        return []
    if isinstance(agents, Mapping):
        return list(agents.items())
    if isinstance(agents, (list, tuple)):
        return list(agents)
    raise ValueError("agents must be a mapping from label to agent or a sequence of (label, agent) tuples")


def _entries(profile, policies, agents=None, *, checkpoint=None):
    """Resolve built-in names and user agents into evaluation entries, in the given order."""
    if not isinstance(policies, (list, tuple)):
        raise ValueError("policies must be a sequence of policy names or (label, agent) tuples")
    entries = []
    for item in policies:
        if isinstance(item, str):
            entries.append(_built_in_entry(profile, item, checkpoint))
        else:
            entries.append(_agent_entry(profile, item))
    entries.extend(_agent_entry(profile, item) for item in _agent_items(agents))
    if not entries:
        raise ValueError("policies must be a nonempty sequence of policy names, or agents must name at least one agent")
    labels = [entry.label for entry in entries]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate policies would repeat observations and inflate scenario counts")
    return entries


def _single_entry(profile, policy, checkpoint):
    if isinstance(policy, str):
        return _built_in_entry(profile, policy, checkpoint)
    return _agent_entry(profile, policy)


def rollout(profile, *, policy, seed, calendar_steps, checkpoint=None, apply_filter=False, trace=False,
            events=True):
    """Run one policy on one scenario; ``policy`` is a built-in name or a (label, agent) tuple."""
    return _rollout(profile, _single_entry(profile, policy, checkpoint), seed=seed,
                    calendar_steps=calendar_steps, apply_filter=apply_filter, trace=trace, events=events)


def _core_order_margin(env, diagnostic):
    rul = diagnostic.get("first_trigger_true_rul")
    if rul is None:
        return None
    # The last safe order is the lead time plus the exchange step (13 steps).
    return int(rul) - (int(env.unwrapped.config["mobilization_lead"]) + 1)


def _rollout(profile, entry, *, seed, calendar_steps, apply_filter=False, trace=False, events=True):
    calendar_steps = _integer(calendar_steps, "calendar_steps", 1)
    seed = _integer(seed, "seed")
    if apply_filter and "Gearbox" not in profile["environment_id"]:
        raise ValueError("The execution filter is defined only for the frozen gearbox contract")
    env = make_resolved(profile, calendar_steps)
    environment_configuration = _environment_configuration(profile, calendar_steps)
    environment_budget = environment_configuration.get("calendar_horizon", environment_configuration.get("horizon"))
    effective_horizon = min(calendar_steps, environment_budget) if environment_budget is not None else calendar_steps
    choose = None
    rows, recorded, total, elapsed, corrections, decisions = [], [], 0.0, 0, 0, 0
    try:
        observation, info = env.reset(seed=int(seed))
        choose = entry.build(env)
        terminated = truncated = False
        while not (terminated or truncated) and elapsed < calendar_steps:
            before = np.asarray(observation).copy()
            proposal = choose(observation, info)
            action = proposal
            if apply_filter:
                from .gearbox import filtered_action
                action = int(filtered_action(env.unwrapped, proposal))
            observation, reward, terminated, truncated, info = env.step(action)
            duration = int(info["duration_calendar_steps"])
            if duration <= 0:
                raise ArithmeticError("Environment returned a nonpositive duration")
            raw = float(info["raw_profit_sum"])
            if not np.isfinite(raw) or not np.isfinite(reward):
                raise ArithmeticError("Nonfinite reward or raw profit")
            total += raw
            corrections += int(action != proposal)
            decisions += 1
            if events:
                for event in info.get("events", []):
                    recorded.append(json_value(event))
            if trace:
                rows.append({"decision": decisions - 1, "calendar_start": elapsed,
                    "calendar_end": elapsed + duration, "observation": before,
                    "next_observation": observation, "proposed_action": proposal, "executed_action": action,
                    "duration_calendar_steps": duration, "reward": float(reward), "raw_profit": raw,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "ledger": info.get("ledger", {}), "events": info.get("events", [])})
            elapsed += duration
        diagnostic = env.unwrapped.diagnostics() if hasattr(env.unwrapped, "diagnostics") else {}
        if not trace:
            diagnostic.pop("trace", None)
        public_metrics = info.get("metrics", {})
        cumulative = diagnostic.get("cumulative_metrics", {})
        failures = int(diagnostic.get("failures", public_metrics.get("failures", info.get("failures", 0))))
        planned = float(diagnostic.get("planned_replacements", public_metrics.get("planned_starts", cumulative.get("planned_renewals", 0))))
        corrective = float(diagnostic.get("corrective_replacements", public_metrics.get("corrective_starts", cumulative.get("corrective_renewals", 0))))
        core = "CoreCycle" in profile["environment_id"]
        resource = "Resource" in profile["environment_id"]
        return {
            "policy": entry.label, "policy_kind": entry.kind, "seed": int(seed), "filtered": bool(apply_filter),
            "nominal_calendar_steps": int(calendar_steps), "actual_calendar_steps": elapsed,
            "budget_overshoot": max(0, elapsed - calendar_steps), "decisions": decisions,
            "raw_profit": total, "profit_per_1000_calendar_steps": 1000 * total / elapsed,
            "failures": failures, "planned_interventions": planned, "corrective_interventions": corrective,
            "ended_lives": int(round(planned)) + failures,
            "discarded_operating_steps": diagnostic.get("discarded_operating_steps"),
            "normalized_discarded_life": diagnostic.get("normalized_discarded_life"),
            "first_trigger_true_rul": diagnostic.get("first_trigger_true_rul"),
            "order_margin": _core_order_margin(env, diagnostic) if core else None,
            "invalid_actions": int(public_metrics.get("invalid_actions", 0)) if resource else None,
            "effective_accounting_horizon": effective_horizon,
            "environment_configuration": environment_configuration,
            "environment_calendar_budget": environment_budget,
            "environment_budget_overshoot": max(0, elapsed - environment_budget) if environment_budget is not None else 0,
            "corrected_proposals": corrections, "intervention_fraction": corrections / decisions,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "accounting_cutoff": elapsed >= effective_horizon and not terminated,
            "final_public_metrics": info.get("metrics", {}), "analysis_diagnostics": diagnostic,
            "trace": rows, "events": recorded if events else None,
        }
    finally:
        if choose is not None:
            choose.close()
        env.close()


def evaluate(profile, *, policies=(), agents=None, seed=None, episodes, calendar_steps, checkpoint=None,
             apply_filter=False, fixture=None, events=True):
    """Run built-in policies and user agents on identical scenarios.

    ``policies`` holds built-in names and optionally (label, agent) tuples;
    ``agents`` adds agents as a mapping from label to callable or as (label, agent)
    tuples. ``events=False`` omits the per-scenario event lists from the units.
    ``episodes`` scenario seeds are derived from ``seed``; a ``fixture`` supplies the
    scenario seeds instead, and then ``seed`` may be omitted.
    """
    entries = _entries(profile, policies, agents, checkpoint=checkpoint)
    return _evaluate_entries(profile, entries, seed=seed, episodes=episodes, calendar_steps=calendar_steps,
                             apply_filter=apply_filter, fixture=fixture, events=events)


def _scenario_seeds(profile, *, seed, episodes, calendar_steps, fixture):
    contract = _fixture_contract(profile, calendar_steps)
    if fixture is None:
        seeds = [int(np.random.SeedSequence([int(seed), i]).generate_state(1, dtype=np.uint32)[0]) for i in range(episodes)]
    else:
        if not isinstance(fixture, dict) or fixture.get("schema") != FIXTURE_SCHEMA:
            raise ValueError("Fixture requires schema maintenance_arena_synthetic_fixture_v2 with environment and calendar bindings; regenerate unbound v1 fixtures explicitly")
        if fixture.get("profile") != profile["base_profile"] or fixture.get("scenario_contract") != contract:
            raise ValueError("Fixture environment configuration or calendar budget is incompatible")
        supplied = fixture.get("seeds")
        if not isinstance(supplied, list) or not supplied:
            raise ValueError("Fixture must contain a nonempty list of nonnegative integer seeds")
        seeds = [_integer(s, "fixture seed") for s in supplied]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate scenario seeds cannot be counted as independent synthetic scenarios")
    return contract, seeds


def _summary(entry, selected, core, resource):
    raw_profit = sum(r["raw_profit"] for r in selected)
    calendar = sum(r["actual_calendar_steps"] for r in selected)
    failures = sum(r["failures"] for r in selected)
    ended = sum(r["ended_lives"] for r in selected)
    summary = {"policy": entry.label, "independent_synthetic_scenarios": len(selected),
        "mean_profit_per_1000_calendar_steps": float(np.mean([r["profit_per_1000_calendar_steps"] for r in selected])),
        "raw_profit": raw_profit,
        "actual_calendar_steps": calendar,
        "failures": failures,
        "planned_interventions": sum(r["planned_interventions"] for r in selected),
        "corrective_interventions": sum(r["corrective_interventions"] for r in selected),
        "mean_normalized_discarded_life": (float(np.mean([r["normalized_discarded_life"] for r in selected if r["normalized_discarded_life"] is not None]))
            if any(r["normalized_discarded_life"] is not None for r in selected) else None),
        "corrected_proposals": sum(r["corrected_proposals"] for r in selected),
        # Fields added with evaluation schema v3; the fields above keep their v2 meaning.
        "policy_kind": entry.kind,
        "pooled_profit_per_1000_calendar_steps": 1000 * raw_profit / calendar,
        "ended_lives": ended,
        "failures_per_1000_lives": 1000 * failures / ended if ended else None,
        "failures_per_1000_calendar_steps": 1000 * failures / calendar,
        "orders_placed": None, "late_orders": None, "late_order_share": None,
        "order_margin_mean": None, "order_margin_sd": None,
        "invalid_actions": sum(r["invalid_actions"] for r in selected) if resource else None}
    if core:
        margins = np.asarray([r["order_margin"] for r in selected if r["order_margin"] is not None], dtype=float)
        late = int(np.sum(margins < 0))
        summary.update(orders_placed=int(margins.size), late_orders=late,
                       late_order_share=late / margins.size if margins.size else None,
                       order_margin_mean=float(np.mean(margins)) if margins.size else None,
                       order_margin_sd=float(np.std(margins, ddof=1)) if margins.size > 1 else None)
    return summary


def _evaluate_entries(profile, entries, *, seed, episodes, calendar_steps, apply_filter=False, fixture=None, events=True):
    """Evaluate resolved entries on identical scenarios (also used by ``benchmark``)."""
    episodes = _integer(episodes, "episodes", 1)
    if seed is None and fixture is None:
        raise ValueError("seed is required unless a fixture supplies the scenario seeds")
    seed = None if seed is None else _integer(seed, "seed")
    calendar_steps = _integer(calendar_steps, "calendar_steps", 1)
    contract, seeds = _scenario_seeds(profile, seed=seed, episodes=episodes, calendar_steps=calendar_steps, fixture=fixture)
    rows = []
    for entry in entries:
        for unit, episode_seed in enumerate(seeds):
            row = _rollout(profile, entry, seed=episode_seed, calendar_steps=calendar_steps,
                           apply_filter=apply_filter, trace=False, events=events)
            row.pop("trace")
            row["paired_unit"] = unit
            rows.append(row)
    core = "CoreCycle" in profile["environment_id"]
    resource = "Resource" in profile["environment_id"]
    summaries = [_summary(entry, [r for r in rows if r["policy"] == entry.label], core, resource) for entry in entries]
    return {"schema": EVALUATION_SCHEMA, "scope": SCOPE,
            "fixture": {"schema": FIXTURE_SCHEMA, "profile": profile["base_profile"],
                        "scenario_contract": contract, "seeds": seeds},
            "aggregation": AGGREGATION, "metric_definitions": dict(METRIC_DEFINITIONS),
            "unit_events": "recorded" if events else "omitted",
            "summary": summaries, "units": rows}


def write_trace(directory, result):
    write_json(Path(directory) / "trajectory.json", result)
    rows = result["trace"]
    if rows:
        with (Path(directory) / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(json_value(value), allow_nan=False) if isinstance(value, (list, dict, np.ndarray)) else value for key, value in row.items()})
