"""Grid-search calibration of the core rules, its two selection criteria, its seed split, its JSON record,
the calibrate command and the recalibration of benchmark references."""
import json
import statistics
import warnings

import numpy as np
import pytest

from maintenance_arena import benchmark, calibration
from maintenance_arena.calibration import CalibrationWarning, FamilyCoverageWarning, GridEdgeWarning, calibrate
from maintenance_arena.cli import main
from maintenance_arena.evaluation import _Entry, BUILT_IN, _evaluate_entries, evaluate, make_fixture
from maintenance_arena.policies import Policy
from maintenance_arena.profiles import resolve_profile


CORE = "paper-core-continuous-v1"
SMALL = {"age": (70, 80, 83, 86, 95), "health": (0.30, 0.33, 0.36, 0.40), "projection": (13, 14, 15, 16, 17)}


def _evaluation_seeds(root, count):
    return [int(np.random.SeedSequence([root, i]).generate_state(1, dtype=np.uint32)[0]) for i in range(count)]


def _quiet_calibrate(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        return calibrate(*args, **kwargs)


# Grids and seeds.

def test_default_grids_are_the_252_candidate_roster():
    grids = calibration.default_grids()
    assert grids["age"] == tuple(range(1, 160)) and all(type(value) is int for value in grids["age"])
    assert grids["health"] == tuple(integer / 100.0 for integer in range(20, 96))
    assert grids["health"][16] == 0.36 and grids["health"][0] == 0.2 and grids["health"][-1] == 0.95
    assert grids["projection"] == tuple(range(8, 25))
    assert sum(len(values) for values in grids.values()) == 252
    grids["age"] = ()
    assert len(calibration.default_grids()["age"]) == 159


def test_parse_grid_reads_numbers_and_inclusive_ranges():
    assert calibration.parse_grid("age", ["1:5", "9"]) == (1, 2, 3, 4, 5, 9)
    assert calibration.parse_grid("health", ["0.20:0.95:0.01"]) == calibration.default_grids()["health"]
    assert calibration.parse_grid("projection", ["8:24"]) == calibration.default_grids()["projection"]
    assert calibration.parse_grid("projection", ["14.5", "15,16"]) == (14.5, 15, 16)
    for rule, tokens, message in [("age", ["5:1"], "START <= STOP"), ("age", ["1:2:0"], "positive STEP"),
                                  ("age", ["x"], "not a number"), ("age", ["1:2:3:4"], "START:STOP"),
                                  ("age", ["2.5"], "whole operating steps"), ("health", ["1.5"], r"\[0, 1\]"),
                                  ("projection", ["3", "3"], "duplicate"), ("age", ["-1"], "nonnegative")]:
        with pytest.raises(ValueError, match=message):
            calibration.parse_grid(rule, tokens)


def test_calibration_seeds_follow_the_documented_stream_and_avoid_evaluation_seeds():
    seeds = calibration.calibration_seeds(20261002, 800)
    assert seeds == [int(np.random.SeedSequence(20261002, spawn_key=(1, i)).generate_state(1, dtype=np.uint32)[0])
                     for i in range(800)]
    assert len(set(seeds)) == 800 and seeds == calibration.calibration_seeds(20261002, 800)
    # The same root gives different test scenarios in evaluate and benchmark.
    assert not set(seeds) & set(_evaluation_seeds(20261002, 800))
    excluded = {seeds[1], seeds[5]}
    bank = calibration.calibration_seeds(20261002, 800, exclude=excluded)
    assert len(bank) == len(set(bank)) == 800 and not set(bank) & excluded
    assert bank[:5] == [seeds[0], seeds[2], seeds[3], seeds[4], seeds[6]]
    with pytest.raises(ValueError, match="lives must be an integer >= 1"):
        calibration.calibration_seeds(1, 0)


def test_calibration_and_test_seeds_are_disjoint():
    derived = calibration.calibration_seeds(7, 6)
    result = _quiet_calibrate(CORE, grids=SMALL, seed=7, lives=6, exclude_seeds=[derived[0], derived[3]], verify=False)
    assert len(result.seeds) == 6 and not set(result.seeds) & {derived[0], derived[3]}
    assert [row["reason"] for row in result.skipped_seeds] == ["excluded test seed", "excluded test seed"]
    assert [row["index"] for row in result.skipped_seeds] == [0, 3]
    assert result.excluded_seed_count == 2
    with pytest.raises(ValueError, match="must be disjoint"):
        calibrate(CORE, grids=SMALL, seeds=[1, 2, 3], exclude_seeds=[3], verify=False)
    with pytest.raises(ValueError, match="Duplicate calibration seeds"):
        calibrate(CORE, grids=SMALL, seeds=[1, 1], verify=False)


# Scoring and selection.

EXACTNESS_CASES = [
    (CORE, None, 720),
    (CORE, {"env_config": {"revenue_per_operating_step": 255.3, "replacement_cost": 517.7,
                           "downtime_cost_per_step": 25.1, "failure_event_cost": 7003.9}}, 720),
    (CORE, {"env_config": {"sensor_level": "stress", "observation_route": "FILTERED_CAUSAL_EWLS8"}}, 720),
    (CORE, {"env_config": {"calendar_horizon": 110, "training_start_offsets": [20]}}, 90),
    ("paper-core-discrete-v1", {"env_config": {"family": "bearing_late_knee", "eol_operating_age": 40}}, 720),
]


@pytest.mark.parametrize("profile_id,overrides,calendar_steps", EXACTNESS_CASES)
@pytest.mark.parametrize("recovery_contract", ["immediate_exchange_v1", "preserve_preparation_v1"])
def test_every_candidate_scores_exactly_as_the_full_simulator(profile_id, overrides, calendar_steps, recovery_contract):
    if recovery_contract == "preserve_preparation_v1":
        overrides = {"env_config": {**(overrides or {}).get("env_config", {}),
                                    "recovery_contract": recovery_contract}}
    profile = resolve_profile(profile_id, overrides)
    seeds = _evaluation_seeds(4242, 10)
    lives, constants = calibration._record_lives(profile, seeds, calendar_steps)
    macro = calibration._macro_profits(constants)
    grids = {"age": (0, 5, 26, 27, 30, 83, 100, 159, 400), "health": (0.2, 0.36, 0.5, 0.95, 1.0),
             "projection": (0, 13, 14.5, 15, 24, 40)}
    for rule, grid in grids.items():
        scored = calibration._score_grid(rule, lives, grid, macro)
        summaries = calibration._summaries(scored, macro[3])
        entries = [_Entry(f"{rule}-{j}", BUILT_IN, lambda env, p=calibration._with_thresholds(profile, {rule: value}):
                          Policy(rule, p, env)) for j, value in enumerate(grid)]
        record = _evaluate_entries(profile, entries, seed=0, episodes=1, calendar_steps=calendar_steps,
                                   fixture=make_fixture(profile, seeds, calendar_steps=calendar_steps), events=False)
        by_label = {summary["policy"]: summary for summary in record["summary"]}
        for j in range(len(grid)):
            units = sorted((u for u in record["units"] if u["policy"] == f"{rule}-{j}"), key=lambda u: u["paired_unit"])
            for i, unit in enumerate(units):
                rul = int(scored["rul"][i, j])
                assert (unit["raw_profit"], unit["actual_calendar_steps"], unit["failures"],
                        unit["planned_interventions"], unit["first_trigger_true_rul"],
                        unit["normalized_discarded_life"]) == (
                    float(scored["raw"][i, j]), int(scored["elapsed"][i, j]), int(scored["failure"][i, j]),
                    int(scored["planned"][i, j]), None if rul < 0 else rul,
                    float(scored["normalized_discarded"][i, j])), (rule, grid[j], unit["seed"])
            summary = by_label[f"{rule}-{j}"]
            for key, value in summaries[j].items():
                assert summary[key] == value, (rule, grid[j], key)


def test_pooled_selection_maximizes_pooled_profit_and_is_verified_by_the_simulator():
    result = calibrate(CORE, grids=SMALL, criterion="pooled", seed=11, lives=24)
    assert result.rules == ("age", "health", "projection") and result.criterion == "pooled"
    for rule in result.rules:
        rows = result.table[rule]
        best = max(row["pooled_profit_per_1000_calendar_steps"] for row in rows)
        selected = [row for row in rows if row["selected"]]
        assert len(selected) == 1 and selected[0]["rank"] == 1
        assert selected[0]["pooled_profit_per_1000_calendar_steps"] == best
        assert selected[0]["threshold"] == result.selected[rule] and result.row(rule) == selected[0]
        assert sorted(row["rank"] for row in rows) == list(range(1, len(rows) + 1))
        assert [row["threshold"] for row in rows] == list(SMALL[rule])
    assert result.verification["performed"] and result.verification["identical"]
    assert result.verification["simulator_pooled_profit_per_1000_calendar_steps"] == {
        rule: result.row(rule)["pooled_profit_per_1000_calendar_steps"] for rule in result.rules}
    defaults = result.policy_defaults()
    assert set(defaults) == {"age", "health", "projection"}
    assert type(defaults["age"]["age_interval"]) is int and type(defaults["projection"]["projection_threshold"]) is int
    assert result.defaults["projection"]["threshold"] == 15 and result.defaults["age"]["threshold"] == 83
    for rule, row in result.defaults.items():
        # The selected threshold is never worse than the default on the calibration lives.
        assert row["interval_low"] <= row["interval_high"] and row["difference_per_1000_calendar_steps"] >= 0


def test_the_default_zero_failure_criterion_selects_the_best_family_equal_profit_without_failures():
    result = calibrate(CORE, grids=SMALL, seed=11, lives=24)
    assert result.criterion == calibration.DEFAULT_CRITERION == "zero_failure"
    assert sum(result.families.values()) == 24 and set(result.families) == set(calibration.FAMILY_IDS)
    present = [family for family, count in result.families.items() if count]
    for rule in result.rules:
        rows = result.table[rule]
        for row in rows:
            assert row["zero_failure_eligible"] == (row["failures"] == 0
                                                  and row["mean_normalized_discarded_life_over_planned"] is not None)
            assert set(row["family_profit_per_1000_calendar_steps"]) == set(present)
            assert row["family_equal_profit_per_1000_calendar_steps"] == statistics.fmean(
                row["family_profit_per_1000_calendar_steps"].values())
        eligible = [row for row in rows if row["zero_failure_eligible"]]
        best = max(row["family_equal_profit_per_1000_calendar_steps"] for row in eligible)
        chosen = result.row(rule)
        assert chosen["selected"] and chosen["rank"] == 1 and chosen["failures"] == 0
        assert chosen["family_equal_profit_per_1000_calendar_steps"] == best
        # Eligible candidates come first in the ranking; candidates with failures follow.
        assert sorted(row["rank"] for row in eligible) == list(range(1, len(eligible) + 1))
    # On these lives a threshold with failures is excluded in every rule (age 95, health 0.30,
    # projection 13 and 14), whatever its profit.
    assert {rule: [row["threshold"] for row in result.table[rule] if not row["zero_failure_eligible"]]
            for rule in result.rules} == {"age": [95], "health": [0.3], "projection": [13, 14]}
    assert result.selected == {"age": 86, "health": 0.33, "projection": 15}
    assert result.verification["identical"]
    record = result.to_dict()
    assert record["selection_criterion"] == "zero_failure" and record["families"] == result.families
    assert record["criterion"] == calibration.CRITERION_TEXTS["zero_failure"] == calibration.CRITERION
    assert record["tie_break"] == calibration.TIE_BREAK_TEXTS["zero_failure"]
    assert "Selection by the zero-failure criterion" in result.to_markdown()


def _constructed_bank(rows, budget=720):
    """Hand-made calibration lives (family, lifetime) from age 0; health and projection are not used."""
    return [calibration._Life(index, 0, lifetime, min(budget, lifetime), budget, np.ones(lifetime),
                              np.full(lifetime, np.inf), family) for index, (family, lifetime) in enumerate(rows)]


# The core's prices with a failure that costs no more than its replacement, so that a late order can pay.
CHEAP_FAILURES = {"revenue_per_operating_step": 255.0, "replacement_cost": 520.0, "downtime_cost_per_step": 25.0,
                  "failure_event_cost": 0.0, "mobilization_lead": 12, "downtime_steps": 6}


def _criteria_rows(rule, lives, grid, constants):
    macro = calibration._macro_profits(constants)
    scored = calibration._score_grid(rule, lives, grid, macro)
    summaries = calibration._summaries(scored, macro[3])
    metrics = calibration._criterion_metrics(scored, [life.family for life in lives], summaries)
    return [{**summary, **metric} for summary, metric in zip(summaries, metrics)]


def test_family_equal_profit_and_eligibility_on_a_constructed_bank():
    # Three linear lives of 110 steps and one late-knee life of 100 steps: the families are unbalanced.
    lives = _constructed_bank([("linear_reference", 110)] * 3 + [("bearing_late_knee", 100)])
    rows = _criteria_rows("age", lives, (85, 90), CHEAP_FAILURES)
    # Age 85 orders at true RUL 25 and 15: four planned exchanges, each 97 operating steps (85 + 12 of
    # lead time), 24,040 cash in 104 calendar steps. Every life earns the same rate.
    planned_85 = 1000 * 24040 / 104
    assert rows[0]["failures"] == 0 and rows[0]["planned_interventions"] == 4
    assert rows[0]["pooled_profit_per_1000_calendar_steps"] == pytest.approx(planned_85, rel=1e-15)
    assert rows[0]["family_equal_profit_per_1000_calendar_steps"] == pytest.approx(planned_85, rel=1e-15)
    # Discarded life of the planned exchanges: 110 - 97 = 13 steps of 110, and 100 - 97 = 3 of 100.
    assert rows[0]["mean_normalized_discarded_life_over_planned"] == statistics.fmean([13 / 110] * 3 + [3 / 100])
    assert rows[0]["zero_failure_eligible"]
    # Age 90 is planned on the linear lives (25,315 cash in 109 steps) but orders at true RUL 10 on the
    # late-knee life, which fails after 9 more operating steps (22,950 + 2,295 - 545 - 150 = 24,550 cash
    # in 106 steps).
    linear, knee = 1000 * 25315 / 109, 1000 * 24550 / 106
    assert rows[1]["failures"] == 1 and rows[1]["planned_interventions"] == 3
    assert rows[1]["family_profit_per_1000_calendar_steps"] == {
        "linear_reference": pytest.approx(linear, rel=1e-15), "bearing_late_knee": pytest.approx(knee, rel=1e-15)}
    # Equal weight per family, although the linear family has three lives and the knee family one ...
    assert rows[1]["family_equal_profit_per_1000_calendar_steps"] == pytest.approx((linear + knee) / 2, rel=1e-15)
    # ... whereas the pooled profit weights every life by its calendar steps.
    pooled_90 = 1000 * (3 * 25315 + 24550) / (3 * 109 + 106)
    assert rows[1]["pooled_profit_per_1000_calendar_steps"] == pytest.approx(pooled_90, rel=1e-15)
    assert rows[1]["family_equal_profit_per_1000_calendar_steps"] < rows[1]["pooled_profit_per_1000_calendar_steps"]
    # The discarded life counts only the planned exchanges: 8 steps of 110 on each linear life; the
    # failed life is not averaged in, unlike in mean_normalized_discarded_life over all lives.
    assert rows[1]["mean_normalized_discarded_life_over_planned"] == pytest.approx(8 / 110, rel=1e-15)
    assert rows[1]["mean_normalized_discarded_life"] == pytest.approx(3 * (8 / 110) / 4, rel=1e-15)
    assert not rows[1]["zero_failure_eligible"]
    # The late order pays here, so the pooled criterion selects age 90 with its failure; the
    # zero-failure criterion selects the only candidate without a failure.
    assert pooled_90 > planned_85
    assert calibration._select("age", (85, 90), rows, "pooled") == [1, 0]
    assert calibration._select("age", (85, 90), rows, "zero_failure") == [0, 1]
    # A candidate without a failure is still ineligible when no life ends in a planned exchange: with a
    # budget of 50 steps, age 200 never orders and no life reaches its end.
    short = _constructed_bank([("linear_reference", 110), ("bearing_late_knee", 100)], budget=50)
    rows = _criteria_rows("age", short, (30, 200), CHEAP_FAILURES)
    assert rows[1]["failures"] == 0 and rows[1]["planned_interventions"] == 0
    assert rows[1]["mean_normalized_discarded_life_over_planned"] is None and not rows[1]["zero_failure_eligible"]
    assert rows[1]["pooled_profit_per_1000_calendar_steps"] > rows[0]["pooled_profit_per_1000_calendar_steps"]
    assert calibration._select("age", (30, 200), rows, "zero_failure") == [0, 1]
    assert calibration._select("age", (30, 200), rows, "pooled") == [1, 0]


def _candidate(eligible, family_equal, discarded, pooled=0.0, failures=None):
    return {"zero_failure_eligible": eligible, "family_equal_profit_per_1000_calendar_steps": family_equal,
            "mean_normalized_discarded_life_over_planned": discarded,
            "pooled_profit_per_1000_calendar_steps": pooled, "failures": (0 if eligible else 1) if failures is None
            else failures}


def test_zero_failure_tie_breaks_follow_the_documented_ordering():
    # The criterion selects min(eligible, key=(-family_equal_profit, discarded_life, later_key,
    # candidate_id)) with later_key = -age for the age rule and the threshold itself otherwise.
    ages = (80, 83, 86, 89)
    rows = [_candidate(True, 10.0, 0.2), _candidate(True, 10.0, 0.1), _candidate(True, 10.0, 0.1),
            _candidate(False, 99.0, 0.0)]
    # A failing candidate ranks last despite the highest profit; the lower discarded life beats the
    # later age; at equal profit and discarded life the higher age wins.
    assert calibration._select("age", ages, rows, "zero_failure") == [2, 1, 0, 3]
    for rule, grid in (("health", (0.30, 0.33, 0.36)), ("projection", (14, 15, 16))):
        equal = [_candidate(True, 5.0, 0.1)] * 3
        # At equal profit and discarded life the lower health or projection threshold wins.
        assert calibration._select(rule, grid, equal, "zero_failure") == [0, 1, 2]
        # The family-equal profit decides before everything else.
        better = [_candidate(True, 5.0, 0.1), _candidate(True, 5.0, 0.1), _candidate(True, 6.0, 0.5)]
        assert calibration._select(rule, grid, better, "zero_failure")[0] == 2
        # A candidate without a planned exchange has no discarded life and is not eligible.
        missing = [_candidate(False, 7.0, None, failures=0), _candidate(True, 5.0, 0.1), _candidate(True, 5.0, 0.2)]
        assert calibration._select(rule, grid, missing, "zero_failure") == [1, 2, 0]
    with pytest.raises(ValueError, match="criterion must be one of zero_failure, pooled"):
        calibration._select("age", ages, rows, "family")


def test_pooled_ties_prefer_fewer_failures_then_the_later_acting_threshold():
    # Ages above every lifetime never order: equal profit, all lives fail. The latest age wins the tie.
    with pytest.warns(GridEdgeWarning, match="age threshold 300 is the largest"):
        result = calibrate(CORE, rules=["age"], grids={"age": (300, 200, 250)}, criterion="pooled", seed=3, lives=4,
                           verify=False)
    assert [row["pooled_profit_per_1000_calendar_steps"] for row in result.table["age"]].count(
        result.row("age")["pooled_profit_per_1000_calendar_steps"]) == 3
    assert result.selected["age"] == 300
    assert calibration._select("health", (0.3, 0.4), [{"pooled_profit_per_1000_calendar_steps": 1.0, "failures": 0}] * 2,
                               "pooled") == [0, 1]
    assert calibration._select("health", (0.3, 0.4), [{"pooled_profit_per_1000_calendar_steps": 1.0, "failures": 2},
                                                      {"pooled_profit_per_1000_calendar_steps": 1.0, "failures": 0}],
                               "pooled") == [1, 0]


def test_the_zero_failure_criterion_stops_when_a_rule_has_no_candidate_without_failures():
    # The same ages as above: every life fails, so no candidate is eligible.
    with pytest.raises(calibration.NoEligibleThresholdError,
                       match=r"No age threshold is eligible under the zero-failure criterion.*every age threshold of "
                             r"the grid fails on at least one of the 4 calibration lives \(fewest failures: 4 at 300\)"
                             r".*--criterion pooled"):
        calibrate(CORE, rules=["age"], grids={"age": (300, 200, 250)}, seed=3, lives=4, verify=False)
    # Starts at true RUL 5 make every order late, whatever the grid.
    late = resolve_profile(CORE, {"env_config": {"training_start_offsets": [5]}})
    with pytest.raises(ValueError, match="No projection threshold is eligible"):
        calibrate(late, rules=["projection"], seed=3, lives=5, verify=False)
    with pytest.raises(ValueError, match="criterion must be one of zero_failure, pooled; got 'oldest'"):
        calibrate(CORE, criterion="oldest", lives=2, verify=False)


def test_a_wear_family_missing_from_the_bank_is_reported_under_the_zero_failure_criterion():
    # Root 3 with 4 lives draws no late-knee life; the family-equal profit then averages three families.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridEdgeWarning)
        with pytest.warns(FamilyCoverageWarning, match="no life of the wear family bearing_late_knee"):
            result = calibrate(CORE, rules=["health"], grids={"health": (0.3, 0.36, 0.4)}, seed=3, lives=4,
                               verify=False)
    assert result.families["bearing_late_knee"] == 0 and sum(result.families.values()) == 4
    assert any("family-equal profit averages the 3 families present" in text for text in result.warnings)
    assert len(result.row("health")["family_profit_per_1000_calendar_steps"]) == 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridEdgeWarning)
        warnings.simplefilter("error", FamilyCoverageWarning)
        # Not reported under the pooled criterion, nor for a configuration with a single family.
        calibrate(CORE, rules=["health"], grids={"health": (0.3, 0.36, 0.4)}, criterion="pooled", seed=3, lives=4,
                  verify=False)
        single = calibrate(resolve_profile(CORE, {"env_config": {"family": "bearing_late_knee"}}), rules=["health"],
                           grids={"health": (0.3, 0.36, 0.4)}, seed=3, lives=4, verify=False)
    assert single.families == {"linear_reference": 0, "concave_running_in": 0, "convex_accelerating_fatigue": 0,
                               "bearing_late_knee": 4}
    row = single.row("health")
    assert row["family_equal_profit_per_1000_calendar_steps"] == pytest.approx(
        row["pooled_profit_per_1000_calendar_steps"], rel=1e-12)


def test_an_optimum_at_a_grid_edge_raises_a_warning_and_is_recorded():
    # Thresholds below 13 order too late; 13 is the best of this grid and its largest value.
    with pytest.warns(GridEdgeWarning, match="largest value of its grid") as caught:
        result = calibrate(CORE, rules=["projection"], grids={"projection": (10, 11, 12, 13)}, criterion="pooled",
                           seed=5, lives=12, verify=False)
    assert len(caught) == 1 and result.selected["projection"] == 13
    assert result.warnings and "Widen the grid" in result.warnings[0]
    assert "- Warning: The selected projection threshold 13" in result.to_markdown()
    # Equal profit (every life fails at thresholds 8 to 12) goes to the later-acting threshold 8.
    with pytest.warns(GridEdgeWarning, match="smallest value of its grid"):
        tied = calibrate(CORE, rules=["projection"], grids={"projection": (8, 9, 10, 11, 12)}, criterion="pooled",
                         seed=5, lives=12, verify=False)
    assert tied.selected["projection"] == 8 and tied.row("projection")["failures"] == 12
    assert len({row["pooled_profit_per_1000_calendar_steps"] for row in tied.table["projection"]}) == 1
    # Under the zero-failure criterion, 13 fails on 8 of these lives, and 14 is the smallest threshold
    # without a failure.
    with pytest.warns(GridEdgeWarning, match="projection threshold 14 is the smallest value") as caught:
        edge = calibrate(CORE, rules=["projection"], grids={"projection": (14, 15, 16, 17)}, seed=5, lives=12,
                         verify=False)
    assert len(caught) == 1 and edge.selected["projection"] == 14 and edge.row("projection")["failures"] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FamilyCoverageWarning)  # three lives cannot cover four families
        with pytest.warns(GridEdgeWarning, match="single value"):
            single = calibrate(CORE, rules=["health"], grids={"health": [0.36]}, seed=5, lives=3, verify=False)
    assert single.selected == {"health": 0.36}
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridEdgeWarning)
        interior = calibrate(CORE, rules=["projection"], grids={"projection": (13, 14, 15, 16, 17)}, seed=5,
                             lives=12, verify=False)
    assert interior.selected["projection"] == 14 and interior.warnings == ()
    assert interior.row("projection", 13)["failures"] == 8 and not interior.row("projection", 13)["zero_failure_eligible"]


def test_the_default_core_calibration_on_800_lives_selects_the_profile_defaults():
    # The documented acceptance run: root 20261002, 800 lives, default 252-candidate grids. The
    # zero-failure criterion selects the profile's thresholds, age 83, health 0.36 and projection 15.
    result = calibrate(CORE, seed=20261002, lives=800, verify=False)
    assert result.criterion == "zero_failure"
    assert result.selected == {"age": 83, "health": 0.36, "projection": 15}
    assert result.warnings == () and sum(len(values) for values in result.grids.values()) == 252
    assert sum(result.families.values()) == 800 and all(result.families.values())
    for rule in result.rules:
        assert result.row(rule)["failures"] == 0
        assert result.defaults[rule]["on_grid"] and result.defaults[rule]["grid_steps_from_selected"] == 0
        assert result.defaults[rule]["difference_per_1000_calendar_steps"] == 0
        assert result.defaults[rule]["family_equal_difference_per_1000_calendar_steps"] == 0
    # The pooled criterion, applied to the same table, selects health 0.35, which fails on 2 of these
    # lives; age and projection are the same.
    pooled = {rule: result.grids[rule][calibration._select(rule, result.grids[rule], result.table[rule], "pooled")[0]]
              for rule in result.rules}
    assert pooled == {"age": 83, "health": 0.35, "projection": 15}
    assert result.row("health", 0.35)["failures"] == 2 and not result.row("health", 0.35)["zero_failure_eligible"]
    assert (result.row("health", 0.35)["pooled_profit_per_1000_calendar_steps"]
            > result.row("health", 0.36)["pooled_profit_per_1000_calendar_steps"])


def test_calibration_under_a_modified_configuration_reports_overrides():
    overrides = {"env_config": {"sensor_level": "stress", "observation_route": "ROBUST_MULTISCALE_234"}}
    profile = resolve_profile(CORE, overrides)
    result = _quiet_calibrate(profile, rules=["projection"], grids={"projection": (13, 15, 17, 19, 21, 23)},
                              seed=2, lives=30)
    assert result.profile == profile["id"] and result.configuration_sha256 == profile["configuration_sha256"]
    assert result.environment_configuration == profile["env_config"]
    assert result.overrides() == {"env_config": overrides["env_config"],
                                  "policy_defaults": {"projection": {"projection_threshold": result.selected["projection"]}}}
    # The overrides recreate the calibrated rule as a policy of a regular modified profile.
    calibrated = resolve_profile(CORE, result.overrides())
    replay = evaluate(calibrated, policies=["projection"], seed=0, episodes=1, calendar_steps=720,
                      fixture=make_fixture(calibrated, result.seeds, calendar_steps=720), events=False)
    assert replay["summary"][0]["pooled_profit_per_1000_calendar_steps"] == result.row("projection")[
        "pooled_profit_per_1000_calendar_steps"]


@pytest.mark.parametrize("arguments,message", [
    ({"profile": "paper-gearbox-v1"}, "defined for the core profiles"),
    ({"rules": ["age", "oracle"]}, "nonempty selection"),
    ({"rules": []}, "nonempty selection"),
    ({"rules": ["age", "age"]}, "Duplicate rules"),
    ({"rules": ["age"], "grids": {"health": [0.3]}}, "not calibrated"),
    ({"grids": {"age": []}}, "empty"),
    ({"grids": {"age": [-2]}}, "nonnegative"),
    ({"grids": {"health": [1.2]}}, r"\[0, 1\]"),
    ({"grids": {"projection": [float("nan")]}}, "finite"),
    ({"grids": {"age": [True]}}, "finite numbers"),
    ({"lives": 0}, "lives must be an integer >= 1"),
    ({"seed": -1}, "calibration seed must be an integer >= 0"),
    ({"calendar_steps": 0}, "calendar_steps must be an integer >= 1"),
    ({"confidence": 1.0}, "confidence"),
])
def test_invalid_calibration_requests_are_rejected(arguments, message):
    request = {"profile": CORE, "lives": 2, "verify": False, **arguments}
    with pytest.raises(ValueError, match=message):
        calibrate(request.pop("profile"), **request)


# The JSON record.

def test_the_record_round_trips_through_json_and_detects_changes(tmp_path):
    result = _quiet_calibrate(CORE, grids=SMALL, seed=8, lives=6, verify=False)
    record = result.to_dict()
    assert record["schema"] == calibration.SCHEMA and len(record["sha256"]) == 64
    assert record["seeds"] == list(result.seeds) and record["lives"] == 6
    assert record["seed_derivation"]["root"] == 8 and record["seed_derivation"]["stream"] == 1
    assert {rule: len(rows) for rule, rows in record["table"].items()} == {rule: len(v) for rule, v in SMALL.items()}
    assert record["policy_defaults"] == result.policy_defaults()
    text = json.dumps(record, allow_nan=False)
    assert calibration.CalibrationResult.from_dict(json.loads(text)) == result
    path = tmp_path / "calibration.json"
    path.write_text(text, encoding="utf-8")
    assert calibration.load_calibration(path).to_dict() == record
    tampered = json.loads(text)
    tampered["selected"]["projection"] = 16
    with pytest.raises(ValueError, match="sha256 does not match"):
        calibration.CalibrationResult.from_dict(tampered)
    with pytest.raises(ValueError, match="needs schema"):
        calibration.CalibrationResult.from_dict({"schema": "other"})
    # The selection criterion is part of the record and its hash.
    assert record["selection_criterion"] == "zero_failure" and sum(record["families"].values()) == 6
    unnamed = {key: value for key, value in json.loads(text).items() if key not in ("sha256", "selection_criterion")}
    unnamed["sha256"] = calibration.canonical_hash(unnamed)
    with pytest.raises(ValueError, match="must name its selection_criterion"):
        calibration.CalibrationResult.from_dict(unnamed)
    pooled = _quiet_calibrate(CORE, grids=SMALL, criterion="pooled", seed=8, lives=6, verify=False)
    assert pooled.to_dict()["selection_criterion"] == "pooled" and pooled.to_dict()["sha256"] != record["sha256"]
    assert calibration.CalibrationResult.from_dict(pooled.to_dict()).criterion == "pooled"
    assert pooled.to_dict()["criterion"] == calibration.CRITERION_TEXTS["pooled"]


# Recalibrated benchmark references.

def test_benchmark_keeps_the_profile_thresholds_for_the_published_configuration():
    result = benchmark.score(profile=CORE, seeds=[1, 2], references=("age", "health", "projection", "reactive"))
    assert result.calibration is None and result.calibration_origin is None
    assert {row["policy"]: row["threshold"] for row in result.rows} == {
        "oracle": None, "age": 83, "health": 0.36, "projection": 15, "reactive": None}
    assert "Rule thresholds: age 83 (policy_defaults), health 0.36 (policy_defaults), projection 15 (policy_defaults)." in result.notes
    assert result.to_dict()["calibration"] is None


def test_benchmark_recalibrates_the_rules_when_the_environment_changes():
    profile = resolve_profile(CORE, {"env_config": {"failure_event_cost": 2000.0}})
    result = benchmark.score(profile=profile, seed=11, episodes=8, references=("age", "projection"),
                             calibration_lives=20)
    assert result.calibration_origin == "automatic" and result.calibration["rules"] == ["age", "projection"]
    assert not set(result.calibration["seeds"]) & set(result.scenario_seeds)
    assert result.calibration["seed_derivation"]["root"] == 11 and result.calibration["lives"] == 20
    selected = result.calibration["selected"]
    assert result.row("age")["threshold"] == selected["age"] and result.row("projection")["threshold"] == selected["projection"]
    assert not any("not recalibrated" in note for note in result.notes)
    assert any(note.startswith("Rules recalibrated because the environment configuration differs from "
                               f"{CORE}: selected by the zero-failure criterion (no failure, then the highest "
                               "family-equal profit) on 20 calibration lives") for note in result.notes)
    summary = result.to_dict()["calibration"]
    assert summary["origin"] == "automatic" and summary["sha256"] == result.calibration["sha256"] and summary["verified"]
    assert summary["selection_criterion"] == result.calibration["selection_criterion"] == "zero_failure"
    # The recalibrated references are the built-in rules with the selected thresholds.
    calibrated = resolve_profile(CORE, {"env_config": {"failure_event_cost": 2000.0},
                                        "policy_defaults": result.calibration["policy_defaults"]})
    replay = evaluate(calibrated, policies=["age", "projection"], seed=0, episodes=1, calendar_steps=720,
                      fixture=make_fixture(calibrated, result.scenario_seeds, calendar_steps=720), events=False)
    for summary in replay["summary"]:
        assert summary["pooled_profit_per_1000_calendar_steps"] == result.row(summary["policy"])[
            "pooled_profit_per_1000_calendar_steps"]
    # The same calibration, run separately and supplied, gives the same table.
    stored = calibrate(profile, rules=["age", "projection"], seed=11, lives=20, exclude_seeds=result.scenario_seeds)
    assert stored.to_dict() == result.calibration
    supplied = benchmark.score(profile=profile, seed=11, episodes=8, references=("age", "projection"),
                               calibration=stored.to_dict())
    assert supplied.calibration_origin == "supplied" and supplied.rows == result.rows


@pytest.mark.filterwarnings("ignore::maintenance_arena.calibration.FamilyCoverageWarning")
def test_explicit_thresholds_are_kept_and_recalibration_can_be_forced_or_disabled():
    profile = resolve_profile(CORE, {"env_config": {"failure_event_cost": 2000.0},
                                     "policy_defaults": {"projection": {"projection_threshold": 17}}})
    result = benchmark.score(profile=profile, seeds=[4, 5], references=("health", "projection"), calibration_lives=10)
    assert result.calibration["rules"] == ["health"] and result.row("projection")["threshold"] == 17
    assert "The projection threshold is set explicitly by the configuration and was kept." in result.notes
    assert any("not recalibrated" in note for note in result.notes)
    forced = benchmark.score(profile=profile, seeds=[4, 5], references=("projection",), recalibrate=True,
                             calibration_lives=10)
    assert forced.calibration_origin == "requested" and forced.calibration["rules"] == ["projection"]
    published = benchmark.score(profile=CORE, seeds=[4, 5], references=("projection",), recalibrate=True,
                                calibration_lives=10, criterion="pooled")
    assert published.calibration_origin == "requested" and published.calibration["selection_criterion"] == "pooled"
    assert published.to_dict()["calibration"]["selection_criterion"] == "pooled"
    assert any("selected by pooled profit on 10 calibration lives" in note for note in published.notes)
    disabled = benchmark.score(profile=profile, seeds=[4, 5], references=("health",), recalibrate=False)
    assert disabled.calibration is None and disabled.row("health")["threshold"] == 0.36
    with pytest.raises(ValueError, match="recalibrate=False keeps the profile's policy_defaults"):
        benchmark.score(profile=profile, seeds=[4, 5], references=("health",), recalibrate=False, criterion="pooled")
    with pytest.raises(ValueError, match="criterion must be one of"):
        benchmark.score(profile=profile, seeds=[4, 5], references=("health",), criterion="best")


def test_supplied_calibrations_must_match_the_benchmark():
    profile = resolve_profile(CORE, {"env_config": {"failure_event_cost": 2000.0}})
    stored = _quiet_calibrate(profile, rules=["projection"], grids={"projection": (14, 15, 16)}, seed=1, lives=5,
                              verify=False)
    with pytest.raises(ValueError, match="another environment configuration"):
        benchmark.score(profile=CORE, seeds=[1], references=("projection",), calibration=stored)
    with pytest.raises(ValueError, match="calendar steps"):
        benchmark.score(profile=profile, seeds=[1], calendar_steps=500, references=("projection",), calibration=stored)
    with pytest.raises(ValueError, match="must be disjoint"):
        benchmark.score(profile=profile, seeds=[stored.seeds[2], 99], references=("projection",), calibration=stored)
    with pytest.raises(ValueError, match="either calibration or recalibrate"):
        benchmark.score(profile=profile, seeds=[1], calibration=stored, recalibrate=True)
    # A supplied record keeps its criterion; a criterion passed with it must be the same.
    assert stored.criterion == "zero_failure"
    kept = benchmark.score(profile=profile, seeds=[1], references=("projection",), calibration=stored,
                           criterion="zero_failure")
    assert kept.calibration_origin == "supplied" and kept.calibration["selection_criterion"] == "zero_failure"
    with pytest.raises(ValueError, match="selected with the criterion 'zero_failure', not 'pooled'"):
        benchmark.score(profile=profile, seeds=[1], references=("projection",), calibration=stored, criterion="pooled")
    with pytest.raises(ValueError, match="core profiles only"):
        benchmark.score(profile="paper-gearbox-v1", seeds=[1], calendar_steps=48, references=(), recalibrate=True)
    with pytest.raises(ValueError, match="CalibrationResult"):
        benchmark.score(profile=profile, seeds=[1], calibration="calibration.json")
    gearbox = resolve_profile("paper-gearbox-v1", {"env_config": {"horizon": 96}})
    result = benchmark.score(profile=gearbox, seeds=[1], calendar_steps=48, references=())
    assert any("available for the core rules only" in note for note in result.notes)


# The calibrate command and the benchmark options.

def test_cli_calibrate_writes_the_selection_table_and_usable_overrides(tmp_path, capsys):
    output = tmp_path / "calibrated"
    assert main(["calibrate", "--profile", CORE, "--seed", "9", "--lives", "12", "--age-grid", "70:90:5",
                 "--health-grid", "0.30:0.40:0.02", "--projection-grid", "14", "15", "--output", str(output)]) == 0
    captured = capsys.readouterr()
    table = (output / "calibration.md").read_text(encoding="utf-8")
    assert captured.out == table + f"\nResults: {output.resolve()}\n"
    assert "Selection by the zero-failure criterion" in table
    assert "maintenance-arena: warning: The selected projection threshold 14 is the smallest value" in captured.err
    record = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    result = calibration.CalibrationResult.from_dict(record)
    assert record["selection_criterion"] == result.criterion == "zero_failure"
    assert result.grids == {"age": (70, 75, 80, 85, 90), "health": (0.3, 0.32, 0.34, 0.36, 0.38, 0.4),
                            "projection": (14, 15)}
    assert record["grid_sources"] == {"age": "user", "health": "user", "projection": "user"}
    assert record["verification"]["identical"] and record["warnings"] == list(result.warnings)
    assert result.seeds == tuple(calibration.calibration_seeds(9, 12))
    overrides = json.loads((output / "overrides.json").read_text(encoding="utf-8"))
    assert overrides == {"policy_defaults": result.policy_defaults()}
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete" and run["command"] == "calibrate"
    assert run["result"]["policy_defaults"] == result.policy_defaults() and run["result"]["sha256"] == record["sha256"]
    assert run["result"]["selection_criterion"] == "zero_failure" and run["arguments"]["criterion"] == "zero_failure"
    # The overrides file evaluates the calibrated rules; on the calibration lives it reproduces the table.
    fixture = tmp_path / "calibration_fixture.json"
    fixture.write_text(json.dumps(make_fixture(resolve_profile(CORE), list(result.seeds), calendar_steps=720)),
                       encoding="utf-8")
    replay = tmp_path / "replay"
    assert main(["evaluate", "--profile", CORE, "--seed", "0", "--policy", "age", "health", "--config",
                 str(output / "overrides.json"), "--fixture", str(fixture), "--output", str(replay)]) == 0
    replayed = json.loads((replay / "evaluation.json").read_text(encoding="utf-8"))
    for summary in replayed["summary"]:
        assert summary["pooled_profit_per_1000_calendar_steps"] == result.row(summary["policy"])[
            "pooled_profit_per_1000_calendar_steps"]


def test_cli_calibrate_excludes_the_seeds_of_test_fixtures(tmp_path, capsys):
    derived = calibration.calibration_seeds(4, 3)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(make_fixture(resolve_profile(CORE), [derived[1], 17], calendar_steps=720)),
                       encoding="utf-8")
    output = tmp_path / "calibrated"
    assert main(["calibrate", "--profile", CORE, "--seed", "4", "--lives", "3", "--rule", "projection",
                 "--projection-grid", "14:16", "--criterion", "pooled", "--exclude-fixture", str(fixture),
                 "--output", str(output)]) == 0
    assert "Selection by pooled profit" in capsys.readouterr().out
    record = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    assert derived[1] not in record["seeds"] and len(record["seeds"]) == 3 and record["excluded_seed_count"] == 2
    assert record["seed_derivation"]["skipped"] == [{"index": 1, "seed": derived[1], "reason": "excluded test seed"}]
    assert record["selection_criterion"] == "pooled" and record["criterion"] == calibration.CRITERION_TEXTS["pooled"]
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["inputs"]["exclude_fixtures"][0]["supplied_path"] == str(fixture)
    assert run["result"]["selection_criterion"] == "pooled"


def test_cli_calibrate_reports_a_rule_without_eligible_thresholds(tmp_path, capsys):
    output = tmp_path / "no-eligible"
    assert main(["calibrate", "--profile", CORE, "--seed", "3", "--lives", "4", "--rule", "age",
                 "--age-grid", "200:300:50", "--output", str(output)]) == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and "No age threshold is eligible under the zero-failure criterion" in error
    assert "--criterion pooled" in error
    # The calibration ran, so its run record states the failure.
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed" and not (output / "calibration.json").exists()
    assert main(["calibrate", "--profile", CORE, "--seed", "3", "--lives", "4", "--rule", "age",
                 "--age-grid", "200:300:50", "--criterion", "pooled", "--output", str(tmp_path / "pooled")]) == 0


@pytest.mark.parametrize("arguments,message", [
    (["--profile", "paper-resource-v1"], "defined for the core profiles"),
    (["--profile", CORE, "--lives", "0"], "lives must be an integer >= 1"),
    (["--profile", CORE, "--age-grid", "9:1"], "START <= STOP"),
    (["--profile", CORE, "--rule", "projection", "--age-grid", "80"], "--age-grid was given"),
    (["--profile", CORE, "--exclude-fixture", "{missing}"], "--exclude-fixture file does not exist"),
    (["--profile", CORE, "--exclude-fixture", "{not_fixture}"], "is not a fixture.json"),
])
def test_cli_calibrate_validates_before_creating_the_output(tmp_path, capsys, arguments, message):
    not_fixture = tmp_path / "other.json"
    not_fixture.write_text(json.dumps({"seeds": [1]}), encoding="utf-8")
    paths = {"{missing}": str(tmp_path / "missing.json"), "{not_fixture}": str(not_fixture)}
    output = tmp_path / "rejected"
    assert main(["calibrate", "--seed", "1", "--output", str(output)] + [paths.get(a, a) for a in arguments]) == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and message in error, error
    assert not output.exists()


def test_cli_benchmark_recalibrates_for_a_changed_configuration(tmp_path, capsys):
    config = tmp_path / "cheap_failures.yaml"
    config.write_text("env_config:\n  failure_event_cost: 2000.0\n", encoding="utf-8")
    common = ["benchmark", "--profile", CORE, "--config", str(config), "--seed", "3", "--episodes", "4",
              "--reference", "projection"]
    automatic = tmp_path / "automatic"
    assert main(common + ["--calibration-lives", "6", "--output", str(automatic)]) == 0
    assert ("Rules recalibrated because the environment configuration differs from paper-core-continuous-v1: "
            "selected by the zero-failure criterion") in capsys.readouterr().out
    stored = json.loads((automatic / "calibration.json").read_text(encoding="utf-8"))
    record = json.loads((automatic / "benchmark.json").read_text(encoding="utf-8"))
    assert record["calibration"]["sha256"] == stored["sha256"] and record["calibration"]["lives"] == 6
    assert record["calibration"]["selection_criterion"] == stored["selection_criterion"] == "zero_failure"
    assert not set(stored["seeds"]) & set(record["fixture"]["seeds"])
    run = json.loads((automatic / "run.json").read_text(encoding="utf-8"))
    assert run["result"]["calibration"]["origin"] == "automatic"
    supplied = tmp_path / "supplied"
    assert main(common + ["--calibration", str(automatic / "calibration.json"), "--criterion", "zero_failure",
                          "--output", str(supplied)]) == 0
    capsys.readouterr()
    again = json.loads((supplied / "benchmark.json").read_text(encoding="utf-8"))
    assert again["calibration"]["origin"] == "supplied" and again["rows"] == record["rows"]
    pooled = tmp_path / "pooled"
    assert main(common + ["--calibration-lives", "6", "--criterion", "pooled", "--output", str(pooled)]) == 0
    assert "selected by pooled profit on 6 calibration lives" in capsys.readouterr().out
    assert json.loads((pooled / "calibration.json").read_text(encoding="utf-8"))["selection_criterion"] == "pooled"
    assert json.loads((pooled / "benchmark.json").read_text(encoding="utf-8"))["calibration"][
        "selection_criterion"] == "pooled"
    kept = tmp_path / "kept"
    assert main(common + ["--no-recalibrate", "--output", str(kept)]) == 0
    assert "not recalibrated" in capsys.readouterr().out and not (kept / "calibration.json").exists()


@pytest.mark.parametrize("arguments,message", [
    (["--no-recalibrate", "--calibration", "{calibration}"], "exclude each other"),
    (["--no-recalibrate", "--criterion", "zero_failure"], "--criterion and --no-recalibrate exclude each other"),
    (["--calibration", "{calibration}", "--criterion", "pooled"],
     "selected with the criterion 'zero_failure', not 'pooled'"),
    (["--calibration-lives", "0"], "calibration lives must be an integer >= 1"),
    (["--calibration", "{calibration}", "--fixture", "{overlap}"], "must be disjoint"),
    (["--calibration", "{calibration}", "--calendar-steps", "300"], "calendar steps"),
    (["--calibration", "{not_calibration}"], "needs schema"),
])
def test_cli_benchmark_validates_calibration_options_before_creating_the_output(tmp_path, capsys, arguments, message):
    stored = _quiet_calibrate(CORE, rules=["projection"], grids={"projection": (14, 15)}, seed=2, lives=3, verify=False)
    paths = {"{calibration}": tmp_path / "calibration.json", "{overlap}": tmp_path / "fixture.json",
             "{not_calibration}": tmp_path / "other.json"}
    paths["{calibration}"].write_text(json.dumps(stored.to_dict()), encoding="utf-8")
    paths["{overlap}"].write_text(json.dumps(make_fixture(resolve_profile(CORE), [stored.seeds[0]], calendar_steps=720)),
                                  encoding="utf-8")
    paths["{not_calibration}"].write_text("{}", encoding="utf-8")
    output = tmp_path / "rejected"
    command = ["benchmark", "--profile", CORE, "--seed", "1", "--output", str(output)]
    assert main(command + [str(paths.get(a, a)) for a in arguments]) == 2
    error = capsys.readouterr().err
    assert len(error.splitlines()) == 1 and message in error, error
    assert not output.exists()
