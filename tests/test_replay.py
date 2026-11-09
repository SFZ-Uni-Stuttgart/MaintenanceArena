"""Batch first orders, threshold rules and the article's calibration and selection rules."""
import math

import numpy as np
import pytest

from maintenance_arena.banks import ScenarioBankEnv, build_bank, subset_bank
from maintenance_arena.core import policy_action
from maintenance_arena.replay import (CUT_ANCHOR_INDEX, CUT_GRID, Economics, age_orders, calibrate_age, calibrate_cut,
                                      calibrate_per_law, calibrate_rule, calibrate_two_indicator, cut_orders,
                                      decision_series, first_at_or_above, first_at_or_below, first_orders, grid,
                                      greedy_orders, per_law_orders, projected_time, replay, score_orders,
                                      select_failure_budget, select_maximum, threshold_candidates, threshold_orders,
                                      two_indicator_orders, two_indicator_rule_orders)


@pytest.fixture(scope="module")
def series():
    return decision_series(build_bank("source", split="development", seed=21, pair_count=16))


def scalar_first(values, slices, test):
    result = []
    for start, stop in slices:
        hit = [i for i in range(stop - start) if test(values[start + i])]
        result.append(hit[0] if hit else -1)
    return np.asarray(result)


def test_grids_and_cut_grid():
    assert grid(.2, .6, .0025, 4)[:3] == [.2, .2025, .205] and len(grid(.2, .6, .0025, 4)) == 161
    assert len(grid(8., 40., .05, 2)) == 641
    assert len(CUT_GRID) == 123 and CUT_GRID[CUT_ANCHOR_INDEX] == .5
    assert all(a < b for a, b in zip(CUT_GRID, CUT_GRID[1:]))


def test_vectorized_first_crossings_equal_scalar_searches(series):
    rng = np.random.default_rng(2)
    values = rng.random(len(series.states))
    for threshold in (.05, .3, .99):
        np.testing.assert_array_equal(first_at_or_below(values, series.slices, [threshold])[0],
                                      scalar_first(values, series.slices, lambda v: v <= threshold))
        np.testing.assert_array_equal(first_at_or_above(values, series.slices, [threshold])[0],
                                      scalar_first(values, series.slices, lambda v: v >= threshold))
        np.testing.assert_array_equal(first_orders(values > threshold, series.slices),
                                      scalar_first(values, series.slices, lambda v: v > threshold))


def test_rule_orders_equal_the_environment_rules(series):
    bank = build_bank("source", split="development", seed=21, pair_count=16)
    projection = projected_time(series.states)
    for threshold in (13., 15., 20.):
        orders = threshold_orders(series, [{"family": "projection", "threshold": threshold}], projection)[0]
        for life in (0, 5, 11):
            env = ScenarioBankEnv(subset_bank(bank, [life]))
            observation, _ = env.reset()
            age = 0
            while policy_action("projection", env, observation, projection_threshold=threshold) == 0:
                observation, _, done, _, _ = env.step(0)
                age += 1
                assert not done
            assert age == orders[life]
    health = threshold_orders(series, [{"family": "health", "threshold": .36}])[0]
    np.testing.assert_array_equal(health, scalar_first(series.health, series.slices, lambda v: v <= .36))
    expected = [(1. - (-math.log(float(s[0])) / math.log(5.))) / max(float(s[1]) / 12., 1e-12) if s[2] >= .5
                else math.inf for s in series.states]
    np.testing.assert_array_equal(projection, expected)


def test_two_indicator_and_age_orders(series):
    projection = projected_time(series.states)
    orders = two_indicator_orders(series, [.3], [14.], [12.5], projection)[0, 0, 0]
    health = series.health
    direct = first_orders(((health > .3) & (projection <= 14.)) | ((health <= .3) & (projection <= 12.5)), series.slices)
    np.testing.assert_array_equal(orders, direct)
    np.testing.assert_array_equal(two_indicator_rule_orders(series, {"kind": "block_0", "health_split": .3,
                                                                     "a_above_split": 14., "b_at_or_below_split": 12.5},
                                                            projection), direct)
    ages = age_orders(series, [0, 90, 200])
    assert np.all(ages[0] == 0) and np.all(ages[2] == -1)
    lengths = np.asarray([b - a for a, b in series.slices])
    np.testing.assert_array_equal(ages[1], np.where(lengths > 90, 90, -1))


def test_selection_rules():
    assert select_maximum([1., 2., 2. - 5e-10, 1.5], [0., .1, .0, 0.]) == 2
    assert select_maximum([2., 2., 2.], [0., 0., 0.], [.3, .1, .1]) == 1
    assert select_maximum([2., 1.], [1., 0.]) == 0
    assert select_failure_budget([.02, .005, .01, 0.], [.0, .3, .2, .5], .01) == 2
    assert select_failure_budget([.5], [.0], .01) is None


def test_replay_accounting_and_scores():
    lives = np.asarray([127, 127, 127, 127])
    outcome = replay(lives, np.asarray([115, 116, -1, 0]))
    np.testing.assert_array_equal(outcome["failed"], [False, True, True, False])
    # A late order keeps its completed preparation: one step is left after the failure.
    np.testing.assert_array_equal(outcome["calendar"], [134, 135, 146, 19])
    np.testing.assert_array_equal(outcome["cash"], [127 * 255 - 695, 127 * 255 - 695 - 7000 - 25,
                                                    127 * 255 - 695 - 7000 - 300, 12 * 255 - 695])
    np.testing.assert_array_equal(outcome["unused_fraction"], [0., 0., 0., 1.])
    immediate = replay(lives[:2], np.asarray([116, -1]), Economics(recovery_contract="immediate_exchange_v1"))
    np.testing.assert_array_equal(immediate["calendar"], [134, 134])
    with pytest.raises(ValueError):
        replay(lives, np.asarray([0, 0, 0, 128]))


def test_calibrations_choose_the_best_equal_law_rate(series):
    economics = Economics()
    grids = {"health": [.3, .36], "projection": [13., 15., 20.]}
    result = calibrate_rule(series, grids, economics=economics)
    candidates = threshold_candidates(grids)
    scores = score_orders(series, threshold_orders(series, candidates), economics)
    assert scores.rate[result["selected"]["candidate_id"]] >= scores.rate.max() - 1e-9
    assert [c["candidate_id"] for c in result["candidates"]] == list(range(5))
    per_law = calibrate_per_law(series, grids, economics=economics)
    assert sorted(per_law) == sorted(set(series.families))
    orders = per_law_orders(series, per_law)
    for law, rule in per_law.items():
        rows = [i for i, family in enumerate(series.families) if family == law]
        np.testing.assert_array_equal(orders[rows], threshold_orders(series, [rule])[0, rows])
    age = calibrate_age(series, range(70, 110), economics=economics)["selected"]
    assert 70 <= age["age"] < 110
    two = calibrate_two_indicator(series, candidates, [{"health_split": [.3], "a": [14., 15.], "b": [13., 14.]}],
                                  economics=economics)["selected"]
    assert two["profit_rate"] >= result["selected"]["profit_rate"] - 1e-9 and two["candidate_id"] < 9


def test_probability_cuts(series):
    rng = np.random.default_rng(1)
    order = rng.random(len(series.states)).astype(np.float32)
    probabilities = np.stack([1 - order, order], axis=1).astype(np.float32)
    np.testing.assert_array_equal(cut_orders(probabilities, series.slices, [.5])[0],
                                  first_orders(order.astype(np.float64) >= .5, series.slices))
    np.testing.assert_array_equal(greedy_orders(probabilities, series.slices),
                                  first_orders(probabilities[:, 1] > probabilities[:, 0], series.slices))
    result = calibrate_cut(series, probabilities, economics=Economics(), failure_budget=.5)
    assert 0 <= result["selected"]["cut_index"] < len(CUT_GRID)
    assert result["failure_budget"] is None or result["failure_budget"]["failure_share"] <= .5
