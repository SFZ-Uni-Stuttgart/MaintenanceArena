"""The four new degradation laws of the adaptation study: shapes, ranges and environment selection."""
import math

import numpy as np
import pytest

from maintenance_arena.core import CoreCycleEnv, validate_config
from maintenance_arena.curves import (FAMILY_IDS, LAW_IDS, TARGET_FAMILY_IDS, TARGET_SHAPE_RANGES, materialize_health,
                                      normalized_damage_fraction, sample_curve, sample_target_shape, shape_parameters)

U = np.linspace(0., 1., 129)


def test_identifiers_and_ranges_follow_the_article_table():
    assert TARGET_FAMILY_IDS == ("paris_inspired", "two_knee", "gompertz", "power_law")
    assert LAW_IDS == (*FAMILY_IDS, *TARGET_FAMILY_IDS)
    assert {law: dict(ranges) for law, ranges in TARGET_SHAPE_RANGES.items()} == {
        "paris_inspired": {"w": (.15, .45), "q": (.55, .8), "p": (.7, 1.4)},
        "two_knee": {"w": (.08, .22), "k1": (.3, .45), "k2": (.72, .86), "v": (.2, .4), "s": (.025, .045)},
        "gompertz": {"A": (4., 8.), "i": (.4, .65)}, "power_law": {"beta": (2.8, 3.4)}}


@pytest.mark.parametrize("law", TARGET_FAMILY_IDS)
def test_shapes_are_drawn_in_order_within_their_ranges(law):
    rng = np.random.default_rng(5)
    shape = sample_target_shape(law, rng)
    reference = np.random.default_rng(5)
    expected = [(name, float(reference.uniform(low, high))) for name, (low, high) in TARGET_SHAPE_RANGES[law]]
    assert list(shape) == expected
    for _ in range(200):
        for name, value in sample_target_shape(law, rng):
            low, high = dict(TARGET_SHAPE_RANGES[law])[name]
            assert low <= value <= high


@pytest.mark.parametrize("law", TARGET_FAMILY_IDS)
def test_damage_is_normalized_and_strictly_increasing(law):
    rng = np.random.default_rng(11)
    for _ in range(50):
        damage = normalized_damage_fraction(law, U, sample_target_shape(law, rng))
        assert damage[0] == 0. and damage[-1] == 1.
        assert np.all(np.diff(damage) > 0)


def test_damage_formulas():
    paris = normalized_damage_fraction("paris_inspired", U, [("w", .3), ("q", .7), ("p", 1.)])
    expected = .3 * U + .7 * ((1. - .7 * U) ** -1. - 1.) / ((1. - .7) ** -1. - 1.)
    np.testing.assert_allclose(paris[1:-1], expected[1:-1], rtol=1e-14)
    power = normalized_damage_fraction("power_law", U, [("beta", 3.)])
    np.testing.assert_allclose(power[1:-1], U[1:-1] ** 3., rtol=1e-14)
    a, i = 6., .5
    b = math.log(a) / i
    gompertz = normalized_damage_fraction("gompertz", U, [("A", a), ("i", i)])
    expected = (np.exp(-a * np.exp(-b * U)) - math.exp(-a)) / (math.exp(-a * math.exp(-b)) - math.exp(-a))
    np.testing.assert_allclose(gompertz[1:-1], expected[1:-1], rtol=1e-13)
    # Gompertz damage accelerates until the inflection at u = i and slows afterwards.
    rate = np.diff(gompertz)
    assert np.argmax(rate) == pytest.approx(i * 128, abs=2)
    knee = normalized_damage_fraction("two_knee", U, [("w", .1), ("k1", .35), ("k2", .8), ("v", .3), ("s", .03)])
    rate = np.diff(knee)
    assert rate[60] > 3 * rate[20] and rate[120] > 2 * rate[60]


def test_health_is_float32_from_one_to_the_failure_level():
    for law in TARGET_FAMILY_IDS:
        health = materialize_health(law, 128, sample_target_shape(law, np.random.default_rng(3)))
        assert health.dtype == np.float32 and health.shape == (129,)
        assert health[0] == np.float32(1.) and health[-1] == np.float32(.2)
        assert np.all(np.diff(health.astype(np.float64)) < 0)


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError, match="sample_target_shape"):
        shape_parameters("gompertz", .5)
    with pytest.raises(ValueError, match="shape parameters"):
        normalized_damage_fraction("gompertz", U, [("A", 5.)])
    with pytest.raises(ValueError):
        materialize_health("power_law", 128)
    with pytest.raises(ValueError):
        sample_target_shape("linear_reference", np.random.default_rng(0))


def test_sampled_curves_and_environments_select_a_new_law():
    curve = sample_curve(np.random.default_rng(4), family="power_law")
    assert curve["family_id"] == "power_law" and 96 <= curve["eol_operating_age"] <= 160
    assert [name for name, _ in curve["shape_parameters"]] == ["beta"]
    with pytest.raises(ValueError, match="discrete"):
        sample_curve(np.random.default_rng(4), source="discrete", family="gompertz")
    env = CoreCycleEnv({"family": "gompertz"})
    env.reset(seed=8)
    assert env.diagnostics()["family_id"] == "gompertz"
    with pytest.raises(ValueError, match="discrete"):
        validate_config({"source": "discrete", "family": "two_knee"})
    # The default random source keeps drawing the four source laws only.
    env = CoreCycleEnv()
    families = set()
    for seed in range(40):
        env.reset(seed=seed)
        families.add(env.diagnostics()["family_id"])
    assert families == set(FAMILY_IDS)
