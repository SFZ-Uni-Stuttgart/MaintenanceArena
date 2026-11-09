"""Age-shift study: lifetime twins, the age input, the zero age column and the scaled noise."""
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from maintenance_arena import observations
from maintenance_arena import replay as rp
from maintenance_arena.banks import ScenarioBankEnv, build_bank

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "synthetic"
if not (EXPERIMENTS / "age_shift_banks.py").is_file():
    pytest.skip("the experiment runners are not available next to the tests", allow_module_level=True)
_environment, _previous = dict(os.environ), sys.modules.get("common")
sys.path.insert(0, str(EXPERIMENTS))
try:
    ab = importlib.import_module("age_shift_banks")  # common.py pins BLAS threads, the test process keeps its own
finally:
    sys.path.remove(str(EXPERIMENTS))
    os.environ.clear()
    os.environ.update(_environment)
    # The helper imports the experiments' ``common`` module; leave no trace for other tests.
    if _previous is None:
        sys.modules.pop("common", None)
    else:
        sys.modules["common"] = _previous

SOURCE = {"kind": "source"}
DOUBLED = {"kind": "scale", "factor": 2}
WIDE = {"kind": "wide", "scale": 0.3}


@pytest.mark.parametrize("law", ["source", "gompertz"])
def test_source_lifetime_reproduces_the_package_banks(law):
    split = "development"
    built = build_bank(law, split=split, seed=91, pair_count=8, sensor_level="gaussian_005")
    again = ab.lifetime_bank(law, split=split, seed=91, pair_count=8, lifetime=SOURCE, sensor_level="gaussian_005")
    assert again["sha256"] == built["sha256"]


@pytest.mark.parametrize("law", ["source", "power_law"])
def test_twins_keep_shape_quantile_and_address(law):
    source = ab.lifetime_bank(law, split="development", seed=92, pair_count=8, lifetime=SOURCE)
    twin = ab.lifetime_bank(law, split="development", seed=92, pair_count=8, lifetime=DOUBLED)
    wide = ab.lifetime_bank(law, split="development", seed=92, pair_count=8, lifetime=WIDE)
    assert twin["measurement_seed"] == source["measurement_seed"] and twin["lifetime"] == DOUBLED
    for a, b, w in zip(source["pairs"], twin["pairs"], wide["pairs"]):
        for side in ("active", "successor"):
            first, second, broad = a[side], b[side], w[side]
            assert first["shape_parameters"] == second["shape_parameters"] == broad["shape_parameters"]
            assert first["stream_address"] == second["stream_address"] == broad["stream_address"]
            assert second["t_cont"] == 2 * first["t_cont"]
            assert second["eol_operating_age"] == math.floor(2 * first["t_cont"] + .5)
            assert broad["t_cont"] == 128. * (first["t_cont"] / 128.) ** 2.5
            health = np.asarray(second["health"], dtype=np.float64)
            assert health[0] == 1. and health[-1] == np.float32(.2) and np.all(np.diff(health) < 0)


def test_lifetime_ranges_of_the_twins_and_the_wide_distribution():
    assert ab.continuous_lifetime(1e-12, DOUBLED) == pytest.approx(191.)
    assert ab.continuous_lifetime(1 - 1e-12, DOUBLED) == pytest.approx(321.)
    assert ab.continuous_lifetime(1e-12, WIDE) == pytest.approx(61.545, abs=1e-3)
    assert ab.continuous_lifetime(1 - 1e-12, WIDE) == pytest.approx(225.358, abs=1e-3)


def test_repeated_float32_health_is_lowered_by_one_step():
    values, changed = ab.strictly_decreasing(np.asarray([1., 1., .9, .2], dtype=np.float32))
    assert changed == 1 and values[1] == np.nextafter(np.float32(1.), np.float32(0.))
    assert np.all(np.diff(values.astype(np.float64)) < 0)


def test_age_input_is_the_clipped_operating_age_and_matches_the_batch_states():
    bank = ab.lifetime_bank("source", split="development", seed=93, pair_count=2, lifetime=DOUBLED)
    series = rp.decision_series(bank)
    batch = ab.age_states(series)
    plain = ScenarioBankEnv(bank, {"recovery_contract": "preserve_preparation_v1"})
    aged = ab.AgeScenarioBankEnv(bank, {"recovery_contract": "preserve_preparation_v1"})
    first, _ = plain.reset()
    observation, _ = aged.reset()
    start, stop = series.slices[0]
    for age in range(stop - start):
        assert observation.shape == (5,) and np.array_equal(observation[:4], first)
        assert observation[4] == np.float32(min(age / 160., 4.))
        assert np.array_equal(observation, batch[start + age])
        first, *_ = plain.step(0)
        observation, *_ = aged.step(0)
    assert ab.age_column(np.asarray([0, 160, 800])).tolist() == [0., 1., 4.]


@pytest.mark.parametrize("route", ["RAW_CAUSAL", "FILTERED_CAUSAL_EWLS8"])
def test_scaled_noise_keeps_the_streams_and_changes_only_the_standard_deviation(route, monkeypatch):
    bank = ab.lifetime_bank("source", split="development", seed=94, pair_count=4, lifetime=DOUBLED,
                            sensor_level="gaussian_005")
    sigma, bias, missing = observations.SENSOR_LEVELS["gaussian_005"]
    same = ab.scaled_series(bank, observation_route=route, sigma=sigma)
    half = ab.scaled_series(bank, observation_route=route, sigma=sigma / 2)
    expected = rp.decision_series(bank, observation_route=route)
    assert np.array_equal(same.states, expected.states) and np.array_equal(same.measured_health, expected.measured_health)
    assert same.slices == expected.slices and same.families == expected.families
    assert np.array_equal(same.last_productive, expected.last_productive)
    monkeypatch.setitem(observations.SENSOR_LEVELS, "gaussian_005", (sigma / 2, bias, missing))
    expected = rp.decision_series(bank, observation_route=route)
    assert np.array_equal(half.states, expected.states) and np.array_equal(half.measured_health, expected.measured_health)


def test_scaled_noise_cells_divide_the_noise_by_the_lifetime_factor():
    config = json.loads((EXPERIMENTS / "age_shift.json").read_text(encoding="utf-8"))
    scaled = [cell for cell in config["scaled_noise"]["cells"] if "noise_factor" in config["cells"][cell]]
    assert scaled == ["Bs", "B4s"]
    for cell in scaled:
        layout = config["cells"][cell]
        assert layout["noise_factor"] * config["lifetimes"][layout["lifetime"]]["factor"] == 1


def test_zero_age_column_gives_the_initial_outputs_of_the_four_input_network():
    pytest.importorskip("torch")
    from maintenance_arena.learning.collector import PROFILE, role_seeds
    from maintenance_arena.learning.ppo import PPOAgent
    seed = 26101601
    four = PPOAgent(PROFILE, seed, role_seeds=role_seeds(seed))
    five = ab.age_agent_class()(seed, role_seeds=role_seeds(seed))
    rng = np.random.default_rng(1)
    states = rng.random((256, 4)).astype(np.float32)
    states[:, 3] = 2 * states[:, 3] - 1
    ages = (4 * rng.random((256, 1))).astype(np.float32)
    for a, b in zip(four.batch_output(states), five.batch_output(np.concatenate([states, ages], axis=1))):
        assert np.array_equal(a, b)
    weights = five.policy_state()["policy_body.0.weight"]
    assert tuple(weights.shape) == (64, 5) and not weights[:, 4].any()


def test_intervals_resample_runs_and_lives_together():
    prices = rp.Economics()
    last = np.asarray([120, 130, 140, 150])
    order = [("linear_reference", i) for i in range(4)]
    rows = [{"method": "m", "group": f"A/{law}", "run": run, "unit": unit, "last_productive_step": int(t),
             "order_step": int(t) - 12 - run % 3} for (law, unit), t in zip(order, last) for run in range(5)]
    table = ab.life_arrays(rows, order, prices)
    tables = {("A", "m"): table, ("A", "n"): table}
    groups = [{"laws": [law for law, _ in order], "cells": ["A"]}]
    samples = ab.runs_and_lives(tables, groups, 5, replicates=40, seed=3, chunk=16)
    points = ab.point_values(tables)
    zero = ab.contrast(samples, points, [(1, "A", "m"), (-1, "A", "n")], "unused_life")
    assert zero == {"estimate": 0., "ci95": [0., 0.]}
    assert points[("A", "m", "failures")] == 0. and samples[("A", "m", "unused_life")].shape == (40,)
