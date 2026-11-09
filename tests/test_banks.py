"""Finite scenario banks: construction, bindings and the environment that replays them."""
import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from maintenance_arena.banks import (LAW_SETS, ScenarioBankEnv, build_bank, component_measurements, concatenate_banks,
                                     stream_seed, subset_bank, validate_bank, with_sensor_level)
from maintenance_arena.core import policy_action
from maintenance_arena.curves import FAMILY_IDS
from maintenance_arena.replay import Economics, decision_series, replay

PRESERVE = {"recovery_contract": "preserve_preparation_v1"}


def health_digest(component):
    return hashlib.sha256(np.asarray(component["health"], dtype=np.float32).tobytes()).hexdigest()[:16]


def test_article_banks_are_rebuilt_exactly():
    # First lives of calibration banks of the article (law, split and seed fix every life).
    cases = [("source", "source_train", 2257774815, 256, "linear_reference", 123, 123.38109611324109, [],
              "278411b88c605e9d", "f74410fff5bc1b84854c0fd4bea229f40f69c00eada00cf76b14166876189957"),
             ("paris_inspired", "adaptation", 1683905899, 1, "paris_inspired", 132, 132.09021329260295,
              [["w", 0.2559404316500561], ["q", 0.5691461866870089], ["p", 1.071697659123119]], "b6c824e44ba7657f", None),
             ("two_knee", "adaptation", 2768037144, 1, "two_knee", 143, 143.15027125349678,
              [["w", 0.17606709754321898], ["k1", 0.3097817258546663], ["k2", 0.7385146857153242],
               ["v", 0.3485827556013704], ["s", 0.03330537730867278]], "85b670f00b54e59b", None),
             ("gompertz", "adaptation", 4159879460, 1, "gompertz", 112, 111.53050106325466,
              [["A", 5.1016868173937375], ["i", 0.6090147278952823]], "bb768c7c04f5ce09", None),
             ("power_law", "adaptation", 1413115294, 1, "power_law", 135, 134.605934968559,
              [["beta", 3.3273505819470532]], "598f268f054150a6", None)]
    for law, split, seed, pairs, family, eol, t_cont, shape, digest, sha in cases:
        bank = build_bank(law, split=split, seed=seed, pair_count=pairs)
        first = bank["pairs"][0]["active"]
        assert (first["family_id"], first["eol_operating_age"], first["t_cont"], first["shape_parameters"]) == (
            family, eol, t_cont, shape)
        assert health_digest(first) == digest
        if sha is not None:
            assert bank["sha256"] == sha
    assert stream_seed("source", "development", 2026102322, "measurements") == 11808588171289594505


def test_construction_is_deterministic_balanced_and_validated():
    bank = build_bank("source", split="development", seed=91, pair_count=16)
    assert bank == build_bank("source", split="development", seed=91, pair_count=16)
    validate_bank(bank)
    for side in ("active", "successor"):
        assert Counter(pair[side]["family_id"] for pair in bank["pairs"]) == {family: 4 for family in FAMILY_IDS}
    assert all(96 <= pair["active"]["eol_operating_age"] <= 160 for pair in bank["pairs"])
    tampered = copy.deepcopy(bank)
    tampered["pairs"][3]["active"]["health"][5] = .5
    with pytest.raises(ValueError, match="hash"):
        validate_bank(tampered)
    with pytest.raises(ValueError):
        build_bank("gompertz", split="source_train", seed=1, pair_count=4)
    with pytest.raises(ValueError):
        build_bank("source", split="adaptation", seed=1, pair_count=4)
    with pytest.raises(ValueError):
        build_bank("weibull", split="development", seed=1, pair_count=4)
    assert set(LAW_SETS) == {"source", "paris_inspired", "two_knee", "gompertz", "power_law"}


def test_subsets_concatenation_and_sensor_relabelling_keep_lives():
    bank = build_bank("two_knee", split="development", seed=3, pair_count=6)
    part = subset_bank(bank, [4, 1])
    assert [pair["pair_index"] for pair in part["pairs"]] == [4, 1] and part["parent_sha256"] == bank["sha256"]
    other = build_bank("two_knee", split="development", seed=4, pair_count=3)
    joined = concatenate_banks(bank, other)
    assert joined["pair_count"] == 9 and joined["pairs"][6:] == other["pairs"]
    assert joined["measurement_seed"] == bank["measurement_seed"] and len(joined["segments"]) == 2
    with pytest.raises(ValueError, match="share"):
        concatenate_banks(bank, bank)
    noisy = with_sensor_level(bank, "gaussian_005")
    assert noisy["pairs"] == bank["pairs"] and noisy["sha256"] != bank["sha256"]


def run_life(env, order=None, *, threshold=None):
    observation, _ = env.reset()
    observations, age, done = [observation], 0, False
    while not done:
        if threshold is not None:
            action = policy_action("projection", env, observation, projection_threshold=threshold)
        else:
            action = int(order is not None and age == order)
        observation, _, done, truncated, _ = env.step(action)
        assert not truncated
        observations.append(observation)
        age += 1
    return env.diagnostics(), observations


@pytest.mark.parametrize("lead", [6, 12, 24])
@pytest.mark.parametrize("recovery", ["immediate_exchange_v1", "preserve_preparation_v1"])
def test_replayed_first_orders_equal_the_simulator(lead, recovery):
    bank = build_bank("source", split="development", seed=12, pair_count=4)
    economics = Economics(mobilization_lead=lead, recovery_contract=recovery)
    for index, pair in enumerate(bank["pairs"]):
        eol = pair["active"]["eol_operating_age"]
        for order in (0, eol - lead - 2, eol - lead - 1, eol - lead, eol - 1, None):
            env = ScenarioBankEnv(subset_bank(bank, [index]), {"recovery_contract": recovery, "mobilization_lead": lead})
            native, _ = run_life(env, order)
            expected = replay([eol - 1], [-1 if order is None else order], economics)
            assert native["calendar_steps"] == expected["calendar"][0]
            assert native["raw_profit"] == expected["cash"][0]
            assert native["failures"] == int(expected["failed"][0])
            assert native["first_trigger_age"] == order


@pytest.mark.parametrize("sensor", ["exact", "gaussian_005", "light"])
@pytest.mark.parametrize("route", ["RAW_CAUSAL", "FILTERED_CAUSAL_EWLS8"])
def test_environment_observations_equal_the_batch_states(sensor, route):
    bank = build_bank("gompertz", split="development", seed=5, pair_count=3, sensor_level=sensor)
    series = decision_series(bank, observation_route=route, mobilization_lead=24)
    env = ScenarioBankEnv(bank, {"observation_route": route, "mobilization_lead": 24})
    for life in range(3):
        _, observations = run_life(env)
        np.testing.assert_array_equal(np.stack(observations[:series.slices[life][1] - series.slices[life][0]]),
                                      series.life_states(life))
    with pytest.raises(RuntimeError, match="exhausted"):
        env.reset()


def test_measurements_follow_the_component_not_the_policy():
    bank = build_bank("source", split="development", seed=8, pair_count=4, sensor_level="gaussian_005")
    early, late = ScenarioBankEnv(bank, PRESERVE), ScenarioBankEnv(bank, PRESERVE)
    for _ in range(4):
        early.reset()
        early.step(1)
        late.reset()
        for _ in range(30):
            late.step(0)
    assert early._initial["measured_health"] == late._initial["measured_health"]
    expected = component_measurements(bank["pairs"][3]["active"], "gaussian_005", bank["measurement_seed"])
    assert late._initial["measured_health"] == expected["measured_health"].tolist()
    rewound = ScenarioBankEnv(bank, PRESERVE)
    rewound.reset(seed=bank["measurement_seed"])
    other = ScenarioBankEnv(bank, PRESERVE)
    other.reset(seed=7)
    first = component_measurements(bank["pairs"][0]["active"], "gaussian_005", bank["measurement_seed"])
    assert rewound._initial["measured_health"] == first["measured_health"].tolist()
    assert other._initial["health"] == rewound._initial["health"]
    assert other._initial["measured_health"] != rewound._initial["measured_health"]


def test_rewind_diagnostic_start_and_snapshot():
    bank = build_bank("power_law", split="development", seed=6, pair_count=3, sensor_level="light")
    env = ScenarioBankEnv(bank, {**PRESERVE, "observation_route": "FILTERED_CAUSAL_EWLS8"})
    first, _ = env.reset()
    env.reset()
    again, _ = env.reset(seed=bank["measurement_seed"])
    np.testing.assert_array_equal(first, again)
    assert env.bank_cursor == 1 and env.remaining_pair_count == 2
    for _ in range(20):
        env.step(0)
    state = json.loads(json.dumps(env.snapshot(), allow_nan=False))
    restored = ScenarioBankEnv(bank, {**PRESERVE, "observation_route": "FILTERED_CAUSAL_EWLS8"})
    restored.restore(state)
    left, right = env.step(1), restored.step(1)
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1:] == right[1:]
    np.testing.assert_array_equal(env.reset()[0], restored.reset()[0])
    assert env.diagnostics()["pair_index"] == 1 and env.diagnostics()["bank_laws"] == "power_law"
    with pytest.raises(ValueError, match="another scenario bank"):
        ScenarioBankEnv(build_bank("power_law", split="development", seed=7, pair_count=3, sensor_level="light"),
                        {**PRESERVE, "observation_route": "FILTERED_CAUSAL_EWLS8"}).restore(state)
    start = ScenarioBankEnv(bank)
    observation, _ = start.reset(options={"start_age": 50})
    assert start.policy_state()["operating_age"] == 50 and bank["pairs"][0]["start_age"] == 0
    series = decision_series(bank, observation_route="RAW_CAUSAL")
    np.testing.assert_array_equal(observation, series.life_states(0)[50])


def test_bank_owns_components_and_sensor_level():
    bank = build_bank("source", split="development", seed=1, pair_count=4)
    for config in ({"family": "gompertz"}, {"eol_operating_age": 100}, {"sensor_level": "light"},
                   {"training_start_offsets": [13]}):
        with pytest.raises(ValueError):
            ScenarioBankEnv(bank, config)
    with pytest.raises(RuntimeError):
        ScenarioBankEnv(bank).configure_training(seed=1)
