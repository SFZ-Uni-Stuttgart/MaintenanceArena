"""Core mechanics, accounting and resumability."""
import json
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from maintenance_arena.core import CoreCycleEnv, policy_action, validate_config
from maintenance_arena.curves import FAMILY_IDS, build_training_bank, START_OFFSETS


def fixed(**kwargs):
    return CoreCycleEnv({"source": "discrete", "family": "linear_reference", "eol_operating_age": 128, **kwargs})


@pytest.mark.parametrize("remaining,productive,failure", [(13, 12, False), (12, 11, True), (1, 0, True)])
def test_exact_replacement_boundary_and_exchange(remaining, productive, failure):
    env = fixed()
    env.reset(seed=3, options={"start_age": 128 - remaining})
    obs, reward, terminated, truncated, info = env.step(1)
    expected = [255.] * productive + [-7545. if failure else -545.] + [-25.] * 6
    assert [row["raw_profit"] for row in info["ledger"]] == expected
    assert info["duration_calendar_steps"] == len(expected)
    assert info["gamma_pow_duration"] == .999 ** len(expected)
    assert reward == math.fsum(.999 ** i * (r / 1000.) for i, r in enumerate(expected))
    assert terminated and not truncated
    assert info["failures"] == int(failure)
    np.testing.assert_array_equal(obs, [1., 0., 0., -1.])
    trace = env.diagnostics()["trace"]
    assert trace[productive]["component_before"] != trace[productive]["component_after"]
    assert trace[productive]["operating_age_after"] == 0
    assert all(row["operating_age_after"] == 0 for row in trace[productive:])
    with pytest.raises(RuntimeError):
        env.step(0)


def test_reactive_calendar_costs_and_no_truth_in_public_info():
    env = fixed()
    obs, info = env.reset(seed=4)
    assert obs.dtype == np.float32
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy_action("reactive", env, obs))
        forbidden = {"family_id", "operating_age", "true_rul", "eol_operating_age", "true_health", "draw_id"}
        assert not forbidden.intersection(info)
        assert all(not forbidden.intersection(row) for row in info["ledger"])
    diag = env.diagnostics()
    assert diag["calendar_steps"] == 134
    assert diag["raw_profit"] == 127 * 255 - 520 - 175 - 7000
    assert diag["first_trigger_true_rul"] is None
    assert diag["failures"] == 1


@pytest.mark.parametrize("remaining,productive,preparation", [(13, 12, 0), (12, 11, 1), (7, 6, 6), (1, 0, 12)])
def test_preserved_preparation_boundary_costs_and_primitive_parity(remaining, productive, preparation):
    config = {"recovery_contract": "preserve_preparation_v1"}
    macro, primitive = fixed(**config), fixed(**config)
    for env in (macro, primitive):
        env.reset(seed=3, options={"start_age": 128 - remaining})
    obs, reward, terminated, truncated, info = macro.step(1)
    failure = preparation > 0
    expected = [255.] * productive
    if failure:
        expected += [-7025.] + [-25.] * (preparation - 1)
    expected += [-545.] + [-25.] * 6
    assert [row["raw_profit"] for row in info["ledger"]] == expected
    assert info["duration_calendar_steps"] == 19
    assert info["gamma_pow_duration"] == .999 ** 19
    assert reward == math.fsum(.999 ** i * (r / 1000.) for i, r in enumerate(expected))
    assert info["episode_raw_profit"] == sum(expected)
    assert terminated and not truncated
    assert info["failures"] == info["corrective_replacements"] == int(failure)
    assert info["planned_replacements"] == int(not failure)
    rows = [primitive._primitive(1)]
    while primitive._order_pending or primitive._downtime_remaining:
        rows.append(primitive._primitive(0))
    assert rows == info["ledger"]
    trace = macro.diagnostics()["trace"]
    assert trace == primitive.diagnostics()["trace"]
    assert all(row["component_before"] == row["component_after"] for row in trace[:12])
    assert trace[12]["component_before"] != trace[12]["component_after"]
    assert trace[12]["corrective_replacement"] == failure
    assert sum(row["failure"] for row in trace) == int(failure)
    assert sum(row["replacement_cost"] for row in trace) == 520
    if failure:
        assert trace[productive]["failure"] and not trace[productive]["corrective_replacement"]
        assert all(row["operating_age_after"] == 128 for row in trace[productive:12])
        assert all(row["revenue"] == 0 for row in trace[productive:])
    np.testing.assert_array_equal(obs, [1., 0., 0., -1.])


def test_preserved_preparation_reactive_starts_full_lead_after_failure():
    env = fixed(recovery_contract="preserve_preparation_v1", calendar_horizon=129)
    env.reset(seed=4)
    done = False
    while not done:
        _, _, done, truncated, info = env.step(0)
        assert not truncated
    assert [row["event"] for row in info["ledger"]] == ["failure"] + ["corrective_mobilize"] * 11 + ["corrective_exchange"] + ["downtime"] * 6
    assert info["duration_calendar_steps"] == 19
    assert info["calendar_steps"] == 146 and info["calendar_overshoot"] == 17
    assert info["episode_raw_profit"] == 127 * 255 - 520 - 19 * 25 - 7000
    assert info["failures"] == info["corrective_replacements"] == 1
    assert env.diagnostics()["first_trigger_true_rul"] is None
    assert sum(row["order_placed"] for row in env.diagnostics()["trace"]) == 0


def test_preserved_preparation_snapshot_during_failed_recovery():
    original = fixed(recovery_contract="preserve_preparation_v1", sensor_level="light")
    original.reset(seed=45, options={"start_age": 125})
    for action in (1, 0, 0):
        original._primitive(action)
    assert original._failures == 1 and original._corrective == 0 and not original._renewed
    assert original._mobilization_remaining == 9 and original._age == 128
    old_component = original._active["draw_index"]
    state = json.loads(json.dumps(original.snapshot(), allow_nan=False))
    restored = fixed(recovery_contract="preserve_preparation_v1", sensor_level="light")
    restored.restore(state)
    left, right = original.step(0), restored.step(0)
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1:] == right[1:]
    assert original.snapshot() == restored.snapshot()
    ledger = left[-1]["ledger"]
    assert len(ledger) == 16 and not any(row["failure"] for row in ledger)
    assert sum(row["failure_cost"] for row in ledger) == 0
    assert original._active["draw_index"] != old_component and original._age == 0
    assert not original._corrective_pending
    np.testing.assert_array_equal(original.reset()[0], restored.reset()[0])
    assert original.snapshot() == restored.snapshot()


def test_legacy_recovery_configuration_and_snapshot_remain_compatible():
    assert validate_config({"recovery_contract": "immediate_exchange_v1"}) == validate_config(None)
    env = fixed()
    env.reset(seed=7, options={"start_age": 127})
    snapshot = env.snapshot()
    snapshot["state"].pop("_corrective_pending")
    restored = fixed()
    restored.restore(snapshot)
    assert env.step(0)[1:] == restored.step(0)[1:]
    with pytest.raises(ValueError, match="configuration differs"):
        fixed(recovery_contract="preserve_preparation_v1").restore(snapshot)


@pytest.mark.parametrize("route", ["RAW_CAUSAL", "FILTERED_CAUSAL_EWLS8", "ROBUST_MULTISCALE_234"])
@pytest.mark.parametrize("sensor_level", ["light", "gaussian_005"])
def test_snapshot_json_roundtrip_keeps_rng_reconstruction_and_next_cycle(route, sensor_level):
    config = {"observation_route": route, "sensor_level": sensor_level}
    original, restored = CoreCycleEnv(config), CoreCycleEnv(config)
    original.reset(seed=45)
    for _ in range(41):
        original.step(0)
    state = json.loads(json.dumps(original.snapshot(), allow_nan=False))
    restored.restore(state)
    for action in (0, 0, 1):
        left, right = original.step(action), restored.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]
    assert original.diagnostics() == restored.diagnostics()
    left, right = original.reset(), restored.reset()
    np.testing.assert_array_equal(left[0], right[0])
    assert original.snapshot() == restored.snapshot()


@pytest.mark.parametrize("recovery", ["immediate_exchange_v1", "preserve_preparation_v1"])
def test_gaussian_sensor_keeps_latent_stream_clock_and_fixed_action_economics(recovery):
    exact = CoreCycleEnv({"recovery_contract": recovery})
    noisy = CoreCycleEnv({"recovery_contract": recovery, "sensor_level": "gaussian_005"})
    replay = CoreCycleEnv({"recovery_contract": recovery, "sensor_level": "gaussian_005"})
    for cycle in range(3):
        seed = 917 if cycle == 0 else None
        exact_obs, _ = exact.reset(seed=seed)
        noisy_obs, _ = noisy.reset(seed=seed)
        replay_obs, _ = replay.reset(seed=seed)
        np.testing.assert_array_equal(noisy_obs, replay_obs)
        assert exact._active["health"] == noisy._active["health"]
        assert exact._successor["health"] == noisy._successor["health"]
        assert exact._active["measured_health"] == exact._active["health"]
        assert exact._active["measured_health"] != noisy._active["measured_health"]
        saved = noisy.snapshot()
        noisy.policy_state()
        noisy.diagnostics()
        noisy.render()
        assert noisy.snapshot() == saved  # Inspection must not draw another noisy reading.
        step = 0
        done = False
        while not done:
            # Reactive, planned and late orders exercise the same latent clocks.
            trigger = None if cycle == 0 else (10 if cycle == 1 else exact._active["eol_operating_age"] - 7)
            action = int(trigger is not None and step >= trigger)
            clean_result, noisy_result, replay_result = exact.step(action), noisy.step(action), replay.step(action)
            assert clean_result[1:4] == noisy_result[1:4]
            for key in ("ledger", "events", "episode_raw_profit", "failures", "calendar_steps"):
                assert clean_result[-1][key] == noisy_result[-1][key]
            np.testing.assert_array_equal(noisy_result[0], replay_result[0])
            assert noisy_result[1:] == replay_result[1:]
            done = clean_result[2]
            step += 1
        assert noisy.snapshot() == replay.snapshot()


def test_calendar_limit_and_full_macro_overshoot():
    env = fixed(calendar_horizon=3)
    env.reset(seed=1)
    assert not env.step(0)[3]
    assert not env.step(0)[3]
    result = env.step(0)
    assert result[3] and not result[2]
    env.reset(seed=1)
    _, _, terminated, truncated, info = env.step(1)
    assert terminated and not truncated
    assert info["calendar_steps"] == 19 and info["calendar_overshoot"] == 16


def test_strong_rules_use_only_causal_policy_information():
    env = fixed()
    obs, _ = env.reset(seed=6, options={"start_age": 83})
    assert policy_action("age", env, obs, age_interval=83) == 1
    assert policy_action("health", env, obs, health_threshold=.01) == 0
    assert policy_action("projection", env, obs, projection_threshold=100) == 1
    # Changing evaluator-only metadata cannot affect the rule.
    env._initial["eol_operating_age"] = 999999
    assert policy_action("age", env, obs, age_interval=83) == 1
    assert policy_action("projection", env, obs, projection_threshold=100) == 1


def test_reset_seed_and_gym_interface():
    env = CoreCycleEnv()
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=29)
    curve = env.diagnostics()["eol_operating_age"]
    env.step(0)
    second, _ = env.reset(seed=29)
    np.testing.assert_array_equal(first, second)
    assert curve == env.diagnostics()["eol_operating_age"]
    for invalid in (True, -1, 2, .5):
        with pytest.raises(ValueError):
            env.step(invalid)


@pytest.mark.parametrize("seed", [None, 73])
def test_rejected_reset_preserves_active_cycle_and_random_streams(seed):
    env = fixed(sensor_level="light", observation_route="ROBUST_MULTISCALE_234")
    env.reset(seed=29)
    for _ in range(5):
        env.step(0)
    saved = env.snapshot()
    for options in ({"unknown": 1}, {"start_age": True}, {"start_age": 128}):
        with pytest.raises(ValueError):
            env.reset(seed=seed, options=options)
        assert env.snapshot() == saved
    reference = fixed(sensor_level="light", observation_route="ROBUST_MULTISCALE_234")
    reference.restore(saved)
    actual, expected = env.step(1), reference.step(1)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    np.testing.assert_array_equal(env.reset()[0], reference.reset()[0])
    assert env.snapshot() == reference.snapshot()


def test_registered_environment_snapshot_excludes_gym_and_profile_metadata():
    import maintenance_arena
    env = maintenance_arena.make("paper-core-continuous-v1").unwrapped
    env.reset(seed=39)
    snapshot = json.loads(json.dumps(env.snapshot(), allow_nan=False))
    assert "spec" not in snapshot["state"] and "arena_profile" not in snapshot["state"]
    restored = maintenance_arena.make("paper-core-continuous-v1").unwrapped
    np.testing.assert_array_equal(restored.restore(snapshot), env._observation)


def test_finite_continuous_training_blocks_lhs_and_resumable_consumption():
    from collections import Counter
    bank = build_training_bank(901, pair_count=64)
    for start in (0, 32):
        cells = [(p["active"]["family_id"], p["start_slot_index"]) for p in bank["pairs"][start:start + 32]]
        assert len(set(cells)) == 32
    for family in FAMILY_IDS:
        rows = [p["active"] for p in bank["pairs"] if p["active"]["family_id"] == family]
        assert sorted(row["lifetime_stratum_index"] for row in rows) == list(range(16))
        if family != "linear_reference":
            assert sorted(row["shape_stratum_index"] for row in rows) == list(range(16))
    assert Counter(p["successor"]["family_id"] for p in bank["pairs"]) == {f: 16 for f in FAMILY_IDS}
    env = CoreCycleEnv()
    env.configure_training(seed=901, pair_count=32)
    env.reset(seed=81)
    env.step(1)
    snapshot = json.loads(json.dumps(env.snapshot(), allow_nan=False))
    assert "_training_bank" not in snapshot["state"]
    restored = CoreCycleEnv()
    restored.restore(snapshot)
    for _ in range(31):
        left, right = env.reset(), restored.reset()
        np.testing.assert_array_equal(left[0], right[0])
        assert env.step(1)[1:] == restored.step(1)[1:]
    with pytest.raises(RuntimeError, match="exhausted"):
        env.reset()


def test_discrete_training_schedule_preserves_96_cells_and_successors():
    bank = build_training_bank(902, source="discrete", pair_count=96)
    expected = [(slot, eol, family) for slot in range(8) for eol in (112, 128, 144) for family in FAMILY_IDS]
    for pair, (slot, eol, family) in zip(bank["pairs"], expected, strict=True):
        assert pair["start_slot_index"] == slot
        assert pair["active"]["family_id"] == family
        assert pair["active"]["eol_operating_age"] == eol
        offset = START_OFFSETS[slot]
        assert pair["start_age"] == (0 if offset is None else eol - offset)
        np.testing.assert_array_equal(pair["active"]["health"], pair["successor"]["health"])


@pytest.mark.parametrize("config", [{"gamam": .9}, {"mobilization_lead": 10}, {"downtime_steps": 7}, {"gamma": 0}, {"reward_scale": 0}, {"eol_operating_age": True}, {"recovery_contract": "restart_preparation"}])
def test_invalid_config_does_not_silently_change_contract(config):
    with pytest.raises(ValueError):
        CoreCycleEnv(config)
