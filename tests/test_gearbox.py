from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from maintenance_arena.gearbox import GearboxEnv, exact_model, filtered_action, primitive_rewards
from maintenance_arena.gearbox_filter import build_filter
from maintenance_arena import gearbox_model


def test_transition_kernel_matches_preserved_original_source_fixture():
    """Portable identity of the 8,640-transition kernel.

    The fixture was generated from an independent reference implementation of the
    model, not from this package. The test needs only the fixture's array hashes.
    """
    fixture = json.loads((Path(__file__).parent / "fixtures/gearbox_transition_identity.json")
                         .read_text(encoding="utf-8"))
    assert fixture["schema_version"] == "maintenance_arena_gearbox_transition_identity_v1"
    model = exact_model()
    action_map, audit = build_filter(model)
    arrays = {name: getattr(model, name) for name in
              ("reward_raw", "duration", "renews", "next_damage", "immediate_metrics")}
    arrays["action_filter"] = action_map
    arrays["observations"] = np.asarray([
        [gearbox_model.observation(model.spec, damage, opportunity) for opportunity in (0, 1)]
        for damage in range(model.spec.damage_capacity)])
    assert asdict(model.spec) == fixture["model_spec"]
    assert list(gearbox_model.METRIC_NAMES) == fixture["metric_names"]
    assert model.reward_raw.size == fixture["transition_count"] == 8640
    assert audit == fixture["filter_audit"]
    assert set(arrays) == set(fixture["arrays"])
    for name, value in arrays.items():
        canonical = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
        expected = fixture["arrays"][name]
        assert canonical.dtype.str == expected["dtype"], name
        assert list(canonical.shape) == expected["shape"], name
        assert hashlib.sha256(canonical.tobytes(order="C")).hexdigest() == expected["sha256"], name
    for name, value in fixture["scalar_audits"].items():
        assert getattr(model, name) == value


def test_all_8640_macro_transitions_and_accounting():
    env = GearboxEnv()
    model = exact_model()
    checked = 0
    for damage in range(1440):
        for opportunity in (0, 1):
            for action in (0, 1, 2):
                env.reset(seed=987, options={"damage": damage, "opportunity": opportunity})
                obs, reward, terminated, truncated, info = env.step(action)
                index = damage, opportunity, action
                rows = primitive_rewards(model, *index)
                assert info["duration_calendar_steps"] == model.duration[index] == len(rows)
                assert info["raw_profit_sum"] == math.fsum(x / 100 for x in rows)
                assert math.fsum(rows) == model.immediate_metrics[index][0]
                assert reward == math.fsum(.999 ** k * (x / 100) / 1000 for k, x in enumerate(rows))
                assert info["gamma_pow_duration"] == .999 ** len(rows)
                assert not terminated and not truncated
                next_damage = 0 if model.renews[index] else model.next_damage[index]
                np.testing.assert_array_equal(obs, gearbox_model.observation(model.spec, next_damage, int(env.policy_state()["opportunity"])))
                ledger = info["ledger"]
                assert ledger["net_profit"] == pytest.approx(ledger["revenue"] - ledger["replacement_cost"] - ledger["downtime_cost"] - ledger["failure_cost"], abs=1e-10)
                checked += 1
    assert checked == 8640
    env.close()


def test_boundary_real_successor_cutoff_and_resume_snapshot():
    env = GearboxEnv({"horizon": 10})
    env.reset(seed=531, options={"damage": 1355, "opportunity": 1})
    saved = env.snapshot()
    expected_rng = np.random.Generator(np.random.PCG64())
    expected_rng.bit_generator.state = deepcopy(saved["rng_state"])
    expected_opportunity = int(expected_rng.integers(0, 2))
    first = env.step(2)
    assert first[2:4] == (False, True)
    assert first[4]["failures"] == 0
    assert first[4]["duration_calendar_steps"] == 19
    assert first[4]["calendar_budget"] == 10
    assert first[4]["calendar_overshoot"] == 9
    assert int(env.policy_state()["opportunity"]) == expected_opportunity
    assert env.snapshot()["rng_state"] == expected_rng.bit_generator.state
    with pytest.raises(RuntimeError):
        env.step(0)
    restored = GearboxEnv({"horizon": 10})
    restored.restore(saved)
    second = restored.step(2)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    env.reset(seed=531, options={"damage": 1356, "opportunity": 1})
    late = env.step(2)
    assert late[4]["failures"] == 1
    assert late[4]["duration_calendar_steps"] == 18
    with pytest.raises(ValueError):
        GearboxEnv({"gamma": .99})
    with pytest.raises(ValueError):
        GearboxEnv().restore(saved)


def test_filter_is_separate_and_boundary_has_no_rescue_below85():
    env = GearboxEnv()
    action_map, audit = build_filter(exact_model())
    assert all(audit["checks"].values())
    for q in (84, 85, 91, 96, 97, 200):
        env.reset(seed=2, options={"damage": 1440 - q, "opportunity": 1})
        for proposed in range(3):
            assert filtered_action(env, proposed) == action_map[1440 - q, 1, proposed]
    env.reset(seed=2, options={"damage": 1355, "opportunity": 1})
    assert filtered_action(env, 0) == 2
    # The base simulator accepts the raw unsafe proposal unchanged.
    _, _, _, _, info = env.step(0)
    assert info["events"][0]["action"] == 0
    assert env.policy_state()["remaining_damage"] == 73
    env.reset(seed=2, options={"damage": 1356, "opportunity": 1})
    assert filtered_action(env, 0) == 0


def test_gymnasium_interface_and_reset_required():
    env = GearboxEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    check_env(env, skip_render_check=True)
    for invalid in (True, -1, 3, .5):
        with pytest.raises(ValueError):
            env.step(invalid)


def test_reset_rejection_is_harmless_and_render_respects_mode():
    env = GearboxEnv()
    assert env.render() is None
    visible = GearboxEnv(render_mode="ansi")
    assert "reset required" in visible.render()
    env.reset(seed=51)
    env.step(0)
    saved = env.snapshot()
    for options in ({"unknown": 1}, {"damage": 1440}, {"damage": 7, "opportunity": 2}):
        with pytest.raises(ValueError):
            env.reset(seed=82, options=options)
        assert env.snapshot() == saved
    assert env.render() is None
    visible.reset(seed=51, options={"damage": np.int64(7), "opportunity": np.int64(1)})
    assert "remaining_damage=1433" in visible.render()
    _, info = visible.reset(seed=51)
    assert info["calendar_budget"] is None and info["calendar_overshoot"] == 0
