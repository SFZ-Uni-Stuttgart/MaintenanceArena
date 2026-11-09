"""The preparation time L as an option of the core task: event clock, observations and rules."""
import json
import math

import numpy as np
import pytest

from maintenance_arena.core import MOBILIZATION_LEADS, CoreCycleEnv, policy_action, validate_config
from maintenance_arena.curves import materialize_health
from maintenance_arena.observations import ROUTE_IDS, Reconstructor, observed_damage, reconstruct, valid_state


def fixed(lead, **kwargs):
    return CoreCycleEnv({"source": "discrete", "family": "linear_reference", "eol_operating_age": 128,
                         "mobilization_lead": lead, **kwargs})


def test_supported_leads_and_unchanged_default():
    assert MOBILIZATION_LEADS == (6, 12, 24)
    assert validate_config({"mobilization_lead": 12}) == validate_config(None)
    for lead in MOBILIZATION_LEADS:
        assert validate_config({"mobilization_lead": lead})["mobilization_lead"] == lead
    for bad in (0, 10, 13, True, 12.):
        with pytest.raises(ValueError):
            validate_config({"mobilization_lead": bad})
    # The package's own training curriculum is defined for the published lead only.
    with pytest.raises(ValueError, match="mobilization_lead=12"):
        CoreCycleEnv({"mobilization_lead": 6}).configure_training(seed=1, pair_count=32)
    assert CoreCycleEnv().configure_training(seed=1, pair_count=32)["pair_count"] == 32


@pytest.mark.parametrize("lead", MOBILIZATION_LEADS)
@pytest.mark.parametrize("recovery", ["immediate_exchange_v1", "preserve_preparation_v1"])
def test_last_safe_order_is_lead_plus_one_steps_before_failure(lead, recovery):
    for remaining in (lead + 1, lead):
        env = fixed(lead, recovery_contract=recovery)
        env.reset(seed=2, options={"start_age": 128 - remaining})
        _, _, terminated, _, info = env.step(1)
        failure = remaining == lead
        assert terminated and info["failures"] == int(failure)
        profits = [row["raw_profit"] for row in info["ledger"]]
        if not failure:
            assert profits == [255.] * lead + [-545.] + [-25.] * 6
        elif recovery == "preserve_preparation_v1":
            assert profits == [255.] * (lead - 1) + [-7025.] + [-545.] + [-25.] * 6
        else:
            assert profits == [255.] * (lead - 1) + [-7545.] + [-25.] * 6


@pytest.mark.parametrize("lead", MOBILIZATION_LEADS)
def test_reactive_failure_waits_for_the_whole_preparation(lead):
    env = fixed(lead, recovery_contract="preserve_preparation_v1")
    env.reset(seed=4)
    done = False
    while not done:
        _, _, done, _, info = env.step(0)
    assert info["duration_calendar_steps"] == lead + 7
    assert info["episode_raw_profit"] == 127 * 255 - 520 - (lead + 7) * 25 - 7000


def test_observation_scales_rate_and_deadline_with_the_lead():
    for health, rate in ((.8, .3), (.4, .9), (.25, .02), (.9, 0.)):
        assert valid_state(health, rate).tobytes() == valid_state(health, rate, 12).tobytes()
        for lead in MOBILIZATION_LEADS:
            state = valid_state(health, rate, lead)
            remaining = (1. - observed_damage(float(state[0]))) / max(float(state[1]) / lead, 1e-12)
            assert state[3] == np.float32(np.clip((lead + 1. - remaining) / (lead + 1.), -1., 1.))
    values = materialize_health("convex_accelerating_fatigue", 128)
    for lead in MOBILIZATION_LEADS:
        states = reconstruct(values, None, "RAW_CAUSAL", lead)["states"]
        increments = np.diff([observed_damage(v) for v in values])
        np.testing.assert_array_equal(states[1:, 1], np.float32(np.clip(lead * np.maximum(increments, 0.), 0., 1.)))
        np.testing.assert_array_equal(states[0], [1., 0., 0., -1.])


@pytest.mark.parametrize("route", ROUTE_IDS)
def test_routes_share_the_estimate_and_differ_only_in_scaling(route):
    rng = np.random.default_rng(9)
    values = np.clip(materialize_health("bearing_late_knee", 128) + rng.normal(0., .005, 129), 0., 1.).astype(np.float32)
    available = rng.random(129) > .05
    available[0] = True
    base = reconstruct(values, available, route)["states"]
    for lead in (6, 24):
        states = reconstruct(values, available, route, lead)["states"]
        np.testing.assert_array_equal(states[:, [0, 2]], base[:, [0, 2]])
        valid = base[:, 2] > .5
        unsaturated = valid & (base[:, 1] < .95) & (states[:, 1] < .95)
        np.testing.assert_allclose(states[unsaturated, 1] / lead, base[unsaturated, 1] / 12, rtol=1e-6)
        np.testing.assert_array_equal(states[~valid], base[~valid])


def test_reconstructor_snapshot_fields_and_lead_check():
    assert "mobilization_lead" not in Reconstructor().snapshot()
    other = Reconstructor("RAW_CAUSAL", 24)
    other.step(1.)
    snapshot = other.snapshot()
    assert snapshot["mobilization_lead"] == 24
    with pytest.raises(ValueError, match="lead"):
        Reconstructor("RAW_CAUSAL").restore(snapshot)
    restored = Reconstructor("RAW_CAUSAL", 24)
    restored.restore(snapshot)
    assert restored.step(.99).tobytes() == other.step(.99).tobytes()
    with pytest.raises(ValueError):
        Reconstructor("RAW_CAUSAL", 0)


@pytest.mark.parametrize("route", ["RAW_CAUSAL", "FILTERED_CAUSAL_EWLS8"])
def test_snapshot_roundtrip_with_another_lead(route):
    config = {"mobilization_lead": 24, "observation_route": route, "sensor_level": "light",
              "recovery_contract": "preserve_preparation_v1"}
    original, restored = CoreCycleEnv(config), CoreCycleEnv(config)
    original.reset(seed=45)
    for _ in range(30):
        original.step(0)
    restored.restore(json.loads(json.dumps(original.snapshot(), allow_nan=False)))
    left, right = original.step(1), restored.step(1)
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1:] == right[1:]
    assert left[-1]["duration_calendar_steps"] <= 31
    with pytest.raises(ValueError, match="configuration differs"):
        CoreCycleEnv({**config, "mobilization_lead": 12}).restore(original.snapshot())


def test_projection_rule_uses_the_lead_and_needs_a_threshold():
    env = fixed(6)
    observation, _ = env.reset(seed=1, options={"start_age": 100})
    with pytest.raises(ValueError, match="explicit threshold"):
        policy_action("projection", env, observation)
    health, rate = float(observation[0]), float(observation[1])
    remaining = (1. - (-math.log(health) / math.log(5.))) / max(rate / 6, 1e-12)
    assert policy_action("projection", env, observation, projection_threshold=remaining) == 1
    assert policy_action("projection", env, observation, projection_threshold=remaining - 1e-9) == 0
    default = fixed(12)
    observation, _ = default.reset(seed=1, options={"start_age": 100})
    assert policy_action("projection", default, observation) == policy_action("projection", default, observation,
                                                                              projection_threshold=14)
