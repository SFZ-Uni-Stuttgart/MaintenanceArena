"""Rule calibration for preparation times other than twelve steps."""
import numpy as np
import pytest

from maintenance_arena import calibration
from maintenance_arena.calibration import _projected_remaining, calibrate, default_grids
from maintenance_arena.core import CoreCycleEnv, policy_action
from maintenance_arena.profiles import resolve_profile


def test_default_grids_follow_the_preparation_time():
    assert default_grids() == default_grids(12) == calibration.default_grids()
    assert default_grids(12)["projection"] == tuple(range(8, 25))
    assert default_grids(24)["projection"] == tuple(range(20, 37))
    assert default_grids(6)["projection"] == tuple(range(2, 19))
    assert default_grids(24)["age"] == default_grids()["age"] and default_grids(6)["health"] == default_grids()["health"]


@pytest.mark.parametrize("lead", [6, 12, 24])
def test_projected_remaining_is_the_projection_rule_of_the_lead(lead):
    env = CoreCycleEnv({"mobilization_lead": lead, "family": "convex_accelerating_fatigue", "eol_operating_age": 128})
    observation, _ = env.reset(seed=3)
    for _ in range(90):
        observation, *_ = env.step(0)
    remaining = _projected_remaining(observation, lead)
    assert np.isfinite(remaining)
    assert policy_action("projection", env, observation, projection_threshold=remaining) == 1
    assert policy_action("projection", env, observation, projection_threshold=np.nextafter(remaining, 0.)) == 0
    if lead == 12:
        assert _projected_remaining(observation) == remaining


@pytest.mark.parametrize("lead", [6, 24])
def test_calibration_scores_other_leads_like_the_simulator(lead):
    profile = resolve_profile("paper-core-continuous-v1", {
        "env_config": {"mobilization_lead": lead, "recovery_contract": "preserve_preparation_v1"},
        "policy_defaults": {"projection": {"projection_threshold": lead + 3}}})
    result = calibrate(profile, rules=("projection",), seed=5, lives=24, criterion="pooled",
                       grids={"projection": list(range(lead - 2, lead + 9))}, bootstrap_samples=50)
    # The full simulator re-runs the selected threshold and must give identical results.
    assert result.verification["performed"] and result.verification["identical"]
    assert result.grids["projection"] == tuple(range(lead - 2, lead + 9))
    rows = {row["threshold"]: row for row in result.table["projection"]}
    # Thresholds below the last safe order L + 1 fail more often than generous ones.
    assert rows[lead - 2]["failures"] >= rows[lead + 8]["failures"]
    assert rows[lead + 8]["order_margin_mean"] > rows[lead - 2]["order_margin_mean"]
    default = calibrate(profile, rules=("projection",), seed=5, lives=24, criterion="pooled", bootstrap_samples=50,
                        verify=False)
    assert default.grids["projection"] == default_grids(lead)["projection"]
