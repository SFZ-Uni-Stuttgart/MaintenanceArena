"""Gymnasium API compatibility of the four registered environments on the base install."""
import warnings

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import pytest

import maintenance_arena


# The resource observation spaces have intentionally unbounded upper limits
# (counts, hours and costs); Gymnasium's checker warns about that, nothing else.
INFINITE_BOUND = "A Box observation space maximum value is infinity"


@pytest.mark.parametrize("environment_id", sorted(maintenance_arena.ENVIRONMENTS))
def test_registered_environment_passes_the_gymnasium_environment_checker(environment_id):
    env = gym.make(environment_id)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env.unwrapped)
    finally:
        env.close()
    messages = [str(warning.message) for warning in caught]
    assert not [message for message in messages if INFINITE_BOUND not in message], messages
    if "Resource" in environment_id:
        assert any(INFINITE_BOUND in message for message in messages)
    else:
        assert not messages


@pytest.mark.parametrize("environment_id", sorted(maintenance_arena.ENVIRONMENTS))
def test_registered_environment_has_no_hidden_step_limit(environment_id):
    env = gym.make(environment_id)
    try:
        assert env.spec.max_episode_steps is None
        assert env.unwrapped.metadata["render_fps"] == 4
    finally:
        env.close()
