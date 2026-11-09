"""The input sets of the article's ablation as a wrapper and as a function."""
import gymnasium as gym
import numpy as np
import pytest

import maintenance_arena  # noqa: F401  (registers the environment IDs)
from maintenance_arena.wrappers import INPUT_SET_NAMES, INPUT_SETS, InputSet, input_set, zero_deadline


def test_masks_and_names():
    assert INPUT_SET_NAMES == {"full": "Full inputs", "deadline_only": "Deadline only", "health_rate": "Health + rate"}
    observation = np.asarray([.8, .4, 1., -.3], dtype=np.float32)
    np.testing.assert_array_equal(input_set(observation, "full"), observation)
    np.testing.assert_array_equal(input_set(observation, "deadline_only"), np.float32([0., 0., 0., -.3]))
    np.testing.assert_array_equal(input_set(observation, "health_rate"), zero_deadline(observation))
    assert np.signbit(input_set(observation, "health_rate")[3])
    batch = np.stack([observation, observation])
    assert input_set(batch, "deadline_only").shape == (2, 4)
    with pytest.raises(ValueError):
        input_set(observation, "age_only")
    with pytest.raises(ValueError):
        input_set(observation[:3], "full")


@pytest.mark.parametrize("name", sorted(INPUT_SETS))
def test_wrapper_masks_reset_and_step_without_changing_the_task(name):
    plain = gym.make("maintenanceArena/CoreCycle-v0")
    wrapped = InputSet(gym.make("maintenanceArena/CoreCycle-v0"), name)
    assert wrapped.observation_space == plain.observation_space
    first, info = wrapped.reset(seed=4)
    reference, reference_info = plain.reset(seed=4)
    np.testing.assert_array_equal(first, input_set(reference, name))
    for action in (0, 0, 1):
        observation, reward, terminated, truncated, info = wrapped.step(action)
        expected = plain.step(action)
        np.testing.assert_array_equal(observation, input_set(expected[0], name))
        assert (reward, terminated, truncated) == expected[1:4] and info == expected[4]
    with pytest.raises(ValueError):
        InputSet(gym.make("maintenanceArena/Gearbox-v0"), name)
