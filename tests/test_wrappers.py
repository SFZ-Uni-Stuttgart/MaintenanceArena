"""The ZeroDeadline and CalendarBudget wrappers and the environments that make_track builds."""
import gymnasium as gym
import numpy as np
import pytest

import maintenance_arena  # noqa: F401  (registers the environment IDs)
from maintenance_arena import evaluation, tracks
from maintenance_arena.core import REPLACE, WAIT
from maintenance_arena.policies import Policy
from maintenance_arena.profiles import resolve_profile
from maintenance_arena.wrappers import CalendarBudget, ZeroDeadline, zero_deadline


CORE = "paper-core-continuous-v1"
GEARBOX = "paper-gearbox-v1"


def _make(profile_id, calendar_steps=None):
    profile = resolve_profile(profile_id)
    env = gym.make(profile["environment_id"], config=evaluation._environment_configuration(profile, calendar_steps))
    env.unwrapped.arena_profile = profile
    return env


def _run(env, actions, seed):
    """Transitions of a fixed action sequence until the episode ends."""
    rows = [env.reset(seed=seed)]
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append((observation, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return rows


# ZeroDeadline.

def test_zero_deadline_zeroes_only_index_3_and_leaves_the_environment_unchanged():
    actions = [WAIT] * 60 + [REPLACE]
    plain = _run(_make(CORE), actions, seed=41)
    wrapped_env = ZeroDeadline(_make(CORE))
    wrapped = _run(wrapped_env, actions, seed=41)
    assert len(plain) == len(wrapped) and plain[-1][2] and wrapped[-1][2]
    assert wrapped_env.observation_space == wrapped_env.unwrapped.observation_space
    nonzero = 0
    for left, right in zip(plain, wrapped, strict=True):
        observation, zeroed = left[0], right[0]
        assert zeroed.dtype == np.float32 and wrapped_env.observation_space.contains(zeroed)
        assert zeroed[3] == 0.0 and np.array_equal(zeroed[:3], observation[:3])
        nonzero += observation[3] != 0
        # Reward, flags and info are passed through untouched.
        assert left[1:] == right[1:] if len(left) == 5 else left[1] == right[1]
    assert nonzero > 50  # the coordinate does carry values in the unwrapped task
    assert wrapped_env.unwrapped.config == _make(CORE).unwrapped.config


@pytest.mark.parametrize("profile_id", ["paper-gearbox-v1", "paper-resource-v1", "paper-resource-lifetime-v1"])
def test_zero_deadline_is_defined_for_the_core_only(profile_id):
    with pytest.raises(ValueError, match="core task"):
        ZeroDeadline(_make(profile_id))


def test_zero_deadline_function_copies_single_and_batched_observations():
    batch = np.arange(20, dtype=np.float32).reshape(5, 4) / 20
    result = zero_deadline(batch)
    assert np.all(result[:, 3] == 0) and np.array_equal(result[:, :3], batch[:, :3])
    assert batch[0, 3] != 0  # the input is not modified
    assert zero_deadline([0.5, 0.1, 1.0, -0.2]).tolist() == pytest.approx([0.5, 0.1, 1.0, 0.0])
    for bad in (np.zeros(3), np.float32(1.0), np.zeros((2, 5))):
        with pytest.raises(ValueError, match="four coordinates"):
            zero_deadline(bad)


# CalendarBudget.

def test_calendar_budget_truncates_on_calendar_steps_like_the_evaluation_loop():
    # The gearbox never ends by itself; its replacements take up to 19 calendar steps.
    budget, seed = 200, 77
    reference = evaluation.rollout(resolve_profile(GEARBOX), policy="separable_q96", seed=seed, calendar_steps=budget)
    env = CalendarBudget(_make(GEARBOX), budget)
    policy = Policy("separable_q96", env.unwrapped.arena_profile, env)
    observation, info = env.reset(seed=seed)
    decisions = cash = 0
    durations = []
    terminated = truncated = False
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(policy(observation, info))
        decisions += 1
        cash += info["raw_profit_sum"]
        durations.append(info["duration_calendar_steps"])
        assert env.elapsed_calendar_steps == sum(durations)
        assert truncated == (env.elapsed_calendar_steps >= budget)
    assert not terminated and truncated
    assert max(durations) > 1 and env.unwrapped.config["horizon"] is None
    assert decisions == reference["decisions"] and env.elapsed_calendar_steps == reference["actual_calendar_steps"]
    assert cash == pytest.approx(reference["raw_profit"], abs=1e-9)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.reset(seed=seed)
    assert env.elapsed_calendar_steps == 0
    env.step(0)


def test_calendar_budget_counts_calendar_steps_not_decisions():
    # A core REPLACE runs to the renewal in one decision of 19 calendar steps, which ends the
    # episode; the environment's termination takes precedence over the budget.
    env = CalendarBudget(_make(CORE), 10)
    env.reset(seed=3)
    _, _, terminated, truncated, info = env.step(REPLACE)
    assert terminated and not truncated and info["duration_calendar_steps"] == 19
    assert env.elapsed_calendar_steps == 19
    env.reset(seed=3)
    for step in range(1, 11):
        _, _, terminated, truncated, _ = env.step(WAIT)
        assert not terminated and truncated == (step == 10)


def test_calendar_budget_passes_every_transition_through_unchanged():
    budget = 150
    actions = [0, 1, 0, 2, 0, 0, 1] * 40
    plain_env = _make(GEARBOX)
    plain = [plain_env.reset(seed=8)]
    elapsed = 0
    for action in actions:
        row = plain_env.step(action)
        plain.append(row)
        elapsed += row[4]["duration_calendar_steps"]
        if elapsed >= budget:
            break
    wrapped = _run(CalendarBudget(_make(GEARBOX), budget), actions, seed=8)
    assert len(plain) == len(wrapped)
    for left, right in zip(plain[1:-1], wrapped[1:-1], strict=True):
        assert np.array_equal(left[0], right[0]) and left[1:] == right[1:]
    assert np.array_equal(plain[-1][0], wrapped[-1][0]) and plain[-1][1] == wrapped[-1][1]
    assert not plain[-1][3] and wrapped[-1][3]  # only the truncation flag differs


@pytest.mark.parametrize("budget", [0, -5, 2.5, True, "720", None])
def test_calendar_budget_rejects_invalid_budgets(budget):
    with pytest.raises(ValueError, match="integer >= 1"):
        CalendarBudget(_make(CORE), budget)


class _NoDuration(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (1,), np.float32)
    action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 0.0, False, False, {}


def test_calendar_budget_needs_the_transition_duration():
    env = CalendarBudget(_NoDuration(), 5)
    env.reset()
    with pytest.raises(ValueError, match="duration_calendar_steps"):
        env.step(0)


# Environments of the tracks.

@pytest.mark.parametrize("name", tracks.track_names())
def test_make_track_builds_the_scored_environment_with_its_budget_and_input(name):
    track = tracks.load_track(name)
    env = tracks.make_track(name)
    scored = evaluation.make_resolved(track.resolved_profile(), track.calendar_steps)  # as the benchmark builds it
    try:
        assert env.unwrapped.arena_profile["configuration_sha256"] == track.profile_configuration_sha256
        assert env.unwrapped.config == scored.unwrapped.config
        layers, layer = [], env
        while layer is not env.unwrapped:
            layers.append(type(layer))
            layer = layer.env
        assert CalendarBudget in layers and (ZeroDeadline in layers) == (track.agent_observation == "zero_deadline")
        observation, info = env.reset(seed=123)
        assert env.observation_space.contains(observation)
        if track.agent_observation == "zero_deadline":
            assert observation[3] == 0
    finally:
        env.close()
        scored.close()


def test_a_gearbox_track_episode_ends_where_the_benchmark_stops():
    track = tracks.load_track("gearbox-v1")
    seed = track.test_seeds[0]
    reference = evaluation.rollout(track.resolved_profile(), policy="dp", seed=seed, calendar_steps=track.calendar_steps)
    env = tracks.make_track(track)
    policy = Policy("dp", env.unwrapped.arena_profile, env)
    observation, info = env.reset(seed=seed)
    decisions, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(policy(observation, info))
        decisions += 1
    assert truncated and decisions == reference["decisions"]
    assert env.get_wrapper_attr("elapsed_calendar_steps") == reference["actual_calendar_steps"]
    env.close()
