"""Masked inference and exact native continuation at optimizer boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from maintenance_arena.resource import ResourceEnv
from maintenance_arena.resource_lifetime import ResourceLifetimeEnv
from maintenance_arena.learning.resource_ppo import (
    load_policy, public_observation, train, make_training_env, create_model,
    _initialize_from_pair, _binding, evaluate_model, check_calendar_budget,
)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal_tree(x, y)
    else:
        assert a == b


def caller_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.get_num_threads()


@pytest.mark.parametrize("cls,profile", [(ResourceEnv, "paper-resource-lifetime-v1"),
                                         (ResourceLifetimeEnv, "paper-resource-v1"),
                                         (ResourceEnv, "unknown-profile")])
def test_resource_training_rejects_a_false_task_identity(tmp_path, cls, profile):
    before = caller_state()
    with pytest.raises(ValueError, match="profile"):
        train(cls, profile=profile, seed=133, calendar_steps=2048, output=tmp_path / "unused")
    equal_tree(before, caller_state())
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("seed", [True, -1, 1.5, 2**32])
def test_resource_training_rejects_invalid_seeds_before_allocating(tmp_path, seed):
    with pytest.raises(ValueError, match="seed"):
        train(lambda: pytest.fail("invalid seed allocated an environment"),
              profile="paper-resource-v1", seed=seed, calendar_steps=2048, output=tmp_path / "unused")


def test_calendar_budget_error_states_the_required_multiple(tmp_path):
    assert check_calendar_budget(4096) == 2048
    assert check_calendar_budget(64, {"n_envs": 2, "hyperparameters": {"n_steps": 16}}) == 32
    with pytest.raises(ValueError, match=r"n_envs \* n_steps = 8 \* 256 = 2048; got 512, use for example 2048$"):
        check_calendar_budget(512)
    with pytest.raises(ValueError, match="got 3000, use for example 2048 or 4096"):
        check_calendar_budget(3000)
    with pytest.raises(ValueError, match="n_envs must be a positive integer"):
        check_calendar_budget(2048, {"n_envs": 0})
    with pytest.raises(ValueError, match="positive cumulative integer"):
        check_calendar_budget(0)
    # Training raises the same message before it creates its output directory.
    with pytest.raises(ValueError, match="= 2048; got 1000"):
        train(ResourceEnv, profile="paper-resource-v1", seed=7, calendar_steps=1000, output=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_unsupported_evaluation_preserves_normalizer_flags():
    class Normalizer:
        training = True
        norm_reward = True
    normalizer = Normalizer()
    with pytest.raises(ValueError, match="unknown resource evaluation"):
        evaluate_model(None, normalizer, {}, [], [], env_class=object)
    assert normalizer.training and normalizer.norm_reward


@pytest.mark.parametrize("cls,profile", [(ResourceEnv, "paper-resource-v1"),
                                         (ResourceLifetimeEnv, "paper-resource-lifetime-v1")])
def test_native_resume_reproduces_uninterrupted_updates_and_masks(tmp_path, cls, profile):
    # The odd horizon crosses auto resets and leaves pending episode rewards
    # and nonzero VecNormalize returns at the saved optimizer boundary.
    configuration = {"n_envs": 2, "horizon": 19,
        "contexts": {"nominal": {}, "slow": {"normal_lead_range": [24, 48]}},
        "hyperparameters": {"n_steps": 16, "batch_size": 16, "n_epochs": 2}}
    factory = lambda: cls({"horizon": 60})
    started = time.perf_counter()
    before = caller_state()
    full = train(factory, profile=profile, seed=471, calendar_steps=128,
                 output=tmp_path / "full", config=configuration)
    equal_tree(before, caller_state())
    first = train(factory, profile=profile, seed=471, calendar_steps=64,
                  output=tmp_path / "first", config=configuration)
    resumed = train(factory, profile=profile, seed=471, calendar_steps=128,
                    output=tmp_path / "resumed", resume=first["checkpoint"], config=configuration)
    equal_tree(before, caller_state())
    print(f"resource {cls.__name__} three training portions: {time.perf_counter()-started:.3f}s")
    assert resumed["new_calendar_steps"] == 64
    assert resumed["checkpoint_relative"] == "checkpoint.json"
    assert Path(resumed["checkpoint"]) == tmp_path / "resumed" / resumed["checkpoint_relative"]
    ma =MaskablePPO.load(tmp_path / "full/model.zip", device="cpu")
    mb = MaskablePPO.load(tmp_path / "resumed/model.zip", device="cpu")
    equal_tree(ma.policy.state_dict(), mb.policy.state_dict())
    equal_tree(ma.policy.optimizer.state_dict(), mb.policy.optimizer.state_dict())
    equal_tree(ma._last_obs, mb._last_obs)
    equal_tree(ma._last_episode_starts, mb._last_episode_starts)
    assert ma._n_updates == mb._n_updates
    sa = torch.load(tmp_path / "full/training_state.pt", weights_only=False)
    sb = torch.load(tmp_path / "resumed/training_state.pt", weights_only=False)
    equal_tree(sa["environment"]["discounted_reward_returns"], sb["environment"]["discounted_reward_returns"])
    equal_tree(sa["environment"]["vector"]["_seeds"], sb["environment"]["vector"]["_seeds"])
    equal_tree(sa["environment"]["vector"]["_options"], sb["environment"]["vector"]["_options"])
    for wa, wb in zip(sa["environment"]["workers"], sb["environment"]["workers"], strict=True):
        # Monitoring wall-clock time has no scientific effect and naturally
        # differs between interrupted and uninterrupted execution.
        for row in (wa, wb):
            row.pop("monitor_elapsed")
            row["monitor"].pop("episode_times")
            row["environment"]["environment"].pop("action_rng_state")
        equal_tree(wa, wb)
    env = factory()
    observation, _ = env.reset(seed=812)
    before_load = caller_state()
    policy = load_policy(resumed["checkpoint"], profile=profile, env=env)
    equal_tree(before_load, caller_state())
    before = deepcopy(policy.normalizer.obs_rms.__dict__)
    for _ in range(40):
        np.testing.assert_array_equal(public_observation(env), observation)
        action = policy.predict(observation)
        assert env.action_masks()[action]
        observation, _, _, _, _ = env.step(action)
    equal_tree(before, policy.normalizer.obs_rms.__dict__)
    assert not policy.normalizer.training and not policy.normalizer.norm_reward
    policy.close()
    env.close()
    wrong = ResourceLifetimeEnv() if cls is ResourceEnv else ResourceEnv()
    with pytest.raises(ValueError, match="observation order"):
        load_policy(resumed["checkpoint"], profile=profile, env=wrong)
    with pytest.raises(ValueError, match="resume cannot change"):
        train(factory, profile=profile, seed=472, calendar_steps=160,
              output=tmp_path / "wrong_seed", resume=resumed["checkpoint"], config=configuration)
    manifest_path = tmp_path / "resumed/checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["normalizer"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="normalizer hash"):
        load_policy(manifest_path, profile=profile, env=wrong)


def test_native_resume_refuses_legacy_weight_only_models(tmp_path):
    metadata = {"kind": "resource_legacy", "profile": "paper-resource-v1", "files": {}}
    target = tmp_path / "checkpoint.json"
    target.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="legacy weights cannot resume"):
        train(ResourceEnv, profile="paper-resource-v1", seed=51, calendar_steps=2048,
              output=tmp_path / "new", resume=target)


def test_explicit_initialization_preserves_coordinate_pair_but_resets_learning_state(tmp_path):
    source_env = make_training_env({}, {"nominal": {}}, 133, n_envs=2)
    target_env = make_training_env({}, {"nominal": {}}, 929, n_envs=2)
    try:
        source = create_model(source_env, 133)
        source_env.obs_rms.mean[:] = .3
        source_env.obs_rms.var[:] = 2.
        source_env.obs_rms.count = 117.
        source_env.ret_rms.mean, source_env.ret_rms.var, source_env.ret_rms.count = 2., 3., 50.
        source.save(tmp_path / "model.zip")
        source_env.save(tmp_path / "normalizer.pkl")
        manifest = {"kind": "resource_legacy", "profile": "paper-resource-v1", "model_seed": 133,
            "files": {"model": _binding(tmp_path / "model.zip"),
                      "normalizer": _binding(tmp_path / "normalizer.pkl")}}
        path = tmp_path / "checkpoint.json"
        path.write_text(json.dumps(manifest))
        target = create_model(target_env, 929)
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
        record = _initialize_from_pair(target, target_env, path, "paper-resource-v1", ResourceEnv, {})
        equal_tree(source.policy.state_dict(), target.policy.state_dict())
        equal_tree(source_env.obs_rms.__dict__, target_env.obs_rms.__dict__)
        equal_tree(source_env.ret_rms.__dict__, target_env.ret_rms.__dict__)
        equal_tree(torch_state, torch.get_rng_state())
        equal_tree(numpy_state, np.random.get_state())
        assert not target.policy.optimizer.state
        assert target.num_timesteps == 0
        assert target.seed == 929
        assert not target_env.returns.any()
        assert target_env.training and target_env.norm_reward
        assert record["source_model_seed"] == 133
        with pytest.raises(ValueError, match="mutually exclusive"):
            train(ResourceEnv, profile="paper-resource-v1", seed=929, calendar_steps=2048,
                  output=tmp_path / "unused", resume=path, initialize=path)
    finally:
        source_env.close()
        target_env.close()
