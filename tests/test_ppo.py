from pathlib import Path
import time

import pytest

torch = pytest.importorskip("torch")
from maintenance_arena.gearbox import GearboxEnv
from maintenance_arena.learning.ppo import duration_aware_gae, train
from maintenance_arena.learning.checkpoints import atomic_write, load_native, load_policy


def test_checkpoint_publication_is_atomic_and_exclusive(tmp_path):
    target = tmp_path / "checkpoint.pt"
    def interrupted(handle):
        handle.write(b"incomplete checkpoint")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        atomic_write(target, interrupted)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
    def reject(candidate):
        assert candidate.read_bytes() == b"unverified"
        raise ValueError("roundtrip validation failed")
    with pytest.raises(ValueError, match="roundtrip"):
        atomic_write(target, lambda handle: handle.write(b"unverified"), reject)
    assert not target.exists()
    atomic_write(target, lambda handle: handle.write(b"complete"))
    with pytest.raises(FileExistsError):
        atomic_write(target, lambda handle: handle.write(b"replacement"))
    assert target.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [target]


def test_calendar_gamma_decision_lambda_and_renewal_masks():
    values = duration_aware_gae(macro_rewards=[1., 2.], values=[.2, .3], next_values=[.3, .4],
                               durations=[1, 19], bootstrap_masks=[1., 0.], trace_masks=[1., 0.],
                               gamma=.999, gae_lambda_per_decision=.95)
    assert values.deltas[1] == pytest.approx(1.7)
    assert values.advantages[0] == pytest.approx(1 + .999 * .3 - .2 + .999 * .95 * 1.7)
    continuing = duration_aware_gae(macro_rewards=[2.], values=[.3], next_values=[.4],
                                   durations=[19], bootstrap_masks=[1.], trace_masks=[0.],
                                   gamma=.999, gae_lambda_per_decision=.95)
    assert continuing.advantages[0] == pytest.approx(2 + .999 ** 19 * .4 - .3)


def test_initialization_and_exact_resume_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(lambda: None, profile="paper-gearbox-v1", seed=1,
              calendar_steps=1, output="must-not-be-created", resume="old.pt", initialize="weights.pt")


@pytest.mark.parametrize("profile", ["paper-gearbox-v1", "paper-core-continuous-v1"])
def test_native_checkpoint_resume_matches_uninterrupted_updates(tmp_path, profile):
    started = time.perf_counter()
    if profile == "paper-gearbox-v1":
        factory = lambda: GearboxEnv()
        altered_config = {"horizon": 100, "gamma": .999, "reward_scale": 1000.}
    else:
        from maintenance_arena.core import CoreCycleEnv
        factory = lambda: CoreCycleEnv()
        altered_config = {"calendar_horizon": 100}
    whole = train(factory, profile=profile, seed=832016, calendar_steps=2048, output=tmp_path / "whole")
    checkpoint = sorted((tmp_path / "whole").glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[0]
    resumed = train(factory, profile=profile, seed=832016, calendar_steps=2048, output=tmp_path / "resumed", resume=checkpoint)
    assert whole["final_parameter_sha256"] == resumed["final_parameter_sha256"]
    assert whole["checkpoint_relative"] == resumed["checkpoint_relative"] == "final.pt"
    assert Path(whole["checkpoint"]) == (tmp_path / "whole" / whole["checkpoint_relative"]).resolve()
    final_a = load_native(whole["checkpoint"], profile=profile)
    final_b = load_native(resumed["checkpoint"], profile=profile)
    assert final_a["collector"] == final_b["collector"]
    assert final_a["environment"] == final_b["environment"]
    policy = load_policy(whole["checkpoint"], profile=profile, config=factory().config)
    obs, _ = factory().reset(seed=123)
    assert policy.predict(obs) in (0, 1, 2)
    with pytest.raises(ValueError, match="profile"):
        load_policy(whole["checkpoint"], profile="paper-core-discrete-v1")
    with pytest.raises(ValueError, match="configuration"):
        load_policy(whole["checkpoint"], profile=profile, config=altered_config)
    print(f"custom PPO train/resume validation seconds: {time.perf_counter() - started:.3f}")
