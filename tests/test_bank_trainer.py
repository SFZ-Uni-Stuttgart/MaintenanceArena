"""PPO training on scenario banks: role seeds, milestones, initialization and exact resumption."""
import pytest

torch = pytest.importorskip("torch")
import numpy as np

from maintenance_arena.banks import build_bank
from maintenance_arena.learning.collector import BankTrainer, load_checkpoint, policy_agent, role_seeds
from maintenance_arena.learning.ppo import parameter_hash
from maintenance_arena.replay import decision_series, policy_probabilities

PRESERVE = {"recovery_contract": "preserve_preparation_v1"}


@pytest.fixture(scope="module")
def bank():
    return build_bank("source", split="source_train", seed=33, pair_count=192)


def test_role_seeds_of_the_article_runs():
    assert role_seeds(26101601) == {"network": 3382318844, "environment": 3733068830, "action": 2932503442,
                                    "shuffle": 566942177}
    with pytest.raises(ValueError):
        role_seeds(-1)


def test_rollouts_stop_at_milestones_and_resume_exactly(bank, tmp_path):
    straight = BankTrainer(bank, 7, env_config=PRESERVE)
    first = straight.train_until(700)
    assert first["actual_calendar_steps"] - 700 <= 18 and first["rollout_updates"] == 1
    straight.save(tmp_path / "checkpoint_700.pt")
    with pytest.raises(FileExistsError):
        straight.save(tmp_path / "checkpoint_700.pt")
    final = straight.train_until(2048)
    assert final["rollout_updates"] == 3
    resumed = BankTrainer(bank, 7, env_config=PRESERVE, resume=tmp_path / "checkpoint_700.pt")
    assert resumed.calendar == first["actual_calendar_steps"]
    assert resumed.train_until(2048) == final
    assert parameter_hash(resumed.agent.policy_state()) == parameter_hash(straight.agent.policy_state())
    assert resumed.logs == straight.logs
    assert resumed.env.snapshot() == straight.env.snapshot()
    with pytest.raises(ValueError, match="seed"):
        BankTrainer(bank, 8, env_config=PRESERVE, resume=tmp_path / "checkpoint_700.pt")
    with pytest.raises(ValueError):
        BankTrainer(bank, 7, env_config={**PRESERVE, "mobilization_lead": 6}, resume=tmp_path / "checkpoint_700.pt")


def test_initialization_loads_weights_only(bank, tmp_path):
    source = BankTrainer(bank, 7, env_config=PRESERVE)
    source.train_until(1024)
    source.save(tmp_path / "source.pt")
    target = build_bank("gompertz", split="adaptation", seed=4, pair_count=32)
    tuned = BankTrainer(target, 11, env_config=PRESERVE, initialize=tmp_path / "source.pt")
    assert parameter_hash(tuned.agent.policy_state()) == load_checkpoint(tmp_path / "source.pt")["policy_parameter_sha256"]
    assert tuned.agent.optimizer.state_dict()["state"] == {} and tuned.agent.rollout_updates == 0
    assert tuned.initialized_from["mode"].startswith("weights only")
    fresh = BankTrainer(target, 11, env_config=PRESERVE)
    assert parameter_hash(fresh.agent.policy_state()) != parameter_hash(tuned.agent.policy_state())
    with pytest.raises(ValueError, match="mutually exclusive"):
        BankTrainer(target, 11, initialize=tmp_path / "source.pt", resume=tmp_path / "source.pt")


def test_input_sets_and_preparation_times_train(bank):
    masked = BankTrainer(bank, 7, env_config=PRESERVE, input_set="deadline_only")
    np.testing.assert_array_equal(masked.observation[:3], [0., 0., 0.])
    masked.train_until(512)
    assert not np.any(masked.observation[:3])
    long = BankTrainer(bank, 7, env_config={**PRESERVE, "mobilization_lead": 24})
    result = long.train_until(1024)
    assert result["actual_calendar_steps"] - 1024 <= 30


def test_checkpoints_load_into_evaluation_agents(bank, tmp_path):
    trainer = BankTrainer(bank, 9, env_config=PRESERVE)
    trainer.train_until(512)
    saved = trainer.save(tmp_path / "checkpoint.pt")
    agent = policy_agent(tmp_path / "checkpoint.pt")
    states = decision_series(build_bank("source", split="development", seed=2, pair_count=4)).states
    np.testing.assert_array_equal(policy_probabilities(agent, states), trainer.agent.batch_output(states)[1])
    assert saved["policy_parameter_sha256"] == parameter_hash(agent.policy_state())
