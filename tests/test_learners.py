"""The comparison learners of the article: recipes, exact resumption and greedy replay."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from maintenance_arena.banks import build_bank
from maintenance_arena.learning.collector import BankTrainer, single_thread
from maintenance_arena.learning.learners import (ALGORITHMS, LearnerTrainer, greedy_first_orders,
                                                 load_learner_checkpoint, load_policy, make_trainer, parameter_count,
                                                 recipe)
from maintenance_arena.learning.ppo import parameter_hash, recipe_config
from maintenance_arena.replay import decision_series

PRESERVE = {"recovery_contract": "preserve_preparation_v1"}
TABLES = Path(__file__).resolve().parents[1] / "paper" / "rebuild_tables.py"


@pytest.fixture(scope="module")
def bank():
    return build_bank("source", split="source_train", seed=33, pair_count=192)


def test_recipes_and_parameter_counts():
    assert recipe("PPO")["settings"] == recipe_config("paper-core-continuous-v1")
    dqn, ddqn = recipe("DQN"), recipe("DDQN")
    assert {**dqn, "algorithm": "DDQN"} == ddqn
    assert dqn["architecture"] == {"width": 64, "depth": 2} and dqn["settings"]["replay_capacity"] == 100000
    counts = {algorithm: parameter_count(algorithm) for algorithm in ALGORITHMS}
    assert counts == {"PPO": 9155, "DQN": 4610, "DDQN": 4610}
    if TABLES.is_file():
        spec = importlib.util.spec_from_file_location("rebuild_tables_for_learners", TABLES)
        tables = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tables)
        assert tables.PARAMETERS == {"PPO": 9155, "DQN": 4610, "Double DQN": 4610}
    with pytest.raises(ValueError):
        recipe("RPPO")
    assert isinstance(make_trainer("PPO", build_bank("source", split="source_train", seed=1, pair_count=4), 3,
                                   total_calendar_steps=1), BankTrainer)


@pytest.mark.parametrize("algorithm,total,cut", [("DQN", 3072, 1024), ("DDQN", 3072, 2500)])
def test_training_stops_at_milestones_and_resumes_exactly(bank, tmp_path, algorithm, total, cut):
    straight = LearnerTrainer(bank, 7, algorithm=algorithm, total_calendar_steps=total, env_config=PRESERVE)
    first = straight.train_until(cut)
    assert 0 <= first["actual_calendar_steps"] - cut <= 18
    path = tmp_path / "checkpoint.pt"
    saved = straight.save(path)
    with pytest.raises(FileExistsError):
        straight.save(path)
    assert load_learner_checkpoint(path)["policy_parameter_sha256"] == saved["policy_parameter_sha256"]
    final = straight.train_until(total)
    assert final["optimizer_minibatches"] > 0
    resumed = LearnerTrainer(bank, 7, algorithm=algorithm, total_calendar_steps=total, env_config=PRESERVE,
                             resume=path)
    assert resumed.calendar == first["actual_calendar_steps"]
    assert resumed.train_until(total) == final
    assert parameter_hash(resumed.agent.policy_state()) == parameter_hash(straight.agent.policy_state())
    assert resumed.logs == straight.logs
    assert resumed.env.snapshot() == straight.env.snapshot()
    with pytest.raises(ValueError, match="seed"):
        LearnerTrainer(bank, 8, algorithm=algorithm, total_calendar_steps=total, env_config=PRESERVE, resume=path)
    with pytest.raises(ValueError, match="total_calendar_steps"):
        LearnerTrainer(bank, 7, algorithm=algorithm, total_calendar_steps=total + 1, env_config=PRESERVE, resume=path)


def test_runs_with_the_same_seed_start_on_the_same_components(bank):
    trainers = [make_trainer(algorithm, bank, 7, total_calendar_steps=4096, env_config=PRESERVE)
                for algorithm in ALGORITHMS]
    assert len({trainer.observation.tobytes() for trainer in trainers}) == 1
    assert len({trainer.env.snapshot()["core"]["state"]["_bank_cursor"] for trainer in trainers}) == 1


def test_greedy_replay_of_all_lives_equals_one_life_at_a_time(bank, tmp_path):
    series = decision_series(build_bank("source", split="development", seed=2, pair_count=8))
    paths = {}
    for algorithm in ALGORITHMS:
        trainer = make_trainer(algorithm, bank, 5, total_calendar_steps=3072, env_config=PRESERVE)
        trainer.train_until(3072 if algorithm in ("DQN", "DDQN") else 1024)
        paths[algorithm] = tmp_path / f"{algorithm}.pt"
        trainer.save(paths[algorithm])
    for algorithm, path in paths.items():
        policy = load_policy(path)
        assert policy.algorithm == algorithm
        orders = greedy_first_orders(policy, series)
        single = []
        with single_thread():
            for life in range(series.lives):
                actions = [int(policy.actions(state[None])[0]) for state in series.life_states(life)]
                single.append(actions.index(1) if 1 in actions else -1)
        np.testing.assert_array_equal(orders, single)
