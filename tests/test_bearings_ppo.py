"""Bearing PPO: initialization, trainer equality with a direct economic replay, evaluation and checkpoints."""
from dataclasses import replace
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from maintenance_arena.bearings import (BearingHistory, ReplayEconomics, decision_series, design_observations,
                                        pooled_rate)
from maintenance_arena.bearings.ppo import BearingPPOAgent, TrainingSettings, load_policy, save_policy, train_policy
from maintenance_arena.learning.ppo import PPOAgent, RolloutEntry


def _observations(design="full", lives=(30, 45, 60)):
    result = []
    for seed, life in enumerate(lives):
        rng = np.random.default_rng(seed)
        minutes = np.abs(rng.normal(1.0, 0.1, (life, 6))) + 0.5
        minutes[life // 2:, 0:4] *= np.linspace(1, 5, life - life // 2)[:, None]
        history = BearingHistory("femto", f"B{seed}", minutes)
        forecast = None if design == "condition_only" else np.maximum(np.linspace(life + 5, -5, life), 0)
        result.append(design_observations(decision_series(history, forecast), design))
    return result


def _reference_training(observations, seed, calendar_steps, economics):
    """Rollouts written directly with ReplayEconomics.macro, as described in the article."""
    agent = BearingPPOAgent(seed, observations[0].shape[1])
    gamma, quota = agent.config["gamma"], agent.config["rollout_calendar_steps"]
    lives = [len(x) for x in observations]
    rng = np.random.Generator(np.random.PCG64(agent.roles["environment"]))
    weight = 100.0 * float(np.mean(lives))
    cash = pooled_rate(economics.outcome(life, None) for life in lives) * weight
    rho = cash / weight
    bearing, minute, calendar = int(rng.integers(len(lives))), 1, 0
    while calendar < calendar_steps:
        limit = min(calendar_steps, (calendar // quota + 1) * quota)
        entries = []
        while calendar < limit:
            observation = observations[bearing][minute - 1]
            action, log_probability, value = agent.sample_action(observation)
            rewards, terminal = economics.macro(lives[bearing], minute, action)
            discounted = math.fsum(gamma ** j * (r - rho) / 1000.0 for j, r in enumerate(rewards))
            successor = np.zeros_like(observation) if terminal else observations[bearing][minute]
            successor_value = 0.0 if terminal else agent.value(successor)
            entries.append(RolloutEntry(observation.copy(), action, log_probability, value, discounted, successor.copy(),
                                        successor_value, len(rewards), float(not terminal), float(not terminal),
                                        float(math.fsum(rewards)), calendar, calendar + len(rewards)))
            calendar += len(rewards)
            if terminal:
                outcome = economics.outcome(lives[bearing], minute if action == 1 else None)
                cash, weight = 0.99 * cash + outcome.cash, 0.99 * weight + outcome.calendar
                rho = cash / weight
                bearing, minute = int(rng.integers(len(lives))), 1
            else:
                minute += 1
        entries[-1] = replace(entries[-1], trace_mask=0.0)
        agent.update(entries)
    return agent


def test_agent_keeps_core_roles_and_replaces_only_the_first_layers():
    agent, core = BearingPPOAgent(2026101600, 7), PPOAgent("paper-core-continuous-v1", 2026101600)
    assert agent.roles == core.roles and agent.config == core.config
    state, core_state = agent.policy_state(), core.policy_state()
    assert state["policy_body.0.weight"].shape == state["value_body.0.weight"].shape == (64, 7)
    for name in ("policy_body.2.weight", "value_body.2.weight", "action_head.weight", "value_head.weight"):
        assert torch.equal(state[name], core_state[name])
    again = BearingPPOAgent(2026101600, 7).policy_state()
    assert all(torch.equal(state[name], again[name]) for name in state)
    assert len(agent.optimizer.param_groups[0]["params"]) == len(list(agent.network.parameters()))
    with pytest.raises(ValueError, match="policy inputs"):
        agent.batch_output(np.zeros((2, 6), dtype=np.float32))


@pytest.mark.training
@pytest.mark.parametrize("design,contract", [("full", "preserve_preparation_v1"),
                                             ("condition_only", "immediate_exchange_v1")])
def test_trainer_equals_a_direct_economic_replay(design, contract):
    observations = _observations(design)
    economics = ReplayEconomics(recovery_contract=contract)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference = _reference_training(observations, 11, 2048, economics)
    finally:
        torch.set_num_threads(previous)
    result = train_policy(observations, 11, 2048, economics=economics)
    assert torch.get_num_threads() == previous
    trained, expected = result.agent.policy_state(), reference.policy_state()
    assert all(torch.equal(trained[name], expected[name]) for name in expected)
    assert result.agent.optimizer_updates == reference.optimizer_updates and result.calendar_steps >= 2048
    assert result.episodes > 0 and 0 <= result.failures <= result.episodes


@pytest.mark.training
def test_greedy_orders_checkpoints_and_input_validation(tmp_path):
    observations = _observations()
    result = train_policy(observations, 5, 1024, settings=TrainingSettings(log_every_rollouts=1))
    agent = result.agent
    assert len(result.logs) == 1 and set(result.logs[0]) >= {"rollout", "train_rate_recent", "rho"}
    for matrix in observations:
        probabilities = agent.order_probabilities(matrix)
        order = agent.greedy_order_minute(matrix)
        hits = np.flatnonzero(probabilities > 1 - probabilities)
        assert order == (None if hits.size == 0 else int(hits[0] + 1))
    path = tmp_path / "policy.pt"
    save_policy(path, agent, design="Full", training_bearings=["B0", "B1", "B2"])
    loaded, metadata = load_policy(path)
    assert metadata == {"design": "Full", "training_bearings": ["B0", "B1", "B2"]}
    assert all(torch.equal(value, loaded.policy_state()[name]) for name, value in agent.policy_state().items())
    assert [loaded.greedy_order_minute(m) for m in observations] == [agent.greedy_order_minute(m) for m in observations]
    with pytest.raises(FileExistsError):
        save_policy(path, agent)
    with pytest.raises(ValueError, match="float32"):
        train_policy([m.astype(np.float64) for m in observations], 5, 1024)
    with pytest.raises(ValueError, match="calendar_steps"):
        train_policy(observations, 5, 0)
