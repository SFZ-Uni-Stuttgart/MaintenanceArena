"""Noise design: both state estimates, the Rule on two estimates and the selection rules."""
import numpy as np
import pytest

from maintenance_arena import replay as rp
from maintenance_arena import selection as sel
from maintenance_arena.banks import build_bank

LAWS = ("bearing_late_knee", "concave_running_in", "convex_accelerating_fatigue", "linear_reference")
GRID = {"health": [0.25, 0.3, 0.36, 0.45], "projection": [8, 12, 15, 18, 24]}


@pytest.fixture(scope="module")
def noisy():
    bank = build_bank("source", split="development", seed=91, pair_count=24, sensor_level="gaussian_005")
    return (rp.decision_series(bank, observation_route="RAW_CAUSAL"),
            rp.decision_series(bank, observation_route="FILTERED_CAUSAL_EWLS8"))


def test_rule_on_two_estimates_combines_both_indicators(noisy):
    raw, line = noisy
    candidates = rp.threshold_candidates(GRID)
    single_raw, single_line = rp.threshold_orders(raw, candidates), rp.threshold_orders(line, candidates)
    both = rp.dual_threshold_orders(raw, line, candidates, "both")
    either = rp.dual_threshold_orders(raw, line, candidates, "either")
    # "either" orders at the first age at which one estimate meets the threshold.
    missing = (single_raw < 0) & (single_line < 0)
    first = np.where(single_raw < 0, single_line, np.where(single_line < 0, single_raw, np.minimum(single_raw, single_line)))
    assert np.array_equal(either, np.where(missing, -1, first))
    # "both" never orders before either estimate alone.
    ordered = both >= 0
    assert np.all(both[ordered] >= single_raw[ordered]) and np.all(both[ordered] >= single_line[ordered])
    result = rp.calibrate_dual_rule(raw, line, GRID, "both", economics=rp.Economics(recovery_contract="preserve_preparation_v1"))
    assert result["selected"]["combine"] == "both" and len(result["candidates"]) == len(candidates)
    with pytest.raises(ValueError):
        rp.dual_threshold_orders(raw, line, candidates, "neither")


def test_dual_states_order_both_estimates(noisy):
    pytest.importorskip("torch")
    from maintenance_arena.learning.dual import DUAL_TO_LINE_FIT, dual_states, split_dual
    raw, line = noisy
    states = dual_states(raw.states, line.states)
    assert states.shape == (len(raw.states), 8) and states.dtype == np.float32
    assert np.array_equal(states[:, 4:], line.states[:, [1, 3, 0, 2]])
    back_raw, back_line = split_dual(states)
    assert np.array_equal(back_raw, raw.states) and np.array_equal(back_line, line.states)
    assert DUAL_TO_LINE_FIT == (6, 4, 7, 5)


def test_eight_input_policy_starts_as_the_four_input_policy(noisy):
    pytest.importorskip("torch")
    from maintenance_arena.learning.collector import PROFILE, role_seeds
    from maintenance_arena.learning.dual import DualPPOAgent, dual_states
    from maintenance_arena.learning.ppo import PPOAgent
    raw, line = noisy
    four = PPOAgent(PROFILE, 5, role_seeds=role_seeds(5))
    eight = DualPPOAgent(6, role_seeds=role_seeds(6))
    eight.initialize_four(four.policy_state())
    first = eight.network.policy_body[0].weight.detach()
    assert np.all(first[:, 4:].numpy() == 0.)
    np.testing.assert_allclose(eight.batch_output(dual_states(raw.states, line.states))[1],
                               four.batch_output(raw.states)[1], rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        eight.batch_output(raw.states)


def test_two_estimate_environment_matches_the_batch_states(tmp_path):
    pytest.importorskip("torch")
    from maintenance_arena.learning.collector import load_checkpoint, policy_agent
    from maintenance_arena.learning.dual import DualBankTrainer, DualPPOAgent, DualRouteBankEnv, dual_states
    bank = build_bank("source", split="source_train", seed=12, pair_count=96, sensor_level="gaussian_005")
    env = DualRouteBankEnv(bank, {"recovery_contract": "preserve_preparation_v1"})
    observation, _ = env.reset()
    raw = rp.decision_series(bank, observation_route="RAW_CAUSAL")
    line = rp.decision_series(bank, observation_route="FILTERED_CAUSAL_EWLS8")
    expected = dual_states(raw.life_states(0), line.life_states(0))
    seen = [observation]
    for _ in range(len(expected) - 1):
        observation, _, terminal, _, _ = env.step(0)
        assert not terminal
        seen.append(observation)
    assert np.array_equal(np.asarray(seen), expected)
    with pytest.raises(RuntimeError):
        env.restore(env.snapshot())
    with pytest.raises(ValueError):
        DualRouteBankEnv(bank, {"observation_route": "FILTERED_CAUSAL_EWLS8"})
    trainer = DualBankTrainer(bank, 3, env_config={"recovery_contract": "preserve_preparation_v1"})
    trainer.train_until(600)
    trainer.save(tmp_path / "checkpoint.pt")
    assert tuple(load_checkpoint(tmp_path / "checkpoint.pt")["agent"]["policy_state"]["policy_body.0.weight"].shape) == (64, 8)
    assert isinstance(policy_agent(tmp_path / "checkpoint.pt"), DualPPOAgent)


def outcomes(cash, calendar, failed):
    return {"cash": np.asarray(cash, float), "calendar": np.asarray(calendar, float), "failed": np.asarray(failed, float)}


def test_screen_values_are_equal_law_rates_and_failure_shares():
    laws = list(LAWS) * 2
    reference = outcomes([[100.] * 8], [[10.] * 8], [[0.] * 8])
    arm = outcomes([[90.] * 8, [80.] * 8], [[10.] * 8, [10.] * 8], [[1, 0, 0, 0, 0, 0, 0, 0], [0.] * 8])
    values = sel.screen_values({"reference": reference, "arm": arm}, laws, reference="reference")
    assert values["reference"]["R"] == pytest.approx(100.)
    assert values["arm"]["R"] == pytest.approx(85.)
    assert values["arm"]["per_run_F"] == pytest.approx([12.5, 0.])
    assert values["arm"]["family_F"]["bearing_late_knee"] == pytest.approx(25.)
    assert values["arm"]["F"] == pytest.approx(6.25)


def value(R, F, worst=None):
    return {"R": R, "F": F, "family_F": {law: F if worst is None else worst for law in LAWS}}


def test_rule_estimate_and_design_selection():
    values = {"raw": value(97., .6), "line": value(99.3, .25), "both": value(99.35, .5), "either": value(97., .6),
              "a": value(99.4, .2), "b": value(99.42, .19), "c": value(99.6, .9), "d": value(99.9, .1, worst=2.5)}
    rule = sel.select_rule_estimator(values, ["raw", "line", "both", "either"])
    # Within 0.10 points of the best passing retention, the lower failure share wins.
    assert rule["rule"] == "line" and rule["passing"] == ["raw", "line", "both", "either"]
    design = sel.select_design(values, ["a", "b", "c", "d"], {"a": 1, "b": 2, "c": 1, "d": 1}, rule["F_rule"])
    assert design["bound_pct"] == pytest.approx(.5)
    assert design["passing"] == ["a", "b"]          # c above the bound, d above the 2% law cap
    assert design["design"] == "b" and design["ties"]["retention_zone"] == ["a", "b"]
    tied = sel.select_design({"a": value(99., .2), "b": value(99., .2)}, ["b", "a"], {"a": 1, "b": 2}, .25)
    assert tied["design"] == "a"                    # equal failures: the smaller budget
    none = sel.select_design({"a": value(99., 1.5)}, ["a"], {"a": 1}, .25)
    assert none["design"] is None and none["passing"] == []
    fallback = sel.select_rule_estimator({"x": value(99., 3.), "y": value(98., 2.)}, ["x", "y"])
    assert fallback["rule"] == "y" and fallback["passing"] == []
    assert sel.matching_rule(values, "both", {"both": ["both", "either"]}) == "both"


def test_outcome_arrays_read_outcome_rows():
    rows = [{"method": m, "group": f"noisy/{law}", "run": run, "unit": unit, "cash": 10. * unit, "calendar_steps": 5,
             "failed": str(unit == 1)}
            for m, runs in (("arm", ("7", "8")), ("Last-safe reference", ("",))) for run in runs
            for unit, law in ((1, "linear_reference"), (0, "bearing_late_knee"))]
    arrays, laws = sel.outcome_arrays(rows, law=lambda row: row["group"].split("/", 1)[1])
    assert laws == ["bearing_late_knee", "linear_reference"]
    assert arrays["arm"]["cash"].shape == (2, 2) and arrays["arm"]["failed"][0].tolist() == [0., 1.]
    with pytest.raises(ValueError):
        sel.outcome_arrays(rows + rows[:1], law=lambda row: row["group"].split("/", 1)[1])
