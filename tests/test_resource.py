"""Hand-computable checks for resource timing, economics and causal inputs."""

from dataclasses import replace
import unittest

import numpy as np
from gymnasium import error as gym_error
from gymnasium.utils.env_checker import check_env

from maintenance_arena.resource import (
    COST_FIELDS,
    ResourceMaintenanceConfig,
    ResourceMaintenanceEnv,
)


def config(**overrides):
    value = dict(n_assets=1, initial_ages=(0,), contribution_per_asset_hour=(10,),
                 horizon=20, shift_hours=24, shift_period=24,
                 normal_lead_range=(2, 2), express_lead_range=(1, 1))
    value.update(overrides)
    return ResourceMaintenanceConfig.from_mapping(value)


class ResourceMaintenanceTests(unittest.TestCase):
    def test_cash_ledger_lead_delivery_and_joint_capacity(self):
        env = ResourceMaintenanceEnv(config(
            inventory_capacity=1, initial_stock=1, pm_work_hours=1,
            part_cost=100, normal_order_cost=5, holding_cost_per_part_hour=2,
            crew_wage_per_hour=3, pm_setup_cost=7, reward_scale=1,
        ))
        env.reset(seed=11)
        self.assertFalse(env.action_masks()[env.encode_action(1, 0)])
        joint = env.encode_action(1, 1)
        self.assertTrue(env.action_masks()[joint])
        _, reward, terminated, truncated, info = env.step(joint)
        self.assertEqual(reward, -215)  # 100 initial + 100 part + 5 order + 7 setup + 3 wages
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["metrics"]["parts_consumed"], 1)
        self.assertEqual(env.policy_state()["stock"], 0)
        self.assertEqual(env.policy_state()["pipeline"], [{"age": 1, "mode": 1}])
        _, reward, _, _, info = env.step(0)
        self.assertEqual(reward, 7)
        self.assertEqual(env.policy_state()["stock"], 1)  # delivered at decision time 2
        self.assertEqual(info["ledger"]["holding_cost"], 0)
        _, reward, _, _, info = env.step(0)
        self.assertEqual(reward, 5)
        self.assertEqual(info["cumulative_ledger"]["net_profit"], -203)
        self.assertEqual(info["cumulative_ledger"]["initial_stock_cost"], 100)
        self.assertEqual(info["metrics"]["inventory_conservation_residual"], 0)
        ledger = info["cumulative_ledger"]
        self.assertEqual(ledger["net_profit"], ledger["contribution"] - sum(ledger[key] for key in COST_FIELDS))

    def test_shift_pause_reserves_crew_and_renews_only_at_completion(self):
        env = ResourceMaintenanceEnv(config(
            n_assets=2, initial_ages=(0, 0), contribution_per_asset_hour=(10, 20),
            initial_stock=2, inventory_capacity=2, pm_work_hours=3,
            shift_hours=2, shift_period=4,
        ))
        env.reset(seed=2)
        original = env._assets[0].curve.copy()
        for decision, remaining in enumerate((2, 1, 1, 1, 0)):
            _, _, _, _, info = env.step(env.encode_action(0, 1) if decision == 0 else 0)
            self.assertEqual(env.policy_state()["assets"][0]["repair_work_remaining"], remaining)
            if remaining:
                self.assertEqual(env.free_crews, 0)
                self.assertFalse(env.action_masks()[env.encode_action(0, 2)])
                np.testing.assert_array_equal(env._assets[0].curve, original)
                self.assertEqual(env._assets[0].age, 0)
        self.assertEqual(info["metrics"]["crew_work_hours"], 3)
        self.assertEqual(info["metrics"]["paid_crew_hours"], 3)
        self.assertEqual(info["metrics"]["down_asset_hours"], 5)
        self.assertEqual(info["metrics"]["operating_asset_hours"], 5)
        self.assertEqual(info["metrics"]["completed_replacements"], 1)
        self.assertEqual(env.free_crews, 1)
        self.assertEqual(env._assets[0].renewal, 1)
        self.assertEqual(env.policy_state()["assets"][0]["health"], 1)
        # An earlier-index job completes at interval end; its crew was still
        # unavailable to a later-index failed machine throughout that hour.
        env = ResourceMaintenanceEnv(config(
            n_assets=2, initial_ages=(0, 0), contribution_per_asset_hour=(10, 20),
            initial_stock=2, inventory_capacity=2, pm_work_hours=1,
        ))
        env.reset(seed=12)
        env._assets[1].curve = np.array([1.0, 0.2])
        env._assets[1].age, env._assets[1].failed = 1, True
        _, _, _, _, info = env.step(env.encode_action(0, 1))
        self.assertEqual(env.free_crews, 1)
        self.assertEqual(info["metrics"]["no_crew_wait_hours"], 1)
        self.assertEqual(info["metrics"]["failed_waiting_asset_hours"], 1)

    def test_failure_is_end_of_hour_once_and_has_no_automatic_repair(self):
        env = ResourceMaintenanceEnv(config(cm_work_hours=2, failure_event_cost=1500))
        env.reset(seed=3)
        # Test fixture: a component crosses its failure threshold in one hour.
        env._assets[0].curve = np.array([1.0, 0.2])
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["ledger"]["contribution"], 10)
        self.assertEqual(info["ledger"]["failure_cost"], 1500)
        self.assertEqual(info["metrics"]["down_asset_hours"], 0)
        self.assertTrue(env.policy_state()["assets"][0]["failed"])
        self.assertEqual(env.policy_state()["assets"][0]["failure_wait_hours"], 0)
        for _ in range(2):
            _, _, _, _, info = env.step(0)
            self.assertEqual(info["ledger"]["contribution"], 0)
            self.assertEqual(info["ledger"]["failure_cost"], 0)
        self.assertEqual(info["metrics"]["failures"], 1)
        self.assertEqual(info["metrics"]["failed_waiting_asset_hours"], 2)
        env.step(env.encode_action(0, 1))
        self.assertTrue(env.policy_state()["assets"][0]["failed"])
        self.assertEqual(env.policy_state()["assets"][0]["repair_work_remaining"], 1)
        _, _, _, _, info = env.step(0)
        self.assertFalse(env.policy_state()["assets"][0]["failed"])
        self.assertEqual(info["metrics"]["failures"], 1)
        self.assertEqual(info["metrics"]["corrective_starts"], 1)
        self.assertEqual(info["metrics"]["completed_replacements"], 1)

    def test_invalid_joint_action_is_atomic_wait_and_stock_conserves(self):
        env = ResourceMaintenanceEnv(config(initial_stock=0))
        env.reset(seed=44)
        invalid_joint = env.encode_action(2, 1)
        self.assertFalse(env.action_masks()[invalid_joint])
        _, _, _, _, info = env.step(invalid_joint)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["metrics"]["parts_ordered"], 0)
        self.assertEqual(info["ledger"]["service_cost"], 0)
        rng = np.random.default_rng(100)
        while not env._done:
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
            _, _, _, _, info = env.step(action)
            self.assertFalse(info["invalid_action"])
            self.assertEqual(info["metrics"]["inventory_conservation_residual"], 0)
            state = env.policy_state()
            self.assertGreaterEqual(state["stock"], 0)
            self.assertLessEqual(state["inventory_position"], env.config.inventory_capacity)
            self.assertGreaterEqual(state["free_crews"], 0)
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=44)
        with self.assertRaises(ValueError):
            env.step(env.action_space.n)

    def test_keyed_randomness_repeats_and_is_independent_of_actions(self):
        a = ResourceMaintenanceEnv(config(inventory_capacity=3, initial_stock=1))
        b = ResourceMaintenanceEnv(config(inventory_capacity=3, initial_stock=1))
        obs_a, _ = a.reset(seed=71)
        obs_b, _ = b.reset(seed=71)
        np.testing.assert_array_equal(obs_a, obs_b)
        for renewal in range(3):
            np.testing.assert_array_equal(a._make_curve(0, renewal), b._make_curve(0, renewal))
        # Consuming distinct policy-side events must not shift exogenous draws.
        a.step(a.encode_action(2, 1))
        b.step(0)
        for hour in (1, 7, 12):
            for mode in (1, 2):
                self.assertEqual(a._delivery_lead(hour, mode), b._delivery_lead(hour, mode))
        for renewal in range(3):
            np.testing.assert_array_equal(a._make_curve(0, renewal), b._make_curve(0, renewal))
        new_obs, _ = a.reset(seed=71)
        np.testing.assert_array_equal(obs_a, new_obs)

    def test_observation_contract_and_gymnasium_cutoff(self):
        env = ResourceMaintenanceEnv(config(horizon=3))
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=8)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(len(obs), len(env.observation_names))
        forbidden = ("rul", "lifetime", "family", "horizon", "arrives_at", "operating_age")
        self.assertFalse(any(token in " ".join(env.observation_names) for token in forbidden))
        self.assertFalse(any(token in str(env.policy_state()) for token in forbidden))
        self.assertEqual(info["action_mask"].dtype, bool)
        # Changing a hidden order realization does not reveal it in observation.
        env.step(env.encode_action(1, 0))
        before = env._observation().copy()
        env._pipeline[0] = replace(env._pipeline[0], arrives_at=19)
        np.testing.assert_array_equal(before, env._observation())
        env.step(0)
        _, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)  # continuing task: value bootstrap remains appropriate
        self.assertEqual(info["metrics"]["pipeline_units"], 1)
        self.assertNotEqual(info["cumulative_ledger"]["parts_cost"], 0)
        self.assertEqual(env.policy_state()["phase"], 3)

    def test_configuration_rejects_invalid_resource_or_time_contract(self):
        for overrides in (
            {"not_a_resource_setting": 1},
            {"initial_stock": 4}, {"shift_hours": 25}, {"normal_lead_range": (0, 2)},
            {"reward_scale": 0}, {"pm_work_hours": 1.5}, {"crew_count": 0},
            {"initial_ages": (120,)}, {"contribution_per_asset_hour": (float("nan"),)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                config(**overrides)


from copy import deepcopy
import unittest

import numpy as np

from maintenance_arena.resource import ResourceMaintenanceEnv
from maintenance_arena.resource_lifetime import (
    LifetimeResourceEnv, LifetimePolicyConfig, choose_lifetime_action, run_episode,
)


class AgeOnlyProxy:
    """Deliberately provides no Health/rate or private simulator attributes."""
    __slots__ = ("state", "mask", "n_assets")

    def __init__(self, env):
        self.state, self.mask = env.policy_state(), env.action_masks()
        self.n_assets = len(self.state["assets"])
        for asset in self.state["assets"]:
            del asset["health"], asset["log_damage_rate"]

    def policy_state(self):
        return deepcopy(self.state)

    def action_masks(self):
        return self.mask.copy()

    def encode_action(self, mode, job):
        return mode * (self.n_assets + 1) + job


class LifetimeResourceTests(unittest.TestCase):
    def test_operating_age_pauses_and_resets_only_at_completed_exchange(self):
        env = LifetimeResourceEnv(dict(n_assets=1, initial_ages=(50,),
            contribution_per_asset_hour=(100,), lifetime_range=(80, 80)))
        env.reset(seed=981234)
        for _ in range(7):
            env.step(0)
        self.assertEqual(env.policy_state()["assets"][0]["operating_age"], 57)
        _, _, _, _, info = env.step(env.encode_action(0, 1))
        self.assertEqual(env.policy_state()["assets"][0]["operating_age"], 57)
        self.assertEqual(env.diagnostics()["analysis_pm_starts"][0]["unused_life_hours"], 23)
        self.assertEqual(env.policy_state()["assets"][0]["repair_work_remaining"], 3)
        completed = False
        for _ in range(20):
            _, _, _, _, info = env.step(0)
            completed = any(e["type"] == "service_complete" for e in info["events"])
            self.assertEqual(env.policy_state()["assets"][0]["operating_age"], 0 if completed else 57)
            if completed:
                break
        self.assertTrue(completed)
        env.step(0)
        self.assertEqual(env.policy_state()["assets"][0]["operating_age"], 1)
        self.assertEqual(env._observation()[-1], 1)
        env.close()

        failed = LifetimeResourceEnv(dict(n_assets=1, initial_ages=(79,),
            contribution_per_asset_hour=(100,), lifetime_range=(80, 80)))
        failed.reset(seed=981235)
        failed.step(0)
        for _ in range(3):
            failed.step(0)
        self.assertTrue(failed.policy_state()["assets"][0]["failed"])
        self.assertEqual(failed.policy_state()["assets"][0]["operating_age"], 80)
        failed.step(failed.encode_action(0, 1))
        self.assertEqual(failed.policy_state()["assets"][0]["operating_age"], 80)
        for _ in range(30):
            _, _, _, _, info = failed.step(0)
            if any(e["type"] == "service_complete" for e in info["events"]):
                break
            self.assertEqual(failed.policy_state()["assets"][0]["operating_age"], 80)
        self.assertEqual(failed.policy_state()["assets"][0]["operating_age"], 0)
        failed.close()

    def test_lifetime_quantiles_shapes_and_delivery_stream_are_paired(self):
        envs = [LifetimeResourceEnv({"lifetime_range": bounds}) for bounds in
                ((160, 160), (144, 176), (120, 200), (80, 240))]
        for env in envs:
            obs, _ = env.reset(seed=981236)
            self.assertEqual(obs.shape, (58,))
        for asset in range(3):
            for renewal in range(6):
                records = []
                for env in envs:
                    env._make_curve(asset, renewal)
                    rec = env.lifecycle_metadata(asset, renewal)
                    low, high = env.config.lifetime_range
                    self.assertEqual(rec["lifetime"], low + int(rec["lifetime_quantile"] * (high-low+1)))
                    records.append(rec)
                for rec in records[1:]:
                    self.assertEqual(rec["lifetime_quantile"], records[0]["lifetime_quantile"])
                    self.assertEqual(rec["family"], records[0]["family"])
                    self.assertEqual(rec["shape_rng_state_after_curve"], records[0]["shape_rng_state_after_curve"])
        legacy = ResourceMaintenanceEnv()
        obs, _ = legacy.reset(seed=981236)
        self.assertEqual(obs.shape, (55,))
        self.assertNotIn("operating_age", legacy.policy_state()["assets"][0])
        for at, mode in ((0, 1), (73, 2), (144, 1)):
            leads = [env._delivery_lead(at, mode) for env in envs]
            self.assertEqual(leads, [legacy._delivery_lead(at, mode)] * 4)
        legacy.close()
        for env in envs:
            env.close()

    def test_age_and_reactive_never_read_health_and_masks_keep_legacy_semantics(self):
        scheduled = LifetimeResourceEnv({"initial_ages": (136, 0, 0), "lifetime_range": (160, 160)})
        scheduled.reset(seed=981240)
        for _ in range(4):
            scheduled.step(0)
        scheduled_policy = LifetimePolicyConfig(method="age", age_interval=159, base_stock=1)
        self.assertEqual(scheduled.decode_action(choose_lifetime_action(AgeOnlyProxy(scheduled), scheduled_policy))[1], 1)
        scheduled.step(0)  # Missed ideal phase4: a late feasible start is retained.
        self.assertEqual(scheduled.decode_action(choose_lifetime_action(AgeOnlyProxy(scheduled), scheduled_policy))[1], 1)
        scheduled.close()
        for method in ("age", "reactive"):
            env = LifetimeResourceEnv({"lifetime_range": (80, 80), "horizon": 240})
            env.reset(seed=981237)
            policy = LifetimePolicyConfig(method=method, age_interval=65,
                                          procurement="deadline_express", base_stock=2)
            for _ in range(240):
                # Explicit call to the inherited implementation audits the
                # optimized subclass mask over real operational transitions.
                np.testing.assert_array_equal(env.action_masks(), ResourceMaintenanceEnv.action_masks(env))
                action = choose_lifetime_action(AgeOnlyProxy(env), policy)
                self.assertEqual(action, choose_lifetime_action(env, policy))
                _, _, _, _, info = env.step(action)
                self.assertFalse(info["invalid_action"])
            env.close()

    def test_window_accounting_and_analysis_only_pm_life(self):
        result = run_episode(dict(lifetime_range=(160, 160)),
            LifetimePolicyConfig(method="age", age_interval=159, base_stock=3),
            981238, [(0, 72), (72, 240), (0, 240)])
        a, b, whole = result["windows"]
        for key in whole["cash_ledger"]:
            self.assertAlmostEqual(a["cash_ledger"][key] + b["cash_ledger"][key], whole["cash_ledger"][key], places=8)
        self.assertEqual(a["analysis_pm"]["count"] + b["analysis_pm"]["count"], whole["analysis_pm"]["count"])
        self.assertEqual(whole["analysis_pm"]["unused_life_hours"] + whole["analysis_pm"]["operating_age_hours"], whole["analysis_pm"]["installed_life_hours"])
        self.assertEqual(whole["cumulative_metrics"]["failures"], 0)
        self.assertNotIn("true_lifetime", whole["closing_state"]["assets"][0])


class StandaloneResourceTests(unittest.TestCase):
    def test_rejected_actions_and_resets_preserve_live_state_and_age(self):
        for cls in (ResourceMaintenanceEnv, LifetimeResourceEnv):
            with self.subTest(environment=cls.__name__):
                env = cls(render_mode="ansi")
                self.assertIn("reset required", env.render())
                env.reset(seed=410)
                env.step(0)
                env.step(env.encode_action(1, 1))
                saved = env.snapshot()
                for invalid in (True, False, np.bool_(True)):
                    with self.assertRaises(ValueError):
                        env.step(invalid)
                    with self.assertRaises(ValueError):
                        env.encode_action(invalid, 0)
                    with self.assertRaises(ValueError):
                        env.encode_action(0, invalid)
                    self.assertEqual(saved, env.snapshot())
                for seed, options in ((92, {"stock": 3}), (-1, None)):
                    with self.assertRaises((ValueError, gym_error.Error)):
                        env.reset(seed=seed, options=options)
                    self.assertEqual(saved, env.snapshot())
                replay = cls()
                replay.restore(saved)
                for action in (0, 0, 0):
                    actual, expected = env.step(action), replay.step(action)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    self.assertEqual(actual[1:4], expected[1:4])
                    self.assertEqual(env.snapshot(), replay.snapshot())

    def test_snapshot_restores_busy_jobs_deliveries_rng_and_lifetime_records(self):
        for cls in (ResourceMaintenanceEnv, LifetimeResourceEnv):
            with self.subTest(environment=cls.__name__):
                original = cls({"horizon": 240})
                original.reset(seed=90125)
                original.step(original.encode_action(1, 1))
                saved = original.snapshot()
                replay = cls({"horizon": 240})
                replay.restore(saved)
                self.assertEqual(replay.policy_state(), original.policy_state())
                for t in range(200):
                    feasible = np.flatnonzero(original.action_masks())
                    action = int(feasible[t % len(feasible)])
                    a, b = original.step(action), replay.step(action)
                    np.testing.assert_array_equal(a[0], b[0])
                    self.assertEqual(a[1:4], b[1:4])
                    self.assertEqual(a[4]["ledger"], b[4]["ledger"])
                    self.assertEqual(a[4]["events"], b[4]["events"])
                    self.assertEqual(original.diagnostics(), replay.diagnostics())
                np.testing.assert_array_equal(original.reset()[0], replay.reset()[0])
                with self.assertRaises(ValueError):
                    cls({"horizon": 241}).restore(saved)

    def test_privileged_lifetime_data_is_only_in_diagnostics(self):
        env = LifetimeResourceEnv()
        env.reset(seed=34)
        obs, reward, terminated, truncated, info = env.step(env.encode_action(0, 1))
        for forbidden in ("true_lifetime", "unused_life_hours", "analysis_pm_starts", "arrives_at"):
            self.assertNotIn(forbidden, str(info))
            self.assertNotIn(forbidden, str(env.policy_state()))
        self.assertTrue(env.diagnostics()["analysis_pm_starts"])
        self.assertEqual(info["duration_calendar_steps"], 1)
        self.assertEqual(info["gamma_pow_duration"], 0.995)
        self.assertEqual(info["raw_profit_sum"], reward * env.config.reward_scale)


if __name__ == "__main__":
    unittest.main()
