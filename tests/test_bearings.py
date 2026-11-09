"""Bearing replay without recorded data: parsing, statistics, features, economics, environment, rules, metrics."""
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from maintenance_arena.bearings import (
    age_clock, median_order, policy_map, transfer_timing,
    BearingHistory, BearingReplayEnv, INPUT_DESIGNS, OBSERVATION_NAMES, RULE_FAMILIES, ReplayEconomics,
    acquisition_files, acquisition_statistics, apply_rule, bearing_metrics, calibrate_forecast_rule,
    calibrate_indicator_rule, check_layout, dataset_directory, decision_series, design_observations, design_summary,
    family_orders, family_parameters, forecast_rule_order, forecaster_features, hashed_folds, histories,
    indicator_candidates, load_statistics, outcome_row, paired_bootstrap, policy_observations, pooled_bootstrap,
    pooled_summary, prepare, rank_candidates, read_acquisition, read_outcomes, search_rule_families, to_minutes,
    write_outcomes)
from maintenance_arena.bearings.features import ewm, running_in_reference, trailing_slope
from maintenance_arena.bearings.rules import RuleCandidate, family_size, first_trigger, grid_values, rule_score

FEMTO_NAMES = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2", "Bearing1_3",
               "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7", "Bearing2_3", "Bearing2_4", "Bearing2_5",
               "Bearing2_6", "Bearing2_7", "Bearing3_3"]
FEMTO_OUTER_FOLDS = [["Bearing2_1", "Bearing3_1", "Bearing1_7", "Bearing2_5"],
                     ["Bearing3_2", "Bearing1_5", "Bearing1_6", "Bearing2_3"],
                     ["Bearing2_2", "Bearing2_4", "Bearing3_3"], ["Bearing1_3", "Bearing1_4", "Bearing2_7"],
                     ["Bearing1_1", "Bearing1_2", "Bearing2_6"]]


def _write_femto(path, signal, separator):
    lines = [separator.join(["9", "39", str(i % 60), str(1000 * i), repr(float(h)), repr(float(v))])
             for i, (h, v) in enumerate(signal)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_xjtu(path, signal):
    rows = ["Horizontal_vibration_signals,Vertical_vibration_signals"] + [f"{float(h)!r},{float(v)!r}" for h, v in signal]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _signals(seed, count, samples=24):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.3 + 0.05 * i, (samples, 2)) for i in range(count)]


def _expected_statistics(x):
    centered = x - x.mean(axis=0)
    variance = np.mean(centered ** 2, axis=0)
    kurtosis = np.mean(centered ** 4, axis=0) / np.maximum(variance, 1e-12) ** 2
    return np.concatenate([np.sqrt(np.mean(x ** 2, axis=0)), np.max(np.abs(x), axis=0), kurtosis])


@pytest.fixture
def femto_root(tmp_path):
    raw = tmp_path / "FEMTO"
    layout = {("Learning_set", "Bearing1_2"): (",", 7), ("Learning_set", "Bearing1_1"): (";", 13),
              ("Full_Test_Set", "Bearing1_3"): (",", 6)}
    signals = {}
    for (folder, bearing), (separator, count) in layout.items():
        directory = raw / folder / bearing
        directory.mkdir(parents=True)
        signals[bearing] = _signals(len(signals), count)
        for index, signal in enumerate(signals[bearing], start=1):
            _write_femto(directory / f"acc_{index:05d}.csv", signal, separator)
        (directory / "temp_00001.csv").write_text("9;39;39;0;30.1\n", encoding="utf-8")
    return tmp_path, signals


@pytest.fixture
def xjtu_root(tmp_path):
    raw = tmp_path / "XJTU-SY"
    signals = {}
    for folder, bearing, count in (("35Hz12kN", "Bearing1_1", 3), ("37.5Hz11kN", "Bearing2_1", 2),
                                   ("40Hz10kN", "Bearing3_1", 4)):
        directory = raw / folder / bearing
        directory.mkdir(parents=True)
        signals[bearing] = _signals(10 + len(signals), count, samples=32)
        for index, signal in enumerate(signals[bearing], start=1):
            _write_xjtu(directory / f"{index}.csv", signal)
    return tmp_path, signals


def test_femto_layout_parsing_and_statistics(femto_root):
    pytest.importorskip("pandas")
    root, signals = femto_root
    files = acquisition_files(root, "femto")
    assert list(files) == ["Bearing1_1", "Bearing1_2", "Bearing1_3"]
    assert [p.name for p in files["Bearing1_1"]][:2] == ["acc_00001.csv", "acc_00002.csv"]
    assert all(len(files[name]) == len(signals[name]) for name in files)
    for name, paths in files.items():
        for path, signal in zip(paths, signals[name]):
            values = read_acquisition(path, "femto")
            # pandas' default float parser is not always correctly rounded; it is kept for parity.
            np.testing.assert_allclose(values, signal, rtol=1e-12, atol=0)
            np.testing.assert_allclose(acquisition_statistics(values), _expected_statistics(values), rtol=1e-12)


def test_xjtu_header_columns_and_one_acquisition_per_minute(xjtu_root):
    pytest.importorskip("pandas")
    root, signals = xjtu_root
    files = acquisition_files(root, "xjtu_sy")
    assert list(files) == ["Bearing1_1", "Bearing2_1", "Bearing3_1"]
    np.testing.assert_allclose(read_acquisition(files["Bearing2_1"][1], "xjtu_sy"), signals["Bearing2_1"][1],
                               rtol=1e-12, atol=0)
    statistics = prepare(root, "xjtu_sy", root / "cache" / "xjtu_sy.npz")
    assert [h.life for h in histories(statistics, "xjtu_sy")] == [3, 2, 4]
    files["Bearing1_1"][0].write_text("1.0,2.0\n3.0,4.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_acquisition(files["Bearing1_1"][0], "xjtu_sy")


def test_reader_rejects_wrong_columns_and_numbering_gaps(femto_root):
    pytest.importorskip("pandas")
    root, _ = femto_root
    path = root / "FEMTO/Learning_set/Bearing1_2/acc_00002.csv"
    path.write_text("1,2,3,4,5\n1,2,3,4,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="six FEMTO columns"):
        read_acquisition(path, "femto")
    path.unlink()
    with pytest.raises(ValueError, match="1..N"):
        acquisition_files(root, "femto")
    with pytest.raises(ValueError, match="unknown bearing dataset"):
        acquisition_files(root, "ims")


def test_layout_check_names_what_is_missing_and_accepts_other_layouts(femto_root):
    root, signals = femto_root
    report = check_layout(root, "femto")
    assert not report["ok"] and report["folders"]["Learning_set"] == {"Bearing1_1": 13, "Bearing1_2": 7}
    assert f"bearing {root / 'FEMTO' / 'Learning_set' / 'Bearing2_1'}" in report["missing"]
    assert any("Bearing1_3: 6 acquisitions, the article used 2375" in line for line in report["problems"])
    (root / "FEMTO/Learning_set/Bearing1_1/acc_00005.csv").unlink()
    (root / "FEMTO/Learning_set/Extra_1").mkdir()
    report = check_layout(root, "femto")
    assert any(line.endswith("Bearing1_1: acquisitions 5 of 1..13") for line in report["missing"])
    assert any("no files matching" in line for line in report["missing"]) and report["notes"]
    other = root / "elsewhere"
    (root / "FEMTO").rename(other)
    assert check_layout(root, "femto")["missing"] == [f"directory {root / 'FEMTO'}"]
    assert check_layout(None, "femto", directory=other)["folders"]["Full_Test_Set"] == {"Bearing1_3": 6}
    with pytest.raises(ValueError, match="data root"):
        dataset_directory(None, "xjtu_sy")
    xjtu = check_layout(root, "xjtu_sy")
    assert not xjtu["ok"] and xjtu["missing"] == [f"directory {root / 'XJTU-SY'}"]


def test_layout_check_flags_incomplete_xjtu_sy_downloads(xjtu_root):
    root, _ = xjtu_root
    report = check_layout(root, "xjtu_sy")
    assert report["folders"]["35Hz12kN"] == {"Bearing1_1": 3} and not report["ok"]
    assert any(line.endswith("Bearing1_1: 3 acquisitions, the article used 123") for line in report["problems"])
    assert f"bearing {root / 'XJTU-SY' / '40Hz10kN' / 'Bearing3_4'}" in report["missing"]


def test_explicit_dataset_directories_replace_the_data_root(femto_root):
    pytest.importorskip("pandas")
    root, _ = femto_root
    moved = root / "copies" / "femto raw"
    moved.parent.mkdir()
    (root / "FEMTO").rename(moved)
    files = acquisition_files(None, "femto", directory=moved)
    assert list(files) == ["Bearing1_1", "Bearing1_2", "Bearing1_3"]
    statistics = prepare(None, "femto", root / "cache.npz", directory=moved)
    assert [len(v) for v in statistics.values()] == [13, 7, 6]
    with pytest.raises(FileNotFoundError, match="layout check"):
        acquisition_files(root, "femto")


def test_prepare_writes_a_cache_once_in_bearing_order(femto_root):
    pytest.importorskip("pandas")
    root, signals = femto_root
    cache = root / "out" / "femto.npz"
    statistics = prepare(root, "femto", cache)
    loaded = load_statistics(cache)
    assert list(loaded) == list(statistics) == ["Bearing1_1", "Bearing1_2", "Bearing1_3"]
    for name, values in loaded.items():
        np.testing.assert_array_equal(values, statistics[name])
        assert values.shape == (len(signals[name]), 6) and values.dtype == np.float64
    with pytest.raises(FileExistsError):
        prepare(root, "femto", cache)
    parallel = prepare(root, "femto", root / "out" / "parallel.npz", jobs=2)
    assert all(np.array_equal(parallel[name], statistics[name]) for name in statistics)
    lives = [h.life for h in histories(loaded, "femto")]
    assert lives == [math.ceil(13 / 6), math.ceil(7 / 6), 1]


def test_statistics_known_values_and_variance_floor():
    alternating = np.array([[1.0, 2.0], [-1.0, 2.0], [1.0, 2.0], [-1.0, 2.0]])
    np.testing.assert_allclose(acquisition_statistics(alternating), [1.0, 2.0, 1.0, 2.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        acquisition_statistics(np.ones((3, 3)))


def test_minutes_aggregate_and_keep_the_partial_last_minute():
    acquisitions = np.ones((13, 6))
    acquisitions[:6, 0] = [1, 1, 1, 1, 1, 7]
    acquisitions[12, 2] = 9.0
    acquisitions[6:12, 4] = np.arange(6)
    minutes = to_minutes(acquisitions, 6)
    assert minutes.shape == (3, 6)
    assert minutes[0, 0] == pytest.approx(math.sqrt((5 + 49) / 6))
    assert minutes[2, 2] == 9.0 and minutes[1, 4] == 2.5
    np.testing.assert_array_equal(to_minutes(acquisitions, 1), acquisitions)


def _history(life=80, seed=0, name="synthetic"):
    rng = np.random.default_rng(seed)
    minutes = np.abs(rng.normal(1.0, 0.1, (life, 6))) + 0.5
    minutes[life // 2:, 0:4] *= np.linspace(1, 6, life - life // 2)[:, None]
    return BearingHistory("femto", name, minutes)


def test_feature_primitives_known_values():
    np.testing.assert_allclose(trailing_slope(np.arange(12.0) * 2, 10), [0, 0] + [2.0] * 10)
    np.testing.assert_allclose(ewm(np.array([1.0, 3.0]), 0.5), [1.0, 2.0])
    np.testing.assert_array_equal(running_in_reference(np.array([[3.0], [1.0], [2.0]]), 2), [[3.0], [2.0], [2.0]])


def test_features_and_observations_are_causal_prefixes():
    history = _history()
    forecast = np.linspace(100, 0, history.life)
    full_features = forecaster_features(history)
    full = decision_series(history, forecast)
    assert full_features.shape == (80, 15) and forecaster_features(history, age=False).shape == (80, 14)
    for k in (1, 3, 11, 40, 79):
        part = BearingHistory("femto", "cut", history.minutes[:k])
        np.testing.assert_array_equal(forecaster_features(part), full_features[:k])
        series = decision_series(part, forecast[:k])
        for key, values in series.items():
            np.testing.assert_array_equal(values, full[key][:k])


def test_policy_observations_and_input_designs():
    history = _history()
    forecast = np.linspace(100, 0, history.life)
    with_age, without_age = decision_series(history, forecast), decision_series(history, forecast[::-1].copy())
    obs = policy_observations(with_age)
    assert obs.dtype == np.float32 and obs.shape == (80, 7)
    np.testing.assert_array_equal(obs[:, 0], np.clip(forecast / 125, 0, 1.2).astype(np.float32))
    np.testing.assert_array_equal(obs[:, 1], np.clip((13 - (forecast + 1)) / 13, -1, 1).astype(np.float32))
    np.testing.assert_array_equal(obs[:, 6], np.clip(np.arange(1, 81) / 250, 0, 4).astype(np.float32))
    widths = {key: len(design.observation_names) for key, design in INPUT_DESIGNS.items()}
    assert widths == {"full": 7, "no_age": 6, "no_policy_age": 6, "no_forecaster_age": 7, "condition_only": 3}
    # The mixed designs recombine the two forecasts without changing any coordinate.
    np.testing.assert_array_equal(design_observations(with_age, "no_policy_age"), obs[:, :6])
    np.testing.assert_array_equal(design_observations(without_age, "no_forecaster_age"),
                                  np.column_stack([design_observations(without_age, "no_age"), obs[:, 6]]))
    condition = design_observations(decision_series(history), "condition_only")
    np.testing.assert_array_equal(condition, obs[:, 3:6])
    with pytest.raises(ValueError, match="forecast"):
        design_observations(decision_series(history), "full")


def test_economics_known_boundaries_for_both_recovery_contracts():
    econ = ReplayEconomics()
    safe, late, reactive = econ.outcome(100, 88), econ.outcome(100, 89), econ.outcome(100, None)
    assert (safe.failed, safe.calendar, safe.margin, safe.unused) == (False, 107, 0, 0)
    assert safe.cash == 255 * 100 - 520 - 25 * 7
    assert (late.failed, late.calendar, late.margin) == (True, 108, -1)
    assert late.cash == 255 * 100 - 520 - 7000 - 25 * 8
    assert reactive.calendar == 119 and reactive.cash == 255 * 100 - 520 - 7000 - 25 * 19
    early = econ.outcome(100, 10)
    assert early.unused == 78 and early.unused_fraction == pytest.approx(78 / 88)
    assert econ.oracle_order(100) == 88
    immediate = ReplayEconomics(recovery_contract="immediate_exchange_v1")
    assert immediate.outcome(100, None).calendar == 107 and immediate.outcome(100, 95).calendar == 107
    with pytest.raises(ValueError):
        econ.outcome(100, 0)
    with pytest.raises(ValueError):
        ReplayEconomics(recovery_contract="other")


@pytest.mark.parametrize("contract", ["preserve_preparation_v1", "immediate_exchange_v1"])
def test_environment_ledgers_equal_the_economics(contract):
    econ = ReplayEconomics(recovery_contract=contract)
    lives = [1, 5, 12, 13, 14, 30]
    rng = np.random.default_rng(3)
    banks = [rng.uniform(0, 0.5, (life, 7)).astype(np.float32) for life in lives]
    env = BearingReplayEnv(banks, economics=econ, observation_names=OBSERVATION_NAMES)
    for index, life in enumerate(lives):
        for order in [*range(1, life + 1), None]:
            observation, info = env.reset(options={"bearing": index})
            np.testing.assert_array_equal(observation, banks[index][0])
            cash, calendar = info["raw_profit_sum"], info["duration_calendar_steps"]
            assert (cash, calendar) == (econ.revenue, 1)
            for minute in range(1, life + 1):
                action = int(order is not None and minute == order)
                expected, terminal = econ.macro(life, minute, action)
                observation, reward, terminated, truncated, info = env.step(action)
                assert [row["raw_profit"] for row in info["ledger"]] == expected
                assert reward == pytest.approx(math.fsum(0.999 ** i * c / 1000 for i, c in enumerate(expected)))
                assert terminated == terminal and not truncated
                assert "last_productive_minute" not in info and info["gamma_pow_duration"] == 0.999 ** len(expected)
                cash, calendar = cash + info["raw_profit_sum"], calendar + info["duration_calendar_steps"]
                if terminated:
                    break
                np.testing.assert_array_equal(observation, banks[index][minute])
            outcome = econ.outcome(life, order)
            assert (cash, calendar) == (outcome.cash, outcome.calendar)
            assert (info["episode_raw_profit"], info["calendar_steps"]) == (outcome.cash, outcome.calendar)
            assert info["failures"] == int(outcome.failed) == info["corrective_replacements"]
            assert env.episode_outcome() == outcome and env.diagnostics()["order_minute"] == order
            with pytest.raises(RuntimeError):
                env.step(0)


def test_environment_event_order_and_gymnasium_contract():
    env = BearingReplayEnv([np.full((20, 3), 0.1, dtype=np.float32)], names=["B"],
                           observation_names=("rms_level", "rms_trend", "peak_level"))
    env.reset(seed=1)
    for _ in range(9):
        env.step(0)
    _, _, terminated, _, info = env.step(1)
    assert [e["event"] for e in info["events"]] == ["mobilize"] * 10 + ["failure_during_mobilization"] \
        + ["corrective_mobilize"] + ["corrective_exchange"] + ["downtime"] * 6
    assert terminated and info["events"][0]["order_placed"]
    assert env.policy_state() == {"minute": 20} and env.diagnostics()["bearing"] == "B"
    check_env(BearingReplayEnv([np.full((30, 2), 0.2, dtype=np.float32)] * 3,
                               observation_names=("forecast", "age")), skip_render_check=True)
    with pytest.raises(ValueError, match="bounds"):
        BearingReplayEnv([np.full((5, 1), 3.0, dtype=np.float32)], observation_names=("forecast",))
    first = [BearingReplayEnv([np.zeros((3, 1), np.float32)] * 5).reset(seed=7)[0] for _ in range(2)]
    np.testing.assert_array_equal(*first)


def test_hashed_folds_reproduce_the_article_assignment():
    salt = "bearings_when_to_act_v1|bearing_folds|femto"
    assert hashed_folds(FEMTO_NAMES, salt, 5) == FEMTO_OUTER_FOLDS
    groups = hashed_folds([f"u{i}" for i in range(11)], "new data", 3)
    assert sorted(sum(groups, [])) == sorted(f"u{i}" for i in range(11)) and [len(g) for g in groups] == [4, 4, 3]
    with pytest.raises(ValueError):
        hashed_folds(["a", "a"], "salt", 2)


def test_rule_grid_order_scores_and_calibration():
    candidates = indicator_candidates()
    families = [c.family for c in candidates]
    assert families == sorted(families, key=["age", "level", "peak", "forecast", "trend"].index)
    counts = {family: families.count(family) for family in dict.fromkeys(families)}
    assert counts["level"] == counts["peak"] == 3 * 81 and counts["forecast"] == 181 and counts["trend"] == 12 * 91
    ages = [-c.threshold for c in candidates if c.family == "age"]
    assert ages[0] == 5 and ages[-1] == 3000 and all(a == round(a) for a in ages) and ages == sorted(set(ages))
    history = _history()
    series = decision_series(history, np.linspace(100, 0, history.life))
    assert apply_rule(RuleCandidate("age", 0.0, -30.0), series) == 30
    assert apply_rule(RuleCandidate("forecast", 0.0, 50.0), series) == first_trigger(series["forecast"], 50.0)
    level = apply_rule(RuleCandidate("level", 0.3, -0.5), series)
    assert level is not None and level > history.life // 2
    selection = calibrate_indicator_rule([series], ReplayEconomics())
    assert set(selection) == {"age", "level", "peak", "forecast", "trend", "best"}
    best = selection["best"]
    assert best.training_profit_rate == max(s.training_profit_rate for s in selection.values())
    # The last safe order (minute 68) is reachable, so the best rule earns the oracle rate.
    econ = ReplayEconomics()
    oracle = econ.outcome(80, 68)
    assert best.training_profit_rate == oracle.cash / oracle.calendar and best.training_failures == 0


def test_forecast_rule_uses_the_float32_coordinate_and_smallest_tied_cutoff():
    forecast = np.array([0.2, 12 / 125, 0.01], dtype=np.float32)
    assert forecast_rule_order(forecast, 12) == 2 and forecast_rule_order(forecast[:1], 12) is None
    with pytest.raises(ValueError):
        forecast_rule_order(forecast.astype(np.float64), 12)
    inputs = [np.linspace(0.8, 0.0, 60).astype(np.float32)]
    selection = calibrate_forecast_rule(inputs, grid_values(0, 90, 0.5))
    orders = {cutoff: forecast_rule_order(inputs[0], cutoff) for cutoff in grid_values(0, 90, 0.5)}
    tied = [c for c, o in orders.items() if o == orders[selection.cutoff]]
    assert selection.cutoff == min(tied) and selection.candidates == 181


def _rows():
    econ = ReplayEconomics()
    rows = []
    for design, orders in (("PPO", {"A": [40, 46], "B": [None, 85]}), ("Rule", {"A": [48], "B": [95]})):
        for bearing, life in (("A", 60), ("B", 100)):
            for seed, order in enumerate(orders[bearing]):
                rows.append(outcome_row(design, "femto", bearing, econ.outcome(life, order), fold=0,
                                        seed=seed if design == "PPO" else None))
    return rows


def test_equal_bearing_metrics_average_seeds_first(tmp_path):
    econ = ReplayEconomics()
    rows = _rows()
    metrics = {(m.design, m.bearing): m for m in bearing_metrics(rows)}
    a = [econ.outcome(60, 40), econ.outcome(60, 46)]
    oracle = econ.outcome(60, 48)
    expected = 100 * ((a[0].cash + a[1].cash) / 2) / ((a[0].calendar + a[1].calendar) / 2) / (oracle.cash / oracle.calendar)
    assert metrics[("PPO", "A")].retained_profit == pytest.approx(expected)
    assert metrics[("Rule", "A")].retained_profit == pytest.approx(100.0)
    assert metrics[("PPO", "B")].failure == 50.0 and metrics[("PPO", "A")].unused_life == pytest.approx(100 * (10 / 48) / 2)
    assert metrics[("PPO", "B")].unused_life == pytest.approx(100 * (3 / 88) / 2)
    summary = {row["design"]: row for row in design_summary(rows)}
    assert summary["PPO"]["failure_pct"] == 25.0 and summary["PPO"]["replays"] == 4
    contrast = paired_bootstrap(rows, "PPO", "Rule", dataset="femto", replicates=200, seed=5)
    assert contrast["point"] == pytest.approx(summary["PPO"]["retained_profit_pct"] - summary["Rule"]["retained_profit_pct"])
    rng = np.random.Generator(np.random.PCG64(5))
    d = np.array([metrics[("PPO", b)].retained_profit - metrics[("Rule", b)].retained_profit for b in ("A", "B")])
    draws = [d[rng.integers(0, 2, 2)].mean() for _ in range(200)]
    assert contrast["ci95"] == pytest.approx(list(np.quantile(draws, [0.025, 0.975])))
    path = tmp_path / "outcomes.csv"
    write_outcomes(path, rows)
    assert read_outcomes(path) == rows
    with pytest.raises(FileExistsError):
        write_outcomes(path, rows)
    with pytest.raises(ValueError, match="same"):
        paired_bootstrap(rows[:-1], "PPO", "Rule", dataset="femto", replicates=10)


def test_forecaster_cross_fit_holds_out_inner_folds(monkeypatch):
    sklearn = pytest.importorskip("sklearn.ensemble")
    from maintenance_arena.bearings import Forecaster, cross_fit

    histories_ = [_history(40 + 5 * i, seed=i, name=f"B{i}") for i in range(5)]
    fitted = []

    class Probe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y):
            self.rows = len(x)
            fitted.append(self)
            return self

        def predict(self, x):
            return np.full(len(x), float(self.rows)) - 1000.0 * (x[:, -1] > 1e9)

    monkeypatch.setattr(sklearn, "HistGradientBoostingRegressor", Probe)
    oof, evaluation = cross_fit(histories_[:4], [["B0", "B2"], ["B1"], ["B3"]], [histories_[4]], age=False)
    lives = {h.name: h.life for h in histories_}
    total = sum(lives[n] for n in ("B0", "B1", "B2", "B3"))
    assert oof["B0"][0] == total - lives["B0"] - lives["B2"] and oof["B1"][0] == total - lives["B1"]
    assert evaluation[0][0] == total and fitted[0].kwargs["early_stopping"] is False
    with pytest.raises(ValueError, match="partition"):
        cross_fit(histories_[:4], [["B0"], ["B1"]])
    real = Forecaster(age=True)
    monkeypatch.undo()
    prediction = real.fit(histories_[:3]).predict(histories_[3])
    assert prediction.shape == (histories_[3].life,) and np.all(prediction >= 0)


def _plain_order(family, parameters, series):
    """The plain minute-by-minute definition of a family's first order (0 for none)."""
    score = rule_score(RuleCandidate(parameters["condition"], parameters["parameter"], 0.0), series)
    for t in range(1, len(score) + 1):
        condition = bool(score[t - 1] <= parameters["threshold"])
        if {"single": condition, "guarded": condition and t >= 10,
                "after_age": condition and t >= parameters.get("minimum_age", 0),
                "or_age": t >= 10 and (condition or t >= parameters.get("age_limit", 0))}[family.kind]:
            return t
    return 0


def test_rule_family_engine_equals_the_plain_definition():
    rng = np.random.default_rng(4)
    series = decision_series(_history(120, seed=2), np.linspace(140, 0, 120) + rng.normal(0, 3, 120))
    assert [f.name for f in RULE_FAMILIES][:5] == ["age", "level", "peak", "forecast", "forecast_trend"]
    assert len(RULE_FAMILIES) == 16 and family_size(RULE_FAMILIES[-1]) == 181 * 192
    for family in RULE_FAMILIES:
        orders = family_orders(family, series)
        assert orders.dtype == np.int32 and len(orders) == family_size(family)
        for index in rng.choice(len(orders), size=min(len(orders), 120), replace=False):
            assert orders[index] == _plain_order(family, family_parameters(family, int(index)), series)


def test_rule_search_refits_families_and_chooses_by_inner_cross_validation():
    economics = ReplayEconomics()
    series = {f"B{i}": decision_series(_history(60 + 15 * i, seed=i),
                                       np.maximum(np.linspace(70 + 15 * i, -10, 60 + 15 * i), 0)) for i in range(6)}
    training = {name: series[name] for name in ("B0", "B1", "B2", "B3")}
    evaluation = {name: series[name] for name in ("B4", "B5")}
    search = search_rule_families(training, [["B0", "B1"], ["B2"], ["B3"]], evaluation, economics=economics)
    assert set(search["families"]) == {f.name for f in RULE_FAMILIES} and search["chosen"] == search["tied"][0]
    calibrated = calibrate_indicator_rule(list(training.values()), economics)
    for base, name in (("age", "age"), ("level", "level"), ("peak", "peak"), ("forecast", "forecast"),
                       ("trend", "forecast_trend")):
        entry, selection = search["families"][name], calibrated[base]
        assert entry["refit_training_rate"] == selection.training_profit_rate
        assert (entry["refit"]["parameter"], entry["refit"]["threshold"]) == (selection.candidate.parameter,
                                                                              selection.candidate.threshold)
        assert entry["evaluation_orders"]["B4"] == apply_rule(selection.candidate, evaluation["B4"])
    best = max(entry["inner_cv_rate"] for entry in search["families"].values())
    assert search["families"][search["chosen"]]["inner_cv_rate"] == best
    with pytest.raises(ValueError, match="partition"):
        search_rule_families(training, [["B0"]], evaluation)


def test_pooled_bootstrap_and_candidate_ranking():
    economics = ReplayEconomics()
    rows = _rows()
    result = pooled_bootstrap(rows, "PPO", "Rule", dataset="femto", replicates=300, seed=9)
    ppo = {"A": [economics.outcome(60, 40), economics.outcome(60, 46)],
           "B": [economics.outcome(100, None), economics.outcome(100, 85)]}
    rule = {"A": [economics.outcome(60, 48)], "B": [economics.outcome(100, 95)]}
    oracle = {"A": [economics.outcome(60, 48)], "B": [economics.outcome(100, 88)]}

    def rate(group, bearings):
        cash = sum(np.mean([o.cash for o in group[b]]) for b in bearings)
        return cash / sum(np.mean([o.calendar for o in group[b]]) for b in bearings)

    def difference(bearings):
        return (rate(ppo, bearings) - rate(rule, bearings)) / rate(oracle, bearings)

    rng = np.random.Generator(np.random.PCG64(9))
    draws = [difference(["AB"[i] for i in rng.integers(0, 2, 2)]) for _ in range(300)]
    assert result["point"] == pytest.approx(difference("AB"))
    assert result["ci95"] == pytest.approx(list(np.quantile(draws, [0.025, 0.975])))
    assert result["share_positive"] == pytest.approx(np.mean(np.asarray(draws) > 0))
    summary = pooled_summary(rows, "PPO", dataset="femto")
    assert (summary["failure_outcomes"], summary["replays"], summary["no_order_outcomes"]) == (1, 4, 1)
    assert summary["pooled_retained_profit_pct"] == pytest.approx(100 * rate(ppo, "AB") / rate(oracle, "AB"))
    twin = [{**row, "design": "PPO long"} for row in rows if row["design"] == "PPO"]
    ranked = rank_candidates(rows + twin, {"PPO long": {"inputs": "full", "calendar_steps": 819200},
                                           "PPO": {"inputs": "full", "calendar_steps": 409600},
                                           "Rule": {"inputs": 3, "calendar_steps": 0}}, dataset="femto")
    profits = {r["design"]: r["profit_per_calendar_minute"] for r in ranked["ranking"]}
    assert profits["PPO"] == profits["PPO long"]
    assert [r["design"] for r in ranked["ranking"] if r["design"].startswith("PPO")] == ["PPO", "PPO long"]
    assert ranked["best_by_inputs"]["full"] == "PPO" and ranked["selected"] == max(profits, key=profits.get)


def test_median_order_and_policy_map():
    assert median_order([5, None, 3, None, 7]) == 7 and median_order([None, 1, None, 2, None]) is None
    assert median_order([4]) == 4
    probabilities = np.zeros((2, 30, 2), dtype=np.float32)
    probabilities[:, :, 0] = 1
    probabilities[0, 20:] = [0.2, 0.8]
    probabilities[1, 25:] = [0.4, 0.6]
    rows = policy_map({"long": {"probabilities": probabilities, "rule_order": 18},
                       "short": {"probabilities": probabilities[:, :20]}})
    assert [r["bearing"] for r in rows] == ["short", "long"] and rows[1]["actor_orders"] == [21, 26]
    assert rows[1]["last_safe_order"] == 18 and rows[1]["rule_order"] == 18 and rows[0]["actor_orders"] == [None, None]
    mean = rows[1]["mean_order_probability"]
    assert mean[24] == pytest.approx(0.4) and mean[25] == pytest.approx(0.7)


def _transfer_rows():
    economics = ReplayEconomics()
    lives = {"X1": 100, "X2": 50, "X3": 900, "X4": 400}
    orders = {"Full": {"X1": [80, 70, None, 95, 88], "X2": [10, 12, 14, 16, 18], "X3": [300, 310, 320, 330, 340],
                       "X4": [None, None, None, 380, 1]},
              "No age": {"X1": [88, 88, 88, 88, 88], "X2": [39, 40, 41, 42, 43], "X3": [880, 870, 860, 850, None],
                         "X4": [380, 381, 382, 383, 384]}}
    rows = [outcome_row(design, "XJTU-SY", bearing, economics.outcome(lives[bearing], order), seed=seed)
            for design, by_bearing in orders.items() for bearing, seeds in by_bearing.items()
            for seed, order in zip(range(5, 10), seeds)]
    rows += [outcome_row("Indicator Rule", "XJTU-SY", b, economics.outcome(lives[b], o))
             for b, o in (("X1", 90), ("X2", 30), ("X3", 888), ("X4", 200))]
    return rows


def test_transfer_timing_medians_status_and_unused_share():
    timing = {row["bearing"]: row for row in transfer_timing(_transfer_rows())}
    assert list(timing) == ["X2", "X1", "X4", "X3"]
    full = timing["X1"]["designs"]["Full"]
    assert (full["order"], full["status"], full["margin"]) == (88, "timely", 0) and full["unused_life_pct"] == 0
    assert full["seed_failures"] == 2 and full["seed_orders"] == [80, 70, None, 95, 88]
    late = timing["X2"]["designs"]["No age"]
    assert late["status"] == "late" and late["unused_life_pct"] is None
    assert timing["X4"]["designs"]["Full"]["status"] == "no_order" and timing["X4"]["designs"]["Full"]["seed_failures"] == 3
    rule = timing["X3"]["designs"]["Indicator Rule"]
    assert rule["status"] == "timely" and rule["unused_life_pct"] == pytest.approx(0.0)
    assert timing["X2"]["designs"]["Full"]["unused_life_pct"] == pytest.approx(100 * (38 - 14) / 38)
    with pytest.raises(ValueError, match="same"):
        transfer_timing([r for r in _transfer_rows() if r["bearing"] != "X3" or r["design"] != "Full"])


def test_age_clock_collects_training_and_longest_target_orders():
    rows = _transfer_rows()
    fits = [{"design": design, "dataset": "XJTU-SY", "fold": None, "seed": seed, "training_bearings": ["F1", "F2"],
             "training_lives": [120, 468], "training_orders": [100, None]}
            for design in ("Full", "No age") for seed in range(5, 10)]
    clock = age_clock(fits, rows, ("Full", "No age"))
    assert [b["bearing"] for b in clock["long_bearings"]] == ["X1", "X4", "X3"]
    assert clock["long_bearings"][-1]["last_safe_order"] == 888 and clock["longest_training_life"] == 468
    first = clock["designs"]["Full"][0]
    assert first["seed"] == 5 and first["training_orders"] == {"F1": 100, "F2": None}
    assert first["target_orders"] == {"X1": 80, "X4": None, "X3": 300}
    with pytest.raises(ValueError, match="no transfer fits"):
        age_clock(fits, rows, ("No policy age",))
