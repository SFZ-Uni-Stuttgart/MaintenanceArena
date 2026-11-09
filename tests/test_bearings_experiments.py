"""The bearing experiment runners end to end on small synthetic statistics caches."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT / "experiments" / "bearings"
pytestmark = pytest.mark.skipif(not (EXPERIMENTS / "common.py").is_file(),
                                reason="experiments/bearings is not part of this checkout")


def _cache(path, names, seed):
    rng = np.random.default_rng(seed)
    arrays = {}
    for index, name in enumerate(names):
        count = 90 + 30 * index
        rms = 0.5 + rng.uniform(0, 0.05, (count, 2))
        rms[count * 2 // 3:] *= np.linspace(1, 4, count - count * 2 // 3)[:, None]
        peak = 4 * rms + rng.uniform(0, 0.1, (count, 2))
        arrays[name] = np.hstack([rms, peak, 3 + rng.uniform(0, 1, (count, 2))])
    np.savez_compressed(path, **arrays)


@pytest.mark.training
def test_runners_write_outcome_rows_for_every_design(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    import common
    import summarize
    from maintenance_arena.bearings import read_outcomes

    femto = [f"S{i}" for i in range(1, 7)]
    configs, cache = tmp_path / "configs", tmp_path / "cache"
    configs.mkdir()
    cache.mkdir()
    settings = json.loads((EXPERIMENTS / "configs" / "settings.json").read_text(encoding="utf-8"))
    settings["forecaster"]["max_iter"] = 10  # the runner logic, not the forecaster, is under test
    (configs / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    (configs / "folds.json").write_text(json.dumps({
        "bearings": femto, "outer_folds": [femto[:3], femto[3:]],
        "inner_folds": [[[n] for n in femto[3:]], [[n] for n in femto[:3]]],
        "all_bearings_inner_folds": [femto[0:2], femto[2:4], femto[4:6]]}), encoding="utf-8")
    experiments = json.loads((EXPERIMENTS / "configs" / "experiments.json").read_text(encoding="utf-8"))
    experiments["seeds"] = {"femto_cross_validation": [[1, 2], [3, 4]], "xjtu_sy_transfer": [5, 6]}
    (configs / "experiments.json").write_text(json.dumps(experiments), encoding="utf-8")
    _cache(cache / "femto_statistics.npz", femto, 0)
    _cache(cache / "xjtu_sy_statistics.npz", ["Bearing1_1", "Bearing2_1", "Bearing3_1"], 1)
    common_options = ["--cache-dir", str(cache), "--config-dir", str(configs), "--calendar-steps", "1024", "--seeds", "1"]
    assert common.main("femto_cross_validation", [*common_options, "--output", str(tmp_path / "cv")]) == 0
    assert common.main("transfer", [*common_options, "--output", str(tmp_path / "transfer")]) == 0
    assert common.main("forecast_rules", [*common_options, "--output", str(tmp_path / "forecast")]) == 0
    cv, transfer = read_outcomes(tmp_path / "cv" / "outcomes.csv"), read_outcomes(tmp_path / "transfer" / "outcomes.csv")
    designs = ["Full", "No age", "Condition only", "Indicator Rule", "Reactive"]
    assert [r["design"] for r in cv][::6] == designs and len(cv) == 30 and {r["dataset"] for r in cv} == {"FEMTO"}
    assert len(transfer) == 15 and {r["fold"] for r in transfer} == {None}
    assert {r["dataset"] for r in transfer} == {"XJTU-SY"} and {r["seed"] for r in transfer if r["design"] == "Full"} == {5}
    assert len(read_outcomes(tmp_path / "forecast" / "outcomes.csv")) == 36
    record = json.loads((tmp_path / "cv" / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "COMPLETE" and record["article_settings"] is False
    fits = json.loads((tmp_path / "cv" / "fits.json").read_text(encoding="utf-8"))
    assert len(fits) == 6 and all((tmp_path / "cv" / fit["policy"]).is_file() for fit in fits)
    with pytest.raises(FileExistsError):
        common.main("femto_cross_validation", [*common_options, "--output", str(tmp_path / "cv")])
    assert summarize.main([str(tmp_path / "cv" / "outcomes.csv"), "--reference", "Indicator Rule",
                           "--output", str(tmp_path / "summary"), "--config-dir", str(configs)]) == 0
    summary = json.loads((tmp_path / "summary" / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["summary"]) == 5 and len(summary["contrasts"]) == 12 and len(summary["pooled_contrasts"]) == 4

    candidates = ["Full (409600 steps)", "No age (819200 steps)"]
    assert common.main("candidate_selection", [*common_options, "--output", str(tmp_path / "select"),
                                               "--designs", *candidates]) == 0
    selection = json.loads((tmp_path / "select" / "selection.json").read_text(encoding="utf-8"))["FEMTO"]
    assert {r["design"] for r in selection["ranking"]} == set(candidates) and selection["selected"] in candidates
    assert common.main("rule_search", [*common_options, "--output", str(tmp_path / "search")]) == 0
    search = read_outcomes(tmp_path / "search" / "outcomes.csv")
    assert len(search) == 18 * 6 and {r["design"] for r in search} >= {"Cross-validated rule",
                                                                       "Rule family by held-out profit"}
    chosen = json.loads((tmp_path / "search" / "selections.json").read_text(encoding="utf-8"))
    assert set(chosen["Cross-validated rule"]) == {"FEMTO/fold0", "FEMTO/fold1"}
    assert common.main("risk_classifier", [*common_options, "--output", str(tmp_path / "classifier")]) == 0
    classified = read_outcomes(tmp_path / "classifier" / "outcomes.csv")
    assert len(classified) == 2 * (6 + 3) and {r["seed"] for r in classified} == {None}
    thresholds = json.loads((tmp_path / "classifier" / "selections.json").read_text(encoding="utf-8"))
    assert set(thresholds["Risk classifier: No age"]) == {"FEMTO/fold0", "FEMTO/fold1", "XJTU-SY/transfer"}
    import run_nested_selection
    nested = ["Full (409600 steps)", "No age (409600 steps)"]
    assert run_nested_selection.main(["--cache-dir", str(cache), "--config-dir", str(configs), "--calendar-steps",
                                      "1024", "--seeds", "1", "--folds", "0", "--designs", *nested,
                                      "--output", str(tmp_path / "nested")]) == 0
    chosen = json.loads((tmp_path / "nested" / "selection.json").read_text(encoding="utf-8"))
    assert set(chosen) == {"0"} and chosen["0"]["selected"] in nested and len(chosen["0"]["ranking"]) == 2
    assert summarize.main([str(tmp_path / "select" / "outcomes.csv"), "--select", "candidate_selection",
                           "--output", str(tmp_path / "ranked"), "--config-dir", str(configs)]) == 0

    import figures
    plot = ["--plot"] if importlib.util.find_spec("matplotlib") else []
    assert common.main("age_designs", [*common_options, "--output", str(tmp_path / "age"),
                                       "--evaluations", "xjtu_sy_transfer"]) == 0
    forecast_rules = str(tmp_path / "forecast" / "outcomes.csv")
    with pytest.raises(ValueError, match="no FEMTO orders of 'Forecast rule: Full'"):
        figures.main(["policy-map", "--run", str(tmp_path / "cv"), "--output", str(tmp_path / "unmarked"),
                      "--config-dir", str(configs)])
    assert figures.main(["policy-map", "--run", str(tmp_path / "cv"), "--rule-outcomes", forecast_rules,
                         "--output", str(tmp_path / "map"), "--config-dir", str(configs), *plot]) == 0
    policy = json.loads((tmp_path / "map" / "policy_map.json").read_text(encoding="utf-8"))
    assert policy["rule_design"] == "Forecast rule: Full" and len(policy["bearings"]) == 6
    assert [r["life"] for r in policy["bearings"]] == sorted(r["life"] for r in policy["bearings"])
    assert figures.main(["transfer-timing", "--outcomes", str(tmp_path / "transfer" / "outcomes.csv"), forecast_rules,
                         "--output", str(tmp_path / "timing"), "--config-dir", str(configs), *plot]) == 0
    timing = json.loads((tmp_path / "timing" / "transfer_timing.json").read_text(encoding="utf-8"))["bearings"]
    assert len(timing) == 3 and set(timing[0]["designs"]) == {"Full", "Forecast rule: Full", "No age",
                                                              "Forecast rule: No age"}
    assert figures.main(["age-clock", "--runs", str(tmp_path / "transfer"), str(tmp_path / "age"),
                         "--output", str(tmp_path / "clock"), "--config-dir", str(configs), *plot]) == 0
    clock = json.loads((tmp_path / "clock" / "age_clock.json").read_text(encoding="utf-8"))
    assert set(clock["designs"]) == {"Full", "No forecaster age", "No policy age", "No age"}
    assert all(len(seed["training_orders"]) == 6 for seeds in clock["designs"].values() for seed in seeds)
    if plot:
        assert all((tmp_path / name / f"{stem}.png").is_file() for name, stem in
                   (("map", "policy_map"), ("timing", "transfer_timing"), ("clock", "age_clock")))


def test_layout_check_reports_missing_folders(tmp_path, capsys, monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    import prepare_statistics

    bearing = tmp_path / "FEMTO" / "Learning_set" / "Bearing1_1"
    bearing.mkdir(parents=True)
    for index in (1, 2, 4):
        (bearing / f"acc_{index:05d}.csv").write_text("", encoding="utf-8")
    assert prepare_statistics.main(["--data-root", str(tmp_path), "--check", "--datasets", "femto"]) == 1
    output = capsys.readouterr().out
    assert "Bearing1_1 (3)" in output and "acquisitions 3 of 1..4" in output
    assert f"missing: folder {tmp_path / 'FEMTO' / 'Full_Test_Set'}" in output and "Bearing2_1" in output
    assert prepare_statistics.main(["--xjtu-sy-dir", str(tmp_path / "nowhere"), "--check", "--datasets", "xjtu_sy"]) == 1
    assert "missing: directory" in capsys.readouterr().out
