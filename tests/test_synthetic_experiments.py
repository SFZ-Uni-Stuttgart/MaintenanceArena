"""Configurations and seed derivations of the synthetic experiment runners."""
import importlib.util
import json
import os
import py_compile
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "synthetic"
RUNNERS = ("source_learning", "input_sets", "preparation_time", "adaptation", "measurement_noise", "benchmarks",
           "learner_comparison", "age_shift", "failure_cost")
if not EXPERIMENTS.is_dir():
    pytest.skip("the experiment runners are not available next to the tests", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("synthetic_experiment_common", EXPERIMENTS / "common.py")
common = importlib.util.module_from_spec(_spec)
_environment = dict(os.environ)
_spec.loader.exec_module(common)  # the runners pin BLAS threads; keep the test process unchanged
os.environ.clear()
os.environ.update(_environment)


def config(name):
    return json.loads((EXPERIMENTS / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", RUNNERS)
def test_runners_compile_and_configurations_are_complete(name, tmp_path):
    py_compile.compile(str(EXPERIMENTS / f"{name}.py"), cfile=str(tmp_path / f"{name}.pyc"), doraise=True)
    values = config(name)
    assert {"study", "description", "environment", "roles"} <= set(values)
    assert values["environment"]["recovery_contract"] == "preserve_preparation_v1"
    assert len(values["roles"]) == 10


def test_article_seeds_are_derived_from_the_configurations():
    source = config("source_learning")
    bank = common.resolve_bank(source, source["training"]["bank"], role=26101601)
    assert (bank["seed"], bank["pairs"], bank["split"]) == (2777621814, 10781, "source_train")
    assert common.resolve_bank(source, source["rule"]["calibration_bank"])["seed"] == 2257774815
    assert common.resolve_bank(source, source["development_bank"])["seed"] == 2222611548
    inputs = config("input_sets")
    assert common.resolve_bank(inputs, inputs["training"]["bank"], role=26101801)["seed"] == 1896708079
    banks = (inputs["rule"]["calibration_bank"], inputs["diagnostic_bank"], inputs["test_bank"])
    assert [common.resolve_bank(inputs, spec)["seed"] for spec in banks] == [2485111423, 3414505248, 3416595760]
    preparation = config("preparation_time")
    assert common.resolve_bank(preparation, preparation["training"]["bank"], role=26101701)["seed"] == 1246971712
    assert common.resolve_bank(preparation, preparation["rule"]["calibration_bank"])["seed"] == 2485111423
    assert common.resolve_bank(preparation, preparation["test_bank"])["seed"] == 3416595760
    adaptation = config("adaptation")
    law = "paris_inspired"
    assert common.resolve_bank(adaptation, adaptation["training"]["bank"], law=law, role=26101601)["seed"] == 2727233850
    assert common.seed_of(adaptation, adaptation["learner_seed_purpose"], law, 26101601) == 3059666802
    assert common.resolve_bank(adaptation, adaptation["rule"]["calibration_bank"], law=law)["seed"] == 1683905899
    assert common.resolve_bank(adaptation, adaptation["test_bank"], law=law)["seed"] == 3033974346
    noise = config("measurement_noise")
    assert common.resolve_bank(noise, noise["training"]["extension"], role=26101601)["seed"] == 103587399
    assert common.resolve_bank(noise, noise["calibration_bank"])["seed"] == 2257774815
    assert common.resolve_bank(noise, noise["test_bank"])["seed"] == 4086758883
    # The learner comparison uses the run index, not the training seed, as role.
    learners = config("learner_comparison")
    banks = [common.resolve_bank(learners, learners["training"]["bank"], role=run) for run in (0, 9)]
    assert [(bank["seed"], bank["pairs"]) for bank in banks] == [(1532560933, 10784), (855851505, 10784)]
    assert common.resolve_bank(learners, learners["diagnostic_bank"])["seed"] == 717418085
    test = common.resolve_bank(learners, learners["test_bank"])
    assert (test["seed"], test["split"], test["pairs"]) == (3642636118, "final_evaluation", 2000)


def test_noise_development_configuration():
    noise = config("measurement_noise")
    dev = noise["development"]
    seeds = [common.resolve_bank(noise, spec["bank"])["seed"] for spec in dev["rounds"]]
    assert seeds == [1920905703, 963687806, 1727438466, 4108557447]
    for spec in dev["rounds"]:
        assert set(spec["designs"] + spec["descriptive"]) <= set(dev["designs"])
        assert set(spec["controls"]) <= set(dev["controls"]) and set(spec["rules"]) <= set(dev["rules"])
        assert noise["sensor_level"] in spec["cells"].values() and spec["decision"] in ("greedy", "cut")
    for name, design in dev["designs"].items():
        run = dev["runs"][design["run"]]
        assert design["steps"] in run["milestones"] and design["noise_steps"] == design["steps"]
        source = dev["source_steps"] if run["start"] == "source" else 0
        assert design["total_steps"] == design["steps"] + source
    assert dev["rounds"][-1]["designs"][-1] == noise["confirmed_design"]
    assert dev["runs"][dev["designs"][noise["confirmed_design"]]["run"]]["milestones"] == noise["training"]["milestones"]


def test_noise_runner_scales_all_development_runs_alike():
    pytest.importorskip("torch")
    import sys
    sys.path.insert(0, str(EXPERIMENTS))
    environment, previous = dict(os.environ), sys.modules.get("common")
    try:
        spec = importlib.util.spec_from_file_location("synthetic_measurement_noise", EXPERIMENTS / "measurement_noise.py")
        noise_runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(noise_runner)
    finally:
        sys.path.remove(str(EXPERIMENTS))
        os.environ.clear()
        os.environ.update(environment)
        # The runner imports the experiments' ``common`` module; leave no trace for other tests.
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    noise = config("measurement_noise")
    full = noise_runner.development_milestones(noise, None)
    assert full["line_fit_fresh"][-1] == 819200 and full["raw_tuned"] == [12800, 51200, 102400, 204800]
    quick = noise_runner.development_milestones(noise, 8192)
    assert quick["raw_fresh"] == [512, 1024, 2048, 4096, 8192] and quick["both_tuned"] == [128, 512, 1024, 2048]
    assert noise_runner.design_milestone(noise, quick, "raw_tuned_204800") == 2048
    with pytest.raises(SystemExit):
        noise_runner.development_milestones(noise, 4)


def test_figure_configuration_seeds(tmp_path):
    figures = config("figures")
    banks = [common.resolve_bank(figures, spec) for spec in figures["preparation_maps"]["banks"]]
    assert [(bank["seed"], bank["pairs"]) for bank in banks] == [(3967981468, 128), (26101753, 1920)]
    spec = figures["adaptation_maps"]["bank"]
    assert [common.resolve_bank(figures, spec, law=law)["seed"] for law in figures["adaptation_maps"]["laws"]] == \
        [577581305, 818724502, 3152975885, 1478178822]
    py_compile.compile(str(EXPERIMENTS / "figures.py"), doraise=True, cfile=str(tmp_path / "figures.pyc"))


def test_quick_check_budgets_scale_the_milestones():
    assert common.scaled([51200, 102400, 204800], None) == [51200, 102400, 204800]
    assert common.scaled([51200, 102400, 204800], 2048) == [512, 1024, 2048]
    assert common.scaled([12800, 51200], 1024) == [256, 1024]
    # A configured milestone as the budget keeps the exact beginning of the full runs.
    assert common.scaled([51200, 102400, 204800], 102400) == [51200, 102400]
    with pytest.raises(SystemExit):
        common.scaled([51200, 102400, 204800], 2)
    with pytest.raises(SystemExit):
        common.scaled([51200], 60000)
