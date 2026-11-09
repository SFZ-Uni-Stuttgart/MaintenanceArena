"""Equal-life decision metrics, their paired bootstrap and the outcome tables."""
import numpy as np
import pytest

from maintenance_arena.metrics import (OUTCOME_COLUMNS, life_table, paired_bootstrap, read_outcomes, summarize,
                                       summarize_outcomes, write_outcomes)
from maintenance_arena.replay import Economics, replay


def rows(method, orders, lives, runs=None, group="g"):
    result = []
    for k, order_list in enumerate(orders):
        outcome = replay(lives, np.asarray(order_list), Economics())
        for unit, (life, order) in enumerate(zip(lives, order_list)):
            result.append({"method": method, "group": group, "run": "" if runs is None else runs[k], "unit": unit,
                           "last_productive_step": int(life), "order_step": "" if order < 0 else int(order),
                           "cash": float(outcome["cash"][unit]), "calendar_steps": int(outcome["calendar"][unit]),
                           "failed": bool(outcome["failed"][unit]),
                           "unused_fraction": float(outcome["unused_fraction"][unit])})
    return result


def test_known_endpoints():
    # A last productive step of 128 leaves 116 steps to the last safe order (L = 12).
    lives = [128, 128, 128, 128]
    for order, failure, unused in ((0, 0., 100.), (116, 0., 0.), (117, 100., 0.), (-1, 100., 0.)):
        table = life_table(rows("m", [[order]], [128]))
        assert (table.failure_pct[0], table.unused_life_pct[0]) == (failure, unused)
    reference = summarize(life_table(rows("oracle", [[116] * 4], lives)))
    assert reference == {"lives": 4, "runs": 1, "failure_pct": 0., "unused_life_pct": 0., "retained_profit_pct": 100.}


def test_runs_are_averaged_before_the_profit_ratio():
    table = life_table(rows("ppo", [[0], [116]], [128], runs=[1, 2]))
    early, safe = replay([128], np.asarray([0])), replay([128], np.asarray([116]))
    rate = (early["cash"][0] + safe["cash"][0]) / (early["calendar"][0] + safe["calendar"][0])
    assert table.retained_profit_pct[0] == pytest.approx(100 * rate / (safe["cash"][0] / safe["calendar"][0]))
    assert table.unused_life_pct[0] == 50. and table.runs == 2
    with pytest.raises(ValueError, match="same number of runs"):
        life_table(rows("ppo", [[0, 1]], [128, 128], runs=[1]) + rows("ppo", [[0]], [128], runs=[2]))


def test_paired_bootstrap_is_deterministic_and_paired():
    rng = np.random.default_rng(0)
    lives = rng.integers(100, 160, 60)
    first = life_table(rows("a", [list(lives - 12 - rng.integers(0, 4, 60))], lives))
    second = life_table(rows("b", [list(lives - 12 - rng.integers(-1, 3, 60))], lives))
    tables = {"a": first, "b": second}
    left = paired_bootstrap(tables, contrasts=[("a", "b"), ("a", "a")], replicates=400, seed=3)
    right = paired_bootstrap(tables, contrasts=[("a", "b"), ("a", "a")], replicates=400, seed=3)
    assert left == right
    assert left["contrasts"]["a - a"]["failure_pct"] == {"estimate": 0., "ci95": [0., 0.]}
    for metric, value in left["contrasts"]["a - b"].items():
        assert value["ci95"][0] <= value["estimate"] <= value["ci95"][1], metric
    strata = paired_bootstrap(tables, replicates=200, seed=3, strata=[i % 2 for i in range(60)])
    assert strata["strata"] == 2
    with pytest.raises(ValueError):
        paired_bootstrap(tables, contrasts=[("a", "c")])


def test_outcome_tables_roundtrip(tmp_path):
    data = rows("Rule", [[100, -1, 30]], [127, 140, 150]) + rows("PPO", [[100, 120, 30]], [127, 140, 150], runs=[7])
    path = tmp_path / "outcomes.csv"
    assert write_outcomes(path, data) == 6
    with pytest.raises(FileExistsError):
        write_outcomes(path, data)
    loaded = read_outcomes(path)
    assert tuple(loaded[0]) == OUTCOME_COLUMNS
    summary = {row["method"]: row for row in summarize_outcomes(loaded)}
    direct = summarize(life_table([row for row in data if row["method"] == "Rule"]))
    assert summary["Rule"]["retained_profit_pct"] == pytest.approx(direct["retained_profit_pct"], rel=1e-15)
    assert summary["Rule"]["failure_pct"] == pytest.approx(100 / 3)
