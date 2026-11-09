"""Analysis helpers of the article's synthetic figures: health paths, map states, bins and training curves."""
import math

import numpy as np
import pytest

from maintenance_arena import analysis as an
from maintenance_arena.banks import build_bank


@pytest.fixture(scope="module")
def bank():
    return build_bank("source", split="development", seed=5, pair_count=16)


def test_health_paths_take_the_first_lives_of_each_law(bank):
    paths = an.health_paths(bank, per_law=2)
    assert len(paths) == 8 and sorted({p["law"] for p in paths}) == sorted({p["active"]["family_id"] for p in bank["pairs"]})
    first = paths[0]
    assert first["pair_index"] == 0 and first["last_safe_order_age"] == first["end_of_life"] - 13
    assert first["health_at_last_safe_order"] == first["health"][first["last_safe_order_age"]]
    assert len(first["health"]) == first["end_of_life"] + 1
    assert an.health_paths(bank, pairs=[3])[0]["pair_index"] == 3
    with pytest.raises(ValueError):
        an.health_paths(bank)


def test_map_states_cover_whole_lives(bank):
    states = an.map_states(bank, mobilization_lead=24, lives=[1, 0])
    eol = [bank["pairs"][i]["active"]["eol_operating_age"] for i in (0, 1)]
    assert len(states.states) == sum(eol)
    assert states.life.tolist() == [0] * eol[0] + [1] * eol[1]
    assert states.age[:eol[0]].tolist() == list(range(eol[0])) and states.remaining[0] == eol[0]
    assert not states.valid[0] and states.valid[1:eol[0]].all()
    health = bank["pairs"][0]["active"]["health"]
    damage = [min(1., -math.log(h) / math.log(5.)) for h in health[:3]]
    assert states.damage_rate[:3].tolist() == pytest.approx([0., damage[1] - damage[0], damage[2] - damage[1]])
    unclipped = states.valid & (24 * states.damage_rate < 1)
    np.testing.assert_allclose(states.observed_rate[unclipped], states.damage_rate[unclipped], atol=1e-6)
    later = an.map_states(bank, lives=[0], min_age=1)
    assert len(later.states) == eol[0] - 1 and later.age[0] == 1 and later.valid.all()
    chosen = an.first_lives_per_law(bank, 2)
    assert all(len(v) == 2 for v in chosen.values()) and min(min(v) for v in chosen.values()) == 0
    with pytest.raises(ValueError):
        an.first_lives_per_law(bank, 50)


def test_binned_means_weight_states_equally():
    x = np.array([.21, .21, .23, .29, .40])
    y = np.array([.001, .001, .002, .004, .001])
    values = np.array([1., 0., .5, .2, .9])
    x_edges, y_edges = an.edges(.2, .3, .02), an.edges(0., .004, .002)
    counts, means = an.binned_mean(x, y, values, x_edges, y_edges)
    assert counts.sum() == 4                      # x = .40 lies outside the view; y = .004 on the right edge
    assert counts[0, 0] == 2 and means[0, 0] == pytest.approx(.5)
    assert means[4, 1] == pytest.approx(.2) and np.isnan(means[2, 0])
    coordinates, native = an.native_means(x[:2], y[:2], values[:2])
    assert coordinates.tolist() == [[.21, .001]] and native.tolist() == [.5]
    assert an.in_view(x, y, x_edges, y_edges, where=values > .4).tolist() == [True, False, True, False, False]
    probabilities = np.array([[.2, .4], [.6, .8]], dtype=np.float32)
    assert an.ensemble_mean(probabilities).tolist() == pytest.approx([.4, .6])


def test_failure_share_curve_interpolates_the_cumulative_counters():
    log = [{"actual_calendar_steps": 1000 * k, "episodes": 10 * k, "failures": 2 * k} for k in range(1, 5)]
    grid, share, steps = an.failure_share_curve(log, 4000, 2000)
    assert grid.tolist() == [0., 2000., 4000.]
    assert share.tolist() == pytest.approx([20., 20.]) and steps.tolist() == pytest.approx([100., 100.])
    with pytest.raises(ValueError):
        an.failure_share_curve(log, 4000, 3000)
    with pytest.raises(ValueError):
        an.failure_share_curve(log, 8000, 2000)
