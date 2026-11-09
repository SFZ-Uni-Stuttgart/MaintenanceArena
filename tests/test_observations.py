"""Prefix causality of the simulated observations."""
import math

import numpy as np
import pytest

from maintenance_arena.curves import materialize_health
from maintenance_arena.observations import Reconstructor, reconstruct, measurement_path, ROUTE_IDS


@pytest.mark.parametrize("route", ROUTE_IDS)
def test_prefix_causality_missingness_and_reset(route):
    values = materialize_health("convex_accelerating_fatigue", 128)
    available = np.ones(len(values), dtype=bool)
    available[8:12] = False
    full = reconstruct(values, available, route)
    prefix = reconstruct(values[:31], available[:31], route)
    np.testing.assert_array_equal(full["states"][:31], prefix["states"])
    estimator = Reconstructor(route)
    for value in values[:30]:
        estimator.step(float(value))
    estimator.reset()
    np.testing.assert_array_equal(estimator.step(1.), [1., 0., 0., -1.])
    with pytest.raises(ValueError):
        Reconstructor(route).step(None)


def test_route_specific_gap_and_startup_semantics():
    measurements = np.asarray([1., .98, .97, .95, .93, .92, .91, .90], dtype=np.float32)
    available = np.asarray([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    raw = reconstruct(measurements, available, "RAW_CAUSAL")
    ewls = reconstruct(measurements, available, "FILTERED_CAUSAL_EWLS8")
    robust = reconstruct(measurements, available, "ROBUST_MULTISCALE_234")
    assert not raw["states"][5, 2]
    assert ewls["states"][6, 2] and not ewls["states"][7, 2]
    assert robust["states"][7, 2]  # Do not add EWLS's gap limit to this method.
    assert not robust["states"][2, 2] and robust["states"][3, 2]


def test_sensor_stationary_bias_clipping_and_paired_presets():
    health = np.asarray([1., .8, .5, .2], dtype=np.float32)
    kwargs = dict(white_noise=np.asarray([1., 0., -1., 2.]), initial_bias=2.,
                  bias_innovations=np.asarray([0., 1., -2., 3.]), missing_uniforms=np.asarray([0., .009, .04, .2]))
    exact = measurement_path(health, level="exact", **kwargs)
    np.testing.assert_array_equal(exact["measured_health"], health)
    assert exact["available"].all()
    light = measurement_path(health, level="light", **kwargs)
    assert light["available"].tolist() == [True, False, True, True]
    assert light["bias"][0] == .00025 / math.sqrt(1 - .98 ** 2) * 2
    assert light["bias"][1] == .98 * light["bias"][0] + .00025
    assert light["measured_health"][0] == 1.


def test_gaussian_only_preset_has_no_bias_or_missingness():
    health = np.asarray([1., .8, .5, .2, 0.], dtype=np.float32)
    white = np.asarray([1., 0., -1., 2., -1.])
    measured = measurement_path(health, level="gaussian_005", white_noise=white,
                                initial_bias=100., bias_innovations=np.ones(5),
                                missing_uniforms=np.zeros(5))
    expected = np.clip(health.astype(float) + .005 * white, 0., 1.).astype(np.float32)
    np.testing.assert_array_equal(measured["measured_health"], expected)
    np.testing.assert_array_equal(measured["bias"], np.zeros(5))
    assert measured["available"].all()


def test_raw_noisy_differences_use_measured_prefix_and_preserve_validity_contract():
    measured = np.asarray([.8, .81, .79, .78], dtype=np.float32)
    result = reconstruct(measured)
    # Two available readings are valid even when noise creates an apparent recovery.
    np.testing.assert_array_equal(result["states"][1], np.asarray([.81, 0., 1., -1.], dtype=np.float32))
    damage = -np.log(measured.astype(float)) / math.log(5.)
    expected_rate = np.float32(12. * (damage[2] - damage[1]))
    remaining = (1. - damage[2]) / (float(expected_rate) / 12.)
    np.testing.assert_array_equal(result["states"][2], np.asarray(
        [measured[2], expected_rate, 1., np.clip((13. - remaining) / 13., -1., 1.)],
        dtype=np.float32))
    changed_future = measured.copy()
    changed_future[3] = .2
    np.testing.assert_array_equal(reconstruct(changed_future)["states"][:3], result["states"][:3])
