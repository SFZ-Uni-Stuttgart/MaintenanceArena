"""Checks on the recorded datasets; skipped unless the data are named by environment variables.

MAINTENANCE_ARENA_DATA names a data root with FEMTO/ and XJTU-SY/ as described in docs/BEARINGS.md;
MAINTENANCE_ARENA_FEMTO_DIR and MAINTENANCE_ARENA_XJTU_SY_DIR name a dataset directory directly.
"""
import math
import os

import numpy as np
import pytest

from maintenance_arena.bearings import (acquisition_files, acquisition_statistics, check_layout, dataset_directory,
                                        read_acquisition)

ROOT = os.environ.get("MAINTENANCE_ARENA_DATA")
DIRECTORIES = {"femto": os.environ.get("MAINTENANCE_ARENA_FEMTO_DIR"),
               "xjtu_sy": os.environ.get("MAINTENANCE_ARENA_XJTU_SY_DIR")}

FEMTO_ACQUISITIONS = {"Bearing1_1": 2803, "Bearing1_2": 871, "Bearing2_1": 911, "Bearing2_2": 797, "Bearing3_1": 515,
                      "Bearing3_2": 1637, "Bearing1_3": 2375, "Bearing1_4": 1428, "Bearing1_5": 2463,
                      "Bearing1_6": 2448, "Bearing1_7": 2259, "Bearing2_3": 1955, "Bearing2_4": 751,
                      "Bearing2_5": 2311, "Bearing2_6": 701, "Bearing2_7": 230, "Bearing3_3": 434}
FEMTO_LIVES = {"Bearing1_1": 468, "Bearing1_2": 146, "Bearing2_1": 152, "Bearing2_2": 133, "Bearing3_1": 86,
               "Bearing3_2": 273, "Bearing1_3": 396, "Bearing1_4": 238, "Bearing1_5": 411, "Bearing1_6": 408,
               "Bearing1_7": 377, "Bearing2_3": 326, "Bearing2_4": 126, "Bearing2_5": 386, "Bearing2_6": 117,
               "Bearing2_7": 39, "Bearing3_3": 73}
# XJTU-SY records one acquisition per minute, so these counts are also the lives.
XJTU_SY_ACQUISITIONS = {"Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158, "Bearing1_4": 122, "Bearing1_5": 52,
                        "Bearing2_1": 491, "Bearing2_2": 161, "Bearing2_3": 533, "Bearing2_4": 42, "Bearing2_5": 339,
                        "Bearing3_1": 2538, "Bearing3_2": 2496, "Bearing3_3": 371, "Bearing3_4": 1515, "Bearing3_5": 114}


def _directory(dataset):
    if DIRECTORIES[dataset] is None and ROOT is None:
        pytest.skip("set MAINTENANCE_ARENA_DATA or the dataset directory variable")
    directory = dataset_directory(ROOT, dataset, directory=DIRECTORIES[dataset])
    if not directory.is_dir():
        pytest.skip(f"{directory} is not available")
    return directory


def test_femto_layout_inventory_and_lifetimes():
    directory = _directory("femto")
    report = check_layout(None, "femto", directory=directory)
    assert report["ok"], report["missing"] + report["problems"]
    files = acquisition_files(None, "femto", directory=directory)
    assert {name: len(paths) for name, paths in files.items()} == FEMTO_ACQUISITIONS
    assert list(files) == list(FEMTO_ACQUISITIONS)
    assert {name: math.ceil(len(paths) / 6) for name, paths in files.items()} == FEMTO_LIVES


def test_femto_acquisitions_reproduce_reference_statistics():
    pytest.importorskip("pandas")
    files = acquisition_files(None, "femto", directory=_directory("femto"))
    first = read_acquisition(files["Bearing1_1"][0], "femto")
    assert first.shape == (2560, 2)
    np.testing.assert_allclose(acquisition_statistics(first), [0.56174565773522, 0.4358014232064301, 2.01, 1.591,
                                                               2.868534971991775, 2.9649195538613085], rtol=1e-12)
    last = read_acquisition(files["Bearing2_7"][-1], "femto")
    np.testing.assert_allclose(acquisition_statistics(last), [5.400414711686791, 2.0040052515018205, 16.804, 17.547,
                                                              3.3917561191377623, 15.352467556226008], rtol=1e-12)


def test_xjtu_sy_layout_has_fifteen_bearings():
    pytest.importorskip("pandas")
    directory = _directory("xjtu_sy")
    report = check_layout(None, "xjtu_sy", directory=directory)
    assert report["ok"], report["missing"] + report["problems"]
    files = acquisition_files(None, "xjtu_sy", directory=directory)
    assert list(files) == [f"Bearing{c}_{i}" for c in (1, 2, 3) for i in range(1, 6)]
    assert {name: len(paths) for name, paths in files.items()} == XJTU_SY_ACQUISITIONS
    first = read_acquisition(files["Bearing1_1"][0], "xjtu_sy")
    assert first.shape == (32768, 2) and np.isfinite(acquisition_statistics(first)).all()
