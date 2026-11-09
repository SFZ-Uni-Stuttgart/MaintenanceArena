"""Values that the article quotes in its text, computed from the saved outcomes in paper/."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
if not (PAPER / "order_margins.py").is_file():
    pytest.skip("paper/ is not available next to the tests", allow_module_level=True)


def _load(name):
    sys.path.insert(0, str(PAPER))
    try:
        spec = importlib.util.spec_from_file_location(f"paper_{name}", PAPER / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PAPER))
    return module


def test_order_margins_of_the_source_and_new_laws_and_the_xjtu_bearings():
    values = _load("order_margins").compute()
    assert values["source_laws"]["PPO"] == {"median_margin": 1.0, "late_pct": 0.0, "rows": 20000}
    assert values["source_laws"]["Naive projection rule"]["late_pct"] == 63.95
    cut = values["new_laws"]["Fine-tuned PPO, calibrated cut"]
    assert [cut[law]["median_margin"] for law in ("Paris-inspired", "Two-knee", "Gompertz", "Power-law")] == [2, 2, 1, 0]
    xjtu = values["xjtu_per_bearing"]
    assert xjtu["femto_longest_last_productive_minute"] == 468 and len(xjtu["rows"]) == 15
    assert xjtu["beyond_range_all_negative"] and xjtu["beyond_range_span_pp"] == [-16.7, -83.97]


def test_transfer_split_reproduces_the_article_interval():
    pytest.importorskip("numpy")
    values = _load("transfer_split").compute()
    assert values["all_15"] == {"bearings": 15, "point": -17.0267, "ci95": [-34.2654, -0.9191]}
    assert values["within_femto_range"]["bearings"] == 10 and values["beyond_femto_range"]["bearings"] == 5
    assert values["beyond_femto_range"]["point"] == -54.5845
