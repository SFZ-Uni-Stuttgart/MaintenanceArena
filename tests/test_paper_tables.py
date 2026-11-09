"""The data tables of the article and the additional tables, rebuilt from the saved outcomes in paper/, equal their
table files."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "paper" / "rebuild_tables.py"
STEMS = ("table2_source_comparison", "table3_measurement_noise", "table4_degradation_change", "table5_bearings",
         "tableA2_learners", "additional_source_controls", "additional_calibration_comparisons",
         "additional_learner_comparison", "additional_bearing_comparison")


@pytest.fixture(scope="module")
def rebuild():
    if not SCRIPT.is_file():
        pytest.skip("paper/ is not available next to the tests")
    spec = importlib.util.spec_from_file_location("rebuild_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def texts(rebuild):
    return rebuild.build()


def test_the_nine_tables_have_expected_copies(rebuild, texts):
    assert tuple(table.stem for table in rebuild.TABLES) == STEMS == tuple(texts)
    assert sorted(path.name for path in rebuild.EXPECTED.iterdir()) == sorted(f"{stem}.tex" for stem in STEMS)


@pytest.mark.parametrize("stem", STEMS)
def test_rebuilt_table_equals_the_article_table_byte_for_byte(rebuild, texts, stem):
    expected = (rebuild.EXPECTED / f"{stem}.tex").read_bytes()
    assert b"\r" not in expected and expected.endswith(b"\\end{tabular}}\n")
    assert texts[stem].encode("utf-8") == expected


def test_check_command_passes_from_another_working_directory(rebuild, tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count(": identical") == len(STEMS)


def test_check_command_reports_a_changed_cell_with_status_1(rebuild, tmp_path, capsys):
    expected = tmp_path / "expected"
    shutil.copytree(rebuild.EXPECTED, expected)
    path = expected / "table4_degradation_change.tex"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("& 99.51\\\\", "& 99.52\\\\"), encoding="utf-8", newline="\n")
    assert path.read_text(encoding="utf-8") != original
    assert rebuild.main(["--check", "--expected", str(expected)]) == 1
    captured = capsys.readouterr()
    assert "Table 4 (table4_degradation_change.tex): differs" in captured.err
    assert "+Fine-tuned PPO & 1.76 & 0.73 & 99.51\\\\" in captured.err
    assert captured.out.count(": identical") == len(STEMS) - 1


def test_output_writes_the_tables_and_markdown_once(rebuild, tmp_path, capsys):
    output = tmp_path / "tables"
    assert rebuild.main(["--output", str(output)]) == 0
    for stem in STEMS:
        assert (output / f"{stem}.tex").read_bytes() == (rebuild.EXPECTED / f"{stem}.tex").read_bytes()
    markdown = (output / "tables.md").read_text(encoding="utf-8")
    assert "| Per-law rule (privileged) | 0.05 | <0.01 | 99.98 |" in markdown
    assert "| PPO: Full | 27.06 | 33.10 | 85.51 | 14.67 | 46.56 | 88.31 |" in markdown
    assert "| Double DQN | 14.58 | 8.82 | 94.38 | 7/10 | 4,610 |" in markdown
    # Every table prints the article's names instead of the stored ones.
    assert "| Time rule, τ = 13 | 63.95 | 0 | 85.21 |" in markdown
    assert "| Source PPO + filter + calibration | 0.77 | 6.50 | 0 | 2.69 |" in markdown
    assert "| Fine-tuned PPO + calibration | 0.21 | 0.86 | 99.89 |" in markdown
    assert "| PPO: No prediction age | 29.41 | 44.97 | 83.45 | 13.33 | 43.66 | 88.72 |" in markdown
    assert "| Time rule: Full | 41.18 | 36.78 | 79.12 | 26.67 | 32.33 | 84.54 |" in markdown
    assert "| Indicator rule | 35.29 | 41.19 | 81.40 | 20.00 | 27.54 | 85.90 |" in markdown
    assert "Forecast rule" not in markdown and "calibrated cut" not in markdown
    with pytest.raises(SystemExit) as stop:
        rebuild.main(["--output", str(output)])
    assert stop.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_first_order_accounting_and_life_weights(rebuild):
    # T = 128: an immediate order, the last safe order, one step late and no order.
    assert rebuild.consequences(128, 0) == (2365.0, 19.0, 0, 116.0)
    assert rebuild.consequences(128, 116) == (31945.0, 135.0, 0, 0.0)
    assert rebuild.consequences(128, 117) == (24920.0, 136.0, 1, 0.0)
    assert rebuild.consequences(128, None) == (24645.0, 147.0, 1, 0.0)
    runs = [rebuild.Outcome("PPO", "", "Linear", "", str(run), "0", 128, order) for run, order in enumerate((0, 116))]
    summary = rebuild.summarize(runs)
    # Cash and calendar steps are averaged over the runs of a life before their ratio is taken.
    assert summary["failures"] == 0 and summary["unused_life"] == 50
    assert summary["retained_profit"] == pytest.approx(100 * ((2365 + 31945) / (19 + 135)) / (31945 / 135))
    assert rebuild.cell(0.0) == "0" and rebuild.cell(100.0) == "100" and rebuild.cell(0.0025) == r"$<0.01$"
    assert rebuild.cell(0.125) == "0.13" and rebuild.cell(99.995) == "100.00"


@pytest.mark.parametrize("old,new,message", [
    ("PPO: Full,FEMTO,4,0,Bearing1_1,468,220\n", "", r"Bearing1_1 has runs \['1', '2', '3', '4'\]"),
    ("Forecast rule: Full,FEMTO,4,,Bearing1_1,468,285", "Forecast rule: Full,FEMTO,4,,Bearing1_1,468,469",
     "order_step 469 lies after the last productive step 468"),
    ("Reactive,FEMTO,4,,Bearing1_1,468,", "Reactive,FEMTO,4,,Bearing1_1,467,", "not evaluated on the same lives"),
    ("Forecast rule: Full,FEMTO,4,,Bearing1_1", "Forecast rule: Full,FEMTO,3,,Bearing1_1", "one outer fold"),
    ("Indicator Rule,XJTU-SY,,,Bearing1_1", "Indicator rule,XJTU-SY,,,Bearing1_1", "unknown method 'Indicator rule'"),
    ("Bearing1_1,468,220\n", "Bearing1_1,468,220.0\n", "order_step must be a nonnegative integer"),
    ("last_productive_step,order_step\n", "last_productive_step,order\n", "columns"),
])
def test_invalid_outcomes_are_rejected_with_a_clear_message(rebuild, tmp_path, old, new, message):
    text = (rebuild.OUTCOMES / "bearings.csv").read_text(encoding="utf-8")
    assert old in text
    (tmp_path / "bearings.csv").write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match=message):
        rebuild.read_study("bearings", tmp_path)
