"""Raw bearing recordings: file layout, per-acquisition statistics and minute histories.

Each acquisition (FEMTO: 2,560 samples every 10 s; XJTU-SY: 32,768 samples once a
minute; both at 25.6 kHz from a horizontal and a vertical accelerometer) is reduced to
RMS, peak absolute amplitude and kurtosis per channel. Every statistic depends on its own
acquisition only. Raw files are read with pandas, which the ``prepare`` step requires.

Below a data root, the datasets live in ``FEMTO/{Learning_set,Full_Test_Set}/BearingC_R/
acc_NNNNN.csv`` and ``XJTU-SY/{35Hz12kN,37.5Hz11kN,40Hz10kN}/BearingC_R/N.csv``. Every
function also accepts the dataset directory itself, so other layouts need no copying.
"""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import re
import uuid

import numpy as np

STATISTICS = ("rms_h", "rms_v", "peak_h", "peak_v", "kurt_h", "kurt_v")


@dataclass(frozen=True)
class DatasetLayout:
    """Folders of one dataset and the bearings and acquisition counts used in the article.

    ``expected_acquisitions`` is None where a count is not part of the layout check.
    """
    name: str
    directory: str
    folders: tuple[str, ...]
    file_pattern: str
    acquisitions_per_minute: int
    bearings: tuple[tuple[str, ...], ...]
    expected_acquisitions: Mapping[str, int] | None = None


_FEMTO_COUNTS = {"Bearing1_1": 2803, "Bearing1_2": 871, "Bearing2_1": 911, "Bearing2_2": 797, "Bearing3_1": 515,
                 "Bearing3_2": 1637, "Bearing1_3": 2375, "Bearing1_4": 1428, "Bearing1_5": 2463,
                 "Bearing1_6": 2448, "Bearing1_7": 2259, "Bearing2_3": 1955, "Bearing2_4": 751,
                 "Bearing2_5": 2311, "Bearing2_6": 701, "Bearing2_7": 230, "Bearing3_3": 434}
_XJTU_SY_COUNTS = {"Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158, "Bearing1_4": 122, "Bearing1_5": 52,
                   "Bearing2_1": 491, "Bearing2_2": 161, "Bearing2_3": 533, "Bearing2_4": 42, "Bearing2_5": 339,
                   "Bearing3_1": 2538, "Bearing3_2": 2496, "Bearing3_3": 371, "Bearing3_4": 1515, "Bearing3_5": 114}
DATASETS = {
    "femto": DatasetLayout("femto", "FEMTO", ("Learning_set", "Full_Test_Set"), r"acc_(\d+)\.csv", 6,
                           (tuple(list(_FEMTO_COUNTS)[:6]), tuple(list(_FEMTO_COUNTS)[6:])), _FEMTO_COUNTS),
    "xjtu_sy": DatasetLayout("xjtu_sy", "XJTU-SY", ("35Hz12kN", "37.5Hz11kN", "40Hz10kN"), r"(\d+)\.csv", 1,
                             tuple(tuple(f"Bearing{c}_{i}" for i in range(1, 6)) for c in (1, 2, 3)), _XJTU_SY_COUNTS),
}


def layout(dataset: str) -> DatasetLayout:
    if dataset not in DATASETS:
        raise ValueError(f"unknown bearing dataset {dataset!r}; expected one of {sorted(DATASETS)}")
    return DATASETS[dataset]


def dataset_directory(data_root: str | os.PathLike | None, dataset: str, *,
                      directory: str | os.PathLike | None = None) -> Path:
    """The folder holding a dataset's condition folders: ``directory`` if given, else below ``data_root``."""
    spec = layout(dataset)
    if directory is not None:
        return Path(directory)
    if data_root is None:
        raise ValueError(f"name a data root or the {spec.directory} directory")
    return Path(data_root) / spec.directory


def _numbered(bearing: Path, pattern: re.Pattern) -> list[tuple[int, Path]]:
    return sorted((int(match.group(1)), path) for path in bearing.iterdir() if (match := pattern.fullmatch(path.name)))


def acquisition_files(data_root: str | os.PathLike | None, dataset: str, *,
                      directory: str | os.PathLike | None = None) -> dict[str, list[Path]]:
    """Bearing name -> acquisition files in acquisition order.

    Bearings follow the layout's folder order and, within a folder, sorted names. This
    order fixes the bearing indices of folds and of PPO sampling. Files must be numbered
    1..N without gaps; other files, such as FEMTO's ``temp_*.csv``, are ignored.
    """
    spec = layout(dataset)
    pattern = re.compile(spec.file_pattern)
    root = dataset_directory(data_root, dataset, directory=directory)
    result: dict[str, list[Path]] = {}
    for folder in spec.folders:
        condition = root / folder
        if not condition.is_dir():
            raise FileNotFoundError(f"missing {dataset} folder {condition}; run the layout check")
        for bearing in sorted((path for path in condition.iterdir() if path.is_dir()), key=lambda p: p.name):
            numbered = _numbered(bearing, pattern)
            if not numbered:
                raise ValueError(f"{bearing}: no acquisition files matching {spec.file_pattern}")
            if [index for index, _ in numbered] != list(range(1, len(numbered) + 1)):
                raise ValueError(f"{bearing}: acquisition numbering is not 1..N")
            if bearing.name in result:
                raise ValueError(f"duplicate bearing name {bearing.name}")
            result[bearing.name] = [path for _, path in numbered]
    return result


def check_layout(data_root: str | os.PathLike | None, dataset: str, *,
                 directory: str | os.PathLike | None = None) -> dict:
    """Bearings and acquisition counts found, and every missing folder, bearing or file.

    Lists file names only and opens no file. ``missing`` names what the layout needs and
    does not find: the directory, condition folders, the article's bearings, acquisition
    files and numbers. ``problems`` lists repeated numbers and acquisition counts that differ
    from the article's, for example a truncated test set or an incomplete download. ``notes``
    lists additional bearings. ``ok`` is True when nothing is missing and there is no problem.
    """
    spec = layout(dataset)
    root = dataset_directory(data_root, dataset, directory=directory)
    pattern = re.compile(spec.file_pattern)
    report = {"dataset": dataset, "directory": str(root), "folders": {}, "missing": [], "problems": [], "notes": []}
    if not root.is_dir():
        report["missing"].append(f"directory {root}")
    for folder, expected in zip(spec.folders, spec.bearings):
        condition = root / folder
        found: dict[str, int] = {}
        report["folders"][folder] = found
        if not condition.is_dir():
            if root.is_dir():
                report["missing"].append(f"folder {condition}")
            continue
        for bearing in sorted((p for p in condition.iterdir() if p.is_dir()), key=lambda p: p.name):
            numbered = _numbered(bearing, pattern)
            found[bearing.name] = len(numbered)
            indices = {index for index, _ in numbered}
            if bearing.name not in expected:
                report["notes"].append(f"additional bearing {bearing}")
            if not numbered:
                report["missing"].append(f"{bearing}: no files matching {spec.file_pattern}")
                continue
            gaps = sorted(set(range(1, max(indices) + 1)) - indices)
            if gaps:
                shown = ", ".join(map(str, gaps[:10])) + (", ..." if len(gaps) > 10 else "")
                report["missing"].append(f"{bearing}: acquisitions {shown} of 1..{max(indices)}")
            if len(indices) != len(numbered):
                report["problems"].append(f"{bearing}: repeated acquisition numbers")
            if (bearing.name in expected and spec.expected_acquisitions
                    and len(numbered) != spec.expected_acquisitions[bearing.name]):
                report["problems"].append(f"{bearing}: {len(numbered)} acquisitions, the article used "
                                          f"{spec.expected_acquisitions[bearing.name]}")
        report["missing"] += [f"bearing {condition / name}" for name in expected if name not in found]
    report["ok"] = not report["missing"] and not report["problems"]
    return report


def read_acquisition(path: str | os.PathLike, dataset: str) -> np.ndarray:
    """Horizontal and vertical acceleration of one acquisition file as an (n, 2) array.

    FEMTO files have no header, ``;`` or ``,`` separators and six columns, of which
    columns 4 and 5 are the horizontal and vertical acceleration. XJTU-SY files have the
    header ``Horizontal_vibration_signals,Vertical_vibration_signals`` and two columns.
    """
    import pandas as pd

    layout(dataset)
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline()
    if dataset == "xjtu_sy":
        if not first.startswith("Horizontal_vibration_signals"):
            raise ValueError(f"{path}: expected the XJTU-SY header Horizontal_vibration_signals")
        data = pd.read_csv(path, sep=",", header=0, dtype=np.float64, engine="c").to_numpy()
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"{path}: expected two XJTU-SY columns")
    else:
        separator = ";" if ";" in first else ","
        data = pd.read_csv(path, sep=separator, header=None, dtype=np.float64, engine="c").to_numpy()
        if data.ndim != 2 or data.shape[1] != 6:
            raise ValueError(f"{path}: expected six FEMTO columns")
        data = data[:, 4:6]
    if len(data) == 0 or not np.isfinite(data).all():
        raise ValueError(f"{path}: an acquisition needs finite samples")
    return data


def acquisition_statistics(signal: np.ndarray) -> np.ndarray:
    """RMS, peak absolute amplitude and kurtosis per channel, in ``STATISTICS`` order.

    RMS is uncentered. Kurtosis is the centered fourth moment divided by the squared
    variance, with the variance floored at 1e-12.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0:
        raise ValueError("expected a nonempty (n, 2) acquisition")
    centered = x - x.mean(axis=0)
    variance = (centered ** 2).mean(axis=0)
    rms = np.sqrt((x ** 2).mean(axis=0))
    peak = np.abs(x).max(axis=0)
    kurtosis = (centered ** 4).mean(axis=0) / np.maximum(variance, 1e-12) ** 2
    return np.concatenate([rms, peak, kurtosis])


def _file_statistics(task: tuple[Path, str]) -> tuple[np.ndarray, int]:
    signal = read_acquisition(*task)
    return acquisition_statistics(signal), len(signal)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def prepare(data_root: str | os.PathLike | None, dataset: str, output: str | os.PathLike, *,
            jobs: int = 1, directory: str | os.PathLike | None = None) -> dict[str, np.ndarray]:
    """Compute the per-acquisition statistics of every bearing and save a compressed cache.

    The files are read from ``directory`` or from the dataset's folder below ``data_root``.
    The ``.npz`` cache holds one (N, 6) float64 array per bearing, in bearing order. All
    acquisitions of a bearing must have the same length. An existing file is never
    replaced. ``jobs`` processes read the files in parallel.
    """
    destination = Path(output)
    if destination.suffix != ".npz":
        raise ValueError("the statistics cache must be a .npz file")
    if destination.exists():
        raise FileExistsError(f"refusing to replace the statistics cache {destination}")
    jobs = _positive_integer(jobs, "jobs")
    files = acquisition_files(data_root, dataset, directory=directory)
    statistics: dict[str, np.ndarray] = {}
    pool = ProcessPoolExecutor(jobs) if jobs > 1 else None
    try:
        for name, paths in files.items():
            tasks = [(path, dataset) for path in paths]
            rows = list(pool.map(_file_statistics, tasks, chunksize=16) if pool else map(_file_statistics, tasks))
            lengths = {length for _, length in rows}
            if len(lengths) != 1:
                raise ValueError(f"{name}: acquisitions differ in length {sorted(lengths)}")
            statistics[name] = np.vstack([row for row, _ in rows])
    finally:
        if pool is not None:
            pool.shutdown()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **statistics)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return statistics


def load_statistics(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Read a statistics cache written by ``prepare``, in bearing order."""
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    if not result:
        raise ValueError(f"{path}: the statistics cache is empty")
    for name, values in result.items():
        if (values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != len(STATISTICS)
                or len(values) == 0 or not np.isfinite(values).all()):
            raise ValueError(f"{path}: {name} must be a finite nonempty (N, 6) float64 array")
    return result


def to_minutes(acquisitions: np.ndarray, per_minute: int) -> np.ndarray:
    """Aggregate acquisitions to minutes: quadratic-mean RMS, maximum peak, mean kurtosis.

    Minute m holds acquisitions per_minute*(m-1)+1 .. per_minute*m. A final partial
    minute is kept, because the bearing operated during it.
    """
    per_minute = _positive_integer(per_minute, "per_minute")
    values = np.asarray(acquisitions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(STATISTICS) or len(values) == 0:
        raise ValueError("expected nonempty (N, 6) acquisition statistics")
    count = -(-len(values) // per_minute)
    minutes = np.empty((count, len(STATISTICS)))
    for minute in range(count):
        block = values[per_minute * minute:per_minute * (minute + 1)]
        minutes[minute, 0:2] = np.sqrt((block[:, 0:2] ** 2).mean(axis=0))
        minutes[minute, 2:4] = block[:, 2:4].max(axis=0)
        minutes[minute, 4:6] = block[:, 4:6].mean(axis=0)
    return minutes


@dataclass(frozen=True, eq=False)
class BearingHistory:
    """Minute statistics of one recorded bearing; the last row is the last productive minute T."""
    dataset: str
    name: str
    minutes: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.minutes, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(STATISTICS) or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"{self.name}: minute statistics must be a finite nonempty (T, 6) array")
        object.__setattr__(self, "minutes", values)

    @property
    def life(self) -> int:
        """Last productive minute T, the number of recorded minutes."""
        return int(self.minutes.shape[0])


def histories(statistics: Mapping[str, np.ndarray], dataset: str) -> list[BearingHistory]:
    """Minute histories of all bearings of a statistics cache, in its bearing order."""
    per_minute = layout(dataset).acquisitions_per_minute
    return [BearingHistory(dataset, name, to_minutes(values, per_minute)) for name, values in statistics.items()]
