# Third-party notices

MaintenanceArena contains no vendored third-party source code, and its wheel and source
distribution include no external dataset.

The packages below are dependencies that pip installs separately. Each one is distributed
under its own license. The license entries are the ones declared in the package metadata
of the versions pinned in `constraints/`. Their own dependencies are not listed here and
carry their own licenses.

## Runtime dependencies

| Package | Pinned version | Declared license |
|---|---|---|
| NumPy | 2.4.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| Gymnasium | 1.2.3 | MIT |
| PyYAML | 6.0.3 | MIT |

## Optional dependencies

| Extra | Package | Pinned version | Declared license |
|---|---|---|---|
| `rl` | PyTorch (`torch`, CPU build) | 2.8.0+cpu | BSD-3-Clause |
| `rl` | Stable-Baselines3 | 2.8.0 | MIT |
| `rl` | sb3-contrib | 2.8.0 | MIT |
| `plots` | Matplotlib | 3.10.6 | Matplotlib license (PSF-based) |
| `bearings` | pandas | 2.3.0 | BSD-3-Clause |
| `bearings` | scikit-learn | 1.8.0 | BSD-3-Clause |
| `bearings` | SciPy (a scikit-learn dependency) | 1.15.3 | BSD-3-Clause (BSD license classifier) |
| `bearings` | joblib (a scikit-learn dependency) | 1.5.3 | BSD-3-Clause |
| `bearings` | threadpoolctl (a scikit-learn dependency) | 3.6.0 | BSD-3-Clause |
| `dev` | pytest | 9.0.3 | MIT |
| `dev` | build | 1.5.0 | MIT |
| `dev` | Twine | 6.2.0 | Apache-2.0 |

## Build backend

| Package | Pinned version | Declared license |
|---|---|---|
| setuptools | 80.9.0 | MIT |
| wheel | 0.45.1 | MIT |

## Data

Data supplied separately, such as recorded sensor data, comes with its own terms. The MIT license of this project does not change the rights to third-party data. The FEMTO and XJTU-SY bearing recordings are not part of this repository. [docs/BEARINGS.md](docs/BEARINGS.md) names their providers and the publications to cite.

The saved outcomes in `paper/outcomes/` are decisions of the article's experiments: first orders and last productive steps of synthetic lives and of the replayed bearing histories. They contain no recorded signals.
