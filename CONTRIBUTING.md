# Contributing to MaintenanceArena

Improvements to the simulation mechanics, the workflows, the documentation and reproducibility are welcome. Before you change a decision problem, read [the environment contracts](docs/ENVIRONMENTS.md) and [the extension guidance](docs/EXTENDING.md). Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Report security problems privately, as [SECURITY.md](SECURITY.md) describes.

## Development environment

Create a separate CPython 3.12 environment:

```shell
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux or `.venv\Scripts\Activate.ps1` in PowerShell. If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process Bypass` or call `.venv\Scripts\python` directly. Then install the base package and the build and test tools with your platform's constraints, and run the base checks:

```shell
python -m pip install -c constraints/python312-windows-cpu.txt pip setuptools wheel build pytest twine
python -m pip install --no-build-isolation -c constraints/python312-windows-cpu.txt -e .
python -m pytest tests/test_public_workflows.py tests/test_release_engineering.py -q
```

On Linux, use `constraints/python312-linux-cpu.txt`. The first command installs the pinned setuptools that the `--no-build-isolation` install needs. The base checks run without PyTorch or Matplotlib, so a base-only installation stays usable. For the optional features, install the CPU build of PyTorch and then the extras:

```shell
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0+cpu"
python -m pip install --no-build-isolation -c constraints/python312-windows-cpu.txt -e ".[rl,plots,bearings]"
python -m pip check
python -m pytest tests -q
```

The Windows and Linux CI workflow runs the same stages. Training tests run on the CPU with one PyTorch thread and small budgets, so they check behavior, not convergence. The tests need no dataset or trained model.

## Changes and evidence

Describe the affected behavior, the reason and the smallest useful validation. For an event or accounting change, add a short synthetic trajectory that shows the difference. Keep the observation order, random-generator ownership, event timing, masks and cost definitions unless the change defines a new task or profile identity. A changed configuration or implementation does not inherit earlier published validation.

Keep the true lifetime, future degradation, latent identities and evaluation diagnostics out of policy inputs, and state what each baseline receives. Report all independent units and negative results. Keep the reproduction of stored results, frozen-policy evaluation and fresh training apart. Do not pick favorable seeds after seeing the results.

Do not replace a retained result, checkpoint or record while fixing software. A new run needs a new output directory, and temporary work belongs in an ignored directory such as `outputs/`. Leave run outputs, datasets, caches, credentials and model binaries out of contributions.

Never change the meaning of an existing output field. Add a new field and bump the schema identifier instead. Evaluation schema v3 did this for the pooled profit, failures per 1,000 lives and the order-margin fields (see [Workflows](docs/WORKFLOWS.md#compare-policies-on-paired-synthetic-units)).

`tests/test_benchmark.py`, `tests/test_calibration.py`, `tests/test_tracks.py` and `tests/test_wrappers.py` cover the benchmark layer: `benchmark`, `calibrate`, `tracks`, the oracle and the wrappers. The oracle stays on the evaluator side and must not appear in `profiles.json`, an observation, `info` or the agent call. After a change that could alter a score, run `MAINTENANCE_ARENA_FULL_TRACK=1 python -m pytest tests/test_tracks.py -k documented`, which takes a little over a minute. It recomputes the default track and compares it with the table in `docs/BENCHMARK.md` and its excerpt in `README.md`.

`tests/test_synthetic_experiments.py` and `tests/test_bearings_experiments.py` run every experiment runner of the article end to end on small budgets and synthetic data. The module tests add to them (`test_banks.py`, `test_replay.py`, `test_metrics.py`, `test_bearings*.py` and others). `tests/test_paper_tables.py` rebuilds the article's tables from `paper/outcomes` and compares them byte for byte with `paper/expected`. Tests that need the recorded bearing data run only when `MAINTENANCE_ARENA_DATA` names a data root ([docs/BEARINGS.md](docs/BEARINGS.md#data)). A change to the article's banks, training or replay must keep the runners' outputs identical or come with a new configuration.

Document every change of public behavior and add a `CHANGELOG.md` entry. [Releasing](docs/RELEASING.md) lists every place that names the version, and `tests/test_release_engineering.py` checks that they agree.

## Identity freeze

`tests/test_identity_freeze.py` pins the identity of the v1 tasks. It checks:

- the `configuration_sha256` of the five v1 profiles
- the hashes of the default environment configurations
- the environment configuration, recipe and observation hashes that saved checkpoints bind
- the bytes of the three CRLF files
- the serialization of `provenance.json`
- short golden evaluation runs of every built-in conventional policy

A failure means that a change altered a task identity, a checkpoint binding or the simulation. Do not update the constants to make the test pass. Put new behavior in a new profile, environment class or wrapper with its own identifier. Keep it out of the validated configuration when it is unused. Do not add keys to the core or gearbox default configuration, `recipe_config()` or the observation contract, because existing checkpoints bind their hashes.

## Source files, line endings and the export

Text files use UTF-8 and LF line endings. Three files keep recorded CRLF bytes, because their SHA-256 values belong to the task and implementation identity: `src/maintenance_arena/data/profiles.json`, `src/maintenance_arena/gearbox_filter.py` and `src/maintenance_arena/gearbox_model.py`. `.gitattributes` marks them `-text`, so Git leaves their bytes alone. A release-engineering test fails if any other file contains a carriage return. Edit the three files only as part of a deliberate identity change.

After any edit under `src/maintenance_arena`, refresh the inventory with `python tools/record_implementation.py --refresh-current --project .`. The release checker rejects a stale one.

`src/maintenance_arena/data/benchmark_tracks.json` stores the seed lists of the tracks. `python tools/make_benchmark_tracks.py --check` derives the seed lists and profile hashes again from each track's definition and compares them with the file, as the portable suite does. A published track never changes, so a different scenario set or reference list is a new track with a new name. To add one, append its definition fields and run `python tools/make_benchmark_tracks.py --write`.

One allowlist in `tools/export_project.py` defines the source ZIP, the source distribution and the wheel:

- `METADATA` lists the top-level files, and `REQUIRED` those that must exist.
- `TREE_SUFFIXES` lists the exported directories and their file suffixes.
- `EXCLUDED_DIRECTORIES` excludes caches, run outputs, models and data at any depth.

`MANIFEST.in` must select the same files for the source distribution. `tools/check_release.py` rejects any difference between the built source distribution and the export, and `tests/test_release_engineering.py` applies `MANIFEST.in` the way setuptools does to compare the two. The `package-data` patterns in `pyproject.toml` must match the runtime suffixes of `src/maintenance_arena`.

A new file with an allowed suffix in an existing tree is picked up automatically. A new top-level file needs an entry in `METADATA` and in an `include` line of `MANIFEST.in`. A new directory or suffix needs entries in `TREE_SUFFIXES` and `MANIFEST.in`, and for runtime files in `package-data`. [Releasing](docs/RELEASING.md) describes the release procedure.

## Reporting and attribution

A bug report needs the package version, the platform, the resolved profile and configuration, the seed, the command, the exception and a small synthetic reproduction. The issue and pull-request templates in `.github` list these items. Do not attach private or restricted data.

Contribute only code and documentation that you may license under the project's [MIT license](LICENSE), and keep the required notices of [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Credit must reflect actual, verified work.
