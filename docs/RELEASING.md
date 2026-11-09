# Releasing MaintenanceArena

A release consists of a Git tag, a wheel, a source distribution and, optionally, a standalone source ZIP. No step below uploads anything. Pushing a tag, uploading to a package index and archiving are separate actions of the maintainers.

## 1. Prepare the source

Work in the pinned development environment of [Contributing](../CONTRIBUTING.md). After the last edit under `src/maintenance_arena`, refresh the implementation inventory:

```shell
python tools/record_implementation.py --refresh-current --project .
```

This rewrites only `current_implementation` in `provenance.json`. The release checker rejects a missing or stale inventory, for example after a change of the profile data. If you changed the benchmark tracks, also run `python tools/make_benchmark_tracks.py --check`, which compares the packaged file byte for byte with its rebuild.

`tests/test_identity_freeze.py` pins the v1 profiles in `src/maintenance_arena/data/profiles.json` and the three CRLF files listed in `.gitattributes`, as [Identity freeze](../CONTRIBUTING.md#identity-freeze) describes. A behavior change needs a new profile ID, never an edited v1 profile.

Set the new version in `pyproject.toml`, `src/maintenance_arena/__init__.py` and `CITATION.cff`. Add a `CHANGELOG.md` heading for it. Update the README status line, the README heading "Scope of ... and roadmap", the versions in `SECURITY.md` and the artifact names below. `tests/test_release_engineering.py` checks that they agree, and the refresh above records the version in `provenance.json`. Any edit to a `.py` file under `src/maintenance_arena` changes `implementation_sha256` in new run records, even without a change in behavior. The changelog should mention it. Then run the tests:

```shell
python -m pytest tests -q
```

After a change that could alter a score, also recompute the documented default track, which takes a little over a minute. In bash:

```bash
MAINTENANCE_ARENA_FULL_TRACK=1 python -m pytest tests/test_tracks.py -k documented
```

In PowerShell, first set `$env:MAINTENANCE_ARENA_FULL_TRACK = "1"`, then run the same `pytest` command.

The test suite also rebuilds the article's tables from `paper/outcomes`, as `python paper/rebuild_tables.py --check` does. After a change to the scenario banks, the training or the replay, rerun the affected runners in `experiments/` with their quick-check options. Their outputs must stay unchanged.

## 2. Build and check the distributions

Build both distributions in an isolated environment:

```shell
python -m build
```

The isolated build installs the backend that `pyproject.toml` requires (`setuptools>=77.0.3` and `wheel`) in a temporary environment, independent of the active setuptools. The CI workflow instead installs the pinned versions from `constraints/` and runs `python -m build --no-isolation`, which fails if the active setuptools predates the required license metadata. The wheel is built from the source distribution, which shows that the source distribution is complete. Then check both artifacts:

```shell
python tools/check_release.py --dist dist
python -m twine check --strict dist/maintenance_arena-1.0.0-py3-none-any.whl dist/maintenance_arena-1.0.0.tar.gz
```

The release checker requires a current implementation inventory and compares every runtime byte of the wheel with the source project. It checks the entry point, the version, the README, the license files, the dependency metadata with its extras and the allowlisted files of the source distribution. It rejects extra wheel content, unexpected source files, links and non-portable paths, and its JSON output records the artifact hashes. For another version, use the actual file names. Neither check shows that documentation links resolve on a package index.

Install the pinned dependencies into a fresh virtual environment, then the wheel with `python -m pip install --force-reinstall --no-deps` and its path. Run `maintenance-arena list` and the examples from an unrelated directory without `PYTHONPATH`. The base workflows must work without the optional extras. Test the `rl`, `plots` and `bearings` extras separately.

## 3. Export the source project (optional)

The exporter writes a standalone source ZIP and refuses to overwrite an existing file:

```shell
python tools/export_project.py --project . --output outputs/maintenance_arena-1.0.0-project.zip
```

The ZIP holds the allowlist of `tools/export_project.py`. It contains the top-level files `pyproject.toml`, `README.md`, `CHANGELOG.md`, `CITATION.cff`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `MANIFEST.in`, `provenance.json`, `.gitignore` and `.gitattributes`. It also contains the trees `.github`, `src/maintenance_arena`, `docs`, `tests`, `examples`, `experiments`, `paper`, `tools` and `constraints` with fixed file suffixes. Hidden files, caches, run outputs, models and data directories are skipped. `EXPORT_MANIFEST.json` in the ZIP lists the SHA-256 and size of every member. `MANIFEST.in` selects the same files for the source distribution, and `tests/test_release_engineering.py` checks this. The CI workflow validates an export in isolation on every push and pull request.

The same source bytes give the same ZIP and inventory on the same Python and zlib runtime, whatever the file modification times. Wheels and source distributions carry no such guarantee across packaging backends or operating systems. Hash each artifact and keep the validated dependency versions with the release record. Write every new artifact to a new path and keep earlier ones intact.

## 4. Tag and publish

1. Check the line endings with `git ls-files --eol`. The three `-text` files of `.gitattributes` must show `i/crlf w/crlf`, and no other file may show `crlf` or `mixed`. The CI workflow repeats this check, and `.gitattributes` makes it independent of the local `core.autocrlf` setting.
2. Commit the release state, create an annotated tag such as `git tag -a v1.0.0 -m "MaintenanceArena 1.0.0"`, and push both.
3. The workflow `.github/workflows/maintenance-arena.yml` runs on hosted Windows and Linux runners. Record its result with the release.
4. Before uploading to a package index, replace the README's relative links with absolute URLs, because the index shows the README outside the repository.
5. When the release is archived with a persistent identifier, add the identifier to `CITATION.cff` and to the README's citation section.

The packaging follows the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/), the [setuptools guidance on source distributions](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html) and [Twine's distribution check](https://twine.readthedocs.io/en/stable/#twine-check).
