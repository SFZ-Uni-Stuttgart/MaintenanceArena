# Security policy

## Supported versions

MaintenanceArena is a research simulator. Security problems are fixed in the latest version, which is currently 1.0.0.

| Version | Supported |
| --- | --- |
| 1.0.x | Yes |

## Reporting a vulnerability

Please do not report security problems in a public issue. E-mail a private report to the maintainer, Giuseppe Mannone (giuseppe.mannone@ima.uni-stuttgart.de).

Include the package version, the platform and Python version, the affected command or function, the impact and the smallest reproduction you can share. Do not attach private or restricted data. Reports stay private until a fix is available.

## Trust boundary of saved models

Treat saved models and training checkpoints as code. Load only files that you created or that come from a source you trust.

- Core and gearbox checkpoints (`.pt` files) are read with `torch.load(..., weights_only=True)`.
- Resource models use the Stable-Baselines3 formats. `model.zip` and the paired `normalizer.pkl` are unpickled, and a native resume state (`training_state.pt`) is read with `torch.load(..., weights_only=False)`. A crafted file of these kinds can run code when it is loaded.
- Checkpoint manifests bind every file to a SHA-256 hash. The hash detects accidental changes but does not prove where a manifest and its files come from.

The package downloads no models or data, and its commands write results only into a new `--output` directory.
