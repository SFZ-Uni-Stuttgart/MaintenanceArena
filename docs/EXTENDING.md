# Extending MaintenanceArena

Start from a scientific question and name what changes: the dynamics, the available information, the interventions, the objective, the resources or the policy. A configuration override is enough only when the implementation already supports the change.

## Configuration changes and recorded identities

The CLI reads YAML or JSON files through `--config`, and their sections `env_config`, `training` and `policy_defaults` merge recursively into the selected profile. A policy override can change the supported settings of an existing policy but cannot add a policy class. Profile resolution rejects unknown policy settings, and the environment and training implementations check their own fields.

This resource override changes the illustrative part cost and the stock target of the `rolling` replenishment rule:

```yaml
env_config:
  part_cost: 900.0
policy_defaults:
  rolling:
    base_stock: 2
```

Run it with `--profile paper-resource-v1 --config resource_change.yaml`. The run record keeps the original profile as `base_profile` and the full resolved configuration. The resolved configuration gets a `configuration_sha256` and an identifier of the form `paper-resource-v1+<12 hash characters>`, while the published profile and its results stay unchanged.

In Python, pass the same override mapping to `maintenance_arena.make`:

```python
import maintenance_arena

env = maintenance_arena.make(
    "paper-resource-v1",
    config={"env_config": {"part_cost": 900.0},
            "policy_defaults": {"rolling": {"base_stock": 2}}},
)
profile = env.unwrapped.arena_profile
print(profile["id"], profile["configuration_sha256"])
env.close()
```

`gymnasium.make("maintenanceArena/Resource-v0", config={...})` passes only the environment configuration, without the conventional-policy defaults of a profile.

CLI resource runs (`simulate`, `evaluate`) use `--calendar-steps` as their recorded accounting horizon. They can therefore run 3,600 uninterrupted hours, although the profile's default horizon is 720 hours. Direct construction in Python uses the supplied `horizon`, and training uses its own.

## Supported parameters and contract boundaries

| Area | Existing capability | Change that needs implementation and a new contract |
| --- | --- | --- |
| Core | Continuous or discrete source, degradation law (`family`), fixed lifetime (`eol_operating_age`), preparation time 6, 12 or 24, recovery contract, observation route, sensor preset, economic parameters and optional cutoff. | Another action, failure threshold, downtime length (fixed at 6), preparation time or event sequence. |
| Gearbox | Optional horizon. `gamma` and `reward_scale` accept only the published 0.999 and 1,000, and other keys are rejected. | Other damage increments or capacity, operating revenues, lead times, action set, filter boundary or discount. |
| Resource | Fields of `ResourceMaintenanceConfig`, including fleet, stock and crew sizes, shifts, lead ranges, workloads and itemized costs. | Operating or load control, shared physical degradation, new supply rules or a different renewal order. |
| Resource lifetime | The resource fields, public recorded age, separately drawn lifetimes and shapes, and the supported age, condition and procurement policies. | Sensor-derived age, a new stochastic process model, or loading an old 55-input checkpoint into the 58-input task. |

A new fleet size needs initial-age and contribution arrays of matching length. It changes the observation and the action space, and a new inventory capacity changes the observation, so old policy weights and normalizers no longer fit. Training contexts must keep one observation order and action space across resets.

An override changes only what it names. A changed lifetime range in the resource-lifetime task keeps the selected policy settings, and noisy core observations do not fit a new reconstructor, actor or residual treatment. The thresholds of the core rules `age`, `health` and `projection` can be selected again with `calibrate`, and `benchmark` does this by default when `--config` changes the environment of a core profile ([Workflows](WORKFLOWS.md#calibrate-the-core-rules)).

The exact gearbox kernel and its filter belong together. A physical change of the gearbox means rebuilding the kernel, checking its primitive accounting and deriving the new intervention and filter boundary. The published guarantee from `q >= 85` does not carry over to changed mechanics.

## Implement a new environment or observation method

`maintenance_arena.__init__.ENVIRONMENTS` holds the registry and `data/profiles.json` the profiles. `policies.environment_module` dispatches the conventional policies, and the CLI picks the learning implementation by task family. There is no plugin loader, so a new task needs its environment, a registration, a distinct profile, matching policy dispatch and the learning and evaluation support it actually offers.

Existing identities stay fixed. Saved core and gearbox checkpoints bind the hashes of the validated environment configuration, the PPO recipe and the observation contract. A new key in the core or gearbox default configuration, in `recipe_config()` or in the observation contract would make every saved checkpoint unloadable, and an edit of a v1 profile would change its `configuration_sha256`. `tests/test_identity_freeze.py` pins these identities ([Contributing](../CONTRIBUTING.md#identity-freeze)).

Put a new feature into a wrapper, a new environment class or a new profile, and keep it out of the validated configuration while it is unused. The wrappers in `maintenance_arena.wrappers` show the pattern. `ZeroDeadline` changes what an agent sees and `CalendarBudget` when its episode ends, but neither changes the wrapped environment. A benchmark track likewise fixes scenarios on an unchanged profile ([Benchmark tracks and protocol](BENCHMARK.md)).

Keep six distinctions when you extend these components:

- Keep policy observations apart from diagnostics. Declare the actor's fields and their order to match the structured public state. Future degradation, sampled lifetimes, realized delivery times and hindsight labels belong in `diagnostics()` or analysis output, and a `snapshot()` must not reach a policy.
- Keep physical state apart from ownership. Return copies rather than writable views of internal assets or queues, keep shared exact-kernel arrays read-only and close the environments, vector carriers and normalizers you open. Loading a policy must not alter another caller's environment.
- Keep calendar time apart from decisions. Return the actual `duration_calendar_steps`, `gamma_pow_duration`, raw cash sum, events and ledger of a macro action, and discount every primitive step rather than once per macro. Productive preparation, exchange and downtime stay separate events.
- Keep renewal apart from the rollout cutoff. Cycle termination in the core, continuing renewal in the gearbox and resource tasks and accounting truncation call for different bootstrapping. Keep the sampled successor instead of redrawing it at a renewal or at the end of a collector update.
- Keep available resources apart from future supplies. A resource job depends on the parts and crew available at its start, not on pending orders, and masks must agree with the simulator.
- Keep randomness apart from hidden information. The component, context, delivery, sensor, lifetime and shape streams stay separate, and restoring a native snapshot preserves later draws and automatic resets. Policies sharing exogenous stream keys can still produce different event histories, so paired seeds do not imply identical calendar paths.

Record the information routes and units of each new method. A recurrent policy, an execution filter and an online learning mechanism are different methods and need separately identifiable execution records. A filter's corrections must stay visible in both the proposed and the executed actions.

## Add a policy, learner or checkpoint format

A conventional policy uses the existing public state and the feasible-action interface. Keep strong reference policies intact and name the information each one uses. An age policy, for example, must not read true remaining life or health.

The published profiles fix the packaged actor shapes and recipes. A new architecture or observation order needs loader support and its own compatibility identity, not a permissive partial weight load or padded or truncated inputs. Resource inference loads the selected model together with its normalizer.

Native continuation checkpoints are written at completed update boundaries. They hold the learner, environment, collector, normalization and random-generator state, so add every new stateful component to both snapshot and restoration. `maintenance_arena.learning.learners` is a small example of a learner on scenario banks: a fixed recipe, a trainer with milestones, checkpoints and exact resumption, and a greedy policy for the replay. `--initialize` starts a new run from compatible saved weights, and `--resume` continues one native run. Keep this distinction in metadata, seed handling and exposure counters.

Test a changed mechanism with a focused synthetic check, such as a known event order and cash balance, causal input access, a fixed seeded sequence or split against uninterrupted continuation. Compare the unchanged parts with the previous implementation, for example with the golden runs of `tests/test_identity_freeze.py`. A short training run checks execution and saving, not reliability or transfer.

## Evidence and publication ownership

A new scientific comparison records its hypothesis, inputs, strong baselines, independent units, data splits, aggregation and resource budget, and it reports every independent unit and negative outcome. Do not tune the illustrative costs to reverse a ranking, and do not count already inspected data as untouched confirmation.

The installed package must work without a source checkout or data directory. Optional evidence such as stored results or trained models belongs in its own manifest-bound bundle that preserves the original identities and hashes. A portable or redacted copy gets its own location and transformation record and does not replace the earlier record. Publishing a new task, profile, dataset adapter or evidence bundle requires correct attribution and verified software and data rights.
