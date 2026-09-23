# Task 4 Report: Expose O6 Target Object Selection

## Status

Implemented and locally verified.

Feature commit: `feat: expose O6 target object selection` plus the follow-up
P1 fix commit recorded below.
Baseline: `3d40cfc` (`fix: harden object catalog scene integration`).

## Scope implemented

- Added optional play CLI `--object-id OBJECT_ID`; omission leaves `args.object_id`
  as `None` and preserves the legacy `/Box` play path.
- Added wrapper state:
  - `target_object_id: str | None`
  - `obstacle_object_ids: tuple[str, ...]`
- Added pure target-ID resolution with deterministic enabled-instance ordering,
  explicit-target precedence, disabled-instance filtering, and unknown-ID rejection.
  The legacy `box` target remains available for backward compatibility.
- Extended the snapshot/task-goal boundary with pose-only `ObjectState` obstacle
  entries and a `target_object` alias for the existing `BimanualSnapshot.box`.
- Extended `ObjectMpcInput` with `target_object_pose_b` and
  `obstacle_object_poses_b`, while normalizing the existing
  `target_box_pose_b` field as a backward-compatible alias.
- Forwarded selected target and obstacle poses through `BimanualRuntime`; the
  object MPC consumes only CPU float64 pose tensors and does not load catalog
  geometry or perform network access.
- Preserved the existing prior validation/startup order: malformed prior
  configuration still fails before touching `env.unwrapped` or the Isaac scene.
- Fixed the Play entrypoint P1: `--object-id` now selects the default local
  catalog automatically, materializes deterministic `object_class_000`
  instances, validates the selected ID and resolved USD assets before
  `AppLauncher`/`gym.make`, and passes `object_catalog` plus
  `object_instances` into `M1DualPandaO6BimanualEnvCfg`.
- Added optional `--object-catalog` and `--object-assets-root` overrides for
  explicitly prepared local catalogs. Omitting `--object-id` and
  `--object-catalog` still constructs the legacy Box config without loading
  the catalog.
- Scoped catalog preflight to the instances loaded by the selected Play scene:
  an explicit target loads and validates only its `object_class_000` instance,
  while an omitted target or explicit `box` keeps the full configured
  instance set. This prevents an unselected class with a missing USD asset
  from blocking single-target Play.

## TDD evidence

Initial RED, from the repository root with the pinned `go2` environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD/Go2Pvcnn \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_object_target_contract.py
```

Result:

```text
ImportError: cannot import name 'ObjectState' from
'go2_pvcnn.control.m1_bimanual_coordination'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.95s
```

Focused GREEN:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_object_target_contract.py
```

Result: `6 passed in 0.78s`, exit `0`.

## Regression evidence

The required object/catalog and existing O6 contract set was run with:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_object_target_contract.py \
  tests/test_m1_object_catalog.py \
  tests/test_m1_object_scene.py \
  tests/test_m1_rialto_object_assets.py \
  tests/test_m1_bimanual_object_mpc.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_dual_panda_o6_contracts.py \
  tests/test_m1_dual_panda_o6_entrypoints_static.py \
  tests/test_m1_dual_panda_o6_env_static.py \
  tests/test_m1_dual_panda_o6_verification.py
```

Result: `113 passed in 13.07s`, exit `0`.

Additional focused passes:

- Object/catalog/scene/target tests: `25 passed in 0.91s`.
- Object MPC/runtime/O6 contracts: `42 passed in 1.90s`.
- Existing O6 verification after startup-order fix: `9 passed in 0.89s`.
- Target/runtime/object-MPC recheck: `35 passed in 1.73s`.
- Play/catalog target follow-up: `12 passed in 0.78s`.
- Object/catalog/scene/O6 entrypoint follow-up: `77 passed in 15.91s`.

Selected-asset regression TDD evidence:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD/Go2Pvcnn \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_object_target_contract.py \
  -k 'prepare_object_scene or default_catalog'
```

Initial RED: `3 failed, 1 passed, 10 deselected in 0.81s`.
The book fixture reached the missing cup asset, and the existing two-class
fixture still returned both instances.

Focused GREEN:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD/Go2Pvcnn \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_object_target_contract.py
```

Result: `14 passed in 0.84s`, exit `0`.

The selected target and O6 regression set was rerun with:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_object_target_contract.py \
  tests/test_m1_object_scene.py \
  tests/test_m1_bimanual_object_mpc.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_dual_panda_o6_contracts.py \
  tests/test_m1_dual_panda_o6_entrypoints_static.py \
  tests/test_m1_dual_panda_o6_env_static.py \
  tests/test_m1_dual_panda_o6_verification.py
```

Result: `98 passed in 17.05s`, exit `0`.

Static checks:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  go2_pvcnn/control/m1_bimanual_coordination/contracts.py \
  go2_pvcnn/control/m1_bimanual_coordination/task_goal.py \
  go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py \
  go2_pvcnn/control/m1_bimanual_coordination/runtime.py \
  go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  scripts/m1_dual_panda_o6_bimanual_play.py \
  tests/test_m1_object_target_contract.py
```

Result: exit `0`.

Scoped `git diff --check` for the Task 4 files is clean. Repository-wide
`git diff --check` remains affected by an unrelated pre-existing blank line at
`.superpowers/sdd/task-6-brief.md:61`.

## Follow-up commit and concerns

Follow-up commit: `2c8d52a` (`fix: wire Play object-id to catalog-backed
config`) plus the selected-asset preflight fix recorded in the final commit
history.

- No Isaac Sim/GPU0 startup or physical manipulation run was performed.
- The default catalog has local bottle/book source USD files in this worktree,
  but the cup converted USD is absent. A default `--object-id book_000`
  preflight now succeeds, while `--object-id cup_000` fails before simulation
  with `AssetPreparationRequiredError` and the explicit offline preparation
  command. A prepared partial catalog fixture verifies the same selected-class
  behavior without network access.
- The pre-existing catalog test
  `test_committed_catalog_missing_assets_require_explicit_preparation` still
  expects bottle to be the first missing class. Because this worktree contains
  untracked local bottle source assets, the full unfiltered catalog test
  currently reports `1 failed, 35 passed in 15.87s` with cup as the first
  missing class; no catalog test or asset file was changed by this fix.
- Existing hand contact sensor filters remain Box-specific in the environment
  configuration. Target pose/obstacle goal wiring is covered, but target-specific
  contact sensor initialization and contact-rich behavior belong to a later smoke
  task.
- Catalog USD geometry, hashes, and network-free preparation remain covered by
  the prior catalog tasks; this task does not download or resolve assets at
  runtime.
