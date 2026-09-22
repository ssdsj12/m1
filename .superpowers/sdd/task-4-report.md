# Task 4 Report: Expose O6 Target Object Selection

## Status

Implemented and locally verified.

Feature commit: `feat: expose O6 target object selection`.
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

## Concerns and unverified boundaries

- No Isaac Sim/GPU0 startup or physical manipulation run was performed.
- The default play script still constructs the default environment config, so a
  non-legacy `--object-id` requires a caller-provided catalog-backed environment
  configuration with matching `object_instances`; omitted `--object-id` remains
  the Box default.
- Existing hand contact sensor filters remain Box-specific in the environment
  configuration. Target pose/obstacle goal wiring is covered, but target-specific
  contact sensor initialization and contact-rich behavior belong to a later smoke
  task.
- Catalog USD geometry, hashes, and network-free preparation remain covered by
  the prior catalog tasks; this task does not download or resolve assets at
  runtime.
