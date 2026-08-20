# M1 Panda Student S1 Task 6: Batched Teacher Isolation

## Purpose

Complete T400.9 Task 6 by extracting the live PhysX-to-Teacher adapter and
providing one isolated rolling Teacher per Student collection environment.

## Stage

M1 + Panda Student S1 runtime foundation. Related todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md).

## Changes

- Added `runtime_adapter.py` with explicit `env_index` selection for robot,
  contact, Jacobian, bias-force and state tensors.
- Added `BatchedRollingTeacherBank` with one adapter per Teacher, per-env reset,
  matched mission dispatch, and shared mutable-controller-state rejection.
- Kept C0 compatibility exports in `m1_panda_wbc_play.py` and routed the C1a
  rolling adapter through the shared implementation.

## Verification

Command:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_batched_rolling_teacher.py tests/test_m1_panda_wbc_play_static.py tests/test_m1_panda_wbc_roll_play_static.py tests/test_m1_panda_rolling_teacher.py tests/test_m1_panda_rolling_wbc.py tests/test_m1_panda_wbc_safety.py
```

Result: `70 passed`.

Additional checks: `py_compile` passed for all changed Python modules and
`git diff --check` exited `0`.

## Boundary

No Student environment, collection CLI, or training run was started in this
task. Tasks 7-12 remain open.

## Git Refs

- Baseline ref: `15780da`
- Candidate ref: current Task 6 commit
- Key files: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py`,
  `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/batched_rolling_teacher.py`,
  `Go2Pvcnn/scripts/m1_panda_wbc_play.py`,
  `Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py`
