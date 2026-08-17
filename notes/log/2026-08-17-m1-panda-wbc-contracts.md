# T400.8b Task 1 M1 + Panda WBC contracts

## Purpose

Freeze the C0 controller dimensions, canonical 23-joint effort order, runtime joint-index mapping, and finite tensor boundary before implementing kinematics.

## Stage

T400.8b / C0 deterministic Teacher / Task 1

## Related todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [C0 implementation plan](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)

## Input conditions

- Baseline ref: `6e888ad`
- Pure PyTorch environment: `/home/xk/miniconda3/envs/go2/bin/python`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- Existing graphify working-tree artifacts were left unstaged.

## Procedure and evidence

1. Baseline asset/smoke static tests: `28 passed`.
2. RED: the new contract test failed during collection with `ModuleNotFoundError: No module named 'go2_pvcnn.control'`.
3. Added the pure control package, frozen constants, immutable `WbcJointMap`, and `require_tensor`.
4. Focused GREEN: `18 passed`.
5. Existing asset/smoke regression: `28 passed`.
6. `py_compile` and `git diff --check`: exit `0`.

## Result

- Coordination/generalized/controlled dimensions are fixed at `10/31/23`.
- Controlled order is 12 M1 legs, four M1 wheels, then seven Panda arm joints.
- Runtime articulation order is resolved by exact names; duplicate and missing names fail before control starts.
- Shape, dtype, device, and finite-value validation returns the original tensor without hidden conversion.
- No Isaac environment, QP, WBC, play, or training code changed.

## Follow-up

Proceed to Task 2 coordinated kinematics and singularity diagnostics using the frozen contracts.

## Git refs

- Baseline Ref: `6e888ad`
- Candidate Ref: `ac5c9fc`
- Key Files:
  - [contracts](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/contracts.py)
  - [contract tests](../../Go2Pvcnn/tests/test_m1_panda_wbc_contracts.py)
