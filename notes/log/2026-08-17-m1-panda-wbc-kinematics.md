# T400.8b Task 2 coordinated kinematics

## Purpose

Implement and verify the pure-PyTorch planar-base plus Panda spatial Jacobian, damped pseudoinverse, and singularity diagnostics used by the C0 motion distributor.

## Stage

T400.8b / C0 deterministic Teacher / Task 2

## Related todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [C0 implementation plan](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)

## Input conditions

- Baseline ref: `f008c99`
- Pure PyTorch environment: `/home/xk/miniconda3/envs/go2/bin/python`
- Spatial row convention: `[linear xyz, angular xyz]`

## Procedure and evidence

1. Initial RED: collection failed because `kinematics.py` did not exist.
2. Added analytic planar x/y/yaw spatial columns, coordinated 6×10 concatenation, SVD damped pseudoinverse, and Panda singularity metrics.
3. Initial focused GREEN: `18 passed`.
4. Review found zero damping on a rank-deficient matrix produced `0/0`; the new test failed with an all-NaN pseudoinverse.
5. Added SVD tolerance semantics for the zero-damping Moore–Penrose case.
6. Final contracts + kinematics regression: `37 passed`.
7. `py_compile` and `git diff --check`: exit `0`.

## Result

- Base Jacobian linear columns match central finite differences.
- Coordinated Jacobian shape and order are fixed at `[..., 6, 10]` with base columns first.
- Batched float32/float64 operations preserve dtype and device.
- Minimum singular value and manipulability preserve batch dimensions.
- Rank-deficient damped and zero-damping cases stay finite.
- No Isaac, QP, environment, play, or training code changed.

## Follow-up

Proceed to Task 3 velocity-bound intersection and prioritized motion distribution.

## Git refs

- Baseline Ref: `f008c99`
- Candidate Ref: `11c34a1`
- Key Files:
  - [kinematics](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/kinematics.py)
  - [kinematics tests](../../Go2Pvcnn/tests/test_m1_panda_wbc_kinematics.py)
