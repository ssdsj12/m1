# T400.8b Task 3 bounded prioritized motion distribution

## Purpose

Implement the C0 high-level velocity-bound intersection and Panda-first prioritized distribution with M1 planar activation, null-space singularity avoidance, and ordered degradation.

## Stage

T400.8b / C0 deterministic Teacher / Task 3

## Related todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [C0 implementation plan](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)

## Input conditions

- Baseline ref: `65cbc37`
- Pure PyTorch environment: `/home/xk/miniconda3/envs/go2/bin/python`
- Coordination order: M1 planar x/y/yaw followed by Panda joints 1–7.

## Procedure and evidence

1. RED: collection failed because `constraints.py` did not exist.
2. Added the exact position/velocity/acceleration bound intersection.
3. Added deterministic arm-first active-set redistribution, base activation, P1 null-space P3, and `psi` before `phi` scaling.
4. First implementation run had one expected-value failure; investigation showed the test had ignored the tighter ±2 acceleration bound. Both hand-calculated expected bounds were corrected without changing the formula.
5. Review RED showed a moving base could be used while `base_active=False` when acceleration bounds excluded zero; the explicit activation gate fixed this state inconsistency.
6. Focused motion-distribution suite: `19 passed`.
7. Contracts + kinematics + distribution regression: `56 passed`.
8. `py_compile` and `git diff --check`: exit `0`.

## Result

- Arm-only execution remains the default when full-rank and feasible.
- Saturated arm freedom is frozen and missing motion redistributes to M1.
- Rank loss, `sigma_min < 0.1`, infeasible arm bounds, or an unreachable zero base velocity activates M1.
- P3 motion is projected into the active P1 null space.
- Null-space scale `psi` reaches zero before end-effector scale `phi` is reduced.
- Batch outputs remain finite and inside the intersected velocity bounds.
- No QP, Isaac environment, play, or training code changed.

## Follow-up

Proceed to Task 4 project-owned float64 reference QP backend.

## Git refs

- Baseline Ref: `65cbc37`
- Candidate Ref: `2a8cd9b`
- Key Files:
  - [constraints](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py)
  - [motion distribution](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py)
  - [tests](../../Go2Pvcnn/tests/test_m1_panda_motion_distribution.py)
