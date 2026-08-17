# T400.8b Task 7 deterministic WBC Teacher orchestration

## Purpose

Compose the seeded six-dimensional target, 50 Hz motion distribution, 200 Hz standing WBC, safety supervisor, and impedance boundary into one Isaac-independent C0 Teacher.

## Stage

T400.8b / C0 deterministic Teacher / Task 7

## Related todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [C0 implementation plan](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)

## Input conditions

- Baseline ref: `ab55314`
- Pure PyTorch environment: `/home/xk/miniconda3/envs/go2/bin/python`
- Isaac data enters only through immutable `TeacherState`/`StandingWbcInput` tensor snapshots.

## Procedure and evidence

1. RED: collection failed because `teacher.py` did not exist.
2. Added a seeded sum-of-sinusoids pose target with bounded 0.05–0.25 Hz components, ≤0.08 m position amplitude, ≤0.15 rad orientation-vector amplitude, and analytic twist/acceleration.
3. Added four-step distribution scheduling, every-step WBC, continuous arm-target integration, safety override re-solve, effort impedance, and verified-target fallback.
4. Dependency injection tests recorded motion updates at steps `0/4/8` and WBC on every step without importing Isaac.
5. Focused trajectory/Teacher: `10 passed`.
6. Tasks 1–7 regression: `110 passed`.
7. `py_compile` and `git diff --check`: exit `0`.

## Result

- Same seed and reset state reproduce target pose, twist, and effort.
- High-level velocity is integrated continuously between 50 Hz updates.
- HOLD/RETRACT arm targets still pass through WBC before impedance.
- A motion/WBC failure advances safety and reuses only the last verified target; the failed/unverified feedforward effort is never reused.
- Teacher output contains effort, q/qd targets, target pose/twist, motion diagnostics, QP result, safety state, and termination flag.
- No Isaac environment, play, or training code changed.

## Follow-up

Proceed to Task 8 isolated Isaac Lab effort-control environment and Gym registration.

## Git refs

- Baseline Ref: `ab55314`
- Candidate Ref: `055d126`
- Key Files:
  - [trajectory](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/trajectory.py)
  - [Teacher](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py)
  - [Teacher tests](../../Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py)
