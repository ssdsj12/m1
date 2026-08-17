# T400.8b Task 6 WBC impedance and safety supervision

## Purpose

Add the shared 23-channel effort impedance boundary and the balance-first degradation state machine required between deterministic WBC and the articulation.

## Stage

T400.8b / C0 deterministic Teacher / Task 6

## Related todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [C0 implementation plan](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)

## Input conditions

- Baseline ref: `8c1b61d`
- Pure PyTorch environment: `/home/xk/miniconda3/envs/go2/bin/python`
- State order: TRACK, SCALE, HOLD, RETRACT, TERMINATE.

## Procedure and evidence

1. RED: collection failed because `impedance.py` did not exist.
2. Added finite all-or-nothing feedforward plus position/velocity feedback with symmetric 23-channel effort clamps.
3. Added two-unsafe-sample escalation, twenty-safe-sample recovery, terminal latch, wheel-stop gates, held target, and rate-limited retract target.
4. First focused run found one over-specific test message and one real non-finite fallback defect: TERMINATE returned the old retract cache instead of the last finite target.
5. Kept the common 23-shape validator and moved the last finite target into the terminal output cache on non-finite signals.
6. Focused safety/impedance: `15 passed`.
7. Standing WBC + safety: `26 passed`.
8. Tasks 1–6 regression: `100 passed`.
9. `py_compile` and `git diff --check`: exit `0`.

## Result

- Effort is `tau_ff + kp(q_des-q) + kd(qd_des-qd)` followed by symmetric limits.
- Warning/critical orientation, contact loss, lateral slip, and QP failure advance with hysteresis.
- HOLD and RETRACT stop wheel motion; retract advances by at most the configured per-step rate.
- TERMINATE is latched until reset.
- Non-finite state immediately terminates and emits the last finite arm target rather than NaN or a home-pose snap.
- No trajectory, Teacher orchestration, Isaac environment, play, or training code changed.

## Follow-up

Proceed to Task 7 seeded trajectory and deterministic Teacher orchestration.

## Git refs

- Baseline Ref: `8c1b61d`
- Candidate Ref: `8d96922`
- Key Files:
  - [impedance](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/impedance.py)
  - [safety](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py)
  - [tests](../../Go2Pvcnn/tests/test_m1_panda_wbc_safety.py)
