# T400.8b Task 5 standing whole-body QP

## Purpose

Construct and solve the C0 balance-first whole-body problem over 31 generalized accelerations and four three-axis wheel contact forces, then recover the 23 controlled efforts.

## Stage

T400.8b / C0 deterministic Teacher / Task 5

## Related todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [C0 implementation plan](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)

## Input conditions

- Baseline ref: `50001fd`
- Pure PyTorch environment: `/home/xk/miniconda3/envs/go2/bin/python`
- One unbatched C0 problem; runtime adapters must squeeze the single PhysX environment dimension.

## Procedure and evidence

1. RED: collection failed because `standing_wbc.py` did not exist.
2. Added 43-variable problem construction, 18 hard equalities, four friction pyramids, qdd bounds, and 46 torque inequalities.
3. Added mount-wrench generalized-force mapping, actuated-row torque recovery, balance-first objectives, and finite failure diagnostics with no new effort command.
4. First focused run found one test fixture dtype mismatch (`external_wrench` float32 versus float64 state); the fixture was corrected and the strict dtype gate retained.
5. Focused standing WBC: `11 passed`.
6. QP + standing WBC: `29 passed`.
7. Tasks 1–5 regression: `85 passed`.
8. `py_compile` and `git diff --check`: exit `0`.

## Result

- Floating-base dynamics and stationary contact acceleration are hard equalities.
- Each wheel contact has positive normal force and a four-sided friction pyramid.
- Generalized acceleration and recovered 23-effort limits are hard bounds.
- External six-dimensional wrench maps through `J_mount.T` with the approved dynamics sign.
- Balance curvature exceeds arm tracking by the frozen priority weights.
- An infeasible QP returns finite qdd/contact diagnostics and `effort=None`.
- No Isaac environment, play, or training code changed.

## Follow-up

Proceed to Task 6 impedance output and balance-first safety state machine.

## Git refs

- Baseline Ref: `50001fd`
- Candidate Ref: `91cc1b1`
- Key Files:
  - [standing WBC](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/standing_wbc.py)
  - [standing WBC tests](../../Go2Pvcnn/tests/test_m1_panda_standing_wbc.py)
