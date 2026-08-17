# T400.8 C0 prioritized WBC Teacher foundation plan

## Result

The user confirmed the written prioritized WBC Teacher–Student specification. A ten-task, single-agent, test-driven implementation plan now covers the C0 deterministic Teacher foundation without changing runtime code.

## Frozen C0 boundary

- New Gym ID: `Isaac-M1-Panda-Wbc-Teacher-C0-v0`.
- One unified M1 + Panda articulation; 23 controlled joints and two open position-controlled fingers.
- 10-dimensional base–arm motion distribution at 50 Hz.
- 31-dimensional generalized dynamics and a 43-variable standing QP at 200 Hz.
- Project-owned CPU float64 reference QP before any batched GPU backend.
- Balance/contact feasibility remains above arm tracking.
- GPU0 acceptance uses an 8-step stationary smoke followed by a 2,000-step moving-target run.
- Existing A0/A1 60-observation/16-action routes remain unchanged.

## Plan structure

1. Dimension, name, shape, dtype, device, and finite-value contracts.
2. Planar-base/Panda coordinated kinematics and singularity metrics.
3. Bound-aware three-priority motion distribution.
4. Deterministic active-set reference QP.
5. Standing dynamics/contact WBC formulation.
6. Effort impedance and balance-first safety state machine.
7. Seeded band-limited trajectory and deterministic Teacher composition.
8. Isolated Isaac Lab effort environment and Gym registration.
9. PhysX dynamics adapter and standalone play entry point.
10. Static regression, GPU0 runtime acceptance, runbook, and evidence.

## Review

- Required implementation paths, public interfaces, dimensions, update rates, test commands, expected results, and commits are explicit.
- C0 acceptance gates match the approved specification: end-effector error, singularity response, QP feasibility, wheel slip, body orientation, collision/contact, joint limits, finite state, and no arm snap.
- C1/C2 locomotion, C3 wrench/domain randomization, batched GPU QP, Student, grasping, and real maximum-load work are explicitly excluded.
- No unresolved placeholder text is present.

## References

- Approved design: `docs/superpowers/specs/2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md`
- Baseline ref: `14ace5b`
- Runtime code changed: no

## Next step

Execute the plan inline with `executing-plans`, in the user-required single-agent mode. Stop at each verification checkpoint and do not begin C1–C4 work from this plan.
