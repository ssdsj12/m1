# T400.8b Task 4 deterministic reference QP

## Purpose

Implement the project-owned, single-problem CPU float64 QP backend used as the numerical reference for C0 standing WBC.

## Stage

T400.8b / C0 deterministic Teacher / Task 4

## Related todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [C0 implementation plan](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)

## Input conditions

- Baseline ref: `1cbb0ae`
- Pure PyTorch environment: `/home/xk/miniconda3/envs/go2/bin/python`
- Standard form: equality, upper-form inequality, and lower/upper variable bounds.

## Procedure and evidence

1. RED: collection failed because `qp_backend.py` did not exist.
2. Implemented CPU float64 canonicalization, Hessian symmetrization, deterministic active-set selection, equality KKT solves via `torch.linalg.lstsq`, bound expansion, multiplier removal, and residual diagnostics.
3. First GREEN attempt exposed an empty-inequality reduction in unconstrained and equality-only cases; both failed at `torch.max` on a zero-length vector.
4. Added the zero-inequality branch without changing constrained behavior.
5. Focused QP suite: `18 passed`.
6. Tasks 1–4 pure regression: `74 passed`.
7. `py_compile` and `git diff --check`: exit `0`.

## Result

- Unconstrained, equality-only, box, active inequality, redundant, and mixed problems pass analytic checks.
- Repeated solves are bitwise deterministic in solution, active set, and iteration count.
- Float32/device input is detached into a CPU float64 reference solve.
- Inconsistent bounds and contradictory inequalities return `success=False` with a finite diagnostic solution.
- Malformed shapes and non-finite coefficients fail before solving.
- No WBC formulation, Isaac environment, play, or training code changed.

## Follow-up

Proceed to Task 5 standing WBC problem construction and torque recovery.

## Git refs

- Baseline Ref: `1cbb0ae`
- Candidate Ref: `e746eb9`
- Key Files:
  - [QP backend](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/qp_backend.py)
  - [QP tests](../../Go2Pvcnn/tests/test_m1_panda_qp_backend.py)
