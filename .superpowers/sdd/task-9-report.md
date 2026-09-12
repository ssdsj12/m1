# Task 9 Report

Implemented atomic optional fingertip-prior integration for precontact and contact O6 Hand MPC.

- Disabled mode is exact: unchanged deterministic precontact and one unchanged contact QP, with no
  prior import/query or second QP.
- Prior mode always computes a same-cycle baseline first. Precontact uses a bounded six-rate PSD
  projection with persistent O6 latch mapping; contact adds only a PSD Hessian/gradient soft term
  to the unchanged hard problem.
- Prior/query/validation/second-solve failures accept the same-cycle baseline without old-state
  pollution. Baseline infeasibility keeps the original `_last_safe` fallback and skips the prior.
- Runtime forwards real per-side palm-frame positions and folded Jacobians together with O6 state,
  contacts, and phase. Artifact/config construction remains Task 10.
- RED: expected missing input/constructor API (`21 failed, 3 passed`); adversarial property RED also
  proved fail-closed validation. Focused GREEN: `31 passed`; requested Hand/QP: `44 passed`; Tasks
  1–9 relevant regression: `198 passed in 26.58s`.
- Default fresh import loads no prior/offline/HF/HDF5/Isaac module. Python compilation and scoped
  diff checks are part of the final verification.

No real production artifact, Isaac physical run, GPU timing, or 30-trial acceptance was performed.
