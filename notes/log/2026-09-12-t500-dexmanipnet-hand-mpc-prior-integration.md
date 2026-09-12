# T500 DexManipNet Atomic Hand MPC Prior Integration

## Purpose

Record Task 9 integration of the optional frozen fingertip prior into the O6 precontact and
contact controllers. Task 10 artifact/configuration wiring and physical acceptance are not part of
this change.

## Stage And Related Todo

T500.5 / Task 9; see the [T500 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Contract

- `HandMpcInput` now carries the measured palm-frame `(5,3)` fingertip positions. The default
  runtime provider forwards each side's own CPU-float64 positions, folded `(15,6)` Jacobian, O6
  `q/qd`, contact mask, mission phase, and current wrench target without crossing left/right state.
- With `expert_prior=None`, contact planning still performs exactly one unchanged baseline QP;
  precontact planning uses the existing deterministic reference directly. Exact tensor and
  diagnostics equality is covered.
- With a prior configured, APPROACH/PRELOAD first produces the same-cycle deterministic baseline,
  then solves a six-rate projection with joint position/rate bounds, strong baseline-rate tracking,
  and a PSD `J qdot` soft term. Persistent latched finger axes are exact zero using the O6
  thumb-two-axis/four-single-axis mapping; latched Cartesian target rows are removed separately.
- Contact phases first solve the unchanged QP. Only a feasible finite baseline permits a prior
  query and second QP. The added term changes only Hessian/gradient; force/wrench equalities,
  friction inequalities, contact locks, joint bounds, and feasibility meaning remain identical.
- Query timeout/disable/exception, malformed or nonfinite/negative-precision target, and rejected
  second QP all accept the same-cycle baseline and update `_last_safe` once. Baseline infeasibility
  still uses the older safe fallback and never queries the prior.
- Diagnostics add configured/enabled/rejection state, component/probability, inference time,
  precision range, Mahalanobis/prior cost, and baseline-to-regularized tip-rate delta. Existing
  feasibility and fallback fields retain their prior meaning.

## TDD And Verification

RED was observed with `21 failed, 3 passed`: `HandMpcInput` rejected the new position field and
`O6HandMpc` rejected `expert_prior`. A later adversarial target-property test also failed by
propagating `RuntimeError`; output-boundary validation was then made fail-closed. One initial GREEN
run exposed a `4.9e-18` KKT residual on a latch-fixed axis, which is now canonicalized to exact zero.

Focused Hand/Runtime:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_hand_mpc.py tests/test_m1_bimanual_runtime.py
```

Result before the final two edge-case additions: `28 passed in 0.91s`; final focused verification:
`31 passed in 0.92s`.

Requested Hand/QP regression:

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_hand_mpc.py tests/test_m1_bimanual_runtime.py \
  tests/test_m1_bimanual_full_action_teacher.py tests/test_m1_bimanual_o6_contact.py
```

Final result: `44 passed in 1.50s`.

Tasks 1–9 relevant regression (all expert-prior suites plus the four Hand/runtime consumers):
`198 passed in 26.58s`. A fresh-process import check also confirmed default `hand_mpc` import loads
no `expert_fingertip_prior`, Hugging Face, HDF5, or Isaac module.

## Git Refs

- Baseline Ref: `83e3955cdacdbffca57cdabe77fd552b55e34d31`
- Candidate Ref: Task 9 feature commit (this change)
- Key Files: `hand_mpc.py`, `runtime.py`, and their focused tests

## Follow-up

Task 10 may construct two independently validated artifacts/adapters in the wrapper and expose the
new per-side diagnostics. No real artifact, Isaac simulation, GPU timing, or physical 30-trial gate
was run for Task 9.
