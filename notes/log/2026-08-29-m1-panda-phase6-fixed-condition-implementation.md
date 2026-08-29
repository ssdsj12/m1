# 2026-08-29 M1 + Panda Phase 6 fixed-condition implementation

## Scope

Implement the approved stability-first Phase 6 promotion path without changing
the frozen 103D observation, 8D residual contract, Phase 5 hard gates, WBC/QP
projection, physical limits, or 4000-step seeds 42/43/44 evaluation contract.

## Implemented

- Wrench reward is dimensionless and bounded using the existing physical limits
  `(30,30,50,15,15,8)` while raw physical wrench error remains diagnostic.
- Short training performs exactly 100 updates and emits five immutable
  checkpoints at completed updates 0/25/50/75/100. Online logic is safety-only;
  it cannot rank candidates or publish `model_best.pt`.
- Promotion calibrates PhysX noise from nine zero-vs-zero runs, evaluates five
  candidates across three seeds in 15 fresh Isaac Sim processes, applies
  tolerance-aware stability-first comparison, and publishes the sole accepted
  checkpoint atomically.
- Long training requires an accepted promotion manifest and revalidates the
  asset/config/reward/checkpoint SHA-256 lineage.

## Verification before GPU promotion

- Phase 6 focused CPU suite: `57 passed`.
- RNE/coordination/runner regression: `50 passed`.
- Phase 5 GPU0 regression, seed 42, 4000 steps: `accepted=true`, finite,
  MPC/QP feasible rate `1.0`, four wheel contacts, zero reset/base contact/joint
  violation, max roll `0.00331017 rad`, max pitch `0.00007648 rad`, EE position
  error `0.00753717 m`, EE orientation error `0.01774015 rad`, force cosine
  `0.9999999974`, moment cosine `0.9999970431`.

## Current gate

Implementation and preflight regression are complete. GPU0 short training,
fixed-condition promotion, and conditional 3000-update long training remain
runtime gates; no Phase 6 checkpoint is accepted until promotion writes
`accepted=true`.
