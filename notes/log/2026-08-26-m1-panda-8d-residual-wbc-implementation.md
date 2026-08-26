# M1 + Panda 8D Residual WBC implementation

## Scope

Implemented the approved first version (Phase 1–4) as an independent path. Existing 103/23 Coordinated PPO, Folded Load, C0, and C1a behavior remains available. Arm MPC and PPO training are intentionally excluded.

## Result

- Added strict eight-channel residual contracts in the order `[Fx,Fy,Fz,Mx,My,Mz,delta_height,delta_stance]`, physical scaling, slew limits, finite checks, selective reset, and mount-wrench feedback.
- Injected the six-axis wrench and height/stance offsets into the existing WBC/QP, with continuous singularity-aware base participation and safety-state scaling.
- Added the exact 103-wide residual observation, runtime mount-wrench/leg-limit access, per-environment controller isolation, Gym registration, and a deterministic play/probe entrypoint.
- Preserved the legacy 23D action and Teacher routes; this entrypoint does not load or train PPO.

## Verification

- Focused TDD was performed task by task, including valid RED failures.
- Final combined regression: `185 passed`.
- Python compile and `git diff --check`: exit `0` before GPU acceptance.
- GPU acceptance is recorded separately in [2026-08-26-m1-panda-8d-residual-wbc-gpu0-smoke.md](2026-08-26-m1-panda-8d-residual-wbc-gpu0-smoke.md).

## Commits

`a19de7b`, `8b36459`, `d00925e`, `0a727fe`, `4d34356`, `0e22a54`, `7b6dc73`.
