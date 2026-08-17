# T400.8b Task 9 deterministic WBC play entry point

## Purpose

Wire live Isaac/PhysX floating-base dynamics into the deterministic C0 Teacher and apply its canonical 23 efforts once per 200 Hz environment step.

## Stage

T400.8b / C0 deterministic Teacher / Task 9

## Procedure and evidence

1. RED: the absent entry point produced `5 failed` for CLI, PhysX reads, Teacher state/action wiring, diagnostics/summary, and no-learning constraints.
2. Added the C0-only CLI, one-environment guard, deterministic seed, zero-motion option, periodic diagnostics, and atomic JSON summary.
3. Added live reads for generalized mass, Coriolis/centrifugal, gravity, and body Jacobian tensors every physics step; tensors are reordered into 31 generalized and 23 controlled coordinates.
4. A GPU0 API probe exposed a `float32` gain versus `float64` QP boundary. Added an executable regression test and fixed all Teacher gains to `float64`.
5. A second probe exposed Isaac Sim 5.1 legacy bias APIs returning only 25 joint values for a floating-base 31-dimensional mass matrix. Added an executable 25→31 compatibility test and selected the full compensation APIs when required, while retaining the required legacy reads.
6. Focused play contract: `7 passed`; play+environment+Teacher: `20 passed`; compile/diff exit `0`.
7. Final 1-step GPU0 API probe reached one real `env.step`, wrote finite JSON, and exited `0`. It reported `QP feasible=0/1` and initial wheel contacts `3/4`, so this is wiring evidence only, not C0 acceptance.

## Result

- No policy runner, model loading, learning, optimizer, or manifest path exists.
- The script reads live 31-dimensional floating-base dynamics, calls the Teacher, applies exactly 23 efforts, and steps the environment once per cycle.
- Dynamic acceptance remains open; the first-frame infeasible QP and missing contact must be diagnosed in Task 10 without weakening its gates.

## Follow-up

Run the Task 10 GPU0 8-step no-motion smoke under systematic debugging, then attempt the 2,000-step moving-target acceptance only after all short gates pass.

## Git refs

- Baseline Ref: `bb7edf1`
- Candidate Ref: `6a424b4`
- Key Files:
  - [play entry point](../../Go2Pvcnn/scripts/m1_panda_wbc_play.py)
  - [play tests](../../Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py)
