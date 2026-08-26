# M1 + Panda 8D Residual WBC GPU0 smoke

## Device and commands

- Device: NVIDIA RTX 5070 via `CUDA_VISIBLE_DEVICES=0`, Isaac task `Isaac-M1-Panda-Residual-Wbc-v0`.
- Combined asset SHA-256 remained `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`.
- Zero residual: 64 warm-up + 256 measured steps.
- Direction probes: axes `0..7`, each at `+0.1` and `-0.1`, 64 warm-up + 128 measured steps per run.

See [runbook](../../Go2Pvcnn/docs/m1_panda_residual_wbc_runbook.md) for the reproducible commands.

## Zero-residual result

Exit `0`, `steps_complete`, all values finite, QP `256/256`, four wheel contacts throughout, base contact/joint-limit/reset counts all zero, safety state `TRACK` for `256/256`. Maximum roll/pitch were `0.0010422/0.00001447 rad`; maximum EE position error was `0.00020371 m`.

The normalized residual was exactly zero. The measured physical residual maximum was `0.0125645` because the approved mount-wrench feedback remains active; zero-command equivalence without feedback is covered by the CPU/QP tests.

## Sixteen directional probes

All 16 processes exited `0` with `steps_complete` and finite state. Across the runs:

- QP feasible rate: minimum/maximum `1.0/1.0`.
- Minimum wheel contacts: `4`; base contacts, joint-limit violations, and resets: `0`.
- Maximum absolute roll/pitch: `0.00197487/0.000144573 rad`.
- Maximum EE position error: `0.000269226 m`.
- Maximum requested normalized residual: `0.1`.
- Maximum physical residual / filtered mount wrench / feedback correction: `5.02660/0.652453/5.02660` in their channel units.

## Known warning boundary

Isaac still emits the previously known disjointed-transform warning for `Panda/root_joint` and deprecated PhysX force API warnings. No corresponding snap, reset, contact loss, limit violation, or non-finite metric occurred in these acceptance runs.
