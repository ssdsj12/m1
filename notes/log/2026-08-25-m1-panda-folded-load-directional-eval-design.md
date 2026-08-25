# M1 + Panda Folded-Load Directional Evaluation Design

## Purpose

Record the approved observability-only design that will identify the exact L0-C0 fixed-evaluation failure direction before any retraining decision.

## Stage And Todo

- Stage: T400.10b post-L0 diagnosis
- Todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Upstream diagnosis: [L0 stop diagnosis](2026-08-25-m1-panda-folded-load-l0-stop-diagnosis.md)
- Design: [directional evaluation diagnostics](../../docs/superpowers/specs/2026-08-25-m1-panda-folded-load-directional-eval-diagnostics-design.md)

## Selected Contract

- Add forward/reverse/left/right count, metric name, RMSE, limit, contact rate, orientation rate, and pass state.
- Keep all existing global and directional thresholds unchanged.
- Make top-level `directional_pass` derive from the serialized per-direction pass values.
- Re-evaluate the preserved `model_best.pt` in an isolated diagnostic directory without overwriting original artifacts.
- Do not resume or redesign training until the failed direction is identified.

## Verification Status

Design only. No runtime code, checkpoint, report, manifest, curriculum state, or training process changed.

## Git Refs

- Diagnostic baseline: `6ca6807`
- Current Work Ref: `codex/m1-panda-ppo-stability`
