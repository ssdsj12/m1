# T500 GPU RTI common-model implementation plan

## Purpose / stage / related todo

Translate the user-approved [common-model design](../../docs/superpowers/specs/2026-09-16-t500-gpu-rti-common-model-design.md)
into executable TDD/review gates. Related [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Plan result

Focused plan:
[2026-09-16-t500-gpu-rti-common-model.md](../../docs/superpowers/plans/2026-09-16-t500-gpu-rti-common-model.md).
The original GPU plan now marks this focused plan as the governing Task5 detail.

It splits Task5 into three independently testable/reviewable changes:

1. 110-state/55-control robot-box KKT transition, substep checks and dense GN direction;
2. four parallel complete-horizon candidates with deterministic atomic selection;
3. Task3 warm-start integration and private `[B,25,43]` publication.

The plan freezes exact tensor shapes, interfaces, reaction sign, grids, failure
isolation, CPU float64/GPU float32 parity, memory guards, commands and scoped
commits. It keeps expert contribution zero, backend fail-closed and all later
scan/Triton/graph/Isaac work out of scope.

## Self-review / result / follow-up

Checked design coverage, placeholder patterns and cross-task type names. Added
the missing111-D measured state for Task3 reporting, explicit current
`GpuReducedDynamics` input, palm residual blocks, state bounds and fixed hard
inequality tensors. No product code or tests run in this planning step.

Execution preflight caught and corrected an omitted base state: the internal state
is110-D (base tangent pose/twist12 + active q/qd86 + box12), then maps back to the
existing111-D quaternion-based reporting layout. This preserves the current fixed
base and the approved later sliding-base extension without a contract change.

Execution mode remains the user's earlier Subagent-Driven selection. Next:
extract focused Task1 brief, implement with TDD, independently review, then
advance one task at a time.

- Last Verified Product Commit: `1d12e53`.
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`.
