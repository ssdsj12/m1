# T500 GPU RTI common-model design

## Purpose / stage / related todo

Resolve the Task5 common-model gap recorded in
[the investigation](2026-09-16-t500-gpu-rti-coupled-model-gap.md), without
starting implementation. Related [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Decision

The user accepted continuing with the recommended boundary. A focused written
design is now available at
[GPU RTI common-model design](../../docs/superpowers/specs/2026-09-16-t500-gpu-rti-common-model-design.md).

- Public/accepted action remains `[B,25,43]`.
- Each RTI node privately augments 43 efforts with left/right base-frame 6-D
  grasp wrenches.
- A shared wrench drives existing box rigid-body dynamics and enters robot KKT
  with equal-and-opposite palm-Jacobian reaction.
- The40ms node uses two20ms robot/arm and four10ms O6 substeps; no variable
  contact-event branch.
- Dynamics, limits, friction, collision and complete-horizon validation remain
  hard. Palm/object closure and target tracking remain soft.
-30D fingertip force/contact-mode optimization is explicitly out of scope.

The design defines atomic row-local rejection, fixed line-search alphas, E0-off
zero prior contribution, component boundaries and TDD fixtures. It does not
change product code, current fail-closed GPU backend, public CPU contracts,
packages, artifacts or Isaac state.

## Self-review / result / follow-up

Checked for placeholders, contradictions, scope growth and ambiguous frame/sign
semantics. No TBD/TODO remains. Frame, spatial order, reaction sign, common grid,
hard/soft boundary and public action boundary are explicit. The design remains
focused on Task5; scan/Triton/graphs/runtime/physics stay later tasks.

Next gate: user review of the written spec, followed by a revised Task5 TDD plan.
No implementation or test claim in this log.

- Last Verified Product Commit: `1d12e53`.
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`.
