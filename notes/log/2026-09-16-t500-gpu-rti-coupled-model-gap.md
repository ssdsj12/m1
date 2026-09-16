# T500 eager RTI common-model decision gap

## Purpose / stage / related todo

Task5 pre-implementation investigation, child T500 GPU common-model contract.
Related [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).
Reviewed infrastructure `1d12e53`; current parent status ref `de88fa2`.

## Read-only procedure / evidence

Inspect approved architecture and current CPU dynamics/teacher/object/arm/hand/WBC.
No tests, product code, packages, GPU workload, training or Isaac changes.

- `full_action_teacher.py:263`: independent node effort QPs; arm/hand nominal
  pinned. `:247` copies upstream box/palm/wrench trajectories unchanged.
- `reduced_dynamics.py:condense_constrained_dynamics`: robot KKT only wheel
  constraints; no variable grasp-wrench/object input.
- `object_mpc.py:218`: rigid-body acceleration driven by independent left/right
  wrenches; `:246` condenses40ms object horizon.
- `dual_arm_mpc.py`: existing20ms acceleration-controlled arm model.
- `hand_mpc.py:23`:10ms hand grid,20nodes; contact QP uses qdot/forces and
  one-step compliance. No common40ms effort-driven force-state transition.
- `whole_body_qp.py:110`: hierarchical inverse dynamics/PD tracking produces
  nominal efforts, not a coupled effort/object forward identity.

## Result and new open child

Task5 **NEEDS_CONTEXT**, not E0 blockage or solver failure. A generic GPU residual
solver without an authoritative assembler cannot prove real bimanual prediction.
Do not silently publish copied targets as planned object/palm/grasp trajectories.
No RED/GREEN runs: expected coupled physics must be approved before writing tests.

Open child: define reduced state and augmented controls, robot grasp reaction
load,40ms transition composition (2 arm /4 hand substeps), hard-versus-soft
palm/object/hand linkage with unchanged public safety limits. This is necessary
because approved RTI architecture alone does not specify these physical relations.

Recommendation pending user decision: keep43efforts as output, allow left/right
6D grasp-wrench auxiliary variables internally, reuse existing rigid-body grasp
model/constraints; represent reaction in robot dynamics; compose existing grids,
keep tracking linkage soft and safety constraints hard. Do not add contact-mode
search/perception/Residual or new public contract. This is a proposal, not an
approved/implemented model. Alternative: GPU-port existing hierarchy first,
which preserves model behavior but does not satisfy joint RTI planning by itself.

## Follow-up / refs

Obtain focused model approval, amend approved design and Task5 assembler/fixtures
plan, then resume TDD. Tasks2–3 remain reviewed and tested; no GPU Play readiness.

- Last Verified Product Commit: `1d12e53`.
- Investigation Ref: `de88fa2`.
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`.
