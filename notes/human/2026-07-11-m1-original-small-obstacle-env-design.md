# M1 Original Small-Obstacle Environment Adaptation

## Goal

Use the small-obstacle environment from
`/home/xk/coding/Go2Pvcnn-costmap-teacher-ablation` for M1 stage-two training
while preserving the accepted M1 rolling and flat-wave policies.

## Source Of Truth

The environment behavior comes from the original task
`TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg` and its
dependencies:

- flat terrain generator and semantic obstacle course importer;
- row-based small-obstacle density curriculum;
- semantic height scanner and 16x16 elevation/semantic observations;
- goal-anchored commands;
- batch MPC trajectory teacher and reference rewards;
- semantic collision and body/foot clearance rewards.

The corresponding modules already present in `M1/Go2Pvcnn` are byte-for-byte
copies of the original project's modules. They remain the local runtime copy so
Python never has to import two different `go2_pvcnn` packages.

## Adaptation Boundary

Create an M1 subclass of the original flat-small task. Override only robot
specific contracts:

- use `M1_CFG` and attach the scanner to `BASE_LINK`;
- expose 12 leg position actions and 4 wheel velocity actions;
- replace Go2 body and joint selectors with M1 selectors;
- use M1-safe reset posture, command speeds, termination limits, and rewards;
- retain the existing wave action preparation so stage-two initialization can
  preserve the accepted stage-two-A policy.

Do not recreate obstacle boxes in this task. The original semantic course owns
obstacle placement, dimensions, semantic IDs, terrain levels, and curriculum.

## Task IDs

- Training: `Isaac-M1-Pvcnn-Flat-Small-Avoidance-v0`
- Play: `Isaac-M1-Pvcnn-Flat-Small-Avoidance-Play-v0`

The current deterministic 1/5/10 mm tasks remain available as diagnostic probes
but are not the long-term stage-two training environment.

## Verification

Static tests verify inheritance, source environment reuse, M1 action semantics,
sensor attachment, registration, and absence of a custom obstacle asset. Runtime
verification creates a small headless environment, checks observation/action
dimensions and semantic obstacles, then runs a short rollout initialized from
the accepted flat-wave checkpoint before launching a longer curriculum run.
