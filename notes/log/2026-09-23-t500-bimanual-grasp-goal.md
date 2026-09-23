# T500 bimanual perception grasp-goal boundary

## Purpose

Add a CPU-only perception-to-task-goal adapter for catalog objects without
changing the object MPC, arm MPC, hand MPC, mission phases, or Play runtime.

## Implementation

- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/grasp_goal.py`
  accepts object pose plus explicit dimensions/OBB, a perception-supplied
  `OrientedBoundingBox`, or an object-frame point cloud.
- Point-cloud geometry uses deterministic CPU PCA and positive projected
  extents; explicit OBB frame signs are preserved.
- The output is immutable and contains object-frame OBB facts, world/base-frame
  left/right palm positions and orientations, five fingertip positions and
  contact normals per side, and clamp/lift criteria.
- `grasp_profile` is validated against the catalog profiles and controls the
  conservative normal-force prior. Left/right targets remain mirror symmetric.
- The package root exports the adapter and its records; no solver or task-flow
  consumer was changed.

## Exact verification

Environment: `/home/xk/miniconda3/envs/go2/bin/python` with pytest plugin
autoload disabled.

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_grasp_goal.py
14 passed in 0.86s

/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  go2_pvcnn/control/m1_bimanual_coordination/grasp_goal.py \
  go2_pvcnn/control/m1_bimanual_coordination/__init__.py
exit 0
```

The nearby compatibility command ran 57 tests from the grasp-goal, catalog,
object-target, and object-MPC suites.  The 55 relevant tests passed; two
existing catalog assertions failed because the current worktree catalog uses
`resolved/book.usda` and reaches the missing `cup` asset before `bottle`.
Those failures are outside this change and the catalog files were not edited.

## Limitations

- This is a deterministic geometric heuristic, not a learned grasp detector or
  collision-aware IK planner. It does not inspect meshes, infer mass/friction,
  or run a camera/VLM pipeline.
- Point clouds must already be registered in the object frame and contain
  non-degenerate 3-D geometry. Symmetric PCA eigenspaces can still depend on
  upstream OBB tie-breaking when the point cloud has exactly repeated extents.
- Contact force, slip, tilt, lift height, and hold duration are conservative
  task-goal criteria; they are not physical guarantees until the existing
  controllers and Isaac simulation validate them.
- The adapter is intentionally not wired into Play or MPC yet; downstream
  wiring can consume the stable goal records in a later task.

## Git refs

- Baseline: parent worktree state before this focused change.
- Candidate: current working tree; commit recorded by the implementing agent.
- Key files: `grasp_goal.py`, `__init__.py`,
  `tests/test_m1_bimanual_grasp_goal.py`.
