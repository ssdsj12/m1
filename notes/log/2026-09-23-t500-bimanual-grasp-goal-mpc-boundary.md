# T500 bimanual grasp-goal to O6 MPC boundary

## Purpose

Wire the existing CPU `grasp_goal.py` adapter into the O6 runtime/object-MPC
goal boundary while preserving the legacy `Box` behavior when no catalog goal
is configured.

## Changes

- Added optional catalog dimensions (including `dimensions_m` JSON alias) to
  `ObjectClassRecord`.
- Added `build_catalog_grasp_goal()` and the injectable
  `build_catalog_grasp_goal_provider()` to resolve explicit dimensions, OBBs,
  or point clouds first. Conservative class dimensions are available only via
  explicit `allow_default_dimensions=True`.
- Added an optional immutable `grasp_goal` to `ObjectMpcInput`; the existing
  `BimanualObjectMpc.plan(sample)` signature and solver horizon remain intact.
  Catalog goals shift bilateral palm targets with the planned object
  translation; `None` uses the existing hard-coded Box target path unchanged.
- Carried the goal through `ManipulationTarget` and `BimanualRuntime`, exposing
  bilateral palm/fingertip/contact targets and clamp/lift criteria through the
  existing runtime target. Catalog-target wrapper construction accepts an
  injectable geometry provider and explicit default-dimension opt-in.

## Exact verification

Environment: `/home/xk/miniconda3/envs/go2/bin/python`, pytest plugin
autoload disabled.

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_grasp_goal_boundary.py
4 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_grasp_goal_boundary.py tests/test_m1_bimanual_grasp_goal.py tests/test_m1_bimanual_object_mpc.py tests/test_m1_bimanual_runtime.py tests/test_m1_object_target_contract.py
62 passed

/home/xk/miniconda3/envs/go2/bin/python -m py_compile go2_pvcnn/control/m1_bimanual_coordination/{__init__.py,object_catalog.py,task_goal.py,motion_primitives.py,object_mpc.py,runtime.py} go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py tests/test_m1_bimanual_grasp_goal_boundary.py
exit 0
```

The broader catalog command was also run; its two pre-existing assertions fail
because the current worktree catalog uses `resolved/book.usda` and reaches the
missing `cup` asset before `bottle`. No catalog asset or those tests were
changed.

## Limitations

- This is CPU contract wiring and deterministic geometry, not live camera,
  point-cloud, or VLM sensor integration. Geometry remains injectable through
  the provider; point clouds must already be registered in the object frame.
- Default dimensions are deliberately opt-in and are conservative heuristics,
  not mesh measurements. No mass/friction/collision-aware grasp or IK claim is
  made.
- The existing hand and arm solver signatures are unchanged. The object MPC
  consumes the optional goal's palm positions; fingertip/contact targets and
  clamp/lift criteria are exposed on the runtime goal for downstream consumers,
  but are not asserted as physical guarantees until Isaac validation.

## Git refs

- Baseline: `eae14a3` adapter plus existing O6 catalog/MPC boundary.
- Candidate: current worktree commit after this change.
- Key files: `grasp_goal.py`, `task_goal.py`, `object_catalog.py`,
  `motion_primitives.py`, `object_mpc.py`, `runtime.py`,
  `m1_dual_panda_o6_bimanual_wrapper.py`, and
  `tests/test_m1_bimanual_grasp_goal_boundary.py`.
