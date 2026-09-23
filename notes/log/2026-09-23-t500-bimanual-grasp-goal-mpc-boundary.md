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
- Follow-up hardening converts the frame-kinematics `(xyz, rotvec)` runtime
  pose to the grasp adapter's `(xyz, quaternion)` contract before goal
  generation, with a non-axis-aligned rotation regression.
- Catalog profiles are canonicalized through the supported grasp-goal mapping:
  `two_hand_stable -> symmetric_two_hand` and `stable -> generic`; unsupported
  values fail with the supported profile and alias list.
- A caller-supplied `BimanualRuntime` retains its identity while receiving the
  wrapper's catalog grasp-goal provider, including the provider's geometry and
  default-dimension options.

## Exact verification

Environment: `/home/xk/miniconda3/envs/go2/bin/python`, pytest plugin
autoload disabled.

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_grasp_goal_boundary.py tests/test_m1_bimanual_grasp_goal.py tests/test_m1_bimanual_object_mpc.py tests/test_m1_bimanual_runtime.py
50 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_prior_startup_binding.py -k 'wrapper_consumes_bound_workers_without_reloading_or_owning_them or attaches_catalog_grasp_goal_provider'
2 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_object_catalog.py -k 'load_catalog_resolves_all_six_classes_and_verifies_hashes or load_catalog_validates_materialized_usd_against_resolved_sha_only or validate_instances_rejects_duplicate_ids_and_sorts_deterministically or missing_catalog_uses_legacy_box_fallback or object_catalog_can_be_deepcopied'
5 passed

/home/xk/miniconda3/envs/go2/bin/python -m compileall -q go2_pvcnn/control/m1_bimanual_coordination go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py tests/test_m1_bimanual_grasp_goal_boundary.py tests/test_m1_bimanual_prior_startup_binding.py
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

## Review hardening: closed-loop grasp/lift pipeline

The follow-up review of commit `5e913ce` found five contract gaps in the
catalog closed-loop boundary. They are now covered in the pipeline, runtime,
and mission state machine:

- `DONE` is the only successful terminal phase. `TERMINATED` remains a
  fallback result, including after safe release.
- Catalog clamp metrics use the current snapshot fingertip force vectors,
  projected onto each configured contact normal. Vertical force is measured
  separately for lift load; the legacy Box path retains its predicted-force
  behavior.
- Catalog goals enforce `max_normal_force_n`, `min_normal_alignment`,
  `max_slip_speed_m_s`, and `lift.hold_time_s`. Legacy Box thresholds remain
  unchanged.
- Runtime retains the latest mission state so fallback reasons reach
  `BimanualGraspLiftStep.fallback_reason`.
- A pipeline injected with a runtime carrying a different grasp goal rejects
  the conflict before calling the runtime setter; matching or absent goals
  remain compatible.

Regression coverage includes terminal success/fallback semantics, fallback
reason propagation, conflicting-goal rejection without mutation, current
force projection versus vertical load, force/alignment caps, goal slip speed,
and per-goal hold duration.

Fresh verification from the feature worktree:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=Go2Pvcnn:Go2Pvcnn/tests \
  python -m pytest --import-mode=importlib -q tests/test_m1_bimanual_*.py
749 passed in 308.84s
```

Focused grasp/runtime/state-machine verification also passes (`32 passed`).
Compileall and the final focused command are run on the candidate commit; no
Isaac physical or GPU acceptance claim is made by this CPU contract fix.

Post-commit verification at the final candidate reran the complete bimanual
regression with the same command: `749 passed in 303.57s`. Compileall and
`git diff --check HEAD^ HEAD` also exited `0`.
