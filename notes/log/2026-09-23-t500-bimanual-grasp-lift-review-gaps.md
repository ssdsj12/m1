# T500 bimanual grasp-lift review-gap hardening

## Purpose

Close the remaining review gaps identified after `cd6f2ed`:

- enforce catalog lift `min_vertical_force_n` and `max_tilt_rad` on every
  `HOLD` update;
- route any `HOLD` force/tilt loss through the safe fallback so the mission
  cannot reach `DONE`;
- bind a pipeline-injected catalog goal to `BimanualRuntime` reset state;
- reject conflicting runtime/pipeline goals before mutating runtime state.

## Changes

- `BimanualMission` now checks catalog lift force and tilt independently from
  clamp readiness while in `HOLD`, returning `lift_criteria_failed` and entering
  the existing safe-release path on loss. The legacy `Box` path is unchanged,
  and low lift measurements while still in `LIFT` continue to wait for the
  normal LIFT gate.
- `BimanualRuntime.bind_grasp_goal()` atomically validates and binds the goal
  in both `_grasp_goal` and the current mission. `reset()` therefore recreates
  the mission with the same catalog goal.
- `BimanualGraspLiftPipeline` invokes the runtime binder after all conflict
  checks, preserving the existing generic mission-setter fallback for simple
  test/probe runtimes.

## Verification

Environment: `/home/xk/miniconda3/envs/go2/bin/python`, pytest plugin
autoload disabled.

RED tests before the fix:

- two `HOLD` force/tilt cases stayed in `HOLD`;
- the injected runtime goal was lost after `reset()`.

Focused regression:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  --import-mode=importlib -q \
  tests/test_m1_bimanual_grasp_lift_pipeline.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_bimanual_state_machine.py
35 passed in 0.90s
```

Complete bimanual regression:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:tests \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  --import-mode=importlib -q tests/test_m1_bimanual_*.py
752 passed in 305.81s
```

Compile check:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  /home/xk/miniconda3/envs/go2/bin/python -m compileall -q \
  go2_pvcnn/control/m1_bimanual_coordination \
  tests/test_m1_bimanual_grasp_lift_pipeline.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_bimanual_state_machine.py
exit 0
```

No Isaac, GPU, camera, or physical acceptance claim is made by this CPU
contract fix.

## Git refs

- Baseline: `cd6f2ed`
- Candidate: working tree before focused commit
- Key files: `state_machine.py`, `runtime.py`, `grasp_lift_pipeline.py`,
  `tests/test_m1_bimanual_grasp_lift_pipeline.py`
