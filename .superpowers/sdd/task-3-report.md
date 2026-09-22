# Task 3 Report: Deterministic multi-object O6 scene configuration

## Status

Implemented and locally verified.

Feature commit: `b44a29a` (`feat: add deterministic multi-object O6 scene config`).

## Scope implemented

- Added `build_object_scene_cfg(...)` with catalog validation, stable object-ID ordering,
  enabled-instance filtering, deterministic `{ENV_REGEX_NS}/Objects/{object_id}` paths,
  per-class USD path/mass/uniform-scale settings, and initial pose handling.
- Added `select_target_and_obstacles(...)` with explicit target lookup and deterministic
  obstacle partitioning; disabled instances are excluded.
- Added opt-in `object_catalog` and `object_instances` fields to the dual Panda/O6
  environment config. The default `None` path leaves the existing `scene.box` and
  `{ENV_REGEX_NS}/Box` configuration unchanged.
- Added focused tests for target selection, obstacles, unique paths, class settings,
  disabled instances, and legacy catalog behavior.

## RED evidence

Command, from `Go2Pvcnn`:

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_object_scene.py
```

Observed before implementation:

```text
FFFF                                                                     [100%]
4 failed in 0.83s
```

All failures were the expected missing-module failure for
`go2_pvcnn.tasks.m1_object_scene`.

## GREEN and focused regression evidence

The same focused command after implementation:

```text
.....                                                                    [100%]
5 passed in 0.83s
```

Task 2 catalog plus existing O6 environment static checks:

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_object_scene.py tests/test_m1_object_catalog.py \
  tests/test_m1_dual_panda_o6_env_static.py
```

```text
...................                                                      [100%]
19 passed in 0.85s
```

Additional verification:

```bash
python -m py_compile \
  Go2Pvcnn/go2_pvcnn/tasks/m1_object_scene.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py \
  Go2Pvcnn/tests/test_m1_object_scene.py
```

Exit code was `0`. Scoped `git diff --check` for the three feature files also exited
`0` with no output.

## Concerns / unverified boundary

- No Isaac Sim startup was run in this environment; the focused scene tests use small
  Isaac Lab configuration stubs, while existing static checks cover the legacy scene
  contract.
- A repository-wide `git diff --check` still reports an unrelated pre-existing blank
  line in `.superpowers/sdd/task-6-brief.md`; no feature file is implicated.
