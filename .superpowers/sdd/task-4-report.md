# Task 4 Report

Status: DONE_WITH_CONCERNS

Git Ref: unavailable

## Changes

- Added `Go2Pvcnn/go2_pvcnn/assets/m1_panda.py` with `M1_PANDA_CFG`, combined USD path, body constants, 25-DOF contract, legal Panda home pose, and three Panda hold actuators.
- Added `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py`, derived from the old M1 smoke structure but overriding the robot, observations, and actions.
- Added Gym id `Isaac-M1-Panda-Smoke-v0` to `register_m1_envs.py`.
- Added static and lightweight behavior coverage in `test_m1_panda_smoke_cfg_static.py`.
- Updated T400 dashboard, branch, progress, independent log, and log index.

## Contract Evidence

- Actions are exactly 12 M1 leg position actions plus 4 M1 wheel velocity actions.
- `joint_pos_rel` and `joint_vel_rel` each use `SceneEntityCfg(... joint_names=list(M1_JOINT_NAMES))`; action history comes from the 16-term action manager.
- Panda joints are not action terms. Panda actuators only hold the configured home pose.
- Source scan contains no wrench integration; no Task 5 function, IK, OSC, or training was added.
- Lightweight import behavior proves combined cfg nested state does not share with or mutate the original `M1_CFG`.

## RED / GREEN

- RED: focused new test `3 failed in 0.05s`, exit `1`, because both new modules were absent.
- GREEN: focused new test `3 passed in 0.01s`, exit `0`.
- Planned static suite after Fix Round 2: `30 passed in 0.03s`, exit `0`.
- Fix Round 2 test-file `py_compile`: exit `0`.

## Import / Runtime Evidence

- Bounded loco + headless AppLauncher isolated import: exit `0` and reported `dof=25`, USD match true, original cfg unchanged, actions `(12,4)`, and observation joints `16`.
- No physical environment creation or simulation step was run for Task 4.
- After Fix Round 1, normal tasks-package import and `gym.spec("Isaac-M1-Panda-Smoke-v0")` exit `0`; all 21 M1 IDs register without loading their cfg modules eagerly.

## Baseline

- Required three-file static baseline is green (`24 passed`). Old M1 asset/smoke baseline is `4 passed`.
- Expanded `tests/test_m1*_static.py`: `51 passed, 1 failed`; the failure is the existing wave-wrapper source-token contract and does not involve Task 4 files.

## Self-review And Warnings

- Confirmed all Task 4 outputs named in the brief exist.
- Confirmed the old `M1_CFG` and old smoke source were not modified.
- Confirmed no Task 5+ implementation was introduced.
- Gym registration is verified through the real default tasks-package import and `gym.spec`.
- Concern: actual physics startup/step of `M1PandaSmokeEnvCfg` remains unverified in this task.

## Fix Round 1

- Root-cause runtime RED: normal package import reliably exited `17` at the eager `m1_small_obstacle_env_cfg` import because installed IsaacLab lacks `MultiMeshRayCasterCfg`.
- The installed official IsaacLab task pattern uses lazy `module:Class` strings. `register_m1_envs.py` now uses this form for all 21 existing/new M1 IDs and has no task cfg imports. No obstacle implementation was changed, no exception is swallowed, and old cfg modules still load when their environments are actually resolved.
- Round 1 AST RED/GREEN was `1 failed, 4 passed` -> `5 passed`; it substantially strengthened coverage, but the later review correctly found that its action-name prefilter and non-Panda registry shape-only checks did not yet enforce the complete contracts.
- Real public import/Gym evidence: exit `0`; Panda spec has the exact manager entry point and lazy cfg string; M1 registry count is `21`; no next import blocker appeared.
- Real configclass evidence through the normal package path: exit `0`; original `M1_CFG` values unchanged and `spawn`, `init_state`, `actuators` identities are distinct; action types/dims and both observation selectors are exact.
- Deferred old-environment semantics evidence: explicitly resolving the old small-obstacle spec's unchanged module mapping exits `23` with the original `MultiMeshRayCasterCfg` ImportError; the failure is deferred, not swallowed.
- Fix Round 1 verification: planned `24 passed`; old M1 smoke baseline `4 passed`; pycompile exit `0`; full M1 static `51 passed, 1` pre-existing unrelated wave failure.

## Fix Round 2

- Production code was not changed. The sole remaining Important was a regression-test bypass.
- The action validator now inspects every call-valued assignment in `M1PandaSmokeActionsCfg` without prefiltering names, requires the full ordered term list to equal `leg_pos, wheel_vel`, and then checks every required parameter.
- The registry validator contains an explicit reviewed 21-item ordered `ID -> module:Class` table. Every registration must also have the exact `isaaclab.envs:ManagerBasedRLEnv` entry point, `disable_env_checker=True`, exactly two kwargs in the expected order, and `rsl_rl_cfg_entry_point=None`.
- Mutation evidence covers six invalid inputs: extra Panda action, wrong old cfg target, wrong manager, disabled checker, extra kwarg, and non-None RSL. Each mutated source is required to raise `AssertionError`; focused suite result is `11 passed in 0.03s`.
- Verification: planned suite `30 passed in 0.03s`; old M1 asset/smoke baseline `4 passed in 0.01s`; relevant test-file pycompile exit `0`.
- AppLauncher was not rerun because Fix Round 2 changed no production code; Round 1 runtime evidence remains applicable. Git Ref: unavailable.
