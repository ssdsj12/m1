# 2026-08-25 M1 + Panda folded-load MDP and environment

## Purpose

Record Task 4 TDD evidence for the isolated folded-load locomotion MDP, reward, environment, and Gym registration.

## Stage

T400.10b / folded-load curriculum Task 4.

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-25-m1-panda-folded-load-locomotion-curriculum.md)

## Command And Procedure

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_folded_load_mdp.py \
  tests/test_m1_panda_folded_load_env_static.py \
  tests/test_m1_panda_coordinated_mdp.py \
  tests/test_m1_panda_coordinated_env_static.py \
  tests/test_m1_panda_coordinated_learning_env_static.py
```

The RED run failed because the new MDP module did not exist. The first GREEN run produced `23 passed, 1 failed`; the only failure was a test assuming an explicit `panda_finger_joint1` key while the accepted asset uses the regex key `panda_finger_joint.*`. The test was aligned to the real unchanged asset and expanded to check both arm effort limits.

## Input Conditions

- Baseline ref: `f26b863`
- CPU pure/static verification only; no Isaac application or GPU physics step.
- Existing combined asset was read but not modified.
- Existing coordinated Gym ID/config/MDP remained in the regression set.

## Key Metrics

- Final focused plus legacy regression: `29 passed in 0.86s`.
- New ID: `Isaac-M1-Panda-Folded-Load-v0`.
- Boundary: 103 observations, 23 actions, decimation 1, `dt=.005` (200 Hz).
- Dynamic payload: `M1_PANDA_CFG`, fold defaults `0,-.569,0,-2.810,0,3.037,.741`, finger `.04`, PD `80/4`, effort limits `87/12 Nm`.
- Rewards: exact approved X/yaw tracking, lateral/alive/height/stability/slide/first-16 action/rate/torque/termination terms.
- Removed learned objectives: no base-target, EE-tracking, or folded-arm reward.
- Default events: deterministic reset/friction and no external-force event.

## Result

Pass at pure/static level. Real action masking at the environment step, command reset lifecycle, selected-environment DR writes, Panda fold dynamics, and GPU locomotion remain unverified.

## Follow-Up

Implement Task 5 wrapper command lifecycle, exact-zero arm boundary, reset DR, episode metrics, and fold diagnostics.

## Git Refs

- Baseline Ref: `f26b863`
- Candidate Ref: pending Task 4 commit
- Key Files:
  - `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_folded_load.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
