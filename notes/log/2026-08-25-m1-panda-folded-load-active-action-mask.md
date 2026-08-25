# 2026-08-25 M1 + Panda folded-load active-action mask

## Purpose

Record Task 2 TDD evidence for excluding the seven Panda coordinates from sampling and PPO optimization while preserving the 23-output checkpoint shape.

## Stage

T400.10b / folded-load curriculum Task 2.

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-25-m1-panda-folded-load-locomotion-curriculum.md)

## Command And Procedure

```bash
cd Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_rsl_rl_active_action_mask.py \
  tests/test_m1_panda_teacher_noise_std.py \
  tests/test_rsl_ppo_adaptive_schedule.py \
  tests/test_m1_panda_teacher_checkpoint.py
```

The valid vendored-module RED was `9 failed`: the old ActorCritic ignored `active_action_mask`, emitted nonzero inactive actions, retained random inactive final rows, and did not reject invalid masks. A first GREEN run left two final-row failures and exposed Boolean advanced-index zeroing as a copy rather than an in-place view; direct assignment fixed the defect.

## Input Conditions

- Baseline ref: `1aef9ed`
- Vendored RSL-RL forced with `PYTHONPATH=rsl_rl`.
- No Isaac application, GPU, policy checkpoint, or training process used.

## Key Metrics

- Final focused plus legacy regression: `63 passed in 1.72s`.
- Scalar and log std modes both pass.
- Inactive sampled actions, distribution means, and inference actions are exact zero.
- Log probability and entropy sum only active dimensions.
- Inactive actor output-row, bias, and std-parameter gradients are exact zero.
- Inactive final rows remain exact zero after Adam update and state-dict round trip.
- Runtime mask buffer is non-persistent, preserving legacy checkpoint state keys.

## Result

Pass for the generic ActorCritic mask boundary. This does not yet configure the folded-load policy, implement PPO KL abort, or verify the 23-action Isaac environment boundary.

## Follow-Up

Implement Task 3 exact folded-load PPO configuration and remaining-minibatch KL abort.

## Git Refs

- Baseline Ref: `1aef9ed`
- Candidate Ref: pending Task 2 commit
- Key Files:
  - `Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic.py`
  - `Go2Pvcnn/tests/test_rsl_rl_active_action_mask.py`
