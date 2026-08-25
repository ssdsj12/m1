# 2026-08-25 M1 + Panda folded-load PPO guard

## Purpose

Record Task 3 TDD evidence for the exact 200 Hz PPO configuration, zero-output L0 actor, active-dimension KL, and update-local KL abort.

## Stage

T400.10b / folded-load curriculum Task 3.

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-25-m1-panda-folded-load-locomotion-curriculum.md)

## Command And Procedure

```bash
cd Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_folded_load_ppo.py \
  tests/test_rsl_rl_active_action_mask.py \
  tests/test_rsl_ppo_adaptive_schedule.py \
  tests/test_m1_panda_teacher_noise_std.py \
  tests/test_m1_panda_coordinated_train_cfg.py
```

The RED run failed at collection because the config getter did not exist. The first GREEN run had one test-boundary mismatch: existing adaptive LR decreases only for `KL > 2*desired_kl`, while the test used equality `0.020`. The abort threshold correctly fired; the test was corrected to `0.021` to preserve legacy adaptive semantics.

## Input Conditions

- Baseline ref: `b478377`
- Vendored RSL-RL forced with `PYTHONPATH=rsl_rl`.
- Synthetic CPU rollout storage; no Isaac application, GPU, checkpoint, or real training.

## Key Metrics

- Final focused plus legacy regression: `35 passed in 1.62s`.
- Exact config: 256 steps, `gamma=.9995`, `lambda=.995`, 2 epochs, 4 minibatches, LR `1e-5` bounded `[1e-6,1e-4]`, desired KL `.01`, abort `.015`, std `.005` bounded `[.005,.02]`, grad norm `.5`.
- L0 actor output rows and bias initialize to exact zero.
- KL ignores inactive dimensions.
- A `0.021` KL on minibatch two performs one optimizer step, aborts the remaining seven, lowers LR, clears storage, and reports finite losses.
- Abort state resets on the next update; `kl_abort_threshold=None` completes all minibatches.

## Result

Pass for the pure PPO/config layer. The runner does not yet log the new diagnostics, and no Isaac locomotion task or GPU smoke has run.

## Follow-Up

Implement Task 4 isolated folded-load MDP, rewards, 103/23 environment config, and Gym registration.

## Git Refs

- Baseline Ref: `b478377`
- Candidate Ref: pending Task 3 commit
- Key Files:
  - `Go2Pvcnn/agent/m1_panda_folded_load_train_cfg.py`
  - `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py`
  - `Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic.py`
  - `Go2Pvcnn/tests/test_m1_panda_folded_load_ppo.py`
