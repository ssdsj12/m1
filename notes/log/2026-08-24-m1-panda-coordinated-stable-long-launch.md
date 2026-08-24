# 2026-08-24 M1 + Panda coordinated stable long launch

## Launch

- Run: `coordinated_stable_fresh_s42_64x600_20260824`
- Worktree: `/home/xk/coding/M1/.worktrees/m1-panda-ppo-stability/Go2Pvcnn`
- Background process PID: `1128844`
- GPU: `CUDA_VISIBLE_DEVICES=0`, runtime device `cuda:0`
- Contract: 64 envs, at most 600 updates, seed 42, fresh zero-action actor, 256 rollout steps, 200 Hz.
- A1 `model_10402.pt` is provenance-only and was not loaded into actor/critic/optimizer.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_coordinated_train.py \
  --num_envs 64 --max_iterations 600 --seed 42 \
  --run_name coordinated_stable_fresh_s42_64x600_20260824 \
  --init-a1-checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_10402.pt \
  --device cuda:0 --headless
```

Stdout: `Go2Pvcnn/logs/m1_panda_coordinated/coordinated_stable_fresh_s42_64x600_20260824.stdout.log` relative to the worktree.

## First Healthy Update

- manifest status `running`，PID/observation/action 为 `1128844 / 103 / 23`。
- `model_0.pt` generated；total timesteps `16384`；iteration time `4.29 s`。
- finite TensorBoard values: KL `0.0117993`，LR `2.25e-6`，physical std `0.0101509`。
- reset diagnostic sampled flag `1`；joint position/velocity max `0.0299908/0.0499771`。
- first logged base-contact/bad-orientation both `0`。

## Monitoring Boundary

此记录只证明长训进程启动和第一个 update 健康。窗口尚不足，不声明收敛、eligible best、抓取或实机能力。最终只读取 `run_manifest.json`、`best_checkpoint.json` 和 checkpoint SHA；guard 可以因 catastrophe/patience 提前结束。
