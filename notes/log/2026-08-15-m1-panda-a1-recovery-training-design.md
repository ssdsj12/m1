# 2026-08-15 M1 + Panda A1 抗扰恢复训练设计

## Decision

用户批准不从退化的 `model_5999.pt` 盲目续训，改用满幅 checkpoint 筛选、独立 fork、噪声语义修复和 500-iteration 分块训练。

## Evidence

- A1 后 100 轮：`time_out≈0.0966`、`base_contact≈0.6813`、`bad_orientation≈0.2247`。
- 渐进扰动 32-env/1000-step 对照：`model_2700` 为 64 timeout，`model_4500` 为 40 timeout/23 base contact，`model_5999` 为 0 timeout/75 base contact。
- vendored ActorCritic 忽略 `noise_std_type="log"`，并对非负 raw std 应用 softplus；raw `0.01/0.002` 实际约为 `0.698/0.694`。
- 官方本地 RSL-RL 和仓库 ActorCriticCNN 使用直接 scalar std 或 exp(log_std)，确认当前 Teacher 训练语义不一致。

## Approved Boundary

- 满幅评估：64 env、2000 steps、seed 42/43/44、`20 N / 5 Nm`。
- 最终门：timeout `>=80%`、base contact `<=10%`、bad orientation `<=10%`。
- fork 重置 optimizer，学习率 `1e-4`，scalar std 下限 `0.001`，恢复 disturbance global progress。
- 每 500 iterations 训练后满幅复评；连续两块比最佳 survival 低超过 0.10 时停止。
- 不修改 reward、composer、Student、抓取或零间隙资产。

## Artifact

- [书面规格](../../docs/superpowers/specs/2026-08-15-m1-panda-a1-recovery-training-design.md)

## State

书面规格等待用户复核；尚未修改 PPO、训练/Play 入口，也未启动 recovery run。

## Git Refs

- Branch: `main`
- Base: `8872421d02eb93b04b150d025148c8a93e78dd09`
- Current Work Ref: dirty working tree
