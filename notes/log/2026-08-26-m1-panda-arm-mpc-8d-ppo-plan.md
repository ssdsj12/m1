# M1 + Panda Arm MPC + 8D Residual PPO 实施计划

- 日期：2026-08-26
- 任务：T400.13 Phase 5–6
- 状态：计划完成，进入单代理 Inline Execution

## 决策

批准规格已展开为 11 个 TDD 任务：Arm MPC 合同与 QP、PhysX 动力学提取、WBC 前馈接入、103→8 多分支 ActorCritic、稳定优先奖励与 guard、独立 8D 环境、训练入口、probe/eval/play、GPU0 门和运行手册。

实施严格保持 `Arm MPC -> 8D residual -> safety projection -> WBC/QP -> 23 effort`，首轮任务仍仅为 M1 原地平衡与 Panda 小幅六轴末端运动。长训最多 3000 update，但只有 Phase 5、zero residual 和 guarded short train 全部通过后才允许启动。

## 执行边界

- 用户要求单代理，采用 `executing-plans` Inline Execution，不派发子代理。
- 每个任务执行 RED、最小 GREEN、相关回归和独立提交。
- 不改旧 Gym ID、旧 23D actor/checkpoint 或旧训练默认值。
- 不把生成 checkpoint 纳入 Git。
- 当前两个 `graphify-out/cache/last_query_stamp` 工作区变化不属于本任务，不暂存、不覆盖。

## 计划

- [2026-08-26-m1-panda-arm-mpc-8d-ppo.md](../../docs/superpowers/plans/2026-08-26-m1-panda-arm-mpc-8d-ppo.md)
- [批准规格](../../docs/superpowers/specs/2026-08-26-m1-panda-arm-mpc-8d-ppo-design.md)
