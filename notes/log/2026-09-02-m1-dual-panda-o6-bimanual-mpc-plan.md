# M1 + 双 Panda + 双 O6 分层双手 MPC 实施计划

## Purpose

把用户批准的 T500 规格展开为可逐项执行、逐项评审的 TDD 实施计划。

## Stage

T500 implementation planning。

## Related Todo

- [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)

## Procedure

- 映射现有 M1 + Panda builder、PXR verifier、Arm MPC、WBC/QP、Isaac cfg/wrapper 和 Gym registry 模式。
- 冻结新增文件结构与跨任务接口。
- 按资产闭合、组合 articulation、物理门、控制合同、三层 MPC、WBC、runtime、Isaac 环境和正式验收拆分 13 个任务。
- 为每个任务写入 RED、最小 GREEN、focused regression、真实 Isaac gate 和独立 commit。
- 对照批准规格检查覆盖、占位符和类型一致性。

## Input Conditions

- 批准规格：[2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
- 设计提交：`8787f00`
- 当前未授权修改运行时代码或 USD。

## Key Metrics

- 13 个独立 TDD 任务。
- 43 主动通道；运行时测量物理 DOF，不硬编码预计 53。
- 25/50/100/200 Hz 多速率调度。
- seeds 42/43/44 × 10 trials，30/30 才接受。

## Result

计划已写入 [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)。placeholder scan 和 `git diff --check` 通过；Task 11/12 的 probe 创建顺序和跨模块类型定义已在自检中修正。

## Conclusion

计划阶段完成，尚未修改资产、控制代码、Gym 注册或仿真状态。下一步由用户选择 Subagent-Driven 或 Inline Execution。

## Follow-up

按用户选择调用对应执行 skill；开始实施前使用隔离 worktree，并从 Task 1 的 O6 资产闭合 RED 开始。

## Git Refs

- Baseline Ref: `8787f00`
- Candidate Ref: plan commit pending
- Key Files:
  - `docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md`
