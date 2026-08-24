# 2026-08-24 M1 + Panda Coordinated PPO 稳定重训实施计划

## Purpose

把已批准的稳定重训规格转换为可按单代理 TDD 顺序执行的逐文件计划，并在编码前冻结接口、验证命令和破坏性 checkpoint 清理边界。

## Stage

T400.10a / implementation planning。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [approved specification](../../docs/superpowers/specs/2026-08-24-m1-panda-coordinated-ppo-stability-design.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md)

## Procedure

- 复核当前 coordinated train/wrapper/env cfg、vendored runner/PPO、ActorCritic std、Teacher wrench scheduler、Isaac Lab reset/material event 和既有测试。
- 使用现有 graph 先查询 `training/wrench/reset`；图谱只覆盖旧 rigid-object reset，因此以源码路径和测试为最终依据。
- 将实现拆成九个可独立 RED→GREEN→commit 的任务，并固定 GPU0 probe、64×50 guarded short train、旧 checkpoint SHA 审计和 64×600 fresh long launch。
- 自审规格覆盖、占位符、类型/接口一致性和默认 Play/runner 向后兼容边界。

## Key Decisions

- 新建 coordinated 专用 PPO cfg，不污染 A0/A1 cfg。
- runner 只提供通用不可变 iteration summary/callback；M1/Panda 排名与 checkpoint I/O 留在专用模块。
- 无 eligible best 时 `model_final.pt` 回退到 diagnostic best，但 manifest 必须 `accepted=false`。
- 腿/Panda/轮 joint reset 由一次原子事件完成，避免多个 reset event 的执行顺序覆盖。
- intermittent wrench 固定为 `0.25 s` 周期、前 `20%` on，避免 200 Hz Bernoulli 抖动。
- checkpoint 清理使用 dry-run 默认与 `planned→completed` 两阶段原子审计；只允许精确 long-v4 路径。

## Result

计划完成初步自审：覆盖 200 Hz/103/23、adaptive KL/LR、std clamp、100-episode best、早停回退、末端 wrench、reset/摩擦 DR、TensorBoard、GPU probe/short train、审计删除、文档对齐和长期任务启动。当前仍未修改运行代码、执行测试、删除 checkpoint 或启动新训练；下一步是用户选择/确认单代理 Inline Execution。

## Git Refs

- Baseline Ref: `288919d`
- Candidate Ref: plan commit containing this log
- Key Files: `docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md`
