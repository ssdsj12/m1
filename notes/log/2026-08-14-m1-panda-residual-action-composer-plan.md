# M1 + Panda 受限残差动作组合器实施计划记录

## Purpose

把已批准并经用户书面复核的残差组合器规格拆成可逐项执行、逐项验证的 TDD 实施步骤。

## Stage

T400 follow-on control foundation / implementation planning。

## Related Todo

- [T400 M1 + Panda 六轴力感知 Teacher–Student](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Procedure

1. 将配置和初始状态、动作合成、reset/原子异常处理拆成三个独立 RED→GREEN 任务。
2. 为每个任务写明准确文件、接口、测试代码、最小实现、命令和预期结果。
3. 增加第四个回归与 repository memory 对齐任务。
4. 对照规格检查覆盖范围、占位符、类型名称和非目标边界。

## Result

实施计划共 4 个任务，覆盖配置默认值、16 维通道契约、物理单位映射、幅值/变化率限制、逐环境状态、选择性 reset、梯度语义、诊断克隆、输入校验、失败原子性、foundation 回归和笔记对齐。

## Verification

- 实施计划 `737` 行。
- 任务数 `4`。
- 禁止的占位符和泛化实现语句扫描无匹配。
- 构造器、`compose()`、`reset()` 和三个诊断属性在所有任务中命名一致。
- 本阶段只写计划，未修改生产代码或运行行为测试。

## Follow-up

按用户先前确定的单代理约束，使用 `executing-plans` 在当前会话逐任务执行，并在每个 RED/GREEN 边界汇报证据。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: unavailable
- Git Ref: unavailable（`/home/xk/coding/M1` 不是 Git 工作树）
- Key Files:
  - [Implementation plan](../../docs/superpowers/plans/2026-08-14-m1-panda-residual-action-composer.md)
  - [Approved design](../../docs/superpowers/specs/2026-08-14-m1-panda-residual-action-composer-design.md)
