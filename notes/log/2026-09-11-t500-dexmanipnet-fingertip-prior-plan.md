# T500 DexManipNet 指尖先验实施计划记录

## Purpose

把已批准的 DexManipNet 指尖概率先验规格拆成可执行、可审查的逐文件 TDD 任务。

## Stage

T500.5 implementation planning；未下载数据、未修改运行时代码、未训练模型。

## Related Todo

- [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)

## Procedure

- 复核已批准设计的外部 pin、几何边界、教师学习、学生蒸馏、原子 Hand QP 和验证门。
- 检查当前 Hand MPC、O6 mimic Jacobian、runtime/wrapper、Probe/Play 和验证入口。
- 将工作拆为 12 个独立任务，每项包含精确文件、接口、RED、GREEN、回归和提交命令。

## Key Metrics

- 12 个任务。
- 100 Hz、20 节点、五指 `15D` 速度、四分量混合、`42D` 网络输入。
- 教师相对零速度基线至少改善 `10%`；学生 NLL 增量不超过 `0.05 nat/dim`。
- 学生 CPU p99 小于 `2 ms`；正式整机门仍为三 seed 共 `30/30`。

## Result

实施计划写入
[T500 DexManipNet 指尖先验计划](../../docs/superpowers/plans/2026-09-11-t500-dexmanipnet-fingertip-prior.md)。

## Conclusion

计划阶段完成。下一步由用户选择 Subagent-Driven 或 Inline Execution；执行时从 Task 1 的合同
RED 测试开始，不能直接下载数据或先写生产代码。

## Follow-up

按用户选择调用对应执行 skill，逐任务执行并在每个审查门更新计划 checkbox 和仓库 notes。

## Git Refs

- Baseline Ref: `1275812`
- Candidate Ref: plan commit pending
- Key Files:
  - `docs/superpowers/plans/2026-09-11-t500-dexmanipnet-fingertip-prior.md`
  - `docs/superpowers/specs/2026-09-11-t500-dexmanipnet-fingertip-prior-design.md`
