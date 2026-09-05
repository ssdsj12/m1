# 2026-09-05 M1 + 右 Panda + 右 O6 资产基座实施计划

## Purpose

把已批准的 T600 工程路线图收敛为首个可执行子阶段 T600.1，冻结单 articulation、29 主动通道、资产 manifest 和 2000 步 Isaac 物理硬门的逐文件 TDD 实施顺序。

## Stage

T600.1 implementation planning。

## Related Todo

- [T600](../todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md)

## Procedure

- 使用 graphify 查询现有资产关系；查询只命中无关 rigid-object 节点，因此不作为设计证据。
- 直接审计主分支 T400 单 Panda builder/config/verifier。
- 直接审计本地 T500 已提交的 O6 source closure、双臂 builder、43 通道合同和 2000 步验证日志。
- 明确排除 T500 worktree 的未提交实验代码和探针产物。
- 将 T600.1 拆为源闭合复核、单臂资产、29 通道合同、物理验证和文档验收五个任务。

## Input Conditions

- T600 路线图提交：`3ceedb6`。
- 执行分支：`o6_400`。
- O6 vendor 源位于 `/home/xk/coding/o6asset`，约 `67 MB`，执行时只读。
- 可审计 T500 source closure commit：`ac95a9b`；后续 T500 提交和 dirty worktree 不进入 T600.1 lineage。
- 现有 M1 + Panda 25-DOF 资产和 T400 公开合同保持不变。

## Key Decisions

- T600 单臂资产不包含 T500 公共 yaw 平台。
- Panda 保持 `panda_joint1..7` 与 `panda_link0..8`，便于后续复用 T400 单臂控制接口。
- 右 O6 rigid body/joint 使用 `right_` 前缀；6 active + 5 mimic。
- 冻结 29 active，预计并运行时核对 34 physical DOF。
- 资产根为 `/M1SinglePandaO6/BASE_LINK`，只有两个 assembly fixed joints。
- final gate 要求四轮接触比例 `1.0`、零非足接触、零限位/非有限/reset/失稳。

## Result

逐文件 TDD 实施计划已写入[计划文档](../../docs/superpowers/plans/2026-09-05-m1-single-panda-o6-asset-foundation.md)。没有修改运行代码、资产或训练状态。

## Conclusion

T600.1 已具备执行计划。T600.2 自由空间控制、T600.3 Contact-aware MPC、T600.4 多模态感知和 T600.5 Residual 继续保持未实施、未验证。

## Follow-up

用户选择 Subagent-Driven 或 Inline Execution 后，在 `o6_400` 的隔离 worktree 中执行 Task 1，并在每个 task 后做规格与质量复核。

## Git Refs

- Baseline Ref: `3ceedb6`
- Candidate Ref: T600.1 plan commit (this commit)
- Key Files:
  - `docs/superpowers/plans/2026-09-05-m1-single-panda-o6-asset-foundation.md`
  - `notes/todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md`
