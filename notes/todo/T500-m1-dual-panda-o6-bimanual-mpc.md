# T500 M1 + 双 Panda + 双 O6 分层双手 MPC

## Current State

交互设计和书面规格均已由用户确认。首版使用 M1、公共单轴回转平台、左右两条 Panda 和左右 O6，完成固定 `0.5 kg` 箱体的确定性双手夹持、抬升 `0.10 m`、保持 `3 s`、下降和释放。采用 object MPC、双 Arm MPC、双 Hand MPC 和 200 Hz WBC/QP；第一阶段不训练 RL。

规格已写入 [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)，13 任务 TDD [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)已完成并自检。当前没有运行时代码、资产或训练进程变化。

## Open Children

- T500.1：书面规格已确认。
- T500.2：逐文件 TDD 实施计划已完成，等待用户选择执行方式。
- T500.3：实施前冻结双臂平台精确安装变换和 O6 规范化资产 manifest。

## Closed Children Archive

- 机械拓扑、首个箱体任务、仿真真值、确定性控制、公共 yaw 平台、分层 MPC、资产边界、安全回退和验收门已完成交互确认。

## Related Logs

- [2026-09-02 设计记录](../log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
- [2026-09-02 实施计划](../log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-plan.md)

## Git Refs

- Last Feature Commit: none
- Last Verified Commit: design-only
- Current Work Ref: `main`
- Key Files:
  - [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
  - [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)
  - [现有单臂 MPC](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py)
  - [现有单臂约束](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py)

## Next Step

等待用户选择 Subagent-Driven 或 Inline Execution。开始实施时先创建隔离 worktree，再从 Task 1 的 O6 资产闭合 RED 开始。

## Node Details

### T500.1 书面规格复核

重点检查 43 主动控制通道、53 预计物理 DOF 的运行时确认方式、公共 yaw 平台、双 O6 mimic 语义、原子回退和 30/30 固定条件验收。

### T500.2 实施计划

实施计划应按资产输入闭合、单 articulation 构建、静态/物理门、纯控制合同、object MPC、双 Arm MPC、双 Hand MPC、WBC/QP、Isaac 环境和完整任务验收拆分 TDD 任务。
