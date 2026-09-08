# T500 M1 + 双 Panda + 双 O6 分层双手 MPC

## Current State

2026-09-06 更新：单轴右掌姿态 MPC 已实现，但 1600 步物理门失败。
350 步诊断精确复现第 184/270/329 步的 Arm 不可行/限位/安全拒绝，
确认姿态坐标差与空间角位移存在不一致，且首目标相对实测姿态的误差增长。
见[诊断日志](../log/2026-09-06-right-palm-orientation-boundary-diagnosis.md)。
以下资产 Tasks 1–3 描述保留为历史基线。

交互设计和书面规格均已由用户确认。首版使用 M1、公共单轴回转平台、左右两条 Panda 和左右 O6，完成固定 `0.5 kg` 箱体的确定性双手夹持、抬升 `0.10 m`、保持 `3 s`、下降和释放。采用 object MPC、双 Arm MPC、双 Hand MPC 和 200 Hz WBC/QP；第一阶段不训练 RL。

规格已写入 [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)，13 任务 TDD [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)正在以 Inline Execution 执行。Tasks 1–3 已完成：左右 O6 资产规范化入库，组合 articulation 已生成并通过最终 Isaac 2000 步硬门。

## Open Children

- T500.4：右掌姿态传递语义与跟踪边界；阻塞 1600 步物理门和正式 30 条验收。

- T500.1：书面规格已确认。
- T500.2：逐文件 TDD 实施计划正在 Inline Execution 执行，Task 4 进行中。
- T500.3：已冻结双臂平台安装变换和 O6 规范化资产 manifest；53 物理 DOF、43 主动通道运行时确认通过。

## Closed Children Archive

- 机械拓扑、首个箱体任务、仿真真值、确定性控制、公共 yaw 平台、分层 MPC、资产边界、安全回退和验收门已完成交互确认。

## Related Logs

- [2026-09-02 设计记录](../log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
- [2026-09-02 实施计划](../log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-plan.md)
- [2026-09-02 Tasks 1–3 资产验证](../log/2026-09-02-m1-dual-panda-o6-asset-tasks1-3.md)

## Git Refs

- Last Feature Commit: `a17912d`
- Last Verified Commit: `a17912d`
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`
- Key Files:
  - [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
  - [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)
  - [现有单臂 MPC](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py)
  - [现有单臂约束](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py)

## Next Step

先修正 T500 姿态适配中的 SO(3) 插值、空间误差和角速度，保持公共合同；
以相同 350 步样本做单变量对照，再评估实测姿态到首目标的跟踪约束。

## Node Details

### T500.1 书面规格复核

重点检查 43 主动控制通道、53 预计物理 DOF 的运行时确认方式、公共 yaw 平台、双 O6 mimic 语义、原子回退和 30/30 固定条件验收。

### T500.2 实施计划

实施计划应按资产输入闭合、单 articulation 构建、静态/物理门、纯控制合同、object MPC、双 Arm MPC、双 Hand MPC、WBC/QP、Isaac 环境和完整任务验收拆分 TDD 任务。
