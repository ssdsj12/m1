# T500 M1 + 双 Panda + 双 O6 分层双手 MPC

## Current State

2026-09-11 更新：T500.5 DexManipNet 指尖运动先验设计已获用户逐段批准。完整 FAVOR 与
OakInk V2 将经源手 URDF FK 转为掌坐标五指尖短时概率分布，先训练离线教师 ensemble，
再蒸馏冻结学生并只作为 O6 Hand MPC
可退让软代价；模型不接收任务/对象 ID，也不决定掌、物体、机械臂或底盘轨迹。Task 1 已
冻结固定 pin、掌坐标五指窗口、四分量分布、student metadata 和 Inspire/Shadow 左右手
registry；合同与当前 dual-Panda 合同共 `16 passed`。尚未下载数据、训练模型或修改运行时。
见[合同验证](../log/2026-09-11-t500-dexmanipnet-prior-contracts.md)。

2026-09-06 更新：单轴右掌姿态 MPC 已实现，但 1600 步物理门失败。
350 步诊断精确复现第 184/270/329 步的 Arm 不可行/限位/安全拒绝，
确认姿态坐标差与空间角位移存在不一致，且首目标相对实测姿态的误差增长。
见[诊断日志](../log/2026-09-06-right-palm-orientation-boundary-diagnosis.md)。
以下资产 Tasks 1–3 描述保留为历史基线。

2026-09-11 更新：本轮控制与训练改进已整理为功能提交 `265fbcf`。改动包含
SO(3) 几何插值/空间姿态误差、接触摘要与逐 body Box 过滤、举升动作原语与
向量控制入口，以及可恢复的 10k/40k 固定总步数账本和 teacher 数据准备链。
T500 专项测试 `224 passed`，Python 编译和 staged diff 检查通过；正式 30/30
物理验收仍未执行，不能据此宣称整套举升任务已验收通过。见
[上传验证日志](../log/2026-09-11-t500-github-upload-verification.md)。

交互设计和书面规格均已由用户确认。首版使用 M1、公共单轴回转平台、左右两条 Panda 和左右 O6，完成固定 `0.5 kg` 箱体的确定性双手夹持、抬升 `0.10 m`、保持 `3 s`、下降和释放。采用 object MPC、双 Arm MPC、双 Hand MPC 和 200 Hz WBC/QP；第一阶段不训练 RL。

规格已写入 [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)，13 任务 TDD [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)正在以 Inline Execution 执行。Tasks 1–3 已完成：左右 O6 资产规范化入库，组合 articulation 已生成并通过最终 Isaac 2000 步硬门。

## Open Children

- T500.5：DexManipNet 掌坐标五指尖概率先验；Task 1 合同已完成，Task 2 安全下载/解压待执行。

- T500.4：右掌姿态传递语义与跟踪边界；阻塞 1600 步物理门和正式 30 条验收。

- T500.1：书面规格已确认。
- T500.2：逐文件 TDD 实施计划正在 Inline Execution 执行，Task 4 进行中。
- T500.3：已冻结双臂平台安装变换和 O6 规范化资产 manifest；53 物理 DOF、43 主动通道运行时确认通过。

## Closed Children Archive

- 机械拓扑、首个箱体任务、仿真真值、确定性控制、公共 yaw 平台、分层 MPC、资产边界、安全回退和验收门已完成交互确认。

## Related Logs

- [2026-09-11 DexManipNet 指尖先验实施计划](../log/2026-09-11-t500-dexmanipnet-fingertip-prior-plan.md)

- [2026-09-11 DexManipNet 指尖先验合同冻结](../log/2026-09-11-t500-dexmanipnet-prior-contracts.md)

- [2026-09-11 DexManipNet 指尖先验设计](../log/2026-09-11-t500-dexmanipnet-fingertip-prior-design.md)

- [2026-09-11 远端快进合入与本地验证](../log/2026-09-11-t500-remote-fast-forward-local-verification.md)

- [2026-09-11 根 README 状态说明](../log/2026-09-11-t500-readme-status.md)

- [2026-09-08 O6 原始资产迁入仓库](../log/2026-09-08-o6-source-relocation.md)

- [2026-09-02 设计记录](../log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
- [2026-09-02 实施计划](../log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-plan.md)
- [2026-09-02 Tasks 1–3 资产验证](../log/2026-09-02-m1-dual-panda-o6-asset-tasks1-3.md)

## Git Refs

- Last Feature Commit: `265fbcf`
- Last Verified Commit: `037595b`（本机 T500 专项静态/纯控制测试）
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`
- Key Files:
  - [DexManipNet 指尖先验实施计划](../../docs/superpowers/plans/2026-09-11-t500-dexmanipnet-fingertip-prior.md)
  - [DexManipNet 指尖先验设计](../../docs/superpowers/specs/2026-09-11-t500-dexmanipnet-fingertip-prior-design.md)
  - [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
  - [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)
  - [现有单臂 MPC](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py)
  - [现有单臂约束](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py)

## Next Step

由用户选择 T500.5 Subagent-Driven 或 Inline Execution 后，从合同 RED 开始执行 12 任务计划；运行时物理主线仍需在
恢复服务器 Vulkan/DRM 访问后运行跨 seed 物理门并核对固定步数账本，
再执行正式 30/30 举升、保持、下降和释放验收；若首节点仍追不上实测姿态，
再单独评估跟踪感知边界。

## Node Details

### T500.1 书面规格复核

重点检查 43 主动控制通道、53 预计物理 DOF 的运行时确认方式、公共 yaw 平台、双 O6 mimic 语义、原子回退和 30/30 固定条件验收。

### T500.2 实施计划

实施计划应按资产输入闭合、单 articulation 构建、静态/物理门、纯控制合同、object MPC、双 Arm MPC、双 Hand MPC、WBC/QP、Isaac 环境和完整任务验收拆分 TDD 任务。
