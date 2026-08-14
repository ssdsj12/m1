# M1 + Panda Asset/Wrench Foundation Plan

## Purpose

将已批准的 M1 + Panda 六轴力感知设计拆出第一个可独立验收的实施阶段：本地资产闭包、单 articulation 装配、安装点六维力坐标契约和确定性验证。

## Stage

Implementation planning / local USD asset / Robot Assembler / mount wrench observation.

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Procedure

1. 读取已批准设计和当前 M1 资产、smoke env、测试结构。
2. 检查本地 Isaac Lab 2.1.0 的 Franka 配置、URDF converter、Robot Assembler 和 incoming wrench API。
3. 将大规格按独立子系统拆分，先规划资产与 wrench foundation。
4. 写出 TDD 步骤、精确路径、接口、命令、预期结果和非 Git 校验点。
5. 自审规格覆盖、占位符、类型和 Robot Assembler 固定关节语义。

## Key Findings

- 当前 M1 floating overlay 仍使用 `/home/xk/ros2_ws/...` 绝对 sublayer，尚不是真正项目内离线闭包。
- 本机 Isaac Sim URDF importer 自带 `panda_arm_hand.urdf`、视觉 mesh 和碰撞 mesh，可作为本地 Panda 源。
- `RobotAssembler(single_robot=True)` 在 attach mount 下创建 `AssemblerFixedJoint`；它是固定约束 Prim，不属于 25 个可控 DOF。
- Isaac Lab 的 `body_incoming_wrench` 返回世界坐标系 incoming wrench，因此项目需要将力旋转到 `BASE_LINK`，并用 `r × F` 将力矩平移到 M1 base 原点。

## Result

计划已写入 [foundation plan](../../docs/superpowers/plans/2026-08-14-m1-panda-asset-wrench-foundation.md)。共 6 个任务，终点为：项目内本地资产、单个 25-DOF articulation、16维 M1 动作保持不变、六轴安装反力通过六方向真实仿真 probe。

未修改运行代码，未生成 Panda/combined USD，未执行仿真。

## Follow-up

用户选择 subagent-driven 或 inline execution 后执行本计划。残差控制、Teacher–Student、IK/OSC、抓取和实机部分保留为后续独立计划。

## Git Refs

- Baseline Ref: unavailable（`/home/xk/coding/M1` 不是 Git 工作树）
- Candidate Ref: filesystem working copy
- Key Files:
  - [implementation plan](../../docs/superpowers/plans/2026-08-14-m1-panda-asset-wrench-foundation.md)
  - [approved design](../../docs/superpowers/specs/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
