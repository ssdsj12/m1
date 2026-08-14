# T400 M1 + Panda 六轴力感知 Teacher–Student

## Current State

用户已确认完整设计：Panda 作为本地离线 USD 与 M1 组成统一 articulation；实机在 Panda 与 M1 安装界面使用六轴力/力矩传感器；Teacher 使用仿真理想安装点总六维反力，Student 使用带噪六轴测量、本体感知和时序估计器，输出对 M1 基础运动策略的受限平衡残差。首个任务是 M1 驻停时完成 0–3 kg 抓取和短距离搬运。

设计文档已完成、自审并由用户批准。Asset/wrench foundation Task 1 的项目自有离线资产输入已完成；Task 2 已将本地 Panda URDF 转换为 USD，并通过 RobotAssembler 生成组合资产。Task 2→3 阻断已修复：生成器会删除 RobotAssembler `_refresh_asset` 的三个错误 list edits、移除 Panda attach root APIs、保留 disabled `root_joint` 与 articulation 内 fixed mount，并将两条项目 reference 相对化。独立审查 Fix Round 1 已补齐 exact root 与完整 mount body0/body1/enabled/exclusion 门，并增加真实 list-op/dependency/root/mount 行为测试。Task 3 轻量测试 `16 passed`、PXR behavior exit `0`；正式资产和完整资产树搬移副本的 CPU verifier 均退出 `0`，得到 exact M1 root、25 DOF、required bodies 唯一、初始化/reset/write/step/update 完成。

## Open Children

- [ ] T400.3 实施前完成 Panda/M1/六轴传感器最坏工况机械验算。

## Closed Children Archive

- [x] T400.0 需求澄清、路线比较和分节设计确认。
- [x] T400.1 用户审阅并批准书面设计。
- [x] T400.2 完成首个 asset/wrench foundation 实施计划。
- [x] T400.2a 用户选择并启动 foundation plan 执行。
- [x] T400.2b.1 完成 Task 1：项目自有离线 M1/Panda 资产输入。
- [x] T400.2b.2 完成 Task 2：本地 Panda 转换与 M1 + Panda 单 articulation USD 装配。
- [x] T400.2b.2r 修复 Task 2→3 持久化 arcs/双 root 阻断，并在 CPU 完成 Task 3 验收；15 个 `OmniPBR.mdl` 明确分类为 Isaac Sim built-in resolver boundary。

## Related Logs

- [设计记录与自审](../log/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
- [Asset/wrench foundation plan](../log/2026-08-14-m1-panda-asset-wrench-foundation-plan.md)
- [Task 1 offline asset inputs](../log/2026-08-14-m1-panda-offline-asset-inputs.md)
- [Task 2 single articulation build](../log/2026-08-14-m1-panda-single-articulation-build.md)
- [Task 3 offline/topology verification](../log/2026-08-14-m1-panda-offline-topology-verification.md)
- [Task 2→3 asset repair](../log/2026-08-14-m1-panda-task23-asset-repair.md)

## Git Refs

- Last Feature Commit: unavailable（`/home/xk/coding/M1` 不是 Git 工作树）
- Last Verified Commit: unavailable
- Current Work Ref: filesystem working copy
- Key Files:
  - [设计文档](../../docs/superpowers/specs/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
  - [第一阶段实施计划](../../docs/superpowers/plans/2026-08-14-m1-panda-asset-wrench-foundation.md)

## Next Step

进入 Task 4 前仍需由主计划确认。当前资产验收为 Isaac-Sim-offline：15 个 bare `OmniPBR.mdl` 依赖由兼容 Isaac Sim 的内置 MDL resolver 提供，并不等价于脱离 Isaac Sim 材质库的严格项目自包含。另保留 PhysX 对 disabled Panda `root_joint` 的 disjointed-transform warning；一步内 mount 相对位移 `4.7527e-05 m`，低于 `1e-4 m` 门限，未观察到 snap。

## Node Details

### T400.1 书面设计审阅

重点检查统一 articulation 与六轴传感器接口、Teacher/Student 可见信息边界、12 腿位置加 4 轮速残差动作、安全降级状态机，以及 0–3 kg 静止抓取验收范围。

### T400.3 机械验算前置门

策略设计不代表 M1 机架、转接板、传感器和执行器已经具备承载能力。任何最大负载或快速摆臂实机测试前，必须独立验证 Panda 本体质量、物体质量、最大力臂、动态载荷和安全系数。
