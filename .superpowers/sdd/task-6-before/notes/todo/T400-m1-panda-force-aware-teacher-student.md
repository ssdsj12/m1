# T400 M1 + Panda 六轴力感知 Teacher–Student

## Current State

用户已确认完整设计：Panda 作为本地离线 USD 与 M1 组成统一 articulation；实机在 Panda 与 M1 安装界面使用六轴力/力矩传感器；Teacher 使用仿真理想安装点总六维反力，Student 使用带噪六轴测量、本体感知和时序估计器，输出对 M1 基础运动策略的受限平衡残差。首个任务是 M1 驻停时完成 0–3 kg 抓取和短距离搬运。

设计文档已完成、自审并由用户批准。Asset/wrench foundation Task 1–4 的本地资产、单 articulation、拓扑验证和 isolated smoke 已完成。Task 5 Fix Round 1 已纠正 raw PhysX joint-frame/about-joint-origin 边界：先由 mount actor pose 转 world，再由 pure helper 转到 `BASE_LINK` origin/frame；正式 USD child local zero/identity 已进入 builder/verifier/PXR 资产契约。计划 `26 passed`、expanded `42 passed`，rebuild/checksum/PXR/CPU verifier 与真实 Isaac/AppLauncher checks exit `0`；当前等待复审。

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
- [x] T400.2b.4 完成 Task 4：combined asset cfg、isolated M1-only smoke cfg、Gym 注册与静态/导入验证。
- [x] T400.2b.5 完成 Task 5：base-frame/base-origin mount wrench 纯变换、adapter、导出和 smoke observation。

## Related Logs

- [设计记录与自审](../log/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
- [Asset/wrench foundation plan](../log/2026-08-14-m1-panda-asset-wrench-foundation-plan.md)
- [Task 1 offline asset inputs](../log/2026-08-14-m1-panda-offline-asset-inputs.md)
- [Task 2 single articulation build](../log/2026-08-14-m1-panda-single-articulation-build.md)
- [Task 3 offline/topology verification](../log/2026-08-14-m1-panda-offline-topology-verification.md)
- [Task 2→3 asset repair](../log/2026-08-14-m1-panda-task23-asset-repair.md)
- [Task 4 combined cfg and isolated smoke](../log/2026-08-14-m1-panda-combined-smoke-cfg.md)
- [Task 5 base-frame mount wrench](../log/2026-08-14-m1-panda-base-frame-mount-wrench.md)

## Git Refs

- Last Feature Commit: unavailable（`/home/xk/coding/M1` 不是 Git 工作树）
- Last Verified Commit: unavailable
- Current Work Ref: filesystem working copy
- Key Files:
  - [设计文档](../../docs/superpowers/specs/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
  - [第一阶段实施计划](../../docs/superpowers/plans/2026-08-14-m1-panda-asset-wrench-foundation.md)
  - [Task 4 combined asset cfg](../../Go2Pvcnn/go2_pvcnn/assets/m1_panda.py)
  - [Task 4 isolated smoke cfg](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py)
  - [Task 5 wrench transform and adapter](../../Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py)

## Next Step

Task 5 Fix Round 1 等待独立复审，不声明 PASS。raw incoming 命名保留 parent-on-child；moment 使用 `+(p_mount-p_base)×F` 平移到 `BASE_LINK` actor origin。Task 6 仍只负责 live 数值/符号与 sensor-facing convention 校准，未触碰。机械验算仍是最大载荷实机测试前置门。

## Node Details

### T400.1 书面设计审阅

重点检查统一 articulation 与六轴传感器接口、Teacher/Student 可见信息边界、12 腿位置加 4 轮速残差动作、安全降级状态机，以及 0–3 kg 静止抓取验收范围。

### T400.3 机械验算前置门

策略设计不代表 M1 机架、转接板、传感器和执行器已经具备承载能力。任何最大负载或快速摆臂实机测试前，必须独立验证 Panda 本体质量、物体质量、最大力臂、动态载荷和安全系数。
