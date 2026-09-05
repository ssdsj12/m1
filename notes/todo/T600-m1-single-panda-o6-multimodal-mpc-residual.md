# T600 M1 + 右 Panda + 右 O6 多模态 MPC + Residual

## Current State

工程路线图的交互设计和书面规格已由用户批准。首条纵向闭环固定为 M1 原地 + 右 Panda + 右 O6，在 Isaac Lab 中完成固定 `0.5 kg` 箱体的抓取、抬升、保持、下降和释放；先建立真值 nominal MPC 基线，再替换为 RGB-D + LiDAR 状态估计，最后训练受限 9D Residual。

正式路线图见[设计文档](../../docs/superpowers/specs/2026-09-05-m1-single-panda-o6-multimodal-mpc-residual-roadmap-design.md)。T600.1 已按五任务 TDD [实施计划](../../docs/superpowers/plans/2026-09-05-m1-single-panda-o6-asset-foundation.md)完成：项目自有单 articulation、34 物理 DOF、29 主动通道和 GPU0 2000 步物理硬门均已验收。当前进入 T600.2 真值自由空间控制的独立设计/计划周期。

## Open Children

- T600.2：Isaac 真值自由空间 Arm/O6 控制。
- T600.3：真值 Contact-aware MPC 和固定任务 30/30 基线。
- T600.4：仿真 RGB-D + LiDAR 融合及 ObjectState 替换。
- T600.5：冻结 nominal MPC 后的受限 9D Residual 训练。
- T600.6：联合验收与消融。

## Closed Children Archive

- T600.1：M1 + 右 Panda + 右 O6 单 articulation 和 29 通道接口基座已完成；运行时测得 34 物理 DOF，GPU0 2000 步硬门通过。
- 首个本体、任务、仿真边界、推进方式、系统架构、阶段划分、验收门、9D Residual 和代码边界已完成交互确认。

## Related Logs

- [2026-09-05 工程路线图设计](../log/2026-09-05-m1-single-panda-o6-roadmap-design.md)
- [2026-09-05 T600.1 实施计划](../log/2026-09-05-m1-single-panda-o6-asset-plan.md)
- [2026-09-05 T600.1 资产基座验收](../log/2026-09-05-m1-single-panda-o6-asset-foundation.md)

## Git Refs

- Last Feature Commit: `b673ad96f8b535b4f1c69cfada663b027f0a36a1`
- Last Verified Commit: `8215b1b376286823725145df9a280a15a6bd9308`
- Current Work Ref: `o6_400`
- Baseline Ref: `c8bb5e8`
- Key Files:
  - [工程路线图设计](../../docs/superpowers/specs/2026-09-05-m1-single-panda-o6-multimodal-mpc-residual-roadmap-design.md)
  - [T500 双手路线](T500-m1-dual-panda-o6-bimanual-mpc.md)
  - [T400 M1 + Panda 路线](T400-m1-panda-force-aware-teacher-student.md)

## Next Step

为 T600.2 单独完成交互设计和逐文件 TDD 实施计划。下一阶段只建立 Isaac 真值自由空间 Arm/O6 控制，不提前实现接触 MPC、多模态感知或 Residual 训练。

## Node Details

### T600.1 资产与接口基座

目标是项目自有、可重定位的 M1 + 右 Panda + 右 O6 单 articulation。冻结 29 主动通道；预计 34 物理 DOF，但必须由 PXR/Isaac 探针确认。零命令 2000 步物理门通过后才能开始控制开发。

### T600.2 真值自由空间控制

使用统一原子状态快照和 Isaac 箱体真值，验证 M1 平衡、Panda 掌心跟踪、O6 张合和预抓取，不建立计划接触。

### T600.3 真值 Contact-aware MPC

引入箱体动力学、指尖接触、摩擦锥、夹持力与滑移约束。固定 seeds `42/43/44` 各 10 次全部成功后冻结 nominal MPC。

### T600.4 仿真多模态感知

依次完成离线回放、在线开环、真值 shadow 对比和估计闭环。RGB-D + LiDAR 只通过统一 `ObjectState` 向控制层提供结果。

### T600.5 受限 Residual

Residual 固定为末端 6D 小位姿、夹持力、base height 和 base pitch 共 9D，并经过物理缩放、幅值/变化率限制和 WBC/QP 安全投影。仅 `accepted=true` 的 SHA-pinned checkpoint 可晋级。

### T600.6 联合验收

完成 truth/多模态、position/contact-aware、nominal/residual 以及 Residual 输入消融。Residual 必须提高随机化鲁棒性，同时保留固定条件 30/30 安全基线。
