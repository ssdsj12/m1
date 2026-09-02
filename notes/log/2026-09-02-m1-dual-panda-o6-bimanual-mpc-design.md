# M1 + 双 Panda + 双 O6 分层双手 MPC 设计

## Purpose

把用户提供的 O6 左右手资产与 M1 项目现有 Panda MPC/WBC 基础收敛为首个可实施、可验收的确定性双手操作设计。

## Stage

T500 design only。

## Related Todo

- [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)

## Procedure

- 读取 M1 仓库约束、todo、日志和现有单 Panda Arm MPC/WBC 设计与代码入口。
- 检查 O6 左右 URDF、关节/mimic 语义和完整 USD 分层目录。
- 核对官方 O6 主动/被动自由度、重量、速度和抓握能力参数。
- 与用户逐项确认机械拓扑、首个任务、感知边界、控制层级、平台结构、资产处理、安全与验收。
- 比较分层耦合、统一非线性和物体/手集中三种 MPC 路线。

## Input Conditions

- M1 项目现有单 Panda 25-DOF 组合资产和 50 Hz Arm MPC / 200 Hz WBC/QP。
- 用户提供左右 O6 完整 USD 分层和 URDF/STL。
- 固定箱体 `0.12 x 0.18 x 0.10 m`、`0.5 kg`。
- M1 首版原地，Isaac 真值状态，确定性模型控制。

## Key Metrics

- 43 个主动控制通道。
- 预计 53 个运行时物理 DOF，必须由 PXR/Isaac 探针确认。
- object/Arm/Hand/WBC 频率为 25/50/100/200 Hz。
- 抬升 `>=0.10 m`，保持 `>=3 s`，三 seed 共 30 次全部成功。

## Result

用户逐段批准分层 MPC、资产/环境、安全和验收设计。正式规格已写入 [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)。

## Conclusion

设计阶段完成；没有修改运行时代码、资产或训练状态。下一步必须等待用户复核书面规格，再编写实施计划。

## Follow-up

用户批准书面规格后调用 `writing-plans`，把 T500 拆成逐文件 TDD 实施任务。

## Git Refs

- Baseline Ref: `cd6cd58`
- Candidate Ref: design commit pending
- Key Files:
  - `docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md`
  - `notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md`
