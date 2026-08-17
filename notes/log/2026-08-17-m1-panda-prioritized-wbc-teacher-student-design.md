# M1 + Panda 优先级 WBC Teacher–Student 设计

## Purpose

记录用户批准的论文方法迁移设计：在 M1 轮式移动过程中同时控制 Panda 操作，并保持最初定义的六维力感知动态抗扰平衡。

## Stage

T400.8 / design

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Input Conditions

- 参考论文：Xie et al., *Prioritized Multi-task Motion Coordination of Physically Constrained Quadruped Manipulators*, DOI 10.34133/cbsystems.0203。
- 用户选择边移动边操作、轮式移动加腿部姿态调节。
- 用户选择论文 WBC 作为确定性 Teacher，后续蒸馏 Student。
- 用户确认动态抗扰平衡必须保留并高于末端轨迹跟踪。

## Procedure

1. 读取并结构化论文的运动分配、约束、奇异规避、WBC/QP和实验章节。
2. 核对当前 M1+Panda 25-DOF articulation、16 维 M1-only 动作、Panda 隐式驱动器、`sim.dt=0.005 s` 和 `decimation=4`。
3. 比较完整 WBC、论文分配器加现有 A1、WBC Teacher–Student 三条路线。
4. 分四部分完成架构、优先级、Teacher–Student 课程、代码与验证设计，并逐段获得用户批准。
5. 写入并自审正式规格。

## Result

设计批准并写入 [正式规格](../../docs/superpowers/specs/2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md)。采用 50 Hz 运动分配、200 Hz 滚轮式 WBC/QP、23 维全身协调动作、安装六维力反馈和动态平衡最高优先级。现有 A0/A1 保持独立，仅作为诊断基础和行为对照。

## Verification

- 规格占位符和必需章节扫描：pass。
- 本次新增文件及新增目标的相对链接检查：pass。
- scoped `git diff --check`：pass。
- 全量笔记链接扫描仍报告两个既有无关缺口：`notes/todo.md` 的旧 HTML 目标和 `notes/log/index.md` 的日志模板占位目标；本次不扩展范围修复。
- 未修改运行代码。
- 未运行 Isaac Sim、训练或行为评估。

## Conclusion

下一步是用户复核书面规格；批准后使用 writing-plans 编写逐文件 TDD 实施计划。不得从本日志推断论文 WBC 已实现或当前 checkpoint 可用于新 Student。

## Follow-up

- 用户书面复核规格。
- 复核通过后生成实施计划。
- 实施前协调 T400.6 零间隙资产冻结点和 T400.3 实机机械验算门。

## Git Refs

- Baseline Ref: `13dc83e`
- Candidate Ref: `9effc43`
- Key Files:
  - [设计规格](../../docs/superpowers/specs/2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md)
  - [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
