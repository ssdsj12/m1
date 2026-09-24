# 2026-09-05 M1 + 右 Panda + 右 O6 工程路线图设计

## Purpose

把用户提供的《M1 + 双臂双灵巧手多模态感知 MPC + Residual 整体研究规划》转化为与当前仓库状态一致、可以逐阶段实施和验收的工程路线图。

## Stage

T600 design only。

## Related Todo

- [T600](../todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md)

## Procedure

- 提取并阅读用户提供的 Word 研究规划。
- 核对 T400 单 Panda WBC/Residual 路线和 T500 双 Panda/O6 路线。
- 检查主分支状态以及独立 T500 worktree 的已提交和未提交边界。
- 与用户逐项确认工程路线图定位、首个本体、固定任务、纯仿真边界、推进方式、系统架构、阶段门、9D Residual 和代码边界。
- 比较双基线纵向闭环、严格感知优先和控制/感知并行三种推进方式，采用双基线纵向闭环。

## Input Conditions

- M1 原地稳定。
- 右 Panda 7-DOF 和右 O6 6 主动 + 5 mimic。
- Isaac Lab 纯仿真。
- 固定 `0.5 kg` 箱体，唯一任务为抓取、抬升、保持、下降和释放。
- T400/T500 只提供经提交、SHA 和测试确认的可复用基础；拒绝 checkpoint 和 dirty worktree 不形成 lineage。

## Key Contracts

- 29 个主动控制通道；预计 34 物理 DOF，运行时确认优先。
- truth 和 RGB-D + LiDAR 使用同一 `ObjectState`。
- nominal contact-aware MPC 先通过固定三 seed 30/30，再启动 Residual。
- Residual 为受限 9D：末端 6D、夹持力、base height、base pitch。
- zero/pilot/short/promotion/conditional-long 逐级晋级，只有 `accepted=true` 可继承。

## Result

用户逐段批准四部分设计。正式路线图已写入[设计文档](../../docs/superpowers/specs/2026-09-05-m1-single-panda-o6-multimodal-mpc-residual-roadmap-design.md)。

## Conclusion

T600 设计阶段完成；没有修改运行时代码、资产或训练状态。路线明确先完成单臂单 O6 纵向闭环，再开启双手、移动操作、VLA 或真实硬件的新设计周期。

## Follow-up

用户复核书面路线图后，调用 `writing-plans`，首份计划只覆盖 T600.1 资产与接口基座。

## Git Refs

- Baseline Ref: `72dd8df`
- Candidate Ref: T600 design commit (this commit)
- Key Files:
  - `docs/superpowers/specs/2026-09-05-m1-single-panda-o6-multimodal-mpc-residual-roadmap-design.md`
  - `notes/todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md`
