# M1 + Panda 受限残差动作组合器设计记录

## Purpose

把总设计中的 12 腿位置加 4 轮速受限平衡残差拆成可独立实施的纯 PyTorch 接口，并在实现前锁定单位、状态、异常和测试契约。

## Stage

T400 follow-on control foundation / design。

## Related Todo

- [T400 M1 + Panda 六轴力感知 Teacher–Student](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Procedure

1. 核对已完成的 asset/wrench foundation 和总设计实施顺序。
2. 用户选择先实现独立残差接口，不同时加载 checkpoint 或实现 Teacher/Student、IK/OSC。
3. 比较纯 PyTorch 组合器、自定义 Isaac Lab `ActionTerm` 和 RSL-RL wrapper 三种落点。
4. 用户批准纯 PyTorch 方案、逐环境状态、归一化网络输出和物理单位安全边界。
5. 分节确认架构、异常处理、梯度语义和验收范围。

## Result

设计通过对话审批并写入独立规格。首版默认值为腿部 `0.05 rad` 幅值和 `0.01 rad/step` 变化率、轮部 `1.0 rad/s` 幅值和 `0.2 rad/s/step` 变化率；控制周期基于当前 M1 smoke 配置的 20 ms。组合器只限制残差，不裁剪基础动作。

## Verification

- 规格占位符、矛盾、范围和歧义自检已写入文档。
- 本阶段是设计记录，未运行仿真或代码测试。

## Follow-up

用户复核书面规格后，使用 `writing-plans` 编写 TDD 实施计划。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: unavailable
- Git Ref: unavailable（`/home/xk/coding/M1` 不是 Git 工作树）
- Key Files:
  - [Residual composer design](../../docs/superpowers/specs/2026-08-14-m1-panda-residual-action-composer-design.md)
  - [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
