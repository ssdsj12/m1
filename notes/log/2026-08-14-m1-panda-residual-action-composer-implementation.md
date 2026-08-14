# M1 + Panda Residual Action Composer Implementation

## Purpose

实现可供后续 Teacher、Student、评估和部署适配层共同复用的 16 维 M1 受限残差动作组合器。

## Stage

T400 follow-on control foundation / pure PyTorch implementation。

## Related Todo

- [T400 M1 + Panda 六轴力感知 Teacher–Student](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Files

- [Composer](../../Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py)
- [Focused tests](../../Go2Pvcnn/tests/test_m1_residual_action.py)
- [Approved design](../../docs/superpowers/specs/2026-08-14-m1-panda-residual-action-composer-design.md)
- [Implementation plan](../../docs/superpowers/plans/2026-08-14-m1-panda-residual-action-composer.md)

## RED Evidence

1. Task 1：仅创建配置/状态测试后，pytest 在收集阶段以 `ModuleNotFoundError: No module named 'go2_pvcnn.tasks.m1_residual_action'` 失败。
2. Task 2：添加动作合成测试后，目标 subset 为 `5 failed, 13 deselected`，五项均因缺少 `compose()` 失败。
3. Task 3：添加 reset/原子失败测试后，目标 subset 为 `7 failed, 9 passed, 18 deselected`；七项均因缺少 `reset()` 失败，已有 compose 原子校验保持通过。

## GREEN Evidence

1. Task 1：配置默认值、非法配置、构造状态和克隆诊断 `13 passed`。
2. Task 2：加入物理映射、幅值/变化率限制、诊断和梯度后，focused file `18 passed`。
3. Task 3：加入完整/选择性 reset 与索引校验后，focused file `34 passed`，`py_compile` exit `0`。

统一测试解释器为既有 T400 authority：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest
```

当前 base Python 没有 pytest；这属于 shell 环境差异，没有安装或修改依赖。

## Regression Evidence

最终组合回归覆盖 residual composer、M1 asset、M1 + Panda asset、combined smoke cfg、mount wrench 和 wrench probe static：

```text
89 passed in 1.18s
```

生产模块与测试的 `py_compile` exit `0`；占位符扫描无匹配。实现前同一 foundation subset 基线为 `55 passed in 0.85s`。

## Result

实现完成：

- 网络残差逐元素裁剪到 `[-1, 1]`；
- 前 12 维映射为腿关节位置残差，后 4 维映射为轮关节速度残差；
- 物理幅值和每周期变化率分别受配置约束；
- 每个环境保存独立、detached 的历史残差；
- 支持全量或选择性 reset；
- shape、dtype、device、非有限值和 reset 索引错误在状态更新前失败；
- 当前步保留梯度，诊断属性返回克隆；
- 基础动作原样参与相加，不在组合器内裁剪。

## Limitations

本阶段没有加载 checkpoint、接入 Isaac Lab 环境、训练 Teacher/Student、运行 Isaac Sim dynamics、实现 Panda IK/OSC，或建立实机安全限值。已批准的默认边界仅是仿真初始值，不构成实机安全认证。

## Follow-up

下一软件阶段为 Teacher 随机六维扰动平衡基线的独立设计与计划。T400.3 最坏工况机械验算继续作为最大载荷实机测试前置门。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: unavailable
- Git Ref: unavailable（`/home/xk/coding/M1` 不是 Git 工作树）
