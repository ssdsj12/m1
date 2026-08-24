# 2026-08-24 M1 + Panda runner iteration callback

## Purpose

为 coordinated 自动早停与最佳 checkpoint 控制器提供通用、不可变、与具体任务解耦的逐迭代接口。

## Stage

T400.10a / implementation Task 3。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md)

## Procedure

- 先为 frozen summary/result、episode/environment metric detach、finite gate 和可选 callback 写测试。
- 观察旧 runner 在测试收集阶段因接口缺失产生 RED。
- 在通用 runner 内收集每轮完成 episode reward/metric、KL/LR 和可选环境诊断。
- callback 返回非空原因时停止循环；普通调用不传 callback 时保持周期与最终 checkpoint 行为。

## Key Metrics

- RED：测试收集失败，缺少 `IterationSummary`。
- focused GREEN：`13 passed`。
- Teacher/coordinated 代表入口回归：`46 passed`。
- py_compile / `git diff --check`：exit `0`。

## Result

通过。`OnPolicyRunner.learn()` 现在返回 frozen `LearnResult`，可选 callback 接收 detached frozen `IterationSummary`；环境诊断以 finite scalar pair 复制并写入 `DomainRandomization/*`。通用 runner 未引入 M1/Panda 专用指标名。

## Git Refs

- Baseline Ref: `a8d9800`
- Candidate Ref: Task 3 commit containing this log
- Key Files: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`, `Go2Pvcnn/tests/test_rsl_runner_iteration_callback.py`
