# 2026-08-24 M1 + Panda bounded adaptive PPO

## Purpose

让 coordinated 配置中的 adaptive KL 真正受控生效，并把学习率和物理 action std 限制在批准边界内。

## Stage

T400.10a / implementation Task 2。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md)

## Procedure

- 为 LR 调节、上下界、physical std clamp、非法边界和 runner TensorBoard 标签编写测试。
- 观察旧 PPO 拒绝新配置且缺少诊断标签的 RED。
- 新增通用 bounded adaptation/helper，替换旧硬编码 `1e-5/1e-2`，并把 std clamp 扩展为 min/max。
- 运行 focused、Teacher std、runner checkpoint、train cfg/static 回归与 compile/diff check。

## Key Metrics

- RED：`7 failed, 7 passed`。
- 首次 GREEN：`16 passed`。
- 最终相关回归：`58 passed`。
- py_compile / `git diff --check`：exit `0`。

## Result

通过。PPO 新增默认向后兼容的 LR bounds，并为 coordinated 使用 `[1e-6,3e-4]`；`_adapt_learning_rate()` 记录 finite KL 和 increase/hold/decrease，`_clamp_policy_std()` 在实际 std 单位执行 `[0.005,0.05]`。runner 写入 `Loss/kl` 与 `Loss/lr_adjustment`。尚未加入 iteration callback 或训练 guard。

## Git Refs

- Baseline Ref: `a84a902`
- Candidate Ref: Task 2 commit containing this log
- Key Files: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py`, `Go2Pvcnn/tests/test_rsl_ppo_adaptive_schedule.py`
