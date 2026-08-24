# 2026-08-24 M1 + Panda Coordinated PPO 专用配置

## Purpose

为 200 Hz coordinated 训练建立独立 PPO 配置，避免继续从 A0/A1 Teacher 配置复制后在入口临时覆盖。

## Stage

T400.10a / implementation Task 1。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md)

## Procedure

- 新增精确配置测试并观察缺失导出的有效 RED。
- 新建 `get_m1_panda_coordinated_train_cfg()` 并从 `agent` 导出。
- 运行 focused GREEN、既有 coordinated/Teacher train static 回归、py_compile 和 diff check。

## Key Metrics

- RED：`2 failed`，原因均为缺失 `get_m1_panda_coordinated_train_cfg`。
- GREEN：新配置 `2 passed`。
- 相关训练入口回归：`40 passed`。
- py_compile：exit `0`。
- `git diff --check`：exit `0`。

## Result

通过。配置冻结 `num_steps_per_env=256`、`gamma=0.9995`、`lambda=0.995`、adaptive KL `0.01`、LR `[1e-6,3e-4]`、physical std `[0.005,0.05]`，并保证每次调用返回独立对象。尚未接入训练入口或修改 PPO 实现。

## Git Refs

- Baseline Ref: `cba86e3`
- Candidate Ref: Task 1 commit containing this log
- Key Files: `Go2Pvcnn/agent/m1_panda_coordinated_train_cfg.py`, `Go2Pvcnn/tests/test_m1_panda_coordinated_train_cfg.py`
