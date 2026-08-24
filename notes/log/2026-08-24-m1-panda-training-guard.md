# 2026-08-24 M1 + Panda coordinated training guard

## Purpose

为稳定重训增加严格的 100-episode 最佳模型判定、灾难/耐心/上限停止条件，以及可审计的原子 checkpoint 回退链。

## Stage

T400.10a / implementation Task 4。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md)

## Procedure

- 先覆盖 100 episode 门、字典序、早期 tie、finite gate、eligible/diagnostic 分离及三种停止原因。
- 观察模块缺失 RED。
- 实现纯 `TrainingGuard`，每项指标使用同一最近 100 episode 窗口。
- 用同目录临时文件与 `os.replace` 原子发布 best checkpoint、JSON 和 rollback final。
- 以真实 `torch.save` fake runner 验证 load flags、SHA 和回退后的参数。

## Key Metrics

- RED：测试收集失败，guard 模块不存在。
- focused GREEN：`8 passed`。
- Task 1–4 相关回归：`23 passed`。
- py_compile / `git diff --check`：exit `0`。

## Result

通过。排名严格按 hard failure、timeout、task score、reward、较早 iteration；eligible 门为 timeout `>=0.90` 且两类失败各 `<=0.05`。首次 eligible 后启用 25-update catastrophe 和 50-update patience；上限默认 600。finalize 使用 `load_optimizer=False, keep_std=True` 回退，缺少 eligible 时明确标记 `completed_without_eligible_best`、`accepted=false`。

## Git Refs

- Baseline Ref: `ebe1126`
- Candidate Ref: Task 4 commit containing this log
- Key Files: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_training_guard.py`, `Go2Pvcnn/tests/test_m1_panda_coordinated_training_guard.py`
