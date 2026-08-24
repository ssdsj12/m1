# 2026-08-24 M1 + Panda stable train entrypoint

## Purpose

把专用 200 Hz PPO、训练域随机化、逐迭代 guard、原子 best 与自动 final rollback 串入 fresh coordinated 训练入口。

## Stage

T400.10a / implementation Task 7。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md)

## Procedure

- 重写入口静态契约并为纯 schema-2 manifest builder 编写精确字段测试。
- 观察旧 fixed/frozen/missing-controller 路径 RED：`4 failed, 2 passed`。
- 入口改用 coordinated cfg、显式 reset/friction DR、seeded wrench wrapper 与 guard callback。
- actor 末层保持精确零初始化，但 std 必须 trainable。
- 修正 Isaac Lab stale reset log：runner 只在实际 done step 收集 episode log，wrapper 将 reset 聚合量扩展为与完成 episode 数一致的 guard metrics。

## Key Metrics

- RED：`4 failed, 2 passed`。
- focused GREEN：`6 passed`。
- Tasks 1–7 related regression：`115 passed`。
- all coordinated tests：`53 passed`。
- py_compile / `git diff --check`：exit `0`。

## Result

通过。入口默认 `64 env / seed 42 / max 600`，拒绝超过 600；A1 checkpoint 仅记录 provenance，不加载 actor。schema 2 冻结 PPO/DR/guard 合同，训练停止后按 eligible best（否则 diagnostic best）回退并记录 source/final SHA 与 `accepted`。尚未进行 GPU0 物理 probe 或短训。

## Git Refs

- Baseline Ref: `05cf455`
- Candidate Ref: Task 7 commit containing this log
- Key Files: `Go2Pvcnn/scripts/m1_panda_coordinated_train.py`, `Go2Pvcnn/tests/test_m1_panda_coordinated_train_manifest.py`
