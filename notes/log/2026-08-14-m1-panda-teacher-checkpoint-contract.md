# M1 + Panda Teacher Checkpoint Contract

## Purpose

实现 A0/A1 checkpoint、相邻 run manifest、严格 frozen ActorCritic 和 SHA-256 契约，阻止不兼容的旧观察栈进入 A1。

## Stage

T400.5b Task 2 / checkpoint and resume boundary。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## RED Evidence

1. 原子 manifest/hash 测试因缺少 checkpoint 模块而收集失败。
2. stage/shape/base-hash 测试因缺少 `module_sha256` 而收集失败。
3. frozen actor/run manifest 测试因缺少 `build_run_manifest` 而收集失败。

## GREEN Evidence

- 原子 IO/hash：`9 passed`。
- manifest + 实际 actor/critic/std tensor shape：`25 passed`。
- strict frozen actor + 完整 run manifest：`33 passed`。
- Task 1+2+composer 回归：`118 passed in 1.22s`。
- checkpoint 模块与测试 `py_compile`：exit `0`。

## Contracts Verified

- schema `1`、stage、60 observation、16 action、hidden `[256,128]` 同时由 manifest 和 state dict 验证。
- actor/critic 各层与 `std` 的实际 shape 全部检查，非有限 tensor 失败。
- resume 可要求 optimizer state。
- A1 resume 可要求 exact `base_checkpoint_sha256`。
- frozen A0 以 strict state load、`eval()`、`requires_grad=False` 构造。
- module hash 在 inference 后稳定，在任一参数修改后变化。
- manifest 使用同目录临时文件、fsync 和 atomic replace；失败时不遗留临时文件。

## Result

通过。572/586 维 checkpoint 即使伪造 60 维 manifest，也会被 `actor.0.weight`/critic shape gate 拒绝。

## Limitations

本任务未接入训练脚本或 runner；当前阶段 checkpoint validator 由 Task 6 调用。

## Follow-up

执行 Task 3 reward/env cfg/Gym wiring。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable
- Key Files:
  - [Checkpoint contract](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py)
  - [Tests](../../Go2Pvcnn/tests/test_m1_panda_teacher_checkpoint.py)
  - [Plan](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-a0-a1-training.md)
