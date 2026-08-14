# M1 + Panda Teacher A0 Wrapper

## Purpose

实现严格 60/16 RSL-RL boundary、A0 零基础 residual composition、每步 BASE_LINK wrench 写入和逐环境 reset。

## Stage

T400.5b Task 4 / A0 wrapper and force application order。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## RED Evidence

1. 基础 wrapper 测试因模块缺失而收集失败。
2. reset/finite 测试第一次运行先发现测试 fixture 的错误 cat 维度；修正测试后得到有效 RED：`3 failed, 8 passed`，失败分别为 constructor reset 未清外力、done 未清状态、显式 reset 未清外力。

## GREEN Evidence

- 基础 60/16/data-flow：`2 passed`。
- 完整 A0 wrapper：`11 passed`。
- wrapper+scheduler+composer：`96 passed in 1.14s`。
- wrapper/test `py_compile`：exit `0`。

## Contracts Verified

- PPO residual 在 A0 与显式 zero base 经 composer 合成。
- 首步 leg/wheel normalized output 分别为 `0.04` / `0.025`，与 physical slew/action scale 一致。
- live wrench 在底层 `env.step` 前写到 `panda_hand`。
- reset 与 done 均把零 wrench 重新写入 PhysX buffer。
- done 只清对应 composer/scheduler/published residual/diagnostic 行，未 done 环境保持状态。
- action shape/dtype/device/finite 在状态变更前验证。
- observation/reward/body/action-dim/stage/cfg 异常明确失败。
- critic extras 与 policy observation 完全相同，均为 `(N,60)`。

## Result

通过。A0 wrapper 的训练循环接口已具备纯行为证据；真实 ManagerBasedRLEnv 接入留给 CPU smoke。

## Follow-up

执行 A1 frozen actor/two-composer Task 5。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable
- Key Files:
  - [Wrapper](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py)
  - [Tests](../../Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py)
