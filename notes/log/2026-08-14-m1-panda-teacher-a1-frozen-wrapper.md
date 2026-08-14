# M1 + Panda Teacher A1 Frozen Wrapper

## Purpose

实现 A1 对 A0 actor 的严格冻结接入、双层受限残差合成、逐环境 reset 和参数哈希漂移检测。

## Stage

T400.5b Task 5 / A1 frozen actor and two-level residual composition。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## RED Evidence

- 原有 A0 回归保持 `11 passed`。
- 有效 A1 RED 为 `8 failed`，均因 wrapper 明确抛出 `NotImplementedError`，覆盖 frozen actor 构造、推理输出、双 composer、选择性 reset 与 hash 契约。
- 首轮 GREEN 的唯一失败是测试算术期望错误：`-0.25 * 0.05 = -0.0125 rad`，不是实现缺陷；修正精确目标后通过。

## GREEN Evidence

- 完整 wrapper：`19 passed`。
- wrapper + checkpoint + disturbance + composer：`137 passed in 1.32s`。
- wrapper/test `py_compile`：exit `0`。

## Contracts Verified

- A1 只接受 `eval()`、所有参数 `requires_grad=False` 且实现 `act_inference` 的 `torch.nn.Module`。
- frozen actor 接收缓存的当前 60 维观测，并在 `torch.no_grad()` 内产生 16 维 base residual。
- base residual 与 trainable residual 分别由独立 composer 限幅、限速后顺序合成。
- actor 输出 shape/dtype/device/finite 在任何 wrapper 状态变化前验证。
- done/reset 同时清除两个 composer；未结束环境保留历史。
- 初始 actor SHA-256 被固定记录；正常 step 不改变 hash，参数漂移会报告 initial/current hash 并失败。

## Result

通过。A1 frozen chain 的纯行为边界已锁定；下一步接入 PPO 配置、训练入口和 checkpoint/resume 流程。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable
- Key Files:
  - [Wrapper](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py)
  - [Tests](../../Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py)
