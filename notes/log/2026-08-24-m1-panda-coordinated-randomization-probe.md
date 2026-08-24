# 2026-08-24 M1 + Panda coordinated randomization probe

## Purpose

在 GPU0/Isaac Sim 5.1 中验证训练 reset/friction DR、seed 复现、selective reset 隔离和 Panda-hand wrench 到 mount 反力的真实物理链。

## Stage

T400.10a / implementation Task 8 probe。

## Procedure

- 8 environments，seed 42，200 Hz，256 steps，GPU0 headless。
- 同一 seed 重置两批完整 state，并比较 root/joint tensors。
- 只重置 env0，验证其他 env state 完全不变。
- 检查 root/joint/soft-limit/material 实际值边界。
- wrapper 在 `panda_hand` 写逐环境 wrench，读取真实 `mount_wrench_b` 响应。

## Key Metrics

- exit `0`，`hard_gates_passed=true`。
- same-seed match / cross-env diversity / selected-reset isolation：全部 true。
- friction 实测 `[0.806481, 1.192450]`。
- root XY max `0.019408 m`；orientation max `0.042284 rad`。
- leg/arm offset max `0.019914/0.029028 rad`。
- controlled joint velocity offset max `0.049825 rad/s`；Isaac 5.1 对这些旧 actuator 配置报告 zero soft-limit metadata，事件与 probe 仅对正且有限的 soft limit 裁剪，并继续由批准的 `±0.05 rad/s` 硬边界约束。
- applied force/torque norm max `2.686332 N / 0.798839 Nm`。
- mount wrench response max `620.569458`，finite 且 nonzero。
- 256 steps：reset/contact/bad-orientation/non-finite 全部 `0`。

## Result

通过。训练随机化在 GPU0 实际 PhysX 中满足批准边界，Panda 手端外力经机械链产生 mount 反作用；该结果是基础物理/连线验收，不代表策略收敛或硬件载荷验收。

训练入口回归另外确认首次 runner rollout 前显式执行全环境 reset；TensorBoard 记录 `joint_reset_diagnostics_sampled=1`、position max `0.029028 rad`、velocity max `0.049825 rad/s`。这修复了此前构造器把物理演化状态误记为 reset DR 的诊断污染。

## Artifact

- `Go2Pvcnn/tests/artifacts/m1_panda_coordinated_randomization_probe.json`
