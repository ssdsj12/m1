# 2026-08-23 M1 + Panda Coordinated 长时动态门

## Purpose

关闭 T400 Coordinated Teacher 正式长训前的 disabled Panda `root_joint` 长时 snap 风险门。

## Procedure

- Task: `Isaac-M1-Panda-Coordinated-v0`
- Device: GPU0 / RTX 5070 / `cuda:0`
- Isaac Sim / Isaac Lab: 5.1
- `hold`: 8 environments × 2000 physics steps，腿/臂保持 reset 姿态、轮目标速度为零。
- `controlled`: 8 environments × 2000 physics steps，保持底盘并给 Panda joint 0/3 施加 `0.05 rad`, `0.2 Hz` 带限关节目标。
- 两次均通过真实 23-effort action 进入环境；每步测量 `BASE_LINK→panda_link0` 相对位姿、六维 mount wrench、reset/termination、joint limits 和 finite。

## Evidence

- Static TDD: missing script RED `3 failed`; local-source precedence regression RED `1 failed, 2 passed`; final GREEN `3 passed`。
- `hold`: exit `0`, `hard_gates_passed=true`, position drift `2.4958834e-07 m`, orientation drift `5.3401686e-07 rad`, reset/base-contact/bad-orientation/joint-limit/non-finite all `0`。
- `controlled`: exit `0`, `hard_gates_passed=true`, position drift `2.5987634e-07 m`, orientation drift `5.1608117e-07 rad`, reset/base-contact/bad-orientation/joint-limit/non-finite all `0`。
- PhysX still prints the known disabled `Panda/root_joint` disjoint-transform warning, but no measured snap occurred in either 8×2000 run。
- Peak mount wrench was large at startup: hold `1761.72 N / 901.89 Nm`, controlled `2021.06 N / 887.38 Nm`. This is recorded as a transient/sensor-normalization follow-up and is not accepted as a real mechanical load claim。

## Result

The long-horizon rigid-mount dynamics gate passes for stationary hold and small Panda motion. Formal coordinated PPO training remains blocked by a separate learning-contract defect: the current 67-value observation omits Panda/task state and the inherited smoke rewards do not encode navigation or EE tracking.

## Artifacts

- `Go2Pvcnn/logs/m1_panda_coordinated/dynamics_gate_hold_8x2000.json`
- `Go2Pvcnn/logs/m1_panda_coordinated/dynamics_gate_controlled_8x2000.json`

## Git Refs

- Baseline: `7b2673d`
- Candidate: current dynamics-probe work
- Key files: `Go2Pvcnn/scripts/m1_panda_coordinated_dynamics_probe.py`, `Go2Pvcnn/tests/test_m1_panda_coordinated_dynamics_probe_static.py`
