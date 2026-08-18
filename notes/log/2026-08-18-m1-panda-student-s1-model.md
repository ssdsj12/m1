# 2026-08-18 M1 + Panda Student S1 时序模型

## Purpose

执行 T400.9 / Student S1 Task 3，实现显式历史缓冲、GRU 估计器和 23 维安全残差 actor。

## Evidence

- Valid RED: missing `student_model` module caused collection failure，exit `2`。
- Focused GREEN: `9 passed`。
- Student Tasks 1–3 combined: `21 passed`，exit `0`。
- History is explicit `[E,10,100]`; append rolls by one frame and selected-environment reset does not affect peers。
- Network uses `GRU(100,128)`, estimator `128→38` split into `W_hat[6] + latent[32]`, and separate `128→1` safety logit。
- Actor input is exact `100 + 128 + 6 + 32 = 266`; MLP `266→256→128→23` exposes raw logits and bounded `tanh` action。
- Wrong history/observation width, non-finite values, dtype and device mismatch are rejected。
- Gradients reach GRU, estimator and actor; strict state-dict round trip reproduces every output exactly。

## Result

Task 3 passes as pure PyTorch. Model and history are ready for supervised DAgger losses/replay; no dataset or simulator rollout exists yet.

## Links

- Baseline: `688f3fd`
- [model](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_model.py)
- [tests](../../Go2Pvcnn/tests/test_m1_panda_student_model.py)
- [Student plan](../../docs/superpowers/plans/2026-08-18-m1-panda-dagger-student-s1.md)
