# 2026-08-26 M1 + Panda 8D Residual WBC 实施计划

## Result

基于已复核的 T400.12 书面规格，完成八任务单代理 TDD 实施计划。计划从 8D 状态化 contract 开始，依次接入 WBC/QP、连续底盘参与、安全监督、runtime observation、独立 wrapper/play，最后执行 CPU 回归和 GPU0 逐轴 smoke。

## Scope

- 保留旧 103/23、C0/C1a、Folded Load 和 checkpoint。
- 第一版只覆盖 Phase 1–4。
- 不实现 Arm MPC、PPO 长训、抓取或实机部署。

## Next

用户已明确选择单代理并确认开始编码；采用 Inline Execution，按计划 Task 1 开始 RED→GREEN。
