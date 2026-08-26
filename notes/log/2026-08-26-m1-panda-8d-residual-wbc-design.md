# 2026-08-26 M1 + Panda 8D Residual WBC 第一版设计

## Result

用户选择 PDF 推荐的第一版，并批准采用与旧 23D 基线并行的独立 8D 控制链。书面规格覆盖 Phase 1–4：8D contract、6D virtual wrench 到既有 WBC/QP、height/stance 和安装点六维力反馈；Arm MPC 与 PPO 长训不在本阶段。

## Evidence

- 输入文档：`/home/xk/下载/M1 + Panda 协调控制系统优化修改书.pdf`
- 设计规格：`docs/superpowers/specs/2026-08-26-m1-panda-8d-residual-wbc-design.md`
- 既有基线保持：103/23 Coordinated PPO、Folded Load、C0/C1a、旧 checkpoint。
- 本记录仅为设计证据；未修改运行代码，未运行测试或 GPU smoke。

## Next

用户复核书面规格后，编写逐文件单代理 TDD 实施计划，再开始代码修改。
