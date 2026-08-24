# 2026-08-24 M1 + Panda hand wrench curriculum

## Purpose

让 coordinated 训练在 Panda 末端接受可复现、逐环境独立的真实六维外力课程，并向 runner 暴露域随机化诊断。

## Stage

T400.10a / implementation Task 6。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md)

## Procedure

- 先冻结配置、seed 重现性、逐环境独立性、三种 mode duty、课程进度与 selective reset 测试。
- 观察 scheduler 模块缺失 RED。
- 实现 device-local generator 和 continuous/pulse/intermittent 分段调度。
- wrapper 仅在显式开启时解析 Panda hand、在 physics step 前写 wrench，并在 done 后只 reset 对应 row。
- 记录 wrench 与 root/joint reset deviation 的 finite Python scalar diagnostics。

## Key Metrics

- RED：scheduler 模块缺失，测试收集失败。
- scheduler GREEN：`7 passed`。
- scheduler + wrapper GREEN：`11 passed`。
- Teacher/coordinated regression：`75 passed`。
- py_compile / `git diff --check`：exit `0`。

## Result

通过。课程严格使用 `20 N/5 Nm`、`0.25–1.0 s`、`0.10→1.0/50,000 steps` 与 `0.50/0.30/0.20` mode 概率；pulse 和 intermittent 均为 20% duty，后者按 0.25 s 周期避免 200 Hz Bernoulli 抖动。默认 wrapper 不创建 scheduler、不写或清除外力。

## Git Refs

- Baseline Ref: `7d9b807`
- Candidate Ref: Task 6 commit containing this log
- Key Files: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_disturbance.py`, `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_wrapper.py`
