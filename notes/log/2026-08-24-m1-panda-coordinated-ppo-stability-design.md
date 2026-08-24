# 2026-08-24 M1 + Panda Coordinated PPO 稳定长训设计

## Purpose

记录 long v4 后期策略坍塌后的已批准重训设计，在任何运行代码修改或 checkpoint 删除前冻结范围和验收门。

## Stage

T400.10a / Coordinated Teacher PPO stability redesign。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [written specification](../../docs/superpowers/specs/2026-08-24-m1-panda-coordinated-ppo-stability-design.md)

## Input Conditions

- long v4 已完成 5000 updates；第 4022 次开始 base contact，第 4256 次附近 `base_contact=1.0`，最终策略不可接受。
- 用户确认新正式运行从零开始；`model_3500.pt` 仅保留为旧基准。
- 用户批准自动早停并回退、保持 200 Hz 策略、256-step rollout、真实 Panda hand 外力和初始状态域随机化。

## Frozen Design

- `256 / gamma 0.9995 / lambda 0.995`；adaptive KL target `0.01`，学习率范围 `[1e-6, 3e-4]`。
- fresh zero-output actor，实际 std 从 `0.01` 开始并限制 `[0.005, 0.05]`。
- 100 completed episodes 后按 hard-failure、timeout、task score、reward 字典序选 best。
- eligible 后 25-update catastrophe 或 50-update patience 自动停止并回退；最多 600 updates。
- Panda hand 六维真实外力从 10% 课程升到 `20 N / 5 Nm` 满幅；reset/摩擦按规格随机化。
- 精确删除旧 long v4 中编号大于 3500 的 checkpoint，并生成 SHA 审计；清理在规格复核和实施阶段执行。

## Result

交互设计已确认，书面规格已生成并完成初步自审。当前没有运行代码修改、测试运行、GPU 训练或 checkpoint 删除；下一门是用户复核书面规格，然后编写逐文件单代理 TDD 实施计划。

## Git Refs

- Baseline Ref: `b472501`
- Candidate Ref: design commit containing this log
- Key Files: `docs/superpowers/specs/2026-08-24-m1-panda-coordinated-ppo-stability-design.md`
