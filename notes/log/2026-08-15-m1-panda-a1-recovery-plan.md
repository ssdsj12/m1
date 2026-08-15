# 2026-08-15 M1 + Panda A1 抗扰恢复实施计划

## Result

已把批准的 T400.7 规格转换为单代理、七任务 TDD 实施计划。计划覆盖 ActorCritic scalar/log std 语义、扰动课程恢复、满幅 JSON 评估、候选 sweep、独立 recovery fork、静态/真实 smoke 和 GPU0 500-iteration 分块验收。

## Artifact

- [实施计划](../../docs/superpowers/plans/2026-08-15-m1-panda-a1-recovery-training.md)
- [批准规格](../../docs/superpowers/specs/2026-08-15-m1-panda-a1-recovery-training-design.md)

## Execution Boundary

- 用户已要求后续使用单代理，不派生子代理。
- 计划阶段未修改 PPO、训练/Play 入口，也未启动 recovery run。
- 当前工作区已有 Teacher Play 等未提交变更；执行时必须逐文件检查并保护这些改动。

## State

计划等待用户确认执行方式；建议按用户既有偏好选择 inline execution。
