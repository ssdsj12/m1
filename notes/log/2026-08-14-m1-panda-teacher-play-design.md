# 2026-08-14 M1 + Panda Teacher A0/A1 Play 设计

## Purpose

为已完成的 Teacher A0/A1 训练链补充独立、严格匹配训练 wrapper 的 play 入口设计，解决通用 `m1_play.py` 无法复现 A1 frozen-A0 + residual 两级动作链的问题。

## Decision

- 用户批准采用独立 `m1_panda_teacher_play.py`。
- GUI 默认，`--steps 0` 运行到窗口关闭；正 steps 支持 headless smoke。
- 六维扰动默认开启；仅 `--disable-disturbance` 使用显式零 wrench 对照。
- A1 同时要求当前 A1 checkpoint 和 A0 base checkpoint，并严格检查 manifest、60/16 shape 与 base SHA。
- 不训练、不写 checkpoint/manifest，不把当前未验收 A1 描述为部署策略。

## Alternatives Rejected

- 扩展通用 `m1_play.py`：其 wrapper/observation/checkpoint 路径不同，容易形成看似能运行但语义错误的 A1 推理。
- 在训练入口加入 play mode：混合训练与纯推理生命周期，增加误写和参数歧义。

## Design Artifact

- [书面规格](../../docs/superpowers/specs/2026-08-14-m1-panda-teacher-play-design.md)

## Self-review

- 范围检查：只覆盖 Teacher A0/A1 推理、扰动开关、诊断、测试与文档。
- 契约检查：明确默认扰动开启、A1 双 checkpoint、60/16 hard gate、无 optimizer/写盘。
- 歧义检查：明确 `--steps 0`、GUI/headless、零扰动 scheduler 行为与退出清理。
- 状态检查：规格等待用户书面复核；尚未创建实施计划或修改运行代码。

## Verification

文档阶段只执行占位符、链接目标和关键契约扫描；未运行 Isaac Sim。

## Git Refs

- Repository state correction at implementation handoff: Git worktree on `main`
- Base HEAD: `8872421d02eb93b04b150d025148c8a93e78dd09`
- Current Work Ref: uncommitted working tree
