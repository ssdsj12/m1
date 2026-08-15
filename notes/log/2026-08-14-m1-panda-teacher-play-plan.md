# 2026-08-14 M1 + Panda Teacher A0/A1 Play 实施计划

## Result

用户确认书面设计后，已生成三任务、单代理、TDD 实施计划：

1. wrapper 默认开启/显式关闭扰动；
2. strict A0/A1 专用 play 入口；
3. runbook、人机文档、全量静态回归与 GPU0 三段 smoke。

## Key Contracts

- 60 observation / 16 action 不变。
- A0 zero-base；A1 frozen A0 + A1 residual。
- GUI 与六维扰动默认开启；`--disable-disturbance` 保持策略链但清零外力。
- checkpoint/manifest/tensor/base SHA 全部在 runner load 和第一个 step 前验证。
- 不调用 learn，不写 checkpoint/manifest。
- 用户指定单代理 inline execution，不派子代理。

## Artifact

- [实施计划](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-play.md)

## Self-review

规格覆盖、类型一致性、执行顺序、禁止写盘、GPU0 验收和占位符均已逐项检查。执行期间最终确认 `/home/xk/coding/M1` 为 Git 工作树；Codex 未创建提交，测试证据仍同步写入 notes/log。

## Git Refs

- Branch: `main`
- Base HEAD observed at final handoff: `8872421d02eb93b04b150d025148c8a93e78dd09`
- Current Work Ref: uncommitted working tree
