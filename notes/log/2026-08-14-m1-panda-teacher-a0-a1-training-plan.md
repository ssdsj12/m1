# M1 + Panda Teacher A0/A1 Training Plan

## Purpose

把已批准的 Teacher A0/A1 书面规格转换为可由单代理按 TDD 顺序执行的逐文件实施计划。

## Stage

T400.5a / implementation planning / privileged Teacher disturbance balance。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Procedure

1. 核对现有 M1 train entrypoint、RSL-RL wrapper/runner、Panda smoke cfg、mount wrench probe、reward helpers 和静态测试模式。
2. 将实现拆成七个独立验收任务：扰动、checkpoint、cfg、A0 wrapper、A1 frozen chain、训练入口、CPU smoke/runbook。
3. 为每个任务写出精确文件、接口、RED 测试、实现内容、GREEN 命令和笔记步骤。
4. 对照专项 spec 逐节检查覆盖。
5. 扫描占位符和省略号，并把所有简写替换为确切签名、配置、参数和命令。
6. 检查跨任务类型、函数名、维度、stage 和 hash 契约。

## Key Metrics

- 实施计划：7 个任务，约 950 行。
- A0/A1 observation/action：60/16。
- A1 checkpoint 双重门：manifest 维度/stage 与真实 state-dict shape。
- A1 frozen 双重门：`requires_grad=False` 且训练前后 module SHA-256 相同。
- CPU acceptance：A0 initial/resume、A1 initial/resume 四段链。
- Placeholder scan：clean。
- Git：`/home/xk/coding/M1` 不是 Git 工作树。

## Result

计划自审通过。专项 spec 的环境、扰动、动作、奖励、PPO、checkpoint、异常、测试和命令要求都有明确任务；旧 572/586 维 checkpoint 不能通过 A1 validation；每项运行时代码都有先 RED 后 GREEN 的步骤。

用户此前明确要求后续使用单代理，因此执行选项锁定为 `executing-plans` inline execution，不提供或启动 subagent 路径。

## Conclusion

计划可执行。下一步读取并使用 `executing-plans`，从 Task 1 开始，不跳过 RED 证据。

## Follow-up

执行 [Teacher A0/A1 implementation plan](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-a0-a1-training.md)。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable
- Key Files:
  - [Approved spec](../../docs/superpowers/specs/2026-08-14-m1-panda-teacher-a0-a1-training-design.md)
  - [Implementation plan](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-a0-a1-training.md)
  - [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
