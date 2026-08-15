# 2026-08-14 M1 + Panda Teacher Play Wrapper 扰动开关

## Scope

为 `M1PandaTeacherEnvWrapper` 增加默认开启、显式关闭的扰动 gate；不改变 A0/A1 动作组合、60/16 维度或训练默认行为。

## RED

首次 pytest 启动被系统 ROS `launch_testing` 自动插件污染阻断：go2 Python 3.11 加载 `/opt/ros/humble/lib/python3.10` 插件后缺少 `lark`。根因确认后使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 隔离非项目插件。

有效 RED：`tests/test_m1_panda_teacher_wrapper.py` 为 `5 failed, 19 passed`。失败原因均是 `disturbance_enabled` 参数/属性尚不存在。

## GREEN

- constructor 新增 strict bool `disturbance_enabled=True`。
- 默认路径仍推进 stage scheduler 并在 physics step 前施加 wrench。
- disabled 路径调用既有 `clear_external_wrench`，不推进 scheduler。
- disabled 的 effective `current_wrench_b` 与 `max_abs_wrench_seen` 保持零。
- A1 frozen actor 与双 composer 在 disabled 模式保持原动作链。

## Verification

```text
wrapper focused: 24 passed
wrapper + disturbance + residual composer: 109 passed
py_compile: exit 0
```

## Git Refs

- Branch: `main`
- Base HEAD observed at final handoff: `8872421d02eb93b04b150d025148c8a93e78dd09`
- Current Work Ref: uncommitted working tree
