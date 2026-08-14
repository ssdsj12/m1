# M1 + Panda Teacher A0/A1 Static Regression

## Purpose

验证已批准 A0/A1 Teacher 实施的扰动、checkpoint、环境、wrapper、训练入口、smoke driver 与既有 M1/Panda foundation 回归。

## Result

通过。

## Evidence

- 计划指定 9 文件最终回归：`214 passed in 1.40s`。
- 7 个生产/入口文件 `py_compile`：exit `0`。
- runbook 与 `m1_panda_teacher*.py` placeholder scan：无匹配。
- `git -C /home/xk/coding/M1 rev-parse --is-inside-work-tree`：按既有仓库状态失败，未初始化 Git。

## Runtime Fixes Captured by Tests

- Gym 字符串 `env_cfg_entry_point` 必须通过 `isaaclab_tasks.utils.parse_env_cfg` 解析。
- `base_xy_drift_l2` reward 必须显式接收 `SceneEntityCfg("robot")`。
- resume 在 runner load 后把下一学习迭代设为 checkpoint iteration + 1，避免覆盖加载的 checkpoint。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Last Feature Commit: unavailable
- Last Verified Commit: unavailable
