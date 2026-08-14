# M1 + Panda Teacher Environment Wiring

## Purpose

实现 Teacher 专用稳定奖励、A0/A1 环境配置及两个 lazy Gym ID，同时保持已有 60 维观测和 16 维 M1-only action 契约。

## Stage

T400.5b Task 3 / reward, environment configuration, Gym registration。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## RED Evidence

1. 轻量 reward 测试因缺少 `m1_panda_teacher_rewards.py` 失败。
2. env/Gym 测试产生 `5 failed, 19 passed`：四项因 Teacher cfg 文件缺失，一项因 registry 实际 21 而期望 23。

## GREEN Evidence

- Teacher reward helper：`7 passed`。
- Teacher cfg + combined smoke registry：`24 passed`。
- 加入 M1 asset static 回归：`25 passed in 0.84s`。
- reward/cfg/registry/test `py_compile`：exit `0`。

## Contracts Verified

- 两个 Gym ID 为 `Isaac-M1-Panda-Teacher-A0-v0` 与 `...-A1-v0`，保持 string entry point lazy import。
- A0/A1 都继承 combined smoke 的六个 observation terms，总维度 60。
- 动作仍仅 12 腿位置 + 4 轮速；没有 Panda action term。
- rewards 明确覆盖 alive、0.60 m 高度、竖直速度、水平角速度、姿态、xy 漂移、轮速、trainable residual 幅值/变化率、M1 torque 和 feet slide。
- A0/A1 cfg 捕获完整 force/torque/time/curriculum/mode/pulse 字段。
- timeout/base-contact/bad-orientation 继续继承 smoke termination。

## Result

通过。由于轻量测试环境无法导入仓库大型 `mdp/rewards.py` 依赖链，五个 Teacher helper 被隔离到单一职责模块并通过 `mdp/__init__.py` 导出；cfg 对消费者仍使用统一 `mdp.*` namespace。

## Limitations

真实 Gym 创建与 manager term resolution 尚未运行，将在 Task 7 CPU smoke 验证。

## Follow-up

执行 A0 wrapper Task 4。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable
- Key Files:
  - [Reward helpers](../../Go2Pvcnn/go2_pvcnn/mdp/m1_panda_teacher_rewards.py)
  - [Environment cfg](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_env_cfg.py)
  - [Registry](../../Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py)
  - [Tests](../../Go2Pvcnn/tests/test_m1_panda_teacher_env_cfg_static.py)
