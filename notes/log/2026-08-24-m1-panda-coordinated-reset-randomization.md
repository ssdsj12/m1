# 2026-08-24 M1 + Panda coordinated reset randomization

## Purpose

增加仅由 coordinated 训练入口显式开启的 root、关节初态和摩擦域随机化，同时保持默认配置、Play 和确定性 probe 不随机。

## Stage

T400.10a / implementation Task 5。

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-24-m1-panda-coordinated-ppo-stability.md)

## Procedure

- 先为腿/Panda 独立位置范围、wheel 默认位置、受控速度、soft-limit clamp 和单次原子写入编写测试。
- 观察 missing event/helper RED：`3 failed, 2 errors, 4 passed`。
- 实现 canonical joint 精确解析、范围/finite 预检、完整默认 state clone 和一次 joint-state 写入。
- coordinated cfg 默认全部确定性；显式 helper 设置批准的 root、joint、friction/restitution 范围。

## Key Metrics

- RED：`3 failed, 2 errors, 4 passed`。
- focused GREEN：`9 passed`。
- coordinated cfg regression：`13 passed`。
- py_compile / `git diff --check`：exit `0`。

## Result

通过。训练范围为 root x/y `±0.02 m`、roll/pitch `±0.03 rad`、yaw `±0.05 rad`、线速度 `±0.05`、角速度 `±0.10`；腿/Panda 位置分别 `±0.02/±0.03 rad`，受控速度 `±0.05 rad/s`，摩擦 `[0.8,1.2]`、64 buckets、restitution `0`。关闭 helper 后恢复零扰动与 friction `1.0`。

## Git Refs

- Baseline Ref: `02d7031`
- Candidate Ref: Task 5 commit containing this log
- Key Files: `Go2Pvcnn/go2_pvcnn/mdp/events.py`, `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_env_cfg.py`
