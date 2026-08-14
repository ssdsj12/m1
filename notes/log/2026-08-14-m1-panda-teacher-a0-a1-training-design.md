# M1 + Panda Teacher A0/A1 Training Design

## Purpose

记录经用户分节批准的 Teacher A0/A1 平衡训练专项设计，并确认它与现有 M1 + Panda 资产、60 维 observation、安装点 wrench、16 维 residual composer 和 RSL-RL 基础配置一致。

## Stage

T400.5 / privileged Teacher / random six-dimensional disturbance / A0 zero-base / A1 frozen-base checkpoint。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Procedure

1. 读取仓库强约束、T400 branch、总设计和已完成 foundation/composer 证据。
2. 核对现有 `M1PandaSmokeObservationsCfg`、`M1ResidualActionComposer`、mount-wrench adapter、Gym 注册和 M1 PPO 配置。
3. 将用户批准的架构、扰动、奖励、checkpoint、reset、故障和验收边界写入专项 spec。
4. 扫描 `TBD`、`TODO`、`FIXME`、`待定`、`待确认`。
5. 检查 A0/A1 维度、旧 checkpoint 排除边界、短程验证与长期收敛声明是否一致。
6. 检查 Git 工作树状态。

## Input Conditions

- 用户批准 A0 零基础动作与小幅准静态扰动。
- 用户批准 A1 冻结 A0 60→16 checkpoint 并训练第二级 residual。
- Teacher observation 为 60 维，action 为 16 维。
- A0/A1 都只控制 M1，Panda 固定。
- 当前机器的长期 GPU 收敛受 `sm_120` 与 PyTorch 最高 `sm_90` 不兼容限制。

## Key Metrics

- 专项设计：296 行。
- 观测维度：`3 + 3 + 16 + 16 + 16 + 6 = 60`。
- 动作维度：12 腿位置 + 4 轮速 = 16。
- A0 扰动力/力矩满幅：每轴 `±10 N` / `±2 Nm`。
- A1 扰动力/力矩满幅：每轴 `±20 N` / `±5 Nm`。
- 占位符扫描：无匹配。
- Git：`/home/xk/coding/M1` 不是 Git 工作树。

## Result

设计自审通过。文档明确了：A0 与 A1 的两个 Gym ID、60 维 Teacher observation、两级 composer 时序、逐环境 reset、BASE_LINK-frame 扰动、奖励与终止、严格 checkpoint/manifest 契约、A0→A1 CPU smoke 和四类训练/恢复命令。

旧 572/586 维 PVCNN checkpoint 被明确排除，避免把不兼容观察栈静默接入驻停 Teacher。短程 CPU smoke 与长期策略收敛的证据边界也已分开。

## Conclusion

书面规格可以进入用户 review gate。用户批准书面 spec 后，下一步只能进入 `writing-plans`，尚未修改训练运行时代码或执行仿真。

## Follow-up

请用户审阅 [Teacher A0/A1 专项设计](../../docs/superpowers/specs/2026-08-14-m1-panda-teacher-a0-a1-training-design.md)。获批后编写逐文件 TDD 实施计划。

## Git Refs

- Baseline Ref: unavailable（目录不是 Git 工作树）
- Candidate Ref: filesystem working copy
- Git Ref: unavailable
- Key Files:
  - [专项设计](../../docs/superpowers/specs/2026-08-14-m1-panda-teacher-a0-a1-training-design.md)
  - [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
  - [Combined smoke cfg](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py)
  - [Residual composer](../../Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py)
  - [Mount wrench](../../Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py)
