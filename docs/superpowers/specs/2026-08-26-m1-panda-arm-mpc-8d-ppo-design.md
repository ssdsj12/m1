# M1 + Panda Phase 5–6 Arm MPC + 8D Residual PPO 设计

## 1. 目标与范围

在已验收的 Phase 1–4 `8D Residual WBC` 控制链上继续实现：

- Phase 5：Panda 机械臂 MPC 预测未来末端运动、关节参考和安装点六维 wrench；
- Phase 6：从零训练 8D Residual PPO，仅修正模型误差与未知扰动。

首轮训练任务严格限定为 `M1 原地保持平衡 + Panda 小幅六自由度末端运动`。本规格不包含 M1 rolling、payload、抓取、外界推力、完整导航操作任务、Teacher–Student 蒸馏或实机部署。现有 23D Coordinated PPO、Folded Load、C0/C1a 和 Phase 1–4 Play 必须继续可用。

## 2. 总体架构

控制数据流固定为：

```text
EE target
  -> 50 Hz Arm MPC
  -> future EE / q_ref / qd_ref / predicted mount wrench
state + MPC diagnostics
  -> 8D residual PPO correction
  -> safety projection
  -> 200 Hz whole-body WBC/QP
  -> 23 actuator efforts
```

Arm MPC 只预测和生成参考，不直接操作 Isaac、不直接写关节力矩、不负责 RL 或最终 safety。MPC 每 `20 ms` 重规划，使用 20 个预测节点覆盖 `0.4 s`；WBC/QP 每 `5 ms` 执行，并在两个 MPC 周期之间保持最近一次安全参考。PPO residual 先经过 safety projection，再作为 WBC/QP 的受限目标，不在 QP 求解后直接叠加 actuator effort。

最终 wrench 请求为：

```text
W_cmd = -W_arm_mpc + W_feedback + W_rl
```

其中 `W_arm_mpc` 是 Panda 对 M1 安装界面的预测作用 wrench，`W_feedback` 来自 Phase 4 六轴反馈，`W_rl` 是 PPO 的六维修正。另两维 residual 修正机身高度和站姿宽度。

## 3. Phase 5：线性化末端空间 Arm MPC

### 3.1 选择

采用确定性的线性化末端空间 MPC，不采用 nonlinear shooting MPC，也不训练学习式 wrench predictor。现有运行时已经提供 Panda Jacobian、质量矩阵、科氏力、重力、关节状态和末端状态，可建立可测试的模型式预测链。

### 3.2 输入

- Panda 当前 `q`、`qdot`；
- 当前 EE pose、twist；
- 未来小幅 `x/y/z/roll/pitch/yaw` EE 目标；
- Panda spatial Jacobian；
- articulation mass matrix、Coriolis/centrifugal force、gravity force；
- 关节 position、velocity、acceleration、effort soft limits；
- 上一安全解，用于 warm start 和 fallback。

所有纯控制接口使用有限 `torch.float64` 张量，明确 batch、dtype 和 device 合同；不得静默接受 shape、dtype、device 或非有限值错误。

### 3.3 输出

- 预测未来 EE pose/twist 序列；
- 当前 WBC 周期使用的 Panda `q_ref`、`qd_ref`；
- base frame、base origin 的预测安装点六维 wrench；
- `sigma_min`、最小/平均关节余量、EE tracking error；
- constraint saturation、iteration、feasible 和 fallback reason。

### 3.4 约束与 fallback

MPC 显式约束 Panda 关节位置、速度、加速度和 effort。预测 wrench 必须经过既有 canonical frame conversion，不能创建第二套符号或参考点约定。

任一输入、迭代中间量或输出非有限，或 MPC 不可行时：

1. 丢弃本周期新解；
2. 使用上一安全参考；
3. 若无上一安全参考，使用当前静态安全姿态；
4. 将预测前馈 wrench 置零；
5. 输出明确 fallback reason；
6. 连续失败交给安全监督器缩小 residual 或终止。

异常 MPC 输出不得进入 PPO observation、WBC objective 或执行器命令。

## 4. Phase 6：8D Residual PPO

### 4.1 动作合同

Actor 从零初始化，输出单个严格归一化动作：

```text
r in [-1, 1]^8
[Fx, Fy, Fz, Mx, My, Mz, delta_height, delta_stance]
```

动作继续通过 Phase 1 的 physical scale、slew limit 和 finite atomic validation。不得迁移旧 23D actor 权重，不改变旧 checkpoint shape。

### 4.2 103 维 observation

复用已冻结的 Residual Observation 合同：

| 分组 | 维度 | 内容 |
| --- | ---: | --- |
| M1 proprioception/joints | 59 | base、腿、轮、接触和相关本体状态 |
| Panda state | 20 | `q/qdot`、EE error、manipulability、joint margin |
| Interaction | 6 | filtered mount wrench |
| Task | 6 | EE target/error/twist 的冻结表示 |
| Stability | 4 | COM/support、姿态和安全余量 |
| Temporal | 8 | previous applied residual |
| 合计 | 103 | 严格固定 |

MPC feasible、fallback 或 prediction-error 诊断只在不改变 103D 冻结合同的已有槽位内表达；若无法无损表达，实施阶段必须停下更新规格，不得私自扩维。

### 4.3 多分支单头网络

- M1 本体分支：编码为 128；
- Panda 分支：编码为 64；
- wrench 分支：编码为 32；
- task/stability/temporal 分支：编码为 32；
- fusion：`256 -> 128 -> 8`；
- 第一版只使用一个 8D action head，不拆成 6D/2D 双头。

Critic 可复用相同分组边界但拥有独立参数，输出一个 value。Actor/Critic 必须支持 deterministic inference、严格 checkpoint manifest、finite validation 和旧 RSL-RL runner 的隔离接入。

## 5. 环境、奖励与训练课程

### 5.1 独立环境

新增独立 Gym ID，动作维度严格为 8，观测严格为 103。环境内部编排 Arm MPC、WBC/QP、residual composer 和 safety projection，最终仍向组合 articulation 写 23D effort。旧 Gym ID、旧 wrapper 和旧训练配置不得改变默认行为。

### 5.2 课程

1. **Phase 5 Gate**：residual 固定为零；验证 Arm MPC 的 EE 预测、关节参考、wrench 符号、finite 和 fallback。
2. **Phase 6 Stage 0**：创建新环境和网络，但强制 `residual=0`；验证新链不改变 Phase 1–4 基线。
3. **Phase 6 Stage 1**：M1 原地，Panda 执行逐步增幅的六自由度小幅轨迹；8D PPO 从零学习。

Stage 1 不加入 payload、rolling、抓取、外界 wrench 或大范围 domain randomization。

### 5.3 奖励

奖励按优先级组织：

1. 四轮接触、roll/pitch、base height、support margin、joint limit；
2. EE position/orientation tracking 和 MPC reference tracking；
3. wrench prediction/correction error、wheel slip、动作平滑；
4. residual magnitude、residual rate 和 residual intervention ratio 正则。

任务奖励乘稳定门 `g_safe`；接近失稳时 EE 任务奖励自动衰减，策略不能通过牺牲机器狗平衡追逐 EE target。理想模型区域应推动 `W_rl` 接近零。

### 5.4 PPO 与训练保护

- 环境频率 `200 Hz`；
- rollout `256` steps；
- adaptive KL 和有界学习率；
- bounded physical action std；
- 每次 update 记录 KL、LR、std、grad、MPC/QP、safety 和 physical diagnostics；
- 非有限值、QP 连续失败、失去轮接触、机身触地或关节硬越界立即停止；
- rolling-window best checkpoint、atomic manifest 和 automatic rollback；
- 短门通过后才允许最多 `3000` updates 长训；
- 长期退化时早停并回退最佳 checkpoint。

## 6. Safety projection

Safety 始终位于 RL 后、WBC/QP 求解前：

```text
Arm MPC -> RL residual -> safety projection -> WBC/QP -> actuator
```

约束至少覆盖 joint effort/position/velocity、mount wrench、base orientation、support margin、wheel contact 和 collision。沿用 Phase 4 的 `TRACK/SCALE/HOLD`：

- `TRACK`：residual scale `1.0`；
- `SCALE`：residual scale `0.5`；
- `HOLD`：residual scale `0.0`，重建无 residual 的安全请求。

MPC fallback 与 residual fallback 必须分别记录，不能使用同一个模糊的 `failure` 字段。

## 7. 文件边界

建议新增：

```text
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/residual_actor_critic.py
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_env_cfg.py
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py
Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py
Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py
Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py
Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_play.py
```

新增 Arm MPC、network、environment、wrapper、training guard、static entrypoint 和 GPU smoke 测试。只在必要位置扩展 registry、runtime adapter、WBC Teacher optional input 和 vendored RSL-RL model factory；不得顺带重构旧训练链。

GUI Play 创建环境前禁用 IsaacLab manager control window，只保留仿真 viewport，以避免 `env.close()` 后延迟 UI callback 访问已删除的 `viewport_camera_controller`。

## 8. 验收标准

### 8.1 Phase 5

- 全部 Arm MPC 输入、状态、输出和 fallback diagnostics finite；
- MPC feasible rate `>= 0.99`；
- QP feasible rate `1.0`；
- 四轮持续接触；
- base contact、joint-limit violation 和 unexpected reset 均为零；
- `abs(roll)`、`abs(pitch) <= 10 deg`；
- EE position error `<= 0.015 m`；
- EE orientation error `<= 0.08 rad`；
- 非零运动段仅在 measured force norm `>= 2 N` 或 measured moment norm `>= 0.2 Nm` 的样本上计算对应预测/测量 wrench 方向一致性，均要求 `>= 0.8`；
- 零 residual 不得增加 Phase 1–4 固定种子基线的任何 hard-failure count，EE position/orientation error 不得恶化超过 `10%`。

### 8.2 Phase 6

- 短训首先通过所有 Phase 5 物理硬门；
- trained policy 在固定种子集合上优于 zero-residual baseline；
- 比较至少覆盖 roll/pitch RMS、base-height RMS、EE position/orientation error、wrench correction error、slip、contact violation 和 residual intervention ratio；
- normalized residual 不得长期饱和，finite/limit/safety gate 不得被 reward 改善掩盖；
- 固定种子评估中任一 residual channel 的 `abs(normalized_action) >= 0.99` 样本比例必须 `< 1%`；
- 没有 eligible best 时 manifest 必须为 `accepted=false`，不得启动或宣称长训成功。

“优于 zero-residual baseline”按固定种子的 stability-first 字典序判定：先比较 hard-failure count，再比较 roll/pitch 与 base-height RMS，随后比较 EE error，最后比较 residual intervention ratio。任何 hard-failure 增加都直接判定失败。

## 9. 明确非目标

- 不实现 M1 rolling、payload 或抓取；
- 不加入外界推力或完整 domain randomization；
- 不实现 Phase 7 coordinated navigation/manipulation；
- 不进行 Teacher–Student 蒸馏；
- 不删除或覆盖旧 checkpoint；
- 不声称实机载荷、安全或抓取能力。

## 10. 完成定义

Phase 5–6 只有在纯/静态测试、CPU reference、GPU0 Phase 5 probe、Phase 6 zero-residual gate、guarded short train、固定种子评估、best/rollback manifest 和运行手册均有精确证据后才完成。3000-update 长训只在短门通过后启动；启动不是验收，只有 guard 和固定种子评估可以给出 accepted 结论。
