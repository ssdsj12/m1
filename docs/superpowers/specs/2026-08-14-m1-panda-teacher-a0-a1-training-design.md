# M1 + Panda Teacher A0/A1 平衡训练设计

## 1. 目的与范围

本设计定义 M1 + Panda 统一 articulation 的首个可执行 Teacher 平衡训练闭环。它消费已经完成的本地组合 USD、`BASE_LINK` 原点六维安装反力观测，以及 16 维受限残差动作组合器。

本阶段只训练 M1 原地驻停时对 Panda 安装界面扰动的平衡能力，不训练 Panda 操作控制器、Student 估计器、抓取奖励或行走中操作。后续抓取阶段以本阶段得到的稳定基座为前置输入。

实施分为两个连续阶段：

- A0：零基础动作加小幅准静态六维扰动，训练首个稳定残差策略。
- A1：冻结 A0 策略作为基础 M1 checkpoint，在更强、更快的扰动下训练第二级残差策略。

当前 RTX 5070 的 `sm_120` 不在已安装 PyTorch 所支持的最高 `sm_90` 架构范围内。因此本轮验收要求 CPU 短程可执行训练，不把本机长期 GPU 收敛作为完成条件。

## 2. 已有基础与边界

### 2.1 直接复用

- `assets/m1_panda/m1_panda.usd`：M1 与 Panda 的单 articulation 本地资产，共 25 DOF。
- `M1PandaSmokeEnvCfg`：隔离的 M1 + Panda 仿真场景和 16 维 M1 动作接口。
- `mount_wrench_base`：以 M1 `BASE_LINK` 原点为参考点、以 `BASE_LINK` 为表达坐标系的六维安装反力，顺序为 `[Fx, Fy, Fz, Mx, My, Mz]`。
- `M1ResidualActionComposer`：12 个腿位置残差与 4 个轮速残差的幅值及变化率限制。

### 2.2 明确不复用

现有 M1 行走/PVCNN checkpoint 的 actor 输入为 572 或 586 维，并依赖语义扫描和 crossing wrapper；它们与本阶段的 60 维驻停 Teacher 观测不兼容。A1 不隐式迁移或裁剪这些 checkpoint，而是严格加载本阶段 A0 产生的 60→16 checkpoint。

### 2.3 非目标

- 不控制 Panda 的 9 个 DOF；Panda 在本阶段保持固定姿态。
- 不向策略提供扰动命令、物体质量或未来 wrench。
- 不训练 Student、时序估计器或传感器噪声模型。
- 不执行 0–3 kg 抓取课程。
- 不宣称短程 smoke 已达到长期平衡成功率指标。

## 3. 环境与观测契约

新增两个独立 Gym ID：

```text
Isaac-M1-Panda-Teacher-A0-v0
Isaac-M1-Panda-Teacher-A1-v0
```

二者共享相同的 Isaac Lab 环境配置和 60 维 Teacher policy observation：

| 字段 | 维度 | 坐标/语义 |
| --- | ---: | --- |
| M1 base angular velocity | 3 | `BASE_LINK` |
| projected gravity | 3 | `BASE_LINK` |
| M1 joint position | 16 | 默认位置相对量 |
| M1 joint velocity | 16 | 关节顺序与动作接口一致 |
| previous final combined action | 16 | 实际送入 Isaac Lab action manager 的上一动作 |
| ideal mount wrench | 6 | `BASE_LINK` 原点，`[F, M]` |

总维度固定为 `3 + 3 + 16 + 16 + 16 + 6 = 60`。A0 与 A1 actor 输入、critic 输入均使用该组；本阶段没有额外 privileged critic group。

`previous final combined action` 必须来自 wrapper 已完成两级组合、限幅和变化率限制后的最终 16 维动作，不能使用原始 PPO residual。环境 reset 后该字段归零。

## 4. 六维扰动课程

### 4.1 施加位置与坐标

扰动的目标 body 为 `panda_hand`。调度器保存以 M1 `BASE_LINK` 坐标系表达的 wrench，并在每个物理 step 使用当前刚体姿态转换为 PhysX 所需的 `panda_hand` 局部 force/torque 后施加。

这里的输入 torque 是 `panda_hand` 参考点处的自由力矩，不额外做从 `BASE_LINK` 原点到手部的力矩平移；由手部受力对 M1 安装点产生的力矩通过真实动力学自然形成。策略仍只观察安装点测得的 `BASE_LINK` 原点总反力。

每个环境独立维护随机数状态、当前 wrench、保持剩余步数和课程强度。reset 只重置对应环境，并显式清除其外力缓存。

### 4.2 A0 分布

- 保持时间：均匀采样 `1.0–2.0 s`。
- 每个力轴独立采样 `[-10, 10] N`。
- 每个力矩轴独立采样 `[-2, 2] Nm`。
- 训练初期课程强度为满幅的 25%，按训练进度单调增加至 100%。
- 以组合准静态 wrench 为主；保持区间之间允许不连续重采样。

### 4.3 A1 分布

- 保持时间：均匀采样 `0.25–1.0 s`。
- 每个力轴独立采样 `[-20, 20] N`。
- 每个力矩轴独立采样 `[-5, 5] Nm`。
- 分布混合准静态保持、连续线性变化与短时脉冲。
- A1 同样从较低课程强度递进到满幅，不能在首个 rollout 直接启用全部动态扰动。

调度器不把当前目标 wrench 或模式标识写入 policy observation。策略只能从理想安装点反力和机器人响应推断当前扰动。

## 5. 动作架构

### 5.1 A0

A0 PPO actor 输出 16 维归一化 residual：

```text
zero_base[16] + A0_actor_residual[16]
    -> M1ResidualActionComposer
    -> final_action[16]
    -> Isaac Lab action manager
```

`zero_base` 在每一步均显式构造为零。composer 使用已批准默认值：腿动作缩放 0.25、轮动作缩放 8.0、腿残差幅值 0.05 rad、轮残差幅值 1.0 rad/s、腿 slew 0.01 rad/step、轮 slew 0.2 rad/s/step。

### 5.2 A1

A1 冻结 A0 actor 及其基础 composer，另训练一个相同输出维度的 residual actor：

```text
latest_obs[60]
    -> frozen A0 actor
    -> base composer(zero, A0 residual)
    -> frozen_base_action[16]

latest_obs[60]
    -> trainable A1 actor residual[16]

frozen_base_action + A1 residual
    -> residual composer
    -> final_action[16]
```

两个 composer 各有独立历史。环境终止时，两者必须针对相同 `env_ids` 清零，防止前一 episode 的 slew 状态泄漏。

冻结 A0 actor 使用 `eval()` 和 `torch.no_grad()`，参数不进入 A1 optimizer。A1 训练前后必须通过参数摘要验证冻结网络未改变。

## 6. Wrapper 时序契约

专用 wrapper 对 RSL-RL 暴露标准 `VecEnv` 接口，并持有最新的 60 维 observation。

### 6.1 Reset

1. 调用环境 reset。
2. 清除全部 composer 历史。
3. 清除全部扰动及计时状态。
4. 将 previous final action 清零。
5. 验证并缓存初始 60 维 observation。

### 6.2 Step

1. 验证 PPO residual 的 shape、device、dtype 和 finite 状态。
2. A1 使用 step 前缓存的最新 observation 推理冻结 A0 residual；A0 使用零基础动作。
3. 通过 composer 得到 final action。
4. 调度或更新该 step 的每环境 wrench，并施加到 `panda_hand`。
5. 调用底层环境 `step(final_action)`。
6. 把 final action 作为下一 observation 的 previous action 来源。
7. 对完成的 `env_ids` 清除 composer、扰动和缓存状态。
8. 验证返回 observation、reward 和 wrench 均为 finite，并缓存最新 observation。

若底层环境在 step 内自动 reset，wrapper 仍必须根据返回的 done mask 清除自身状态；不能依赖下一次显式 `reset()`。

## 7. 奖励与终止

奖励全部从可解释的平衡目标构成：

- 存活奖励。
- base roll/pitch 稳定奖励或惩罚。
- base 高度相对 `0.60 m` 的误差惩罚。
- base vertical velocity 和 horizontal angular velocity 惩罚。
- 水平位置漂移与轮速惩罚。
- residual 幅值与 residual 变化率惩罚。
- actuator torque 惩罚。
- feet slide 惩罚。

奖励配置使用明确的独立 term 和权重，方便 TensorBoard 消融。A0、A1 默认共享奖励；A1 不通过降低稳定性要求来适应更强扰动。

终止条件为：

- M1 base 发生非期望接触。
- 姿态超过安全阈值并判为跌倒。
- episode timeout。

timeout 必须与真实失败分开记录。短程 smoke 只证明训练链可执行，不作为奖励权重合理或策略收敛的证据。

## 8. PPO 配置

首版沿用当前 M1 RSL-RL 的稳定基础参数：

- rollout：24 steps per environment。
- actor/critic hidden dims：`[256, 128]`。
- PPO epochs：5。
- minibatches：4。
- learning rate：`1e-3`。
- gamma：`0.99`。
- lambda：`0.95`。
- initial action noise std：`0.01`。
- checkpoint save interval：100 iterations；短程验收允许 CLI 覆盖为 1。

A0 与 A1 使用不同 experiment/run 目录，避免 checkpoint 混淆。CLI 可以覆盖 `num_envs`、`max_iterations`、seed、device 和 save interval。

## 9. Checkpoint 契约

### 9.1 A0 基础 checkpoint

A0 checkpoint 必须包含 RSL-RL 的 `model_state_dict`、optimizer 状态、iteration 和 infos。训练入口另写入同一 run 目录的 `run_manifest.json`，至少包括：

- schema version；
- stage=`A0`；
- Gym task ID；
- observation/action dimensions；
- actor hidden dims；
- seed；
- composer 配置；
- disturbance 配置；
- checkpoint 相对路径或最近保存规则。

### 9.2 A1 加载

A1 的 `--base-checkpoint` 必填。进入第一个仿真 step 前严格检查：

- checkpoint 文件和相邻 manifest 存在；
- manifest stage 为 A0；
- observation 为 60、action 为 16；
- actor hidden dims 与当前配置一致；
- actor 首层和末层 tensor shape 匹配；
- checkpoint 能以 strict 模式加载。

A1 manifest 额外记录规范化后的 A0 checkpoint 路径、文件 SHA-256 和冻结 actor 初始参数 SHA-256。

### 9.3 断点续训

`--resume-checkpoint` 仅恢复当前阶段的可训练 PPO runner：

- A0 resume 只能接受 stage=A0。
- A1 resume 只能接受 stage=A1，并要求继续提供与 manifest 哈希一致的 `--base-checkpoint`。
- A1 resume 不从 resume checkpoint 猜测或替换基础 A0 文件。

阶段、维度、结构或哈希不匹配均为硬错误。

## 10. 故障行为

以下情况立即抛出带上下文的错误并停止训练：

- observation、action、reward 或 wrench 含 NaN/Inf；
- action/observation shape 与 16/60 契约不符；
- body/joint 查找不唯一；
- checkpoint/manifest 缺失或不兼容；
- A1 冻结参数在训练期间变化；
- reset 后外力或 composer 历史未清除。

训练入口不得静默回退到随机基础策略、零基础策略或旧 PVCNN checkpoint。

## 11. 文件与模块边界

计划新增或扩展的最小代码面：

- `go2_pvcnn/tasks/m1_panda_teacher_env_cfg.py`：共享场景、观测、奖励、终止和 A0/A1 cfg。
- `go2_pvcnn/tasks/m1_panda_teacher.py`：纯 PyTorch 扰动调度、阶段配置和验证辅助。
- `go2_pvcnn/tasks/m1_panda_teacher_wrapper.py`：A0/A1 动作组合与 VecEnv 时序。
- `go2_pvcnn/agent/m1_panda_teacher_train_cfg.py`：Teacher PPO 配置。
- `go2_pvcnn/tasks/register_envs.py`：两个 Gym ID。
- `scripts/m1_panda_teacher_train.py`：独立训练/恢复入口和 manifest。
- `tests/`：纯单元、静态接线和真实 CPU smoke 测试。

已有通用 mount wrench 与 residual composer 仅在其契约确有缺口时小范围扩展；不复制第二套实现。

## 12. TDD 与验收

### 12.1 纯测试

实施按 RED→GREEN 执行，至少覆盖：

- A0/A1 扰动幅值、保持区间、课程缩放和确定性 seed。
- `BASE_LINK` wrench 到 `panda_hand` 局部输入的旋转转换。
- 每环境独立重采样和选择性 reset。
- A0 零基础动作合成。
- A1 冻结 actor + 两级 composer 合成。
- done env 的历史清除与未 done env 状态保持。
- checkpoint stage/shape/hash 正例和反例。
- manifest 原子写入和 resume 契约。
- 非有限输入在状态更新前失败。
- Gym 注册、60 维 observation、16 维 action 与 reward/termination 静态接线。

### 12.2 真实 Isaac Sim CPU smoke

在单环境 headless CPU 上依次执行：

1. 创建 A0 环境并进行 reset/step，确认 wrench 非零、观测为 60、动作是 16、无非有限值。
2. 运行 A0 一次极短 PPO 迭代，save interval 设为 1，生成 checkpoint 和 manifest。
3. 使用该 A0 checkpoint 启动 A1，完成一次极短 PPO 迭代并保存 A1 checkpoint。
4. 比较 A1 前后冻结 A0 actor 参数摘要，必须完全一致。
5. 验证 A0 和 A1 checkpoint 都可由相应 resume 命令重新加载；允许把 resume 验证限制为初始化或一次极短迭代。

### 12.3 交付命令

最终必须给出四类可直接替换 run 名和 checkpoint 路径的命令：

- A0 正式训练。
- A0 断点续训。
- A1 正式训练，显式传入 A0 base checkpoint。
- A1 断点续训，同时传入相同 A0 base checkpoint 和 A1 resume checkpoint。

命令需要写明当前 CUDA 架构限制，并提供 CPU 短程验证命令；不把未运行的 GPU 命令描述成已验证。

## 13. 后续阶段

A0/A1 通过后，下一阶段才加入 Panda 关节/末端轨迹扰动，再进入静止抓取与 0–3 kg 负载课程。Student 的带噪六轴测量、时序估计器和 Teacher 蒸馏仍按总设计独立实施。最大负载或快速摆臂实机实验继续受 T400.3 机械最坏工况验算约束。
