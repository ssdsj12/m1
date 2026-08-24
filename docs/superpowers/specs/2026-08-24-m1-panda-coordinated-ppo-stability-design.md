# M1 + Panda Coordinated PPO 稳定长训设计

## 目标

修复 `coordinated_teacher_long_v4_64x5000_20260823` 在约第 4022 次更新后出现的 PPO 策略坍塌，并建立一条可审计、可自动止损的 200 Hz Coordinated Teacher 从零训练链。

本阶段保持既有零间隙 M1 + Panda 单 articulation、103 维观测、23 维关节力矩残差动作、平地底盘目标和到达后 Panda 末端跟踪任务不变。新增范围仅包括：适合 200 Hz 的 PPO 时间尺度、KL/学习率自适应、最佳 checkpoint 与自动回退、Panda 末端真实外力课程、初始状态域随机化，以及旧 long v4 checkpoint 清理。

## 已确认的失败基线

旧 long v4 已完成 5000 次更新，但不是可接受模型：

- 第 4000 次更新附近仍为 `time_out=1.0`、`base_contact=0`；
- 第 4022 次开始出现机身接触终止；
- 第 4023 次 value loss 峰值达到约 `27.995`；
- 第 4256 次附近 `base_contact=1.0`；
- 最终 `model_4999.pt` 平均奖励约 `-46.86`、`time_out=0`、`base_contact=1.0`。

失败不是 CUDA、NaN 或资产 snap。主要训练缺陷是：24 步 rollout 在 200 Hz 下仅覆盖 `0.12 s`；固定学习率绕过 adaptive KL；探索 std 永久冻结为 `0.01`；环境初始状态和外力缺少有效多样性；训练始终采用最后 checkpoint，没有最佳模型和自动止损。

## 不变量与边界

- Gym ID 继续使用 `Isaac-M1-Panda-Coordinated-v0`。
- observation 保持精确 `103`，action 保持精确 `23 = 12 legs + 4 wheels + 7 Panda arm`。
- PhysX 和策略动作频率均保持 `200 Hz`，`sim.dt=0.005 s`、`decimation=1`。
- 资产继续使用已接受 SHA-256 `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`。
- 新正式训练从零 actor 开始，不加载 `model_3500.pt`、A1 actor、旧 optimizer 或旧 critic。
- `model_3500.pt` 只作为旧确定性基准保留。
- 不改变 C0/C1a deterministic WBC Teacher、A0/A1、Student S1、抓取或实机接口。
- 本阶段不声称完成抓取、载荷、复杂地形、实机安全或 Student 蒸馏验收。

## 方案选择

### 采用方案：200 Hz 策略 + 长 rollout

保持每个物理步输出一次 23 维残差动作，将 rollout 扩展为 256 步，并按 200 Hz 重新选择折扣尺度。这样不改变执行合同，同时让一次更新覆盖 `1.28 s` 的连续动力学。

### 未采用方案：50 Hz 策略、200 Hz 物理

让动作保持四个物理步可以降低训练难度，但会改变现有 200 Hz residual action 合同，并降低对快速末端外力的响应带宽，因此本阶段不采用。

### 未采用方案：仅修改 gamma/lambda

保留 24 步 rollout 即使提高折扣，也仍只有 `0.12 s` 的真实采样片段，不足以稳定识别延迟摔倒，因此不采用。

## PPO 时间尺度与优化合同

新正式训练固定以下基础合同：

- `num_steps_per_env = 256`；
- `gamma = 0.9995`；
- `lam = 0.995`；
- `learning_rate = 1.0e-4`；
- `schedule = "adaptive"`；
- `desired_kl = 0.01`；
- `num_learning_epochs = 5`；
- `num_mini_batches = 4`；
- `clip_param = 0.2`；
- `max_grad_norm = 1.0`。

KL 自适应必须实际生效，而不是只写入 manifest。PPO 每个 mini-batch 计算新旧高斯策略 KL；当 KL 高于目标上界时降低学习率，低于下界时有限提高。学习率必须限制在 `[1.0e-6, 3.0e-4]`，并将当前 KL、学习率和发生的调节写入 TensorBoard。

新 actor 末层仍以精确零权重和零 bias 初始化，保证初始 residual action 为零。实际 action std 从 `0.01` 开始，但不再 `requires_grad_(False)`。std 采用明确的实际标准差语义，并限制在 `[0.005, 0.05]`；不得重新引入 raw/log/softplus 含义混淆。`entropy_coef` 保持 `0.0`，探索变化由 PPO likelihood 梯度和 std 上下界控制。

新正式运行最多执行 600 次更新。以 64 环境计，总采样上限为 `64 × 256 × 600 = 9,830,400` transitions，约为旧 64×24×5000 运行的 `1.28` 倍。训练长度以 timesteps 为比较基准，不再机械复用 5000 次更新。

## 最佳 checkpoint 排名

新增一个与具体环境解耦的纯 Python `TrainingGuard`。runner 每次更新向 guard 提交不可变指标快照；guard 只做排名、耐心计数和停止决策，不直接读取 TensorBoard 文件。

指标至少累计 100 个完成 episode 后才可形成稳定候选。候选按以下字典序排名：

1. 最小化 `base_contact_rate + bad_orientation_rate`；
2. 最大化 `time_out_rate`；
3. 最大化 `base_target + ee_tracking`；
4. 最大化 `mean_reward`；
5. 指标完全相同时保留较早 iteration，避免无意义覆盖。

一个候选只有同时满足下列稳定门，才是 `eligible best`：

- `time_out_rate >= 0.90`；
- `base_contact_rate <= 0.05`；
- `bad_orientation_rate <= 0.05`；
- 所有排名指标有限；
- 至少 100 个完成 episode。

不满足稳定门的候选仍可保存为诊断 best，但不得标记为 accepted。

## 原子保存、早停与回退

当产生更优候选时：

- runner 先写临时 checkpoint，再通过原子 rename 更新 `model_best.pt`；
- 同步原子写入 `best_checkpoint.json`；
- JSON 记录 iteration、timesteps、完整指标、随机化课程比例、学习率、KL、checkpoint SHA-256 和 `eligible`；
- 普通 `model_<iteration>.pt` 仍按固定间隔保存，便于诊断。

只有首次出现 eligible best 后才启用自动早停：

- 灾难门：`base_contact_rate + bad_orientation_rate > 0.20` 连续 25 次更新；
- 耐心门：连续 50 次更新没有产生更优 eligible checkpoint；
- 上限门：达到 600 次更新。

50 次新更新包含 12,800 control steps，约等于旧配置 533 次更新的数据量。触发任一停止门后，runner 必须加载 `model_best.pt` 的模型权重，保存 `model_final.pt`，并在 run manifest 中记录：停止原因、停止 iteration、best iteration、回退源 SHA、final SHA 和 `accepted`。

如果到训练上限仍没有 eligible best，则保存字典序最优诊断候选并标记 `completed_without_eligible_best`、`accepted=false`；不得把它描述为稳定策略。

## Panda 末端真实外力课程

随机外力施加到 `panda_hand`，不直接施加到 M1 机身或伪造 mount wrench。PhysX 通过 Panda 链和固定安装关节自然产生对 M1 的反作用；103 维观测中的 `mount_wrench_b` 继续读取实际安装反力。

复用现有逐环境六维扰动调度基础，但为 coordinated wrapper 建立独立状态。每个环境独立采样且由训练 seed 完全决定：

- 满幅力上限：每轴绝对值不超过 `20 N`；
- 满幅力矩上限：每轴绝对值不超过 `5 Nm`；
- 保持时间：`0.25–1.0 s`；
- 模式概率：持续 `0.50`、脉冲 `0.30`、间歇 `0.20`；
- 脉冲 on fraction：`0.20`；
- 初始课程比例：`0.10`；
- 在每环境 50,000 个 200 Hz control steps 内线性升到 `1.0`。

wrapper 必须在 `env.step(actions)` 前写入当前 wrench；episode reset 时只清理并重采样对应环境，不能重置其他环境的课程状态。异常或非有限 wrench 必须在进入 PhysX 前失败，不能静默裁剪为零。

训练 TensorBoard 至少记录课程比例、实际施加的力范数、力矩范数和非零扰动环境比例。Play/确定性验证默认关闭随机外力；抗扰验证使用显式开关开启固定 seed 的完整课程或指定满幅。

## 初始状态与物理域随机化

每次 episode reset 对对应环境独立采样：

- root position：`x/y ∈ [-0.02, 0.02] m`，z 不随机；
- root orientation：roll/pitch `[-0.03, 0.03] rad`，yaw `[-0.05, 0.05] rad`；
- root linear velocity：每轴 `[-0.05, 0.05] m/s`；
- root angular velocity：每轴 `[-0.10, 0.10] rad/s`；
- 受控腿关节位置：默认安全姿态叠加 `[-0.02, 0.02] rad`；Panda 关节位置叠加 `[-0.03, 0.03] rad`；两者均裁剪到 soft joint limits；
- 受控关节速度：`[-0.05, 0.05] rad/s`；
- static/dynamic friction：每环境 startup bucket 在 `[0.8, 1.2]` 中采样；
- restitution 保持 `0.0`。

reset 后必须检查 root/joint state 有限且未越 soft limits。随机化值和 seed 写入运行配置；训练中记录 root pose/velocity 和关节扰动的范围统计。确定性 dynamics gate 与默认 Play 不继承训练随机化。

## Runner 接口与兼容性

对 vendored `OnPolicyRunner` 的改动必须是默认关闭、向后兼容的可选迭代回调/guard 接口。其他任务不提供 guard 时，保存频率、训练循环、checkpoint 格式和返回行为必须保持不变。

Coordinated 入口负责构造 guard、随机化 wrapper 和 manifest。通用 PPO 只负责 KL 自适应、可配置学习率边界、std clamp 和迭代指标输出；不能把 M1/Panda 指标名称硬编码进通用 PPO 类。

## 旧 long v4 checkpoint 清理

清理目标严格限制为：

`Go2Pvcnn/logs/m1_panda_coordinated/coordinated_teacher_long_v4_64x5000_20260823`

保留：

- `model_0.pt` 到 `model_3500.pt`；
- `model_3500.pt`；
- TensorBoard event、stdout、原始 `run_manifest.json` 和其他诊断文件。

删除所有编号大于 3500 的 checkpoint。按当前目录预期为 `model_3600.pt` 至 `model_4900.pt` 以及 `model_4999.pt`，实施前必须重新解析实际文件名和数值，打印精确列表，并拒绝路径越出上述 run directory。

原始 manifest 不伪装成以 3500 结束。新增 `checkpoint_pruning.json`，记录清理时间、原始 manifest SHA、每个删除文件名与删除前 SHA、保留上界和 `model_3500.pt` SHA。删除后验证不存在编号大于 3500 的 `model_*.pt`。这些日志 checkpoint 不在 Git 中，删除后只能依赖外部备份恢复。

## 测试与验证

实施严格采用单代理 TDD：

1. 纯单元测试覆盖 guard 排名、eligible gate、指标非有限拒绝、灾难/耐心/上限停止、原子 best/final manifest 和回退；
2. PPO 测试覆盖 adaptive KL 实际调整、学习率上下界、std 实际值上下界和默认无 guard 回归；
3. 静态/纯行为测试覆盖 256/0.9995/0.995、200 Hz 不变量、外力目标 body、随机化边界和 Play 默认关闭；
4. 真实 GPU0 多环境 probe 验证不同环境获得不同但可复现的初始状态与末端 wrench，mount wrench 有物理响应，reset 只影响选中环境；
5. 64 环境短训验证 TensorBoard 中 KL/LR/随机化/终止指标存在，`model_best.pt`、`best_checkpoint.json`、`model_final.pt` 和 manifest 回退链可读且 SHA 一致；
6. 清理前后验证旧 checkpoint 列表和审计 JSON。

短训和 probe 只验证训练链与物理接线。只有新的正式从零长训在多个 seed 下通过 stability/task gates，才能声称长期坍塌得到行为级修复。

## 完成标准

本实现阶段只有在以下条件同时满足时完成：

- 相关纯/静态测试、py_compile 和 diff check 通过；
- 默认 runner 行为无回归；
- GPU0 外力/初始状态随机化 probe 通过；
- GPU0 64-env 短训生成并验证 best/final/manifest 全链；
- long v4 大于 3500 的 checkpoint 已按审计记录删除；
- T400 dashboard、branch page、日志索引和逐次验证日志已对齐；
- 最终报告明确区分“训练基础设施通过”和“新长期策略尚待长训验收”。
