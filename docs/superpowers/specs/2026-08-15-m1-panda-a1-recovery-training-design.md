# M1 + Panda A1 抗扰平衡恢复训练设计

## 1. 目标

在保留现有 A0/A1 checkpoint 和日志的前提下，恢复 M1 + Panda Teacher 在外界六维力/力矩扰动下的驻停平衡训练，并以独立满幅评估决定 checkpoint 选择和训练停止。

本轮只恢复 A1 抗扰平衡能力。Student、Panda 关节控制、抓取、零间隙资产和实机机械验算不混入该训练基线。

## 2. 已有证据与问题定义

正式 GPU0 训练已经完成：

- A0：`model_2999.pt`，扰动上限 `10 N / 2 Nm`；
- A1：`model_5999.pt`，扰动上限 `20 N / 5 Nm`；
- A1 frozen A0 actor SHA-256 始终为 `a7fd58c2753130128f698097eef3159f7f007081f9937f34990a610d8a992457`。

A1 TensorBoard 后 100 轮均值显示后期退化：`time_out≈0.0966`、`base_contact≈0.6813`、`bad_orientation≈0.2247`。相同 seed、32 环境、1000-step 的渐进扰动 Play 对照中，`model_2700` 为 64/64 timeout，`model_4500` 为 40 timeout/23 base contact，`model_5999` 为 0 timeout/75 base contact。该对照只用于发现退化，不作为满幅验收，因为 Play 的课程从 25% 重新开始。

根因调查还确认 vendored `ActorCritic` 与配置语义不一致：

- Teacher 配置声明 `noise_std_type="log"`，旧类却忽略该参数；
- checkpoint 和 TensorBoard 中的 `std` 是原始参数；
- 旧类对该参数应用 `softplus`，所以原始 `0.01` 实际成为约 `0.698`，原始 `0.002` 实际仍约 `0.694`；
- PPO 又把原始参数裁剪为非负，因此有效采样标准差无法低于约 `0.693`；
- 本机安装的官方 RSL-RL 和仓库内 `ActorCriticCNN` 都把 scalar std 直接用作分布标准差。

因此本轮不得从 `model_5999.pt` 盲目追加迭代，也不得继续沿用错误的 softplus/std 组合。

## 3. 选定方案

采用“满幅 checkpoint 筛选 → 独立 fork 恢复 → 500 轮分块训练 → 满幅复评”的路线。

候选 checkpoint 为 `model_2700.pt`、`model_3800.pt`、`model_4500.pt` 和 `model_5999.pt`。先在固定满幅 A1 扰动下排序，再从胜者创建新目录。原 A1 目录和所有 checkpoint 保持只读，不允许 fork 写回来源目录。

不采用：

- 直接从 `model_5999.pt` 续训，因为已有训练和 deterministic Play 都显示明显退化；
- 立即从头重训 A1，因为中期 actor 均值策略仍包含可复用能力；
- 同时修改 reward 或 residual composer，因为会使噪声语义修复无法单独归因。

## 4. 满幅评估契约

Teacher Play 增加显式的满幅评估模式。该模式把扰动 scheduler 的初始课程 step 设置为 `curriculum_steps`，因此第一段采样即使用 A1 上限，而不是等待 75,000 environment steps。

每个候选使用：

- 64 environments；
- 2000 environment steps；
- seeds `42`、`43`、`44`；
- A1 `20 N / 5 Nm`，hold/ramp/pulse 比例保持 `0.50/0.30/0.20`；
- 同一个 A0 base checkpoint 和 frozen hash。

聚合排名按以下字典序执行：

1. timeout survival rate，越高越好；
2. base-contact termination rate，越低越好；
3. bad-orientation termination rate，越低越好；
4. mean episode reward，越高越好。

评估必须输出机器可读 JSON，记录 checkpoint SHA-256、seed、环境数、步数、实际课程 scale、六轴最大施力、各终止计数/比例和 reward。满幅证据要求课程 scale 精确为 `1.0`，且采样历史中每个 force 轴的绝对最大值至少为 `19 N`、每个 torque 轴至少为 `4.75 Nm`；若未达到、出现非有限值或 frozen hash 不符，该行无资格参与排名。

## 5. 标准差语义修复

`ActorCritic` 恢复明确的标准差模式：

- `scalar`：`std` 参数直接作为 Normal 标准差；
- `log`：`log_std` 经 `exp` 后作为 Normal 标准差；
- 未知模式立即失败，不再静默忽略。

恢复分支使用 `scalar`，原因是现有 A1 checkpoint 的 state dict 已包含 `std` 键，使用 scalar 可 strict 加载 actor/critic，不迁移或猜测网络权重。旧 checkpoint 中的 actor 均值网络不变，因此 deterministic inference 不变。

恢复训练的 scalar std 下限为 `0.001`。载入 checkpoint 后、第一次 rollout 前先执行下限裁剪，避免个别通道的 `1e-15` 进入采样。`entropy_coef` 保持 `0`，本轮不同时引入新的探索奖励。

TensorBoard 和 manifest 必须区分：

- raw noise parameter；
- effective action standard deviation；
- noise std mode。

该修复会影响使用 vendored `ActorCritic` 的后续训练，因此必须运行相关全量回归；旧 checkpoint 的 deterministic inference 仍以 actor 均值为准。

## 6. Fork 恢复训练

训练入口增加与原地 resume 区分的 fork 模式：

- 输入 source A1 checkpoint、原 A0 base checkpoint和新 run name；
- 新建 A1 recovery run 目录；
- strict 校验 source stage、60/16、actor hidden dims、base SHA 和 frozen actor SHA；
- 加载 source actor/critic 权重；
- 不加载旧 optimizer state，使用新的 optimizer；
- 保留 source iteration 作为 lineage，但新 run 的本地训练计数和产物不得覆盖来源文件。

恢复 optimizer 初始学习率为 `1e-4`，其余 PPO reward、composer、观测和动作契约不变。

扰动课程起始 step 从 source checkpoint iteration 与 `num_steps_per_env` 推导：

```text
initial_curriculum_step = min(
    source_iteration * num_steps_per_env,
    disturbance_curriculum_steps,
)
```

以 24 steps/env 为例，`model_2700` 从 64,800/75,000 开始，`model_3800` 及以后从满幅开始。该值必须显式写入 manifest，训练时单环境 reset 不得重置全局课程进度。

## 7. 分块训练与监控

GPU0 恢复训练每块 500 iterations，每 100 iterations 保存 checkpoint。每块正常结束后，对最新 checkpoint 执行第 4 节的三 seed 满幅评估。

持续记录：

- timeout/base-contact/bad-orientation 比例；
- mean episode reward/length；
- base height、orientation、linear/angular velocity 和 XY drift reward；
- raw/effective action std；
- learning rate、value/surrogate loss；
- max absolute six-axis wrench；
- frozen actor initial/final SHA；
- 60/16 runtime contract。

最终接受门为三个 seed 聚合后：

- timeout survival rate `>= 0.80`；
- base-contact termination rate `<= 0.10`；
- bad-orientation termination rate `<= 0.10`；
- finite observation/action/reward/wrench；
- 实际扰动达到 `20 N / 5 Nm` 契约；
- frozen A0 actor SHA 不变。

若连续两次块后评估的 survival rate 都比已记录最佳值低超过 0.10，则停止当前分支，保留最佳 checkpoint，不继续堆叠训练。

## 8. Manifest 与失败处理

Recovery manifest 除现有字段外记录：

- `recovery_source_checkpoint` 和 SHA-256；
- source iteration；
- optimizer reset 事实和初始 learning rate；
- noise std mode、minimum raw/effective std；
- initial curriculum step/scale；
- 每次满幅评估 artifact 路径；
- best checkpoint、best metrics 和停止原因。

以下情况立即失败，不静默回退：

- source/base checkpoint 或 manifest 不兼容；
- fork 目标目录已经存在；
- 任何来源文件 checksum 发生变化；
- 满幅模式实际 scale 小于 1.0；
- observation/action/reward/wrench 出现 NaN/Inf；
- frozen A0 actor hash 改变；
- fork 写入来源 run 目录；
- 60/16 或 25-DOF 资产契约改变。

中断时保留最后一个完整 checkpoint 和 manifest 状态，下一次只能从 recovery run 自身 checkpoint 原地 resume；不能把它伪装成新的 source fork。

## 9. 测试与实施顺序

实施采用单代理 TDD：

1. 为 scalar/log std、effective std 和旧 `std` checkpoint strict load 写 RED 测试。
2. 最小修复 ActorCritic 噪声语义并运行 RSL-RL/Teacher 回归。
3. 为 scheduler 初始课程 step 和满幅 Play 写 RED 测试。
4. 实现满幅 JSON 评估与三 seed 聚合。
5. 为 fork 新目录、lineage、optimizer reset 和禁止覆盖写 RED 测试。
6. 实现 recovery fork 和 manifest 扩展。
7. 运行静态、checkpoint、wrapper、train/play 和 CPU smoke。
8. GPU0 执行候选满幅评估并选择 source。
9. GPU0 执行首个 500-iteration recovery block 和三 seed 满幅复评。
10. 按停止条件继续分块，直到达标或触发退化停止。

## 10. 范围边界

本轮不修改：

- Teacher 60 维观测、16 维 M1 residual action；
- reward 权重、residual composer 限幅和 A0 frozen actor；
- Panda 关节、夹爪、IK/OSC 和抓取任务；
- Student estimator/distillation；
- Panda/M1 USD 安装高度、质量、惯量和碰撞；
- 实机六轴传感器和机械承载设计。

零间隙资产 T400.6 必须在本轮恢复训练完成后独立实施和复验；不得把资产高度变化混入 recovery checkpoint 的行为归因。

## 11. 回滚

代码回滚时恢复旧 ActorCritic 和训练入口即可，但旧 softplus/std 训练只能作为历史诊断，不再作为后续正式基线。Recovery fork 不修改原 A1 文件，因此训练数据回滚只需停止新分支并保留其 artifact。
