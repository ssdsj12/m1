# PPO Wire 操作策略设计

## 目标

使用 PPO 学习基于 DeformX reduced Cosserat wire 动力学的双 Panda + 双 O6 操作策略。PPO 负责 wire 操作决策；现有 MPC 仅作为安全投影、关节/碰撞约束和不可行 fallback，不作为主要动作策略。

## 动作接口

策略动作固定为 15D latent：

- 6D：双掌相对 wire 的平移/姿态增量；
- 6D：左右 O6 的闭指、张开、夹持和指尖微调 latent；
- 3D：wire 操作阶段与双手夹持力分配 latent。

latent 经过现有解码器映射为双 Panda 掌心目标、O6 指尖目标和夹持阶段命令。策略不直接输出 43 个或更多关节量，保持低维探索和动作连续性。

## 观测接口

观测不包含具体任务 ID，包含：

- wire reduced Cosserat 状态：中心线采样、切线、曲率/形变和速度；
- 双手掌心位姿、速度、指尖位姿、接触 mask、接触力和滑移；
- 双 Panda 关节位置/速度、限位余量和当前安全投影状态；
- 当前 mission phase 与上一动作 latent。

所有观测经过有限值检查并归一化；缺失或不可信接触数据进入安全 fallback，不伪造接触成功。

## 两阶段训练

### Stage A：Approval PPO

目标是学习稳定接触闭环，不要求完成复杂 wire 形变任务。课程顺序为接近、双手同时接触、闭指、夹持、抬升、保持。奖励只包含距离/姿态误差、接触建立、夹持稳定、抬升和安全惩罚。只有满足完整 acceptance episode 才标记 approval checkpoint。

### Stage B：Wire Action PPO

从 Stage A 的 approval checkpoint 初始化，继续学习 wire 位姿和形变操作。奖励加入 wire 中心线/端点目标、切线方向、形变误差、双手同步、力平衡、滑移和 latent 平滑项。MPC safety projection 失败时使用原子 fallback，并记录失败原因；失败样本不当作成功专家数据。

## Acceptance

一个 episode 只有同时满足以下条件才算成功：每只手至少 2 个有效接触指尖；每个有效接触力大于 `0.2 N`；抬升高度至少 `0.05 m`；保持至少 `1 s`；滑移不超过 `0.02 m`；全程无非有限值、碰撞或不可恢复 QP/Arm MPC 失败。

## 训练和验证边界

- 使用 GPU0 PPO rollout/training；CPU 线程限制为单线程，避免抢占整机 CPU。
- DeformX reduced runtime 作为环境动力学，保持物理状态原子提交。
- 不新增 VLM、感知训练、Residual 学习或接触 MPC；不修改既有 T400/T500 公共合同。
- 每次训练输出 checkpoint、配置、环境/代码 SHA、逐 episode JSONL 和 aggregate manifest。
- 在 Stage A approval 未达到前，不启动 Stage B 长训练。

## 失败处理

接触丢失、力不足、滑移超限、预测/安全投影不可行或非有限值会终止当前 episode 或回退到最近安全状态。fallback 不会污染 PPO 的成功回放；失败原因必须写入 diagnostics。

## 验证计划

1. CPU：观测/动作 shape、latent 解码、奖励分项、终止条件、fallback 原子性和 determinism。
2. GPU0 smoke：固定 seed 验证 wire 接触、双手同步、显存和单核 CPU 限制。
3. Stage A：先获得 acceptance checkpoint，再允许 Stage B。
4. Stage B：固定 seed 回归和多 seed 泛化，输出逐 episode JSONL 与 SHA-pinned manifest。
