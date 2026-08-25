# M1 + Panda 折叠负载 3000 轮持续训练设计

## 目标

让 folded-load 课程的每个阶段最多完整训练 3000 次 PPO 更新，避免策略刚开始学习时被 `eligible_patience_50_updates` 过早终止。训练期间只允许真正的灾难性保护触发提前停止；正常的“暂时没有改善”不再结束训练。

本设计只修改训练生命周期、更新上限和 checkpoint 周期，不修改机器人资产、Panda PD 参数、动作掩码、观测、奖励、命令范围、域随机化或固定评估阈值。

## 已确认的问题

当前训练在第 23 次更新首次得到 eligible snapshot 后立即开始累计 patience。之后 50 次没有产生更优 eligible rank，便在第 74 次更新触发 `eligible_patience_50_updates`。

这不代表策略已经收敛。L0-C0 训练命令幅值从零到上限均匀采样，而固定评估使用最大幅值命令。几乎不运动的策略在训练分布上也可能满足平均 RMSE 门槛，却会在最大幅值的前进、后退、左转和右转评估中全部失败。因此第一次 eligible snapshot 不能作为启动早停计时器的依据。

## 训练生命周期

### 正常结束

- `max_iterations` 的合法范围改为 `1..3000`。
- 课程入口默认 `max_iterations=3000`。
- 单阶段训练入口允许显式传入最多 3000。
- PPO runner 是正常迭代上限的唯一控制者；training guard 不再用硬编码更新数结束训练。
- 达到请求的更新数后，manifest 使用 `requested_iterations_complete` 表示正常完成。

### 允许的提前停止

以下保护保持即时生效：

- PPO、环境状态或诊断值出现 NaN/Inf；
- inactive Panda action 不再严格为零；
- Panda 折叠误差、执行器利用率或关节限位裕度越界；
- 当前 episode 的硬失败率连续达到已有严重阈值；
- 进程级异常、CUDA/Isaac Sim 错误或用户显式终止。

以下条件删除：

- `eligible_patience_50_updates`；
- training guard 内部的 `max_iterations_600`。

删除 patience 只影响正常学习期间的停止决策，不削弱安全保护。

## 最佳模型与周期 checkpoint

- PPO 周期 checkpoint 从每 25 轮保存一次改为每 100 轮保存一次。
- 3000 轮最多产生约 31 个周期 checkpoint（包含初始/最终边界），控制磁盘占用。
- `model_best.pt` 继续保持“满足训练资格且 rank 最优”的语义，不能被非 eligible 策略覆盖。
- 每次出现更优 eligible snapshot 时仍原子更新 `model_best.pt` 和 `model_best.json`。
- 即使没有 eligible best，周期 checkpoint 仍提供恢复和离线诊断能力。
- 不自动删除旧实验、失败报告或 checkpoint。

## 训练后评估与课程推进

- 只有完整训练结束且存在 eligible `model_best.pt` 时，才执行 seed 42/43/44 固定评估。
- 固定评估继续使用前进/后退 `vx_rmse <= 0.04`、左/右 `wz_rmse <= 0.12` 以及原有接触、姿态和静止门。
- 三个种子全部通过才创建 `model_final.pt` 并进入下一阶段。
- 任一方向或种子失败时，保留所有训练 checkpoint 和评估报告，课程停止，不自动降低阈值、不自动进入下一阶段。
- 本次修改不把评估失败反馈回同一次训练进程；3000 轮完成后再做一次严格验收。

## 命令与运行目录

正式课程使用全新实验根目录，不能复用 `foundation-v1`、`foundation-v2` 或 `foundation-v3`：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
TERM=xterm-256color \
CONDA_PREFIX=/home/xk/miniconda3/envs/go2 \
CUDA_VISIBLE_DEVICES=0 \
/home/xk/miniconda3/envs/go2/bin/python -u \
scripts/m1_panda_folded_load_curriculum.py \
  --start_stage L0-C0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --device cuda:0 \
  --experiment_root logs/m1_panda_folded_load/foundation-v4 \
  --headless
```

在当前执行环境中，长训必须使用持续 PTY 会话；普通 `nohup ... &` 子进程会在工具调用结束时被回收。`foundation-v4` 只是建议的新目录名，启动前仍必须验证它不存在。

## 测试与验收

### 单元测试

- 首次 eligible 后连续超过 50 次不改善不会停止。
- 第 600、2999 次更新不会由 guard 主动停止。
- 非有限值、inactive action 泄漏、fold hard failure 和已有硬失败率条件仍立即/按原窗口停止。
- train 与 curriculum CLI 接受 3000，拒绝 3001。
- 默认课程更新数为 3000。
- PPO `save_interval` 为 100。
- 编排器向每个阶段传递 3000，而不改变评估种子和严格 lineage 行为。

### 回归验证

- folded-load guard、训练入口、curriculum、orchestrator 和 PPO 配置聚焦测试全部通过。
- Python 编译检查通过。
- 8×1 smoke 能正常完成且不会因 guard 的 patience 停止。
- 正式启动后至少观察到第 1 次学习更新、GPU 0 显存上升以及 manifest `status=running`，才能报告训练已启动。

## 不在本次范围

- 不改变训练命令采样分布；
- 不改变方向 rank 定义；
- 不引入训练中周期固定评估；
- 不调整奖励权重、学习率、KL、网络或动作噪声；
- 不引入外力、主动 Panda 运动、抓取、Student 或实机部署。

这些内容只有在完整 3000 轮仍无法通过方向验收时，才根据保留的曲线、checkpoint 和方向报告另行设计。
