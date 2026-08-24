# 2026-08-23 M1 + Panda Coordinated Teacher PPO 长训

## 目标

把旧的 67-observation smoke 前置替换为可学习的 103-observation / 23-action 协调任务，在 GPU0 通过动态、PPO 和行为 sanity 门后启动 5000-iteration 正式长训。

## 冻结合同

- 单一零间隙 M1+Panda articulation；资产 SHA-256 `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`。
- policy observation 为精确 103：本体状态、23 关节位置/速度、底盘目标误差、EE 误差、desired twist、四轮接触、六维安装 wrench 和 previous action。
- action 为 canonical `12 legs + 4 wheels + 7 Panda`，wrapper 在 PhysX 前 clamp `[-1,1]`。最终 residual scale 为腿 `5`、轮 `50`、臂 `2`；0.1 m/s nominal wheel feedforward 根据 `30 Nms/rad` 阻尼和 `0.095 m` 半径解析生成，到达后屏蔽 wheel residual。
- fresh actor 输出层精确零初始化，探索 std 冻结为 `0.01`，PPO 固定学习率 `1e-4`，每 100 iterations 保存。
- balance-first：目标/速度奖励均受高度和倾斜门控，终止代价权重 `-10000`；底盘到达后才启用 EE tracking。
- A1 `model_10402.pt` 只记录 lineage/SHA，不迁移不兼容的 60→16 actor 权重。

## 验证证据

- 纯/静态合同：`14 passed`；相关模块 `py_compile` exit `0`。
- 真实 GPU0 2-env×20-step 配置/物理探针：103/23、三 action term、13 reward term；无 reset、base contact、bad orientation、joint limit 或 non-finite。
- GPU0 8-env×1 iteration：完成并生成 `model_0.pt`，manifest `completed`。
- 早期诊断证明两个必须修复的失败模式：
  - `1e-3` + 随机 actor 在约 140–255 iterations 后产生 `base_contact=1.0`；
  - 安全版本长期保持 `base_target≈0.380`、EE=0，说明 `desired_twist=0` 和 `1 Nm` wheel authority 只能原地站立。
- 最终 GPU0 64-env×600 sanity：越过全部历史故障窗口；第 595 轮 `time_out=1.0`、`base_contact=0`、`bad_orientation=0`、termination penalty `0`，mean reward `128.53`、base-target `2.4825`、EE tracking `1.0423`、std `0.01`。这证明训练已进入 EE 阶段，但不代表策略收敛或实机可用。

## 正式长训

- run：`Go2Pvcnn/logs/m1_panda_coordinated/coordinated_teacher_long_v4_64x5000_20260823`
- PID：`683345`
- GPU：`CUDA_VISIBLE_DEVICES=0` / `cuda:0`
- 配置：64 env，seed 42，5000 requested iterations，save interval 100。
- stdout：`Go2Pvcnn/logs/m1_panda_coordinated/coordinated_teacher_long_v4_64x5000_20260823.stdout.log`
- 最终状态：完成全部 5000 updates，但行为验收失败。第 4000 次更新附近仍为 `time_out=1.0`、`base_contact=0`；第 4022 次首次出现 base contact，第 4023 次 value loss 峰值约 `27.995`，第 4256 次附近 `base_contact=1.0`；最终 mean reward 约 `-46.86`、`time_out=0`、`base_contact=1.0`。未发现 CUDA、NaN、traceback 或资产 snap，因此该 run 只保留为 PPO 后期坍塌证据，不作为可部署 checkpoint。

后续稳定重训设计见 [2026-08-24 specification](../../docs/superpowers/specs/2026-08-24-m1-panda-coordinated-ppo-stability-design.md)。

## 边界

本长训只训练平地上的协调底盘目标与到达后 EE pose tracking。六维安装 wrench 仍在 observation 中；启动 wrench 峰值归 T400.11 单独调查。不得把训练运行外推为抓取、最大载荷或实机安全验收。
