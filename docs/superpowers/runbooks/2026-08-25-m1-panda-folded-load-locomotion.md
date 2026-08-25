# M1 + Panda 折叠负载基础运动 Runbook

## 当前范围

本链只训练 M1 携带动态、PD 折叠 Panda 的前进、后退和转向。策略接口保持 103 维观测和 23 维输出，但仅动作 0:16 生效；Panda 7 维动作始终严格为零。没有外力事件、Panda 主动运动、抓取、Student 或实机能力声明。

所有命令固定 GPU 0。folded-load 任务独立使用 `panda_joint4=-2.650 rad` 折叠目标和 shoulder `120/8` PD，不修改全局 Panda 默认。`Go2Pvcnn/` 中没有 `./isaaclab.sh`，本机真实 launcher 是 `/home/xk/coding/IsaacLab/isaaclab.sh`。非交互式启动必须同时提供 `TERM=xterm-256color` 和 Go2 环境的 `CONDA_PREFIX=/home/xk/miniconda3/envs/go2`。

## 1. 8 环境物理 Probe

先确认动态组合 articulation 能以零策略动作稳定 step，inactive 动作严格为零，fold/effort/limit 均在门内，并能读到非零且有限的安装反力：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_probe.py --num_envs 8 --steps 16 --device cuda:0 --report logs/m1_panda_folded_load/probe-pd120-j4m2650-8x16.json --headless
```

短 probe 通过后必须再覆盖完整 PPO horizon：

```bash
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_probe.py --num_envs 8 --steps 256 --device cuda:0 --report logs/m1_panda_folded_load/probe-pd120-j4m2650-8x256.json --headless
```

两份原子报告都必须为 `passed=true`，且 shell 状态为 0，才能进入 smoke。失败报告和旧的 `80/4` 诊断必须保留，不得覆盖或删除。

## 2. 8×1 入口 Smoke

目标是验证训练入口、一次 rollout/update、manifest 和 checkpoint 写入，不是行为收敛：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_train.py --stage L0-C0 --num_envs 8 --max_iterations 1 --device cuda:0 --run_dir logs/m1_panda_folded_load/smoke-pd120-8x1 --headless
```

## 3. 64×10 稳定性 Smoke

输出目录必须不存在：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_train.py --stage L0-C0 --num_envs 64 --max_iterations 10 --device cuda:0 --run_dir logs/m1_panda_folded_load/smoke-pd120-64x10 --headless
```

两个 smoke 可以 clean exit 0，但由于 episode window/三种子门不完整，其 manifest 必须保持 `accepted=false`，不得作为下一 stage 的 parent，也不得称为 locomotion acceptance。

## 4. 单 Stage 与固定评估

正式单阶段训练示例：

```bash
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_train.py \
  --stage L1-C1 --parent_manifest /ABS/foundation-v4/L0-C0/run_manifest.json \
  --num_envs 4096 --max_iterations 3000 --device cuda:0 \
  --run_dir /ABS/foundation-v4/L1-C1 --headless
```

`--max_iterations` 的默认值和硬上限均为 3000。每 100 个 PPO update 保存一次常规 checkpoint。正常学习不会因为 50 轮没有刷新 eligible best 或到达旧的 600 轮边界而停止；只有 NaN/Inf、inactive action 泄漏、折叠误差、关节 effort/limit 越界或硬失败率超限等灾难性安全条件可以提前终止。到达请求轮数后再使用 eligible best 执行固定评估，平台期本身不代表失败。

只有 `training_eligible=true` 的 `model_best.pt` 才执行三次评估：

```bash
for seed in 42 43 44; do
  TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_eval.py \
    --stage L1-C1 --run_dir /ABS/foundation-v4/L1-C1 \
    --seed "$seed" --num_envs 64 --device cuda:0 --headless || break
done
```

第三份报告完成后，只有 aggregate 通过才出现 SHA-identical `model_final.pt` 和 `accepted=true`。

## 5. 完整八阶段长期课程

实验根目录不得包含 `L0-C0/`：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python -u scripts/m1_panda_folded_load_curriculum.py --start_stage L0-C0 --num_envs 4096 --max_iterations 3000 --device cuda:0 --experiment_root logs/m1_panda_folded_load/foundation-v4 --headless
```

长期训练应在持续保留的 PTY 会话中直接运行上面的 `python -u` 命令，并通过该会话轮询；不要依赖会被宿主回收的短命 `nohup`/后台子进程。

编排器只按 L0-C0 → L1-C1 → L1-C2 → L1-C3 → L1-C4 → L2-D1 → L2-D2 → L2-D3 顺序推进。任何 train/eval/accepted SHA 失败都会停止，并在 `curriculum_state.json` 指向上一 accepted checkpoint；不会自动降低难度继续。

## 6. 检查与监控

```bash
python -m json.tool logs/m1_panda_folded_load/foundation-v4/curriculum_state.json
python -m json.tool logs/m1_panda_folded_load/foundation-v4/L0-C0/run_manifest.json
tensorboard --logdir logs/m1_panda_folded_load/foundation-v4 --port 6006
```

重点检查 `Loss/kl`、`Loss/kl_max`、`Loss/kl_aborted`、`Loss/grad_norm`、`Policy/active_action_std_min/max`，以及 inactive action、fold error、effort utilization、joint-limit proximity 和 hard-failure rates。训练完成的唯一判据是 L2-D3 manifest 具有三份通过的 seed 报告和 `accepted=true`；Panda 运动、外力、抓取、Student 与硬件部署仍需独立设计和验收。
