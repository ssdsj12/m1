# M1 + Panda Teacher A0/A1 训练 Runbook

## 当前可执行范围

本入口训练 M1 驻停受力平衡 Teacher：A0 从零基础动作学习小幅准静态六维扰动，A1 严格冻结同一条 A0 checkpoint，再学习第二级受限 residual。当前阶段不训练 Student，不控制 Panda 关节，也不执行抓取。

本机 RTX 5070 是 `sm_120`，当前 PyTorch 构建最高支持 `sm_90`，因此已验证命令使用 `--device cpu`。只有升级到支持该架构的 PyTorch/Isaac Lab 组合后，才把它替换为受支持的 CUDA device。

## A0 正式训练

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
/home/xk/miniconda3/envs/loco/bin/python \
  /home/xk/coding/M1/Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A0 --num_envs 64 --max_iterations 3000 \
  --run_name a0_force_balance --device cpu --headless
```

默认输出在 `logs/m1_panda_teacher/a0/a0_force_balance/`。最终 checkpoint 为 manifest 中的 `final_checkpoint`。

## A0 断点续训

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
/home/xk/miniconda3/envs/loco/bin/python \
  /home/xk/coding/M1/Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A0 --resume-checkpoint /ABS/A0/model_N.pt \
  --max_iterations 1000 --device cpu --headless
```

Resume 复用 checkpoint 所在目录，禁止同时指定 `--run_name`。默认要求 optimizer state；只有明确接受重置优化器时才加 `--reset-optimizer`。

## A1 正式训练

`/ABS/A0/model_N.pt` 必须是本阶段生成且验证通过的 60→16 A0 checkpoint。

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
/home/xk/miniconda3/envs/loco/bin/python \
  /home/xk/coding/M1/Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A1 --base-checkpoint /ABS/A0/model_N.pt \
  --num_envs 64 --max_iterations 3000 \
  --run_name a1_dynamic_force_balance --device cpu --headless
```

## A1 断点续训

A1 resume 必须继续使用创建该 A1 run 时的同一个 A0 base checkpoint；base 文件内容变化会因 SHA-256 不一致而失败。

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
/home/xk/miniconda3/envs/loco/bin/python \
  /home/xk/coding/M1/Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A1 --base-checkpoint /ABS/A0/model_N.pt \
  --resume-checkpoint /ABS/A1/model_M.pt \
  --max_iterations 1000 --device cpu --headless
```

## 四段验收 Smoke

输出目录必须尚不存在。

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 2400 \
  /home/xk/miniconda3/envs/loco/bin/python \
  scripts/m1_panda_teacher_smoke.py \
  --output-root /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher_smoke \
  --device cpu --headless
```

成功时最后一行 JSON 的 `status` 为 `completed`，四个 `returncode` 都为 `0`，A0/A1 checkpoint 校验为 `true`，A1 的 base SHA 等于 A0 checkpoint SHA，且 frozen actor 初末 SHA 相等。每个 child 的 stdout/stderr 路径也在 JSON 中。

## Teacher Play（GPU0）

专用 play 入口严格复现训练 wrapper，而不是通用 `m1_play.py`。GUI 默认打开，同时默认开启六维扰动并采用所选 stage 的分布；只有显式增加 `--disable-disturbance` 才进入零 wrench 对照。`--steps 0` 是默认值，表示运行到关闭窗口；需要自动结束的验证使用 `--headless --steps N`。

GPU0 play 使用 `/home/xk/miniconda3/envs/go2`（PyTorch 2.7.0 + CUDA 12.8）和 `cuda:0`。A0 GUI：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A0 --checkpoint /ABS/A0/model_N.pt \
  --num-envs 1 --device cuda:0
```

A1 GUI。`/ABS/A0/model_N.pt` 必须是创建该 A1 run 时使用的同一 A0 文件，内容 SHA-256 不一致会在仿真 step 前失败：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A1 --base-checkpoint /ABS/A0/model_N.pt \
  --checkpoint /ABS/A1/model_M.pt \
  --num-envs 1 --device cuda:0
```

A1 零扰动对照。这个开关只清除外力，仍然执行 frozen A0 actor、A1 actor 和两个 residual composer：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A1 --base-checkpoint /ABS/A0/model_N.pt \
  --checkpoint /ABS/A1/model_M.pt \
  --num-envs 1 --device cuda:0 --disable-disturbance
```

周期诊断包含 reward、done、当前六轴 wrench 绝对最大值、历史 wrench 最大值和可用的 reset reason。默认扰动模式应出现正的 `max_abs_wrench_seen`；零扰动对照必须保持为 `0`。当前 A1 checkpoint 只用于诊断和可视化，尚未通过长期行为验收，也不代表抓取或实机能力。

## A1 满幅筛选与恢复训练（GPU0）

恢复训练固定使用 `/home/xk/miniconda3/envs/go2`、`CUDA_VISIBLE_DEVICES=0` 和 `cuda:0`。开始前对 A0 base 与四个候选 checkpoint 执行 `sha256sum` 并保存结果；筛选和 fork 后必须再次校验，禁止改写源文件。恢复期间也禁止修改零间隙 M1+Panda 资产，以免策略变化和资产变化混在同一次验收中。

四候选、三 seed 满幅筛选命令如下。`--full-scale-disturbance` 会把课程直接置于 1.0；输出目录必须不存在：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_eval_sweep.py \
  --base-checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_2700.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_3800.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_4500.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_5999.pt \
  --seed 42 --seed 43 --seed 44 --num-envs 64 --steps 2000 \
  --output-dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/eval/a1_fullscale_candidates_20260815 \
  --device cuda:0
```

只从 `ranking.json` 读取 winner，不能凭 TensorBoard 目测选择。第一次恢复创建独立目录，并隐式重置 optimizer、采用 `1e-4` 学习率、保留 checkpoint policy std 后按有效量裁剪至至少 `0.001`，同时从源 iteration 恢复扰动课程：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
RECOVERY_WINNER=$(
  /home/xk/miniconda3/envs/go2/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["winner"]["checkpoint"])' \
    /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/eval/a1_fullscale_candidates_20260815/ranking.json
)
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_train.py \
  --stage A1 \
  --base-checkpoint logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --fork-checkpoint "$RECOVERY_WINNER" \
  --run_name a1_force_balance_recovery_gpu0_20260815 \
  --num_envs 64 --max_iterations 500 --save-interval 100 \
  --device cuda:0 --headless
```

每个 500 iteration block 结束后，用同一个 sweep 入口仅评估该 block 的最终 checkpoint（仍为 seeds 42/43/44、64 env、2000 steps）。未达标时在同一恢复目录续训：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_train.py \
  --stage A1 \
  --base-checkpoint logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --resume-checkpoint /ABS/RECOVERY/RUN/model_N.pt \
  --max_iterations 500 --save-interval 100 \
  --device cuda:0 --headless
```

硬验收条件是三 seed 聚合后 `timeout survival >= 0.80`、base contact `<= 0.10`、bad orientation `<= 0.10`，且每个力轴达到 `19 N`、每个力矩轴达到 `4.75 Nm`、课程 scale 为 `1.0`、数值 finite、frozen A0 hash 不变。若连续两个 block 的 survival 都比历史最佳低超过 `0.10`，停止训练并保留历史最佳，不把最后 checkpoint 冒充最佳。

恢复监控重点 TensorBoard tag：`Policy/mean_noise_parameter` 是存储参数，`Policy/mean_action_std` 是真正采样的有效标准差；兼容 tag `Policy/mean_noise_std` 也表示有效值。同时观察 termination、episode length、reward、base height/orientation 和 residual/torque 项。

## Manifest 与监控

每个 run 目录的 `run_manifest.json` 是恢复和验收依据：启动时为 `running`，正常结束为 `completed`，异常为 `failed`。同时核对 `stage`、`observation_dim=60`、`action_dim=16`、base/resume 路径、checkpoint SHA、frozen actor 初末 SHA，以及 `runtime_contract.max_abs_wrench_b_seen > 0`。

TensorBoard：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/loco/bin/tensorboard --logdir logs/m1_panda_teacher
```

重点关注 episode length、各 termination、value/surrogate loss、action noise、base height/orientation/drift、residual/rate 和 torque/feet-slide 项。短程 smoke 只证明链路可执行，不代表策略收敛。

## 中断与恢复

需要停止时使用一次 `Ctrl+C`，等待环境和 SimulationApp 清理。中断会把 manifest 标为 `failed`；从目录中最后一个完整 `model_<iteration>.pt` 用上述 resume 命令继续。不要手工修改 checkpoint、相邻 `run_manifest.json` 或 A1 使用的 A0 base 文件。

正式上机前仍需完成 Panda/M1/传感器最坏工况机械验算、Student 训练与实机安全状态机；本 runbook 不构成 0–3 kg 抓取能力验收。
