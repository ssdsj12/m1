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
