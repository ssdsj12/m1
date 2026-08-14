# M1 + Panda Teacher A0/A1 Real CPU Smoke

## Purpose

在真实 Isaac Lab/PhysX CPU 环境执行 A0 initial/resume → A1 initial/resume，验证 60/16 环境、非零六维扰动、checkpoint progression、strict base/resume 与 frozen hash。

## Final Command

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 2400 \
  /home/xk/miniconda3/envs/loco/bin/python \
  scripts/m1_panda_teacher_smoke.py \
  --output-root logs/m1_panda_teacher_smoke_20260814_1818 \
  --device cpu --headless
```

## Result

Exit `0`，final JSON `status="completed"`。

- 四段 return code：`0, 0, 0, 0`。
- A0 checkpoint：`model_0.pt → model_1.pt`，最终 SHA-256 `8899254d9049a7ae8c9c47da29cd4dce95a873aabad038e9e9b155d96a6a0326`。
- A1 checkpoint：`model_0.pt → model_1.pt`，manifest base SHA 与上述 A0 SHA 完全相同。
- A1 frozen actor initial/final SHA：均为 `c3fbef05c386124a52ce20758c1aa51f21b9e3c49af6e2dd2ecb01c25d33db8d`。
- A0/A1 strict checkpoint validation：均为 `true`。
- A0 live runtime contract：observation `60`、action `16`、`max_abs_wrench_b_seen=2.0750198364257812`。
- A1 live runtime contract：observation `60`、action `16`、`max_abs_wrench_b_seen=0.35319486260414124`。
- 每段完成 4 timestep、reward/loss 输出有限；这是可执行链验收，不是收敛证明。

## Diagnosed Runtime Failures

1. 首次 A0 在 cfg 访问失败：Gym registry 返回字符串。新增聚焦 RED 后改用 `parse_env_cfg`。
2. 第二次 A0 在 Reward Manager 初始化失败：`base_xy_drift` 漏传 `asset_cfg`。新增静态 RED 后显式绑定 robot。
3. 首轮四段均 exit `0`，但 resume 覆盖 `model_0.pt`。新增 iteration RED 后在 load 后前进一轮，并由 driver 强制检查数值后缀递增。

## Artifacts

- Root: `Go2Pvcnn/logs/m1_panda_teacher_smoke_20260814_1818/`
- A0 manifest: `a0/smoke_a0/run_manifest.json`
- A1 manifest: `a1/smoke_a1/run_manifest.json`
- Child logs: `a0_initial.*.log`, `a0_resume.*.log`, `a1_initial.*.log`, `a1_resume.*.log`

## Known Warnings / Open Scope

- PhysX 仍报告 Panda `root_joint` 初始 body transform disjointed 并可能 snap；既有 topology/CPU foundation 可运行，但最大载荷实机前仍必须完成 T400.3 机械和装配复核。
- OmniHub 不可访问、implicit actuator 参数弃用警告不阻断本次 CPU smoke。
- RTX 5070 `sm_120` 与当前 PyTorch 最大 `sm_90` 不兼容；本轮不声称 CUDA 长期训练或策略收敛。
- Student、Panda 运动/IK/OSC、抓取和实机安全状态机仍开放。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Last Feature Commit: unavailable
- Last Verified Commit: unavailable
