# 2026-08-14 M1 + Panda Teacher A0/A1 Play GPU0 验收

## Scope

完成专用 Teacher play 的最终静态回归，以及 RTX 5070 / `cuda:0` 上的 A0 默认扰动、A1 默认扰动和 A1 零扰动三段真实 headless smoke。

## Environment

- Python: `/home/xk/miniconda3/envs/go2/bin/python`
- PyTorch: `2.7.0+cu128`
- Isaac Sim: `5.1`
- GPU: NVIDIA GeForce RTX 5070, driver `580.159.03`
- Device: `cuda:0`, `CUDA_VISIBLE_DEVICES=0`
- Environments/steps: `1 / 8`

## Checkpoints

- A0: `logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt`
- A0 file SHA-256: `2cfa105e3c2dbcbdaed3b64c986f97124408bd9f4bd00732ad9ef2409161852f`
- A1: `logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_5999.pt`
- A1 file SHA-256: `e6a610919af7a2b10b7daa503a0ca81e74389db4b296b487f96a600c46b13e66`
- Frozen A0 actor SHA-256: `a7fd58c2753130128f698097eef3159f7f007081f9937f34990a610d8a992457`
- A1 manifest base SHA and frozen initial/final hashes match the values above.

## Runtime Results

### A0 default disturbance

- Exit: `0`
- Contract: observation/action `60/16`
- Steps/done: `8/0`
- Reset reasons: bad orientation `0`, base contact `0`, timeout `0`
- `max_abs_wrench_seen=2.449706`

### A1 default disturbance

- Exit: `0`
- Contract: observation/action `60/16`
- Steps/done: `8/0`
- Reset reasons: bad orientation `0`, base contact `0`, timeout `0`
- `max_abs_wrench_seen=4.899412`
- Frozen actor hash unchanged.

### A1 zero disturbance

首次运行在 Isaac Lab 5.1 permanent wrench composer 触发 Warp empty-array incompatibility。新增纯测试先复现为 RED `1 failed`，随后 `clear_external_wrench` 对该精确错误回退到 `[num_instances,num_bodies,3]` 全零 wrench；兼容 focused `4 passed`。

修复后：

- Exit: `0`
- Contract: observation/action `60/16`
- Steps/done: `8/0`
- 六个 `wrench_axis_abs_max` 均为 `0.000`
- `max_abs_wrench_seen=0.000000`
- Frozen actor hash与默认扰动运行相同且未变化。

## Final Static Verification

```text
Teacher play/wrapper/checkpoint/disturbance/env/train/composer: 195 passed
py_compile: exit 0
placeholder scan: exit 0
play learn/manifest-write scan: exit 0
```

测试命令显式设置 `PYTHONPATH=/home/xk/coding/M1/Go2Pvcnn/rsl_rl` 和 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，分别隔离 site-packages 新版 RSL-RL 接口与 ROS pytest 自动插件污染。

## Review

按用户明确的单代理约束未派 code-review subagent。单代理最终审查逐项核对了 checkpoint 顺序、A1 base SHA、frozen hash、默认/零扰动语义、finite/shape gate、退出清理和禁止写盘；未发现未处理的 Critical/Important 问题。

## Known Warnings and Limits

- 仍出现已知 disabled Panda `root_joint` disjointed-transform warning；本变更未修改资产。
- Isaac Lab 报告 actuator 参数和旧 wrench API deprecation；不影响三段 exit `0`。
- 8-step smoke 只证明加载、动作链、外力开关和短程数值契约，不证明 A1 长程平衡或抓取能力通过。

## Git Refs

- Branch: `main`
- Base HEAD observed before final handoff: `8872421d02eb93b04b150d025148c8a93e78dd09`
- Current state: working tree changes, not committed by Codex
