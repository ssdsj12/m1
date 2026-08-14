# M1 + Panda Teacher A0/A1 Play 设计

## 1. 目的与范围

本设计为已经训练完成的 M1 + Panda Teacher A0/A1 checkpoint 增加专用推理与可视化入口。入口必须复现训练时的 60 维观测、16 维动作、A0 零基础动作、A1 冻结 A0 基础策略与双层 residual composer，不能借用现有通用 `m1_play.py` 的不同 wrapper。

Play 用于观察策略在六维安装扰动下的平衡行为、执行短程 smoke 和零扰动对照。它不训练网络、不创建 optimizer、不写 checkpoint 或 manifest，也不把当前未通过行为验收的 A1 checkpoint 描述为可部署策略。

## 2. 方案选择

采用独立入口 `Go2Pvcnn/scripts/m1_panda_teacher_play.py`，不修改通用 M1 play 的既有语义。备选方案包括扩展 `m1_play.py` 或给训练脚本增加 play 模式；二者都会把不同 observation/wrapper/checkpoint 契约混合在同一入口，因此不采用。

运行时复用现有 Teacher 环境配置、`M1PandaTeacherEnvWrapper`、checkpoint validator、冻结 actor loader 和 RSL-RL runner。只有“禁用扰动”的最小控制面以及 play 生命周期与诊断属于新增行为。

## 3. CLI 契约

入口至少提供以下参数：

| 参数 | 契约 |
| --- | --- |
| `--stage {A0,A1}` | 必填，选择环境、wrapper 与 checkpoint 阶段 |
| `--checkpoint PATH` | 必填，待播放的当前阶段 checkpoint |
| `--base-checkpoint PATH` | A1 必填且只能是兼容 A0 checkpoint；A0 不需要 |
| `--num-envs N` | 默认 `1`，必须为正整数 |
| `--device DEVICE` | 默认 `cuda:0`，GPU0 对应 `cuda:0` |
| `--seed N` | 默认沿用训练配置种子 |
| `--steps N` | 默认 `0`；`0` 表示运行到窗口关闭，正整数表示有界步数 |
| `--disable-disturbance` | 默认不设置；设置后进入零扰动对照 |
| `--stats-interval N` | 默认 `100`，必须为正整数，控制诊断输出周期 |
| `--headless` 及 AppLauncher 参数 | 默认打开 GUI；允许无界面 smoke |

非法 stage、缺少必要 checkpoint、A0 携带无意义的 base checkpoint、非正环境数、负 steps 或非正统计周期都必须在启动仿真前失败，并给出明确错误。

## 4. Checkpoint 与策略加载

所有 checkpoint 在第一个仿真 step 前严格验证：

- A0 的 `--checkpoint` 必须由 stage=A0 manifest 描述，观测/动作维度为 60/16，actor hidden dims 与当前配置一致。
- A1 的 `--checkpoint` 必须由 stage=A1 manifest 描述；`--base-checkpoint` 必须是兼容的 A0 checkpoint。
- A1 manifest 记录的 base checkpoint SHA-256 必须与命令行所给 A0 文件一致。
- 推理加载不要求 optimizer 状态，并使用 `load_optimizer=False`；checkpoint 的模型 tensor shape 和 strict actor load 仍然是硬门。
- 不允许回退到随机 actor、零 actor、通用 M1/PVCNN checkpoint 或从 A1 manifest 猜测 base checkpoint 路径。

Runner 只作为构造与加载当前阶段 policy 的兼容层。加载完成后获取 inference policy，整个 play 循环在 `eval()` 和 `torch.inference_mode()` 下运行。

## 5. A0/A1 数据流

### 5.1 A0

```text
Teacher observation[60]
  -> A0 checkpoint actor residual[16]
  -> wrapper: zero_base + residual composer
  -> final M1 action[16]
  -> Isaac Lab
```

### 5.2 A1

```text
Teacher observation[60]
  -> frozen A0 actor -> base composer -> base action[16]
Teacher observation[60]
  -> A1 checkpoint actor residual[16]
base action + A1 residual
  -> second composer -> final M1 action[16]
  -> Isaac Lab
```

A1 的 base actor 必须通过既有 loader 从 `--base-checkpoint` 创建并冻结。Play 与训练使用同一个 wrapper 时序：基础策略读取 step 前 observation，外力在底层 `env.step` 前施加，done 环境的两个 composer 状态同时清零。

## 6. 扰动开关

扰动默认开启，并使用所选 stage 的训练配置：A0 使用小幅准静态课程，A1 使用更强的动态混合课程。这样默认 play 检查的是策略实际训练目标，而不是无载荷静态站立。

`--disable-disturbance` 是显式零扰动基线。启用后 wrapper：

- 不推进扰动 scheduler；
- 对全部环境施加零六维 wrench，并调用现有 clear shim 清除缓存外力；
- 对外报告的 effective current wrench 与 max-abs wrench 都保持为零；
- 保留完全相同的 observation/action/checkpoint 路径，不绕过策略或 composer。

reset 和 done 后都必须继续保持零 wrench，不能让上一 episode 或上一次启用扰动的状态泄漏。

## 7. 生命周期与诊断

默认 GUI 模式在 `simulation_app.is_running()` 为真时循环；窗口关闭后正常退出。`--steps N` 为正时最多执行 N 个 policy step，适合 headless smoke。退出路径必须关闭底层环境和 simulation app，即使加载或 step 抛错也不能故意吞掉原异常。

每 `--stats-interval` 步输出一条紧凑诊断，至少包括：当前 step、平均 reward、累计 done 数、当前六维 wrench 的逐轴最大绝对值、历史最大绝对 wrench，以及可取得时的 `bad_orientation`、`base_contact`、`time_out` reset 计数。终止 term 不存在时标记为 unavailable，不得把缺失误报为零。

所有 observation、policy action、final action、reward 与 effective wrench 必须满足既有 shape 和 finite gate。任何 checkpoint、hash、shape 或 finite 异常立即停止。

## 8. 文件边界

计划中的修改范围：

- 新增 `Go2Pvcnn/scripts/m1_panda_teacher_play.py`。
- 扩展 `M1PandaTeacherEnvWrapper`，加入默认开启的 disturbance enable/disable 契约。
- 新增 play 静态/纯行为测试，并扩展 wrapper 测试覆盖零扰动 reset/step。
- 在 human/AI entrypoint 文档和 Teacher runbook 中加入 GPU0 A0/A1 play 命令。
- 更新 T400 branch、dashboard 与逐次验证日志。

不修改训练默认扰动、奖励、PPO 配置、资产、60/16 observation/action 契约或已有 checkpoint 文件。

## 9. 验收

实施采用单代理 TDD：先以静态与 fake-env 测试锁定 parser、阶段校验、checkpoint preflight、A0/A1 wrapper 选择、默认扰动和显式关闭扰动，再实现入口和 wrapper 最小扩展。

最低验收包括：

1. 新测试先对缺失入口/行为产生预期 RED，再变为 GREEN。
2. Teacher play、wrapper、checkpoint、scheduler、composer 相关回归全部通过。
3. 新增 Python 文件及修改文件通过 `py_compile`，占位符扫描为空。
4. 使用 `/home/xk/miniconda3/envs/go2/bin/python` 和 `--device cuda:0 --headless --num-envs 1 --steps` 完成 A0、A1 短程 smoke；两者默认扰动必须观测到非零 wrench。
5. 至少一个 `--disable-disturbance` smoke 验证 effective/current/max wrench 为零。
6. A1 smoke 验证 base checkpoint SHA 与 frozen actor hash 不变。
7. runbook 给出可直接复制的 GUI 默认扰动命令和零扰动对照命令。

若 GPU runtime 因本机环境失败，必须记录实际错误和 CPU/静态替代证据，不得声称 GPU play 已通过。

## 10. 已知限制

当前 A1 长程训练结果尚未达到行为验收要求，因此新增 play 只提供忠实推理、观察和诊断能力。它不能把 A1 的平衡质量、抓取能力或实机可用性变为已完成状态。Panda IK/OSC、抓取物体、Student 六维力估计、传感器融合和实机安全状态机均不在本次实现范围内。
