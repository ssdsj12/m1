# M1 + Panda Phase 6 Fixed-Condition Promotion Design

## 1. 背景与已复现问题

Phase 5 已通过 seeds `42/43/44` 的独立 4000 步物理验收。Phase 6 的第一轮短训安全完成 51 updates，但 `model_best.pt` 仍是 iteration 0 的精确零输出策略。复现证据表明当前流程存在三个独立缺陷：

1. rolling guard 直接比较连续训练窗口。首个 256-step 窗口包含 trajectory warm-up，其目标 RMS 约为 `0.0113 m / 0.0247 rad`；下一个窗口约为 `0.0332 m / 0.0670 rad`。不同难度窗口不能用于 checkpoint 排名。
2. 固定种子评估对浮点指标使用严格 `<`。同一个精确零输出 checkpoint 在 seed 44 因 reset 后的 PhysX 数值差异获得约 `5.46e-5 rad` 的 roll/pitch 偶然改善并被误判为 accepted；seeds 42/43 则因逐位相同而被拒绝。
3. reward 直接使用约 `50--95` 的原始六维 wrench norm，`-0.2 * wrench_error` 的量级大于稳定与任务正奖励之和，违反稳定优先的奖励层次。

本修订只修复 Phase 6 Stage 1 的奖励量纲、checkpoint 选择和固定种子晋级。Phase 5 门、8D action、103D observation、Arm MPC、WBC/QP、安全投影及动作物理限幅保持不变。

## 2. 方案选择

采用固定条件离线 checkpoint 竞赛：训练期 guard 只执行硬安全保护，候选性能在相同 seeds、相同步数、相同 trajectory contract 下离线比较。

不采用以下替代方案：

- 双环境在线基线：计算更快，但不同 env placement 和求解顺序会引入新的跨环境物理差异。
- 目标难度归一化 rolling rank：依赖人工难度模型，仍不能消除 trajectory phase、reset 和 PhysX 噪声。
- 单纯延长当前短训：不会修复 iteration 0 的先发优势和严格浮点比较的假阳性。

## 3. Reward 量纲修复

训练 reward 中的 wrench prediction/correction error 使用逐通道无量纲残差：

```text
wrench_scale = [30 N, 30 N, 50 N, 15 Nm, 15 Nm, 8 Nm]
e_w = ||(W_measured - W_predicted) / wrench_scale||_2
wrench_penalty = -0.2 * tanh(e_w)
```

量程必须来自 `WholeBodyResidualCfg.physical_limits[:6]` 的单一配置源，不允许在 reward 中维护第二份常量。`tanh` 使该项固定落在 `[-0.2, 0]`，因此无法压过稳定与 EE tracking 奖励。日志和验收继续记录未归一化的原始 wrench error；reward 另记录 normalized wrench error，禁止把二者混为同一字段。

其他 reward 权重、稳定门、residual magnitude/rate/intervention 正则和 Small EE trajectory 暂不改变，以便单变量验证本次根因。

## 4. 训练期安全守卫

短训固定执行 100 updates，rollout 保持 256 control steps。保存候选：

```text
candidate_u000.pt
candidate_u025.pt
candidate_u050.pt
candidate_u075.pt
candidate_u100.pt
```

后缀表示已经完成的 PPO update 数，而不是 runner 内部 iteration index。`u000` 必须是任何 rollout/update 之前的精确零均值 actor；`u100` 必须是第 100 次 update 后保存的模型。训练期 rolling diagnostics 只用于：

- 非有限 optimizer/environment diagnostics 立即停止；
- hard failure、QP/MPC、轮接触、关节限制和 saturation 硬门立即或按现有连续失败合同停止；
- 原子保存周期 checkpoint 和失败 manifest。

不同 trajectory phase 的 rolling rank 不再更新 `model_best.pt`，也不再触发 `eligible_patience`。正常完成 100 updates 不代表 accepted。训练失败时不执行离线竞赛，也不得发布 best。

## 5. 零策略噪声标定

正式竞赛前，在每个 seed `42/43/44` 上执行三对 zero-vs-zero 4000-step 对照。每一对都使用与正式评估相同的 reset、trajectory 和 metrics 路径，并各自在全新的 Isaac Sim 进程中运行；同一进程既不得执行下一对，也不得重建下一个 seed 的环境。

对每个连续指标 `m`，从九个绝对差值建立容差：

```text
tol_m = max(engineering_floor_m, 2 * max(abs(delta_m_zero_zero)))
```

固定 engineering floors：

| metric | floor |
| --- | ---: |
| roll/pitch RMS | `1e-4 rad` |
| base-height RMS | `2e-5 m` |
| EE position error | `5e-5 m` |
| EE orientation error | `5e-5 rad` |
| raw wrench error | `0.1` |
| slip | `2e-5` |
| intervention ratio | `1 / 4000` |

hard-failure count、MPC/QP feasible、four-contact rate 和 saturation 不使用噪声容差，继续执行原硬门。标定 JSON 必须记录九组原始差值、floor、最终 tolerance、命令、seed、步数和资产/配置 SHA。

## 6. 固定条件 checkpoint 竞赛

每个候选 checkpoint 分别在 seeds `42/43/44` 上运行 4000 步，并与该 seed 的 zero-residual baseline 比较。每个 seed 使用一个全新 Isaac Sim 进程，以避免单实例环境重建后的长时吞吐退化。driver 负责合并子进程 JSON；缺失、非零退出、非有限值或 SHA 不一致均 fail closed。

候选首先必须在三个 seed 上全部满足 Phase 5/6 硬门，并满足：

- hard-failure count 不增加；
- MPC feasible rate `>= 0.99`；
- QP feasible rate和 four-contact rate均为 `1.0`；
- 每个 residual channel saturation fraction `< 0.01`；
- 任一连续 rank metric 不得比对应 baseline 恶化超过该 metric 的 tolerance。

通过上述门后，对三个 seed 的连续指标取算术平均，并按以下 tolerance-aware stability-first 顺序比较：

```text
hard failure
roll/pitch RMS
base-height RMS
EE position error
EE orientation error
intervention ratio
```

对当前指标，candidate 比 baseline 小超过 tolerance 即产生真实改善；大超过 tolerance 即失败；容差内视为相等并继续下一项。候选只有在至少一个指标产生真实改善时才可 accepted，完全等价的零策略不能晋级。raw wrench error 和 slip 是强制的非退化诊断：不得恶化超过 tolerance，但不先于稳定/EE 指标决定 best。

多个候选都 accepted 时，在候选之间复用相同 tolerance-aware 顺序选出唯一 best；若全部指标在 tolerance 内相等，选择 completed-update count 更小者。不得复制 checkpoint 后改写其身份；promotion manifest 必须指向原候选并记录 SHA-256。

## 7. Manifest 与长训门

短训目录新增：

- `noise_calibration.json`；
- `candidate_eval/candidate_u<updates>/seed_<seed>.json`；
- `promotion_manifest.json`；
- 仅在竞赛通过后原子发布的 `model_best.pt`。

`promotion_manifest.json` 至少记录候选列表及 SHA、逐 seed baseline/candidate metrics、tolerances、硬门、聚合比较、best iteration、best SHA 和 `accepted`。训练 `run_manifest.json` 只能记录训练是否完整和安全，不能单独授权长训。

long stage 必须同时验证：

1. 短训 run manifest 为安全完成；
2. promotion manifest 为 `accepted=true`；
3. asset/config/reward/checkpoint SHA 全部匹配；
4. `model_best.pt` SHA 与 promotion source SHA 一致。

任一条件不满足时拒绝启动 3000-update 长训。

## 8. 验证顺序

1. TDD 覆盖 wrench 归一化、有界 penalty 和原始诊断不变。
2. TDD 覆盖 tolerance 计算、等价、真实改善、真实退化、三 seed 硬门和 tie-break。
3. TDD 覆盖训练期不再按不同窗口发布 best、100-update 周期候选及 fail-closed manifest。
4. CPU focused regression 与 Phase 5 回归。
5. GPU0 zero-vs-zero 噪声标定。
6. GPU0 100-update 短训。
7. GPU0 候选固定条件竞赛。
8. 仅在 promotion accepted 后启动 long，并持续监控安全门。

Phase 6 只有在固定条件竞赛产生真实改善的 accepted checkpoint 后才通过短门；长训启动本身仍不代表最终训练验收。
