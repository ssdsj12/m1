# T400 M1 + Panda 六轴力感知 Teacher–Student

## Current State

用户已确认完整设计：Panda 作为本地离线 USD 与 M1 组成统一 articulation；实机在 Panda 与 M1 安装界面使用六轴力/力矩传感器；Teacher 使用仿真理想安装点总六维反力，Student 使用带噪六轴测量、本体感知和时序估计器，输出对 M1 基础运动策略的受限平衡残差。首个任务是 M1 驻停时完成 0–3 kg 抓取和短距离搬运。

总设计已完成、自审并由用户批准。Teacher A0/A1 专项书面规格和逐文件单代理 TDD 实施计划均已批准并执行完成。A0 使用零基础动作和小幅准静态扰动；A1 严格冻结本阶段 A0 产生的 60→16 checkpoint，再训练第二级 residual。既有 572/586 维 PVCNN checkpoint 不在本阶段迁移范围内。

Asset/wrench foundation Task 1–6 已完成：用户批准每轴独立 reset/re-equilibrate 后，fresh CPU 六轴 probe exit `0`，七行 artifact 全部 finite/no-reset/pass，六轴 sign count 均为 `50/50`；最终静态 `58 passed`，checksum/PXR/CPU verifier 均 exit `0`。Network-denial runtime 未获权限，仍是明确限制。

受限残差动作组合器子阶段已由单代理按 4-task TDD 计划实现完成。Focused `34 passed`，composer + foundation 回归 `89 passed`，`py_compile` exit `0`。实现采用独立、状态化的纯 PyTorch 组合器，不加载基础策略 checkpoint，也不混入 Teacher/Student、IK/OSC 或 Isaac Lab 环境接入。

Teacher A0/A1 Task 1 已完成：三轮预期 RED 后，纯 PyTorch 扰动配置/scheduler、BASE_LINK→body-local wrench 转换与 clear shim 均为 GREEN；focused `51 passed`，scheduler+wrench/probe 回归 `77 passed`，pycompile/scan exit `0`。下一项是严格 checkpoint/manifest 契约。

Task 2 也已完成：strict manifest/state tensor shape/base hash/frozen actor 契约 focused `33 passed`；Task 1+2+composer 回归 `118 passed`，pycompile exit `0`。旧 572/586 维 checkpoint 无法通过实际 tensor shape gate。

Task 3 已完成：Teacher reward helper `7 passed`；A0/A1 cfg 与 23-ID registry focused `24 passed`，含 M1 asset static `25 passed`，pycompile exit `0`。两个环境保持 60 observation、16 M1-only action 和 smoke termination。

Task 4 已完成：A0 wrapper 纯行为 `11 passed`，wrapper+scheduler+composer 回归 `96 passed`，pycompile exit `0`。zero base、wrench-before-step、done/selective reset、外力清零与 60/16 finite gates 已锁定。

Task 5 已完成：A1 frozen actor 与双 composer 完整 wrapper `19 passed`；wrapper+checkpoint+disturbance+composer 回归 `137 passed`，pycompile exit `0`。eval/frozen gate、pre-step observation、双层 residual、选择性 reset 与 actor hash 漂移检测均已锁定。

Task 6 已完成：独立 PPO cfg、CLI/preflight、strict base/resume、atomic manifest 和 cleanup flow 的 static+checkpoint 回归 `49 passed`，pycompile exit `0`。下一步进入真实 Isaac CPU 四段 smoke。

Task 7 已完成：最终静态 `214 passed`，四段真实 CPU smoke 全部 exit `0`；A0/A1 checkpoint 均由 `model_0` 前进到 `model_1`，A1 base SHA 对齐且 frozen actor 初末 SHA 相同，60/16/nonzero-wrench runtime contract 写入 manifest。正式 runbook 已给出。

Teacher 专用 play 的书面规格已经用户复核确认，三任务单代理 TDD 实施计划已完成并自审：使用独立入口，A0 复现 zero-base residual，A1 严格加载 frozen A0 + A1 residual；GUI 和六维扰动默认开启，显式 `--disable-disturbance` 才进入零 wrench 对照。下一步按计划 inline 执行，当前尚未修改运行代码。

T400.5d Task 1 已完成：有效 RED `5 failed, 19 passed`，GREEN wrapper `24 passed`，wrapper+disturbance+composer 回归 `109 passed`，pycompile exit `0`。wrapper 默认扰动语义不变；显式 disabled 会 clear 外力且不推进 scheduler，A1 动作链仍执行。下一步实现 strict play 入口。

T400.5d Task 2 已完成：入口缺失有效 RED `13 failed`，GREEN focused `13 passed`；play+wrapper+checkpoint+train static 在项目 `rsl_rl` 绑定下 `97 passed`，pycompile 与禁止 learn/manifest-write scan exit `0`。下一步补齐 runbook/人机文档并运行全量静态与 GPU0 三段 smoke。

T400.5d Task 3 已完成：runbook 文档 RED→GREEN；最终静态 `195 passed`，compile/placeholder/no-write scan exit `0`。RTX 5070 GPU0 三段真实 smoke 均最终 exit `0`：A0/A1 默认扰动历史最大 wrench 分别为 `2.449706/4.899412`，A1 零扰动六轴与历史最大值全部为零，三段均为 60/16、8 steps、0 reset，frozen actor SHA 不变。Isaac 5.1 empty-wrench Warp 兼容问题通过新增 RED 测试与 full-zero fallback 修复。

T400.6 零间隙安装设计已获交互批准并写入书面规格：把构建 clearance 从 `0.01 m` 改为 `0.0 m`，保持单 articulation/fixed mount/25 DOF，不允许用零间隙掩盖网格穿透。当前等待书面规格复核，尚未修改构建脚本或资产。

T400.7 A1 抗扰恢复训练设计已获交互批准并写入书面规格。既有 A1 在 `model_5999` 前出现明显后期退化；根因调查确认 vendored ActorCritic 把 raw std 经过 softplus 后形成约 `0.694` 的实际动作噪声，且 resume 会重置扰动课程。选定满幅三 seed checkpoint 筛选、独立 fork、scalar std 修复、课程进度恢复和 500-iteration 分块验收。当前等待书面规格复核，尚未修改训练代码或启动 recovery run。

T400.7 书面规格现已由用户复核确认，七任务单代理 TDD 实施计划已写入。计划按噪声语义、课程/axis 诊断、纯评估、严格 Play/sweep、fork/manifest、静态+真实 smoke、GPU0 分块训练顺序执行。当前尚未修改运行代码或启动 recovery run。

T400.7 Tasks 1–6 已完成：scalar/log 噪声语义、课程恢复、六轴覆盖、strict full-scale Play/sweep、隔离 fork 和 lineage manifest 均已实现。静态回归 `183 passed`、compile exit `0`；GPU0 64×500 满幅 Play 与 8-env 单 iteration fork 均 exit `0`，fork 从 `2700` 生成 `2701`，有效 std floor、课程进度、源/frozen SHA 均验证通过。当前进入 Task 7 正式四候选三 seed 筛选与 500-iteration 分块训练，尚未达到行为验收。

T400.7 Task 7 已执行 20 个 500-iteration recovery blocks。最佳 `model_9700.pt` 的三 seed timeout/contact/orientation 为 `0.701863/0.222360/0.075776`；相比源模型显著改善但未通过 `0.80/0.10/0.10` 联合门。后 7 个 block 在 contact `0.22–0.27` 平台，没有继续相同 PPO 的充分依据；manifest stop reason 为 `recovery_plateau_requires_design_review`。下一步需要批准 reward/termination 或 curriculum redesign，T400.7 仍保持开放。

T400.8 论文优先级 WBC Teacher–Student 交互设计已获批准并写入书面规格。新路线使用 50 Hz 三级运动分配、200 Hz 滚轮式动力学 WBC/QP、动态抗扰平衡最高执行优先级和后续 100-observation/23-action Student；现有 A0/A1 保持独立，不进行 shape 绕过或 checkpoint 冒充迁移。当前等待书面规格复核，尚未修改运行代码。

## Open Children

- [ ] T400.8 复核优先级 WBC Teacher–Student 书面规格，并生成逐文件 TDD 实施计划。
- [ ] T400.7 完成正式满幅 checkpoint 筛选和 500-iteration 分块 GPU0 验收（恢复代码、独立 fork 与真实 smoke 已完成）。
- [ ] T400.6 复核并实施 M1 + Panda 零间隙安装资产修订，完成拓扑、穿透、no-snap 和 GPU0 Play 复验。
- [ ] T400.3 实施前完成 Panda/M1/六轴传感器最坏工况机械验算。

## Closed Children Archive

- [x] T400.0 需求澄清、路线比较和分节设计确认。
- [x] T400.1 用户审阅并批准书面设计。
- [x] T400.2 完成首个 asset/wrench foundation 实施计划。
- [x] T400.2a 用户选择并启动 foundation plan 执行。
- [x] T400.2b.1 完成 Task 1：项目自有离线 M1/Panda 资产输入。
- [x] T400.2b.2 完成 Task 2：本地 Panda 转换与 M1 + Panda 单 articulation USD 装配。
- [x] T400.2b.2r 修复 Task 2→3 持久化 arcs/双 root 阻断，并在 CPU 完成 Task 3 验收；15 个 `OmniPBR.mdl` 明确分类为 Isaac Sim built-in resolver boundary。
- [x] T400.2b.4 完成 Task 4：combined asset cfg、isolated M1-only smoke cfg、Gym 注册与静态/导入验证。
- [x] T400.2b.5 完成 Task 5：base-frame/base-origin mount wrench 纯变换、adapter、导出和 smoke observation。
- [x] T400.2b.6 完成 Task 6：独立每轴 reset 的真实 CPU 六轴 probe、七行 artifact、sign/magnitude/no-reset 校准；network denial 保持 unverified。
- [x] T400.4a 用户复核受限残差动作组合器书面规格并完成 4-task TDD 实施计划。
- [x] T400.4b 单代理完成受限残差动作组合器 RED→GREEN、回归和笔记对齐。
- [x] T400.5b.1 完成 A0/A1 每环境六维扰动 scheduler 与公共 wrench helper。
- [x] T400.5b.2 完成 checkpoint/manifest、strict frozen actor 与 SHA-256 契约。
- [x] T400.5b.3 完成 Teacher reward、A0/A1 env cfg 与两个 lazy Gym ID。
- [x] T400.5b.4 完成 A0 zero-base VecEnv wrapper、外力调用顺序和逐环境 reset。
- [x] T400.5b.5 完成 A1 strict frozen actor、双 composer reset 与 hash 不可变检测。
- [x] T400.5b.6 完成 Teacher PPO 配置、训练入口、strict resume 与 manifest 状态机。
- [x] T400.5 完成已批准 Teacher A0/A1 随机六维扰动平衡训练链。
- [x] T400.5a 完成 A0/A1 专项 spec 自审和逐文件实施计划。
- [x] T400.5b TDD 完成扰动、奖励、wrapper、checkpoint/manifest 与训练入口。
- [x] T400.5c 完成 A0→A1 CPU initial/resume 短程训练、冻结参数和 runtime contract 验证。
- [x] T400.5d 完成 strict A0/A1 Teacher play、默认/零扰动开关、文档、静态回归与 GPU0 三段 smoke。

## Related Logs

- [设计记录与自审](../log/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
- [Asset/wrench foundation plan](../log/2026-08-14-m1-panda-asset-wrench-foundation-plan.md)
- [Task 1 offline asset inputs](../log/2026-08-14-m1-panda-offline-asset-inputs.md)
- [Task 2 single articulation build](../log/2026-08-14-m1-panda-single-articulation-build.md)
- [Task 3 offline/topology verification](../log/2026-08-14-m1-panda-offline-topology-verification.md)
- [Task 2→3 asset repair](../log/2026-08-14-m1-panda-task23-asset-repair.md)
- [Task 4 combined cfg and isolated smoke](../log/2026-08-14-m1-panda-combined-smoke-cfg.md)
- [Task 5 base-frame mount wrench](../log/2026-08-14-m1-panda-base-frame-mount-wrench.md)
- [Task 6 deterministic wrench probe](../log/2026-08-14-m1-panda-wrench-probe.md)
- [Residual action composer design](../log/2026-08-14-m1-panda-residual-action-composer-design.md)
- [Residual action composer plan](../log/2026-08-14-m1-panda-residual-action-composer-plan.md)
- [Residual action composer implementation](../log/2026-08-14-m1-panda-residual-action-composer-implementation.md)
- [Teacher A0/A1 approved design](../../docs/superpowers/specs/2026-08-14-m1-panda-teacher-a0-a1-training-design.md)
- [Teacher A0/A1 implementation plan](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-a0-a1-training.md)
- [Teacher disturbance scheduler](../log/2026-08-14-m1-panda-teacher-disturbance-scheduler.md)
- [Teacher checkpoint contract](../log/2026-08-14-m1-panda-teacher-checkpoint-contract.md)
- [Teacher env wiring](../log/2026-08-14-m1-panda-teacher-env-wiring.md)
- [Teacher A0 wrapper](../log/2026-08-14-m1-panda-teacher-a0-wrapper.md)
- [Teacher A1 frozen wrapper](../log/2026-08-14-m1-panda-teacher-a1-frozen-wrapper.md)
- [Teacher training entrypoint](../log/2026-08-14-m1-panda-teacher-training-entrypoint.md)
- [Teacher final static regression](../log/2026-08-14-m1-panda-teacher-a0-a1-static-regression.md)
- [Teacher real CPU smoke](../log/2026-08-14-m1-panda-teacher-a0-a1-cpu-smoke.md)
- [Teacher training runbook](../../docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md)
- [Teacher play approved design](../../docs/superpowers/specs/2026-08-14-m1-panda-teacher-play-design.md)
- [Teacher play design log](../log/2026-08-14-m1-panda-teacher-play-design.md)
- [Teacher play implementation plan](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-play.md)
- [Teacher play plan log](../log/2026-08-14-m1-panda-teacher-play-plan.md)
- [Teacher play wrapper disturbance gate](../log/2026-08-14-m1-panda-teacher-play-wrapper.md)
- [Teacher play strict entrypoint](../log/2026-08-14-m1-panda-teacher-play-entrypoint.md)
- [Teacher play GPU0 smoke](../log/2026-08-14-m1-panda-teacher-play-gpu0-smoke.md)
- [Zero-clearance mount design](../../docs/superpowers/specs/2026-08-15-m1-panda-zero-clearance-mount-design.md)
- [Zero-clearance mount design log](../log/2026-08-15-m1-panda-zero-clearance-mount-design.md)
- [A1 recovery training design](../../docs/superpowers/specs/2026-08-15-m1-panda-a1-recovery-training-design.md)
- [A1 recovery training design log](../log/2026-08-15-m1-panda-a1-recovery-training-design.md)
- [A1 recovery implementation plan](../../docs/superpowers/plans/2026-08-15-m1-panda-a1-recovery-training.md)
- [A1 recovery plan log](../log/2026-08-15-m1-panda-a1-recovery-plan.md)
- [A1 recovery implementation](../log/2026-08-15-m1-panda-a1-recovery-implementation.md)
- [A1 recovery blocks](../log/2026-08-15-m1-panda-a1-recovery-blocks.md)
- [Prioritized WBC Teacher–Student design](../../docs/superpowers/specs/2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md)
- [Prioritized WBC Teacher–Student design log](../log/2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md)

## Git Refs

- Last Feature Commit: `9effc43`
- Last Verified Base: `8872421d02eb93b04b150d025148c8a93e78dd09`
- Current Work Ref: `main` working tree（未提交）
- Key Files:
  - [设计文档](../../docs/superpowers/specs/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
  - [受限残差动作组合器设计](../../docs/superpowers/specs/2026-08-14-m1-panda-residual-action-composer-design.md)
  - [受限残差动作组合器实施计划](../../docs/superpowers/plans/2026-08-14-m1-panda-residual-action-composer.md)
  - [受限残差动作组合器代码](../../Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py)
  - [受限残差动作组合器测试](../../Go2Pvcnn/tests/test_m1_residual_action.py)
  - [第一阶段实施计划](../../docs/superpowers/plans/2026-08-14-m1-panda-asset-wrench-foundation.md)
  - [Task 4 combined asset cfg](../../Go2Pvcnn/go2_pvcnn/assets/m1_panda.py)
  - [Task 4 isolated smoke cfg](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py)
  - [Task 5 wrench transform and adapter](../../Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py)
  - [Task 6 probe](../../Go2Pvcnn/scripts/m1_panda_wrench_probe.py)
  - [Teacher A0/A1 专项设计](../../docs/superpowers/specs/2026-08-14-m1-panda-teacher-a0-a1-training-design.md)
  - [Teacher A0/A1 实施计划](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-a0-a1-training.md)
  - [Teacher A0/A1 Play 设计](../../docs/superpowers/specs/2026-08-14-m1-panda-teacher-play-design.md)
  - [Teacher A0/A1 Play 实施计划](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-play.md)
  - [Teacher A0/A1 Play 入口](../../Go2Pvcnn/scripts/m1_panda_teacher_play.py)
  - [Teacher A0/A1 Play 测试](../../Go2Pvcnn/tests/test_m1_panda_teacher_play_static.py)
  - [优先级 WBC Teacher–Student 设计](../../docs/superpowers/specs/2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md)

## Next Step

先复核 T400.8 优先级 WBC Teacher–Student 书面规格并生成实施计划；控制器实现前确定 T400.6 零间隙资产冻结点。T400.7 仍保留为旧 A1 诊断线，最大载荷实机测试前仍必须完成 T400.3 机械验算。

## Node Details

### T400.1 书面设计审阅

重点检查统一 articulation 与六轴传感器接口、Teacher/Student 可见信息边界、12 腿位置加 4 轮速残差动作、安全降级状态机，以及 0–3 kg 静止抓取验收范围。

### T400.3 机械验算前置门

策略设计不代表 M1 机架、转接板、传感器和执行器已经具备承载能力。任何最大负载或快速摆臂实机测试前，必须独立验证 Panda 本体质量、物体质量、最大力臂、动态载荷和安全系数。

### T400.4 受限残差动作组合器

首个后续控制子阶段只实现 `base_action[16] + normalized_residual[16] -> combined_action[16]` 的纯 PyTorch 组合器。残差按 12 个腿位置和 4 个轮速度通道映射到物理单位，执行逐环境幅值和每周期变化率限制，并支持指定环境 reset、梯度传播和只读饱和诊断。Isaac Lab 接入与 Teacher 训练留给后续独立阶段。

书面规格已由用户复核；配置/状态、动作合成、reset/异常原子性和回归/笔记四个任务均已完成。最终 focused `34 passed`，组合回归 `89 passed`，未运行 Isaac Sim dynamics。

### T400.5 Teacher 随机六维扰动基线

下一软件子阶段只规划 Teacher 特权观测、随机六维扰动课程、冻结基础策略和 residual composer 接入。Student 估计器、蒸馏、Panda IK/OSC 与抓取仍保持独立后续阶段。

### T400.5d Teacher A0/A1 Play

独立 play 入口严格复现训练 wrapper：A0 使用 zero-base，A1 要求 frozen A0 checkpoint 与 A1 checkpoint。默认开启 stage 对应六维扰动，`--disable-disturbance` 只用于零 wrench 对照；GUI 默认，有限 steps 支持 GPU0/headless smoke。实现、文档、`195 passed` 静态回归与 GPU0 三段 smoke 均已完成；当前 A1 仍只作为诊断 checkpoint，不升级行为验收状态。

### T400.6 零间隙安装

把 `MOUNT_CLEARANCE_M` 从 `0.01` 改为 `0.0`，使 Panda 安装原点直接对齐 M1 `BASE_LINK` 顶面。实施必须重建 USD/checksum，并复验单 articulation、25 DOF、fixed mount、网格无明显穿透、一步 no-snap 与 GPU0 Teacher Play。旧 checkpoint 只保持结构兼容，不自动获得行为验收。

### T400.7 A1 抗扰恢复训练

既有 A1 后期 survival 下降且 base contact 上升，不能从 `model_5999` 盲目续训。恢复路线先以三 seed 满幅 `20 N / 5 Nm` 评估选择 `2700/3800/4500/5999` 候选，再从胜者创建只写新目录的 fork。修复 vendored ActorCritic 的 scalar/log std 语义，恢复 disturbance global progress，重置 optimizer 为 `1e-4`，每 500 iterations 进行满幅复评。最终要求 timeout `>=80%`、base contact 和 bad orientation 各 `<=10%`。

### T400.8 优先级 WBC Teacher–Student

采用论文的三级运动分配和动力学全身控制：Panda 优先跟踪六维末端轨迹，接近关节限制或奇异位形时激活 M1 平面运动；执行层以轮地接触和动态抗扰平衡为最高优先级。第一阶段实现确定性 WBC Teacher，随后用现实可得本体感知、安装六维力和末端目标蒸馏 23 维全身 Student。新任务、日志和 checkpoint contract 与旧 A0/A1 完全隔离。
