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

## Open Children

- [ ] T400.5d 按已确认书面规格和三任务单代理 TDD 计划实现 Teacher A0/A1 play，并完成 GPU0 smoke。
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

## Git Refs

- Last Feature Commit: unavailable（`/home/xk/coding/M1` 不是 Git 工作树）
- Last Verified Commit: unavailable
- Current Work Ref: filesystem working copy
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

## Next Step

按已确认计划单代理 TDD 实施 T400.5d Teacher A0/A1 play，并完成 GPU0 默认扰动与零扰动 smoke；随后回到 Student 估计/蒸馏。最大载荷实机测试前仍必须完成 T400.3 机械验算，Panda IK/OSC 与抓取任务保持开放。

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

独立 play 入口严格复现训练 wrapper：A0 使用 zero-base，A1 要求 frozen A0 checkpoint 与 A1 checkpoint。默认开启 stage 对应六维扰动，`--disable-disturbance` 只用于零 wrench 对照；GUI 默认，有限 steps 支持 GPU0/headless smoke。书面设计和三任务实施计划均已确认，进入单代理 inline TDD。
