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

T400.6 零间隙重基线已完成。最终 surface gap `-8.64e-07 m`、PXR 单 root、CPU/relocation 25 DOF、一步 delta `2.40e-05 m`，且用户明确确认视觉贴合；完整回归 `184 passed`。GPU0 C0 2000 步和 C1a 无摆臂/联合两次 4000 步均 exit `0`，QP `1.0`、全程 TRACK，C1a `hard_gates_passed=true`。接受的组合 USD SHA 为 `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`；Student S1 现仅按批准的 flat-ground DAgger 计划解锁。

T400.7 A1 抗扰恢复训练设计已获交互批准并写入书面规格。既有 A1 在 `model_5999` 前出现明显后期退化；根因调查确认 vendored ActorCritic 把 raw std 经过 softplus 后形成约 `0.694` 的实际动作噪声，且 resume 会重置扰动课程。选定满幅三 seed checkpoint 筛选、独立 fork、scalar std 修复、课程进度恢复和 500-iteration 分块验收。当前等待书面规格复核，尚未修改训练代码或启动 recovery run。

T400.7 书面规格现已由用户复核确认，七任务单代理 TDD 实施计划已写入。计划按噪声语义、课程/axis 诊断、纯评估、严格 Play/sweep、fork/manifest、静态+真实 smoke、GPU0 分块训练顺序执行。当前尚未修改运行代码或启动 recovery run。

T400.7 Tasks 1–6 已完成：scalar/log 噪声语义、课程恢复、六轴覆盖、strict full-scale Play/sweep、隔离 fork 和 lineage manifest 均已实现。静态回归 `183 passed`、compile exit `0`；GPU0 64×500 满幅 Play 与 8-env 单 iteration fork 均 exit `0`，fork 从 `2700` 生成 `2701`，有效 std floor、课程进度、源/frozen SHA 均验证通过。当前进入 Task 7 正式四候选三 seed 筛选与 500-iteration 分块训练，尚未达到行为验收。

T400.7 Task 7 已执行 20 个 500-iteration recovery blocks。最佳 `model_9700.pt` 的三 seed timeout/contact/orientation 为 `0.701863/0.222360/0.075776`；相比源模型显著改善但未通过 `0.80/0.10/0.10` 联合门。后 7 个 block 在 contact `0.22–0.27` 平台，没有继续相同 PPO 的充分依据；manifest stop reason 为 `recovery_plateau_requires_design_review`。下一步需要批准 reward/termination 或 curriculum redesign，T400.7 仍保持开放。

T400.8 C0 Tasks 1–10 已按 TDD 完成并通过 GPU0 硬验收。最终纯/静态回归 `177 passed`；8-step 静止与 2000-step 六维小幅运动均 exit `0`。正式 2000-step 指标为 EE `0.009566 m`、sigma `0.141259`、QP `1.0`、slip `0.007126 m/s`、roll/pitch `0.028045/0.008563 rad`，且 limit/base/self-collision/reset/snap 全零、2000 步均为 TRACK/safe。C0 只证明驻停小幅操作基础；C1/C2 滚动约束是下一独立设计任务。

T400.8c C1a 已按独立书面设计与十二任务单代理计划完成。最终纯/静态回归 `184 passed`；GPU0 无 Panda 轨迹和默认联合轨迹的两次 4000-step 平地直线前进/停车/倒车均 exit `0`、`hard_gates_passed=true`，联合运行 QP 可行率 `1.0`、四轮持续接触、最大 EE 误差 `0.0016413305229732028 m`、最大滚动残差 `0.0016662825831554645 m/s`、最大侧滑 `0.0012335278538215793 m/s`，全程 `TRACK/safe`。C1a 只完成 deterministic Teacher 直线滚动；下一项 C1b 转向必须单独设计并获用户批准，不授权 C2/C3、Student、抓取或实机工作。

Coordinated Teacher PPO 前置已完成：训练入口绑定 `Isaac-M1-Panda-Coordinated-v0`，通过 1 iteration GPU0 smoke，combined policy 为 67 observation / 23 joint-effort action，生成 `model_0.pt`。A1 `model_10402.pt` 仅作为严格存在性/lineage 初始化元数据，不直接迁移 60→16 actor 权重。随后完成资产 PXR 和 2-env×20-step reset/step 复验：`root_joint` 明确 disabled，active `AssemblerFixedJoint`、单 articulation root、25 DOF、finite 均通过，最大 mount 相对位移 `0.000370 m`。PhysX warning 仍打印，长时段 dynamics gate 尚未关闭，正式长训暂缓。

2026-08-23 长时 dynamics gate 已关闭：GPU0 `hold` 与 `controlled` 两次 8-env×2000-step 均通过，最大 mount 相对漂移分别为 `2.50e-7/2.60e-7 m`，姿态漂移 `5.34e-7/5.16e-7 rad`，reset/base-contact/bad-orientation/joint-limit/non-finite 全零。已知 warning 未产生实测 snap。新审计同时确认 67 维观测不含 Panda/任务状态，继承 smoke reward 不含导航/EE tracking；正式长训前必须补齐学习合同。启动 mount wrench 峰值 `2021 N / 902 Nm` 保留为传感瞬态/归一化 follow-up，不作为机械载荷验收。

T400.10 已补齐精确 103-observation / 23-action 协调 PPO 合同并完成真实 GPU0 迭代诊断。失败诊断保留了高学习率跌倒、wheel authority 不足、到达后 wheel residual 漂移和 std 增长四类证据。最终采用 zero-action actor、固定 `1e-4/0.01 std`、balance-gated mission reward、0.1 m/s nominal wheel feedforward、到达后 wheel mask、action L2，以及腿/轮/臂 `5/50/2` residual scale。最终 64-env×600 gate 第 595 轮 timeout `1.0`、contact/orientation `0`、base-target `2.4825`、EE `1.0423`。GPU0 64-env×5000 long v4 后续完成，但约 update 4022 后发生策略坍塌并被拒绝；尚未声明收敛或实机能力。

T400.10 long v4 后续已完成 5000 updates，但在约 4022 updates 后发生 PPO 策略坍塌，最终 `base_contact=1.0`，不可接受。T400.10a 稳定重训交互设计现已批准并写入书面规格：新运行从零开始，保持 200 Hz/103/23，改用 256-step rollout、`gamma=0.9995`、`lambda=0.995`、adaptive KL、best checkpoint 与自动早停回退；Panda hand 接受逐环境 `20 N/5 Nm` 满幅课程，reset/摩擦进行可复现域随机化。旧 `model_3500.pt` 只作为基准保留，之后 checkpoint 将在实施阶段按 SHA 审计删除。当前等待书面规格复核，尚未改运行代码、启动新训练或删除文件。

T400.10a 书面规格已由用户复核批准，九任务单代理 TDD 实施计划已完成并自审。计划按专用 256-step cfg、bounded adaptive PPO、通用 runner callback、100-episode guard/原子回退、原子 joint reset/摩擦 DR、Panda-hand wrench、训练入口、GPU0 probe/64×50 短门、checkpoint 审计清理与 fresh 64×600 启动顺序执行。当前等待 Inline Execution handoff；仍未修改运行代码、删除 checkpoint 或启动新训练。

T400.10a Task 1 已完成：新增 coordinated 专用 PPO cfg，RED `2 failed`、GREEN `2 passed`、相关训练入口回归 `40 passed`，compile/diff exit `0`。精确冻结 256-step、`0.9995/0.995`、adaptive KL/LR bounds 和 physical std bounds；尚未接入入口。下一项是 Task 2 bounded adaptive PPO 实现。

T400.10a Task 2 已完成：RED `7 failed, 7 passed`，最终相关回归 `58 passed`，compile/diff exit `0`。PPO adaptive KL 使用配置化 LR `[1e-6,3e-4]`，physical std 使用 min/max clamp，runner 写入 KL 与 LR 调节方向；默认边界保持旧调用兼容。下一项是 Task 3 generic iteration summary/callback。

T400.10a Task 3 已完成：缺失接口 RED 在测试收集阶段有效触发，focused `13 passed`，Teacher/coordinated 代表入口回归 `46 passed`，compile/diff exit `0`。通用 runner 现在提供 frozen iteration summary/result、finite detached episode/environment metrics、可选 stop-reason callback，并保持普通周期/最终保存。下一项是 Task 4 best-checkpoint guard 与原子 rollback controller。

T400.10a Task 4 已完成：模块缺失 RED 有效触发，focused `8 passed`，Tasks 1–4 相关回归 `23 passed`，compile/diff exit `0`。纯 guard 使用同一最近 100 episode 窗口与批准的字典序/eligible 门，原子 controller 保存 best/JSON 并以 `load_optimizer=False, keep_std=True` 回退 final；无 eligible 时保持 `accepted=false`。下一项是 Task 5 reset/friction 域随机化。

T400.10a Task 5 已完成：missing event/helper RED 为 `3 failed, 2 errors, 4 passed`，focused `9 passed`，coordinated cfg regression `13 passed`，compile/diff exit `0`。新增 canonical joint 单次原子 reset 写入及批准的 root/joint/friction 范围；默认 cfg 与 helper disabled 均保持确定性。下一项是 Task 6 seeded Panda-hand wrench curriculum。

T400.10a Task 6 已完成：scheduler 缺失 RED 有效触发，scheduler `7 passed`，wrapper focused `11 passed`，Teacher/coordinated regression `75 passed`，compile/diff exit `0`。Panda hand 获得 seeded 逐环境 `20 N/5 Nm` 三模式课程，wrapper 默认关闭且不触碰外力状态，done selective reset 与 finite DR diagnostics 均通过。下一项是 Task 7 stable train entrypoint/manifest/final rollback。

T400.10a Task 7 已完成：旧入口 RED `4 failed, 2 passed`，focused GREEN `6 passed`，Tasks 1–7 related `115 passed`，全部 coordinated tests `53 passed`，compile/diff exit `0`。fresh 入口默认 64×600/seed42，A1 仅 provenance；schema2 manifest、trainable std、显式 DR、guard callback 与 best→final SHA rollback 已串通。本阶段尚未做 GPU0 probe/短训。下一项是 Task 8 GPU0 randomization/physics probe 与 64×50 gate。

T400.10a Task 8 已完成：GPU0 8×256 randomization/physics probe 全部 hard gates true，真实 controlled velocity DR max `0.049825 rad/s`，Panda-hand wrench 产生 finite/nonzero mount response；首次 runner rollout 前现显式 reset。64×50 guard/checkpoint/SHA 链通过，但 timeout `0.605156` 未达 eligible 门，故 `accepted=false`，没有收敛声明。Task 8 回归 `137 passed`。

T400.10a Task 9 清理与本地验收已完成：long-v4 中恰好 15 个 >3500 数字 checkpoint 已按两阶段 SHA 审计删除，`model_3500.pt` 和原 manifest 哈希保持；完整 focused suite `141 passed`，asset SHA 保持 `643fd061...`。下一状态是 fresh 64×600 长训监控，只以 guard/manifest 判断接受，不外推抓取或实机能力。

T400.10a fresh 64×600 已在 GPU0 启动，PID `1128844`。首个 update 生成 `model_0.pt`，16,384 timesteps，KL/LR/std 和 reset DR diagnostics 均 finite/bounded；当前 manifest 仍为 `running`，尚无 eligible/accepted 结论。

T400.10b 替代路线已由用户批准：先训练 M1 携带 PD 折叠、保持真实动力学的 Panda 完成前进、后退和转向，不施加外力、不解锁 arm action。Task 1 已以纯 PyTorch TDD 完成八阶段父链、C0-C4 命令、D1-D3 重置范围、命令族采样和 64 环境固定评估表，focused `7 passed`；尚未接入 PPO 或 Isaac。

T400.10b Task 2 已完成：vendored ActorCritic 新增非持久 active-action mask，`[1]*16+[0]*7` 可保持 23 输出/旧 checkpoint key，同时 inactive action、mean、log-prob、entropy、actor/std gradient 和最终 actor 行均严格隔离。有效 RED `9 failed`，最终 mask+legacy 回归 `63 passed`；尚未接入 folded-load PPO 配置或 Isaac action manager。

T400.10b Task 3 已完成：新增专用 256-step/200 Hz PPO cfg、fresh zero-output actor 和 `[1]*16+[0]*7` policy contract；PPO KL 只统计 active 维度，超过 `0.015` 时中止当前 update 剩余 minibatch，并在下一 update 复位状态。最终 config/mask/adaptive/legacy 回归 `35 passed`；尚未创建 Isaac task。

T400.10b Task 4 已完成纯/静态层：新增独立 `Isaac-M1-Panda-Folded-Load-v0`，保持 103 observation、23 action、200 Hz 和未修改的动态 `M1_PANDA_CFG`；base/EE 兼容槽为有限零，desired twist 承载 episode 命令。reward 只含批准的 locomotion/balance/active-action 项，默认 reset/friction 确定且无外力事件。最终新旧回归 `29 passed`；真实 Isaac physics 尚未运行。

T400.10b Task 5 已完成 CPU contract：wrapper 在 `env.step` 前 clone 并 exact-zero `16:23`，按 done env 选择性换命令并输出逐 episode tracking/方向/termination 记录；stage cfg 映射 D1-D3 exact root/leg/friction。新增 leg-only reset event，只写 selected env，轮/Panda 位置和全部 joint velocity 保持默认。最终新旧回归 `34 passed`；真实 Isaac reset/material/contact 仍待 GPU probe。

T400.10b Task 6 已完成纯 guard：catastrophe 不依赖 eligible best，`>0.50×2`、`>0.20×5`、nonfinite/mask/fold immediate stop 均通过；200/400 episode shared + direction + stationary gates和 42/43/44 atomic SHA acceptance 通过。focused `8 passed`；runner/manifest/GPU 尚未接入。

T400.10b Task 7 已完成 CPU 入口合同：L0 强制 fresh，后续 stage 只接受紧邻 `accepted=true` 且 manifest/checkpoint SHA 一致的父阶段；wrapper 用 env-id 保留并一次性排出完整 episode 记录。单阶段 train 串接 guard/best checkpoint/manifest，eval 固定 64 env、确定性均值、平衡命令和 seed 42/43/44；runner 新增 KL max/abort、minibatch、grad、active std 与物理失败诊断。有效 RED `6 failed, 4 passed`，最终 related `72 passed`；真实 Isaac/GPU 尚未运行。

T400.10b Task 8 已完成 CPU 编排合同：严格八阶段顺序，每阶段 train 后固定执行 42/43/44 eval；每次晋级重新验证从 L0 到当前的 accepted 状态、父 manifest path/SHA、父 final path/SHA 与当前 final SHA。首个失败原子写 `curriculum_state.json`、保留上一 accepted rollback 且不创建下一 stage。有效 RED `5 failed`，最终 related `17 passed`；真实 subprocess/GPU 尚未运行。

T400.10b Task 9 已实现 folded-load-only Panda shoulder `120/8` 配置副本、每步 fold position/零速度 target、原子 GPU0 probe 和 runbook；全局 `M1_PANDA_CFG` 仍为 `80/4`。Focused `16 passed`。8×16 probe 通过，fold `0.065997 rad`、joint margin `0.045702 rad`；8×256 probe 失败，fold `0.214941 rad`、joint margin `-0.103241 rad`、effort utilization `1.0`、inactive action `0.0`。joint4 target 到 soft lower limit 仅约 `0.1117 rad`，且 `87 Nm` 已饱和，因此当前目标/力矩约束下单纯提高 Kp 不能关闭门槛。按 stop policy 未启动 8×1、64×10 或长训。

用户随后批准只将 folded-load joint4 target 改为 `-2.650 rad`。TDD 证明 task-local init state 隔离，全局仍为 `-2.810 rad`。复验四级门全部关闭：8×16 margin `0.201676`；8×256 fold/margin `0.231634/0.040066`；8×1 smoke fold/margin `0.190140/0.081560`；64×10 最终 fold/margin `0.185009/0.086691`，effort `0.255274`，hard failure 和 inactive action 均为 `0`。两个 smoke 正确保持 `accepted=false`。64×10 有 9/10 update 触发 KL abort，长训需重点监控但不构成物理门失败。

T400.10b Task 10 已启动 fresh 长期编排器：GPU0、4096 env、每 stage 最多 600 iterations，当前 L0-C0 manifest 为 `running`，curriculum/train PID `2660373/2660477`。首个 update 完成，update4 fold/margin `0.185018/0.086682`、effort `0.255313`、hard failure/inactive action `0`；GPU 占用约 `8.63/12.23GB`，无 OOM。KL `0.02775` 且 abort 仍激活，当前只是健康启动证据，不是 eligible/accepted 或收敛结论。

T400.10b L0-C0 后续在 update74 正常停止，原因是 update23 产生 eligible `model_best.pt` 后连续 50 update 无更优 rank，触发 `eligible_patience_50_updates`；不是 crash、OOM、nonfinite 或物理 hard failure。随后 seed 42/43/44 固定评估均 `passed=false`，编排器正确记录 `stopped`。每份报告 timeout/contact/orientation `1/0/0`，全局 vx/wz RMSE `0.035408/0.106067` 和静止漂移均通过；逐条对照 `evaluate_records` 后，只可能是 forward/reverse 中至少一个 vx RMSE `>0.04`，或 left/right 中至少一个 wz RMSE `>0.12`。当前 artifact 未写出分方向 RMSE，不能进一步诚实归因到某个方向。

T400.10b 分方向 evaluation diagnostics 交互设计已获用户确认并写入书面规格。选定在 `evaluate_records` 唯一计算边界输出 forward/reverse/left/right 的 count、metric、RMSE、limit、contact/orientation rate 和 pass，顶层 `directional_pass` 由同一布尔值汇总。所有旧门槛、训练、reward、命令和 checkpoint 选择不变；现有 SHA-pinned `model_best.pt` 只在隔离诊断目录复评，不覆盖原 artifact。

T400.12 8D Residual WBC 第一版 Phase 1–4 已完成。新增与现有 103/23 Coordinated PPO、Folded Load 和 C0/C1a 并行的独立控制链：8D contract、6D virtual wrench 到既有 WBC/QP、height/stance、连续 base participation、安装点六维力反馈、103D observation、runtime wrapper、Gym 注册与 play 入口。CPU 最终回归 `185 passed`；GPU0 零残差 256 步和八轴正负 `0.1` 共 16 组探针全部通过，QP 可行率 `1.0`，四轮接触、无机身触地/越界/reset。本版不包含 Arm MPC 或 PPO 长训。

T400.13 Phase 5 已完成，Phase 6 v4/v5/v6 的 pilot、100-update short
和 seeds 42/43/44 共 24 个固定条件进程均完整执行。三个 promotion
manifest 均为 `accepted=false`，所以 3000-update long 正确保持未启动。
受保护的 update100→300 bridge、经验 normalizer count 持久化、v6 u100
原子迁移、schema-v3 promotion 和 long 的 optimizer/count 连续恢复已按
批准规格完成；完整 CPU 预检 `134 passed`，compile/diff 通过。下一步是
在 GPU0 执行 v7 bridge，尚无 bridge/promotion/long 的运行验收结论。

## Open Children

- [x] T400.10 补齐 Coordinated Teacher 可学习 observation/action/reward 合同，通过短训行为 sanity 后启动并完成 GPU0 长训（long v4 已完成但因后期策略坍塌被拒绝，由 T400.10a 接续）。
- [ ] T400.10a 监控 Coordinated PPO 稳定重训：实现、GPU probe/短门和旧 checkpoint 审计清理已完成，fresh 64×600 进入长期行为门监控。
- [ ] T400.10b 完成折叠 Panda 动态负载基础运动课程：Tasks 1–8 已完成；Tasks 9–10（GPU probe/runbook、完整回归与长训验收）仍开放。
  - [x] Task 1：新增独立 coordinated 200 Hz PPO 配置并通过 RED→GREEN。
  - [x] Task 2：实现 bounded adaptive KL/LR 与 physical std clamp/诊断。
  - [x] Task 3：实现通用不可变 runner iteration summary/callback。
  - [x] Task 4：实现纯 best-checkpoint guard 与原子 rollback controller。
  - [x] Task 5：实现 training-only reset/friction 域随机化。
  - [x] Task 6：实现 seeded Panda-hand wrench curriculum 与 wrapper diagnostics。
  - [x] Task 7：接入 stable train entrypoint、manifest 与 automatic final rollback。
  - [x] Task 8：执行 GPU0 randomization/physics probe 与 guarded 64×50 短训；基础设施通过，策略 `accepted=false`。
  - [x] Task 9：完成 >3500 checkpoint SHA 审计清理、完整回归与稳定训练 runbook。
  - [ ] Long monitor：等待 fresh 64×600 guard 停止并审计 eligible best/final manifest。
- [ ] T400.11 调查 reset/启动 mount wrench 峰值的物理与传感来源，并冻结训练归一化/裁剪合同；不得外推为实机允许载荷。
- [x] T400.12 按 PDF 推荐顺序完成 8D Residual WBC 第一版 Phase 1–4；CPU 和 GPU0 硬门通过，Arm MPC/PPO 留待独立阶段。
- [ ] T400.13 完成 Phase 6 Residual PPO 晋级和条件长训；Phase 5、三轮
  100-update short 和三轮 24-worker promotion 已完成，均未晋级。bridge/
  normalizer 实现与 CPU 预检已完成；下一步执行 update100→300 GPU0
  bridge 和不变的24进程门，并仅在 `accepted=true` 后启动3000-update long。

- [x] T400.8a 复核优先级 WBC Teacher–Student 书面规格，并生成 C0 逐文件 TDD 实施计划。
- [x] T400.8b 单代理执行 C0 deterministic Teacher foundation，完成静态回归与 GPU0 8+2000-step 验收。
  - [x] Task 1 冻结维度、关节名、规范控制顺序与张量合同。
  - [x] Task 2 实现协调运动学与奇异性诊断。
  - [x] Task 3 实现速度界交集与三级运动分配。
  - [x] Task 4 实现项目自有 float64 参考 QP 后端。
  - [x] Task 5 实现 C0 standing WBC 动力学、接触和力矩恢复。
  - [x] Task 6 实现阻抗输出与 balance-first 安全状态机。
  - [x] Task 7 实现带限轨迹与 deterministic Teacher 编排。
  - [x] Task 8 实现独立 Isaac Lab effort 环境与 Gym 注册。
  - [x] Task 9 实现 PhysX 适配、deterministic play、诊断与原子摘要。
  - [x] Task 10 完成 GPU0 8-step 静止与 2000-step 运动硬验收、runbook 和证据。
- [x] T400.8c 单代理完成 C1a deterministic Teacher 平地直线滚动、Panda 联合运动、GPU0 4000-step 硬验收、C0 回归与运行手册。
- [ ] T400.8d 为 C1b 转向编写独立设计并取得用户批准（尚未授权实现）。
- [ ] T400.7 完成正式满幅 checkpoint 筛选和 500-iteration 分块 GPU0 验收（恢复代码、独立 fork 与真实 smoke 已完成）。
- [x] T400.6 完成 M1 + Panda 零间隙资产、拓扑、穿透、no-snap、视觉和 GPU0 C0/C1a Teacher 重基线。
  - [x] T400.6a Task 1：冻结 `0.0 m` 构建器合同并通过 asset static RED→GREEN。
  - [x] T400.6b Task 2：独立验证父侧安装平面与 `1e-6 m` 容差。
  - [x] T400.6c Task 3：重建组合 USD/checksum，并通过 Isaac Sim 5.1 PXR 单 root/安装平面预检。
  - [x] T400.6d Task 4：通过局部可见表面、拓扑、relocation、no-snap 和用户确认视觉门。
  - [x] T400.6e Task 5–6：C0/C1a GPU0 和最终证据门。
- [ ] T400.9 按批准计划实施 100/10/23 online DAgger Student S1。
  - [x] T400.9a Task 1：冻结 Student observation/history/action 和安全 residual 合同。
  - [x] T400.9b Task 2：抽取部署侧 C1a mission/nominal，并支持 Teacher 外部注入。
  - [x] T400.9c Task 3：实现显式历史缓冲、GRU estimator、安全头与 23 维 actor。
  - [x] T400.9d Task 4：实现确定性安全接管、六项监督损失和严格版本化 DAgger replay。
  - [x] T400.9e Task 5：冻结严格 Student checkpoint/manifest 和 resume/inference 边界。
  - [x] T400.9f Task 6：抽取 runtime adapter 并完成 batched Teacher 隔离；Task 6 回归 `70 passed`。
  - [x] T400.9f Tasks 7–9：100 维观测/S1 注册、可版本化采集 shard、纯监督 Student trainer；CPU 端到端 smoke 通过。
  - [x] T400.9f Tasks 10–12：Student-only Play、三 seed 评估入口与 S1 运行边界文档；CPU contract smoke 通过。
  - [ ] T400.9g Isaac 物理 Teacher side-label 采集、Student-only 物理闭环和 GPU 验收。
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

- [8D Residual WBC implementation](../log/2026-08-26-m1-panda-8d-residual-wbc-implementation.md)
- [8D Residual WBC GPU0 smoke](../log/2026-08-26-m1-panda-8d-residual-wbc-gpu0-smoke.md)
- [Phase 5–6 Arm MPC + 8D PPO design](../../docs/superpowers/specs/2026-08-26-m1-panda-arm-mpc-8d-ppo-design.md)
- [Folded-load curriculum orchestrator Task 8](../log/2026-08-25-m1-panda-folded-load-curriculum-orchestrator.md)
- [Folded-load train/eval entrypoints Task 7](../log/2026-08-25-m1-panda-folded-load-train-eval-entrypoints.md)
- [Folded-load training guard Task 6](../log/2026-08-25-m1-panda-folded-load-training-guard.md)
- [Folded-load wrapper/DR Task 5](../log/2026-08-25-m1-panda-folded-load-wrapper-dr.md)
- [Folded-load MDP/environment Task 4](../log/2026-08-25-m1-panda-folded-load-mdp-env.md)
- [Folded-load PPO guard Task 3](../log/2026-08-25-m1-panda-folded-load-ppo-guard.md)
- [Folded-load active-action mask Task 2](../log/2026-08-25-m1-panda-folded-load-active-action-mask.md)
- [Folded-load curriculum Task 1](../log/2026-08-25-m1-panda-folded-load-curriculum-contracts.md)
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
- [Zero-clearance builder contract](../log/2026-08-18-m1-panda-zero-clearance-builder-contract.md)
- [Zero-clearance mount-plane verifier](../log/2026-08-18-m1-panda-zero-clearance-mount-plane-verifier.md)
- [Zero-clearance asset rebuild](../log/2026-08-18-m1-panda-zero-clearance-asset-rebuild.md)
- [Zero-clearance runtime and visual gates](../log/2026-08-18-m1-panda-zero-clearance-runtime-visual-gates.md)
- [Zero-clearance Teacher rebaseline](../log/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)
- [Zero-clearance Teacher runbook](../../docs/superpowers/runbooks/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)
- [Online DAgger Student S1 plan](../../docs/superpowers/plans/2026-08-18-m1-panda-dagger-student-s1.md)
- [Student S1 contracts](../log/2026-08-18-m1-panda-student-s1-contracts.md)
- [Student S1 deployable mission](../log/2026-08-18-m1-panda-student-s1-mission.md)
- [Student S1 temporal model](../log/2026-08-18-m1-panda-student-s1-model.md)
- [Student S1 DAgger replay](../log/2026-08-18-m1-panda-student-s1-dagger-replay.md)
- [Student S1 checkpoint](../log/2026-08-18-m1-panda-student-s1-checkpoint.md)
- [A1 recovery training design](../../docs/superpowers/specs/2026-08-15-m1-panda-a1-recovery-training-design.md)
- [A1 recovery training design log](../log/2026-08-15-m1-panda-a1-recovery-training-design.md)
- [A1 recovery implementation plan](../../docs/superpowers/plans/2026-08-15-m1-panda-a1-recovery-training.md)
- [A1 recovery plan log](../log/2026-08-15-m1-panda-a1-recovery-plan.md)
- [A1 recovery implementation](../log/2026-08-15-m1-panda-a1-recovery-implementation.md)
- [A1 recovery blocks](../log/2026-08-15-m1-panda-a1-recovery-blocks.md)
- [Prioritized WBC Teacher–Student design](../../docs/superpowers/specs/2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md)
- [Prioritized WBC Teacher–Student design log](../log/2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md)
- [Prioritized WBC Teacher C0 implementation plan](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)
- [Prioritized WBC Teacher C0 plan log](../log/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation-plan.md)
- [M1 + Panda WBC contracts](../log/2026-08-17-m1-panda-wbc-contracts.md)
- [M1 + Panda coordinated kinematics](../log/2026-08-17-m1-panda-wbc-kinematics.md)
- [M1 + Panda prioritized motion distribution](../log/2026-08-17-m1-panda-motion-distribution.md)
- [M1 + Panda reference QP](../log/2026-08-17-m1-panda-reference-qp.md)
- [M1 + Panda standing WBC](../log/2026-08-17-m1-panda-standing-wbc.md)
- [M1 + Panda WBC impedance and safety](../log/2026-08-17-m1-panda-wbc-safety.md)
- [M1 + Panda deterministic WBC Teacher](../log/2026-08-17-m1-panda-wbc-teacher.md)
- [M1 + Panda prioritized WBC Teacher C0 acceptance](../log/2026-08-17-m1-panda-prioritized-wbc-teacher-c0.md)
- [M1 + Panda rolling WBC Teacher C1a design](../../docs/superpowers/specs/2026-08-18-m1-panda-wbc-teacher-c1a-design.md)
- [M1 + Panda rolling WBC Teacher C1a implementation plan](../../docs/superpowers/plans/2026-08-18-m1-panda-wbc-teacher-c1a.md)
- [M1 + Panda rolling WBC Teacher C1a runbook](../../docs/superpowers/runbooks/2026-08-18-m1-panda-wbc-teacher-c1a.md)
- [M1 + Panda rolling WBC Teacher C1a acceptance](../log/2026-08-18-m1-panda-wbc-teacher-c1a.md)
- [Student S1 Task 6 batched Teacher isolation](../log/2026-08-20-m1-panda-student-s1-task6-batched-teacher.md)
- [Student S1 Tasks 7–9 training smoke](../log/2026-08-20-m1-panda-student-s1-training-smoke.md)
- [Student S1 Tasks 10–12 evaluation smoke](../log/2026-08-20-m1-panda-student-s1-evaluation-smoke.md)

## Git Refs

- Last Feature Commit: `545dcaa`
- Last Verified Base: `545dcaa`
- Current Work Ref: `codex/m1-panda-ppo-stability`
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
  - [优先级 WBC Teacher C0 实施计划](../../docs/superpowers/plans/2026-08-17-m1-panda-prioritized-wbc-teacher-foundation.md)
  - [WBC 合同](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/contracts.py)
  - [WBC 合同测试](../../Go2Pvcnn/tests/test_m1_panda_wbc_contracts.py)
  - [协调运动学](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/kinematics.py)
  - [协调运动学测试](../../Go2Pvcnn/tests/test_m1_panda_wbc_kinematics.py)
  - [运动约束](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py)
  - [优先级运动分配](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py)
  - [运动分配测试](../../Go2Pvcnn/tests/test_m1_panda_motion_distribution.py)
  - [参考 QP](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/qp_backend.py)
  - [参考 QP 测试](../../Go2Pvcnn/tests/test_m1_panda_qp_backend.py)
  - [Standing WBC](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/standing_wbc.py)
  - [Standing WBC 测试](../../Go2Pvcnn/tests/test_m1_panda_standing_wbc.py)
  - [WBC 阻抗](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/impedance.py)
  - [WBC 安全监督](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py)
  - [WBC 安全测试](../../Go2Pvcnn/tests/test_m1_panda_wbc_safety.py)
  - [带限轨迹](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/trajectory.py)
  - [Deterministic Teacher](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py)
  - [Teacher 测试](../../Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py)

## Next Step

当前 T400.13 下一步是用户复核 Phase 6 bridge/normalizer 连续性书面
规格；批准后才进入 TDD 实施计划和代码修改。不得直接从缺少 normalizer
count 的 v6 u100 续训，不放宽 promotion 门槛，也不在 `accepted=false`
时创建 long 目录或进程。

协同任务第一版已完成纯 PyTorch mission/零空间辅助和 combined Gym 启动；下一步必须先处理或独立复验 `Panda/root_joint` disjointed body transforms 的 PhysX snap 警告，再进行多环境动态验收。6D mount wrench 契约保持不变，Student S1 暂不更新。

为 C1/C2 单独设计滚动约束与 planar-base 到轮速/接触的一致映射；不得把 C0 驻停验收外推为移动底盘能力。T400.7 仍保留为旧 A1 诊断线，最大载荷实机测试前仍必须完成 T400.3 机械验算。

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

C0 驻停 foundation 与 C1a 平地直线滚动 deterministic Teacher 均已完成硬验收。下一子阶段只允许先为 C1b 转向编写独立设计并取得用户批准；C2/C3、Student、抓取与实机工作仍不在授权范围内。

### T400.9 Phase 5 传感器校准动力学完整验收（完成）

最终采用用户批准的六轴传感器直接校准：RNE 提供 `0.001` 动力学先验，安装传感器提供 `0.999` 观测更新，并以 active-minus-hold 的匹配动态增量验收。GPU0 独立 4000-step seeds 42/43/44 均通过；力矩方向余弦分别为 `0.9999970431 / 0.9999986872 / 0.9999971937`，力方向余弦分别为 `0.9999999974 / 0.9999999927 / 0.9999999810`。三组均无 reset/base contact/关节越界，四轮持续接触，MPC/QP 可行率 `1.0`。最终相关回归 `90 passed`，compileall 与 diff check 通过，Phase 5 完整验收完成。

### T400.13 Phase 6 固定条件 Residual PPO 晋级（运行中）

已完成归一化有界 wrench reward、100-update 五候选、安全监控与性能
选择解耦、九次 zero-pair 噪声标定、十五次 candidate/seed 独立评估和
严格 SHA lineage。v4/v5/v6 每轮均完成完整 24 个 GPU0 worker，但三轮
promotion 均拒绝，未启动 long。v6 u000/u025/u100 为
`aggregate_equivalent`，u050/u075 为 `seed_42_wrench_regression`。

架构诊断确认 u100 normalized action RMS 只有约 `0.00092–0.00323`，100
updates 尚不足以从接近最优的 zero-residual WBC 基线产生超过 PhysX
噪声的改善；同时 RSL empirical normalizer 的 sample count 没有进入
checkpoint。现已在不改 reward、物理限制、seed、4000-step 和 promotion
tolerance 的前提下实现完整模型/optimizer/normalizer/count 连续恢复、
legacy u100 精确迁移、update100→300 bridge 与 schema-v3 晋级。CPU
预检 `134 passed`；GPU0 bridge、24-worker promotion 和条件 long 仍待执行。
