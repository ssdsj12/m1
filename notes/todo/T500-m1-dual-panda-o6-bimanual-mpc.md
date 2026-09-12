# T500 M1 + 双 Panda + 双 O6 分层双手 MPC

## Current State

2026-09-11 更新：T500.5 DexManipNet 指尖运动先验设计已获用户逐段批准。完整 FAVOR 与
OakInk V2 将经源手 URDF FK 转为掌坐标五指尖短时概率分布，先训练离线教师 ensemble，
再蒸馏冻结学生并只作为 O6 Hand MPC
可退让软代价；模型不接收任务/对象 ID，也不决定掌、物体、机械臂或底盘轨迹。Task 1 已
冻结固定 pin、掌坐标五指窗口、四分量分布、student metadata 和 Inspire/Shadow 左右手
registry；Task 2 已实现固定 DexManipNet revision、固定 ManipTrans commit、archive SHA/size
manifest、路径/设备/越界链接拒绝、临时 sibling 解压后原子安装，以及不联网且不改写 evidence
的 `--verify-only`。source checkout 在复用和 verify 时均以 no-optional-lock Git status 拒绝 tracked、
untracked 和 ignored 漂移。Task 3 已实现严格单根 URDF tree、稳定拓扑、fixed/revolute/continuous/
prismatic 批量 FK、递归 mimic 展开和掌坐标五指尖转换，保持 stdlib XML + NumPy 且不依赖 Isaac。
Task 4 已实现精确 `favor`/`oakinkv2` source-side、`lh_main`/`rh_main`/`bh_main` interaction-side、固定手型
joint width、长度/shape/有限性、手根/对象状态、对象 URDF 与全部 successful rollout 的原子审计，并按
累计 reward 和 rollout 名确定性选优；拒绝 HDF5 外部 link/storage 与对象 geometry symlink，输出不含
task/object ID/name、primitive、description 或 text。Task 5 已加入 palm-frame 五指尖规范化、固定
`60 -> 100 Hz` CubicSpline 解析导数、对象 collision mesh 几何接触滞回、七阶段、20 节点窗口、
group-exclusive 80/10/10 split，以及固定 ZIP metadata 的原子 NPZ、audit JSONL 和排序 SHA aggregate；
两次完整合成 CLI 转换逐文件哈希一致。review 已补齐 conversion 前 archive size/SHA 重算、共享 Task 2
readonly ManipTrans commit/tree/dirty 检查、outward mesh normal 门、target-rate spline normal speed 及
audit/shard/aggregate drift verifier；Task 1–5 回归 `108 passed`。见
[review hardening 验证](../log/2026-09-11-t500-dexmanipnet-shard-review-hardening.md)。Task 6 已实现仅离线的
large residual MLP mixture ensemble：训练在读取前重新验证 aggregate、audit 和全部 shard SHA，只保留 42 维掌坐标
几何输入与 `20×5×3` future velocity；distinct fixed seeds、group-exclusive split、NLL 和 acceleration/jerk
规则化、atomic checkpoint/manifest、best validation member selection 均已覆盖。held-out metric 使用整个
ensemble/component predictive mixture（同时含 aleatoric 与成员间 epistemic 不确定性）的精确 80% quantile coverage，
终点以 `0.01 s` 积分 20 节点。合成 two-epoch smoke 有限但 first/endpoint 零基线改善仅 `4.46%/4.59%`、coverage
`1.0`；nonproduction synthetic provenance 写入 verified aggregate 且无论数值如何都不会标记为 production deployable；
resume 在读取 member facts 前验证 writer-identical canonical manifest self-hash，随后验证 selected checkpoint SHA、member/seed、architecture 和 aggregate。Tasks 1–6 `117 passed`。Task 7 的 compact student、mandatory metadata/report integrity checks、temporal objective、independent determinism fingerprint 和 versioned comparison provenance 已完成；synthetic artifact 永久 `production_approved=false`。Tasks 1–7 `133 passed`。尚未下载 8.31 GB
数据、验证真实全量 schema/对象 mesh 接受率、训练真实模型或修改运行时。见 [Task 6 ensemble 验证](../log/2026-09-11-t500-dexmanipnet-expert-ensemble.md)。
[deterministic shard 验证](../log/2026-09-11-t500-dexmanipnet-deterministic-shards.md)。

2026-09-12 更新：Task 8 adds a runtime-only frozen-student adapter. It rejects a non-approved
artifact, packs the exact 42 float32 geometry features from O6 measurements, selects the first
future-node mixture component against baseline tip velocity, bounds precision, removes contacted
rows, and returns finite CPU float64 QP terms. Safe phases, timeout, invalid input/output, and
exceptions disable atomically without cached targets. Tasks 1–8 focused regression: `139 passed`;
Task 9 Hand MPC integration remains open.

2026-09-06 更新：单轴右掌姿态 MPC 已实现，但 1600 步物理门失败。
350 步诊断精确复现第 184/270/329 步的 Arm 不可行/限位/安全拒绝，
确认姿态坐标差与空间角位移存在不一致，且首目标相对实测姿态的误差增长。
见[诊断日志](../log/2026-09-06-right-palm-orientation-boundary-diagnosis.md)。
以下资产 Tasks 1–3 描述保留为历史基线。

2026-09-11 更新：本轮控制与训练改进已整理为功能提交 `265fbcf`。改动包含
SO(3) 几何插值/空间姿态误差、接触摘要与逐 body Box 过滤、举升动作原语与
向量控制入口，以及可恢复的 10k/40k 固定总步数账本和 teacher 数据准备链。
T500 专项测试 `224 passed`，Python 编译和 staged diff 检查通过；正式 30/30
物理验收仍未执行，不能据此宣称整套举升任务已验收通过。见
[上传验证日志](../log/2026-09-11-t500-github-upload-verification.md)。

交互设计和书面规格均已由用户确认。首版使用 M1、公共单轴回转平台、左右两条 Panda 和左右 O6，完成固定 `0.5 kg` 箱体的确定性双手夹持、抬升 `0.10 m`、保持 `3 s`、下降和释放。采用 object MPC、双 Arm MPC、双 Hand MPC 和 200 Hz WBC/QP；第一阶段不训练 RL。

规格已写入 [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)，13 任务 TDD [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)正在以 Inline Execution 执行。Tasks 1–3 已完成：左右 O6 资产规范化入库，组合 articulation 已生成并通过最终 Isaac 2000 步硬门。

## Open Children

- T500.5：DexManipNet 掌坐标五指尖概率先验；Tasks 1–8 合同、下载边界、源手 FK、sequence 审计、deterministic shards、offline ensemble、compact student artifact 和 fail-closed runtime adapter 已完成，下一步为 Task 9 Hand MPC atomic integration；真实数据/ensemble/student production 门仍未运行。

- T500.4：右掌姿态传递语义与跟踪边界；阻塞 1600 步物理门和正式 30 条验收。

- T500.1：书面规格已确认。
- T500.2：逐文件 TDD 实施计划正在 Inline Execution 执行，Task 6 待开始。
- T500.3：已冻结双臂平台安装变换和 O6 规范化资产 manifest；53 物理 DOF、43 主动通道运行时确认通过。

## Closed Children Archive

- 机械拓扑、首个箱体任务、仿真真值、确定性控制、公共 yaw 平台、分层 MPC、资产边界、安全回退和验收门已完成交互确认。

## Related Logs

- [2026-09-12 DexManipNet frozen O6 prior runtime](../log/2026-09-12-t500-dexmanipnet-fingertip-prior-runtime.md)

- [2026-09-11 DexManipNet deterministic fingertip shards](../log/2026-09-11-t500-dexmanipnet-deterministic-shards.md)

- [2026-09-11 DexManipNet shard review hardening](../log/2026-09-11-t500-dexmanipnet-shard-review-hardening.md)

- [2026-09-11 DexManipNet offline expert ensemble](../log/2026-09-11-t500-dexmanipnet-expert-ensemble.md)

- [2026-09-11 DexManipNet compact student distillation](../log/2026-09-11-t500-dexmanipnet-student-distillation.md)

- [2026-09-11 DexManipNet sequence 审计与 rollout 选优](../log/2026-09-11-t500-dexmanipnet-sequence-audit.md)

- [2026-09-11 DexManipNet Isaac-independent URDF FK](../log/2026-09-11-t500-dexmanipnet-urdf-fk.md)

- [2026-09-11 DexManipNet 指尖先验实施计划](../log/2026-09-11-t500-dexmanipnet-fingertip-prior-plan.md)

- [2026-09-11 DexManipNet 指尖先验合同冻结](../log/2026-09-11-t500-dexmanipnet-prior-contracts.md)

- [2026-09-11 DexManipNet 固定下载与安全解压](../log/2026-09-11-t500-dexmanipnet-fetcher.md)

- [2026-09-11 DexManipNet 指尖先验设计](../log/2026-09-11-t500-dexmanipnet-fingertip-prior-design.md)

- [2026-09-11 远端快进合入与本地验证](../log/2026-09-11-t500-remote-fast-forward-local-verification.md)

- [2026-09-11 根 README 状态说明](../log/2026-09-11-t500-readme-status.md)

- [2026-09-08 O6 原始资产迁入仓库](../log/2026-09-08-o6-source-relocation.md)

- [2026-09-02 设计记录](../log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
- [2026-09-02 实施计划](../log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-plan.md)
- [2026-09-02 Tasks 1–3 资产验证](../log/2026-09-02-m1-dual-panda-o6-asset-tasks1-3.md)

## Git Refs

- Last Feature Commit: Task 8 frozen O6 prior runtime（this commit）
- Last Verified Commit: Task 8 focused runtime plus Tasks 1–8 regression (`139 passed`), import boundary and pycompile
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`
- Key Files:
  - [DexManipNet 指尖先验实施计划](../../docs/superpowers/plans/2026-09-11-t500-dexmanipnet-fingertip-prior.md)
  - [DexManipNet 指尖先验设计](../../docs/superpowers/specs/2026-09-11-t500-dexmanipnet-fingertip-prior-design.md)
  - [设计文档](../../docs/superpowers/specs/2026-09-02-m1-dual-panda-o6-bimanual-mpc-design.md)
  - [实施计划](../../docs/superpowers/plans/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md)
  - [现有单臂 MPC](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py)
  - [现有单臂约束](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py)

## Next Step

从 T500.5 Task 9 的 Hand MPC atomic integration 开始继续 12 任务计划；运行时物理主线仍需在
恢复服务器 Vulkan/DRM 访问后运行跨 seed 物理门并核对固定步数账本，
再执行正式 30/30 举升、保持、下降和释放验收；若首节点仍追不上实测姿态，
再单独评估跟踪感知边界。

## Node Details

### T500.1 书面规格复核

重点检查 43 主动控制通道、53 预计物理 DOF 的运行时确认方式、公共 yaw 平台、双 O6 mimic 语义、原子回退和 30/30 固定条件验收。

### T500.2 实施计划

实施计划应按资产输入闭合、单 articulation 构建、静态/物理门、纯控制合同、object MPC、双 Arm MPC、双 Hand MPC、WBC/QP、Isaac 环境和完整任务验收拆分 TDD 任务。
