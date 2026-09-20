# T500 双来源训练与 GPU Play 生产化设计

## 目标与边界

将当前“GPU RTI correctness backend + FAVOR-only 数据身份通过”的状态推进到：

1. OakInkV2 兼容动作引用的全部 90 个 unique object geometry 可验证；
2. FAVOR + OakInkV2 双来源重新转换并取得新的 aggregate SHA；
3. expert ensemble 和 student artifact 通过生产资格；
4. GPU RTI planner 接入 runtime，并经 benchmark、Isaac GPU0 smoke 与正式 30 条验收后开放 GPU Play。

本设计不改变原始 archive/source/run_e，不改变公开 `[B,25,43]` 动作、CPU float64 合同或 T400/T500 state-machine/safety 合同。geometry overlay 是独立外部 SHA manifest/resolver，不覆盖原始数据。

## 1. Geometry overlay

### 输入

- 保留已固定 OakInkV2 archive/revision/SHA；
- 从 1292 个 source-compatible interaction sides 收集约 90 个 unique `ObjURDF/align_ds/...` reference；
- 每个 reference 的 raw mesh 必须来自固定 upstream archive，不能用 repair mesh 替代 raw 坐标而不记录来源。

### 生成与验证

对每个 object reference，在受限 CPU 环境单独生成 COACD/URDF overlay：

- 固定 CoACD、ManipTrans、NumPy、trimesh、native library、seed、线程和全部 recipe 参数；
- 不做隐藏 rescale、registration、centering 或旋转；
- 输出 URDF 使用 zero origin/unit scale，collision mesh 至少一个；
- 验证三角面有限、组件 watertight、正体积、外法向一致、范围/frame 与 raw 一致；
- 二次独立生成必须得到相同文件 SHA；
- 任何一个 reference 失败或漂移都不发布 production geometry manifest。

Manifest 必须绑定：upstream archive/member SHA、reference、raw member、URDF SHA、全部 mesh SHA、recipe/tool/runtime identity、checks 和 aggregate manifest SHA。resolver 在 audit 与 load 使用同一份 immutable manifest，并在每次 resolve 时重新检查文件 SHA。

## 2. 双来源数据转换资格

几何通过后，输出到全新目录，不修改 `run_e`：

- FAVOR 与 OakInkV2 都必须保留；
- 动作轨迹不要求相同，保留各源原始动作多样性；
- 同一物体可以对应多个动作、手侧、阶段和接触路径；
- group-exclusive split 按 sequence/object/group 规则执行，防止泄漏；
- audit JSONL、NPZ shards 和 aggregate manifest 逐文件 SHA 固定；
- aggregate 必须同时记录 source counts、object geometry coverage、accepted/rejected reasons 和 geometry SHA。

生产资格要求：所有被接受的 OakInkV2 side 都必须解析到已验证 geometry；若任一 reference 失败，则本轮只允许生成诊断结果，不得标记 production-approved，也不得启动正式训练。

## 3. Expert 与 student

### Expert

先训练三成员 fingertip residual mixture ensemble：输入仍只包含当前六维手指状态、接触状态和动作阶段；输出短时五指相对运动分布。不得输入 task/object ID，也不决定掌心、物体、机械臂或底盘轨迹。

生产 expert 的 metadata 必须绑定：

- 新双来源 aggregate SHA；
- geometry manifest SHA；
- source/side/group split；
- training seed/member SHA；
- validation NLL、temporal regularization、coverage、endpoint/first-node RMSE；
- writer-identical canonical metadata self-hash。

### Student

蒸馏成冻结 student，运行时只提供可退让 fingertip soft cost；通过 O6 Jacobian 转成关节参考。student 不能改变 hard constraints、掌心/物体轨迹或公开 43D action。artifact、metadata SHA、normalization SHA 必须成对绑定并在 AppLauncher/Isaac 前验证。

## 4. GPU RTI runtime 与 Play

Gate A–C 已完成的 eager planner 继续保持 prior-off correctness backend。runtime 接入分为：

1. Wrapper 在控制周期内构造 CUDA float32 `CoupledRtiInput` 与 `GpuReducedDynamics`；
2. 运行一次 planner，读取仅 43D action、reason、safe_available 和 diagnostics；
3. planner reject 时沿 CPU/OSQP/last-safe 顺序退化，不把 fallback 误报为 accepted；
4. prior-on 只在 production student metadata/pin 合格时作为 soft cost；prior 故障只移除 soft cost；
5. GPU backend 与 reference CPU 在小 fixture 上做 action/state/decision parity。

Play 入口继续 fail-closed，直到 runtime 接线、benchmark manifest 和 Isaac gate 全部完成。`auto` 不得在没有 benchmark manifest 时选择 GPU。

## 5. 验收门

### Geometry/Data

- 90/90 unique geometry reference 成功生成、二次 SHA 一致；
- 双来源 accepted side 全部 geometry-resolved；
- 新 aggregate/audit/shards 逐文件 SHA 一致；
- source counts、coverage、split 和 rejection reasons 可复现。

### Training

- expert 三成员训练可恢复、metadata self-hash 通过；
- student 蒸馏 deterministic repeat 通过；
- artifact production-approved=true 且不含 synthetic provenance；
- runtime metadata/pin 检查通过。

### GPU/runtime

- GPU planner 与 CPU reference 小 fixture parity；
- fixed-shape warmup 后无 host scalar/new CUDA storage；
- benchmark manifest 记录 p50/p95/p99、显存、batch、H25 和完整命令；
- Isaac GPU0 smoke 验证坐标、工作台、最小间距、控制方向、频率和 reset/termination；
- 正式 30 条逐条 JSONL 与 SHA-pinned aggregate manifest 全部通过。

## 明确不做

- 不将 90 个 object reference 当作 90 条相同动作；
- 不用 FAVOR-only 数据冒充双来源生产数据；
- 不把 COACD 失败对象替换成 box/single hull 以通过资格门；
- 不在 benchmark/Isaac 证据前开放 `bimanual-rti-cuda` 或 GPU Play；
- 不新增接触模式 MPC、感知、Residual 或训练 task-ID 输入。
