# T500 DexManipNet 指尖运动先验蒸馏设计

## 目标

把 DexManipNet 的 FAVOR 与 OakInk V2 单手、双手成功轨迹离线转换为与手型无关的
掌坐标系五指尖运动数据，并蒸馏成冻结的短时概率先验。运行时仍由现有 object MPC、
双 Arm MPC 和双 Hand MPC 决定物体、掌心、接触力和关节动作；专家模型只通过可退让的
软代价约束 O6 指尖运动，使未见操作中的手指轨迹更接近专家数据的自然运动分布。

本设计不要求 O6 逐帧模仿任一 DexManipNet 手型，也不把任务 ID、物体类别、物体轨迹、
掌心目标、机械臂状态或底盘状态输入专家模型。

## 外部来源与版本边界

- 数据集：`LiKailin/DexManipNet`，固定 revision
  `3933fae5fe83498fb314a0924aa21d5038fba5a5`。
- 数据内容：`dexmanipnet_favor.tar.gz` 与 `dexmanipnet_oakinkv2.tar.gz`，总压缩大小约
  `8.31 GB`；两部分均纳入审计。
- 源手定义：官方 `ManipTrans/ManipTrans`，固定 commit
  `a3d08cfe3c3a5868a7f057533bcaf759c5af4705`。
- 数据集和官方代码标注为 GPL-3.0。本阶段只做本机研究使用；原始数据、解压内容和
  训练权重默认不提交或重新分发。Git 只保存下载清单、来源 revision、SHA-256、转换代码、
  不含样本的统计报告和模型格式合同。
- 官方来源：<https://huggingface.co/datasets/LiKailin/DexManipNet> 与
  <https://github.com/ManipTrans/ManipTrans>。

## 总体数据流

```text
DexManipNet archives + pinned ManipTrans hand descriptions
    -> archive/schema audit
    -> source-hand URDF forward kinematics
    -> palm-frame five-fingertip positions and velocities
    -> handedness canonicalization and 60 Hz -> 100 Hz resampling
    -> 20-node / 0.2 s windows with contact and phase labels
    -> offline expert ensemble learns the full data distribution
    -> compact four-component student distills the expert ensemble
    -> frozen, SHA-pinned student artifact
    -> O6 measured fingertip geometry + folded real Jacobian
    -> optional soft quadratic cost in each Hand MPC solve
```

数据和模型流程完全离线。Isaac/Hand MPC 运行时不导入 Hugging Face、HDF5、源手 URDF
解析器或 ManipTrans 代码。

## 本地存储与下载

原始数据使用仓库内相对目录：

```text
Go2Pvcnn/data/external/dexmanipnet/
├── downloads/
├── extracted/
│   ├── dexmanipnet_favor/
│   └── dexmanipnet_oakinkv2/
├── source_maniptrans/
├── converted/
├── artifacts/
└── manifests/
```

现有根 `.gitignore` 的 `data/*/` 规则覆盖该目录。下载器必须：

1. 使用固定 Hugging Face revision，不允许静默跟随 `main`。
2. 支持断点续传并在下载前检查可用空间。
3. 对两个归档分别计算 SHA-256，并把实际大小、来源 URL、revision 和哈希写入 manifest。
4. 在解压前拒绝绝对路径、`..` 路径穿越、设备文件和越界符号链接。
5. 使用临时目录解压，完成验证后原子重命名；中断不得产生看似完整的数据目录。
6. 固定 ManipTrans commit，并记录其 Git tree/commit，不从未固定的工作树读取手型定义。

## 源数据审计

每个 sequence 至少需要：

- `seq_info.json`，含 `dexhand`、左右手/交互模式、序列长度及对象路径信息；
- `rollouts.hdf5/rollouts/successful/*`；
- 对应侧的 `q_rh`/`q_lh`、`dq_rh`/`dq_lh` 和手根状态；
- 与序列长度一致的有限数值；
- 可由固定源手 manifest 唯一解析的关节顺序、掌 link 和五个指尖 link。

与官方 loader 一致，每个 sequence 只选择累计 reward 最大的 successful rollout。reward
只用于选择成功 rollout，不作为专家模型输入或训练目标。

审计对每条 sequence 输出一行 JSONL，至少包含来源、sequence、侧、手型、帧数、接受状态、
拒绝原因和输入 SHA。缺字段、长度不一致、非有限值、未知关节顺序、缺少可用 URDF、缺掌/指尖
link、FK 失败、对象几何不可用或接触标签不可确定时必须原子拒绝，不能用零值补齐。

## 手型无关的指尖转换

### 源手 FK

预处理器不依赖 IsaacGym。它从固定的源手 manifest 读取：

- URDF 相对路径；
- 与 HDF5 `q_*` 列严格一致的主动关节顺序；
- 掌 link；
- 拇指、食指、中指、无名指、小指指尖 link 顺序；
- 左右手坐标约定。

使用 URDF origin、axis 和 revolute/prismatic 关节计算每帧掌到指尖变换。固定关节必须参与
链式变换；mimic 关节必须从 master、multiplier 和 offset 展开。任何自由度数量或名称不一致
都拒绝整条侧轨迹。

### 规范坐标与左右手

训练几何统一为右手规范掌坐标。右手保持原掌坐标；左手通过固定反射矩阵镜像位置与速度，
并按相同的五指顺序保存。部署到左 O6 时执行严格逆变换。镜像变换只作用于几何，不交换手指。

模型不接收 `left/right` 标志，因此不能学习依赖侧别的任务行为。

### 速度和采样率

- 源数据按官方 60 FPS 解释；实际帧率与合同不符时拒绝，不能猜测。
- 指尖位置先做带限三次插值，再统一采样到 `100 Hz`。
- 速度由插值轨迹的解析导数或对称带限差分产生，不能直接放大含噪单步差分。
- 每个样本为连续 `20` 节点、`0.2 s` 的未来指尖速度，shape 为 `(20, 5, 3)`。
- 窗口不得跨 sequence、回放边界或不连续时间戳。

### 接触与阶段

接触标签 shape 为 `(5,)`。优先使用经 schema 审计确认的逐指显式接触字段；缺少该字段时，
使用对应对象网格、对象位姿、指尖表面距离和相对法向速度，并带进入/退出双阈值滞回推断。
对象几何或位姿不足时拒绝该侧轨迹，而不是把它标成无接触。

训练使用七个与任务无关的规范阶段：`approach`、`preload`、`grasp`、`manipulate`、`hold`、
`release`、`unknown`。阶段仅由掌/指尖相对速度、接触建立/保持/消失和静止区间推断；
`primitive`、description、对象名称和数据集任务 ID 不进入模型。运行时映射为：

- `APPROACH -> approach`
- `PRELOAD -> preload`
- `GRASP -> grasp`
- `LIFT/LOWER -> manipulate`
- `HOLD -> hold`
- `RELEASE -> release`
- `DONE/HOLD_SAFE/LOWER_SAFE/SAFE_RELEASE/TERMINATED -> 禁用先验`

## 蒸馏数据合同

转换后的每个窗口至少包含：

- `fingertip_position_palm`: `(5, 3)`, float32；
- `fingertip_velocity_palm`: `(5, 3)`, float32；
- `contact_mask`: `(5,)`, bool；
- `phase`: 标准七阶段整数；
- `future_fingertip_velocity_palm`: `(20, 5, 3)`, float32；
- `source_group`: 来源、sequence、侧组成的稳定 group ID；
- `source_sha256`: 原 sequence 输入指纹。

train/validation/test 按完整 `source_group` 确定性划分为 80/10/10。同一 sequence 的左右手
可以属于同一 group，但任何重叠窗口不得跨集合。数据加载器不暴露任务 ID、物体类别或文本描述。

转换产物采用 shard + aggregate manifest。每个 shard 原子写入并记录 SHA-256、样本数、来源、
手型、阶段和接触模式分布。aggregate manifest 固定全部 shard SHA；训练拒绝缺 shard、额外 shard、
哈希不符或 split 重叠。

## 专家学习与概率先验蒸馏

### 输入

运行接口接收 O6 当前六维 `q/qd`、实测掌坐标五指尖位置、折叠真实指尖 Jacobian、五指接触
mask 和 `BimanualPhase`。几何模型的实际网络输入是：

- 当前掌坐标指尖位置 `15`；
- 当前指尖速度 `J(q) qd`，`15`；
- 五指接触 mask，`5`；
- 七阶段 one-hot，`7`。

总输入维数为 `42`。六维 `q/qd` 不直接进入网络，避免把 O6 特有关节坐标错误地与源手关节角
对齐；它们只通过 O6 真实 FK/Jacobian 形成几何状态并参与后续受约束求解。

### 输出合同

教师和学生使用相同的四分量对角高斯混合输出合同：

- mixture logits：`(4,)`；
- future velocity means：`(4, 20, 5, 3)`；
- future velocity log standard deviations：`(4, 20, 5, 3)`。

log standard deviation 在模型内部和运行时都做有限上下界裁剪。模型不得预测关节力矩、接触力、
掌位姿、物体位姿或底盘动作。

### 第一阶段：离线专家学习

先训练由多个较大残差 MLP/temporal decoder 组成的专家 ensemble。每个成员以不同固定 seed
训练四分量混合高斯负对数似然；加速度/jerk 正则只约束时间连续性，不使用任务完成 reward。
ensemble 聚合 aleatoric 与成员间 epistemic 不确定性，用于给蒸馏学生产生软分布目标。

专家 ensemble 只存在于离线训练目录，不能被 Hand MPC 或 Play 直接加载。若 held-out 指标未通过，
不得开始学生蒸馏，也不得把未通过的教师标成可部署先验。
训练器和蒸馏器共享同一严格 ensemble verifier：完整 manifest schema/self-hash、固定且无重复的
成员 seed 顺序、checkpoint 路径与 SHA、dataset/训练身份、有限且内部一致的 held-out 指标及由指标
重新计算的教师 gate 必须同时成立。manifest 自称 `production_deployable` 不能绕过这些验证。

### 第二阶段：紧凑学生蒸馏

运行时学生是单个小型 MLP/temporal decoder，输入和输出合同与教师一致。学生同时最小化：

- 对真实未来指尖轨迹的混合高斯负对数似然；
- 对教师 ensemble 混合分布的固定 Monte-Carlo KL/交叉熵蒸馏损失；
- 与教师一致的有限加速度/jerk 正则。

蒸馏采样使用写入 manifest 的固定 seed 和每状态固定样本数。ensemble 的概率密度与采样都采用
“成员等权、成员内四分量 softmax”：先分别归一化每个成员，再赋予严格 `1/M` 的成员权重，
因此任一成员 logits 的整体平移不能改变密度或固定 generator 的样本。教师样本以绑定 dataset
aggregate SHA、ensemble manifest SHA、采样 seed 和每状态样本数的确定性磁盘分片/memmap 原子发布；
生成过程使用固定的身份所有者文件、固定 staging 路径和逐 chunk 原子 SHA 进度；异常终止后仅在身份、
路径和进度均严格匹配时续写，普通失败则只清理已证明属于本次身份且不含 symlink 的 staging。
学生 batch 按需读取，内存复杂度不得随完整教师样本集增长。混合分量不做基于编号的硬对齐，
避免分量置换造成错误监督。只有冻结学生进入运行 artifact；教师 ensemble 保留在本地、忽略 Git，
用于复核和重新蒸馏。

学生导出为冻结权重和 metadata。metadata 固定模型格式版本、输入/输出顺序、20 节点、100 Hz、
五指顺序、七阶段顺序、镜像矩阵、数据 aggregate SHA、教师 ensemble manifest SHA、教师与蒸馏
seed、代码 commit、学生权重 SHA、每状态样本数、epoch、batch、学习率、软件/CUDA build、稳定设备
指纹、CUBLAS 配置及 trainer/model/contracts/artifact 的语义 SHA。可复现身份只哈希确定性权重、指标、
provenance 和训练合同；`metrics.json` 不含机器相关的最终批准位。每次真实 CPU latency 与由质量门和
`p99 < 2 ms` 共同推导的 `production_approved` 原样保存在独立 qualification，并由 SHA 完整绑定。
加载器先分别校验 deterministic identity 与 qualification，再仅在内存视图中向运行时暴露合成后的
批准状态。不同机器或运行时抖动允许 qualification SHA 不同，不得伪造、取整或用其破坏同配置学生
权重、metrics 和 identity 的比较。

威胁模型明确区分完整性与真实性：artifact 内部 self-hash、文件 SHA、审批重算以及单次打开得到的
不可变字节快照，只能证明同一 artifact 内各字段一致并阻止校验—加载间路径替换；它们不能阻止拥有
目录写权限者用任意模型协调重写全部文件。`save_student_artifact` 因此只是离线 authoring API，不能
充当信任根。部署方若需要确认“这就是被批准的那一份”，必须从 artifact 目录之外提供预先审阅的
`metadata.json` SHA-256，并让 validate/load 的 `expected_metadata_sha256` 严格匹配该外部 pin。
学生使用输出目录专属 resume workspace，在 epoch 边界原子提交模型和优化器状态；最终目录仅在完整
训练、独立重复和质量门通过后原子发布。真实数据 gate 失败必须非零退出，且不得发布可供运行时
误用的最终 artifact；仅可在明确 nondeployable 的诊断目录保留带 SHA 的失败证据。resume 在
`load_state_dict` 前验证模型 exact keys/shape/dtype/finite，并验证 AdamW 参数 ID、moment shape/dtype/finite
及完整参数组语义；任何协调重写 checkpoint 与 progress SHA 仍不得改变训练身份。

## Hand MPC 接入

### 两阶段原子求解

现有 Hand MPC 默认行为保持不变。只有显式配置并通过 artifact 校验后才启用先验：

1. `APPROACH/PRELOAD` 先运行当前确定性闭指控制得到同周期 baseline；其余阶段构建并求解
   当前无先验 Hand QP，得到同周期 baseline `qdot_base`。
2. 运行冻结先验，计算四组未来分布。
3. 用第一节点均值与 `J qdot_base` 的受限 Mahalanobis 距离选择一个混合分量；不使用任务 ID。
4. 对未接触或尚未锁定的手指加入局部二次软代价：

   `(J qdot - mu)^T W (J qdot - mu)`

5. `APPROACH/PRELOAD` 求解六维有界速度投影 QP，同时保留原闭指参考跟踪项和接触锁定轴；
   其余阶段重新求解同一个带原有限位、摩擦锥、接触力和 wrench 目标的 QP。

`W` 由预测方差倒数得到，并同时具有配置的最小/最大特征值及全局权重。已接触并锁定的指轴
权重降为零，避免先验与接触等式冲突。只消费第一节点是现有 100 Hz receding-horizon 接口的
明确边界；完整 20 节点分布保留在诊断和未来真正多节点 Hand MPC 接口中，但本轮不改写公共
25/50/100/200 Hz 调度合同。

### 优先级与失败语义

- 关节限位、速度限位、摩擦锥、接触力、目标 wrench 和安全投影始终高于专家软代价。
- 未配置先验时绕过全部先验代码，结果必须与当前 Hand MPC 一致。
- 显式指定但 metadata、SHA、shape、手指顺序、阶段顺序或频率不兼容时，启动时拒绝。
- 单帧推理超时、抛异常或产生非有限值时，放弃该帧先验并使用同周期已可行 baseline；不能用
  无效结果污染 `_last_safe`。
- 带先验的第二次 QP 不可行时，使用同周期 baseline，而不是回退到更旧动作；记录
  `prior_qp_rejected`。
- baseline 本身不可行时沿用现有安全 fallback，先验无权改变 fallback。

诊断至少记录：先验是否配置/启用、禁用原因、选中分量、混合概率、第一节点方差范围、
Mahalanobis 距离、先验代价、baseline/regularized tip velocity 差、推理耗时和第二次 QP 状态。

## 与现有 z16 Teacher 的关系

DexManipNet 先验与现有 `full_action_teacher -> z16 latent action model` 平行：

- z16 继续压缩 M1 全身 43 通道教师动作和箱体/掌/力任务效果；
- DexManipNet 模型只提供单手指尖自然性分布；
- DexManipNet 数据不能写入现有 `teacher_success.npz` 的 43 维动作标签；
- z16 训练可选择记录“带先验 Hand MPC”生成的新 teacher 数据，但不得把先验网络输出伪装为
  完整 MPC expert action；
- 两个 artifact 使用不同格式版本、目录和 SHA manifest。

## 验证门

### 1. CPU 数据与合同

- 下载 manifest 固定 revision/大小/SHA，安全解压拒绝路径穿越。
- 伪造最小 HDF5/URDF 覆盖 FK、fixed/mimic、五指顺序、左右镜像、60->100 Hz、窗口边界、
  接触滞回、阶段推断和每个原子拒绝原因。
- 左右镜像往返最大位置误差不超过 `1e-6 m`。
- FK 对手算两关节链参考最大误差不超过 `1e-8 m`。
- 同一 revision、seed 和输入 manifest 两次转换必须得到相同 shard SHA 与 aggregate SHA。
- train/validation/test 的 `source_group` 交集必须为空。

### 2. 模型与纯 QP

- 网络输出 shape、dtype、有限值、混合权重和标准差边界准确；保存/加载前后输出一致。
- 教师 held-out 第一节点指尖速度 RMSE 至少比零速度基线低 `10%`。
- 教师 held-out `0.2 s` 指尖终点误差至少比零速度积分基线低 `10%`。
- 教师预测 80% 区间的 held-out 覆盖率位于 `[0.65, 0.95]`；不满足则不能开始蒸馏。
- 学生 held-out 每维 NLL 相比教师 ensemble 的增量不超过 `0.05 nat`，`0.2 s` 终点 RMSE
  相比教师增幅不超过 `5%`，并且仍满足上述两个相对零速度基线的 `10%` 改进门。
- 固定教师、数据、蒸馏 seed 和初始化，两次蒸馏得到相同的指标 JSON 与学生 artifact SHA。
- CPU 单手单样本推理 p99 小于 `2 ms`，测量至少 1000 次且排除前 100 次 warm-up。
- prior 关闭时，已有 Hand MPC 专项回归结果不变。
- prior 启用时验证均值跟踪、方差退让、接触轴屏蔽、限位、摩擦、重复求解一致性、推理异常、
  第二次 QP 拒绝和最近安全状态不污染。

### 3. Isaac GPU0 smoke

单条固定样本分别以 prior off/on 运行，核对左右镜像、O6 实测指尖位置、真实折叠 mimic Jacobian、
100 Hz 更新频率、控制方向、推理耗时、最小间距、关节限位、接触方向和同周期 baseline 回退。
出现非有限值、硬限位、意外 reset 或安全拒绝即失败。

### 4. 未见操作评估

当前 M1 固定箱体任务不向先验提供任务或物体标识，因此属于接口意义上的未见条件。正式比较
使用相同 seeds、初始条件和 object/Arm MPC 参数运行 prior off/on：

- prior-on 的已执行指尖轨迹在冻结先验下的中位 NLL 至少降低 `10%`；
- 指尖速度 jerk 的 p95 不高于 prior-off；
- Hand QP 可行率、滑移 margin、硬限位、安全拒绝和任务完成率不得退化；
- 原 T500 三 seed、每 seed 十次的 30/30 完整物理验收仍是整机完成门。

如果基础 T500 Arm/姿态物理门仍失败，只能报告专家先验局部 smoke 结果，不能把整机任务失败
归因于数据集或宣称未见操作成功。

## 非目标

- 不复现 ManipTrans 的 IsaacGym 强化学习训练。
- 不移植其 residual policy、imitator checkpoint 或任务 reward。
- 不把源手关节角直接映射成 O6 关节角。
- 不让专家模型决定物体轨迹、掌心目标、接触力分配、双手配合、机械臂或底盘动作。
- 不修改 T400 公共合同，不改变 T500 的 43 通道顺序和分层控制频率。
- 不在本轮把现有单节点 Hand QP 重写为完整 20 节点非线性 MPC。

## 完成定义

只有下载/审计/转换 manifest、冻结 artifact、CPU 数据门、模型门、纯 QP 回归和 Isaac GPU0
smoke 全部通过，才允许在 Play/正式实验中启用专家先验。完成这些工程门不等于整机 30/30
验收；整机成功必须继续满足原 T500 物理标准。
