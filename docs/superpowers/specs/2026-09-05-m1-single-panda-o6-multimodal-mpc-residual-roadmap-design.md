# M1 + 右 Panda + 右 O6 多模态 MPC + Residual 工程路线图设计

## 1. 目标

本路线把《M1 + 双臂双灵巧手多模态感知 MPC + Residual 整体研究规划》收敛为一条可逐阶段实现和验收的工程主线。首个闭环系统固定为：

- M1 原地稳定，不使用底盘移动补偿不可达目标；
- 右 Panda 7-DOF；
- 右 Linker Hand O6，6 个主动通道和 5 个 mimic 联动关节；
- Isaac Lab 纯仿真；
- 固定 `0.5 kg` 刚性箱体；
- 唯一任务为接近、抓取、抬升、保持、下降和释放。

路线采用“双基线纵向闭环”：先以 Isaac 真值建立确定性控制基线，再替换为仿真 RGB-D + LiDAR 多模态状态估计，最后冻结 nominal MPC 并训练小范围受限 Residual。

## 2. 与现有路线的关系

本路线登记为独立根任务 `T600`，不改变现有任务的公开合同或默认行为：

- `T400` 提供 M1 + Panda 的 WBC/QP、安装点力反馈、Residual 训练守卫和 checkpoint manifest 经验；其未被接受的 checkpoint 不得作为本路线父模型。
- `T500` 提供 O6 源资产规范化、关节/mimic 语义和双手控制研究成果。只有已提交、具备 SHA manifest 且通过测试的通用资产或纯函数才可复用。
- `codex/t500-dual-panda-o6-mpc` worktree 中未提交的实验文件和探针产物不是 T600 的依赖，也不能作为验收证据。

T600 使用独立 Gym ID `Isaac-M1-SinglePanda-O6-Contact-v0`、模块命名、日志目录、训练配置和 checkpoint schema。不得把单 Panda 两指夹爪 checkpoint、T400 的 8D policy 或 T500 的双手 latent policy 直接加载到 T600。

## 3. 冻结的首版机械合同

主动控制通道顺序固定为：

```text
M1 legs (12)
M1 wheels (4)
right Panda arm (7)
right O6 active joints (6)
```

主动控制通道总数为 `29`。若 importer 将 O6 的 11 个活动关节均报告为物理 DOF，组合 articulation 预计为 `34` 个物理 DOF；最终数目必须由 PXR 和 Isaac 运行时探针测量并写入 manifest，不能硬编码错误计数。

O6 主动通道顺序固定为：

```text
thumb bend, thumb yaw, index bend,
middle bend, ring bend, little bend
```

五个 distal 关节只按资产中的 mimic 关系联动，不进入 policy 或 MPC 的独立动作向量。首版使用右手资产和右侧安装变换，同时保持状态、控制和测试合同可镜像到左侧。

## 4. 总体架构

```text
Isaac truth provider OR RGB-D + LiDAR estimator
                         |
                         v
             timestamped ObjectState
                         +
             RobotState + ContactState
                         |
                         v
            grasp planner + state machine
                         |
                         v
             contact-aware nominal MPC
                         |
                         v
                 bounded 9D residual
                         |
                         v
              safety projection + WBC/QP
                         |
                         v
                  29 actuator commands
```

所有模块消费同一个带单调时间戳的原子快照。感知 provider 可切换，但其下游 `ObjectState` 合同完全一致。MPC 负责名义轨迹和接触力，Residual 只补偿模型误差，WBC/QP 是唯一最终执行边界。

首版控制节拍固定为 Arm MPC `50 Hz`、O6 Hand MPC `100 Hz`、WBC/QP `200 Hz`。感知传感器按各自采样率异步更新，但必须在进入控制器前由状态 provider 完成时间对齐、状态年龄检查和确定性缓存；不得让控制周期阻塞等待新帧。

## 5. 核心数据合同

### 5.1 RobotState

至少包含 M1 base pose/twist、29 个主动通道状态、O6 mimic 状态、右 Panda 掌心 pose/twist、Jacobian、质量矩阵、bias、四轮接触和支撑状态。

### 5.2 ObjectState

至少包含箱体 pose、twist、协方差或置信度、来源、采样时间、状态年龄和有效标记。真值 provider 和多模态 provider 必须输出同一结构，不允许下游根据来源改变控制语义。

### 5.3 ContactState

至少包含 O6 指尖/掌面接触、法向力、切向力、滑移估计、Panda 腕部或安装点 wrench，以及箱体支撑接触。

### 5.4 CommandBundle

一个周期的 arm、hand、base residual 和 WBC 输出必须原子提交。任一字段 shape、dtype、device、finite、时间戳或安全检查失败时，整组新命令均不得写入仿真。

纯控制内核默认使用 `torch.float64`。环境边界可按 Isaac 需要转换 dtype，但必须显式测试，不能发生隐式混用。

## 6. 控制与任务状态机

任务状态机固定为：

```text
APPROACH -> PREGRASP -> PRELOAD -> GRASP -> LIFT
         -> HOLD -> LOWER -> RELEASE -> DONE
```

异常路径固定为：

```text
HOLD_SAFE -> LOWER_SAFE -> SAFE_RELEASE
```

自由空间阶段由 Arm MPC 跟踪掌心参考，O6 保持预抓取形状。接触阶段启用 finger/contact force 预测、摩擦锥、防滑和箱体动力学约束。箱体悬空时禁止突然释放；只有箱体稳定接触支撑面后才能进入释放状态。

## 7. 六阶段工程路线

### T600.1 资产与接口基座

建立项目自有的 M1 + 右 Panda + 右 O6 单 articulation，冻结 29 主动通道、O6 mimic、安装变换、传感器 body 名称、坐标系和 manifest。只复用已验证的 O6 规范化源，不直接依赖 T500 双臂运行时。

验收门：

- 所有 USD reference 可解析且无机器相关绝对依赖；
- 单一 articulation root；
- 29 个主动通道，运行时物理 DOF 与 manifest 一致；
- 质量、惯量、关节限制和安装变换均 finite；
- 零命令连续运行 `2000` 步，四轮持续接触；
- 无机身触地、初始穿透、硬限位、非有限值或意外 reset。

### T600.2 真值自由空间控制

使用 Isaac 箱体真值，完成 M1 原地平衡、右 Panda 掌心轨迹跟踪、O6 张合和预抓取，不建立计划接触。

验收门：

- 掌心位置误差 `<= 0.010 m`；
- 掌心姿态误差 `<= 0.05 rad`；
- O6 主动关节无硬越界，mimic 跟随方向和比例正确；
- WBC/QP feasible rate 为 `1.0`；
- 控制频率和求解时间满足实现计划冻结的实时预算。

### T600.3 真值 Contact-aware MPC

加入箱体状态、指尖接触、摩擦锥、最小法向夹持力、物体动力学和滑移约束，形成冻结的 nominal MPC 基线。

验收门：

- 抬升高度 `>= 0.10 m`；
- 连续 HOLD `>= 3 s`；
- HOLD 位置误差 `<= 0.02 m`；
- HOLD 姿态误差 `<= 0.10 rad`；
- 箱体相对手掌滑移 `<= 0.005 m`；
- 完成受控下降和释放；
- seeds `42/43/44` 各 10 次，共 `30/30` 成功；
- 无掉落、计划外碰撞、硬限位、非有限值或 M1 失稳。

### T600.4 仿真多模态感知

加入 RGB-D 和 LiDAR，完成外参标定、时间同步、点云变换、粗定位、局部点云精化和状态置信度输出。LiDAR 负责中距离粗几何和工作空间，RGB-D 负责近距离箱体位姿；控制器不得直接消费原始点云。

按以下顺序验收：

1. 离线数据集回放；
2. 感知开环在线运行；
3. 真值与估计 shadow 对比；
4. 以估计状态替换真值进入闭环。

验收门：

- 静态位置 RMSE `<= 0.010 m`；
- 静态姿态 RMSE `<= 5 deg`；
- 状态年龄、丢帧率和置信度均被逐周期记录；
- 遮挡或失配时输出明确 invalid/degraded 状态，不产生未标记跳变；
- 闭环失败可区分 perception、planner、MPC、contact 和 safety 原因。
- 使用多模态估计替换真值后，重新通过 T600.3 的 seeds `42/43/44`、共 `30/30` 完整任务门，且安全阈值不放宽。

### T600.5 受限 Residual 学习

冻结已通过 T600.3 的 nominal MPC。Residual 首版动作固定为 `9D`：

```text
[dx, dy, dz, droll, dpitch, dyaw,
 d_grip_force, d_base_height, d_base_pitch]
```

策略输入至少包含 robot state、object state、contact state、nominal command、tracking error 和有限历史。每个动作分量经过物理量程缩放、幅值 clamp、slew-rate 限制和 WBC/QP safety projection。策略不直接输出 29D 力矩。

训练随机化只覆盖已声明变量：箱体质量、质心、摩擦、接触刚度/阻尼和感知位姿噪声。每个随机化范围先通过单因素探针，再允许组合。

训练晋级顺序固定为：

```text
zero residual gate -> 10-update pilot -> 100-update short
                   -> fixed three-seed promotion -> conditional long
```

每次运行写不可变 `run_manifest.json`；每次晋级写 `promotion_manifest.json`。只有 `accepted=true`、源代码/配置/父 checkpoint SHA 匹配且无安全停止的模型，才能作为下一阶段父 checkpoint。

### T600.6 联合验收与消融

至少比较：

- Isaac truth 与 RGB-D + LiDAR；
- position-only MPC 与 contact-aware MPC；
- nominal MPC 与 MPC + Residual；
- Residual proprio-only、`+force` 和 `+vision history`。

统一指标包括任务成功率、物体位姿误差、掌心和手指跟踪误差、接触力误差与峰值、滑移、MPC/QP 求解时间、Residual RMS/饱和率、碰撞率、掉落率和失败原因分布。

Residual 的接受标准不是只看 reward：它必须提高随机化条件下的任务成功率，并且不得降低固定条件 `30/30` 安全基线。

## 8. 安全与错误处理

- 感知状态过期或低置信：停止任务推进，保持最近安全参考；悬空时进入安全下降。
- Arm/Finger MPC 不可行：不提交部分命令，保持最近安全解并记录 solver 原因。
- 检测到滑移：限制高度增长，提高到已验证范围内的夹持参考；持续滑移进入 `LOWER_SAFE`。
- QP 连续失败、非有限值、硬关节越界、计划外碰撞或 M1 失稳：终止正常任务并执行安全下降。
- 无法保证下降：进入受限 hold，停止新规划并报告不可恢复原因。

所有 fallback 原因按 perception、planner、arm MPC、hand MPC、residual、WBC/QP 和 simulator 分开记录，禁止压缩成单一 `failure`。

## 9. 代码所有权边界

建议新增独立命名空间 `m1_single_o6_contact`：

```text
Go2Pvcnn/go2_pvcnn/assets/
  m1_single_panda_o6.py
Go2Pvcnn/go2_pvcnn/perception/m1_single_o6_contact/
  contracts.py
  calibration.py
  rgbd_lidar_fusion.py
  object_state_provider.py
Go2Pvcnn/go2_pvcnn/control/m1_single_o6_contact/
  contracts.py
  grasp_planner.py
  arm_finger_mpc.py
  residual.py
  safety_projection.py
  state_machine.py
  runtime.py
Go2Pvcnn/go2_pvcnn/tasks/
  m1_single_panda_o6_contact_env_cfg.py
  m1_single_panda_o6_contact_wrapper.py
Go2Pvcnn/scripts/
  build_m1_single_panda_o6_asset.py
  verify_m1_single_panda_o6_asset.py
  m1_single_panda_o6_contact_probe.py
  m1_single_panda_o6_residual_train.py
  m1_single_panda_o6_eval.py
```

精确文件和 API 由实施计划冻结。只有真正通用且已有回归覆盖的函数可以抽取到共享模块；旧 T400/T500 公开接口不得为方便新路线而改变。

## 10. 验证体系

1. CPU 单元测试：shape、dtype、device、时间戳、finite、mimic、限幅、状态机和 fallback。
2. PXR/Isaac 资产测试：articulation、DOF、碰撞体、传感器、坐标变换和可重定位性。
3. GPU0 探针：零命令、自由空间、抓取闭环、感知 provider 切换以及 9D 每轴正负响应。
4. 训练/评估审计：配置 SHA、源 lineage、父 checkpoint、seed、逐回合指标、停止原因和晋级决定。

每个阶段必须单独留下验证日志。未运行的门必须明确标记为未验证，不能用静态测试替代 Isaac 物理验收，也不能用平均 reward 替代完整任务成功率。

## 11. 明确非目标

首轮不实现：

- 左 Panda、左 O6 或双手联合控制；
- M1 底盘移动辅助；
- 推、插入、旋拧、工具使用和 handover；
- 真实相机、LiDAR、触觉或 O6 硬件；
- VLA、语言任务规划或长时序 skill selection；
- 随机物体集合或完整 DexVerse benchmark；
- 端到端视觉到力矩策略。

这些能力只有在 T600.6 完成后才进入新的独立设计周期。

## 12. 首个实施计划边界

首份实施计划只覆盖 `T600.1`：建立并验收 M1 + 右 Panda + 右 O6 资产与冻结接口。它必须先审计 T500 已提交资产中可复用的内容，再通过 TDD 建立单臂资产，不启动感知开发、MPC 接触任务或 Residual 训练。

后续每个 T600 子阶段独立执行“设计补充或合同复核 -> 实施计划 -> TDD -> CPU/PXR/GPU 门 -> 日志与晋级”，避免一次计划跨越感知、控制和训练三个高风险边界。
