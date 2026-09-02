# M1 + 双 Panda + 双 O6 分层双手操作 MPC 设计

## 1. 目标与范围

在现有 M1 项目中新增一条完全隔离的双臂双手操作路线：M1 顶部通过一个公共 yaw 回转关节连接刚性双臂平台，平台左右固定两条 Panda 7-DOF 机械臂，左右腕端分别安装 Linker Hand O6 左手和右手。

首个任务严格限定为固定条件仿真：M1 原地保持平衡，双手从已知位姿接近一个 `0.12 x 0.18 x 0.10 m`、`0.5 kg` 的刚性箱体，建立夹持后抬升至少 `0.10 m`，保持至少 `3 s`，再受控下降和释放。

第一阶段只实现确定性模型控制，不训练 PPO/RL，不加入视觉估计、随机物体、移动底盘操作、Student 蒸馏或实机部署。MPC 直接使用 Isaac Lab 的箱体位姿、速度和接触力真值。

现有单 Panda 资产、Gym ID、MPC/WBC、训练配置和 checkpoint 必须继续可用且默认行为不变。

## 2. 已确认机械拓扑

- M1：12 个腿关节和 4 个轮关节。
- 公共回转平台：1 个主动 yaw 关节，范围 `[-90 deg, 90 deg]`。
- 双 Panda：左右各 7 个主动关节，共 14 个；原 Panda 两指夹爪不进入新组合体。
- 双 O6：左右各 6 个主动控制通道和 5 个 mimic 联动关节。
- 主动控制通道合计为 `16 + 1 + 14 + 12 = 43`。
- 若 USD importer 将每只手的 11 个活动关节均报告为物理 DOF，则组合 articulation 预计为 53 DOF；最终数目必须以 PXR 和 Isaac 运行时探测为准，不能只依赖文件推算。

双臂平台是刚体，整体绕 M1 竖直轴旋转；两条 Panda 固定在平台上并共同随平台旋转。首版使用左右对称安装，初始基座中心间距约 `0.40 m`，精确变换在资产阶段通过无碰撞、工作空间覆盖和重心稳定探针确定并写入 manifest。

## 3. 总体控制架构

采用分层耦合 MPC：

```text
box target pose
    -> 25 Hz bimanual object MPC
       -> box trajectory
       -> shared platform yaw trajectory
       -> left/right palm 6D targets
       -> left/right target contact wrench
    -> two 50 Hz arm MPC controllers
       -> left/right Panda q_ref and qd_ref
    -> two 100 Hz O6 hand MPC controllers
       -> 12 active hand position/velocity references
    -> 200 Hz whole-body WBC/QP
       -> 43 active actuator commands
```

物体级 MPC 负责双手之间的全局协调；Arm MPC 负责把掌心目标转为可执行的 Panda 关节参考；Hand MPC 负责 O6 的接触建立、夹持力和防滑；WBC/QP 统一处理 M1 平衡、平台、双臂动力学、接触和最终安全约束。

首版 M1 轮速目标为零，但腿、轮接触和底盘平衡约束保持激活。平台 yaw 可由物体级 MPC 联合规划，但使用低速限制。

## 4. 原子状态与数据流

每个 200 Hz 控制周期由 runtime adapter 生成一个带单一时间戳的原子状态快照。所有子控制器必须使用这一快照或由它确定性预测出的状态，不允许分别从 Isaac 读取不同时刻的数据。

快照至少包含：

- M1 base pose、twist、关节状态、轮接触和支撑状态；
- 平台 yaw、速度和执行器状态；
- 左右 Panda 的 `q/qd`、掌心 pose/twist、Jacobian、质量矩阵和 bias；
- 左右 O6 的 6 个主动通道、mimic 关节状态和指尖/掌面接触；
- 箱体 pose、twist、质量、惯量和接触 wrench；
- 所有控制器上一安全解和连续失败计数。

所有纯控制合同使用显式 batch、shape、dtype、device、finite 和 timestamp 校验。任一字段不合法时拒绝整个新周期，不能提交部分输出。

## 5. 双手物体级 MPC

物体级 MPC 以箱体 pose/twist、平台 yaw 状态和左右聚合接触 wrench 为状态，初始按 25 Hz 重规划并覆盖约 `1 s` 的预测窗。

决策变量包括：

- 箱体期望加速度和预测 pose/twist；
- 平台 yaw、速度和加速度；
- 左右掌心 6D pose/twist；
- 左右期望接触 force/moment；
- 接触、摩擦和可达性松弛量。

约束包括：

- 箱体刚体动力学和重力；
- 左右手力/力矩平衡；
- 线性化摩擦锥与最小法向夹持力；
- 平台位置、速度和加速度限制；
- 双臂工作空间和 manipulability 的保守近似；
- 左右掌心相对箱体的抓握几何；
- 箱体目标 pose/twist 和阶段相关高度限制。

目标函数按安全优先级惩罚箱体跟踪误差、掌心几何误差、接触 wrench 偏差、平台运动、松弛量和控制变化率。

## 6. 左右 Arm MPC

现有单 Panda `LinearizedArmMpc` 保持兼容。实施时只抽取真正通用、已有测试覆盖的单臂核心，然后分别实例化左右控制器；不得把双臂状态塞进旧 `ArmMpcInput`。

每侧继续使用 `50 Hz / 20 nodes / 0.4 s` 合同，输入本侧 Panda 动力学和由物体级 MPC 生成的掌心轨迹，输出：

- `q_ref/qd_ref/qdd`；
- 预测掌心 pose/twist；
- 预测腕部和平台安装作用 wrench；
- feasible、fallback、saturation、manipulability、joint margin 和 tracking diagnostics。

任一侧不可行时，两侧同时停止任务推进，防止另一侧继续对箱体施加不对称运动。

## 7. 左右 O6 Hand MPC

每只 O6 的主动控制顺序冻结为：

```text
thumb bend, thumb yaw, index bend,
middle bend, ring bend, little bend
```

五个 distal 关节继续按 USD/URDF mimic 关系联动，不作为独立决策变量。Hand MPC 初始按 100 Hz 重规划，预测约 `0.2 s`，输入主动关节状态、指尖/掌面接触和本手目标 wrench，输出主动通道 position/velocity reference。

约束覆盖关节范围、厂商速度上限、接触法向力、摩擦、防滑、闭合速率和命令变化率。URDF 中统一的 `effort=100` 不作为真实安全上限。初始目标夹持力为每手约 `8-15 N`，正常阶段双手总抓握力不超过 `40 N`；任何提高必须经过独立验证。

官方 O6 规格参考：<https://linkerbot.cn/>。其公开参数包括 6 个主动自由度、5 个被动自由度、360 g 重量和 O6 指尖/抓握能力；实现不得把营销最大负载直接解释为仿真关节 effort。

## 8. WBC/QP 与约束边界

200 Hz WBC/QP 是唯一最终执行边界，管理 43 个主动控制通道。新双臂双手约束放入独立的 `m1_bimanual_coordination` 模块；只复用现有通用 QP、张量合同和经验证的单臂算法，不改变旧单臂 `constraints.py` 的公开合同。

硬约束至少覆盖：

- M1 姿态、base height、support margin 和四轮接触；
- 平台 yaw、速度、加速度和 effort；
- 双 Panda position、velocity、acceleration、effort 和奇异性余量；
- O6 主动关节范围、速度和抓握力；
- 双臂互碰、机械臂与 M1、手与平台、非接触部位与箱体的最小距离；
- 箱体摩擦锥、力闭合和滑移；
- 接触阶段允许集合。

初始平台速度限制为 `0.25 rad/s`。M1 第一阶段只允许原地平衡，不能使用底盘移动来补偿不可达目标。

## 9. 任务状态机

正常路径：

```text
APPROACH -> PRELOAD -> GRASP -> LIFT -> HOLD -> LOWER -> RELEASE -> DONE
```

- `APPROACH`：无计划接触，掌心到达箱体两侧预抓取位姿。
- `PRELOAD`：建立轻接触并验证左右接触方向。
- `GRASP`：提高夹持力，满足力闭合和防滑条件。
- `LIFT`：箱体抬升至少 `0.10 m`。
- `HOLD`：稳定保持至少 `3 s`。
- `LOWER`：将箱体受控放回支撑面。
- `RELEASE`：确认支撑稳定后张手并撤离。

异常路径：

```text
HOLD_SAFE -> LOWER_SAFE -> SAFE_RELEASE
```

箱体悬空时禁止突然张手。只有箱体已稳定接触支撑面，才允许进入任何 release 状态。

## 10. 安全与回退

1. 物体级 MPC 不可行：保持上一安全双掌/平台轨迹，不再提高箱体高度。
2. 单侧 Arm MPC 不可行：左右两侧同时停止推进并保持最近安全参考。
3. Hand MPC 不可行：保持最近安全夹持参考，禁止突然释放。
4. 连续不可行或检测到滑移：进入 `HOLD_SAFE` 并协同降低箱体。
5. 非有限值、硬关节越界、计划外碰撞、M1 失稳或 QP 连续失败：终止正常任务，执行安全下降；无法保证下降时进入受限 hold 并报告明确原因。

命令采用整周期原子提交：任一子控制器本周期失败，整组新命令都不写入仿真。object/left-arm/right-arm/left-hand/right-hand/WBC 的 fallback reason 必须分别记录，禁止合并为一个模糊的 `failure`。

## 11. 资产构建

新增项目资产目录：

```text
Go2Pvcnn/assets/m1_dual_panda_o6/
  m1_dual_panda_o6.usd
  platform.usd
  o6_left/
  o6_right/
  asset_manifest.json
```

原始 M1、Panda 和 O6 文件保持不变。O6 源资产来自用户提供的完整 USD 分层目录。右手入口与 `configuration/` 同级，可以按现有相对引用解析；左手入口当前位于 `configuration/` 内却仍引用 `configuration/...sensor.usd`，项目副本需要将入口规范到上一层。其余 `base/physics/robot/sensor` 层不重建。

构建器必须：

- 创建刚性平台和主动 yaw joint；
- 复制两套 arm-only Panda 并加 `left_`、`right_` 命名空间；
- 通过固定关节把 O6 `hand_base_link` 连接到对应腕端；
- 清除子资产多余 articulation root、ground、physics scene 和 root joint；
- 输出唯一 articulation root；
- 为手指生成或选用简化凸碰撞体，禁止以高面数 STL 动态三角网格直接承担指尖接触；
- 写入源 SHA、安装变换、关节顺序、mimic 映射、碰撞参数和构建版本 manifest。

## 12. Isaac Lab 环境

新增独立 Gym ID：

```text
Isaac-M1-DualPanda-O6-Bimanual-Lift-v0
```

场景包含组合 articulation、固定支撑台、箱体、左右指尖/掌面/腕部接触传感器、平台和 M1 接触传感器，以及箱体真值状态。首版箱体初始位姿固定，不做 reset 随机化。

控制实现建议位于：

```text
Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/
Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py
Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py
Go2Pvcnn/scripts/build_m1_dual_panda_o6_asset.py
Go2Pvcnn/scripts/verify_m1_dual_panda_o6_asset.py
Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py
```

精确文件清单由后续实施计划冻结。

## 13. 验证与验收

### 13.1 资产静态门

- 所有 USD reference 可解析；
- 单一 articulation root；
- 43 个主动控制通道；
- 运行时物理 DOF 数与 manifest 一致；
- 质量、惯量、关节限制和安装变换 finite；
- 无重复名称、独立 physics scene 或意外 root joint。

### 13.2 物理稳定门

- 零命令和安全初始姿态连续运行至少 2000 个仿真步；
- 平台、双臂和双手安装点无跳变或超限漂移；
- 无 base contact、翻倒、硬关节越界、非有限值或意外 reset；
- 四轮持续接触；
- 手指无初始穿透、接触爆炸或非计划自碰撞。

### 13.3 控制器门

- 每个 MPC 覆盖可行、不可行、非有限输入和上一安全解回退；
- 左右镜像输入产生镜像输出；
- 接触力满足摩擦锥和力闭合；
- 任一子控制器失败时不提交部分命令；
- WBC/QP feasible rate 为 `1.0`。

### 13.4 完整任务门

固定场景使用 seeds `42/43/44`，每个 seed 至少 10 次：

- 抬升高度 `>= 0.10 m`；
- HOLD 连续 `>= 3 s`；
- HOLD 位置误差 `<= 0.02 m`；
- HOLD 姿态误差 `<= 0.10 rad`；
- 箱体相对双掌滑移 `<= 0.005 m`；
- object MPC feasible rate `>= 0.98`；
- 两个 Arm MPC 和两个 Hand MPC feasible rate 各 `>= 0.99`；
- WBC/QP feasible rate `1.0`；
- `abs(roll)`、`abs(pitch) <= 10 deg`；
- 无掉落、计划外碰撞、硬关节越界或非有限值；
- 完成受控下降和释放，箱体稳定留在支撑面。

30 次全部成功才接受固定条件第一版。

## 14. 明确非目标

- 不训练 PPO/RL；
- 不使用视觉或学习式状态估计；
- 不随机化箱体尺寸、质量、摩擦或初始 pose；
- 不允许 M1 底盘移动；
- 不实现 handover、箱体旋转或动态抛接；
- 不声明实机承载、抓握安全或部署能力；
- 不覆盖旧资产、Gym ID 或 checkpoint。

## 15. 完成定义

第一阶段只有在资产依赖闭合、单 articulation、物理稳定、各层 MPC/WBC 纯测试、真实 Isaac GPU smoke、三 seed 共 30 次完整任务以及日志/manifest 全部通过后才完成。GUI 中看见一次成功抓取不构成验收。
