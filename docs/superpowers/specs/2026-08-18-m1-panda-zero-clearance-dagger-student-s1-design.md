# M1 + Panda 零间隙与在线 DAgger Student S1 设计

## 1. 目标

本阶段先把 M1 背部与 Panda `panda_link0` 安装平面之间的人为间隙从 `10 mm` 改为 `0 mm`，冻结新的统一 articulation 资产，再在该资产上训练一个同时协调 M1 和 Panda 的 Student。

Student 使用现实可得的本体感知、安装六轴力/力矩和任务目标，通过 Teacher 数据预热与在线 DAgger 学习 C1a 平地直线滚动和小幅 Panda 末端轨迹。动态平衡、四轮接触和安全降级始终高于末端跟踪。

本设计只授权：

- 零间隙组合资产及其验证；
- 新资产上的 C0/C1a deterministic Teacher 复验；
- Student S1 的 100 维观测、10 帧历史、23 维安全残差和在线 DAgger 训练。

本设计不授权随机六维外力课程、转向、复杂地形、PPO 微调、抓取、夹爪学习控制或实机工作。

## 2. 与既有设计的关系

零间隙部分采用并复核 [M1 + Panda 零间隙安装设计](2026-08-15-m1-panda-zero-clearance-mount-design.md)。WBC、100/10/23 Student 合同和安全优先级继承 [M1 + Panda 优先级 WBC Teacher–Student 设计](2026-08-17-m1-panda-prioritized-wbc-teacher-student-design.md)。C1a Teacher 的任务和硬门继承 [M1 + Panda Rolling WBC Teacher C1a](2026-08-18-m1-panda-wbc-teacher-c1a-design.md)。

既有总体设计把 Student 阶段描述为行为克隆后 PPO 微调。本设计只对第一个 Student S1 子阶段作更具体约束：采用 Teacher 数据预热后在线 DAgger，不在 S1 启动 PPO。后续 PPO 是否需要必须依据 S1 证据另行设计和批准。

## 3. 选定路线

采用严格串行路线：

1. 修改构建源并生成零间隙 USD；
2. 完成几何、拓扑、checksum、no-snap 和视觉检查；
3. 在新资产上重新接受 C0/C1a Teacher；
4. 仅使用新资产和已复验 Teacher 采集预热数据；
5. 启动在线 DAgger，逐级提高 Student 执行比例；
6. 进行无 Teacher 执行的 Student-only 正式验收。

不并行使用旧 `10 mm` 资产训练，也不把 `0–10 mm` 安装间隙作为域随机化。安装高度会改变整体重心和安装点力矩，因此旧资产的轨迹、标签和 checkpoint 只能用于历史对照，不能混入 S1 正式训练集或冒充新基线。

## 4. 零间隙资产门

### 4.1 构建

`Go2Pvcnn/scripts/build_m1_panda_asset.py` 中：

```python
MOUNT_CLEARANCE_M = 0.0
mount_offset_z = base_top_z - base_origin_z + MOUNT_CLEARANCE_M
```

必须通过构建源修改生成 `Go2Pvcnn/assets/m1_panda/m1_panda.usd`，不得手工平移生成后的 Prim。Panda 朝向、水平位置、URDF、网格、质量、惯量和子侧 mount pose 不变。

重建时同步更新 `generated_files.sha256`。`source_files.sha256` 只有在项目自有源输入真实变化时才更新；本阶段不修改 Panda/M1 源输入。

### 4.2 几何与动力学合同

- Panda 安装原点高度等于构建时 `BASE_LINK` 世界包围盒顶面，误差 `<= 1e-6 m`；
- 唯一 articulation root 为 `/M1Panda/BASE_LINK`；
- 总 DOF 为 25，WBC 主动协调 12 腿、4 轮和 7 Panda 臂关节，两指夹爪仍独立位置保持；
- mount 为 enabled `FixedJoint`，`excludeFromArticulation=false`；
- child `localPos1=(0,0,0)`、`localRot1=identity`；
- 一步 mount relative delta `< 1e-4 m`；
- M1/Panda 装配 collision mask 保持启用，不得产生持续内部接触抖动；
- Panda 底座方向正确且没有明显穿入 M1 主体。

零间隙表示安装原点与计算出的顶面没有额外 clearance，不表示允许网格穿透。如果 link0 网格在安装原点以下延伸并产生明显穿透，实施必须停止并回到转接板/安装面设计，不能继续降低高度或关闭更多外部碰撞来掩盖问题。

### 4.3 Teacher 复验

零间隙资产通过静态与 topology verifier 后，必须在 GPU0 重新执行：

- C0 2000-step 联合小幅运动回归；
- C1a 4000-step、关闭 Panda 目标运动基线；
- C1a 4000-step、默认开启 Panda 目标运动正式验收。

C1a 的速度、停车、位移符号、四轮接触、滚动残差、侧滑、roll/pitch、末端误差、QP、安全状态和零异常硬门全部保持不变。任何 Teacher 门失败时不得开始 Student 数据采集。

## 5. Student S1 合同

### 5.1 独立边界

Student 使用独立 Gym ID、日志根目录、数据集版本、manifest、checkpoint 和 Play 入口。不得覆盖或部分加载 A0/A1 的 60-observation/16-action checkpoint，也不得覆盖 C0/C1a Teacher 证据。

S1 任务只包含：

- 平地；
- C1a 五段纵向速度任务 `0.00, +0.05, +0.10, 0.00, -0.05 m/s`；
- 小幅、连续、带限的 Panda 六维末端目标；
- Panda 自重和运动惯性形成的真实安装反力。

S1 不主动施加随机外力，不加入负载、抓取、yaw 转向或复杂地形。

### 5.2 100 维现实观测

每步观测严格为 100 维：

| 分量 | 维度 |
| --- | ---: |
| 基座线速度、角速度、投影重力 | 9 |
| M1 16 关节位置和速度 | 32 |
| Panda 7 关节位置和速度 | 14 |
| 末端六维位姿误差 | 6 |
| 期望末端 twist | 6 |
| 四轮接触标志 | 4 |
| 安装点六维力/力矩 | 6 |
| 上一时刻 23 维动作 | 23 |

合计必须由张量合同测试证明为 100。Student 使用最近 10 帧观测；episode reset 必须同步清空历史、估计器隐状态、上一动作和动作变化率状态。

安装六维输入的坐标系、作用方向和参考点继续与未来实机传感器一致。S1 即使没有随机外力也保留该输入，因为 Panda 自重、加减速和末端运动会产生安装反力。S1 先证明确定性管线；传感器零偏、延迟、丢包和大范围噪声随机化留给后续独立阶段。

### 5.3 时序估计器与动作

10 帧历史送入时序估计器，输出：

- 显式六维安装反力估计 `W_hat[6]`；
- 表示延迟、接触变化和未建模动力学的扰动 latent。

actor 使用当前观测、历史编码、`W_hat` 和 latent，输出 23 维归一化安全残差：

- 12 个 M1 腿关节位置残差；
- 4 个轮关节速度残差；
- 7 个 Panda 臂关节位置残差。

Student 不输出裸力矩。动作通过显式单位映射、逐通道幅值限制、每周期变化率限制、既有阻抗执行层和 balance-first 安全监督器。两指夹爪不属于 23 维动作。

nominal command 由不依赖 Teacher 的任务命令层产生：12 个腿关节使用 settling 后冻结的 C1a 安全站姿，4 个轮关节使用 shaped `vx / wheel_radius`，7 个 Panda 臂关节使用 settling 后冻结的安全弯曲姿态。末端目标只通过六维位姿误差和期望 twist 进入 Student 观测，由 Panda 的 7 维残差完成小幅跟踪，不依赖部署时不可用的 Teacher 或隐式 IK。Teacher 标签是安全处理后的 WBC 目标与同一步 nominal command 的差值，并经过与 Student 完全相同的归一化、限幅和变化率合同。测试必须证明“nominal + Teacher residual”能重构 Teacher 的安全目标。

## 6. 在线 DAgger 数据流

### 6.1 预热

先由零间隙资产上已复验的 Teacher 独立执行，记录：

- 100 维可部署观测及 10 帧历史；
- Teacher 23 维安全残差标签；
- 理想安装六维力、Student 风格六维输入和 `W_hat` 监督目标；
- WBC/QP、接触、末端误差、安全状态与降级原因；
- nominal command、限幅前后动作和执行动作。

预热数据只用于初始化估计器和 actor，防止随机 Student 从第一步制造大量失稳、reset 和无效标签。

### 6.2 DAgger rollout

之后由 Student 在仿真中实际访问状态，Teacher 在相同状态持续计算标签。执行动作由受控混合与安全门决定：

```text
deployable observation history -> Student -> proposed residual
privileged state              -> Teacher -> safe residual label
Student/Teacher mixture + safety supervisor -> executed residual
all state/action/safety evidence -> versioned replay buffer
```

Teacher 执行比例只能在独立验证集通过当前阶段安全门后下降。具体比例和每阶段样本量进入实施计划并由短程测量确定，不在设计中伪造未经运行验证的魔数。

Teacher 接管、`SCALE/HOLD/RETRACT/TERMINATE`、接触丢失和 Student 大误差样本必须保留并提高采样权重，不能只保留正常 TRACK 轨迹。不同环境的 Teacher warm-start、Student 历史、replay 写入和 reset 必须完全隔离。

现有 C1a 正式入口严格限制为单环境。DAgger 实施必须新增批量编排层，使每个环境拥有独立的 command shaper、trajectory、motion distributor、QP warm-start、安全状态和 settling 中心；不得删除 C1a 单环境限制或用跨环境共享状态绕过。先通过小批量隔离测试和 GPU smoke，再扩展到正式 64 环境采集。

### 6.3 损失

S1 至少包含：

- 23 维安全残差蒸馏损失；
- 六维 `W_hat` 监督损失；
- 安全状态/接管边界辅助损失；
- 动作变化率和限幅一致性损失。

各动作组必须按物理尺度归一化，避免轮速或 Panda 通道因数值范围支配总损失。损失权重属于实施期消融参数，必须记录进 manifest，不在本设计中硬编码虚假最优值。S1 不包含 PPO loss 或 privileged critic 优化。

## 7. 安全与失败处理

- 非有限观测、六轴无效、QP 不可行、轮接触不足或姿态越界时，Student proposed action 不得直接执行；
- 安全状态保持 `TRACK -> SCALE -> HOLD -> RETRACT -> TERMINATE`；
- `SCALE` 同时限制底盘与 Panda；
- `HOLD` 冻结 Panda 目标并平滑制动；
- 仅在底盘速度低于既有阈值后进入 `RETRACT`；
- 持续失稳或非有限状态终止 episode，并锁存首个根因；
- 训练日志记录 Teacher 接管次数、接管原因、动作差异、恢复时间和 reset 原因；
- checkpoint/数据集 manifest 的资产 SHA、Teacher commit、100/10/23 合同、动作尺度、控制周期或 DAgger 阶段不一致时，加载前失败。

资产门失败时不覆盖已接受的 `10 mm` 资产证据。Teacher 复验失败时不启动采集。Student 当前 DAgger 级别失败时保留 checkpoint 和证据，回到该级别诊断，不自动扩大课程。

## 8. 测试与验收

### 8.1 资产和 Teacher

- 静态测试先对 `MOUNT_CLEARANCE_M == 0.0` 产生有效 RED，再完成 GREEN；
- checksum、PXR reopen、CPU topology verifier 和一步 no-snap 均 exit `0`；
- 视觉检查确认贴合、方向正确且无明显穿透；
- GPU0 C0 2000-step 和两组 C1a 4000-step 复验通过原有全部硬门。

### 8.2 Student 数学与纯测试

- 100 维单帧、10 帧历史和 23 维动作 shape/dtype/device 合同；
- reset 隔离和历史清零；
- nominal + residual 重构 Teacher 安全目标；
- 动作单位映射、幅值、变化率和饱和诊断；
- `W_hat` 坐标、符号和监督目标；
- replay buffer 版本、环境隔离、困难样本保留和 manifest 拒绝不匹配；
- 固定 seed 下 Teacher-only、混合和 Student-only 动作选择可重复。

### 8.3 Student-only 正式验收

正式评估不允许 Teacher 执行动作，只允许旁路计算 Teacher 对照标签。使用 3 个固定 seed，每个至少 64 个环境，要求：

- 至少 95% episode 完成五段 4000-step 任务；
- Student 平衡成功率达到同资产 Teacher 的至少 95%；
- C1a 的速度 RMSE、停车时间、位移符号、四轮接触、滚动残差、侧滑、roll/pitch、末端误差、QP 可行率、动作连续性和零异常硬门全部适用于 Student；
- 正式成功 episode 中 `HOLD`、`RETRACT`、`TERMINATE` 均为零；
- 无关节越限、机身触地、非有限值、Panda 目标跳变和意外 reset；
- 六维 `W_hat` 误差、23 维动作误差和训练期 Teacher 接管率完整记录，但不能用降低硬安全门换取拟合分数。

只有 Student-only 验收通过后，才允许另行设计随机外力/传感器域随机化阶段；转向、抓取、PPO 和实机仍需各自批准。

## 9. 计划代码边界

实施计划预计涉及：

```text
Go2Pvcnn/scripts/build_m1_panda_asset.py
Go2Pvcnn/scripts/verify_m1_panda_asset.py
Go2Pvcnn/assets/m1_panda/m1_panda.usd
Go2Pvcnn/assets/m1_panda/generated_files.sha256
Go2Pvcnn/tests/test_m1_panda_asset_static.py

Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_contracts.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/dagger.py
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_s1_env_cfg.py
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_dataset.py
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_checkpoint.py
Go2Pvcnn/scripts/m1_panda_wbc_collect.py
Go2Pvcnn/scripts/m1_panda_student_train.py
Go2Pvcnn/scripts/m1_panda_student_play.py
Go2Pvcnn/tests/test_m1_panda_student_*.py
```

实施时可以为清晰边界拆分上述模块，但不得把 A0/A1、C0/C1a 的入口和 checkpoint 就地改造成 Student。先完成零间隙资产与 Teacher 复验，再编写 Student 数据与训练代码；两部分必须有独立提交和回滚点。

## 10. 已知限制

零间隙仿真贴合不等价于真实转接板、螺栓、六轴传感器法兰或机械强度已经验证。T400.3 机械验算仍是实机动态摆臂和负载试验的硬前置。

S1 只证明平地直线滚动和小幅 Panda 运动下的联合蒸馏。它不证明外部冲击鲁棒性、转向、复杂地形、抓取负载或实机可部署性。
