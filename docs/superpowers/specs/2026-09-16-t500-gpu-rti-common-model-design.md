# T500 GPU RTI 共同预测模型设计

## 1. 状态与边界

本设计补齐 `2026-09-15-t500-gpu-mpc-latent-runtime-design.md` 中未定义的
机器人力矩、双手抓持力和箱体运动之间的共同预测合同。用户已接受推荐方向：
保持公开动作 `[25, 43]` 不变，只在 RTI 内部增加左右手各 6 维抓持 wrench。

它不是接触模式 MPC，也不新增感知、Residual、训练入口或公开 T400/T500
合同。生产指尖先验仍取决于 E0 资格；共同模型允许 prior-off 开发。

## 2. 方案取舍

评估过三种路径：

1. **43 维 effort + 12 维内部抓持 wrench（采用）**：真实连接机器人与箱体，
   复用现有箱体/抓持约束，同时不把辅助变量发布为执行动作。
2. 仅 GPU 化现有 Object/Arm/Hand/WBC 层级：风险最低，但仍是串行目标传递，
   不能满足“完整动作由共同预测得到”的目标。
3. 增加 30 维指尖接触力和接触模式：表达力最强，但会变成未获授权的接触 MPC，
   还需要新接触状态估计，因此不采用。

## 3. 决策变量与状态

每个生产节点 `k=0..24`，内部增量控制为：

```text
u_k = [tau_k(43), w_left_k(6), w_right_k(6)]
```

- `tau` 的顺序严格沿用现有 `actuation_matrix[59,43]`；不生成 59 维命令。
- `w_left/right` 是手施加到箱体上的 base-frame spatial wrench，线力在前、
  力矩在后，顺序和现有 Object MPC 完全相同。
- wrench 仅是优化辅助量。对外结果、warm-start、latent encoder 和执行接口仍是
  `[B,25,43]`；诊断可另行记录内部 wrench，不允许其泄漏为公开动作。

共同状态由现有测量派生，不创建第二套公开 dataclass。内部110维状态为基座局部
SO(3)切空间位姿/速度12维、43个主动坐标的 `q/qd`86维及箱体 pose/twist12维；
左右掌状态由该状态和真实 Jacobian 预测，阶段与接触摘要作为固定参数。
该布局即使当前固定基座也保留基座块，以支持之后滑动时不更换合同。
被动/mimic 惯量继续保留在 59 维质量矩阵中。

## 4. 力学耦合

左右掌空间 Jacobian `J_L,J_R` 必须来自与当前 59 维广义坐标和 base frame
一致的真实模型。定义 `w_L,w_R` 为作用于箱体的 wrench，则机器人受到等大反向
wrench。每个节点的约束动力学为：

```text
[ M  -Jwheel^T ] [qdd] = [-b + S tau - J_L^T w_L - J_R^T w_R]
[Jwheel    0    ] [lam]   [                  -wheel_bias          ]
```

箱体使用现有 Object MPC 的刚体质量、惯量、重力、左右抓持偏置和 wrench-to-
acceleration 映射，以 `+w_L,+w_R` 推进 pose/twist。由此同一个辅助 wrench 同时
驱动箱体并反作用到机器人，禁止把两套 wrench 独立优化或只复制上游目标。

符号、frame、spatial 分量顺序必须通过虚功/功率一致性和 CPU float64 fixture
验证；不允许依靠视觉上“运动方向正确”判断。缺失、非有限、frame/layout/phase
身份不一致的 Jacobian 或惯量使该 batch row 原子拒绝。

## 5. 统一 40 ms 网格

生产 RTI 保持 `dt=0.04 s,H=25`。在节点内，`tau,w_L,w_R` 零阶保持：

- 机器人/机械臂执行两个确定性的 20 ms 常加速度子步；
- O6 执行四个 10 ms 子步，使用同一全身 `qdd` 对六个主动轴积分；
- 箱体执行一个 40 ms 常 wrench 刚体步，公式与现有 Object MPC 的离散化一致；
- 每个子步从状态重新计算残差，但首版 Gauss-Newton 线性化固定在当前 nominal
  horizon；Task 6 之前不引入可变子步或接触事件分支。

位置采用 `q += dt*qd + 0.5*dt^2*qdd`，速度采用 `qd += dt*qdd`。
旋转误差和积分沿用当前 SO(3) shortest-path/base-frame spatial 语义，禁止直接将
欧拉角线性相加。节点 0 总是由最新测量覆盖。

## 6. 约束与代价

### 硬约束

- 原有 43 effort 限制、四轮 effort 固定为零和公开 joint/rate/acceleration 限制；
- wheel nonholonomic 约束及动力学/KKT residual；
- 现有 Object MPC 的阶段相关 normal-force、摩擦、moment、platform yaw/rate 限制；
- 已存在的 O6 关节/速度、碰撞和最小间距安全限制；
- 全 horizon 数值有限、维度/设备/layout/phase 身份有效。

不把掌心—箱体几何闭合设为硬等式。当前没有刚性抓取接触模式与闭环约束模型，
强行硬化会制造虚假可行性或过约束。

### 软代价

- 箱体 pose/twist 目标；
- 左右掌相对箱体目标及速度；
- wrench nominal/slew 和左右负载分配；
- arm/O6 nominal、effort/slew 和 terminal tracking；
- E0 可用后才加入五指尖自然运动先验，且仅前 0.20 s、可退让、冲突时归零。

所有权重显式配置、有限且非负。CPU/GPU 比较容差只用于数值验证，不能替代或
修改物理安全阈值。

## 7. RTI、接受与退化

每周期只做一次 Gauss-Newton RTI 修正，固定候选步长
`(1.0,0.5,0.25,0.125)` 并行评估。候选必须同时通过机器人动力学、箱体动力学、
所有硬约束和完整 horizon 有限性检查，才能按 merit 和固定 tie-break 规则接受。

任一侧 Jacobian/wrench/约束失败都拒绝该环境的**整条**候选，保留同一环境最近
接受的完整 `[25,43]` horizon；不能接受另一侧或只接受首动作。若没有有效最近
horizon，则沿既有 OSQP/CPU/last-safe 层级退化。其他 batch row 不受污染。

固定上下界相等时先验证 nominal trajectory，再直接采用唯一值，不调用求解器。
prior-off 和 prior-on 走同一安全验证路径；先验故障只移除软代价。

## 8. 组件边界

Task 5 在既有三个文件边界内增加：

- `lq_problem.py`：固定形状的共同状态/控制合同、40 ms assembler、线性化和硬约束；
- `line_search.py`：四个候选的 batched rollout、merit、完整安全 mask 与确定性选择；
- `planner.py`：Task3 state/dynamics/warm-start 编排、一次 RTI、原子发布。

Task 5 使用 eager PyTorch dense correctness backend。H25→32 scan、Triton、CUDA
Graph 和 runtime 路由分别属于后续 Tasks 6/8，不能提前混入。生产 Play 入口仍在
后端完整实现、benchmark manifest 和正式验证前 fail closed。

## 9. 验证合同

TDD fixture 至少覆盖：

1. 真实符号/框架的机器人—箱体等大反向 wrench parity；
2. 2×20 ms arm、4×10 ms O6 与 1×40 ms box rollout；
3. feasible、active-limit、infeasible、单侧 Jacobian 失败和非有限输入；
4. 箱体、双掌、负载分配、arm/O6 及 prior-off 零贡献的各代价块；
5. 固定四步长、确定性 tie-break、相等边界旁路；
6. whole-horizon atomic rejection、逐 row 隔离及 last-safe 保留；
7. CPU float64 小 fixture 对 GPU float32 的状态、残差、首动作和接受决策 parity；
8. fixed-address workspace 和 warm-up 后显存有界。

测试不能用复制目标、独立节点 QP 或 mock dynamics 证明“共同预测”。Isaac 物理、
性能和正式 30 条验收仍属于 Task 10。

## 10. 明确非目标

- 不优化 59 维独立动作；
- 不增加 30 维指尖力、接触模式切换或抓取/脱离事件优化；
- 不改变现有 CPU float64、state machine、safety projection 或 public thresholds；
- 不因缺失 E0 构造合成生产 prior；
- 不把基础设施测试称为实时、Play 或物理验收。
