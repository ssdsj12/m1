# M1 + Panda 8D Residual WBC 第一版设计

## 目标

按《M1 + Panda 协调控制系统优化修改书》的推荐顺序，先实现可独立验证的第一版 `8D Residual + 既有 WBC/QP + 安装点六维力反馈`。本阶段只覆盖文档 Phase 1–4：冻结 8D 动作合同、将 6D 虚拟修正 wrench 接入 QP、接入机身高度与支撑姿态残差、接入安装点六维力反馈。

现有 `103 observation / 23 joint-effort action` Coordinated PPO、Folded Load 课程、C0 standing Teacher、C1a rolling Teacher 和已有 checkpoint 必须保持可用并作为对照基线。本阶段不训练策略，不实现 Arm MPC，不声称完成抓取、复杂地形、Student 蒸馏或实机能力。

## 范围与不变量

- 新控制链使用独立配置、wrapper、入口和 Gym ID；旧 `Isaac-M1-Panda-Coordinated-v0` 默认行为不变。
- M1 + Panda 继续使用已接受的零间隙单 articulation 资产，不修改 USD、关节顺序、PD 参数和 200 Hz 物理频率。
- 既有 WBC/QP 的 23 个受控关节顺序继续是 `12 legs + 4 wheels + 7 Panda arm`。
- 旧 23D PPO 输出、checkpoint shape、训练入口和消融能力保持兼容。
- 第一版只验证手工给定或全零的 8D residual；PPO actor、multi-branch network 和 Arm MPC 留给后续独立设计。
- 所有新增纯控制计算必须支持 CPU `float64` 参考路径；Isaac wrapper 边界负责设备和 dtype 转换。

## 方案

采用并行新增方案，不在现有 23D wrapper 中加入模式开关。新链路复用已验收的 motion distribution、standing/rolling WBC、QP、阻抗和安全监督，但将 residual 的生成、状态、观测和投影放在独立组件中。

```text
normalized residual[8]
        |
        v
幅值限制 + slew limit + finite gate + per-env state
        |
        +---- physical wrench residual[6]
        +---- delta height[1]
        +---- delta stance[1]
        |
        v
mount wrench bias/filter/feedback + safety projection
        |
        v
WBC reference correction
        |
        v
existing standing/rolling WBC + reference QP
        |
        v
q_des[23], qd_des[23], tau_ff[23]
```

## 8D 动作合同

通道顺序固定为：

```text
0 Fx
1 Fy
2 Fz
3 Mx
4 My
5 Mz
6 delta_height
7 delta_stance
```

策略或手工输入必须是有限的 `[..., 8]` 浮点张量，归一化输入范围为 `[-1, 1]`。超出范围的有限值先裁剪；NaN/Inf 整批拒绝，不允许静默归零。

物理幅值固定为：

```text
Fx/Fy        +/-30 N
Fz           +/-50 N
Mx/My        +/-15 Nm
Mz           +/-8 Nm
delta_height +/-0.04 m
delta_stance +/-0.08 rad
```

每个环境维护独立的上一帧物理 residual。默认每个 5 ms 控制步最大变化量为满幅的 `5%`；配置显式保存八个通道的幅值与 slew limit。reset 只清零指定环境，不能影响其他环境。composer 返回裁剪前后、幅值饱和、slew 饱和和当前物理 residual 的只读诊断。

## 安装点六维力反馈

现有 `m1_panda_mount_wrench_b` 继续提供关于 `BASE_LINK` 原点、在 base frame 表达的 `[Fx,Fy,Fz,Mx,My,Mz]`。第一版使用：

```text
W_cmd = K_w (W_ref - W_measured_filtered) + delta_W_RL
```

- `W_ref` 第一版固定为零，但作为显式六维配置保留。
- filter 使用逐环境一阶低通；reset 时用首个有限样本初始化，避免从零状态制造瞬态。
- 可选 bias calibration 只在静止 warm-up 内更新；进入控制后 bias 冻结。
- 默认反馈增益保守设置为 force 三轴 `0.15`、moment 三轴 `0.10`。
- 合成后的 `W_cmd` 再按 8D 合同中的六维物理上限裁剪。
- `W_cmd` 是经控制器生成的虚拟修正 wrench，通过 `J_mount^T W_cmd` 进入 WBC；原始传感器 wrench 不再次作为物理外力注入，避免双重计数。
- 配置反馈增益为零且 residual 为零时，传入现有 WBC 的修正 wrench 必须精确为零。

启动 mount wrench 已知存在较大瞬态，因此正式 GPU smoke 必须包含 warm-up/filter 诊断；第一版不能用启动峰值推断实机载荷。

## Height 与 Stance 映射

`delta_height` 修改当前 WBC 的机身高度参考，而不是直接写关节动作。高度参考先限制在名义高度 `0.6115 m +/- 0.04 m`，再由现有 base PD 生成 z 方向加速度目标。

`delta_stance` 只修改四个 ABAD 关节的姿态参考。规范腿顺序为 `FAR, FBL, RAR, RBL`，展开符号固定为 `[+1, -1, +1, -1]`；正值表示横向展开，负值表示收窄。修改后的关节参考必须裁剪到 soft limits，再由现有腿部 PD 生成加速度目标。HIP/KNEE、轮和 Panda 参考不由该通道直接修改。

## 连续底盘参与度

`motion_distribution.py` 增加连续参与度：

```text
alpha_base = clip((0.20 - sigma_min) / (0.20 - 0.08), 0, 1)
```

当 `alpha_base=0` 时保持 arm-only 分配；处于 `(0,1)` 时按比例缩放 planar base 的速度上下界；达到 `1` 时允许完整 base authority。若缩放后的问题不可行，可按既有 fallback 提升到完整 base authority，并记录原因。

`MotionDistributionResult` 新增 `base_participation`，既有 `base_active` 布尔字段继续保留，定义为最终底盘 authority 大于零。旧调用方未请求新配置时仍使用相同默认阈值和 fallback；新增特征测试锁定安全区零参与和临界区完整参与。

## WBC/QP 接入

新增纯函数把 residual 命令应用到 `StandingWbcInput` 的副本：

- `external_wrench` 写入有限且已投影的 `W_cmd`；
- `base_acceleration[2]` 使用高度修正后的参考重新计算；
- `leg_acceleration` 中四个 ABAD 元素使用 stance 修正后的参考重新计算；
- 其他字段逐项保持相同，不原地修改输入。

standing 与 rolling WBC 继续使用同一个 `build_standing_wbc_problem` 和 `solve_reference_qp`。本阶段不增加第二套优化器，也不让 RL 直接输出腿、轮或 Panda 力矩。

零 residual、零反馈增益时，修改后的 WBC problem 的 equality、inequality、Hessian、gradient、torque matrix 和 torque offset 必须与旧调用精确相同。

## Safety 与错误处理

新增 residual safety projection 在 WBC 构造之前执行：

- 输入、测量 wrench、滤波状态或合成命令非有限：拒绝当前 step，并将原因报告给现有 safety supervisor；
- `TRACK`：允许完整配置幅值；
- `SCALE`：wrench、height、stance 同时乘现有 safety scale；
- `HOLD/RETRACT/TERMINATE`：residual 精确清零；
- QP 不可行：不得输出未经验证的 effort，沿用现有 HOLD/RETRACT/TERMINATE 处理；
- reset 清理 residual/filter/bias warm-up 状态，只影响指定环境。

所有降级必须产生可枚举的原因和有限诊断，不允许 catch-all 后继续执行旧 residual。

## Observation 第一版边界

新增纯 `residual_observation.py`，定义后续策略可复用的分组输入，但本阶段不建立 PPO 网络。分组至少包含：

- M1 本体状态与 base target；
- Panda 关节、末端误差和 desired twist；
- filtered mount wrench；
- `sigma_min`、joint-limit minimum/mean margin、support margin；
- previous physical residual[8]。

builder 必须检查 batch、dtype、device 和 finite 性，并返回带命名分组与确定性 flatten 顺序的结果。第一版 play/probe 可只使用诊断分组；旧 103D observation 不修改。

## 文件边界

新增：

```text
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/whole_body_residual.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/residual_observation.py
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_residual_wbc_wrapper.py
Go2Pvcnn/scripts/m1_panda_residual_wbc_play.py
Go2Pvcnn/tests/test_m1_panda_whole_body_residual.py
Go2Pvcnn/tests/test_m1_panda_residual_observation.py
Go2Pvcnn/tests/test_m1_panda_residual_wbc.py
Go2Pvcnn/tests/test_m1_panda_residual_qp.py
Go2Pvcnn/tests/test_m1_panda_residual_action_contract.py
Go2Pvcnn/tests/test_m1_panda_residual_wbc_play_static.py
```

修改：

```text
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py
Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py
```

若 wrapper 只包装已有 C0/C1a Gym 环境即可满足 8D 手工控制，则不新增重复的 Isaac env cfg；底层 action manager 仍接收 WBC 生成的 23D effort，新 wrapper 对外暴露 8D。

本阶段不修改：

```text
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_wrapper.py
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py
Go2Pvcnn/agent/m1_panda_coordinated_train_cfg.py
Go2Pvcnn/agent/m1_panda_folded_load_train_cfg.py
Go2Pvcnn/rsl_rl/
```

## 测试顺序

实施严格遵循 TDD：每个生产行为先有能因缺失功能正确失败的测试。

1. 8D 顺序、物理缩放、幅值/slew、选择性 reset、finite 原子性；
2. wrench filter、bias warm-up、反馈符号、最终 clip 和零增益；
3. height/stance 映射、ABAD 顺序/符号、soft-limit clip；
4. 连续 `alpha_base` 边界、中间值、不可行 fallback 和旧结果兼容；
5. residual 应用前后的 WBC input 不变性与零残差 problem 等价；
6. 手工六维 wrench 对 QP generalized force、contact force 和 effort 的有限响应；
7. safety SCALE/zero projection、非有限拒绝和 reset 隔离；
8. observation 分组、flatten 顺序和 previous residual；
9. wrapper 8D→WBC→23D 合同、旧 Gym ID/23D wrapper 回归；
10. 静态入口、CPU reference probe、GPU0 短 smoke。

## 验收门

纯/静态门：

- 新增 focused tests 全通过；
- 现有 motion distribution、standing/rolling WBC、QP、safety、C0/C1a 测试全通过；
- 现有 coordinated/folded-load action shape 和注册测试全通过；
- py_compile 和 diff check exit `0`。

CPU reference 门：

- 零 residual 连续执行具有确定性；
- 新旧 WBC problem 在零 residual/零反馈时数值等价；
- 正负六轴手工 residual 均产生有限且方向一致的 generalized correction；
- QP 成功、effort/contact force finite，输入对象未被原地修改。

GPU0 第一版 smoke：

- 1 env，先完成 warm-up，再运行零 residual 和逐轴小幅 residual；
- 8D 输入、filtered wrench、WBC effort 和 diagnostics 全部 finite；
- QP feasibility `1.0`；
- 四轮持续接触，无 base contact、bad orientation、joint-limit violation、unexpected reset；
- 零 residual 段不新增 Panda snap，EE tracking 不劣于对应旧 WBC 短基线；
- 退出原因是 `steps_complete`。

## 完成定义

第一版只有在代码、测试、CPU reference 和 GPU0 短 smoke 均通过，并且 notes/log 记录命令、退出码、commit、GPU 与关键指标后才完成。完成只代表 8D residual WBC 基础接线成立；Arm MPC、PPO multi-branch actor、正式训练、抓取和实机部署必须另行设计和批准。
