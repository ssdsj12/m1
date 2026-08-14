# M1 + Panda 受限残差动作组合器设计

## 1. 目标与范围

本阶段为 M1 + Panda 六轴力感知 Teacher–Student 系统建立一个独立、可复用的残差动作安全边界。组合器接收冻结基础策略产生的 16 维 M1 动作和未来 Teacher/Student 产生的 16 维归一化残差，将残差转换为物理单位、执行幅值与每周期变化率限制，再转换回现有 M1 动作空间并与基础动作相加。

本阶段不加载基础策略 checkpoint，不新增 Teacher 或 Student 网络，不接入 Isaac Lab `ActionTerm`，不实现 Panda IK/OSC、抓取课程或实机安全状态机。

## 2. 选定方案

采用独立、状态化的纯 PyTorch 组合器。它不持有 Isaac Lab 环境对象，训练 wrapper、评估程序和未来实机适配层均可复用同一实现。

未选择的方案：

- 自定义 Isaac Lab `ActionTerm` 会把基础动作与残差动作耦合进环境动作契约，不利于独立测试和实机复用。
- 直接修改现有 RSL-RL wrapper 会把安全边界绑定到单一训练栈，并增加 Teacher、Student 和部署端出现多套实现的风险。

## 3. 文件边界

- `Go2Pvcnn/go2_pvcnn/tasks/m1_residual_action.py`
  - `M1ResidualActionComposerCfg`
  - `M1ResidualActionComposer`
  - 只读诊断属性
- `Go2Pvcnn/tests/test_m1_residual_action.py`
  - 纯 CPU 单元测试，不依赖 Isaac Sim 或 Isaac Lab

不修改现有 M1/Panda 环境动作配置。本阶段只建立后续控制和训练阶段要消费的组合器接口。

构造接口固定为 `M1ResidualActionComposer(cfg, num_envs, device, dtype=torch.float32)`。`num_envs` 必须为正整数，`dtype` 必须是浮点 dtype；内部状态在构造时一次性建立在指定 device 和 dtype 上。

## 4. 动作契约

`base_action` 和 `normalized_residual` 均为形状 `[num_envs, 16]` 的浮点张量。通道顺序固定为：

1. 前 12 维：`M1_LEG_JOINT_NAMES` 对应的腿关节位置动作；
2. 后 4 维：`M1_WHEEL_JOINT_NAMES` 对应的轮关节速度动作。

归一化残差由未来网络输出。组合器先将其逐元素裁剪到 `[-1, 1]`，再映射为物理残差：

- 腿部为关节位置残差，单位 `rad`；
- 轮部为关节速度残差，单位 `rad/s`。

物理残差经过逐环境、逐关节的每周期变化率限制。随后腿部残差除以 `leg_action_scale`，轮部残差除以 `wheel_action_scale`，得到现有 M1 action 空间中的增量，并与 `base_action` 相加。

组合器只限制残差，不裁剪或改写基础策略动作。最终组合动作的整体裁剪仍由现有环境或 wrapper 负责。

## 5. 配置与首版默认值

首版配置默认使用现有 M1 + Panda smoke action 比例和已批准的仿真初始边界：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `leg_action_scale` | `0.25` | 归一化腿动作到关节位置的比例 |
| `wheel_action_scale` | `8.0` | 归一化轮动作到关节速度的比例 |
| `leg_residual_limit_rad` | `0.05` | 腿部残差绝对幅值上限 |
| `wheel_residual_limit_rad_s` | `1.0` | 轮部残差绝对幅值上限 |
| `leg_slew_limit_rad_per_step` | `0.01` | 腿部每 20 ms 控制周期的最大残差变化 |
| `wheel_slew_limit_rad_s_per_step` | `0.2` | 轮部每 20 ms 控制周期的最大残差变化 |

这些数值是仿真初始值，不构成实机安全认证。所有值保持显式可配置；机械验算和训练评估可在后续阶段收紧或调整。

两个 action scale 必须为正数，四个物理限幅参数必须为非负数。非法配置在组合器创建时立即失败。

## 6. 状态与 reset

组合器为每个环境保存上一次经过变化率限制后的 16 维物理残差。每个环境独立更新，避免 batch 内相互污染。

- `reset()` 清零全部环境的历史残差和诊断掩码。
- `reset(env_ids)` 只清零指定环境；`env_ids` 接受一维整数张量或整数序列，不接受布尔掩码。
- 环境索引越界时调用失败，且任何状态均不改变。

当前步计算保留从组合动作到 `normalized_residual` 的梯度。写入跨步历史状态时执行 `detach()`，避免在长期 rollout 中累积跨控制周期计算图。

## 7. 异常与原子性

每次 `compose()` 在更新历史状态前完成全部校验。以下情况抛出包含具体原因的异常：

- 输入形状不是 `[num_envs, 16]`；
- 两个输入不在同一 device；
- 两个输入不是相同的浮点 dtype；
- 任一输入包含 `NaN` 或 `Inf`；
- 输入 device 或 dtype 与组合器状态不一致。

异常调用不得改变历史残差或诊断掩码。组合器不静默迁移 device，不静默转换 dtype，也不把非有限值替换为零。传感器故障降级和实机 fail-safe 属于后续安全适配层。

## 8. 诊断接口

组合器公开以下只读状态，供后续奖励、训练日志和饱和率统计使用：

- 当前经过变化率限制的物理残差；
- 本周期归一化输入发生幅值裁剪的逐元素布尔掩码；
- 本周期物理残差发生变化率裁剪的逐元素布尔掩码。

公开属性一律返回内部张量的克隆，调用方原地修改返回值不得改变组合器状态。

## 9. 数据流

```text
base_action[16]
normalized_residual[16]
        |
        v
validate shape/device/dtype/finiteness
        |
        v
clip normalized residual to [-1, 1]
        |
        v
map to physical leg rad / wheel rad/s
        |
        v
per-env slew limit against detached history
        |
        +--> update diagnostics and detached history
        v
divide by existing M1 action scales
        |
        v
base_action + residual_action_delta --> combined_action[16]
```

## 10. 测试与验收

测试驱动实现必须覆盖：

- 初始或 reset 后的零残差保持基础动作不变；
- 12 个腿通道和 4 个轮通道的单位映射正确；
- 超出 `[-1, 1]` 的输入被裁剪，物理幅值不超过配置上限；
- 连续多步的正向和负向变化均受每周期变化率限制；
- batch 中指定环境 reset 后归零，其他环境保持原历史；
- 幅值裁剪和变化率裁剪诊断掩码正确；
- 非法配置、shape、dtype、device、非有限值和 reset 索引给出明确错误；
- 异常调用不改变内部状态；
- 当前步输出可以反向传播，保存的历史状态不保留计算图；
- 调用方修改读取到的诊断张量不会改变内部状态。

实现验收要求：

1. `Go2Pvcnn/tests/test_m1_residual_action.py` 全部通过；
2. 现有 M1/Panda asset、smoke cfg 和 wrench foundation 静态回归测试通过；
3. Python 编译检查通过；
4. 代码、测试、T400 branch memory 和验证日志保持一致。

## 11. 后续消费者

本组合器完成后，后续阶段按独立计划接入：

1. Teacher 随机六维扰动平衡基线；
2. Student 时序估计器与蒸馏训练；
3. Panda IK/OSC 和静止抓取课程；
4. 实机六轴传感器适配与安全状态机。

T400.3 最坏工况机械验算仍是最大载荷实机测试的前置门，但不阻止本纯软件组合器的实现与仿真单元测试。

## 12. 自检结论

- 占位符：无。
- 内部一致性：输入维度、通道顺序、单位、默认比例、状态和测试契约一致。
- 范围：仅覆盖独立残差组合器，不混入 Teacher、Student、IK/OSC 或实机部署。
- 歧义处理：组合器不裁剪基础动作；变化率按控制周期而非每秒定义；历史状态保存的是变化率限制后的物理残差；诊断属性返回克隆；reset 索引只接受整数序列或一维整数张量。
- Git Ref: unavailable（`/home/xk/coding/M1` 不是 Git 工作树）。
