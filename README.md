# M1 + 双 Panda + 双 O6 分层双手 MPC

本分支面向 M1 轮足底盘、公共单轴回转平台、左右两条 Franka Panda 机械臂和
左右 Linker Hand O6 灵巧手的双手协同操作。当前目标是在 Isaac Sim / Isaac Lab
中完成固定 `0.5 kg` 箱体的接近、预加载、抓取、举升、保持、下降和释放，并为
后续 MPC/Teacher 数据训练提供可恢复、可审计的运行入口。

> 当前状态：控制与训练基础设施已经实现并通过专项静态/纯控制测试，但尚未通过
> 跨 seed 真实物理门和正式 `30/30` 完整任务验收。因此本分支仍属于开发与验证阶段。

## 本次改进

### 1. 统一 SO(3) 姿态语义

- 使用 SO(3) 插值生成掌面姿态轨迹。
- 使用 `Log(R_target R_measured.T)` 计算空间姿态误差。
- 从相邻旋转的对数映射计算角速度前馈，使其与空间几何 Jacobian 保持一致。
- 保留原有公开 dataclass、43 通道动作顺序和 25/50/100/200 Hz 分层调度合同。

主要实现位于
[`Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/`](Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/)。

### 2. 改进双手接触判断

- 新增统一接触摘要，同时保留 raw contact 与 Box-filtered contact 诊断。
- Box 接触以 filtered force 为准，避免把环境碰撞误判成抓箱接触。
- 采用一个传感器对应一个手指 body 的 Isaac Lab 兼容结构，左右手共配置 18 个
  Box-filtered contact sensor。
- 状态机可基于稳定、可追踪的接触条件进入预加载、抓取和举升阶段。

### 3. 增加有效举升及其他动作基础

- 新增举升相关 motion primitives 和向量控制入口。
- 改进 approach、preload、grasp、lift、hold、lower、release 及安全释放阶段推进。
- 在 Arm MPC、Object MPC、Whole-body QP 和 Isaac Lab wrapper 之间补齐状态、目标、
  可行性与安全回退传递。
- Probe 增加阶段、接触、首节点姿态纠偏和安全事件诊断。

### 4. 固定总仿真步数与多 GPU 运行

- [`m1_dual_panda_o6_10k_sweep.py`](Go2Pvcnn/scripts/m1_dual_panda_o6_10k_sweep.py)：
  10,000 步受控候选/跨 seed 验证入口。
- [`m1_dual_panda_o6_40k_runner.py`](Go2Pvcnn/scripts/m1_dual_panda_o6_40k_runner.py)：
  可恢复的 40,000 env-step 账本和多 GPU worker 调度。
- 运行器记录 reservation、实际消耗、阶段预算和恢复状态，防止重复计步或超预算。
- GPU 由启动参数和空闲资源检查决定，不要求永久绑定固定卡号。

### 5. Teacher 数据链

- 补充采集、准备和合并脚本。
- Teacher 数据可记录状态、任务特征、43 维动作、MPC 可行性和安全事件。
- 训练产物、缓存和本机环境不纳入 Git；GitHub 仅保存源码、测试、文档和受 Git LFS
  管理的资产。

## 已验证内容

当前功能提交为 `265fbcf`，验证记录提交为 `7c6701d`。

```bash
conda activate go2
cd Go2Pvcnn
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_m1_bimanual_*.py tests/test_m1_dual_panda_o6_*.py
```

最近一次结果：

- T500 专项测试：`224 passed`
- 接触与环境针对性回归：`12 passed`
- 改动 Python 模块编译：通过
- `git diff --check`：通过
- 提交内容敏感信息扫描：无匹配

详细证据见
[`notes/log/2026-09-11-t500-github-upload-verification.md`](notes/log/2026-09-11-t500-github-upload-verification.md)。

## 常用入口

```bash
conda activate go2
cd Go2Pvcnn

# 查看物理诊断参数
PYTHONPATH="$PWD" python \
  scripts/m1_dual_panda_o6_bimanual_probe.py --help

# 查看演示/Play 参数
PYTHONPATH="$PWD" python \
  scripts/m1_dual_panda_o6_bimanual_play.py --help

# 查看固定步数运行器参数
PYTHONPATH="$PWD" python \
  scripts/m1_dual_panda_o6_40k_runner.py --help
```

运行 Isaac Sim 前仍需确保当前用户拥有 NVIDIA Vulkan/DRM 设备访问权限，且
`vulkaninfo --summary` 能枚举目标 GPU。

## 尚未完成

- [ ] 在可用 Vulkan/DRM 环境中重新执行真实 Isaac Sim smoke。
- [ ] 完成 seeds `42/43/44` 的跨 seed 物理门，并核对固定总步数账本。
- [ ] 验证完整 `approach → preload → grasp → lift 0.10 m → hold 3 s → lower → release`
  周期无非有限值、硬限位、意外 reset 或安全拒绝。
- [ ] 完成三个 seed、每个 10 次的正式 `30/30` 固定条件验收。
- [ ] 用正式采集数据完成 Teacher 数据合并、离线模型训练、模型加载和无仿真前向验证。
- [ ] 在物理门通过后再评估是否需要跟踪感知的首节点约束；当前不能仅凭专项测试宣称
  MPC 举升已经准确或完全收敛。
- [ ] 配置并验证 WebRTC 端口转发及本地画面；当前服务器的 Vulkan/DRM 权限属于宿主机
  基础设施阻塞，不是 Play 参数能够绕过的问题。

## 文档索引

- [T500 分支状态](notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md)
- [SO(3) 与 10k 多 GPU 设计](docs/superpowers/specs/2026-09-08-t500-so3-10k-multigpu-design.md)
- [SO(3) 与 10k 多 GPU 实施计划](docs/superpowers/plans/2026-09-08-t500-so3-10k-multigpu.md)
- [右掌姿态边界诊断](notes/log/2026-09-06-right-palm-orientation-boundary-diagnosis.md)
- [O6 原始资产迁移记录](notes/log/2026-09-08-o6-source-relocation.md)
- [验证日志索引](notes/log/index.md)
