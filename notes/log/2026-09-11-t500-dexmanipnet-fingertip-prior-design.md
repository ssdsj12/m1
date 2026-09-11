# T500 DexManipNet 指尖先验设计记录

## Purpose

确定 DexManipNet 如何作为 O6 手指自然运动专家数据，同时保持 MPC 对未见物体和未见操作的
任务决策权。

## Stage

T500.5 设计阶段；无运行时代码、训练状态或外部数据变更。

## Related Todo

- [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)

## Procedure

- 核对 Hugging Face 数据集卡、文件列表、容量、revision 和 GPL-3.0 标识。
- 核对官方 ManipTrans loader、HDF5 successful rollout 选择逻辑、`q/dq` 轨迹、60 FPS
  约定、双手训练入口和新手型映射说明。
- 检查本地 Hand MPC、O6 折叠 mimic 指尖 Jacobian、全动作 teacher 与 z16 artifact 合同。
- 与用户逐段确认完整 FAVOR + OakInk V2、掌坐标五指尖几何蒸馏、四分量概率模型和
  可退让 Hand MPC 软代价。

## Input Conditions

- DexManipNet revision `3933fae5fe83498fb314a0924aa21d5038fba5a5`，约 `8.31 GB`。
- ManipTrans commit `a3d08cfe3c3a5868a7f057533bcaf759c5af4705`。
- 本机剩余磁盘约 `476 GB`；`huggingface_hub` 与 `h5py` 可用。
- 当前分支已有 6-active-axis O6 Hand MPC、真实折叠 mimic Jacobian 和 100 Hz 合同。

## Key Decisions

- 不直接复制源手关节角；使用源手 FK 生成掌坐标五指尖位置/速度。
- 运行模型不接收任务 ID、对象类别或掌/物体目标。
- 先由离线大教师 ensemble 学习完整数据分布，再蒸馏为四分量短时学生先验；只有冻结学生
  能进入 MPC，输出 20 节点指尖速度均值/方差。
- Hand MPC 先求 baseline，再选择专家分量并加入可退让二次项；失败时原子使用同周期 baseline。
- 原始数据、缓存和权重不进 Git；Git 保存 pin、SHA、代码、合同和无样本统计。

## Result

用户批准完整设计。书面规格见
[DexManipNet 指尖运动先验蒸馏设计](../../docs/superpowers/specs/2026-09-11-t500-dexmanipnet-fingertip-prior-design.md)。

## Conclusion

设计阶段完成，但尚未下载数据、转换轨迹、训练模型、修改 Hand MPC 或运行验证。下一步是用户
复核书面规格，随后使用 writing-plans 拆分逐文件 TDD 实施计划。

## Follow-up

用户确认书面规格后编写实施计划；计划必须先实现无网络的 schema/FK 测试，再进行完整数据下载。

## Git Refs

- Baseline Ref: `09e16c3`
- Candidate Ref: design commit pending
- Key Files:
  - `docs/superpowers/specs/2026-09-11-t500-dexmanipnet-fingertip-prior-design.md`
  - `notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md`
