# T500 GPU RTI共同模型 Gate C 验证

## 完成范围

Gate C 在 Gate A（共同动力学/LQ）与 Gate B（四alpha完整时域line search）之上新增 eager planner。固定顺序为 warm shift → assemble → nominal rollout/validation → linearize → 一次 direction solve → 一次 line search → 111D reporting staging → 原子 warm publication。

内部优化状态为110D，公开/warm/action结果严格为`[B,25,43]`。报告状态按现有111D布局保存：base position/quaternion/twist、M1/双臂/双手/箱体；base quaternion由`q0⊗Exp(local SO(3) tangent)`重建并做归一化/同半球约束。wrench维度永不进入公开动作或last-safe存储。

失败路径逐row保持：identity/reset或测量失效会清除资格；无历史安全轨迹时输出全零、`accepted=False`、`safe_available=False`、`REASON_NO_SAFE_HORIZON`；有历史时返回完整shifted last-safe，`accepted=False`，而不是伪装成新接受。新增复审修复覆盖 nominal有效但四条alpha都违反hard约束的真实不可行场景，并验证无prior与有prior两种结果。

`bimanual-rti-cuda`仍由配置层在设备/场景创建前明确拒绝；本门没有启用GPU Play，也没有加入prior、scan/Triton、CUDA Graph、runtime、latent、benchmark、Isaac、训练或感知代码。

## 独立复审

首轮复审发现 Important：原“no-safe”用例仅是`input_valid=False`，没有真实四候选hard-infeasible。`aff491f`仅改planner测试补齐该场景；累计复审最终 Spec compliant / Approved，Critical和Important为None。剩余 Minor（主动映射逐组distinct值、完整调用顺序日志）不阻塞本门。

## 父级新鲜验证

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/Go2Pvcnn" \
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q -s \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_planner.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py
```

结果：`42 passed in83.84s`，planner存储`205`个地址稳定，allocated/reserved增量0。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/Go2Pvcnn" \
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q -s \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_state.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_dynamics.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_warm_start.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_lq.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_line_search.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_planner.py \
  Go2Pvcnn/tests/test_m1_bimanual_reduced_dynamics.py \
  Go2Pvcnn/tests/test_m1_bimanual_full_action_teacher.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py
```

结果：`123 passed in209.84s`，state/dynamics/warm/LQ/line-search/planner全部内存诊断稳定，无warning/skip。

## 明确未完成

这只是 prior-off eager correctness backend。仍未完成：GPU Play启用、scan/Triton、CUDA Graph、latent运行时、benchmark manifest、真实 Isaac GPU0 smoke、正式30条物理验收、DexManipNet双来源E0/student资格。当前可继续的下一阶段应从 runtime/benchmark 计划中另立 Gate，避免把本门误称为可直接Play。
