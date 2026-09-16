# T500 GPU RTI共同模型Task1验证

## 范围

本轮完成focused计划Task1：在不改变公开`[B,25,43]`动作合同的前提下，新增私有`[tau43,w_left6,w_right6]`共同LQ工作区。机器人使用`-J_left.T w_left-J_right.T w_right`反作用，箱体接受正向wrench；每个40ms节点包含2个20ms机器人/机械臂检查、4个10ms O6检查和1个40ms箱体步进。

内部优化状态为110D，轨迹Q作用于`x1..x25`，终端P仅叠加于`x25`，掌心残差同样索引`x1..x25`。完整H25控制方向采用1375x1375 dense GN正确性参考实现；没有启用GPU Play，也没有声称实时性能或Isaac物理验收。

## 审查修复

- 首轮审查发现有限输入乘积可能负向溢出为`-inf`并通过`<= upper`；现对wrench与generic hard两类LHS先做逐行有限性拒绝。
- 复审发现动态约束数可超过固定finiteness scratch容量并触发CUDA隐式扩容；现按`max(25*12*110,25*C*12,25*K*55)`预分配，独立覆盖`C=111,K=0`、`C=0,K=25`和空约束。
- 最终独立审查：Spec compliant，Critical/Important/Minor均为None。

## 父级新鲜验证

在`ce6942c`上运行真实GPU0：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/Go2Pvcnn" /home/xk/miniconda3/envs/go2/bin/python \
  -m pytest -q -s Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_lq.py
```

结果：`25 passed in 68.55s`；`111`个存储地址稳定，`allocated_delta=0`，`reserved_delta=0`。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/Go2Pvcnn" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q -s \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_state.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_dynamics.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_warm_start.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py \
  Go2Pvcnn/tests/test_m1_bimanual_reduced_dynamics.py \
  Go2Pvcnn/tests/test_m1_bimanual_full_action_teacher.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py
```

结果：`82 passed in 2.36s`；既有state/dynamics/warm-start内存诊断仍为固定地址和零显存增量。

## 边界与下一步

生产指尖prior/E0仍未合格，GPU Play继续fail-closed。下一步只实施Task2四条完整时域候选的并行line search：固定alpha`(1,.5,.25,.125)`、最低索引平局胜出、任一节点/侧失败整行原子拒绝；last-safe发布留给Task3。
