# T500 GPU RTI共同模型Task2验证

## 范围

本轮新增私有`ParallelLineSearch`。它从Task1的nominal与完整H25×55方向生成固定alpha`(1.0,0.5,0.25,0.125)`四条候选，对每个控制坐标独立clamp，并恢复等上下界元素；不投影wrench或generic hard不等式。

四条候选均执行完整rollout和hard validation。merit与Task1 GN软目标一致：`x1..x25`状态跟踪、`x25`附加terminal、25节点控制代价、`x1..x25`掌心残差。普通行仅接受严格改善且最低索引胜出；等上下界行先验证nominal后以index0旁路。任何节点/侧失败均拒绝整条候选；拒绝输出完整nominal、`accepted=False`、`alpha_index=-1`、`merit=+inf`。

last-safe发布与公开43D边界不属于本任务，留给Task3。没有启用GPU Play、训练、Isaac或性能后端。

## 审查

独立任务审查结论：Spec compliant；Critical/Important/Minor均为None。审查确认精确merit、strict improvement、equality bypass、非有限值fail-closed、最低索引tie-break、整行原子staging和稳态persistent CUDA storage。

## 父级新鲜验证

在`f4c1105`上运行真实GPU0：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/Go2Pvcnn" \
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q -s \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_line_search.py
```

结果：`3 passed in 59.90s`；38个line-search存储地址稳定，`allocated_delta=0`，`reserved_delta=0`。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/Go2Pvcnn" \
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q -s \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_lq.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_line_search.py
```

第一次运行在日期切换后丢失会话句柄，没有作为证据。重新完整运行结果：`28 passed in 127.11s`；LQ 111个与line-search 38个存储地址稳定，allocated/reserved增量均为0。

## 下一步

Task3将按固定顺序编排Task3基础state/dynamics/warm-start、Task1 LQ与Task2 line search，只发布43D effort horizon，并按identity/reset维护最近接受的完整last-safe轨迹。生产E0仍未合格，GPU Play继续fail-closed。
