# M1 双 Panda + O6 四层验证

本阶段只验证已有控制器，不创建训练入口，不新增接触 MPC、感知或 Residual，
也不修改 T400/T500 公共合同。

所有命令均在 `Go2Pvcnn` 根目录、M1 的 `go2` Conda 环境中执行。

## 1. CPU 单元测试

覆盖确定性、轨迹连续性、shape/dtype/device、mimic、fallback 与原子拒绝：

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer cpu
```

## 2. 纯 QP 回归

覆盖 Object/Arm/Hand/WBC 可行性、限位、重复求解一致性，以及错误输入不污染最近安全状态：

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer qp
```

## 3. Isaac GPU0 smoke

单条固定 seed=7 样本，显式固定 `cuda:0`，检查确定性物理复位、坐标、工作台、
最小间距、控制层可行率和控制方向：

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer smoke --smoke-steps 200
```

结果写入 `tests/artifacts/m1_dual_panda_o6_verification/gpu0_smoke.json`。

## 4. 正式 GPU0 验收

固定 seeds 42/43/44，每个 seed 10 条，共 30 条。只有 30 条全部满足硬门，
aggregate manifest 的 `status` 才会是 `passed`：

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer formal --formal-steps 2000
```

输出目录为 `tests/artifacts/m1_dual_panda_o6_verification/`：

- `formal_trials.jsonl`：逐条 trial，一行一个 JSON 对象；
- `formal_report.json`：完整聚合报告；
- `formal_aggregate.manifest.json`：固定 JSONL、report、USD、源码和 Git ref 的 SHA/版本信息。

任意一条失败时命令返回非零状态，但仍会落盘全部诊断与 `failed` manifest，不能将
smoke 通过替代为正式验收通过。
