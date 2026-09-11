# T500 DexManipNet 指尖先验合同冻结

## Purpose

冻结 DexManipNet/ManipTrans 来源 pin、五指掌坐标样本、四分量概率输出和学生 artifact metadata 的无运行时数据依赖合同。

## Stage

T500.5 Task 1：离线预处理与 artifact 的基础合同；不下载数据、不解析 URDF、不训练且不接入 Hand MPC。

## Related Todo

- [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)

## Procedure

### RED

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py
```

在 package 尚不存在时，collection 按预期失败：`ModuleNotFoundError: No module named 'go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior'`；`1 error in 0.80s`。

### GREEN / current-contract regression

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py tests/test_m1_dual_panda_o6_contracts.py
```

结果：`16 passed in 0.78s`。

### Contract review hardening

The Task 1 review requested frozen artifact layout metadata, strict schema types, tuple-only source fields, exact registry assertions, and immutable public constants.

RED command:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py
```

The new negative tests failed at collection as intended because
`MIXTURE_OUTPUT_AXIS_ORDER` was absent: `ImportError: cannot import name 'MIXTURE_OUTPUT_AXIS_ORDER'`; `1 error in 0.81s`.

GREEN / current-contract regression:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py tests/test_m1_dual_panda_o6_contracts.py
```

Result: `30 passed in 0.79s`.

The fix records the exact four-field geometry-only input order and the four output axes in `StudentArtifactMetadata`; rejects bool/float impostors for integer schema fields, non-tuples and non-strict hidden widths, malformed SHA/dtype/nonfinite values, and source string/list impostors. `LEFT_REFLECTION` is now a tuple-of-tuples and `SOURCE_HANDS` a `MappingProxyType`.

### Second review: float32 mixture dtype and source-member negatives

RED command:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py
```

The new dtype cases were red as intended: float16, bfloat16, and float64 distributions each failed because `MixtureDistribution` did not raise `TypeError`; result `3 failed, 17 passed in 0.80s`.

GREEN / current-contract regression:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py tests/test_m1_dual_panda_o6_contracts.py
```

Result: `33 passed in 0.82s`.

`MixtureDistribution` now requires exact `torch.float32` logits, means, and log standard deviations. Added source-spec negatives reject blank and non-string tuple members and duplicate joint/tip entries. No deep read-only tensor wrapping was added: the approved Task 1 contract specifies frozen dataclasses and construction-time validation, while PyTorch has no standard durable read-only tensor interface; clone/property wrapping would change the public tensor API without a design requirement.

### Pinned source-registry audit

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD MANIPTRANS_ROOT=/tmp/maniptrans-contracts /home/xk/miniconda3/envs/go2/bin/python -c '<parse each registered pinned URDF; assert its palm, five tips, and joint order exist>'
```

ManipTrans checkout `a3d08cfe3c3a5868a7f057533bcaf759c5af4705` produced
`verified 4 pinned source-hand specs`. This audit is test-only; the package retains no source checkout or external-data runtime dependency.

## Input Conditions

- Baseline Ref: `b109b684a32ca3b5400fcdfe5aa094aa82c44c74`
- Candidate Ref: `0d5b82124ca1b0a36b257760d88ecd6e7050438c`
- Key Files:
  - [contracts.py](../../Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/contracts.py)
  - [sources.py](../../Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/sources.py)
  - [contract tests](../../Go2Pvcnn/tests/test_m1_bimanual_expert_prior_contracts.py)

## Result

- 固定 DexManipNet revision `3933fae5fe83498fb314a0924aa21d5038fba5a5` 与 ManipTrans commit `a3d08cfe3c3a5868a7f057533bcaf759c5af4705`。
- `ExpertWindow` 只打包 `15` 位置、`15` 速度、`5` 接触和 `7` 阶段 one-hot，网络输入严格为 `(42,)`，目标为 `(20,5,3)`。
- `MixtureDistribution` 固定为四分量、20 节点、五指、xyz 的对角高斯张量；窗口、分布、metadata 和 source-hand spec 在构造时拒绝不兼容 shape、dtype、有限性、enum、SHA、维度或重复名称。
- `SOURCE_HANDS` 显式列出 pinned ManipTrans 的 Inspire/Shadow 左右手；其 URDF 相对路径、关节顺序、掌 link 和 thumb/index/middle/ring/pinky tip links 直接抄自该 commit 的 dexhand definitions 和引用 URDF。

## Conclusion

基础合同与现有 T500 dual-Panda contract suite 均通过。下一步是 Task 2 的固定下载、安全解压与 manifest；本任务没有任何数据、模型或 Isaac 物理结论。

## Follow-up

Task 2 只能消费这些固定 pin 和 registry，且 source URDF 的本地 clone 必须固定在上述 ManipTrans commit。

## Git Refs

- Last Feature Commit: `0d5b82124ca1b0a36b257760d88ecd6e7050438c`
- Last Verified Commit: `0d5b82124ca1b0a36b257760d88ecd6e7050438c` (verification was run immediately before the feature commit)
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`
