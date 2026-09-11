# T500 DexManipNet Sequence Audit And Rollout Selection

## Purpose

为 T500.5 离线转换建立严格 DexManipNet sequence 边界：只接受完整、有限、长度一致且与固定
`SOURCE_HANDS` 精确匹配的几何输入，并确定性选择累计 reward 最高的 successful rollout。

## Stage

T500.5 / Task 4；仅涉及离线 HDF5/JSON 审计与几何数据加载，不改变 Hand MPC、Isaac 环境、训练或
运行时依赖。

## Related Todo

[T500.5 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。

## RED

从 `Go2Pvcnn` 执行：

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_dexmanipnet.py
```

production module 缺失时按预期 collection error：
`ModuleNotFoundError: ... expert_fingertip_prior.dexmanipnet`，`1 error in 0.89s`，exit `2`。

自审发现 rejected audit 未保留输入指纹后，新增 focused RED：

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_dexmanipnet.py -k length_mismatch
```

按预期得到 `1 failed, 15 deselected`：`input_sha256` 长度为 `0` 而不是 `64`。最小修正后同一命令
得到 `1 passed, 15 deselected`。

## Focused GREEN

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_dexmanipnet.py
```

初始结果：`16 passed in 0.85s`。

## Review Hardening RED/GREEN

review 后按缺口分别保持最小 synthetic fixture，并逐项先红后绿：

- source alias：新增 `oakink` 拒绝用例，RED `1 failed, 1 passed, 15 deselected`；移除 alias 后
  GREEN `2 passed, 15 deselected`。公开参数只接受精确 CLI key `favor`、`oakinkv2`。
- `interaction_mode`：导入冻结合同的 RED 在 collection 阶段 exit `2`；冻结
  `lh_main -> {lh}`、`rh_main -> {rh}`、`bh_main -> {lh, rh}` 后 GREEN
  `1 passed, 17 deselected`。其他值为 `invalid_interaction_mode`，请求 side 不在映射集合中为
  `interaction_mode_side_mismatch`。
- HDF5 link：successful group external link、consumed dataset external/soft link 的 RED
  `3 failed, 18 deselected`；改为先检查 link metadata、只解引用 hard link 后 GREEN
  `3 passed, 18 deselected`。
- rejected-input geometry hash：越界 symlink target 改字节会改变 SHA 的 RED
  `1 failed, 21 deselected`；逐 path component `lstat`、只纳入 source root 内非 symlink regular file 后
  GREEN `1 passed, 21 deselected`。
- HDF5 storage：consumed virtual dataset 与 external-raw dataset 的 RED
  `2 failed, 22 deselected`；在 `np.asarray` 前拒绝两种 storage 后 GREEN
  `2 passed, 22 deselected`。

最终 focused suite：`24 passed in 0.89s`。

## Task 1–4 Regression

代码提交前从 `Go2Pvcnn` 执行一次完整相关回归：

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_contracts.py \
  tests/test_m1_bimanual_expert_prior_download.py \
  tests/test_m1_bimanual_expert_prior_urdf_fk.py \
  tests/test_m1_bimanual_expert_prior_dexmanipnet.py
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/dexmanipnet.py \
  tests/test_m1_bimanual_expert_prior_dexmanipnet.py
git -C .. diff --check
```

review hardening 后最终结果：`80 passed in 1.94s`；pycompile 与 diff check 均 exit `0`。

## Input Conditions And Metrics

- 只使用合成最小 HDF5/JSON/URDF fixture；未下载或扫描 8.31 GB production archive。
- pinned ManipTrans checkout `a3d08cfe3c3a5868a7f057533bcaf759c5af4705` 只用于核对真实字段名：
  `q_*`、`dq_*`、`state_*`、`state_manip_obj_*`、`tip_force_*` 和 `reward`。
- `favor` 仅接受 `rh`；`oakinkv2` 接受逐手 `rh`/`lh`，不接受 `oakink` alias；手型必须唯一映射到冻结的
  Inspire/Shadow source-hand key。
- `interaction_mode` 仅接受 OakInk2 schema 的 `lh_main`、`rh_main`、`bh_main`，且必须与请求 side
  分别满足 `{lh}`、`{rh}`、`{lh, rh}` 映射。
- Inspire 与 Shadow 的 joint width 分别严格为 `12` 和 `22`；root/object state 为 `13`，可选
  five-tip force 为 `15`。

## Result

`audit_sequence` 对缺/坏 JSON、无 successful rollout、未知 source/side/hand、错误 joint width、
长度不一致、shape 错误、非有限数组及缺失/越界对象 URDF 返回单个稳定拒绝原因和可用输入 SHA；
不返回任何部分数组。`load_best_successful_rollout` 先审计全部 successful rollouts，再按
`(total_reward, rollout_name)` 最大值选择，因此 reward 相同时选择字典序最大的名字。

HDF5 在任何 dataset 解引用前拒绝 external/soft/其他非-hard link，并在读取 consumed dataset 前拒绝
virtual dataset 与 external raw storage。rejected-input best-effort SHA 不跟随对象路径中的 symlink，故
不会把 source root 外字节计入 provenance。

`LoadedHandSequence` 只包含 `q/dq`、手根与对象 root state、可选 tip force、对象几何路径和必要
provenance。它不含 task/object ID/name、primitive、description 或 instruction text；返回数组均为
只读副本。HDF5 依赖只存在于离线 loader module。

## Conclusion And Follow-up

Task 4 合成 schema 门通过，下一步为 Task 5 的掌坐标指尖转换、规范化、重采样、接触/阶段推断和
确定性 shard。由于 production archive 未下载，本轮没有验证真实 FAVOR/OakInk V2 全量 schema、
对象 URDF/mesh 可用性或实际序列接受率，不能宣称真实数据转换已验收。

## Git Refs

- Baseline Ref: `47c6fbd7018ac8823e542c1da9de79c4c6d61606`
- Candidate Ref: `fdcd40a` (`feat: audit DexManipNet successful rollouts`)
- Review Hardening Ref: `7d21403` (`fix: harden DexManipNet rollout provenance`)
- Key Files: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/dexmanipnet.py` and `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_dexmanipnet.py`
