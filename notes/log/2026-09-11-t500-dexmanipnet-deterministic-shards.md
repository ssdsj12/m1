# T500 DexManipNet Deterministic Fingertip Shards

## Purpose

把 Task 4 已审计的 `LoadedHandSequence` 经固定源手 FK 转成右手规范掌坐标五指尖窗口，并以可复现、
原子、SHA 固定的离线 shard 形式保存，供后续 Task 6 expert ensemble 消费。

## Stage

T500.5 / Task 5；只涉及离线 geometry/provenance preprocessing、storage 和 conversion CLI，不训练模型、
不导入 Isaac，也不修改 Hand MPC 或运行时行为。

## Related Todo

[T500.5 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。

## TDD Evidence

从 `Go2Pvcnn` 执行初始 focused RED：

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_preprocess.py \
  tests/test_m1_bimanual_expert_prior_storage.py
```

两个 production module 尚不存在时按预期 collection error：`2 errors in 0.85s`，均为
`ModuleNotFoundError`。后续 manifest SHA hardening 的 focused RED 为 `1 failed, 6 deselected`：旧 CLI
错误接受 `not-a-sha` archive provenance；最小修正后同一用例通过。

最终 focused GREEN：

```text
21 passed in 5.95s
```

覆盖精确左手反射往返、CubicSpline 解析导数、滞回/七阶段、20 个严格 future 节点、缺失/不安全/
不可用 object collision geometry 原子拒绝、group-exclusive split、shape/dtype/finiteness、固定 ZIP
metadata NPZ、audit JSONL、排序 aggregate manifest，以及两次完整合成 CLI 转换逐文件 SHA 一致。

## Task 1–5 Regression And Static Gates

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_contracts.py \
  tests/test_m1_bimanual_expert_prior_download.py \
  tests/test_m1_bimanual_expert_prior_urdf_fk.py \
  tests/test_m1_bimanual_expert_prior_dexmanipnet.py \
  tests/test_m1_bimanual_expert_prior_preprocess.py \
  tests/test_m1_bimanual_expert_prior_storage.py
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_convert_dexmanipnet.py --help
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/preprocess.py \
  go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/storage.py \
  scripts/m1_dual_panda_o6_convert_dexmanipnet.py \
  tests/test_m1_bimanual_expert_prior_preprocess.py \
  tests/test_m1_bimanual_expert_prior_storage.py
git -C .. diff --check
```

结果：Task 1–5 `101 passed in 7.02s`；CLI help exit `0` 且针对不存在的 root 的测试确认不创建目录；
pycompile 和 diff check 均 exit `0`。

## Implemented Contract

- 源手 FK 输出始终是 palm-relative thumb/index/middle/ring/pinky；右手不变，左手只施加冻结的
  `diag(1,-1,1)`，不交换手指。
- 源时间固定按 60 Hz，目标时间只取原区间内的 100 Hz 节点；位置和速度分别来自同一个
  `scipy.interpolate.CubicSpline` 及其解析一阶导数。
- contact 不读取 `tip_force`、primitive、对象名称或任务字段；只从对象 collision mesh 的 signed
  distance 和 object-relative normal speed 进入双阈值滞回。缺失、越界链接、非 watertight 或查询失败
  的几何会拒绝整条侧轨迹，绝不补 `contact=false`。
- phase 仅由五指 contact 建立/增加/保持/消失和 palm-relative fingertip speed 推断为七个冻结枚举。
- 每个 `ExpertWindow` 保存当前 `(5,3)` 位置/速度、`(5,)` contact、phase 和后续严格 20 个
  `(5,3)` 速度节点；窗口不跨 sequence。
- group split 对 sorted unique group 使用固定 seed 的 80/10/10 分配；同一 group 的窗口只能进入一个
  split，且同一 group 的 source SHA 不得冲突。
- shard 使用排序字段、固定 NPY dtype/order、固定 ZIP timestamp/permission/compression 写 sibling temp
  后 rename；输出目录整体 staged 后原子安装。每个 shard 记录 SHA、source/hand/phase/contact-pattern
  分布；aggregate 先按相对 shard path 排序再计算 SHA，并固定 archive/source manifest SHA。
- audit JSONL canonical-sort 后写入；CLI 只消费本地 pinned manifest/source/extraction，`--help` 不检查、
  写入或下载数据。

## Input Conditions And Limits

- Baseline Ref: `a40590ed240a081c0499e22700568e2a48666122`
- Candidate Ref: `ef60ae1489be0cb314bbbdcefd0d1b848fea4a47`
- 所有测试均使用 synthetic URDF/HDF5/box collision fixtures；未下载或扫描 8.31 GB production archive。
- 尚未验证真实 FAVOR/OakInk V2 object mesh 的 watertight 接受率、完整转换数量/phase/contact 分布或
  production shard SHA；这些属于完整数据 gate，不得从本次 synthetic evidence 推断。
- 本任务未训练 ensemble/student，未运行 Isaac/GPU0/物理仿真，也没有 Hand MPC 行为或成功率结论。

## Conclusion And Follow-up

Task 5 离线 deterministic shard 合同及 Tasks 1–5 CPU 回归通过。下一步为 Task 6：只从 aggregate
manifest 和 hash-verified shards 训练并 gate offline expert ensemble。

## Git Refs

- Last Feature Commit: `ef60ae1` (`feat: convert DexManipNet to fingertip prior shards`)
- Last Verified Commit: `ef60ae1`
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`
- Key Files: `preprocess.py`, `storage.py`, conversion CLI, and their two focused test modules.
