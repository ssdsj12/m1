# T500 DexManipNet Isaac-independent URDF FK

## Purpose

为 T500.5 离线转换提供不依赖 Isaac 的源手 URDF forward kinematics：把 DexManipNet 按
`SourceHandSpec.joint_order` 排列的批量关节角转换为固定五指顺序的掌坐标指尖位置。

## Stage

T500.5 / Task 3；只涉及离线 stdlib XML + NumPy URDF 解析与 FK，不改变 Hand MPC、仿真环境或训练运行时。

## Related Todo

[T500.5 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。

## RED

从 `Go2Pvcnn` 执行：

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_urdf_fk.py
```

production module 缺失时按预期 collection error：
`ModuleNotFoundError: ... expert_fingertip_prior.urdf_fk`，`1 error in 0.93s`，exit `2`。

## Focused GREEN

同一命令在实现后得到 `20 passed in 0.83s`。伪造解析 fixture 覆盖：

- 两 revolute joint 的解析解，位置误差门为 `1e-8 m`；
- fixed、continuous、revolute、prismatic、origin xyz/rpy、axis 和四个 limit 字段；
- master multiplier/offset mimic 展开；
- 与 XML 子节点顺序无关的稳定 joint-name topology；
- `SourceHandSpec` 冻结 thumb/index/middle/ring/pinky 顺序下的 `inv(T_palm) @ T_tip`；
- 重名、未知 joint type/link、multiple parent、零/多 root、零轴、非有限 XML/q、mimic cycle、
  缺 link、未知 joint name 和 q 列数不一致的原子拒绝。

## Task 1–3 Regression

代码提交前从 `Go2Pvcnn` 执行一次完整相关回归：

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_contracts.py \
  tests/test_m1_bimanual_expert_prior_download.py \
  tests/test_m1_bimanual_expert_prior_urdf_fk.py
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/urdf_fk.py \
  tests/test_m1_bimanual_expert_prior_urdf_fk.py
git -C .. diff --check
```

结果：`56 passed in 1.98s`；pycompile 与 diff check 均 exit `0`。

## Result

`UrdfKinematicTree.from_file` 只接受单根、无 multiple parent 且完全连通的 link tree；按 joint name
稳定排序生成 topology。`forward_links` 严格校验二维有限 `q`、列数、joint order 与 requested links，
并返回 `(batch, links, 4, 4)`。mimic 递归使用 master、multiplier 和 offset；mimic cycle 在解析期拒绝。
掌坐标接口返回五个相对 transform 或 `(batch, 5, 3)` 位置，手指顺序直接来自 `SourceHandSpec`。

## Conclusion And Follow-up

伪造 URDF 合同与 Task 1–3 回归通过。由于 Task 2 未执行 8.31 GB production fetch，本轮没有对固定
ManipTrans commit 中的真实 Inspire/Shadow URDF 做兼容或数值对照；Task 4/完整数据门必须补上该证据，
当前不能宣称真实数据转换已验收。

## Git Refs

- Baseline Ref: `158d0da84d449d38ab2ec8d51d94850c037476f6`
- Candidate Ref: `27dc96f` (`feat: add source hand URDF forward kinematics`)
- Key Files: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/urdf_fk.py` and `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_urdf_fk.py`
