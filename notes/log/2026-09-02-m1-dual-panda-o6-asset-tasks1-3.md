# M1 + 双 Panda + 双 O6 组合资产 Tasks 1–3

## Outcome

T500 Inline Execution 的资产阶段通过。项目内现已包含左右 O6 的规范化输入、M1 公共 yaw 平台、两条 arm-only Panda、左右 O6，以及单 articulation 组合 USD。重复的 Panda/O6 rigid-body 与 joint basename 均使用 `left_` / `right_` 前缀隔离。

## Verification

- Task 1 commit：`ac95a9b`；Task 2 commit：`b58a9a2`；Task 3 commit：`a17912d`。
- Isaac runtime：53 physical DOF、43 active controls、2000 physics steps。
- 安装漂移：最大位置 `2.3841863594498136e-07 m`，最大姿态 `4.814712530792817e-07 rad`。
- `nonfinite_count=0`、`hard_joint_limit_count=0`、`unexpected_contact_count=0`、`unexpected_reset_count=0`、`base_instability_count=0`。
- ContactSensor 数据可用，离线依赖错误为空。
- 相关静态/控制回归：`95 passed`；py_compile、`git diff --check` 和资产绝对路径扫描通过。

## Next

Task 4 冻结 43 通道主动关节顺序和运行时名称映射，之后进入原子双手状态/命令合同。
