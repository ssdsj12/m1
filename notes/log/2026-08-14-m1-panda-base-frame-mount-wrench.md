# M1 + Panda Base-Frame Mount Wrench

## Purpose And Scope

完成 T400 foundation Task 5：将 PhysX joint-frame/about-joint-origin incoming wrench 先转换到 world，再平移到 `BASE_LINK` actor origin、旋转到 base frame，并接入未归一化 smoke observation。未加入 noise、Student/Teacher、IK、OSC 或 Task 6。

## Stage And Refs

- Stage: T400 / asset-wrench foundation / Task 5
- Related todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable

## API And Convention

本地 vendor behavior test 证明 raw incoming wrench 是 joint-frame、about joint origin 的 parent-on-child `[N,B,6]` force+torque；IsaacLab high-level “world frame” docstring 与此冲突，不能作为边界依据。正式 USD 的 child `localPos1=0/localRot1=identity`，因此 mount joint frame/origin 与 `panda_link0` actor 重合。adapter 先用 mount quaternion 正向旋转 raw force/torque 到 world；输出顺序固定 `[Fx,Fy,Fz,Mx,My,Mz]`：

```text
M_base_origin_w = M_mount_w + (p_mount_w - p_base_w) x F_w
F_b = R_wb F_w
M_b = R_wb M_base_origin_w
```

不做归一化、裁剪或噪声。adapter 只用 `asset_cfg.name`，显式 mount/base 名分别以 `preserve_order=True` 唯一解析，避免与已有 `body_ids/body_names` 双重语义冲突；Python integer ID 不引入 CPU index tensor。

## TDD Evidence

- Initial RED: `1 failed, 11 passed, 8 errors in 1.79s`, exit `1`；仅因 wrench 模块和 smoke term 缺失。
- First GREEN: `20 passed in 0.92s`, exit `0`。
- Unique-ID falsification RED: 单 name 但重复 IDs，`1 failed, 4 passed in 0.77s`；补 `len(ids)==1` 后通过。
- 覆盖 identity shift、yaw、batch 非零 lever+rotation、incoming force/torque slices、mount/base pose/quaternion、显式 name 优先、missing/duplicate RuntimeError、smoke exact params 和原 12+4 actions。

## Final Verification

统一命令前缀均为 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest`。

- 计划四文件：`25 passed in 0.84s`，exit `0`。
- 加 Task 4 Panda asset focused：`41 passed in 0.86s`，exit `0`。
- 五个相关 Python 文件 `py_compile`：exit `0`。
- loco/source 真实 `quat_rotate_inverse`：exit `0`，+90° base yaw 将 world +Y 转为 base +X。
- loco/source 真实 Isaac math 调生产纯函数：exit `0`，非零 lever+yaw 得约 `[2,0,0,0,0,5]`，shape `(1,6)`。
- loco bootstrap + headless AppLauncher 的公共 `go2_pvcnn.mdp` export、真实 cfg params 与 actions `(12,4)` assertions：exit `0`。

诊断尝试如实保留：未设置 EULA 时 exit `1`；只使用 bootstrap package 时缺 `isaaclab.utils` exit `1`；无 AppLauncher 的完整 manager import 在 `omni.kit` 边界 exit `1`。后续正确 source/AppLauncher 路径均 exit `0`。

## Fix Round 1

- Review 状态：初审 FAIL 后已修复，等待独立复审，不声明 PASS。
- AppLauncher+PXR 正式 joint 证据：body0/base、body1/mount、localPos0 z=`0.1389991939`、localRot0 identity、localPos1 zero、localRot1 identity。
- Adapter RED：旧实现输出约 `[0,-1,0,0,0,2]`，期望 `[1,0,0,0,0,3]`；联合 export/asset contract 为 `4 failed, 22 passed`。
- GREEN：先 `quat_rotate` raw force/torque 到 world，再调用未改 pure helper；builder/verifier/PXR 强制 child zero/identity；export 显式化。
- 最终：计划 `26 passed in 0.81s`，五文件 `42 passed in 0.81s`，Task 2–4 focused `32 passed in 0.04s`，pycompile exit `0`。
- Rebuild/checksum/PXR/CPU verifier 均 exit `0`；2/2 checksum 通过，verifier 25 DOF、step 1、`validation_errors=[]`。
- 真实 Isaac joint→world→base 输出约 `[1,0,0,0,0,3]`，公共 AppLauncher import/cfg/action exit `0`。

## Result

FIX_ROUND_1_AWAITING_REVIEW。16 action contract 不变。Task 6 仅保留 live 数值/符号与 sensor-facing convention 校准；未加入其实现。
