# M1 + Panda 零间隙安装设计

## 1. 目的

将 M1 背部与 Panda `panda_link0` 安装平面之间的人为垂直间隙从 `10 mm` 改为 `0 mm`，使 Panda 底座在视觉和安装坐标上直接落在 M1 `BASE_LINK` 顶面。

本变更只修订组合资产的安装高度，不改变统一 articulation、Panda 朝向、M1/Panda 关节拓扑、动作维度、Teacher 观测或 checkpoint 网络结构。

## 2. 当前状态

组合资产由 `Go2Pvcnn/scripts/build_m1_panda_asset.py` 生成。当前安装偏移为：

```python
MOUNT_CLEARANCE_M = 0.01
mount_offset_z = base_top_z - base_origin_z + MOUNT_CLEARANCE_M
```

正式资产具有：

- 唯一 articulation root `/M1Panda/BASE_LINK`；
- M1 `BASE_LINK` 到 Panda `panda_link0` 的 enabled `AssemblerFixedJoint`；
- `excludeFromArticulation=false`；
- 25 DOF，其中策略仍只控制 M1 的 16 DOF；
- M1/Panda 装配时 `mask_all_collisions=True`，避免两个子机器人在固定安装界面产生内部接触反力。

## 3. 选定方案

采用构建源参数修订，不直接手工平移生成后的 USD Prim：

```python
MOUNT_CLEARANCE_M = 0.0
```

重新运行现有资产构建入口，使固定关节父侧安装位置由 `BASE_LINK` 顶面加 `10 mm` 改为严格的 `BASE_LINK` 顶面。Panda 子侧 `localPos1=(0,0,0)` 和 `localRot1=identity` 保持不变，安装方向也保持不变。

不采用直接编辑 `m1_panda.usd` 的方案，因为下次构建会覆盖手工修改；本轮也不加入实体转接板，因为用户已选择零间隙直接贴合。

## 4. 几何与碰撞契约

“零间隙”定义为构建器计算的 `BASE_LINK` 世界包围盒顶面与 Panda 安装原点之间没有额外 clearance，不代表允许不可控的网格深度穿透。

重建后必须验证：

- Panda 底座位于 M1 背部上方，方向未翻转；
- 安装原点高度等于 `base_top_z`，容差不超过 `1e-6 m`；
- Panda 可视/碰撞网格没有明显穿入 M1 主体；
- 两个子机器人之间仍使用装配 collision mask，不因零间隙产生持续内部接触；
- 不改变地面碰撞、M1 自身碰撞或 Panda 与外部物体的碰撞能力。

如果 Panda link0 网格本身在安装原点以下延伸，验收不得通过简单降低更多高度解决；应停止并回到转接板或几何安装面方案。

## 5. 动力学与控制影响

固定关节和单 articulation 保持不变，因此 Panda 的质量、惯性、关节运动及手端接触反力仍会传递到 M1。高度降低 `10 mm` 会轻微改变整体重心、力臂和安装点力矩，属于真实动力学变化。

Teacher observation/action 仍为 60/16，A0/A1 checkpoint tensor shape 与 manifest 结构仍兼容。但结构兼容不等于行为等价：旧 checkpoint 只允许用于短程对比和诊断，必须重新运行 A0/A1 Play 验证。若后续把零间隙资产作为正式训练/实机基线，应从该资产重新训练或完成足够的长程回归后再接受。

## 6. 实施边界

计划修改范围：

- `Go2Pvcnn/scripts/build_m1_panda_asset.py` 的 clearance 常量；
- 资产构建/校验测试中的零间隙契约；
- 重新生成的 `Go2Pvcnn/assets/m1_panda/m1_panda.usd` 及对应 checksum/manifest；
- verifier 输出或测试，使父侧安装高度可被精确验证；
- T400 工作记忆和资产验证日志。

不修改：

- Panda URDF、网格、质量和惯量；
- Panda 朝向和水平位置；
- Teacher reward、PPO、disturbance 或 play 行为；
- Student、IK/OSC、夹爪或抓取任务；
- 实机机械转接板设计。

## 7. 验证顺序

实施采用单代理 TDD：

1. 静态测试先要求 `MOUNT_CLEARANCE_M == 0.0` 并产生 RED。
2. 扩展构建/验证契约，检查父侧 mount 高度与零 clearance 关系。
3. 重建组合 USD，更新受控 checksum。
4. 运行轻量资产测试、PXR reopen 验证和完整 CPU topology verifier。
5. 验证唯一 articulation、25 DOF、fixed/enabled/in-articulation mount、依赖闭包和一步 no-snap。
6. 进行一次视觉检查，确认 Panda 底座贴合且没有明显网格穿透。
7. 使用 GPU0 分别执行 A0、A1 默认扰动短程 Play，确认 60/16、finite、nonzero wrench 和 frozen actor hash。
8. 记录新资产 checksum、安装高度、no-snap 位移和 Play 结果。

任一 topology、穿透、no-snap 或 finite gate 失败时，不覆盖已接受的资产结论；恢复 `0.01 m` 并重新构建即可回滚。

## 8. 验收标准

- `MOUNT_CLEARANCE_M` 为精确 `0.0`；
- 固定关节父侧安装点等于构建时 `BASE_LINK` 顶面，误差 `<=1e-6 m`；
- 唯一 articulation root、25 DOF 和 required bodies 全部通过；
- mount fixed/enabled、`excludeFromArticulation=false`、child local pose 不变；
- 一步 mount relative delta 继续 `<1e-4 m`；
- 无明显 M1/Panda 网格穿透或内部接触抖动；
- checksum、PXR reopen、CPU verifier 和相关静态回归全部 exit `0`；
- GPU0 A0/A1 Play 均 exit `0`，观测/动作为 60/16，默认扰动非零；
- 文档明确旧 checkpoint 仅完成兼容性复验，不自动升级长期行为或抓取验收。

## 9. 已知限制

零间隙只是仿真安装面贴合，不等价于已设计真实螺栓、转接板、传感器法兰或结构安全系数。最大载荷和快速摆臂实机测试仍受 T400.3 机械验算门约束。
