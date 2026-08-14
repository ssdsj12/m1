# M1 + Panda Task 2→3 Asset Repair

## Purpose And Scope

修复 RobotAssembler live-stage 结果持久化到组合 USD 后的三个错误 composition arcs 与双 articulation root，并完成 Task 3 CPU 验收。Task 4+ 未触碰；GPU `sm_120` 环境问题不在本次范围。

## Stage And Todo

- Stage: T400 / asset-wrench foundation / Task 2→3
- Related todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable

## Scientific Checkpoints

临时候选位于 `/tmp/m1-panda-task23.BkEfa6`，正式资产未在 A/B 阶段被覆盖。

- Checkpoint A 仅从 root layer 删除 `Sdf.Reference("/M1Panda")`、`Sdf.Reference("/M1Panda/Panda")` 与 `Sdf.Payload("/M1Panda/Panda")` 的 item edits。变换退出 `0`；独立重开显示 `invalid_m1panda_unresolved=[]`、剩余 distinct unresolved 仅 `OmniPBR.mdl`，articulation roots 仍为 `[/M1Panda/BASE_LINK, /M1Panda/Panda]`。
- Checkpoint B 仅额外从 `/M1Panda/Panda` 移除 `UsdPhysics.ArticulationRootAPI` 与 `PhysxSchema.PhysxArticulationAPI`。变换退出 `0`；CPU 运行显示唯一 root `/M1Panda/BASE_LINK`、25 DOF、required bodies 存在、一步 physics 完成。旧 verifier 因 15 个 MDL unresolved 仍退出 `1`，但证明了 B 的 root 假设。

Checkpoint B 同时暴露旧 verifier 假阳性：默认全零 joint cfg 使 `panda_joint4=0` 超出 `[-3.072,-0.070]`，初始化 callback 抛错但旧代码继续输出 topology 数据。随后按 TDD 加入官方 Franka home pose 与 `robot.is_initialized` 硬门。

## TDD RED / GREEN

- Serialization/MDL RED: `2 failed, 9 passed`，exit `1`。
- Serialization/MDL GREEN: `11 passed`，exit `0`。
- Initialization RED: focused `1 failed`，缺少合法 Panda home pose 和 initialization evidence。
- Reliable-exit RED: focused `1 failed`，成功路径仍调用可能拖延的 Kit cleanup。
- Mount no-snap RED: focused `1 failed`，缺少一步前后 mount 相对位移证据。
- Final GREEN: `12 passed in 0.01s`，exit `0`；builder/verifier `py_compile` exit `0`。

## Production Build And Checksum

正式 build 使用 `loco` Python、`--headless`，exit `0`。生成器现在：

- 精确清除三个 `_refresh_asset` 错误 list edits；
- 将合法 references 写成 `m1_floating.usda` 与 `panda/panda.usd`；
- 移除 attach 顶层两个 root APIs；
- 保持 `AssemblerFixedJoint` 为 fixed、enabled、`excludeFromArticulation=false`；
- 保持 Panda `root_joint` disabled；
- 用显式 `RuntimeError` 与 Export 返回值检查替代 bare asserts。

Checksum：

- `panda/panda.usd`: `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`
- `m1_panda.usd`: `6acbd32afab08dbfb8963e0f7d990d2988cdfe8ad4fec083d0c9fa1c4585c3ff`
- `sha256sum -c`: `2/2` 成功，exit `0`；manifest 不包含自身。

## CPU Verification

正式资产与复制到新目录的完整资产树分别运行 verifier，均在约 4 秒内可靠 exit `0`。正式 JSON 的关键值：

- `articulation_roots=["/M1Panda/BASE_LINK"]`
- `dof_count=25`
- `runtime_initialized=true`
- `physics_steps=1`
- `mount_joint_is_fixed=true`
- `panda_root_joint_enabled=false`
- `mount_relative_step_delta_m=4.752705353894271e-05`，低于 `1e-4 m` 门限
- `unresolved_dependencies=[]`
- `remote_dependencies=[]`
- `outside_root_dependencies=[]`
- `validation_errors=[]`
- `builtin_mdl_dependencies` 为 15 个 `OmniPBR.mdl`

搬移副本的 8 个真实 USD dependencies 全部解析到新 asset root，证明组合层合法 references 可移动。

## Resolver Boundary And Warnings

15 个 bare `OmniPBR.mdl` 被严格白名单分类为 Isaac Sim built-in MDL resolver boundary；其他任何 unresolved 项仍进入 `unresolved_dependencies` 并使 verifier 失败。这满足兼容 Isaac Sim 安装内的离线使用，但不等价于脱离 Isaac Sim 材质库的严格项目自包含。

保留警告：

- build：Panda `link8` 无质量/碰撞体而获得小的惯量；`panda_joint8` axis 被 importer 调整；DriverShaderCacheManager、无 viewport、IOMMU、Joystick、MaterialX、OmniHub、deprecated dynamic control、no crash reporter；RobotAssembler live stage 与匿名层相对 reference 在构建中短暂报 unresolved，但正式重开无对应 unresolved。
- verify：USD dependency 扫描对 15 个 `OmniPBR.mdl` 打 resolver warning；PhysX 对 disabled `/Panda/root_joint` 打 disjointed-transform warning；空 actuator mapping 打 `0 != 25` warning；以及相同的标准 headless Kit warnings。
- `root_joint` warning 调查：属性在正式层与运行 JSON 中均为 `jointEnabled=false`；PhysX 仍解析该 disabled FixedJoint 并打印 CreateJoint warning。一步前后 mount 相对位移通过门限，未观察到 snap。为保持任务要求的 disabled source root joint，没有删除该 prim。

## Result

DONE_WITH_CONCERNS。Task 2→3 blocker 已解除，CPU 物理验收与资产树搬移验收通过。concerns 限于 built-in MDL resolver 边界、disabled root-joint warning、空 actuator 验证配置 warning 和既有 importer/headless warnings；GPU `sm_120` 未运行且不属于本修复。

## Fix Round 1: Independent Review Important Findings

独立审查指出三个 Important：root predicate 未锁 exact M1 path、mount contract 未锁 body relationships/enabled/exclusion、测试过度依赖 source tokens。

只读 PXR 证据先确认正式资产的 composed 与 authored targets 均为：

- `physics:body0 -> /M1Panda/BASE_LINK`
- `physics:body1 -> /M1Panda/Panda/panda_link0`
- effective `jointEnabled=true`
- `excludeFromArticulation=false`

TDD RED：轻量 suite `4 failed, 12 passed`，exit `1`；PXR behavior runner 因 builder 缺 `_remove_refresh_asset_edits` 退出 `1`。GREEN：轻量 suite `16 passed`，exit `0`；PXR behavior runner 对 in-memory Sdf list ops 证明只移除三个 exact bad objects、保留三个无关 arcs，并对正式重开资产验证 exact root 和完整 mount contract，输出 `{"cleanup":"pass","mount":"pass","roots":["/M1Panda/BASE_LINK"]}`，exit `0`。

Builder 现在在相对化 references 前验证 live composed exact root/mount contract，并在 Export 后独立 reopen 再验证同一 contract；verifier 要求 roots 恰为 `[/M1Panda/BASE_LINK]`，JSON 新增 `mount_body0_targets`、`mount_body1_targets`、`mount_joint_enabled`、`mount_joint_exclude_from_articulation`。依赖分类通过轻量行为测试证明只允许 exact `OmniPBR.mdl`。

重新 build exit `0`，USD hashes 未变化，checksum `2/2` exit `0`。正式与新复制资产树 CPU verifier 均 exit `0`，四个 mount JSON 值分别为 `[/M1Panda/BASE_LINK]`、`[/M1Panda/Panda/panda_link0]`、`true`、`false`，`validation_errors=[]`。原有 warnings/concerns 不变；未处理 review Minor，未触碰 Task 4+。
