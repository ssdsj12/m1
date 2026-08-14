# M1 + Panda 六轴力感知 Teacher–Student 设计记录

## Purpose

记录经用户逐节确认的 M1 + Panda 平衡与静止抓取设计，并验证书面规格不存在明显占位符、范围冲突或输入输出歧义。

## Stage

设计 / Teacher–Student 特权学习 / 六轴力传感器 / M1 残差平衡 / Panda 静止抓取。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Procedure

1. 核对用户已确认的资产、传感器、训练阶段、观测、控制、安全和验收决定。
2. 写入正式设计文档。
3. 搜索 `TBD`、`TODO`、`FIXME`、`待定` 和 `待确认` 等占位内容。
4. 交叉检查物体质量输入边界、六轴信号语义、动作维度、训练阶段和验收指标。
5. 检查目标目录的 Git 状态。

## Input Conditions

- Panda 使用本地离线 USD 依赖闭包。
- 实机安装点使用物理六轴力/力矩传感器。
- Teacher 使用理想总六维反力，Student 使用带噪测量和可部署状态历史。
- M1 基础策略与 Panda IK/OSC 首期冻结。
- 首个操作任务为 M1 驻停、0–3 kg 抓取和短距离搬运。

## Key Metrics

- 设计文档：252 行。
- 占位符扫描：无匹配。
- 明确动作接口：12 个腿关节位置残差 + 4 个轮关节速度残差。
- 明确验收：20 秒随机六维扰动无跌倒率不低于 95%；姿态 P95 小于 10 度；0–3 kg 抓取搬运成功率不低于 80%；抓取阶段无跌倒率不低于 95%。
- Git：`/home/xk/coding/M1` 不是 Git 工作树，无法生成设计提交。

## Result

通过文档级自审。设计覆盖架构、组件边界、数据流、训练、异常处理、安全和验证；未修改运行时代码，未执行仿真或实机测试。

## Conclusion

书面规格可进入用户审阅门。用户批准后才能转入 writing-plans；实施前仍需传感器选型参数和机械最坏工况验算。

## Follow-up

等待用户审阅 [设计文档](../../docs/superpowers/specs/2026-08-14-m1-panda-force-aware-teacher-student-design.md)。

## Git Refs

- Baseline Ref: unavailable（目录不是 Git 工作树）
- Candidate Ref: filesystem working copy
- Key Files:
  - [设计文档](../../docs/superpowers/specs/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
  - [T400 branch memory](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Task 3 Topology Checkpoint

Command:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 120 \
  /home/xk/miniconda3/envs/loco/bin/python Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --device cpu --headless
```

Exit code: `1`.

Verifier JSON:

```json
{
  "root": "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd",
  "dependencies": [
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd",
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/panda/panda.usd",
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/panda/configuration/panda_base.usd",
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/panda/configuration/panda_physics.usd",
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/panda/configuration/panda_sensor.usd",
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_floating.usda",
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd",
    "/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_base.usd"
  ],
  "remote_dependencies": [],
  "outside_root_dependencies": [],
  "unresolved_dependencies": ["/M1Panda", "/M1Panda/Panda", "/M1Panda/Panda", "OmniPBR.mdl (15 occurrences total)"],
  "articulation_roots": ["/M1Panda/BASE_LINK", "/M1Panda/Panda"],
  "mount_joint_is_fixed": true,
  "joint_names": [],
  "body_names": [],
  "dof_count": null,
  "physics_steps": 0,
  "validation_errors": [
    "18 unresolved dependencies",
    "expected one articulation root, found 2",
    "runtime topology check failed because Isaac Lab found multiple articulations under /World/M1Panda"
  ]
}
```

Conclusion: RobotAssembler 的构建期 reference/payload warning 在独立重开后仍存在，不能视为瞬态。FixedJoint prim 有效且类型正确，但依赖闭包、单 articulation、25 DOF、required body uniqueness 和 reset/physics step 尚未通过。完整证据见 [Task 3 log](2026-08-14-m1-panda-offline-topology-verification.md)。

Git Ref: unavailable

## Task 2→3 Repair Checkpoint

Task 2→3 repair 已完成，详细证据见 [repair log](2026-08-14-m1-panda-task23-asset-repair.md)。正式资产 CPU verifier 使用 `--device cpu --headless`，exit `0`：唯一 articulation root `/M1Panda/BASE_LINK`，25 DOF，required bodies 唯一，`runtime_initialized=true`，reset/write/step/update 完成，`physics_steps=1`，真实 USD dependencies 均位于 asset root，remote/outside/unresolved/validation errors 均为空。

15 个 `OmniPBR.mdl` 单独报告为 Isaac Sim built-in MDL resolver boundary；这不是脱离 Isaac Sim 材质库的严格项目自包含。完整资产树搬移到临时新目录后同样 exit `0`。disabled Panda `root_joint` 仍产生 PhysX disjointed-transform warning，但 `jointEnabled=false` 且一步 mount 相对位移为 `4.752705353894271e-05 m < 1e-4 m`，未观察到 snap。

Git Ref: unavailable
