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

## Task 6 Deterministic Wrench Probe Checkpoint

用户随后明确批准 Round 4 架构：每个 case 施加载荷后丢弃 `10` transition steps，再采新的 `50` samples；mean expected sign、expected-sign fraction `>=0.90`、magnitude ratio `>0.20`、finite/no reset 必须同时成立。载荷、坐标转换、clear shim 均保持不变；该批准取代下文旧 checkpoint 的“尚未选择”状态，最终结果以 Round 4 复验为准。

Round 4 TDD 已完成：contract RED `4 failed, 6 passed`，GREEN 后 focused `11 passed`；fresh planned static `53 passed`，pycompile/checksum/PXR/CPU verifier 均 exit `0`。真实 CPU 两次从头运行均在通过 `force_x/force_y` 后于 `force_z` 触发 reset；termination diagnostics 为 `bad_orientation=true`、`base_contact=false`、`time_out=false`、`truncated=false`。这在当时违反 no-reset gate 并形成历史 **BLOCKED** checkpoint；该状态已被下方用户批准的 independent-reset 最终结果取代。

用户随后批准 independent-reset round：初始 row 与每个 axis 均在同一个 env 内先 clear + explicit reset，重新验证 25 DOF/unique bodies，再独立执行 100 settle、50 baseline、10 transition discard、50 samples。其余 gate、载荷、termination、坐标与 clear shim 不变；下方后续 checkpoint 记录最终结果。

Independent-reset 最终 fresh authority exit `0`，原子生成七行 artifact；六轴均为 `50/50` expected-sign samples、finite/no reset/pass。Magnitude ratios：`force_x=1.705708`、`force_y=1.786585`、`force_z=0.919172`、`torque_x=1.269007`、`torque_y=10.788469`、`torque_z=1.180200`。单代理最终 regression `58 passed`，checksum `2/2`、PXR、CPU verifier 与 JSONL validator 均 exit `0`。Task 6 与 asset/wrench foundation 完成；process-enforced network denial 因权限不足仍 unverified，所有后续训练、控制、抓取、传感器、机械与实机阶段仍保持 open。

Task 6 probe 与静态/纯行为契约已实现。历史 Round 3 计划静态 regression exit `0`（`50 passed`），generated checksum `2/2` exit `0`，CPU verifier exit `0`：one articulation、25 DOF、dependencies `8`、remote/outside/unresolved `0`、physics step `1`、validation errors empty；最终 fresh 计数以上方 independent-reset checkpoint 的 `58 passed` 为准。

真实 authority 使用 one-env CPU public Gym create/reset/step。三轮修复依次解决/隔离 Kit close 掩盖返回码、Isaac Lab 2.1 empty-wrench `[0] -> [29,3]`、以及 empty-index `[0] -> [0,3]` 兼容问题。第三轮后真实载荷到达 `force_y`：baseline-subtracted mean `Fy=-27.970840454101562`，expected sign `-1`，magnitude ratio `1.3985420227050782`，但仅 `47/50` samples 为预期符号，严格 50/50 stable-sign gate 失败并 exit `1`。

历史 Round 3 状态为 **BLOCKED**，当时没有生成或手写七行 artifact，且未到达 `force_z` 与 torque cases；上方 Round 4 checkpoint 已取代该运行结论。网络 denial 尝试 `unshare --net true` exit `1`（not permitted），所以 dependency closure 已验证但 denial runtime unverified。完整证据见 [Task 6 log](2026-08-14-m1-panda-wrench-probe.md)。

Git Ref: unavailable

## Task 2→3 Repair Checkpoint

Task 2→3 repair 已完成，详细证据见 [repair log](2026-08-14-m1-panda-task23-asset-repair.md)。正式资产 CPU verifier 使用 `--device cpu --headless`，exit `0`：唯一 articulation root `/M1Panda/BASE_LINK`，25 DOF，required bodies 唯一，`runtime_initialized=true`，reset/write/step/update 完成，`physics_steps=1`，真实 USD dependencies 均位于 asset root，remote/outside/unresolved/validation errors 均为空。

15 个 `OmniPBR.mdl` 单独报告为 Isaac Sim built-in MDL resolver boundary；这不是脱离 Isaac Sim 材质库的严格项目自包含。完整资产树搬移到临时新目录后同样 exit `0`。disabled Panda `root_joint` 仍产生 PhysX disjointed-transform warning，但 `jointEnabled=false` 且一步 mount 相对位移为 `4.752705353894271e-05 m < 1e-4 m`，未观察到 snap。

Git Ref: unavailable
