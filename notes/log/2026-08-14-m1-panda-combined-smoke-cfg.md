# M1 + Panda Combined Config And Isolated Smoke

## Purpose And Scope

完成 T400 foundation Task 4：组合 articulation 配置、只控制 M1 的隔离 smoke 环境和 Gym 注册。未加入 Task 5 mount wrench、IK、OSC 或训练逻辑。

## Stage And Todo

- Stage: T400 / asset-wrench foundation / Task 4
- Related todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable

## Input Conditions And Contracts

- 组合 USD：`Go2Pvcnn/assets/m1_panda/m1_panda.usd`
- DOF：25
- policy action：12 个 `M1_LEG_JOINT_NAMES` position + 4 个 `M1_WHEEL_JOINT_NAMES` velocity
- joint observations：`joint_pos_rel` 和 `joint_vel_rel` 均显式限定 `M1_JOINT_NAMES`；`last_action` 来自仅含上述 16 维的 Action Manager
- Panda：三个 implicit actuator 仅保持合法 Franka home pose，不暴露 Panda action

## TDD RED / GREEN

RED command：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_smoke_cfg_static.py
```

结果：`3 failed in 0.05s`，exit `1`；失败均为两个 Task 4 模块缺失。

GREEN 同命令：`3 passed in 0.01s`，exit `0`。行为测试确认 `M1_PANDA_CFG` 的 spawn/init_state/actuators 与原 `M1_CFG` 不共享，导入后原 USD path、joint defaults 和 actuator keys 均未变化。

## Planned Static Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_smoke_cfg_static.py
```

结果：`22 passed in 0.03s`，exit `0`。旧 M1 smoke contract 保持绿色。

四个 Task 4 Python/test 文件的 `py_compile` exit `0`。

## Real Import Evidence

使用 `OMNI_KIT_ACCEPT_EULA=Y`、loco Python、headless `AppLauncher` 和 60 秒 timeout，预加载隔离 `go2_pvcnn.tasks` namespace 后导入并实例化 `M1PandaSmokeEnvCfg`，exit `0`：

```text
{'dof': 25, 'usd_matches': True, 'original_unchanged': True,
 'actions': (12, 4), 'obs_joints': 16}
```

这证明 Task 4 模块在兼容 IsaacLab 环境中真实可导入，但未创建物理环境或执行仿真 step。

## Baseline And Concerns

全部 `tests/test_m1*_static.py` 结果为 `49 passed, 1 failed in 0.06s`。失败是既有 `test_m1_wave_cfg_static.py::test_wrapper_keeps_wave_wheels_equal_and_negative`，它要求 wrapper 中存在旧文本 `prepared[:, 12:16] = wheel_action.expand(-1, 4)`；Task 4 未修改 wrapper 或 wave cfg，因此不越界修复。

默认 `go2_pvcnn.tasks` 全链真实导入在既有 `m1_small_obstacle_env_cfg.py` 处因当前 loco IsaacLab 不导出 `MultiMeshRayCasterCfg` 而失败。隔离 Task 4 import 成功，Gym 注册由静态测试覆盖；未声称默认全注册链或 `gym.make` 已通过。

## Result

DONE_WITH_CONCERNS。Task 4 的配置、16 维 M1 action/observation 隔离、注册、原 cfg 不污染与真实 package import 均有证据；物理 env step 和既有 wave baseline 失败未验证/未修复。Task 5+ 未触碰。

## Fix Round 1

独立审查指出公共 Gym ID 不可达和测试不可证伪两个 Important。真实 RED 使用 loco/AppLauncher 正常导入 `go2_pvcnn.tasks`，可靠 exit `17`，trace 精确停在 `register_m1_envs.py` eager import `m1_small_obstacle_env_cfg.py` 后缺少 `MultiMeshRayCasterCfg`。IsaacLab 安装内官方任务注册使用 `module:Class` string cfg entry point；因此只修改 `register_m1_envs.py`，移除 task cfg 顶层 imports，并将全部 21 个旧/新 M1 ID 改为 lazy strings。旧 ID、class 映射和真正解析旧 cfg 时的异常语义均保留，不吞异常、不重复注册 Panda ID，也未修改 obstacle 实现。

测试先升级为 AST 行为断言：锁定 action 参数、joint observation 与 selector 一一绑定且无其他 all-joint term、Panda home pose 和三组 actuator 全参数；Round 1 当时对“完整 action/21 项 mapping”的措辞过强，后续审查发现它仍会过滤额外 action，并只精确锁 Panda mapping。Review RED 为 `1 failed, 4 passed in 0.03s`；GREEN 为 `5 passed in 0.01s`。旧 smoke/walk source expectations 的迁移 RED 为 `2 failed, 24 passed`，改为等价 lazy strings 后计划 suite `24 passed`、old M1 smoke baseline `4 passed`、pycompile exit `0`。

真实公共路径验证 exit `0`：`import go2_pvcnn.tasks` 后 `gym.spec("Isaac-M1-Panda-Smoke-v0")` 返回正确 manager entry point 和 `go2_pvcnn.tasks.m1_panda_smoke_env_cfg:M1PandaSmokeEnvCfg`，registry 中 M1 IDs 为 `21`，未暴露下一 import blocker。第二个正常 package import 进程实例化真实 cfg，确认 `copy_isolated=True`、action types 为 position/velocity、dims `(12,4)`、两个 observation joint dims `(16,16)`，actuators 为 M1 两组加 Panda 三组。旧 small-obstacle spec 仍映射原模块；显式解析该 lazy string 可靠 exit `23` 并抛出原 `MultiMeshRayCasterCfg` ImportError，证明错误被推迟到真正配置加载而未被吞掉。

扩展全量 M1 static 为 `51 passed, 1 failed in 0.06s`；唯一失败仍是 Task 4 前已存在的 wave-wrapper 旧 token。物理 env create/step 仍未执行。

## Fix Round 2

Round 1 复审指出测试仍可被两个路径绕过：action test 先按已知名称过滤，registry test 仅锁 21 个 ID 顺序与 Panda 的精确 target。Fix Round 2 未修改生产代码，只把契约整理为两个纯 AST validator。

- action validator 收集 `M1PandaSmokeActionsCfg` 中所有 call-valued 配置赋值，不预过滤名称，并要求完整顺序恰为 `leg_pos, wheel_vel`，再逐项检查 type/asset/joints/scale/default offset/clip。
- registry validator 内置显式 21 项 ordered `ID -> module:Class` mapping；每项同时要求 manager entry point 精确、checker 为 `True`、kwargs keys 恰为 env cfg/RSL、RSL 为 `None`。
- mutation fixtures 将额外 Panda action、任一旧 target、manager、checker、extra kwarg、non-None RSL 分别注入源码，并要求 validator 全部抛 `AssertionError`，证明测试可证伪，而不是从生产源码动态自证。

Focused mutation/contract suite `11 passed in 0.03s`；计划 suite `30 passed in 0.03s`；old M1 asset/smoke baseline `4 passed in 0.01s`；三个相关测试文件 `py_compile` exit `0`。生产代码不变，因此按任务要求不重跑 AppLauncher，沿用 Round 1 的真实 public import/spec 与 configclass 证据。Git Ref: unavailable。
