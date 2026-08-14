# M1 + Panda Offline And Topology Verification

## Purpose

记录 T400 asset/wrench foundation Task 3 的离线依赖闭包、固定安装关节、articulation 拓扑和物理启动验收。

## Stage And Todo

- Stage: T400 / asset-wrench foundation / Task 3
- Related todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Input Conditions

- Asset: `Go2Pvcnn/assets/m1_panda/m1_panda.usd`
- Asset root: `Go2Pvcnn/assets/m1_panda`
- Compatible runtime: `/home/xk/miniconda3/envs/loco/bin/python`, Isaac Sim 4.5.0 / Isaac Lab 2.1.0
- Git Ref: unavailable
- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy

## TDD

Initial RED:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_panda_asset_static.py::test_verifier_checks_offline_and_topology_contracts
```

Result: missing verifier, `1 failed in 0.03s`, exit `1`.

Follow-up RED cases caught JSON emission after Kit shutdown, Kit-masked failure status, missing combined dependency/physics evidence, and the Isaac Lab 2.1 required `actuators` field. Each failed before the matching implementation change.

Final GREEN:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_panda_asset_static.py
```

Result: `9 passed in 0.01s`, exit `0`.

## Real Runtime Verification

GPU command used the compatible interpreter and exited `1`. It additionally reported that this environment's PyTorch has no RTX 5070 `sm_120` kernel image, so a bounded CPU rerun separated that environment issue from the asset issue:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 120 \
  /home/xk/miniconda3/envs/loco/bin/python Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --device cpu --headless
```

Result: exit `1` with one verifier JSON object. Key evidence:

- resolved dependencies: 8; all are under asset root
- `remote_dependencies=[]`
- `outside_root_dependencies=[]`
- unresolved dependencies: 18 entries: three `/M1Panda*` reference/payload paths plus fifteen `OmniPBR.mdl`
- articulation roots: `[/M1Panda/BASE_LINK, /M1Panda/Panda]`, expected exactly one
- mount joint valid and `mount_joint_is_fixed=true`
- Isaac Lab rejected `/World/M1Panda` because it found both articulation roots
- `dof_count=null`, `body_names=[]`, `joint_names=[]`, `physics_steps=0`

The RobotAssembler build-time reference/payload warnings are therefore persistent after reopening, not transient. The CPU run removes the CUDA compatibility issue as the cause of the topology failure.

## Warnings

- Persistent USD unresolved reference/payload warnings for `/M1Panda` and `/M1Panda/Panda`.
- Repeated unresolved `OmniPBR.mdl` material dependencies.
- CPU PhysX warning for a disjointed `/World/M1Panda/Panda/root_joint`.
- GPU-only warning: the installed PyTorch supports through `sm_90`, not RTX 5070 `sm_120`.
- Standard headless Kit warnings: no crash reporter, missing rendering_modes config, OmniHub inaccessible, deprecated dynamic control, IOMMU enabled.

## Result And Follow-up

BLOCKED. The verifier itself is implemented, statically green, emits one JSON object, and reliably returns non-zero on validation failure. The Task 3 acceptance cannot pass because the generated Task 2 asset has unresolved authored arcs and two articulation roots. Fix/rebuild the Task 2 asset, update its generated checksum, then rerun this verifier; a compatible GPU PyTorch build is also needed for the default CUDA route, although CPU already proves the asset topology blocker.

## Key Files

- `Go2Pvcnn/scripts/verify_m1_panda_asset.py`
- `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- `Go2Pvcnn/assets/m1_panda/m1_panda.usd`
- [Task 3 report](../../.superpowers/sdd/task-3-report.md)
