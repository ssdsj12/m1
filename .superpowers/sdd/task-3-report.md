# Task 3 Report

Status: BLOCKED

## Scope And Changes

- Created `Go2Pvcnn/scripts/verify_m1_panda_asset.py`.
  - Opens `m1_panda.usd` independently with USD.
  - Uses `UsdUtils.ComputeAllDependencies` and reports resolved, remote, unresolved, and outside-root dependencies.
  - Checks exactly one `UsdPhysics.ArticulationRootAPI` root.
  - Checks `/M1Panda/Panda/panda_link0/AssemblerFixedJoint` is valid and `UsdPhysics.FixedJoint`.
  - Attempts one Isaac Lab `Articulation`, reset, write, physics step, update, 25 DOF, and exact required body matches.
  - Uses explicit runtime errors/validation errors rather than bare assertions.
  - Emits one verifier-owned JSON object. Failure flushes JSON and uses `os._exit(1)` because Kit cleanup masked an ordinary return code.
- Modified `Go2Pvcnn/tests/test_m1_panda_asset_static.py` with the requested verifier contract plus regressions discovered during real runtime.
- Updated T400 branch/dashboard, verification log/index, and appended the topology checkpoint to the requested design log.
- Did not modify Task 4+ files, generated USDs, builder, or checksums. Did not initialize Git.

## RED / GREEN Evidence

Initial RED command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_panda_asset_static.py::test_verifier_checks_offline_and_topology_contracts
```

Original result: exit `1`; `1 failed in 0.03s`; `FileNotFoundError` for the absent verifier.

Additional real-runtime-driven RED/GREEN cycles:

- JSON-before-close RED: exit `1`, `assert 5051 < 5023`; GREEN `1 passed in 0.01s`.
- forced nonzero exit RED: exit `1`, missing `sys.stdout.flush()`; GREEN passed.
- combined dependency/physics evidence RED: exit `1`, missing `unresolved_dependencies`; GREEN passed.
- Isaac Lab 2.1 actuator mapping RED: exit `1`, missing `actuators={}`; GREEN passed.

Final GREEN command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_panda_asset_static.py
```

Result: exit `0`; `9 passed in 0.01s`.

## Real Isaac Verification

Primary compatible command:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 120 \
  /home/xk/miniconda3/envs/loco/bin/python Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --headless
```

Final GPU result: exit `1`. It produced JSON, found the persistent asset failures below, and additionally failed runtime initialization because the installed PyTorch supports CUDA architectures through `sm_90` but the RTX 5070 is `sm_120`.

Bounded CPU isolation command:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 120 \
  /home/xk/miniconda3/envs/loco/bin/python Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --device cpu --headless
```

Final CPU result: exit `1`.

JSON summary (the command emitted one verifier JSON object):

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
  "unresolved_dependencies_count": 18,
  "unresolved_dependencies_distinct": ["/M1Panda", "/M1Panda/Panda", "OmniPBR.mdl"],
  "articulation_roots": ["/M1Panda/BASE_LINK", "/M1Panda/Panda"],
  "mount_joint_is_fixed": true,
  "joint_names": [],
  "body_names": [],
  "dof_count": null,
  "physics_steps": 0
}
```

## Warnings And Blockers

- Reopen emits unresolved reference `@/M1Panda@`, unresolved reference `@/M1Panda/Panda@`, and unresolved payload `@/M1Panda/Panda@`. These are persistent; the Task 2 RobotAssembler warnings are not transient.
- `ComputeAllDependencies` returns 15 unresolved `OmniPBR.mdl` entries in addition to the three invalid `/M1Panda*` arcs.
- Stage traversal finds two articulation roots, not one. CPU Isaac Lab explicitly raises: `Failed to find a single articulation when resolving '/World/M1Panda'`.
- The mount joint does pass both validity and `FixedJoint` type checks.
- Because Articulation initialization fails, 25 DOF, required body uniqueness, reset completion, and physics step cannot be claimed (`physics_steps=0`).
- GPU default also has an independent local environment blocker: no PyTorch kernel image for RTX 5070 `sm_120`. CPU removes that issue and still proves the asset failure.
- Other observed warnings: disjointed Panda `root_joint`, no crash reporter, missing rendering_modes config, MaterialX extension notice, OmniHub inaccessible, deprecated dynamic control, IOMMU enabled.

## Self-review

- Task 3-only code scope preserved; no Task 4+ files touched.
- No bare `assert` exists in the runtime verifier; `python -O` cannot suppress validation.
- Every expected verifier failure produces nonzero status even when Kit would otherwise mask it.
- The required top-level JSON interface keys are present on normal validation results: `root`, `dependencies`, `joint_names`, `body_names`, `dof_count`, `remote_dependencies`.
- Static tests are green, but overall Task 3 acceptance is correctly BLOCKED rather than reported successful.
- Notes use repository-relative links and record `Git Ref: unavailable` exactly.

## Required Follow-up

Return to Task 2 scope: remove/fix the invalid authored reference/payload arcs, produce exactly one articulation root, close or explicitly resolve material dependencies locally, rebuild `m1_panda.usd`, and update its checksum. Then rerun Task 3 on CPU and on a CUDA-compatible environment until JSON reports zero unresolved/remote/outside-root dependencies, one root, 25 DOF, unique required bodies, and `physics_steps=1` with exit `0`.

Git Ref: unavailable
