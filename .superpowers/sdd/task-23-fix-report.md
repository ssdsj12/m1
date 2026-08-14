# Task 2→3 Asset Fix Report

Status: DONE_WITH_CONCERNS

Date: 2026-08-14

Git Ref: unavailable

## Outcome And Scope

Task 2→3 blocker is repaired without initializing Git or changing Task 4+.

- `build_m1_panda_asset.py` removes only the three RobotAssembler `_refresh_asset` reference/payload list edits, removes both attach-root APIs from `/M1Panda/Panda`, preserves the fixed mount contract and disabled Panda `root_joint`, and exports two legitimate project references as paths relative to the combined layer.
- Bare asserts and unchecked `Export` were replaced by explicit `RuntimeError` gates; imports are used.
- `verify_m1_panda_asset.py` separates the exact allowlisted built-in module `OmniPBR.mdl` into `builtin_mdl_dependencies`; every other unresolved item remains fatal.
- The verifier uses the official Isaac Lab Franka home pose, requires `robot.is_initialized`, performs reset/write/step/update, and forces a reliable JSON-flushed process exit without waiting for hanging Kit cleanup.
- Formal `m1_panda.usd` and `generated_files.sha256` were regenerated. The checksum manifest contains only the two generated USD files, not itself.

## Checkpoint A: Arc-only Candidate

Candidate root: `/tmp/m1-panda-task23.BkEfa6` (copy of the complete asset tree; formal assets untouched).

Mutation: removed item edits for exactly:

```text
Sdf.Reference("/M1Panda")
Sdf.Reference("/M1Panda/Panda")
Sdf.Payload("/M1Panda/Panda")
```

Mutation command exit: `0`; `Layer.Save()` returned `true`.

Independent reopen result:

```json
{"articulation_roots":["/M1Panda/BASE_LINK","/M1Panda/Panda"],"invalid_m1panda_unresolved":[],"m1_ref_delete":[],"m1_ref_prepend":["/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_floating.usda"],"panda_payload_delete":[],"panda_payload_prepend":[],"panda_ref_delete":[],"panda_ref_prepend":["/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/panda/panda.usd"],"unresolved_distinct":["OmniPBR.mdl"]}
```

Reopen command/pipeline exit: `0`. The three `/M1Panda*` unresolved arcs disappeared while both roots remained, independently confirming `_refresh_asset` causality.

## Checkpoint B: API-only Addition

Mutation: on the same A candidate, additionally removed only:

```text
UsdPhysics.ArticulationRootAPI
PhysxSchema.PhysxArticulationAPI
```

from `/M1Panda/Panda`. Both `RemoveAPI` calls and `Layer.Save()` returned `true`; mutation exit `0`.

The first CPU run reached one root, 25 DOF, required bodies and one physics step. It exited `1` only because the then-current verifier still treated the 15 built-in MDL occurrences as fatal. It also exposed a later verifier defect: `panda_joint4=0` is outside `[-3.072,-0.070]`, and Isaac Lab logged an asynchronous initialization traceback that the old verifier did not detect. Work stopped at that finding and added a new TDD gate instead of treating the JSON as acceptance.

## TDD RED / GREEN Evidence

Serialization and MDL boundary RED:

```text
2 failed, 9 passed in 0.03s
EXIT_CODE=1
```

The two expected failures were missing serialization cleanup/relative-root behavior and absent exact MDL allowlist classification.

First GREEN:

```text
11 passed in 0.01s
EXIT_CODE=0
```

Runtime-driven RED cycles:

- legal Panda home pose plus `robot.is_initialized`: focused `1 failed`, exit `1`;
- reliable success exit without Kit cleanup: focused `1 failed`, exit `1`;
- disabled-root warning no-snap measurement: focused `1 failed`, exit `1`.

Final fresh GREEN:

```text
12 passed in 0.01s
py_compile: EXIT_CODE=0
```

## Production Build And Checksum

Build command:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 120 \
  /home/xk/miniconda3/envs/loco/bin/python Go2Pvcnn/scripts/build_m1_panda_asset.py \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --headless
```

Exit: `0`; output root: `/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd`.

Hashes:

```text
1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51  assets/m1_panda/panda/panda.usd
6acbd32afab08dbfb8963e0f7d990d2988cdfe8ad4fec083d0c9fa1c4585c3ff  assets/m1_panda/m1_panda.usd
```

Fresh `sha256sum -c`: both successful, exit `0`.

## Final CPU Authority

Command:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 120 \
  /home/xk/miniconda3/envs/loco/bin/python Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda \
  --device cpu --headless
```

Fresh authoritative exit: `0` in about 4 seconds.

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
  "unresolved_dependencies": [],
  "builtin_mdl_dependencies": ["OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl", "OmniPBR.mdl"],
  "articulation_roots": ["/M1Panda/BASE_LINK"],
  "mount_joint_is_fixed": true,
  "panda_root_joint_enabled": false,
  "runtime_initialized": true,
  "dof_count": 25,
  "joint_names": ["FAR_ABAD_JOINT", "FBL_ABAD_JOINT", "RAR_ABAD_JOINT", "RBL_ABAD_JOINT", "FAR_HIP_JOINT", "FBL_HIP_JOINT", "RAR_HIP_JOINT", "RBL_HIP_JOINT", "panda_joint1", "FAR_KNEE_JOINT", "FBL_KNEE_JOINT", "RAR_KNEE_JOINT", "RBL_KNEE_JOINT", "panda_joint2", "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"],
  "body_names": ["BASE_LINK", "FAR_ABAD_LINK", "FBL_ABAD_LINK", "RAR_ABAD_LINK", "RBL_ABAD_LINK", "panda_link0", "FAR_HIP_LINK", "FBL_HIP_LINK", "RAR_HIP_LINK", "RBL_HIP_LINK", "panda_link1", "FAR_KNEE_LINK", "FBL_KNEE_LINK", "RAR_KNEE_LINK", "RBL_KNEE_LINK", "panda_link2", "FAR_FOOT_LINK", "FBL_FOOT_LINK", "RAR_FOOT_LINK", "RBL_FOOT_LINK", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_link8", "panda_hand", "panda_leftfinger", "panda_rightfinger"],
  "physics_steps": 1,
  "mount_relative_step_delta_m": 4.752705353894271e-05,
  "validation_errors": []
}
```

Required bodies `BASE_LINK`, `panda_link0`, and `panda_hand` are each present exactly once. The verifier completed `sim.reset()`, `robot.reset()`, `write_data_to_sim()`, one `sim.step()`, and `robot.update()` after confirming initialization.

## Relocatability

The entire formal asset tree was copied to `/tmp/m1-panda-task23.BkEfa6/relocated`. Running the same CPU verifier with that directory as `--asset-root` exited `0`; all eight real USD dependencies resolved under the relocated root, with no remote/outside/unresolved entries. This proves the two legal combined-layer references are relative and the asset tree is movable.

## Built-in MDL Boundary

The 15 bare `OmniPBR.mdl` occurrences are reported under `builtin_mdl_dependencies`. The allowlist is exactly `{"OmniPBR.mdl"}`; a different bare MDL name or any other unresolved asset remains in `unresolved_dependencies` and fails validation.

This acceptance is **Isaac-Sim-offline**, relying on the compatible Isaac Sim built-in MDL resolver. It is not the stronger claim that the project is independent of the Isaac Sim material library. No incomplete MDL library was copied and materials were not mass-rewritten.

## Warnings And Concerns

Production build warnings/errors retained in the log:

- Panda `link8` has no mass/collider; importer assigns a small isotropic inertia.
- `panda_joint8` axis is not primary-axis aligned; importer reorients it.
- RobotAssembler `_refresh_asset` emits transient `/M1Panda*` warnings on the anonymous live stage before cleanup.
- After relative references are authored into the anonymous live layer, that live layer temporarily cannot resolve them; independent exported-stage reopen and relocation both resolve them.
- `DriverShaderCacheManager::init() called without a shutdown()`, no viewport visualization node, `removePath` on a missing prototype, joystick remapping, IOMMU, no crash reporter, missing rendering_modes config, MaterialX notice, OmniHub inaccessible, and deprecated dynamic control.

CPU verifier warnings retained in the log:

- `UsdUtils.ComputeAllDependencies` prints 15 warnings for bare `OmniPBR.mdl`; they are classified as the declared built-in resolver boundary.
- PhysX prints a disjointed-body-transform warning for disabled `/World/M1Panda/Panda/root_joint`. The authored/runtime value is `jointEnabled=false`; PhysX still parses the disabled FixedJoint. The relative `panda_link0` to `BASE_LINK` position changed only `4.752705353894271e-05 m` over the accepted step, below the `1e-4 m` no-snap gate.
- Isaac Lab warns that no actuators are configured (`0 != 25`) because this topology verifier intentionally uses `actuators={}`; it does not alter the 25-DOF topology/reset/write/step/update result.
- Standard headless warnings: modules loaded before SimulationApp, no crash reporter, missing rendering_modes config, MaterialX notice, OmniHub inaccessible, deprecated dynamic control, and IOMMU enabled.

The RTX 5070 `sm_120` / installed PyTorch mismatch was not changed or rerun. CPU is the requested local physics acceptance route.

## Self-review

- No Git repository was initialized; Git Ref remains unavailable.
- No installed Isaac Sim/Isaac Lab or environment file was changed.
- Task 4+ was not touched.
- Production builder uses exact cleanup targets and explicit failure checks; fixed mount path/type/enabled/exclusion and disabled source root joint are guarded before export.
- Verifier does not accept arbitrary unresolved MDL/assets and cannot silently pass an asynchronous initialization failure.
- Formal and relocated CPU runs both exited `0`; fresh static tests, `py_compile`, and checksum verification exited `0`.
- T400 dashboard, branch page, design checkpoint, per-verification log, and log index were aligned with repository-relative links.

## Fix Round 1: Independent Review Important Findings

The independent review in `task-23-review.md` withheld approval for three Important gaps. This round is limited to those Task 2/3 findings; the review's Minor long-horizon, strict-MDL, and embedded-`os._exit` topics were not expanded.

### Read-only relationship evidence

Before changing predicates, a read-only PXR probe opened the formal generated asset and printed both composed and root-layer-authored relationship targets:

```text
COMPOSED_BODY0 ['/M1Panda/BASE_LINK']
COMPOSED_BODY1 ['/M1Panda/Panda/panda_link0']
AUTHORED_BODY0 [Sdf.Path('/M1Panda/BASE_LINK')]
AUTHORED_BODY1 [Sdf.Path('/M1Panda/Panda/panda_link0')]
ENABLED True EXCLUDE False
EXIT_CODE=0
```

This established the exact expected target-path semantics without guessing.

### RED

The new lightweight behavior suite initially failed as intended:

```text
4 failed, 12 passed in 0.04s
EXIT_CODE=1
```

The failures were missing executable helpers for exact unresolved classification, exact articulation-root validation, complete mount-contract validation, and builder list-op/serialized validation behavior.

The PXR behavior runner initially failed on the missing cleanup helper:

```text
AssertionError: builder is missing _remove_refresh_asset_edits
EXIT_CODE=1
```

### GREEN implementation

- The verifier now requires `articulation_roots == ["/M1Panda/BASE_LINK"]`, not merely a count of one.
- The builder applies the same exact root invariant on the composed stage before relative reference authoring and on an independent serialized reopen after `Export`.
- Builder and verifier both require the mount to be a `FixedJoint`, `body0 == ["/M1Panda/BASE_LINK"]`, `body1 == ["/M1Panda/Panda/panda_link0"]`, effective `jointEnabled is True`, and `excludeFromArticulation is False`.
- Verifier JSON now reports `mount_body0_targets`, `mount_body1_targets`, `mount_joint_enabled`, and `mount_joint_exclude_from_articulation`.
- Lightweight executable tests mutate inputs to prove only exact `OmniPBR.mdl` is allowlisted, wrong/single/multiple root paths fail, and each wrong mount field fails.
- The PXR behavior runner constructs real in-memory Sdf list ops containing each bad edit plus unrelated references/payloads. It proves the cleanup removes only the three exact bad objects, then reopens the formal USD and checks the exact root and full mount contract.

GREEN results:

```text
lightweight behavior: 16 passed in 0.02s, EXIT_CODE=0
PXR behavior: {"cleanup":"pass","mount":"pass","roots":["/M1Panda/BASE_LINK"]}, EXIT_CODE=0
```

### Rebuild and fresh runtime evidence

The production builder exited `0`, including its new post-Export serialized reopen gate. Generated content remained deterministic:

```text
panda/panda.usd  1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51
m1_panda.usd     6acbd32afab08dbfb8963e0f7d990d2988cdfe8ad4fec083d0c9fa1c4585c3ff
checksum: 2/2 successful, EXIT_CODE=0
```

The formal and newly relocated asset-tree CPU verifier commands both exited `0`. Their JSON includes:

```json
{
  "articulation_roots": ["/M1Panda/BASE_LINK"],
  "mount_body0_targets": ["/M1Panda/BASE_LINK"],
  "mount_body1_targets": ["/M1Panda/Panda/panda_link0"],
  "mount_joint_enabled": true,
  "mount_joint_exclude_from_articulation": false,
  "mount_joint_is_fixed": true,
  "dof_count": 25,
  "runtime_initialized": true,
  "physics_steps": 1,
  "unresolved_dependencies": [],
  "remote_dependencies": [],
  "outside_root_dependencies": [],
  "validation_errors": []
}
```

The relocated root was `/tmp/m1-panda-task23-round1.6YDUIP`; all eight real USD dependencies resolved beneath it.

### Corrected scope of claims

The full fixed-joint enabled/body/in-articulation semantics are now actually guarded rather than inferred from the runtime output. The result remains only the requested one-step Task 3 CPU smoke. It is not a claim of long-horizon holding or orientation stability, and the Isaac Sim built-in MDL boundary remains distinct from strict material-library independence.

Fix Round 1 status: DONE_WITH_CONCERNS. Git Ref: unavailable.
