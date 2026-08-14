# Task 2 -> 3 Root-Cause Diagnosis

Status: DONE

Date: 2026-08-14

## Scope and non-mutation guarantee

This was a root-cause-only investigation. No project source, test, USD, checksum, or generated asset was modified or regenerated. Read-only Kit/PXR probes opened the existing assets and inspected layer/spec/property stacks. The only write is this requested report.

Investigated runtime:

- `/home/xk/miniconda3/envs/loco/bin/python`
- Isaac Sim 4.5.0
- Isaac Lab 2.1.0
- installed assembler extension:
  `/home/xk/miniconda3/envs/loco/lib/python3.10/site-packages/isaacsim/exts/isaacsim.robot_setup.assembler`

Required inputs were read completely:

- `.superpowers/sdd/task-3-report.md`
- `Go2Pvcnn/scripts/build_m1_panda_asset.py`
- `Go2Pvcnn/scripts/verify_m1_panda_asset.py`
- installed `robot_assembler.py` (586 lines)
- installed official `tests/test_robot_assembler.py` (392 lines)

## Executive conclusion

The preferred root-cause hypothesis is:

> `RobotAssembler(single_robot=True)` in Isaac Sim 4.5 is a live-stage assembly operation, not a serialization-safe asset-flattening operation. Its live refresh workaround authors bogus asset-path list edits, and its definition of “single robot” disables the attachment's PhysX articulation without removing the attachment's USD articulation-root schemas. Exporting that transient live stage persists both implementation details. On reopen, USD correctly reports the bogus paths, while Isaac Lab 2.1 counts schemas with `HasAPI` and therefore sees two roots even though PhysX considers the Panda root disabled.

This is one compatibility/serialization mismatch with two directly evidenced manifestations, not an arbitrary collection of asset corruptions:

1. `_refresh_asset` creates the three bad `/M1Panda*` composition arcs.
2. `single_robot=True` leaves the Panda `ArticulationRootAPI` authored and only sets `physxArticulation:articulationEnabled = false`; Isaac Lab ignores that enable flag during root discovery.

The existing geometry/joint composition itself is substantially intact: the fixed mount joint is at the contracted path and is included in the articulation; read-only USD inventory contains exactly 25 enabled movable joints (16 M1 + 9 Panda).

## 1. Exact origin of the unresolved `/M1Panda*` arcs

### Call/data flow

The builder creates two referenced prims and immediately exports the current stage:

- `build_m1_panda_asset.py:64-66`: `/M1Panda` references `m1_floating.usda`; `/M1Panda/Panda` references `panda.usd`.
- `build_m1_panda_asset.py:76-85`: invokes `assemble_articulations(..., single_robot=True)`.
- `build_m1_panda_asset.py:87-88`: exports that same root layer.

Installed `RobotAssembler` then calls `_refresh_asset(base_path)` and `_refresh_asset(attach_path)` in both `assemble_rigid_bodies` (`robot_assembler.py:417-418`) and again for `single_robot=True` (`robot_assembler.py:468-469`).

The implementation says the refresh exists to force immediate live-timeline updates around an acknowledged USD Physics bug (`robot_assembler.py:176-179`). It does this:

```python
payload = Sdf.Payload(prim_path)
reference = Sdf.Reference(prim_path)
```

at `robot_assembler.py:183-195`, then runs Remove/Add commands at the prim.

For both Sdf constructors, the one positional string is `assetPath`, not `primPath`. Therefore:

- `Sdf.Reference("/M1Panda")` means asset `@/M1Panda@`, with empty target prim path.
- `Sdf.Reference("/M1Panda/Panda")` means asset `@/M1Panda/Panda@`.
- `Sdf.Payload("/M1Panda/Panda")` means payload asset `@/M1Panda/Panda@`.

The read-only Sdf representation confirmed that these objects have asset paths, e.g. `Sdf.Reference('/M1Panda')`, rather than `Sdf.Reference(primPath=Sdf.Path('/M1Panda'))`.

### Authored proof in `m1_panda.usd`

The root layer exports exactly these list edits:

```usda
def Xform "M1Panda" (
    delete references = @/M1Panda@
    prepend references = [
        @/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_floating.usda@,
        @/M1Panda@
    ]
)

def Xform "Panda" (
    delete payload = @/M1Panda/Panda@
    prepend payload = @/M1Panda/Panda@
    delete references = @/M1Panda/Panda@
    prepend references = [
        @/home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/panda/panda.usd@,
        @/M1Panda/Panda@
    ]
)
```

Both delete and prepend opinions remain because Remove then Add is authoring list edits, not temporarily unloading/reloading without an authored consequence. The prepended opinions remain active and cause reopen errors. This exactly accounts for all three non-MDL unresolved entries:

- reference `@/M1Panda@`
- reference `@/M1Panda/Panda@`
- payload `@/M1Panda/Panda@`

No key sublayer authored any other external file arc outside its intended local composition; the three invalid arcs all originate in `m1_panda.usd`.

## 2. Why `single_robot=True` still leaves two `ArticulationRootAPI` prims

### Attachment-root transformation performed by RobotAssembler

The imported Panda physics layer initially authors `PhysicsArticulationRootAPI` and `PhysxArticulationAPI` on `panda/root_joint`.

`assemble_rigid_bodies`:

1. finds the attachment articulation root (`robot_assembler.py:378`);
2. moves both root APIs from `root_joint` to the attachment top prim `/M1Panda/Panda` (`robot_assembler.py:385-387`);
3. disables the attachment's stage root joint (`robot_assembler.py:389-393`);
4. creates `/M1Panda/Panda/panda_link0/AssemblerFixedJoint` and first excludes it from articulation (`robot_assembler.py:398-404`).

Then `assemble_articulations(single_robot=True)`:

1. sets `physxArticulation:articulationEnabled = false` on `/M1Panda/Panda` (`robot_assembler.py:460-464`);
2. sets the assembler fixed joint's `physics:excludeFromArticulation = false` (`robot_assembler.py:466`).

It does **not** remove either root API from the Panda prim.

### Authored layer-stack proof

Read-only PXR inspection found:

| Composed prim | Root API source | PhysX enabled source/value |
| --- | --- | --- |
| `/M1Panda/BASE_LINK` | `m1_floating.usda` `/ZJ_V3_URDF_V1_0/BASE_LINK`: prepends `PhysicsArticulationRootAPI`, `PhysxArticulationAPI` | schema default/effective `true` |
| `/M1Panda/Panda` | `m1_panda.usd` `/M1Panda/Panda`: prepends `PhysicsArticulationRootAPI`, `PhysxArticulationAPI` | same root-layer spec authors `physxArticulation:articulationEnabled = false` |

Additional composition details:

- M1's source physics layer originally applies the root APIs at `/ZJ_V3_URDF_V1_0/root_joint`.
- `m1_floating.usda` deactivates that root joint and explicitly applies the APIs at `BASE_LINK`, producing the intended surviving base root.
- Panda's source physics layer applies the APIs at `/panda/root_joint`.
- `m1_panda.usd` authors a delete list-op for those APIs at `/M1Panda/Panda/root_joint`, then prepends them on `/M1Panda/Panda` as a result of `move_articulation_root`.

Thus `stage.Traverse()` plus `prim.HasAPI(UsdPhysics.ArticulationRootAPI)` correctly returns both `/M1Panda/BASE_LINK` and `/M1Panda/Panda`. The second root is disabled for PhysX, but its schema is not absent.

### Why the official example works but Isaac Lab does not

The complete installed official use case is:

`.../isaacsim.robot_setup.assembler/isaacsim/robot_setup/assembler/tests/test_robot_assembler.py::_testRobotToRobotAssemble` (`test_robot_assembler.py:191-296`).

It:

- references UR10e and Allegro Hand into a live stage;
- calls `assemble_articulations` with both `single_robot=False` and `True` (`lines 209-227`);
- for `single_robot=True`, controls the result through one `SingleArticulation(base_robot_path)` and verifies 22 joints (`lines 236-251`);
- verifies one base articulation via `find_all_articulation_base_paths()` (`line 251`).

It does not export the assembly and reopen it as a standalone asset.

Critically, Isaac Sim's `find_all_articulation_base_paths()` counts a root only when all three are true: it has the API, has `physxArticulation:articulationEnabled`, and that value is true (`isaacsim/core/utils/articulations.py:80-86`). Therefore the disabled Panda root is ignored and the official live-stage test passes.

Isaac Lab 2.1's `Articulation._initialize_impl`, by contrast, obtains all child prims using only:

```python
predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
```

(`isaaclab/assets/articulation/articulation.py:1156-1159`) and rejects more than one (`lines 1165-1170`). It does not inspect `physxArticulation:articulationEnabled`. This is the direct cause of the CPU runtime exception.

## 3. Authored references, payloads, and key layer closure

Read-only `Stage.GetUsedLayers()` returned the following real layers:

1. `m1_panda.usd`
2. `m1_floating.usda`
3. `m1/.../configuration/ZJ_V3_URDF_V1_0_physics.usd`
4. `m1/.../configuration/ZJ_V3_URDF_V1_0_base.usd`
5. `panda/panda.usd`
6. `panda/configuration/panda_physics.usd`
7. `panda/configuration/panda_base.usd`
8. `panda/configuration/panda_sensor.usd`

Key authored composition:

| Layer/spec | Intended authored composition |
| --- | --- |
| `m1_panda.usd` `/M1Panda` | absolute reference to project `m1_floating.usda`, plus erroneous `@/M1Panda@` list edits |
| `m1_panda.usd` `/M1Panda/Panda` | absolute reference to project `panda/panda.usd`, plus erroneous `@/M1Panda/Panda@` reference and payload list edits |
| `m1_floating.usda` | relative sublayer `./m1/.../ZJ_V3_URDF_V1_0_physics.usd` |
| M1 physics layer | relative sublayer `ZJ_V3_URDF_V1_0_base.usd` |
| `panda/panda.usd`, Physics=`PhysX` variant | payload `configuration/panda_physics.usd` |
| `panda/panda.usd`, Sensor=`Sensors` variant | payload `configuration/panda_sensor.usd` |
| Panda physics layer | relative sublayer `panda_base.usd` |

The variant-contained Panda payloads do not appear as top-level `PrimSpec` arcs in a shallow Sdf walk, but their presence is confirmed by the complete ASCII export of `panda.usd` and by the stage's used layers.

The two intended references in `m1_panda.usd` are absolute filesystem paths. They resolve and are under `asset_root` on this machine, but the asset is not relocatable as a directory. A strict offline/portable repair should author `m1_floating.usda` and `panda/panda.usd` relatively.

## 4. `OmniPBR.mdl`: real closure failure or normal resolver boundary?

The 15 `OmniPBR.mdl` entries are categorically different from `/M1Panda*`:

- Six are authored as `info:mdl:sourceAsset = @OmniPBR.mdl@` in the M1 base layer.
- Nine are authored the same way in the Panda base layer.
- Isaac Sim's own installed `OmniPBR` material class deliberately authors exactly `Sdf.AssetPath("OmniPBR.mdl")` (`isaacsim/core/api/materials/omni_pbr.py:79-81`).
- Numerous installed NVIDIA test/demo USDs use the same bare MDL module name.
- The runtime contains an MDL material library and MDL search-path mechanism; an `OmniPBR.mdl` file is present under the installed material-library extension.

`UsdUtils.ComputeAllDependencies` uses USD/Ar asset resolution. A bare MDL module name is resolved later by the MDL runtime's module search paths, not necessarily by Ar relative to the USD layer. Consequently `ComputeAllDependencies` reports it unresolved even in a normal Isaac Sim material authored by NVIDIA's own API.

Conclusion:

- For **offline use inside this installed Isaac Sim distribution**, the 15 reports are normal USD-versus-MDL resolver behavior, not evidence that geometry/physics assets are missing. They should not be conflated with the three provably invalid composition arcs.
- For **strict project-owned, relocatable closure independent of a compatible Isaac Sim installation**, `OmniPBR.mdl` remains an external runtime dependency. The clean options are to replace these visual shaders with `UsdPreviewSurface`, or legally vendor a complete MDL module/import/resource closure and author portable paths. Copying one arbitrary `OmniPBR.mdl` test file is not a sufficient closure strategy.
- Therefore the current verifier's blanket rule “every `ComputeAllDependencies` unresolved item is fatal” is too coarse for Isaac Sim built-in MDL modules. A repaired verifier should classify known bare MDL modules separately and explicitly state whether the acceptance contract is “Isaac-Sim-offline” or “project-self-contained without Isaac material libraries.”

## 5. Topology and 25-DOF evidence

Read-only composed-stage inventory found:

- 23 revolute joints
- 2 prismatic joints
- 4 fixed joints
- 25 enabled movable joints total
  - 16 under M1
  - 9 under Panda (7 arm revolute + 2 finger prismatic)
- `/M1Panda/Panda/panda_link0/AssemblerFixedJoint`
  - type `PhysicsFixedJoint`
  - `physics:jointEnabled = true`
  - `physics:excludeFromArticulation = false`
- `/M1Panda/Panda/root_joint`
  - type `PhysicsFixedJoint`
  - `physics:jointEnabled = false`

This does not substitute for a successful PhysX/Isaac Lab 25-DOF runtime query, but it shows why the intended result should be 25 DOF once root discovery succeeds. The mount path contract already survives export/reopen.

## 6. CPU versus `sm_120`

The CPU route is expected to bypass the PyTorch CUDA `sm_120` kernel incompatibility for simulation tensors/PhysX because the verifier passes `device="cpu"` to `SimulationCfg` (`verify_m1_panda_asset.py:109`). Kit may still initialize Vulkan and enumerate the RTX 5070 for headless rendering; that does not mean PyTorch CUDA kernels are being used for the CPU simulation device.

The recorded CPU run contains no `sm_120` kernel-image failure and reaches Isaac Lab articulation initialization. It then fails at the deterministic pre-view discovery check described above: Isaac Lab finds two `HasAPI` roots and raises before creating a PhysX articulation view, before reset, and before any physics step.

Precise conclusion:

- The CPU runtime topology failure observed in Task 3 is purely caused by the two-schema-root discovery condition.
- The CPU verifier still exits nonzero independently because its static dependency/root validation also records the three invalid arcs and the 15 MDL classifications. Thus “CPU process exit 1” is not attributable only to the double root, but the **runtime initialization exception** is.
- After removing the duplicate API root, CPU could reveal a later physics problem (for example the logged disjointed root-joint warning); that has not yet been tested. It is therefore not valid to claim the entire physics acceptance will pass solely from this diagnosis.
- The default CUDA route has a separate environment blocker (`sm_120` unsupported by the installed PyTorch build), independent of the USD topology.

## 7. Minimal repair experiment (proposal only; not executed)

Use one temporary candidate under `/tmp` so no accepted asset/checksum is touched. Keep the existing assembler-generated transforms, fixed joint, root-joint disabling, joint states, and collision mask; change only the two serialization artifacts identified above.

### Candidate transformation

Immediately after `assemble_articulations` and before export, in the temporary experiment:

1. Remove only the bogus root-layer reference/payload list-edit items whose **assetPath** is `/M1Panda` or `/M1Panda/Panda`. Preserve the real M1/Panda asset arcs. Prefer preventing `_refresh_asset` from authoring them when the timeline is stopped; direct list-op cleanup is acceptable for the experiment because it isolates the hypothesis.
2. Remove both `UsdPhysics.ArticulationRootAPI` and `PhysxSchema.PhysxArticulationAPI` from `/M1Panda/Panda` rather than merely setting `articulationEnabled=false`. Preserve the deleted root APIs and disabled fixed root joint at `/M1Panda/Panda/root_joint`.
3. Keep `/M1Panda/BASE_LINK` as the only root.
4. Keep `/M1Panda/Panda/panda_link0/AssemblerFixedJoint` unchanged with `excludeFromArticulation=false`.
5. For the portability portion of the experiment, author the two legitimate root references relative to the candidate layer. This should not change composition, only relocation behavior.

The smallest scientific sequence is two checkpoints, not a bundle of speculative changes:

- Checkpoint A, arc-only: reopen the temporary candidate and require zero `/M1Panda*` unresolved items. The root count should intentionally remain two. This confirms `_refresh_asset` causality independently.
- Checkpoint B, API-only addition: require exactly one `HasAPI` root and then run Isaac Lab CPU initialization. This confirms the schema/enable mismatch independently.

### Acceptance for the temporary experiment

Require all of the following before changing the real builder or generated checksum:

1. Reopen emits no `/M1Panda*` asset warnings.
2. Actual USD file dependencies resolve inside the candidate asset root; no remote URLs; intended root references are relative.
3. MDL entries are reported in a separate `builtin_mdl_dependencies` category, or visuals are converted to an explicitly selected project-owned representation.
4. Exactly one `UsdPhysics.ArticulationRootAPI`: `/M1Panda/BASE_LINK`.
5. The mount joint exists at exactly `/M1Panda/Panda/panda_link0/AssemblerFixedJoint`, is a fixed joint, is enabled, and has `excludeFromArticulation=false`.
6. The Panda source root joint remains disabled.
7. Isaac Lab CPU reports 25 DOF, unique `BASE_LINK`, `panda_link0`, and `panda_hand`, completes reset/write/one physics step/update.
8. Only after CPU success, rerun CUDA in an environment whose PyTorch supports the RTX 5070 `sm_120` architecture.

### Likely production repair shape if the experiment passes

Do not patch installed Isaac Sim files. Encapsulate serialization cleanup in the project builder, or replace the live assembler path with explicit USD authoring of the already-understood fixed joint/root operations. Add reopen regression checks before writing/updating `generated_files.sha256`.

The production contract should be explicit:

- one authored root API, not merely one enabled PhysX root;
- 25 runtime DOF;
- stable mount joint path;
- relative project file dependencies;
- a declared policy for built-in MDL versus fully project-owned visual closure.

## 8. Confidence and remaining concerns

High confidence:

- exact source and authored layer responsible for `/M1Panda*` arcs;
- exact layer/spec responsible for each composed articulation root;
- exact semantic mismatch between official Isaac Sim helper and Isaac Lab 2.1;
- CPU bypasses the observed PyTorch `sm_120` failure and stops at double-root discovery;
- 25 movable joints are present in USD and the mount path/type contract survives reopen.

Not yet verified, by instruction:

- a repaired candidate's PhysX articulation view, reset, and physics step;
- runtime 25-DOF/body-name acceptance after root cleanup;
- whether any later disjointed-joint or dynamics issue appears;
- CUDA acceptance on an `sm_120`-compatible PyTorch build;
- which visual-closure policy the project wants for deployment.

No repair was implemented.
