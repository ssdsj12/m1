# RialTo Object Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pinned RialTo real-USD objects, a validated catalog/resolver, deterministic multi-instance IDs, and explicit target/obstacle wiring for M1/O6 while preserving the legacy Box scene.

**Architecture:** A repository-local asset bundle stores downloaded sources and converted USDs with SHA-256 metadata. A pure-Python catalog/resolver validates classes, paths, hashes, physical defaults, and instance IDs before Isaac startup. The O6 scene adapter optionally consumes catalog instances; absent a catalog scene it retains the existing `/Box` path.

**Tech Stack:** Python 3.11, Isaac Lab/Isaac Sim USD, PyTorch tests, JSON manifests, `usd-core`/Isaac conversion utilities available in the go2 environment.

## Global Constraints

- No runtime network access; all assets are pinned locally with SHA-256.
- Preserve the existing `/Box` default and public M1/O6 contracts.
- Do not add contact-rich screw-cap MPC or VLM behavior in this change.
- Explicit `target_object_id` takes precedence over ranking.
- No asset may escape the repository asset root through catalog resolution.

### Task 1: Add pinned RialTo source manifest and fetch/convert utility

**Files:**
- Create: `Go2Pvcnn/assets/m1_objects/rialto/README.md`
- Create: `Go2Pvcnn/assets/m1_objects/rialto/source_manifest.json`
- Create: `Go2Pvcnn/scripts/m1_rialto_object_assets.py`
- Test: `Go2Pvcnn/tests/test_m1_rialto_object_assets.py`

**Interfaces:**
- `source_manifest.json` records `repo`, `revision`, `source_url`, `source_sha256`, and `resolved_path`.
- `fetch_sources(destination: Path) -> dict[str, Path]` refuses network use unless explicitly requested by the asset-preparation command.
- `convert_usdz_or_glb(source: Path, destination: Path) -> Path` writes a deterministic USD and raises on unsupported input.
- `inspect_usd(path: Path) -> dict[str, object]` reports prim count, bounds, and dependency status.

- [ ] Write tests for manifest schema, SHA recording, unsupported extensions, and deterministic output names.
- [ ] Run `PYTHONPATH=Go2Pvcnn pytest -q Go2Pvcnn/tests/test_m1_rialto_object_assets.py` and verify the new tests fail before implementation.
- [ ] Implement explicit source URLs for `bottle_fixed.usd`, `coffeecup.usdz`, `bowlnrack2.usd`, `book_fixed.usd`, `box.glb`, and `poly.glb`; do not silently substitute another asset.
- [ ] Implement conversion and inspection with clear failure messages for missing converters or unresolved USD dependencies.
- [ ] Run the focused test and record source/output SHA-256 values.
- [ ] Commit: `feat: pin RialTo object assets`

### Task 2: Implement object catalog and resolver

**Files:**
- Create: `Go2Pvcnn/config/m1_object_catalog.json`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_catalog.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Test: `Go2Pvcnn/tests/test_m1_object_catalog.py`

**Interfaces:**
- `ObjectClassRecord(object_class: str, usd_path: Path, sha256: str, mass_kg: float, scale: float, collision_profile: str, grasp_profile: str)`.
- `ObjectInstance(object_id: str, object_class: str, pose: tuple[float, ...], enabled: bool = True)`.
- `load_catalog(path: Path, asset_root: Path) -> ObjectCatalog`.
- `ObjectCatalog.resolve(object_class: str) -> ObjectClassRecord`.
- `ObjectCatalog.validate_instances(instances: Sequence[ObjectInstance]) -> tuple[ObjectInstance, ...]`.

- [ ] Write tests for all six classes, path traversal rejection, SHA mismatch, duplicate IDs, deterministic ordering, and legacy Box fallback.
- [ ] Run focused tests and verify failure.
- [ ] Implement JSON schema with classes `bottle`, `cup`, `bowl`, `book`, `cube`, and `cylinder` (the cylinder record must carry a geometry-inspection note if `poly.glb` is not cylindrical).
- [ ] Implement resolver validation and stable instance ordering.
- [ ] Run focused tests and verify all pass.
- [ ] Commit: `feat: add validated M1 object catalog`

### Task 3: Add multi-instance scene configuration

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_object_scene.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_m1_object_scene.py`

**Interfaces:**
- `build_object_scene_cfg(catalog: ObjectCatalog, instances: Sequence[ObjectInstance], env_regex: str) -> dict[str, RigidObjectCfg]`.
- `select_target_and_obstacles(instances: Sequence[ObjectInstance], target_object_id: str | None) -> tuple[ObjectInstance | None, tuple[ObjectInstance, ...]]`.

- [ ] Test explicit target selection, obstacle partitioning, unique prim paths, and no-catalog legacy Box behavior.
- [ ] Implement deterministic prim paths `{ENV_REGEX_NS}/Objects/{object_id}` and per-class mass/scale settings.
- [ ] Keep existing `scene["box"]` and `/Box` configuration untouched when no catalog is supplied.
- [ ] Run focused tests.
- [ ] Commit: `feat: add deterministic multi-object O6 scene config`

### Task 4: Wire target object IDs into the O6 task/MPC adapter

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/*` only at the task-goal boundary
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Test: `Go2Pvcnn/tests/test_m1_object_target_contract.py`

**Interfaces:**
- CLI option: `--object-id OBJECT_ID` (optional; omission preserves Box default).
- Wrapper state: `target_object_id: str | None`, `obstacle_object_ids: tuple[str, ...]`.
- MPC goal input: target object pose plus obstacle object poses; no geometry or network download in the controller.

- [ ] Test CLI parsing, unknown-ID rejection, explicit target precedence, and obstacle exposure.
- [ ] Add `--object-id` and pass it through the wrapper without changing existing play defaults.
- [ ] Ensure the MPC receives only selected target state and obstacle states through the existing goal boundary.
- [ ] Run all object/catalog and existing O6 contract tests.
- [ ] Commit: `feat: expose O6 target object selection`

### Task 5: Asset and GPU0 smoke verification

**Files:**
- Modify: `Go2Pvcnn/scripts/verify_m1_dual_panda_o6_asset.py`
- Create: `Go2Pvcnn/scripts/verify_m1_rialto_object_catalog.py`
- Test: `Go2Pvcnn/tests/test_m1_rialto_object_verification.py`

- [ ] Add offline checks for all catalog files, SHA values, USD dependencies, unique IDs, and collision metadata.
- [ ] Run CPU/unit checks and verify legacy asset checks remain green.
- [ ] Run GPU0 headless smoke with one bottle, one cup, one cube, and one cylinder candidate; verify target/obstacle partition and contact initialization.
- [ ] Record JSON evidence and aggregate SHA manifest under `artifacts/m1_rialto_object_catalog/`.
- [ ] Commit: `test: verify RialTo object catalog in O6`

