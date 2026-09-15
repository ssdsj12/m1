# T500 OakInk Geometry Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the independently SHA-pinned, fail-closed resolver approved for OakInk geometry overlays.

**Architecture:** Implement a standalone verifier/resolver first; no converter integration or corpus qualification is claimed by this prerequisite. Generated geometry publication and converter integration follow as separately reviewed deliverables after real CoACD validation.

**Tech Stack:** Python standard library, pathlib, JSON, hashlib, XML; pytest.

## Global Constraints

- Preserve original archives, source checkout and run_e unchanged.
- External manifest SHA is supplied by the caller, not trusted from a self-declaration.
- Resolver handles only declared OakInkV2 references; default/FAVOR resolution remains unchanged.
- Reject absolute paths, traversal, symlinks, duplicate paths, undeclared mesh references, hash drift and nonzero origin/nonunit scale.
- No public T400/T500 contract changes, training entrypoint, contact MPC, perception or Residual.
- Geometry verification does not replace finite/watertight/outward/positive-volume checks in preprocessing.
- This task has no authority to set a production corpus pin or mark geometry generation deterministic.

### Task 1: Verified immutable geometry resolver

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/geometry_overlay.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_geometry_overlay.py`

**Interfaces:**
```python
def verify_geometry_overlay(root: Path, manifest_path: Path, expected_sha256: str) -> VerifiedGeometryResolver: ...
class VerifiedGeometryResolver:
    @property
    def manifest_sha256(self) -> str: ...
    def resolve(self, reference: str, *, source: str) -> Path: ...
```

Manifest JSON schema version 1 has exactly `schema_version`, `upstream`, `recipe`, `entries`. Upstream binds repository/revision/archive SHA; recipe is a nonempty JSON object binding tool/version/arguments. Each entry has exactly `reference`, `raw_member`, `raw_sha256`, `urdf`, `urdf_sha256`, `meshes`, `checks`. Each mesh has exactly `path`, `sha256`; checks is a nonempty JSON object reserved for generated validation evidence. References and files are unique normalized POSIX relative paths. Reject duplicate JSON keys, nonfinite JSON numbers, malformed hashes and unsupported schema. Only accept upstream repository `kelvin34501/OakInk-v2`, revision `21705616140d726607027e70d58b7837f442ffd8`, archive SHA `40bb71fb59e1288e5673f32c5bd8fdb501bef15e8b666005b4a6349549983cd2`. Manifest identity is SHA256 of exact manifest bytes. Manifest path and all generated files must stay inside root without any symlink component; reject even contained symlinks. Every URDF visual/collision mesh must be declared, use unit scale and zero origin, resolve relative to its URDF, and stay inside root. Require at least one collision mesh. Verify all referenced bytes at construction and recheck the requested URDF/meshes at resolve to detect subsequent drift. Copy parsed data into private immutable records; do not expose mutable dictionaries. Raise ValueError for rejected identity/path/schema/source/reference/geometry declarations.

- [ ] **Step 1: Write failing fixture tests.** Build a temporary zero-origin/unit-scale one-link URDF and declared mesh, manifest with pinned upstream and recipe/check dictionaries. Check successful resolution and externally computed exact SHA. Include the regression:
```python
def test_wrong_external_pin_rejected(valid_overlay):
    root, manifest, reference = valid_overlay
    with pytest.raises(ValueError):
        verify_geometry_overlay(root, manifest, "0" * 64)
```
Parameterize unsafe paths, duplicate keys/entries/mesh paths, wrong upstream pin, missing mesh, URDF drift, mesh drift, unknown reference, non-OakInk source, nonunit scale, nonzero origin and symlink parent. Assert invalid construction does not change fixture bytes. Assert post-construction file mutation is rejected by resolve.

- [ ] **Step 2: Run red tests.**
```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_expert_prior_geometry_overlay.py -q
```
Expected: import failure before implementation.

- [ ] **Step 3: Implement verifier and resolver.** Separate strict JSON/path/hash helpers from XML declaration verification. Resolve rejects sources other than `oakinkv2`; no fallback path guessing. A frozen record stores reference, URDF path/hash and mesh path/hash tuples; resolver owns a read-only mapping. Hash the bytes used for parsing rather than reopening the manifest. Reuse the same bounded file validation helper for construction and resolution; never relax the existing preprocessing geometry gate.

- [ ] **Step 4: Run focused green tests and source regression.**
```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_expert_prior_geometry_overlay.py tests/test_m1_bimanual_expert_prior_dexmanipnet.py -q
```
Expected: all pass; existing default resolver is untouched.

- [ ] **Step 5: Self-review and scoped commit.**
```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/geometry_overlay.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_geometry_overlay.py
git commit -m "feat: verify independent OakInk geometry overlays"
```
Report tests/commit/concerns, explicitly state this is only a standalone resolver and not accepted geometry or corpus evidence.
