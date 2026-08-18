# M1 + Panda Zero-Clearance Teacher Rebaseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the M1 + Panda asset with zero artificial mount clearance, verify geometry/topology/no-snap, and re-accept C0/C1a before Student collection.

**Architecture:** Change the source builder, extend the independent verifier with a parent mount-plane measurement, regenerate/checksum the USD, then rerun unchanged Teacher gates. The accepted USD SHA becomes a mandatory Student manifest input.

**Tech Stack:** Python 3.11, pytest, Isaac Sim 5.1, Isaac Lab, USD/PXR, PhysX, PyTorch CUDA, SHA-256, GPU0.

## Global Constraints

- Work in `/home/xk/coding/M1`; run code commands from `Go2Pvcnn`.
- Single-agent inline execution only; never stage `graphify-out/`.
- Set `MOUNT_CLEARANCE_M` to exact `0.0`; do not randomize it or hand-edit the output Prim.
- Preserve one root `/M1Panda/BASE_LINK`, 25 DOF, enabled fixed mount, `excludeFromArticulation=false`, child local pose, orientation, collision mask, mass and inertia.
- Mount-plane error `<=1e-6 m`; one-step relative delta `<1e-4 m`.
- Stop before Student work on penetration, jitter, topology/no-snap failure, or any C0/C1a gate failure.

---

### Task 1: Freeze the zero-clearance builder contract

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- Modify: `Go2Pvcnn/scripts/build_m1_panda_asset.py`

**Interfaces:**
- Consumes: `base_top_z`, `base_origin_z`, RobotAssembler offset.
- Produces: `MOUNT_CLEARANCE_M = 0.0` and `mount_offset_z(base_top_z, base_origin_z, clearance_m) -> float`.

- [ ] **Step 1: Write the failing AST/pure test**

```python
def test_builder_uses_exact_zero_mount_clearance_and_top_plane_offset():
    path = ROOT / "scripts" / "build_m1_panda_asset.py"
    tree = ast.parse(path.read_text())
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "MOUNT_CLEARANCE_M"
    }
    assert values == {"MOUNT_CLEARANCE_M": 0.0}
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    namespace = {"np": __import__("numpy")}
    exec(compile(ast.Module(body=[functions["mount_offset_z"]], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["mount_offset_z"](0.42, 0.17, 0.0) == pytest.approx(0.25)
```

Add `import pytest` to the test.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py::test_builder_uses_exact_zero_mount_clearance_and_top_plane_offset
```

Expected: failure for old `0.01` and missing helper.

- [ ] **Step 3: Implement the source change**

```python
MOUNT_CLEARANCE_M = 0.0


def mount_offset_z(base_top_z: float, base_origin_z: float, clearance_m: float) -> float:
    values = (base_top_z, base_origin_z, clearance_m)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("mount offset inputs must be finite")
    if clearance_m < 0.0:
        raise ValueError("mount clearance must be nonnegative")
    return float(base_top_z - base_origin_z + clearance_m)
```

Use `np.array([0.0, 0.0, mount_offset_z(base_top_z, base_origin_z, MOUNT_CLEARANCE_M)])` as `fixed_joint_offset`.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py
git add Go2Pvcnn/scripts/build_m1_panda_asset.py Go2Pvcnn/tests/test_m1_panda_asset_static.py
git commit -m "feat: set M1 Panda mount clearance to zero"
```

### Task 2: Verify the parent mount plane

**Files:**
- Modify: `Go2Pvcnn/scripts/verify_m1_panda_asset.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- Modify: `Go2Pvcnn/tests/run_m1_panda_asset_pxr_behavior.py`

**Interfaces:**
- Produces: `_mount_plane_errors(parent_local_pos, expected_parent_local_pos, tolerance_m)` and JSON fields `mount_parent_local_pos`, `expected_mount_parent_local_pos`, `mount_plane_error_m`.

- [ ] **Step 1: Add a failing pure predicate test**

Extend `_load_verifier_contract_helpers()` with `_mount_plane_errors`, then add:

```python
def test_mount_plane_predicate_enforces_micrometer_tolerance():
    errors = _load_verifier_contract_helpers()["_mount_plane_errors"]
    assert errors((0.0, 0.0, 0.25), (0.0, 0.0, 0.25), 1e-6) == []
    assert errors((0.0, 0.0, 0.250002), (0.0, 0.0, 0.25), 1e-6)
    assert errors(None, (0.0, 0.0, 0.25), 1e-6)
```

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py -k mount_plane
```

Expected: helper lookup failure.

- [ ] **Step 3: Implement measurement and predicate**

Add `MOUNT_PLANE_TOLERANCE_M = 1e-6`, import `UsdGeom`, read `Joint.GetLocalPos0Attr()`, and add:

```python
def _mount_plane_errors(parent_local_pos, expected_parent_local_pos, tolerance_m):
    if parent_local_pos is None:
        return ["mount parent local position is unavailable"]
    error = max(abs(float(a) - float(b)) for a, b in zip(parent_local_pos, expected_parent_local_pos))
    return [] if error <= tolerance_m else [
        f"mount parent plane error {error} m exceeds {tolerance_m} m"
    ]
```

Open the independent `assets/m1_panda/m1_floating.usda`, compute the world BBox top and world origin of `/ZJ_V3_URDF_V1_0/BASE_LINK`, and set expected parent local position to `(0,0,top_z-origin_z)`. Do not compute the expected top from combined `/M1Panda/BASE_LINK`, because its composed Panda descendants would contaminate the bound. Compare that independent value with `Joint.GetLocalPos0Attr()`, call the predicate, and emit all three JSON fields. Mirror this independent-source calculation in `run_m1_panda_asset_pxr_behavior.py` and assert error `<=1e-6`.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py
git add Go2Pvcnn/scripts/verify_m1_panda_asset.py Go2Pvcnn/tests/test_m1_panda_asset_static.py Go2Pvcnn/tests/run_m1_panda_asset_pxr_behavior.py
git commit -m "test: verify zero-clearance mount plane"
```

### Task 3: Rebuild and checksum the asset

**Files:**
- Modify: `Go2Pvcnn/assets/m1_panda/m1_panda.usd`
- Modify if bytes change: `Go2Pvcnn/assets/m1_panda/panda/panda.usd`
- Modify: `Go2Pvcnn/assets/m1_panda/generated_files.sha256`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: generated zero-clearance USD plus exact SHA manifest.

- [ ] **Step 1: Record old hashes and run the reliable builder**

```bash
sha256sum assets/m1_panda/panda/panda.usd assets/m1_panda/m1_panda.usd
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 /home/xk/miniconda3/envs/go2/bin/python scripts/build_m1_panda_asset.py --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --headless
```

Expected: exit `0`, printing the combined USD path.

- [ ] **Step 2: Regenerate and verify checksums**

```bash
sha256sum assets/m1_panda/panda/panda.usd assets/m1_panda/m1_panda.usd > assets/m1_panda/generated_files.sha256
sha256sum -c assets/m1_panda/generated_files.sha256
```

Expected: `2/2 OK`; `source_files.sha256` remains unchanged.

- [ ] **Step 3: Inspect scope and commit generated files**

```bash
git diff --check
git status --short
git add assets/m1_panda/m1_panda.usd assets/m1_panda/generated_files.sha256
git commit -m "build: regenerate zero-clearance M1 Panda asset"
```

If `panda/panda.usd` changes, add it explicitly before committing; otherwise leave it untouched.

### Task 4: Pass topology, relocation, no-snap, and visual gates

**Files:** Modify implementation only after a reproduced failure and failing test.

- [ ] **Step 1: Run static and PXR checks**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_asset_static.py tests/test_m1_panda_smoke_cfg_static.py tests/test_m1_panda_wbc_env_static.py
OMNI_KIT_ACCEPT_EULA=Y /home/xk/miniconda3/envs/go2/bin/python tests/run_m1_panda_asset_pxr_behavior.py
```

Expected: all tests pass; PXR JSON reports exact root/fixed mount and `mount_plane_error_m<=1e-6`.

- [ ] **Step 2: Run CPU topology/no-snap verifier**

```bash
OMNI_KIT_ACCEPT_EULA=Y /home/xk/miniconda3/envs/go2/bin/python scripts/verify_m1_panda_asset.py --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda --device cpu --headless > /tmp/m1_panda_zero_clearance_verify.json
```

Expected: exit `0`, 25 DOF, one physics step, no validation errors, plane error `<=1e-6`, relative delta `<1e-4`.

- [ ] **Step 3: Verify relocation closure**

Use `mktemp -d`, copy the complete `assets/m1_panda` tree, rerun the verifier on that explicit copy, and delete only the temporary directory after success. Expected: no remote/outside/unresolved dependency except the allowed `OmniPBR.mdl` resolver boundary.

- [ ] **Step 4: Perform the GPU0 visual gate**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_wbc_roll_play.py --device cuda:0 --steps 0 --seed 42
```

Inspect front, side and rear mount views. Accept only an upright Panda touching the top plane with no visible penetration or jitter. Otherwise stop; never hide the defect by disabling more collisions or lowering Panda.

### Task 5: Re-establish C0 and C1a on GPU0

**Files:** Produce transient JSON only; runtime correction requires a new failing test.

- [ ] **Step 1: Run C0**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_wbc_play.py --headless --device cuda:0 --steps 2000 --seed 42 --stats-interval 500 --summary-json /tmp/m1_panda_zero_clearance_c0.json
```

Expected: exit `0`, finite, QP rate `1.0`, `TRACK: 2000`, no limit/base/self-collision/reset/snap event.

- [ ] **Step 2: Run C1a without Panda target motion**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_wbc_roll_play.py --headless --device cuda:0 --steps 4000 --seed 42 --disable-target-motion --stats-interval 400 --summary-json /tmp/m1_panda_zero_clearance_c1a_no_arm.json
```

Expected: exit `0`, `hard_gates_passed=true`.

- [ ] **Step 3: Run combined C1a**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_wbc_roll_play.py --headless --device cuda:0 --steps 4000 --seed 42 --stats-interval 400 --summary-json /tmp/m1_panda_zero_clearance_c1a_combined.json
```

Expected: exit `0`, five `800`-step phases and every existing hard gate true.

- [ ] **Step 4: Run the complete 12-file C0+C1a regression**

Use the exact test list in `docs/superpowers/runbooks/2026-08-18-m1-panda-wbc-teacher-c1a.md` with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Expected: all selected tests pass.

### Task 6: Record authority and unlock Student S1

**Files:**
- Create: `notes/log/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md`
- Create: `docs/superpowers/runbooks/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Consumes: exact hashes, commands, exit codes, verifier and GPU JSON.
- Produces: the only accepted asset/Teacher authority for Student S1.

- [ ] **Step 1: Write exact evidence and commands**

Record old/new hashes, builder commit, GPU/driver, each exit code, verifier JSON, visual decision, C0 metrics, and both C1a tables. State explicitly that a failed gate cannot unlock Student.

- [ ] **Step 2: Update T400 narrowly**

Mark T400.6 complete only after every asset/Teacher gate passes; open Student S1 implementation. Do not authorize random force, turning, PPO, grasping or real hardware.

- [ ] **Step 3: Verify and commit documentation**

```bash
git diff --check
rg -n "0\.0|mount_plane_error|25 DOF|4000|hard_gates_passed|Student" docs/superpowers/runbooks/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md notes/log/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md notes/todo/T400-m1-panda-force-aware-teacher-student.md
git add docs/superpowers/runbooks/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md notes/log/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md notes/log/index.md notes/todo/T400-m1-panda-force-aware-teacher-student.md
git commit -m "docs: accept zero-clearance M1 Panda Teacher baseline"
```

## Final Verification

- [ ] Re-run Task 4 checks, Task 5 static suite, `sha256sum -c`, and independent GPU JSON gate calculation at the final commit.
- [ ] Confirm the log contains the exact accepted `m1_panda.usd` SHA consumed by the Student plan.
- [ ] Run `git diff --check` and `git status --short`; only the pre-existing graph cache may remain.
