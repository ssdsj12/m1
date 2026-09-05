# Right O6 Mount Orientation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rotate the right O6 hand by 180 degrees about the Panda wrist's local X axis, preserve the composite articulation contracts, and prove that the right digits—not the palm base—make the intended box contact.

**Architecture:** Keep the asset builder as the sole source of mount calibration. Author the same side-specific quaternion into both the referenced hand's rest transform and its fixed-joint frame, validate the serialized USD independently, then expose geometric and physical evidence through the existing probe and four-layer verifier. Do not compensate for the asset defect in controllers.

**Tech Stack:** Python 3.11, pytest, Isaac Sim 5.1, Isaac Lab, USD/pxr, PhysX.

## Global constraints

- Left hand mount remains identity `(1, 0, 0, 0)`; right hand mount becomes local-X half-turn `(0, 1, 0, 0)` in WXYZ order.
- Preserve one articulation root, 43 active DOFs, 53 physical DOFs, and all existing DOF names/order.
- Do not add a training entrypoint or implement contact MPC, perception, or Residual.
- Do not modify the T400/T500 public contracts.
- Stop physical acceptance testing if asset verification, deterministic reset, finiteness, or joint-limit gates fail.
- Preserve unrelated dirty-worktree changes and diagnostic artifacts.

## Task 1: Make the side-specific hand mount an asset-builder invariant

**Files:**

- Modify: `tests/test_m1_dual_panda_o6_asset_static.py`
- Modify: `scripts/build_m1_dual_panda_o6_asset.py`
- Regenerate: `assets/m1_dual_panda_o6/m1_dual_panda_o6.usd`
- Regenerate: `assets/m1_dual_panda_o6/asset_manifest.json`

- [ ] **Step 1: Write the failing static contract test**

Add an AST-based assertion that the builder defines exactly:

```python
HAND_MOUNT_QUATERNION_WXYZ = {
    "left": (1.0, 0.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0, 0.0),
}
```

Also require the source tokens `mount_matrix * relative`, `local_rot0=mount_quaternion`, and `"hand_mounts"`. The test deliberately checks the multiplication order because `relative * mount_matrix` rotates the wrist translation.

- [ ] **Step 2: Run the static test and observe RED**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_dual_panda_o6_asset_static.py -q
```

Expected: failure because the builder still uses one identity `MOUNT_QUATERNION_WXYZ` for both hands.

- [ ] **Step 3: Generalize joint-frame authoring without changing existing callers**

Change `_set_joint_frames` to accept optional WXYZ rotations:

```python
def _set_joint_frames(
    joint: UsdPhysics.Joint,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_rot0: tuple[float, float, float, float] = IDENTITY_QUATERNION_WXYZ,
    local_rot1: tuple[float, float, float, float] = IDENTITY_QUATERNION_WXYZ,
) -> None:
    ...
```

Keep platform and Panda mounts on their current identity defaults. Define `IDENTITY_QUATERNION_WXYZ` separately from `HAND_MOUNT_QUATERNION_WXYZ` so hand chirality cannot silently affect other assembly joints.

- [ ] **Step 4: Apply the calibration consistently in `assemble_o6`**

Use the side-specific quaternion for both representations:

```python
mount_quaternion = HAND_MOUNT_QUATERNION_WXYZ[side]
mount_matrix = Gf.Matrix4d(1.0)
mount_matrix.SetRotate(
    Gf.Quatd(mount_quaternion[0], Gf.Vec3d(*mount_quaternion[1:]))
)
_set_matrix(hand, mount_matrix * relative)
_set_joint_frames(
    joint,
    wrist_body,
    f"{hand_path}/{side}_hand_base_link",
    (0.0, 0.0, 0.0),
    local_rot0=mount_quaternion,
)
```

Do not add a right-side branch outside the calibration mapping.

- [ ] **Step 5: Validate mount pose and fixed-joint agreement after serialized reopen**

Add small pure helpers that normalize WXYZ quaternions and compare `q` with both `expected` and `-expected` at `1e-7` tolerance. Extend `validate_stage_contract` to inspect, for each side:

1. `<side>_hand_mount_joint.physics:localRot0` equals the configured side quaternion, sign-invariant.
2. `physics:localRot1` remains identity.
3. Read both the hand-to-arm and wrist-to-arm matrices with `UsdGeom.XformCache().ComputeRelativeTransform(..., arm_prim)`; do not call it with `wrist_prim` as the ancestor because the hand and wrist are not in an ancestor/descendant relationship.
4. Recover the authored mount matrix with the algebra matching `hand_matrix = mount_matrix * wrist_matrix`, namely `measured_mount = hand_matrix * wrist_matrix.GetInverse()`. Its translation norm must be at most `1e-7 m`, and its rotation must equal the configured quaternion sign-invariantly.
5. The rest-transform and fixed-joint checks agree for the same side.

Return a `hand_mounts` structure from `validate_stage_contract` containing expected and measured quaternions plus `valid`. Fail `_require` on any invalid side so a serialized mismatch cannot produce a manifest.

Because USD matrix/quaternion conventions can differ, first print the measured reopened values in a one-off builder run if the test fails; adjust only the extraction convention, not the approved WXYZ calibration or multiplication order.

- [ ] **Step 6: Pin the calibration in the generated manifest**

Add:

```python
"hand_mounts": contract["hand_mounts"],
```

The manifest must record both expected and measured values, rather than copying the constants without measurement.

- [ ] **Step 7: Rebuild on GPU0 and rerun the focused test**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/build_m1_dual_panda_o6_asset.py \
  --headless --device cuda:0 \
  --asset-root assets/m1_dual_panda_o6

PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_dual_panda_o6_asset_static.py -q
```

Expected: build exits 0; static test passes; reopened contract reports left identity and right local-X half-turn.

- [ ] **Step 8: Commit only Task 1 files**

```bash
git add tests/test_m1_dual_panda_o6_asset_static.py \
  scripts/build_m1_dual_panda_o6_asset.py \
  assets/m1_dual_panda_o6/m1_dual_panda_o6.usd \
  assets/m1_dual_panda_o6/asset_manifest.json
git commit -m "fix: rotate right O6 wrist mount"
```

## Task 2: Add an independent serialized-asset gate

**Files:**

- Modify: `tests/test_m1_dual_panda_o6_asset_static.py`
- Modify: `scripts/verify_m1_dual_panda_o6_asset.py`
- Update: `assets/m1_dual_panda_o6/asset_manifest.json`

- [ ] **Step 1: Write a failing verifier contract test**

Require the verifier to contain `EXPECTED_HAND_MOUNT_QUATERNION_WXYZ`, `hand_mount_quaternion_wxyz`, `hand_mount_calibration_valid`, and a hard-gate reference to `offline["hand_mount_calibration_valid"]`.

- [ ] **Step 2: Run the test and observe RED**

Use the same focused pytest command as Task 1. Expected: failure because the verifier does not inspect mount calibration independently.

- [ ] **Step 3: Implement the offline mount report**

In `scripts/verify_m1_dual_panda_o6_asset.py`, define its own expected mapping rather than importing builder state. Add `_hand_mount_report(stage)` that reads both fixed-joint quaternions and the hand prim wrist-relative transforms for left/right, performs sign-invariant normalized comparison at `1e-7`, and returns:

```python
{
    "hand_mount_quaternion_wxyz": {"left": ..., "right": ...},
    "hand_mount_joint_quaternion_wxyz": {"left": ..., "right": ...},
    "hand_mount_calibration_valid": True,
}
```

Merge it into `_offline_report`; append a specific `offline_errors` item when invalid. Include `offline["hand_mount_calibration_valid"]` in `hard_gates_passed`.

- [ ] **Step 4: Run the focused static test and asset verification**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_dual_panda_o6_asset_static.py -q

PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/verify_m1_dual_panda_o6_asset.py \
  --headless --device cuda:0 --steps 2000 \
  --asset-root assets/m1_dual_panda_o6
```

Expected: `hard_gates_passed: true`, one articulation root, 43 active controls, 53 physical DOFs, zero nonfinite values, zero hard-limit events, and mount drift within existing tolerances. If any prerequisite gate fails, stop before Task 3 physical testing.

- [ ] **Step 5: Commit the independent gate**

```bash
git add tests/test_m1_dual_panda_o6_asset_static.py \
  scripts/verify_m1_dual_panda_o6_asset.py \
  assets/m1_dual_panda_o6/asset_manifest.json
git commit -m "test: verify O6 mount calibration"
```

## Task 3: Expose reset-time fingertip geometry in the existing probe

**Files:**

- Modify: `tests/test_m1_dual_panda_o6_entrypoints_static.py`
- Modify: `scripts/m1_dual_panda_o6_bimanual_probe.py`
- Generate: `tests/artifacts/m1_dual_panda_o6_right_mount_geometry_3.json`

- [ ] **Step 1: Write a failing probe contract test**

Add a static assertion for the exact report key `initial_fingertip_palm_offsets_m`.

- [ ] **Step 2: Run the test and observe RED**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_dual_panda_o6_entrypoints_static.py -q
```

- [ ] **Step 3: Record the geometry immediately after deterministic reset**

For each side, compute every fingertip position relative to that side's palm position from the same observation snapshot:

```python
offsets = hand.fingertip_positions_b - arm.palm_pose_b[..., :3].unsqueeze(-2)
```

Serialize only finite CPU lists under `initial_fingertip_palm_offsets_m`. Do not alter observations, control targets, state-machine behavior, or public wrapper contracts.

- [ ] **Step 4: Run a three-step GPU0 diagnostic**

Run the probe with its existing CLI arguments and fixed seed:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_bimanual_probe.py \
  --headless --device cuda:0 --steps 3 --seed 0 \
  --report tests/artifacts/m1_dual_panda_o6_right_mount_geometry_3.json
```

Expected evidence: startup/reset/nonfinite counters are zero and the right-hand maximum fingertip Y offset, expressed in the shared M1 base axes relative to the palm origin, is at least `0.05 m`; left/right patterns are chiral rather than identical. This is intentionally a translation-only base-frame diagnostic, not a claim that the offsets were rotated into palm-local axes.

- [ ] **Step 5: Run the existing GPU0 smoke layer**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py \
  --layer smoke --smoke-steps 200
```

Expected: exit 0 and updated SHA-pinned smoke report.

- [ ] **Step 6: Commit probe diagnostics**

```bash
git add tests/test_m1_dual_panda_o6_entrypoints_static.py \
  scripts/m1_dual_panda_o6_bimanual_probe.py \
  tests/artifacts/m1_dual_panda_o6_right_mount_geometry_3.json
git commit -m "test: report initial O6 fingertip geometry"
```

## Task 4: Prove right-digit PRELOAD contact and run regressions

**Files:**

- Update only existing verification artifacts produced by the approved commands.
- Do not change control, box, contact-filter, or MPC code in response to this acceptance step.

- [ ] **Step 1: Run the fixed-seed PRELOAD acceptance trial**

Run the existing probe for 1600 physics steps with seed 0:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_bimanual_probe.py \
  --headless --device cuda:0 --steps 1600 --seed 0 \
  --report tests/artifacts/m1_dual_panda_o6_right_mount_preload_1600.json
```

- [ ] **Step 2: Evaluate physical acceptance atomically**

All conditions must hold in the same report:

- the maximum right-hand box-contact body is a right proximal/distal digit link, not `right_hand_base_link`;
- selected right digit force is greater than zero;
- maximum consecutive bilateral contact steps is at least 20;
- deterministic reset, nonfinite, hard-joint-limit, and hard-failure counts are zero;
- asset/source SHA fields match the rebuilt USD and current source manifest.

If any condition fails, stop and report the measured link/force/phase evidence. Do not mask the failure with controller compensation, a wider box, collision-filter edits, contact MPC, or looser safety thresholds.

- [ ] **Step 3: Run CPU and pure-QP verification layers**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer cpu

PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer qp
```

Expected: deterministic sampling, continuity, shape/dtype/device, Hand MPC constraints, mimic, fallback, atomic rejection, Arm/Hand/WBC feasibility, limits, repeatability, and safe-state non-contamination all pass.

- [ ] **Step 4: Run the complete scoped test suite**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_frame_kinematics.py \
  tests/test_m1_bimanual_full_action_teacher.py \
  tests/test_m1_bimanual_o6_contact.py \
  tests/test_m1_bimanual_object_mpc.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_bimanual_safety_projection.py \
  tests/test_m1_bimanual_state_machine.py \
  tests/test_m1_dual_panda_o6_asset_static.py \
  tests/test_m1_dual_panda_o6_contracts.py \
  tests/test_m1_dual_panda_o6_entrypoints_static.py \
  tests/test_m1_dual_panda_o6_env_static.py \
  tests/test_m1_dual_panda_o6_verification.py -q
```

Expected: all scoped tests pass with no regressions.

- [ ] **Step 5: Review scope before claiming completion**

```bash
git status --short
git diff --check
git diff --stat
```

Verify that no training entrypoint, contact MPC, perception, Residual, or T400/T500 public-contract change was introduced. Preserve all unrelated pre-existing changes.

- [ ] **Step 6: Commit accepted physical evidence**

Stage only the new acceptance report and any aggregate smoke artifact intentionally refreshed by the verification commands:

```bash
git add tests/artifacts/m1_dual_panda_o6_right_mount_preload_1600.json
git commit -m "test: validate right O6 digit preload contact"
```

Do not run the formal 30-trial GPU0 layer until this physical acceptance passes. Once it passes, the next separately approved milestone is the existing formal command that must produce 30 passing JSONL rows and a SHA-pinned aggregate manifest.
