# M1 Dual-Panda O6 Bimanual MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated M1 + shared-yaw platform + dual Panda + dual O6 Isaac Lab task that deterministically grasps, lifts, holds, lowers, and releases the fixed 0.5 kg box through hierarchical MPC and whole-body QP control.

**Architecture:** A project-owned single-articulation asset exposes 43 active control channels. A timestamped atomic snapshot feeds 25 Hz object MPC, two 50 Hz arm MPC instances, two 100 Hz O6 hand MPC instances, and a 200 Hz WBC/QP; a deterministic state machine owns phase transitions and atomic fallback.

**Tech Stack:** Python 3.11, PyTorch float64 pure-control kernels, Isaac Sim 5.1/Isaac Lab, USD/PXR, Gymnasium, pytest, existing `go2_pvcnn.control.m1_panda_coordination` QP and Arm MPC primitives.

## Global Constraints

- Preserve all existing M1/Panda assets, Gym IDs, checkpoint shapes, and default behavior.
- Keep source O6 files under `/home/xk/coding/o6asset` read-only; generated project assets live under `Go2Pvcnn/assets/m1_dual_panda_o6/`.
- Normalize only the left O6 entry-layer location; do not regenerate its base/physics/robot/sensor layers.
- Freeze the active order at 16 M1 + 1 platform + 14 Panda + 12 O6 = 43 channels.
- Treat 53 as an expected runtime physical-DOF count, then record the measured PXR/Isaac value in the manifest; never force a wrong count.
- Use 25/50/100/200 Hz for object/arm/hand/WBC control and fixed-condition simulator truth.
- Do not add PPO/RL, vision, randomization, rolling-base manipulation, handover, or real-hardware code.
- Keep the 0.5 kg box, 0.10 m lift, 3 s hold, and seeds 42/43/44 × 10 trials acceptance unchanged.
- Every pure control API validates exact shape, `torch.float64`, device, finite values, and monotonic timestamp before returning output.
- Each task follows RED → minimal GREEN → focused regression → commit; generated USD and runtime artifacts are committed only when the repository's existing asset policy and checksum manifest require them.

## Locked File Structure

```text
Go2Pvcnn/go2_pvcnn/assets/m1_dual_panda_o6.py
Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/
  __init__.py
  contracts.py
  object_mpc.py
  dual_arm_mpc.py
  hand_mpc.py
  constraints.py
  whole_body_qp.py
  state_machine.py
  runtime.py
Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py
Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py
Go2Pvcnn/scripts/normalize_o6_assets.py
Go2Pvcnn/scripts/build_m1_dual_panda_o6_asset.py
Go2Pvcnn/scripts/verify_m1_dual_panda_o6_asset.py
Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py
Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py
Go2Pvcnn/tests/test_m1_dual_panda_o6_sources.py
Go2Pvcnn/tests/test_m1_dual_panda_o6_asset_static.py
Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py
Go2Pvcnn/tests/test_m1_bimanual_object_mpc.py
Go2Pvcnn/tests/test_m1_bimanual_dual_arm_mpc.py
Go2Pvcnn/tests/test_m1_bimanual_hand_mpc.py
Go2Pvcnn/tests/test_m1_bimanual_whole_body_qp.py
Go2Pvcnn/tests/test_m1_bimanual_state_machine.py
Go2Pvcnn/tests/test_m1_bimanual_runtime.py
Go2Pvcnn/tests/test_m1_dual_panda_o6_env_static.py
Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py
```

---

### Task 1: Close and Normalize O6 Source Assets

**Files:**
- Create: `Go2Pvcnn/scripts/normalize_o6_assets.py`
- Create: `Go2Pvcnn/tests/test_m1_dual_panda_o6_sources.py`
- Create at runtime: `Go2Pvcnn/assets/m1_dual_panda_o6/o6_left/`
- Create at runtime: `Go2Pvcnn/assets/m1_dual_panda_o6/o6_right/`
- Create at runtime: `Go2Pvcnn/assets/m1_dual_panda_o6/source_manifest.json`

**Interfaces:**
- Consumes: `normalize_o6_sources(source_root: Path, destination_root: Path) -> dict[str, object]`.
- Produces: local `o6_left/O6_left.usd`, `o6_right/O6_right.usd`, their `configuration/` layers, STL files, and SHA256 manifest entries with no absolute authored dependencies.

- [ ] **Step 1: Write failing source-closure tests**

```python
def test_normalizer_places_both_entries_above_configuration(tmp_path):
    manifest = normalize_o6_sources(O6_SOURCE_ROOT, tmp_path)
    assert (tmp_path / "o6_left/O6_left.usd").is_file()
    assert (tmp_path / "o6_right/O6_right.usd").is_file()
    assert manifest["left"]["entry"] == "o6_left/O6_left.usd"
    assert manifest["right"]["entry"] == "o6_right/O6_right.usd"

def test_normalizer_is_lossless_and_rejects_missing_layers(tmp_path):
    manifest = normalize_o6_sources(O6_SOURCE_ROOT, tmp_path)
    assert set(manifest["left"]["layers"]) == {"base", "physics", "robot", "sensor"}
    assert all(len(value) == 64 for value in manifest["sha256"].values())
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_sources.py`

Expected: collection fails because `scripts.normalize_o6_assets` does not exist.

- [ ] **Step 3: Implement deterministic normalization**

```python
def normalize_o6_sources(source_root: Path, destination_root: Path) -> dict[str, object]:
    source_root = source_root.resolve(strict=True)
    destination_root.mkdir(parents=True, exist_ok=True)
    sides = {
        "left": source_root / "1、O6，urdf/linkerhand_O6_left.urdf/linkerhand_O6_left.urdf",
        "right": source_root / "1、O6，urdf/linkerhand_O6_right.urdf/linkerhand_O6_right.urdf",
    }
    result: dict[str, object] = {"schema": 1, "sha256": {}}
    for side, root in sides.items():
        entry = root / ("configuration/O6_left.usd" if side == "left" else "O6_right.usd")
        required = {name: root / f"configuration/linkerhand_O6_{side}.urdf_{name}.usd"
                    for name in ("base", "physics", "robot", "sensor")}
        for path in (entry, *required.values()):
            if not path.is_file():
                raise FileNotFoundError(path)
        side_dest = destination_root / f"o6_{side}"
        shutil.copytree(root / "configuration", side_dest / "configuration", dirs_exist_ok=True)
        shutil.copy2(entry, side_dest / f"O6_{side}.usd")
        result[side] = {"entry": f"o6_{side}/O6_{side}.usd", "layers": list(required)}
    return write_sha_manifest(destination_root, result)
```

Copy `meshes/` from the corresponding parent source directory into each local O6 directory, hash every copied file, write JSON atomically, and reject symlinks or authored `http://`, `https://`, `omniverse://`, and `/home/` asset paths.

- [ ] **Step 4: Run focused tests and materialize the project-owned sources**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_sources.py`

Expected: all tests pass.

Run: `cd Go2Pvcnn && python scripts/normalize_o6_assets.py --source-root /home/xk/coding/o6asset --destination-root assets/m1_dual_panda_o6`

Expected: JSON summary reports both sides, all four layers per side, and zero missing dependencies.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/normalize_o6_assets.py Go2Pvcnn/tests/test_m1_dual_panda_o6_sources.py Go2Pvcnn/assets/m1_dual_panda_o6
git commit -m "feat: normalize project-owned O6 assets"
```

### Task 2: Build the Single-Articulation Mechanical Asset

**Files:**
- Create: `Go2Pvcnn/scripts/build_m1_dual_panda_o6_asset.py`
- Create: `Go2Pvcnn/tests/test_m1_dual_panda_o6_asset_static.py`
- Create at runtime: `Go2Pvcnn/assets/m1_dual_panda_o6/platform.usda`
- Create at runtime: `Go2Pvcnn/assets/m1_dual_panda_o6/m1_dual_panda_o6.usd`
- Create at runtime: `Go2Pvcnn/assets/m1_dual_panda_o6/asset_manifest.json`

**Interfaces:**
- Consumes: normalized O6 manifest, existing project M1 source, and existing Panda URDF source.
- Produces: `build_asset(asset_root: Path, force_panda_conversion: bool = False) -> Path` and one articulation rooted at `/M1DualPandaO6/BASE_LINK`.

- [ ] **Step 1: Write static builder-contract tests**

```python
def test_builder_freezes_namespaces_and_mount_contract():
    source = BUILDER.read_text()
    assert 'ROOT_PRIM = "/M1DualPandaO6"' in source
    assert 'PLATFORM_JOINT_NAME = "dual_arm_platform_yaw_joint"' in source
    assert 'LEFT_ARM_PRIM = f"{ROOT_PRIM}/left_arm"' in source
    assert 'RIGHT_ARM_PRIM = f"{ROOT_PRIM}/right_arm"' in source
    assert 'EXPECTED_ACTIVE_DOF_COUNT = 43' in source

def test_builder_never_edits_external_o6_sources():
    tree = ast.parse(BUILDER.read_text())
    assert "source_o6_root" not in {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call) and "write" in ast.unparse(node)}
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_asset_static.py`

Expected: FAIL because the builder is absent.

- [ ] **Step 3: Implement the builder with explicit assembly phases**

```python
@dataclass(frozen=True)
class MountTransform:
    translation: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

LEFT_ARM_MOUNT = MountTransform((0.0, 0.20, 0.0), (1.0, 0.0, 0.0, 0.0))
RIGHT_ARM_MOUNT = MountTransform((0.0, -0.20, 0.0), (1.0, 0.0, 0.0, 0.0))

def build_asset(asset_root: Path, force_panda_conversion: bool = False) -> Path:
    validate_source_manifests(asset_root)
    ensure_arm_only_panda(asset_root, force=force_panda_conversion)
    stage = create_m1_and_platform_stage(asset_root)
    author_platform_revolute_joint(stage, lower=-math.pi / 2, upper=math.pi / 2)
    assemble_panda(stage, side="left", mount=LEFT_ARM_MOUNT)
    assemble_panda(stage, side="right", mount=RIGHT_ARM_MOUNT)
    assemble_o6(stage, side="left", wrist_body="left_panda_link8")
    assemble_o6(stage, side="right", wrist_body="right_panda_link8")
    remove_child_roots_scenes_and_root_joints(stage)
    author_convex_collision_approximations(stage)
    validate_stage_contract(stage)
    return export_reopen_validate_and_manifest(stage, asset_root)
```

Use the existing `build_m1_panda_asset.py` RobotAssembler compatibility branch, but give every duplicated Panda/O6 joint and body a deterministic side prefix. Convert Panda from an arm-only URDF or remove the two-finger branch before O6 assembly; the final stage must not contain `panda_finger_joint.*`.

- [ ] **Step 4: Run static tests, build, and reopen**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_asset_static.py`

Expected: all tests pass.

Run: `CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/build_m1_dual_panda_o6_asset.py --asset-root Go2Pvcnn/assets/m1_dual_panda_o6 --headless`

Expected: exit 0; exported file reopens; exactly one articulation root and five assembly joints are reported (base→platform, platform→two arms, two wrists→two hands).

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/build_m1_dual_panda_o6_asset.py Go2Pvcnn/tests/test_m1_dual_panda_o6_asset_static.py Go2Pvcnn/assets/m1_dual_panda_o6
git commit -m "feat: build M1 dual-arm O6 articulation"
```

### Task 3: Verify Asset Topology and Long-Hold Physics

**Files:**
- Create: `Go2Pvcnn/scripts/verify_m1_dual_panda_o6_asset.py`
- Modify: `Go2Pvcnn/tests/test_m1_dual_panda_o6_asset_static.py`

**Interfaces:**
- Consumes: combined USD and asset manifest.
- Produces: one JSON object containing dependency closure, roots, measured DOF/body/joint names, mount drift, contact/limit/reset counts, and `hard_gates_passed`.

- [ ] **Step 1: Add failing verifier tests**

```python
def test_verifier_requires_runtime_measured_dof_and_2000_steps():
    source = VERIFIER.read_text()
    assert 'EXPECTED_ACTIVE_DOF_COUNT = 43' in source
    assert 'EXPECTED_PHYSICS_STEPS = 2000' in source
    assert '"measured_physical_dof_count"' in source
    assert '"hard_gates_passed"' in source
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_asset_static.py`

Expected: verifier-related assertions fail.

- [ ] **Step 3: Implement offline and runtime gates**

```python
def summarize_runtime(robot, metrics: HoldMetrics, manifest: dict) -> dict:
    active_names = tuple(name for name in robot.joint_names if is_active_control_joint(name))
    return {
        "measured_physical_dof_count": int(robot.num_joints),
        "active_control_count": len(active_names),
        "single_articulation_root": metrics.articulation_root_count == 1,
        "max_mount_position_drift_m": metrics.max_mount_position_drift_m,
        "max_mount_orientation_drift_rad": metrics.max_mount_orientation_drift_rad,
        "nonfinite_count": metrics.nonfinite_count,
        "hard_gates_passed": (
            len(active_names) == 43
            and metrics.physics_steps == 2000
            and metrics.articulation_root_count == 1
            and metrics.nonfinite_count == 0
            and metrics.hard_joint_limit_count == 0
            and metrics.unexpected_reset_count == 0
        ),
    }
```

Update `asset_manifest.json` only after successful reopen and runtime initialization, recording the measured DOF count instead of asserting 53.

- [ ] **Step 4: Verify static and real physics gates**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_asset_static.py`

Expected: all tests pass.

Run: `CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/verify_m1_dual_panda_o6_asset.py --asset-root Go2Pvcnn/assets/m1_dual_panda_o6 --steps 2000 --headless`

Expected: exit 0; `active_control_count=43`, `single_articulation_root=true`, `hard_gates_passed=true`; measured DOF is finite and persisted.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/verify_m1_dual_panda_o6_asset.py Go2Pvcnn/tests/test_m1_dual_panda_o6_asset_static.py Go2Pvcnn/assets/m1_dual_panda_o6/asset_manifest.json
git commit -m "test: verify dual-arm O6 asset physics"
```

### Task 4: Freeze Asset Configuration and 43-Channel Ordering

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/assets/m1_dual_panda_o6.py`
- Modify: `Go2Pvcnn/go2_pvcnn/assets/__init__.py`
- Create: `Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py`

**Interfaces:**
- Produces: `M1_DUAL_PANDA_O6_CFG`, `M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES`, body-name constants, O6 mimic map, and `M1_DUAL_PANDA_O6_ACTIVE_DOF_COUNT = 43`.

- [ ] **Step 1: Write exact ordering tests**

```python
def test_active_order_is_16_plus_1_plus_14_plus_12():
    names = M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES
    assert len(names) == 43
    assert names[:16] == M1_BASE_ACTIVE_JOINT_NAMES
    assert names[16] == "dual_arm_platform_yaw_joint"
    assert names[17:24] == tuple(f"left_panda_joint{i}" for i in range(1, 8))
    assert names[24:31] == tuple(f"right_panda_joint{i}" for i in range(1, 8))
    assert names[31:37] == LEFT_O6_ACTIVE_JOINT_NAMES
    assert names[37:43] == RIGHT_O6_ACTIVE_JOINT_NAMES
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_contracts.py`

Expected: import fails because the asset cfg is absent.

- [ ] **Step 3: Implement isolated actuator configuration**

Define separate actuator groups for M1 legs/wheels, platform, left/right Panda shoulder/forearm, and left/right O6. Set the platform soft limit to ±π/2 and velocity limit `0.25 rad/s`. Set O6 velocities from the approved vendor values and conservative position-control effort/stiffness values; document that these are simulator controller limits, not URDF `100` effort claims.

```python
M1_DUAL_PANDA_O6_CFG = M1_CFG.copy()
M1_DUAL_PANDA_O6_CFG.spawn = M1_DUAL_PANDA_O6_CFG.spawn.replace(usd_path=USD_PATH)
M1_BASE_ACTIVE_JOINT_NAMES = (
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT",
    "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT",
)
M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES = (
    *M1_BASE_ACTIVE_JOINT_NAMES,
    "dual_arm_platform_yaw_joint",
    *(f"left_panda_joint{i}" for i in range(1, 8)),
    *(f"right_panda_joint{i}" for i in range(1, 8)),
    *LEFT_O6_ACTIVE_JOINT_NAMES,
    *RIGHT_O6_ACTIVE_JOINT_NAMES,
)
```

- [ ] **Step 4: Run focused and legacy cfg tests**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_contracts.py tests/test_m1_panda_asset_static.py tests/test_m1_panda_wbc_contracts.py`

Expected: all tests pass and legacy constants remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/assets/m1_dual_panda_o6.py Go2Pvcnn/go2_pvcnn/assets/__init__.py Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py
git commit -m "feat: define dual-arm O6 asset contract"
```

### Task 5: Add Atomic Bimanual State and Command Contracts

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contracts.py`
- Extend: `Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py`

**Interfaces:**
- Produces: `BimanualSnapshot`, `SideArmState`, `SideHandState`, `BoxState`, `BimanualCommand`, and `validate_monotonic_snapshot(previous, current)`.

- [ ] **Step 1: Write strict tensor and timestamp tests**

```python
def test_snapshot_rejects_mixed_timestamp_and_nonfinite_hand_state():
    valid = make_snapshot(timestamp_ns=10)
    with pytest.raises(ValueError, match="monotonic"):
        validate_monotonic_snapshot(valid, replace(valid, timestamp_ns=10))
    bad = replace(valid.left_hand, q=torch.full((6,), float("nan"), dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        replace(valid, left_hand=bad)

def test_command_has_exact_43_channel_order():
    command = make_zero_command()
    assert command.effort.shape == (43,)
    assert command.timestamp_ns > 0
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_contracts.py`

Expected: missing contract imports fail.

- [ ] **Step 3: Implement frozen dataclasses**

```python
@dataclass(frozen=True)
class SideArmState:
    q: torch.Tensor                 # (7,)
    qd: torch.Tensor                # (7,)
    palm_pose_b: torch.Tensor       # (6,)
    palm_twist_b: torch.Tensor      # (6,)
    jacobian_b: torch.Tensor        # (6, 7)
    mass_matrix: torch.Tensor       # (7, 7)
    bias: torch.Tensor              # (7,)

@dataclass(frozen=True)
class SideHandState:
    q: torch.Tensor                 # (6,)
    qd: torch.Tensor                # (6,)
    fingertip_forces_b: torch.Tensor  # (5, 3)
    fingertip_positions_b: torch.Tensor  # (5, 3)
    contact_mask: torch.Tensor      # (5,) bool

@dataclass(frozen=True)
class BoxState:
    pose_b: torch.Tensor            # (6,)
    twist_b: torch.Tensor           # (6,)
    mass: torch.Tensor              # scalar
    inertia_b: torch.Tensor         # (3, 3)
    supported: bool

@dataclass(frozen=True)
class BimanualSnapshot:
    timestamp_ns: int
    base_state: torch.Tensor        # (13,)
    m1_q: torch.Tensor              # (16,)
    m1_qd: torch.Tensor             # (16,)
    platform_q_qd: torch.Tensor     # (2,)
    left_arm: SideArmState
    right_arm: SideArmState
    left_hand: SideHandState
    right_hand: SideHandState
    box: BoxState

@dataclass(frozen=True)
class BimanualCommand:
    timestamp_ns: int
    effort: torch.Tensor            # (43,)
    feasible: bool
    fallback_reasons: tuple[str, ...]

class BimanualPhase(Enum):
    APPROACH = auto(); PRELOAD = auto(); GRASP = auto(); LIFT = auto()
    HOLD = auto(); LOWER = auto(); RELEASE = auto(); DONE = auto()
    HOLD_SAFE = auto(); LOWER_SAFE = auto(); SAFE_RELEASE = auto(); TERMINATED = auto()
```

Use the existing `require_tensor` style and exact CPU float64 contracts for pure control. Boolean contact masks are exact CPU bool. Clone tensors on construction so caller mutation cannot alter an accepted snapshot.

- [ ] **Step 4: Run contract tests**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_contracts.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py
git commit -m "feat: add atomic bimanual control contracts"
```

### Task 6: Implement the 25 Hz Bimanual Object MPC

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_object_mpc.py`

**Interfaces:**
- Consumes: `ObjectMpcInput(snapshot, target_box_pose_b, phase, previous_solution)`.
- Produces: `ObjectMpcSolution(box_pose, box_twist, platform_yaw, left_palm_pose, right_palm_pose, left_wrench, right_wrench, diagnostics)` over 25 nodes / 1.0 s.

- [ ] **Step 1: Write equilibrium, friction, mirror, and fallback tests**

```python
def test_static_box_solution_balances_gravity_and_is_mirrored():
    solution = BimanualObjectMpc().plan(make_centered_hold_input())
    assert solution.diagnostics.feasible
    total_force = solution.left_wrench[0, :3] + solution.right_wrench[0, :3]
    assert total_force[2].item() == pytest.approx(0.5 * 9.81, abs=1e-5)
    assert solution.left_palm_pose[0, 1].item() == pytest.approx(-solution.right_palm_pose[0, 1].item())

def test_infeasible_target_returns_last_safe_without_lift_progress():
    planner = BimanualObjectMpc()
    safe = planner.plan(make_centered_hold_input())
    fallback = planner.plan(make_unreachable_input(previous=safe))
    assert fallback.diagnostics.fallback_used
    assert torch.equal(fallback.left_palm_pose, safe.left_palm_pose)
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_object_mpc.py`

Expected: missing module import failure.

- [ ] **Step 3: Implement condensed rigid-body QP**

```python
@dataclass(frozen=True)
class ObjectMpcCfg:
    dt: float = 0.04
    horizon_steps: int = 25
    box_pose_weight: float = 2000.0
    box_twist_weight: float = 100.0
    wrench_slew_weight: float = 0.1
    platform_motion_weight: float = 10.0
    friction_coefficient: float = 0.8
    per_hand_normal_force_min: float = 8.0
    per_hand_normal_force_max: float = 15.0

@dataclass(frozen=True)
class ObjectMpcDiagnostics:
    feasible: bool
    fallback_used: bool
    fallback_reason: str | None
    force_closure_margin: float
    saturation_fraction: float

@dataclass(frozen=True)
class ObjectMpcInput:
    snapshot: BimanualSnapshot
    target_box_pose_b: torch.Tensor       # (25, 6)
    phase: BimanualPhase
    previous_solution: ObjectMpcSolution | None

@dataclass(frozen=True)
class ObjectMpcSolution:
    box_pose: torch.Tensor                # (25, 6)
    box_twist: torch.Tensor               # (25, 6)
    platform_yaw: torch.Tensor            # (25,)
    left_palm_pose: torch.Tensor          # (25, 6)
    right_palm_pose: torch.Tensor         # (25, 6)
    left_wrench: torch.Tensor             # (25, 6)
    right_wrench: torch.Tensor            # (25, 6)
    diagnostics: ObjectMpcDiagnostics

class BimanualObjectMpc:
    def plan(self, sample: ObjectMpcInput) -> ObjectMpcSolution:
        problem = build_object_qp(sample, self.cfg)
        result = solve_reference_qp(problem, tolerance=1e-7, max_iterations=512)
        return self._accept_or_fallback(sample, result)
```

Linearize SO(3) error in the canonical base frame, use four-sided friction pyramids, constrain yaw to ±π/2 and `0.25 rad/s`, and expose separate feasibility, saturation, force-closure margin, and fallback reason.

- [ ] **Step 4: Run focused and QP regression tests**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_object_mpc.py tests/test_m1_panda_qp_backend.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py Go2Pvcnn/tests/test_m1_bimanual_object_mpc.py
git commit -m "feat: add bimanual object MPC"
```

### Task 7: Coordinate Two Existing 50 Hz Arm MPC Instances

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dual_arm_mpc.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_dual_arm_mpc.py`

**Interfaces:**
- Consumes: `DualArmMpcInput(left: ArmMpcInput, right: ArmMpcInput, object_solution: ObjectMpcSolution)`.
- Produces: `DualArmMpcSolution(left: ArmMpcSolution, right: ArmMpcSolution, both_feasible: bool, synchronized_fallback: bool)`.

- [ ] **Step 1: Write synchronization and legacy-compatibility tests**

```python
def test_one_side_failure_holds_both_last_safe_solutions():
    coordinator = DualArmMpcCoordinator(planner_factory=FakePlannerFactory())
    first = coordinator.plan(make_dual_arm_input(feasible=(True, True)))
    second = coordinator.plan(make_dual_arm_input(feasible=(True, False)))
    assert second.synchronized_fallback
    assert torch.equal(second.left.q_ref, first.left.q_ref)
    assert torch.equal(second.right.q_ref, first.right.q_ref)
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_dual_arm_mpc.py tests/test_m1_panda_arm_mpc.py`

Expected: only the new module tests fail; legacy Arm MPC tests pass.

- [ ] **Step 3: Implement the coordinator without changing `ArmMpcInput`**

```python
@dataclass(frozen=True)
class DualArmMpcInput:
    left: ArmMpcInput
    right: ArmMpcInput
    object_solution: ObjectMpcSolution

@dataclass(frozen=True)
class DualArmMpcSolution:
    left: ArmMpcSolution
    right: ArmMpcSolution
    both_feasible: bool
    synchronized_fallback: bool

class DualArmMpcCoordinator:
    def __init__(self, planner_factory=LinearizedArmMpc):
        self.left = planner_factory()
        self.right = planner_factory()
        self._last_safe: DualArmMpcSolution | None = None

    def plan(self, sample: DualArmMpcInput) -> DualArmMpcSolution:
        left = self.left.plan(sample.left)
        right = self.right.plan(sample.right)
        if left.diagnostics.feasible and right.diagnostics.feasible:
            solution = DualArmMpcSolution(left, right, True, False)
            self._last_safe = clone_dual_solution(solution)
            return solution
        return synchronized_hold(sample, self._last_safe, left, right)
```

Do not modify `m1_panda_coordination/arm_mpc.py`; the new coordinator composes its public classes exactly as they are.

- [ ] **Step 4: Run new and legacy tests**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_dual_arm_mpc.py tests/test_m1_panda_arm_mpc.py tests/test_m1_panda_arm_mpc_runtime.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dual_arm_mpc.py Go2Pvcnn/tests/test_m1_bimanual_dual_arm_mpc.py
git commit -m "feat: coordinate dual Panda arm MPC"
```

### Task 8: Implement Two 100 Hz O6 Hand MPC Controllers

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/hand_mpc.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_hand_mpc.py`

**Interfaces:**
- Consumes: `HandMpcInput(q, qd, fingertip_forces_b, contact_mask, contact_jacobian, wrench_map, target_wrench_b, q_min, q_max, qd_max)`.
- Produces: `HandMpcSolution(q_ref, qd_ref, predicted_forces_b, predicted_wrench_b, diagnostics)` for six active axes.

- [ ] **Step 1: Write force-tracking, velocity, mimic-exclusion, and fallback tests**

```python
def test_hand_mpc_tracks_wrench_with_six_active_axes_only():
    solution = O6HandMpc().plan(make_grasp_input())
    assert solution.q_ref.shape == (6,)
    assert solution.predicted_forces_b.shape == (5, 3)
    assert torch.all(solution.qd_ref.abs() <= make_grasp_input().qd_max + 1e-12)

def test_mimic_joint_names_are_rejected_from_active_input():
    with pytest.raises(ValueError, match="six active"):
        HandMpcInput(q=torch.zeros(11, dtype=torch.float64), **other_fields())
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_hand_mpc.py`

Expected: missing module import failure.

- [ ] **Step 3: Implement a linear compliance/contact QP**

```python
@dataclass(frozen=True)
class HandMpcCfg:
    dt: float = 0.01
    horizon_steps: int = 20
    wrench_weight: float = 500.0
    force_slew_weight: float = 1.0
    joint_rate_weight: float = 0.1
    normal_force_max: float = 10.0
    friction_coefficient: float = 0.8

@dataclass(frozen=True)
class HandMpcDiagnostics:
    feasible: bool
    fallback_used: bool
    fallback_reason: str | None
    wrench_error_norm: float
    slip_margin: float

@dataclass(frozen=True)
class HandMpcInput:
    q: torch.Tensor                       # (6,)
    qd: torch.Tensor                      # (6,)
    fingertip_forces_b: torch.Tensor      # (5, 3)
    contact_mask: torch.Tensor            # (5,) bool
    contact_jacobian: torch.Tensor        # (15, 6)
    wrench_map: torch.Tensor              # (6, 15)
    target_wrench_b: torch.Tensor         # (6,)
    q_min: torch.Tensor                   # (6,)
    q_max: torch.Tensor                   # (6,)
    qd_max: torch.Tensor                  # (6,)

@dataclass(frozen=True)
class HandMpcSolution:
    q_ref: torch.Tensor                   # (6,)
    qd_ref: torch.Tensor                  # (6,)
    predicted_forces_b: torch.Tensor      # (5, 3)
    predicted_wrench_b: torch.Tensor      # (6,)
    diagnostics: HandMpcDiagnostics

class O6HandMpc:
    def plan(self, sample: HandMpcInput) -> HandMpcSolution:
        problem = build_hand_contact_qp(sample, self.cfg)
        result = solve_reference_qp(problem, tolerance=1e-7, max_iterations=256)
        return self._accept_or_hold(sample, result)
```

Model only five fingertip contact forces; the palm pose/wrench remains an Arm/Object MPC responsibility. Encode thumb bend/yaw and four finger bend speeds with side-independent limits; mirror geometry through the supplied Jacobian and wrench map, not by changing control order.

- [ ] **Step 4: Run hand, contract, and QP tests**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_hand_mpc.py tests/test_m1_dual_panda_o6_contracts.py tests/test_m1_panda_qp_backend.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/hand_mpc.py Go2Pvcnn/tests/test_m1_bimanual_hand_mpc.py
git commit -m "feat: add O6 contact hand MPC"
```

### Task 9: Add Bimanual Constraints and Atomic Whole-Body QP

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/constraints.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/whole_body_qp.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_whole_body_qp.py`

**Interfaces:**
- Consumes: one `BimanualSnapshot`, accepted object/arm/hand solutions, collision distances/Jacobians, and contact state.
- Produces: `BimanualWbcSolution(effort: Tensor[43], feasible, fallback_used, diagnostics)` and never writes to Isaac directly.

- [ ] **Step 1: Write 43-channel, collision, and atomic-failure tests**

```python
def test_wbc_returns_exact_43_efforts_and_zero_wheels_for_stationary_task():
    solution = BimanualWholeBodyQp().solve(make_safe_request())
    assert solution.effort.shape == (43,)
    assert torch.allclose(solution.effort[12:16], torch.zeros(4, dtype=torch.float64))

def test_one_invalid_subsolution_rejects_the_entire_new_command():
    controller = BimanualWholeBodyQp()
    safe = controller.solve(make_safe_request())
    bad = controller.solve(make_request(left_hand_feasible=False))
    assert bad.fallback_used
    assert torch.equal(bad.effort, safe.effort)
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_whole_body_qp.py`

Expected: missing modules fail collection.

- [ ] **Step 3: Implement explicit constraint assembly and last-safe fallback**

```python
@dataclass(frozen=True)
class BimanualConstraintSet:
    lower_effort: torch.Tensor   # (43,)
    upper_effort: torch.Tensor   # (43,)
    inequality_matrix: torch.Tensor
    inequality_upper: torch.Tensor
    min_collision_distance: float
    force_closure_margin: float

@dataclass(frozen=True)
class BimanualWbcDiagnostics:
    qp_iterations: int
    min_collision_distance: float
    force_closure_margin: float
    support_margin: float
    fallback_reason: str | None

@dataclass(frozen=True)
class BimanualWbcRequest:
    snapshot: BimanualSnapshot
    object_solution: ObjectMpcSolution
    arm_solution: DualArmMpcSolution
    left_hand_solution: HandMpcSolution
    right_hand_solution: HandMpcSolution
    collision_distances: torch.Tensor
    collision_jacobian: torch.Tensor

@dataclass(frozen=True)
class BimanualWbcSolution:
    effort: torch.Tensor                  # (43,)
    feasible: bool
    fallback_used: bool
    diagnostics: BimanualWbcDiagnostics

class BimanualWholeBodyQp:
    def solve(self, request: BimanualWbcRequest) -> BimanualWbcSolution:
        validate_all_subsolutions(request)
        constraints = build_bimanual_constraints(request)
        result = solve_reference_qp(build_wbc_problem(request, constraints))
        return self._accept_or_last_safe(request, result)
```

Use existing M1 support/base constraints and dynamics helpers. Add platform/dual-arm/O6 bounds, inter-arm and robot/self collision linearization, box non-contact exclusion, and force-closure inequalities. Keep old `m1_panda_coordination/constraints.py` unchanged.

- [ ] **Step 4: Run focused and legacy WBC/QP tests**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_whole_body_qp.py tests/test_m1_panda_qp_backend.py tests/test_m1_panda_standing_wbc.py tests/test_m1_panda_wbc_safety.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/constraints.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/whole_body_qp.py Go2Pvcnn/tests/test_m1_bimanual_whole_body_qp.py
git commit -m "feat: add bimanual whole-body QP"
```

### Task 10: Implement the Phase State Machine and Multirate Runtime

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/state_machine.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_state_machine.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_runtime.py`

**Interfaces:**
- Produces: `BimanualPhase`, `BimanualMission.update(snapshot, diagnostics)`, and `BimanualRuntime.compute(snapshot) -> BimanualCommand`.
- Guarantees: schedules object/arm/hand/WBC at 8/4/2/1 physics-step intervals and atomically advances only after accepted output.

- [ ] **Step 1: Write phase and multirate tests**

```python
def test_normal_phase_sequence_requires_contact_lift_hold_and_support():
    mission = BimanualMission()
    for expected, observation in approved_transition_samples():
        assert mission.update(observation).phase is expected
    assert mission.phase is BimanualPhase.DONE

def test_runtime_cadence_is_25_50_100_200_hz_at_200_hz_physics():
    runtime = make_runtime_with_counting_controllers()
    for step in range(16):
        runtime.compute(make_snapshot(timestamp_ns=step + 1))
    assert runtime.counts == {"object": 2, "arm": 4, "hand": 8, "wbc": 16}
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_state_machine.py tests/test_m1_bimanual_runtime.py`

Expected: missing state machine/runtime imports fail.

- [ ] **Step 3: Implement normal and safe paths with atomic pending transition**

```python
class BimanualRuntime:
    OBJECT_PERIOD = 8
    ARM_PERIOD = 4
    HAND_PERIOD = 2

    def compute(self, snapshot: BimanualSnapshot) -> BimanualCommand:
        validate_monotonic_snapshot(self._last_snapshot, snapshot)
        candidate = self._plan_due_layers(snapshot)
        command = self.wbc.solve(candidate)
        if command.feasible:
            self._commit_candidate(snapshot, candidate, command)
        else:
            command = self._last_safe_or_stationary_hold(snapshot)
        return command
```

Use explicit thresholds: grasp requires both sides force-closure positive for a dwell window; LIFT completes at 0.10 m; HOLD completes after 600 physics steps; RELEASE requires supported box with low twist. Any slip, synchronized Arm fallback, repeated MPC infeasibility, nonfinite, collision, or balance violation enters the approved safe path.

- [ ] **Step 4: Run the pure runtime suite**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_bimanual_state_machine.py tests/test_m1_bimanual_runtime.py tests/test_m1_bimanual_object_mpc.py tests/test_m1_bimanual_dual_arm_mpc.py tests/test_m1_bimanual_hand_mpc.py tests/test_m1_bimanual_whole_body_qp.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/state_machine.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py Go2Pvcnn/tests/test_m1_bimanual_state_machine.py Go2Pvcnn/tests/test_m1_bimanual_runtime.py
git commit -m "feat: orchestrate bimanual MPC mission"
```

### Task 11: Wire the Isolated Isaac Lab Environment

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Create: `Go2Pvcnn/tests/test_m1_dual_panda_o6_env_static.py`

**Interfaces:**
- Produces Gym ID `Isaac-M1-DualPanda-O6-Bimanual-Lift-v0` with a private 43-effort action manager and a deterministic wrapper-owned mission.

- [ ] **Step 1: Write cfg, sensor, box, and registry tests**

```python
def test_env_is_200_hz_fixed_condition_and_43_effort():
    cfg = parse_cfg_ast(ENV_CFG)
    assert cfg["sim.dt"] == 0.005
    assert cfg["decimation"] == 1
    assert cfg["private_action_dim"] == 43
    assert cfg["box.size"] == (0.12, 0.18, 0.10)
    assert cfg["box.mass"] == 0.5

def test_registry_is_isolated():
    assert "Isaac-M1-DualPanda-O6-Bimanual-Lift-v0" in REGISTER.read_text()
    assert "M1DualPandaO6BimanualEnvCfg" in REGISTER.read_text()
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_env_static.py`

Expected: cfg and registry assertions fail.

- [ ] **Step 3: Implement scene and wrapper**

```python
@configclass
class M1DualPandaO6BimanualEnvCfg(M1PandaWbcTeacherEnvCfg):
    private_action_dim = 43
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = M1_DUAL_PANDA_O6_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.box = make_fixed_box_cfg(size=(0.12, 0.18, 0.10), mass=0.5)
        self.scene.o6_contacts = make_o6_contact_sensor_cfg()
        self.sim.dt = 0.005
        self.decimation = 1

class M1DualPandaO6BimanualWrapper:
    def step(self):
        snapshot = self.adapter.snapshot()
        command = self.runtime.compute(snapshot)
        return self.env.step(command.effort.to(self.env.device, torch.float32))
```

Add fingertip/palm/wrist/platform/base contact sensors, a fixed support table, fixed box pose, and no randomization events. The adapter resolves all joint/body IDs once and rejects ambiguous regex matches.

Create the probe here with `--num-envs`, `--steps`, `--seed`, and `--headless`. Its Task 11 responsibility is only startup/finite/action-dimension smoke; Task 12 extends the same file with formal trial aggregation.

- [ ] **Step 4: Run static tests and one-step Isaac smoke**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_env_static.py tests/test_m1_panda_wbc_env_static.py tests/test_m1_panda_arm_mpc_residual_env_static.py`

Expected: all tests pass.

Run: `CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py --num-envs 1 --steps 1 --headless`

Expected: exit 0; action dimension 43; finite snapshot and effort; no reset.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py Go2Pvcnn/tests/test_m1_dual_panda_o6_env_static.py
git commit -m "feat: register bimanual O6 lift environment"
```

### Task 12: Add Probe, Play, and Formal Fixed-Condition Acceptance

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Create: `Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py`
- Create at runtime: `Go2Pvcnn/tests/artifacts/m1_dual_panda_o6_acceptance.json`

**Interfaces:**
- Probe emits per-trial JSON and an aggregate with exact seed/trial counts and every design metric.
- Play uses the same runtime and accepts no policy checkpoint.

- [ ] **Step 1: Write entrypoint and acceptance predicate tests**

```python
def test_acceptance_requires_all_30_trials_and_every_hard_gate():
    trials = [passing_trial(seed, index) for seed in (42, 43, 44) for index in range(10)]
    assert aggregate_acceptance(trials)["accepted"] is True
    trials[7]["box_dropped"] = True
    assert aggregate_acceptance(trials)["accepted"] is False

def test_play_has_no_checkpoint_or_training_surface():
    source = PLAY.read_text()
    assert "--checkpoint" not in source
    assert ".learn(" not in source
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_entrypoints_static.py`

Expected: missing scripts fail.

- [ ] **Step 3: Implement exact metrics and CLI**

```python
def trial_passes(row: dict[str, object]) -> bool:
    return (
        row["lift_height_m"] >= 0.10
        and row["hold_duration_s"] >= 3.0
        and row["hold_position_error_m"] <= 0.02
        and row["hold_orientation_error_rad"] <= 0.10
        and row["relative_palm_slip_m"] <= 0.005
        and row["object_mpc_feasible_rate"] >= 0.98
        and min(row["arm_mpc_feasible_rates"]) >= 0.99
        and min(row["hand_mpc_feasible_rates"]) >= 0.99
        and row["wbc_qp_feasible_rate"] == 1.0
        and row["max_abs_roll_rad"] <= math.radians(10.0)
        and row["max_abs_pitch_rad"] <= math.radians(10.0)
        and row["hard_failure_count"] == 0
        and row["released_supported"] is True
    )
```

Write JSON atomically. Include phase dwell times, fallback counts per layer, max contact forces, collision/limit/reset/nonfinite counts, box final support state, source/asset SHA, Git ref, Isaac version, and exact command.

- [ ] **Step 4: Run static, progressive GPU, formal, and GUI gates**

Run: `cd Go2Pvcnn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_entrypoints_static.py tests/test_m1_bimanual_runtime.py tests/test_m1_dual_panda_o6_env_static.py`

Expected: all tests pass.

Run: `CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py --num-envs 1 --steps 2000 --seed 42 --headless`

Expected: physics hold gates pass before grasp is enabled.

Run: `CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py --seeds 42 43 44 --trials-per-seed 10 --headless --report Go2Pvcnn/tests/artifacts/m1_dual_panda_o6_acceptance.json`

Expected: exit 0 only when 30/30 trials and every hard gate pass; otherwise exit nonzero and retain diagnostics without claiming acceptance.

Run: `CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py --seed 42`

Expected: GUI follows the same deterministic state machine through DONE; GUI appearance is diagnostic, not an acceptance substitute.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py Go2Pvcnn/tests/artifacts/m1_dual_panda_o6_acceptance.json
git commit -m "test: add formal bimanual lift acceptance"
```

### Task 13: Final Regression, Runbook, and Notes Alignment

**Files:**
- Create: `docs/superpowers/runbooks/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-implementation.md`
- Modify when contracts become user-facing: `notes/human/human-02-training-and-entrypoints.md`
- Modify when environment contracts become active: `notes/human/human-03-environment-and-observations.md`

**Interfaces:**
- Produces exact build/verify/probe/play commands, Git/asset SHA evidence, verified metrics, and an honest list of unverified non-goals.

- [ ] **Step 1: Run the complete pure/static regression**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_m1_dual_panda_o6_sources.py \
  tests/test_m1_dual_panda_o6_asset_static.py \
  tests/test_m1_dual_panda_o6_contracts.py \
  tests/test_m1_bimanual_object_mpc.py \
  tests/test_m1_bimanual_dual_arm_mpc.py \
  tests/test_m1_bimanual_hand_mpc.py \
  tests/test_m1_bimanual_whole_body_qp.py \
  tests/test_m1_bimanual_state_machine.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_dual_panda_o6_env_static.py \
  tests/test_m1_dual_panda_o6_entrypoints_static.py \
  tests/test_m1_panda_arm_mpc.py \
  tests/test_m1_panda_qp_backend.py \
  tests/test_m1_panda_wbc_contracts.py
```

Expected: zero failures.

- [ ] **Step 2: Run compile and repository diff gates**

Run: `python -m compileall -q Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination Go2Pvcnn/go2_pvcnn/assets/m1_dual_panda_o6.py Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py Go2Pvcnn/scripts/normalize_o6_assets.py Go2Pvcnn/scripts/build_m1_dual_panda_o6_asset.py Go2Pvcnn/scripts/verify_m1_dual_panda_o6_asset.py Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`

Expected: exit 0.

Run: `git diff --check`

Expected: exit 0.

- [ ] **Step 3: Write the runbook and evidence log with actual outputs**

The runbook must contain the exact normalization, build, verify, one-step smoke, 2000-step hold, 30-trial acceptance, and GUI commands above. The log must record actual test counts and JSON metrics; do not copy expected values into the result section unless observed.

- [ ] **Step 4: Update the T500 tree and human notes**

Mark only actually verified children complete. Keep RL, vision, randomized objects, rolling manipulation, and real hardware explicitly unverified. Update `notes/todo.md`, the T500 branch page, and `notes/log/index.md` with relative links.

- [ ] **Step 5: Commit documentation and final alignment**

```bash
git add docs/superpowers/runbooks/2026-09-02-m1-dual-panda-o6-bimanual-mpc.md notes/todo.md notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md notes/log/index.md notes/log/2026-09-02-m1-dual-panda-o6-bimanual-mpc-implementation.md notes/human/human-02-training-and-entrypoints.md notes/human/human-03-environment-and-observations.md
git commit -m "docs: record bimanual O6 MPC acceptance"
```

## Plan Completion Gate

Before calling the implementation complete, rerun the Task 13 pure/static suite, the asset verifier, the 2000-step hold probe, and the 30-trial formal acceptance from a clean process. Read the fresh outputs, confirm the report's Git and asset SHAs match the candidate, and verify `git status --short` contains no unexpected generated changes. A single GUI success or a partially passing seed set is not completion.
