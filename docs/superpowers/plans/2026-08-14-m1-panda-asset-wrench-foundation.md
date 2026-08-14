# M1 + Panda Asset and Wrench Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local M1 + Franka Panda single-articulation Isaac Lab asset and expose a tested six-dimensional mount wrench in the M1 `BASE_LINK` frame.

**Architecture:** Copy the existing M1 USD dependency set and Isaac Sim's bundled Franka URDF package into one project-owned asset tree, convert Panda locally, and assemble both robots with `RobotAssembler(single_robot=True)`. Add a dedicated smoke environment that keeps M1's 16-action contract explicit and a focused MDP observation that shifts and rotates the incoming Panda mount-joint wrench into the M1 base frame.

**Tech Stack:** Python 3.10, Isaac Lab 2.1.0, Isaac Sim, USD/PXR, PyTorch, Gymnasium, pytest.

## Global Constraints

- Runtime assets must not contain `omniverse://`, HTTP, or other network references.
- The combined robot must be one articulation rooted at the M1 asset.
- The mounting contract is Panda `panda_link0` fixed to M1 `BASE_LINK`.
- Wrench order is exactly `[Fx, Fy, Fz, Mx, My, Mz]` in the M1 `BASE_LINK` frame and about the M1 base origin.
- The force and torque channels remain unnormalized in this foundation phase; normalization belongs to the later Student-observation plan.
- M1 control remains 12 leg joint-position actions plus 4 wheel joint-velocity actions.
- Panda joints are held by their configured drives; this phase does not add IK/OSC or grasping.
- No policy receives payload mass.
- Existing M1 and Go2 task registrations and behavior must remain unchanged.
- `/home/xk/coding/M1` is not currently a Git working tree. Replace each unavailable commit step with the stated checksum checkpoint and record `Git Ref: unavailable` in the task log.

## Scope Boundary

This is the first implementation plan for the approved design. It ends when the combined asset loads offline, exposes 25 DOFs (16 M1 + 7 Panda arm + 2 fingers), steps stably, and reports a correctly transformed mount wrench. Residual control, Teacher training, Student estimation/distillation, IK/OSC, grasping curriculum, hardware sensor drivers, and real-hardware safety are separate follow-on plans that consume the interfaces defined here.

## File Structure

- `Go2Pvcnn/assets/m1_panda/source_manifest.json`: provenance and exact copied-source contract.
- `Go2Pvcnn/assets/m1_panda/m1_floating.usda`: local M1 floating overlay with relative sublayer.
- `Go2Pvcnn/assets/m1_panda/m1/`: copied M1 USD dependency tree.
- `Go2Pvcnn/assets/m1_panda/panda_source/`: copied Franka URDF, meshes, and package metadata.
- `Go2Pvcnn/assets/m1_panda/panda/panda.usd`: generated local Panda root USD and converter outputs.
- `Go2Pvcnn/assets/m1_panda/m1_panda.usd`: generated combined single-articulation root USD.
- `Go2Pvcnn/scripts/build_m1_panda_asset.py`: deterministic conversion and Robot Assembler build entrypoint.
- `Go2Pvcnn/scripts/verify_m1_panda_asset.py`: dependency, articulation, body, joint, and mount-joint verifier.
- `Go2Pvcnn/go2_pvcnn/assets/m1_panda.py`: combined asset constants and `ArticulationCfg`.
- `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py`: pure wrench transform and environment observation adapter.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py`: isolated combined-robot smoke environment.
- `Go2Pvcnn/scripts/m1_panda_wrench_probe.py`: deterministic runtime force/torque probe.
- `Go2Pvcnn/tests/test_m1_panda_asset_static.py`: offline/static asset and config contracts.
- `Go2Pvcnn/tests/test_m1_panda_wrench.py`: tensor-level wrench transform tests.
- `Go2Pvcnn/tests/test_m1_panda_smoke_cfg_static.py`: action, observation, and registry contracts.

---

### Task 1: Establish the project-owned offline asset inputs

**Files:**
- Create: `Go2Pvcnn/assets/m1_panda/source_manifest.json`
- Create: `Go2Pvcnn/assets/m1_panda/m1_floating.usda`
- Create: `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- Copy: `/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0/` to `Go2Pvcnn/assets/m1_panda/m1/ZJ_V3_URDF_V1_0/`
- Copy: `/home/xk/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/data/urdf/robots/franka_description/` to `Go2Pvcnn/assets/m1_panda/panda_source/franka_description/`

**Interfaces:**
- Consumes: the current M1 physics USD and Isaac Sim-bundled `panda_arm_hand.urdf` package.
- Produces: `M1_LOCAL_OVERLAY`, `PANDA_LOCAL_URDF`, and a project-contained source tree used by the builder.

- [ ] **Step 1: Write the failing static asset test**

```python
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "m1_panda"


def test_m1_panda_sources_are_project_owned_and_manifested():
    manifest = json.loads((ASSET_ROOT / "source_manifest.json").read_text())
    assert manifest["m1"]["entry"] == "m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd"
    assert manifest["panda"]["entry"] == "panda_source/franka_description/robots/panda_arm_hand.urdf"
    assert (ASSET_ROOT / manifest["m1"]["entry"]).is_file()
    assert (ASSET_ROOT / manifest["panda"]["entry"]).is_file()


def test_m1_overlay_has_only_a_relative_local_sublayer():
    text = (ASSET_ROOT / "m1_floating.usda").read_text()
    assert "@./m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd@" in text
    assert "/home/" not in text
    assert "omniverse://" not in text
    assert "http://" not in text
    assert "https://" not in text
```

- [ ] **Step 2: Run the test and confirm that the asset root is missing**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q tests/test_m1_panda_asset_static.py
```

Expected: FAIL because `assets/m1_panda/source_manifest.json` does not exist.

- [ ] **Step 3: Copy the two authoritative source trees**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
mkdir -p assets/m1_panda/m1 assets/m1_panda/panda_source
cp -a /home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0 assets/m1_panda/m1/
cp -a /home/xk/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/data/urdf/robots/franka_description assets/m1_panda/panda_source/
```

Expected: both destination entry files exist and `du -sh assets/m1_panda` reports a non-zero local asset tree.

- [ ] **Step 4: Add the manifest and local floating overlay**

Create `source_manifest.json` with exactly:

```json
{
  "m1": {
    "entry": "m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd",
    "source": "/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0"
  },
  "panda": {
    "entry": "panda_source/franka_description/robots/panda_arm_hand.urdf",
    "source": "/home/xk/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/data/urdf/robots/franka_description"
  }
}
```

Create `m1_floating.usda` by copying the existing floating overlay and changing only its sublayer to:

```usda
subLayers = [
    @./m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd@
]
```

- [ ] **Step 5: Run the static test and record a checksum checkpoint**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q tests/test_m1_panda_asset_static.py
find assets/m1_panda -type f ! -name source_files.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > assets/m1_panda/source_files.sha256
```

Expected: `2 passed`; `source_files.sha256` is non-empty and `sha256sum -c assets/m1_panda/source_files.sha256` passes every entry. Record the checksum file in the T400 verification log. Git commit is unavailable in this directory.

---

### Task 2: Convert the local Panda URDF and build the single articulation

**Files:**
- Create: `Go2Pvcnn/scripts/build_m1_panda_asset.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- Generate: `Go2Pvcnn/assets/m1_panda/panda/panda.usd`
- Generate: `Go2Pvcnn/assets/m1_panda/m1_panda.usd`

**Interfaces:**
- Consumes: `assets/m1_panda/m1_floating.usda` and `assets/m1_panda/panda_source/franka_description/robots/panda_arm_hand.urdf`.
- Produces: `build_asset(asset_root: Path) -> Path`, returning the combined root USD; root prim `/M1Panda`; mount bodies `BASE_LINK` and `panda_link0`; fixed-joint prim `/M1Panda/Panda/panda_link0/AssemblerFixedJoint`.

- [ ] **Step 1: Extend the static test with the builder contract**

```python
def test_builder_declares_single_robot_mount_contract():
    source = (ROOT / "scripts" / "build_m1_panda_asset.py").read_text()
    assert 'ROOT_PRIM = "/M1Panda"' in source
    assert 'BASE_MOUNT_FRAME = "/BASE_LINK"' in source
    assert 'PANDA_MOUNT_FRAME = "/panda_link0"' in source
    assert 'MOUNT_JOINT_PATH = f"{PANDA_PRIM}/panda_link0/AssemblerFixedJoint"' in source
    assert "assemble_articulations(" in source
    assert "single_robot=True" in source
```

- [ ] **Step 2: Run the focused test and confirm the builder is absent**

Run: `pytest -q tests/test_m1_panda_asset_static.py::test_builder_declares_single_robot_mount_contract`

Expected: FAIL because `scripts/build_m1_panda_asset.py` does not exist.

- [ ] **Step 3: Implement the deterministic builder**

The script must use this public contract and startup order:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset-root", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
from pxr import Sdf, Usd, UsdGeom

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.robot_setup.assembler import RobotAssembler
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

ROOT_PRIM = "/M1Panda"
PANDA_PRIM = f"{ROOT_PRIM}/Panda"
BASE_MOUNT_FRAME = "/BASE_LINK"
PANDA_MOUNT_FRAME = "/panda_link0"
MOUNT_JOINT_PATH = f"{PANDA_PRIM}/panda_link0/AssemblerFixedJoint"
MOUNT_CLEARANCE_M = 0.01


def _top_z(stage: Usd.Stage, prim_path: str) -> float:
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return float(bbox.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedBox().GetMax()[2])


def build_asset(asset_root: Path) -> Path:
    asset_root = asset_root.resolve()
    panda_dir = asset_root / "panda"
    panda_dir.mkdir(parents=True, exist_ok=True)
    panda_usd = panda_dir / "panda.usd"
    combined_usd = asset_root / "m1_panda.usd"
    urdf = asset_root / "panda_source/franka_description/robots/panda_arm_hand.urdf"

    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(urdf),
            usd_dir=str(panda_dir),
            usd_file_name=panda_usd.name,
            fix_base=True,
            merge_fixed_joints=False,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=80.0, damping=4.0),
                target_type="position",
            ),
        )
    )
    assert Path(converter.usd_path).is_file()

    stage_utils.create_new_stage()
    prim_utils.create_prim(ROOT_PRIM, usd_path=str(asset_root / "m1_floating.usda"))
    prim_utils.create_prim(PANDA_PRIM, usd_path=str(panda_usd))
    stage = stage_utils.get_current_stage()
    base_top_z = _top_z(stage, f"{ROOT_PRIM}/BASE_LINK")
    base_origin_z = float(UsdGeom.Xformable(stage.GetPrimAtPath(f"{ROOT_PRIM}/BASE_LINK")).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2])
    mount_offset = np.array([0.0, 0.0, base_top_z - base_origin_z + MOUNT_CLEARANCE_M])

    assembled = RobotAssembler().assemble_articulations(
        ROOT_PRIM,
        PANDA_PRIM,
        BASE_MOUNT_FRAME,
        PANDA_MOUNT_FRAME,
        fixed_joint_offset=mount_offset,
        fixed_joint_orient=np.array([1.0, 0.0, 0.0, 0.0]),
        mask_all_collisions=True,
        single_robot=True,
    )
    assert str(assembled.fixed_joint.GetPath()) == MOUNT_JOINT_PATH
    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT_PRIM))
    stage.GetRootLayer().Export(str(combined_usd))
    return combined_usd


if __name__ == "__main__":
    try:
        print(build_asset(args.asset_root))
    finally:
        simulation_app.close()
```

- [ ] **Step 4: Run static GREEN and build the asset**

Run:

```bash
cd /home/xk/coding/M1
pytest -q Go2Pvcnn/tests/test_m1_panda_asset_static.py
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/build_m1_panda_asset.py \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda \
  --headless
```

Expected: static tests pass; the command prints the absolute `m1_panda.usd` path and exits `0`.

- [ ] **Step 5: Record generated-asset checksums**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
sha256sum assets/m1_panda/panda/panda.usd assets/m1_panda/m1_panda.usd > assets/m1_panda/generated_files.sha256
```

Expected: two checksum rows. Record them in the T400 log. Git commit is unavailable.

---

### Task 3: Verify offline dependency closure and articulation topology

**Files:**
- Create: `Go2Pvcnn/scripts/verify_m1_panda_asset.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_asset_static.py`

**Interfaces:**
- Consumes: `m1_panda.usd`.
- Produces: JSON-compatible verification output with keys `root`, `dependencies`, `joint_names`, `body_names`, `dof_count`, and `remote_dependencies`.

- [ ] **Step 1: Add a failing static verifier-contract test**

```python
def test_verifier_checks_offline_and_topology_contracts():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    for token in (
        "ComputeAllDependencies",
        "remote_dependencies",
        'EXPECTED_DOF_COUNT = 25',
        '"BASE_LINK"',
        '"panda_link0"',
        '"panda_hand"',
        '"/M1Panda/Panda/panda_link0/AssemblerFixedJoint"',
    ):
        assert token in source
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_m1_panda_asset_static.py::test_verifier_checks_offline_and_topology_contracts`

Expected: FAIL because the verifier file is absent.

- [ ] **Step 3: Implement dependency and runtime topology verification**

Implement a headless Isaac Lab script that:

```python
EXPECTED_DOF_COUNT = 25
REMOTE_PREFIXES = ("omniverse://", "http://", "https://")

stage = Usd.Stage.Open(str(asset_path))
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(asset_path))
dependencies = [layer.identifier for layer in layers] + [str(asset) for asset in assets]
remote_dependencies = [item for item in dependencies if item.startswith(REMOTE_PREFIXES)]
assert not unresolved
assert not remote_dependencies
assert all(Path(item).resolve().is_relative_to(asset_root.resolve()) for item in dependencies if not item.startswith("anon:"))
```

Then instantiate one Isaac Lab `Articulation`, reset it, and assert:

```python
assert robot.num_joints == EXPECTED_DOF_COUNT
assert robot.find_bodies("BASE_LINK")[1] == ["BASE_LINK"]
assert robot.find_bodies("panda_link0")[1] == ["panda_link0"]
assert robot.find_bodies("panda_hand")[1] == ["panda_hand"]
mount_joint = stage.GetPrimAtPath("/M1Panda/Panda/panda_link0/AssemblerFixedJoint")
assert mount_joint.IsValid()
assert mount_joint.IsA(UsdPhysics.FixedJoint)
```

Print one JSON object and exit non-zero on every failed assertion.

- [ ] **Step 4: Run static and real verification**

Run:

```bash
cd /home/xk/coding/M1
pytest -q Go2Pvcnn/tests/test_m1_panda_asset_static.py
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda \
  --headless
```

Expected: pytest passes; JSON reports `dof_count: 25`, an empty `remote_dependencies`, and all required bodies/joint exactly once.

- [ ] **Step 5: Record the topology checkpoint**

Append the verifier JSON and command exit code to `notes/log/2026-08-14-m1-panda-force-aware-teacher-student-design.md`. Git commit is unavailable.

---

### Task 4: Add the combined asset config and isolated smoke environment

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/assets/m1_panda.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_smoke_cfg_static.py`

**Interfaces:**
- Consumes: `M1_JOINT_NAMES`, `M1_LEG_JOINT_NAMES`, `M1_WHEEL_JOINT_NAMES`, and local `m1_panda.usd`.
- Produces: `M1_PANDA_CFG`, `M1_PANDA_MOUNT_BODY_NAME`, `M1_PANDA_BASE_BODY_NAME`, `M1_PANDA_DOF_COUNT`, `M1PandaSmokeEnvCfg`, and Gym id `Isaac-M1-Panda-Smoke-v0`.

- [ ] **Step 1: Write the failing static config test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_combined_cfg_and_smoke_task_keep_m1_action_contract():
    asset = (ROOT / "go2_pvcnn/assets/m1_panda.py").read_text()
    env = (ROOT / "go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py").read_text()
    registry = (ROOT / "go2_pvcnn/tasks/register_m1_envs.py").read_text()
    assert 'M1_PANDA_MOUNT_BODY_NAME = "panda_link0"' in asset
    assert 'M1_PANDA_BASE_BODY_NAME = "BASE_LINK"' in asset
    assert "M1_PANDA_DOF_COUNT = 25" in asset
    assert 'usd_path=M1_PANDA_USD_PATH' in asset
    assert "joint_names=list(M1_LEG_JOINT_NAMES)" in env
    assert "joint_names=list(M1_WHEEL_JOINT_NAMES)" in env
    assert 'id="Isaac-M1-Panda-Smoke-v0"' in registry
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_m1_panda_smoke_cfg_static.py`

Expected: FAIL because both new modules are absent.

- [ ] **Step 3: Implement `M1_PANDA_CFG`**

Start from the current `M1_CFG` rigid/articulation properties and initial M1 pose. Add Panda actuators with the installed Franka limits:

```python
M1_PANDA_USD_PATH = str(Path(__file__).resolve().parents[2] / "assets/m1_panda/m1_panda.usd")
M1_PANDA_BASE_BODY_NAME = "BASE_LINK"
M1_PANDA_MOUNT_BODY_NAME = "panda_link0"
M1_PANDA_DOF_COUNT = 25

M1_PANDA_CFG = M1_CFG.copy()
M1_PANDA_CFG.spawn.usd_path = M1_PANDA_USD_PATH
M1_PANDA_CFG.init_state.joint_pos.update({
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
})
M1_PANDA_CFG.actuators.update({
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"], effort_limit=87.0,
        velocity_limit=2.175, stiffness=80.0, damping=4.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"], effort_limit=12.0,
        velocity_limit=2.61, stiffness=80.0, damping=4.0,
    ),
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"], effort_limit=200.0,
        velocity_limit=0.2, stiffness=2000.0, damping=100.0,
    ),
})
```

- [ ] **Step 4: Implement the smoke environment and registration**

Derive `M1PandaSmokeEnvCfg` from the existing M1 smoke structure, but explicitly scope every joint observation and action to `M1_JOINT_NAMES`. This task must remain independently importable and runnable, so it does not reference the mount-wrench function introduced in Task 5.

The action block must be exactly:

```python
leg_pos = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=list(M1_LEG_JOINT_NAMES),
    scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)},
)
wheel_vel = mdp.JointVelocityActionCfg(
    asset_name="robot", joint_names=list(M1_WHEEL_JOINT_NAMES),
    scale=8.0, use_default_offset=True, clip={".*": (-8.0, 8.0)},
)
```

Register:

```python
gym.register(
    id="Isaac-M1-Panda-Smoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": M1PandaSmokeEnvCfg, "rsl_rl_cfg_entry_point": None},
)
```

- [ ] **Step 5: Run static GREEN**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q tests/test_m1_panda_asset_static.py tests/test_m1_panda_smoke_cfg_static.py tests/test_m1_smoke_cfg_static.py
```

Expected: all tests pass and the old M1 smoke contract remains green. Record the result; Git commit is unavailable.

---

### Task 5: Implement the base-frame mount-wrench observation

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py`
- Modify: `Go2Pvcnn/go2_pvcnn/mdp/__init__.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_smoke_cfg_static.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_wrench.py`

**Interfaces:**
- Consumes: world-frame incoming wrench on `panda_link0`, world positions of `panda_link0` and `BASE_LINK`, and `BASE_LINK` world quaternion.
- Produces: `shift_rotate_wrench_to_base(force_w, torque_w, sensor_pos_w, base_pos_w, base_quat_w) -> torch.Tensor[..., 6]` and `m1_panda_mount_wrench_b(env, asset_cfg, mount_body_name, base_body_name) -> torch.Tensor[num_envs, 6]`.

- [ ] **Step 1: Write tensor-level failing tests**

```python
import torch

from go2_pvcnn.mdp.m1_panda_wrench import shift_rotate_wrench_to_base


def test_identity_frame_keeps_force_and_shifts_moment_to_base_origin():
    result = shift_rotate_wrench_to_base(
        force_w=torch.tensor([[0.0, 10.0, 0.0]]),
        torque_w=torch.tensor([[0.0, 0.0, 2.0]]),
        sensor_pos_w=torch.tensor([[1.0, 0.0, 0.0]]),
        base_pos_w=torch.zeros(1, 3),
        base_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.allclose(result, torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0, 12.0]]))


def test_base_yaw_rotates_world_force_into_base_frame():
    half = 2.0 ** -0.5
    result = shift_rotate_wrench_to_base(
        force_w=torch.tensor([[0.0, 1.0, 0.0]]),
        torque_w=torch.zeros(1, 3),
        sensor_pos_w=torch.zeros(1, 3),
        base_pos_w=torch.zeros(1, 3),
        base_quat_w=torch.tensor([[half, 0.0, 0.0, half]]),
    )
    assert torch.allclose(result[:, :3], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_m1_panda_wrench.py`

Expected: FAIL with `ModuleNotFoundError` for `m1_panda_wrench`.

- [ ] **Step 3: Implement the pure transform**

```python
from __future__ import annotations

import torch
from isaaclab.utils import math as math_utils


def shift_rotate_wrench_to_base(
    force_w: torch.Tensor,
    torque_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
) -> torch.Tensor:
    moment_about_base_w = torque_w + torch.linalg.cross(sensor_pos_w - base_pos_w, force_w, dim=-1)
    force_b = math_utils.quat_rotate_inverse(base_quat_w, force_w)
    moment_b = math_utils.quat_rotate_inverse(base_quat_w, moment_about_base_w)
    return torch.cat((force_b, moment_b), dim=-1)
```

- [ ] **Step 4: Implement the environment adapter**

```python
def m1_panda_mount_wrench_b(
    env,
    asset_cfg,
    mount_body_name: str = "panda_link0",
    base_body_name: str = "BASE_LINK",
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    mount_ids, mount_names = robot.find_bodies(mount_body_name, preserve_order=True)
    base_ids, base_names = robot.find_bodies(base_body_name, preserve_order=True)
    if mount_names != [mount_body_name] or base_names != [base_body_name]:
        raise RuntimeError(f"Expected one mount/base body, got {mount_names=} {base_names=}")
    incoming = robot.root_physx_view.get_link_incoming_joint_force()[:, mount_ids[0], :]
    return shift_rotate_wrench_to_base(
        incoming[:, :3],
        incoming[:, 3:],
        robot.data.body_pos_w[:, mount_ids[0]],
        robot.data.body_pos_w[:, base_ids[0]],
        robot.data.body_quat_w[:, base_ids[0]],
    )
```

Export it from `go2_pvcnn/mdp/__init__.py`, and wire the smoke term as:

```python
mount_wrench_b = ObsTerm(
    func=mdp.m1_panda_mount_wrench_b,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=M1_PANDA_MOUNT_BODY_NAME),
        "mount_body_name": M1_PANDA_MOUNT_BODY_NAME,
        "base_body_name": M1_PANDA_BASE_BODY_NAME,
    },
)
```

Add the following static assertion to `test_m1_panda_smoke_cfg_static.py` in this task:

```python
assert 'body_names=M1_PANDA_MOUNT_BODY_NAME' in env
```

- [ ] **Step 5: Run GREEN and related regression**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q \
  tests/test_m1_panda_wrench.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_smoke_cfg_static.py \
  tests/test_m1_asset_static.py
```

Expected: all tests pass. Record the test count and duration; Git commit is unavailable.

---

### Task 6: Add and run the deterministic real-simulation wrench probe

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_wrench_probe.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_wrench_probe_static.py`
- Modify: `notes/log/2026-08-14-m1-panda-force-aware-teacher-student-design.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/todo.md`
- Modify: `notes/log/index.md`

**Interfaces:**
- Consumes: Gym id `Isaac-M1-Panda-Smoke-v0` and `m1_panda_mount_wrench_b`.
- Produces: JSONL rows for `settle`, `force_x`, `force_y`, `force_z`, `torque_x`, `torque_y`, and `torque_z`, including measured mean wrench and sign/error checks.

- [ ] **Step 1: Write the failing probe-contract test**

```python
from pathlib import Path


def test_probe_covers_all_six_axes_and_clears_external_wrench():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/m1_panda_wrench_probe.py").read_text()
    for case in ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"):
        assert f'"{case}"' in source
    assert "set_external_force_and_torque" in source
    assert "torch.zeros(0, 3" in source
    assert 'TASK_ID = "Isaac-M1-Panda-Smoke-v0"' in source
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_m1_panda_wrench_probe_static.py`

Expected: FAIL because the probe script is absent.

- [ ] **Step 3: Implement the six-axis probe**

The probe must:

1. launch one headless environment;
2. hold zero M1 actions for 100 settling steps;
3. find `panda_hand`, `panda_link0`, and `BASE_LINK` body ids exactly once;
4. record the 50-step mean baseline wrench;
5. apply each local test wrench independently to `panda_hand` for 50 steps using magnitudes `20 N` and `5 N·m`;
6. clear the external wrench with `torch.zeros(0, 3, device=robot.device)` between cases;
7. subtract the baseline, write JSONL, and require the excited measured channel to have stable sign and magnitude greater than 20% of the applied channel;
8. exit non-zero on non-finite data, body lookup mismatch, unexpected reset, or failed channel check.

Use this case table verbatim:

```python
CASES = {
    "force_x": ([20.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "force_y": ([0.0, 20.0, 0.0], [0.0, 0.0, 0.0]),
    "force_z": ([0.0, 0.0, 20.0], [0.0, 0.0, 0.0]),
    "torque_x": ([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
    "torque_y": ([0.0, 0.0, 0.0], [0.0, 5.0, 0.0]),
    "torque_z": ([0.0, 0.0, 0.0], [0.0, 0.0, 5.0]),
}
```

The 20% gate checks signal routing and sign, not force-estimation accuracy; exact static equilibrium values depend on the whole-body controller and contact constraints.

- [ ] **Step 4: Run local tests and the real Isaac Lab smoke**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_panda_wrench.py \
  tests/test_m1_panda_wrench_probe_static.py \
  tests/test_m1_asset_static.py \
  tests/test_m1_smoke_cfg_static.py

cd /home/xk/coding/M1
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/m1_panda_wrench_probe.py \
  --headless \
  --output Go2Pvcnn/tests/artifacts/m1_panda_wrench_probe.jsonl
```

Expected: pytest passes; real probe exits `0`, produces seven finite JSONL rows, reports one 25-DOF articulation, and passes all six channel checks.

- [ ] **Step 5: Run offline denial verification**

Disconnect or block Nucleus/network access for this process, then run:

```bash
cd /home/xk/coding/M1
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda \
  --headless
```

Expected: exit `0` with `remote_dependencies: []`. If the environment cannot enforce network denial, record dependency-closure verification as passed and network-denial execution as unverified; do not claim full offline runtime proof.

- [ ] **Step 6: Align notes and close this foundation phase**

Update the T400 branch and log with:

- exact commands and exit codes;
- generated asset checksums;
- dependency count and remote dependency count;
- body/joint names and 25-DOF result;
- six probe rows and channel pass/fail;
- whether network denial was actually exercised;
- `Git Ref: unavailable` unless the user initializes a repository before execution.

Mark only the asset/wrench foundation child complete. Leave residual policy, Teacher–Student training, IK/OSC, grasping, sensor driver, mechanical validation, and real-hardware validation open.

## Plan Self-Review

- Spec coverage in this phase: local asset closure, single articulation, fixed mount, 25-DOF topology, unchanged M1 hybrid action scope, base-frame six-dimensional total wrench, deterministic tests, and offline verification are covered.
- Deliberately deferred: residual controller, Teacher/Student networks and losses, domain randomization, Panda IK/OSC, object curriculum, real sensor driver, safety state machine, and hardware tests. Each depends on the verified asset/wrench interface produced here.
- Type consistency: the wrench helper always returns `torch.Tensor[num_envs, 6]`; body names and wrench order are identical across asset config, environment, probe, and tests.
- Repository state: commit steps are replaced by checksum/log checkpoints because `/home/xk/coding/M1` has no `.git` directory.
