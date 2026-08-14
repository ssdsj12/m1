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

