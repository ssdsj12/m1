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
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot_setup.assembler")

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.robot_setup.assembler import RobotAssembler
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

ROOT_PRIM = "/M1Panda"
PANDA_PRIM = f"{ROOT_PRIM}/Panda"
BASE_MOUNT_FRAME = "/BASE_LINK"
PANDA_MOUNT_FRAME = "/panda_link0"
MOUNT_JOINT_PATH = f"{PANDA_PRIM}/panda_link0/AssemblerFixedJoint"
MOUNT_CLEARANCE_M = 0.0
EXPECTED_ARTICULATION_ROOT = f"{ROOT_PRIM}/BASE_LINK"
EXPECTED_MOUNT_BODY0 = f"{ROOT_PRIM}/BASE_LINK"
EXPECTED_MOUNT_BODY1 = f"{PANDA_PRIM}/panda_link0"
EXPECTED_MOUNT_CHILD_LOCAL_POS = (0.0, 0.0, 0.0)
EXPECTED_MOUNT_CHILD_LOCAL_ROT = (1.0, 0.0, 0.0, 0.0)


def _top_z(stage: Usd.Stage, prim_path: str) -> float:
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return float(bbox.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedBox().GetMax()[2])


def mount_offset_z(
    base_top_z: float, base_origin_z: float, clearance_m: float
) -> float:
    values = (base_top_z, base_origin_z, clearance_m)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("mount offset inputs must be finite")
    if clearance_m < 0.0:
        raise ValueError("mount clearance must be nonnegative")
    return float(base_top_z - base_origin_z + clearance_m)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _remove_refresh_asset_edits(root_layer: Sdf.Layer) -> None:
    """Remove only the invalid asset-path list edits authored by RobotAssembler."""
    m1_spec = root_layer.GetPrimAtPath(ROOT_PRIM)
    panda_spec = root_layer.GetPrimAtPath(PANDA_PRIM)
    _require(m1_spec is not None, f"missing root-layer spec: {ROOT_PRIM}")
    _require(panda_spec is not None, f"missing root-layer spec: {PANDA_PRIM}")

    m1_spec.referenceList.RemoveItemEdits(Sdf.Reference("/M1Panda"))
    panda_spec.referenceList.RemoveItemEdits(Sdf.Reference("/M1Panda/Panda"))
    panda_spec.payloadList.RemoveItemEdits(Sdf.Payload("/M1Panda/Panda"))


def _validate_stage_contract(stage: Usd.Stage, context: str) -> None:
    articulation_roots = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    _require(
        articulation_roots == [EXPECTED_ARTICULATION_ROOT],
        f"{context}: expected articulation root {[EXPECTED_ARTICULATION_ROOT]}, found {articulation_roots}",
    )

    mount_joint = stage.GetPrimAtPath(MOUNT_JOINT_PATH)
    _require(mount_joint.IsA(UsdPhysics.FixedJoint), f"{context}: mount joint is not fixed: {MOUNT_JOINT_PATH}")
    mount_joint_schema = UsdPhysics.Joint(mount_joint)
    body0_targets = [str(path) for path in mount_joint_schema.GetBody0Rel().GetTargets()]
    body1_targets = [str(path) for path in mount_joint_schema.GetBody1Rel().GetTargets()]
    _require(body0_targets == [EXPECTED_MOUNT_BODY0], f"{context}: unexpected mount body0: {body0_targets}")
    _require(body1_targets == [EXPECTED_MOUNT_BODY1], f"{context}: unexpected mount body1: {body1_targets}")
    child_local_pos = tuple(mount_joint_schema.GetLocalPos1Attr().Get())
    child_local_rot_value = mount_joint_schema.GetLocalRot1Attr().Get()
    child_local_rot = (
        float(child_local_rot_value.GetReal()),
        *tuple(child_local_rot_value.GetImaginary()),
    )
    _require(
        child_local_pos == EXPECTED_MOUNT_CHILD_LOCAL_POS,
        f"{context}: unexpected mount child local position: {child_local_pos}",
    )
    _require(
        child_local_rot == EXPECTED_MOUNT_CHILD_LOCAL_ROT,
        f"{context}: unexpected mount child local rotation: {child_local_rot}",
    )
    _require(mount_joint_schema.GetJointEnabledAttr().Get() is True, f"{context}: mount joint is disabled")
    exclude_attr = mount_joint.GetAttribute("physics:excludeFromArticulation")
    _require(
        exclude_attr.IsValid() and exclude_attr.Get() is False,
        f"{context}: mount joint is excluded from articulation",
    )

    panda_root_joint = stage.GetPrimAtPath(f"{PANDA_PRIM}/root_joint")
    enabled_attr = panda_root_joint.GetAttribute("physics:jointEnabled")
    _require(
        enabled_attr.IsValid() and enabled_attr.Get() is False,
        f"{context}: Panda root_joint must remain disabled",
    )


def _prepare_serialized_root_layer(stage: Usd.Stage, asset_root: Path) -> None:
    """Remove RobotAssembler live-refresh artifacts and author portable references."""
    root_layer = stage.GetRootLayer()
    _remove_refresh_asset_edits(root_layer)
    m1_spec = root_layer.GetPrimAtPath(ROOT_PRIM)
    panda_spec = root_layer.GetPrimAtPath(PANDA_PRIM)

    panda_prim = stage.GetPrimAtPath(PANDA_PRIM)
    _require(panda_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI), "failed to remove Panda root API")
    _require(panda_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI), "failed to remove Panda PhysX root API")

    _validate_stage_contract(stage, "pre-export stage")

    m1_spec.referenceList.RemoveItemEdits(Sdf.Reference(str(asset_root / "m1_floating.usda")))
    panda_spec.referenceList.RemoveItemEdits(Sdf.Reference(str(asset_root / "panda/panda.usd")))
    m1_spec.referenceList.Prepend(Sdf.Reference("m1_floating.usda"))
    panda_spec.referenceList.Prepend(Sdf.Reference("panda/panda.usd"))


def _validate_serialized_asset(combined_usd: Path) -> None:
    reopened_stage = Usd.Stage.Open(str(combined_usd))
    _require(reopened_stage is not None, f"failed to reopen serialized asset: {combined_usd}")
    _validate_stage_contract(reopened_stage, "serialized reopen")


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
    _require(Path(converter.usd_path).is_file(), f"URDF conversion did not produce {panda_usd}")

    stage_utils.create_new_stage()
    prim_utils.create_prim(ROOT_PRIM, usd_path=str(asset_root / "m1_floating.usda"))
    prim_utils.create_prim(PANDA_PRIM, usd_path=str(panda_usd))
    stage = stage_utils.get_current_stage()
    base_top_z = _top_z(stage, f"{ROOT_PRIM}/BASE_LINK")
    base_origin_z = float(
        UsdGeom.Xformable(stage.GetPrimAtPath(f"{ROOT_PRIM}/BASE_LINK"))
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()[2]
    )
    mount_offset = np.array(
        [
            0.0,
            0.0,
            mount_offset_z(base_top_z, base_origin_z, MOUNT_CLEARANCE_M),
        ]
    )

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
    _require(
        str(assembled.fixed_joint.GetPath()) == MOUNT_JOINT_PATH,
        f"unexpected mount joint path: {assembled.fixed_joint.GetPath()}",
    )
    _prepare_serialized_root_layer(stage, asset_root)

    mount_joint = stage.GetPrimAtPath(MOUNT_JOINT_PATH)
    _require(mount_joint.IsA(UsdPhysics.FixedJoint), f"mount joint is not fixed: {MOUNT_JOINT_PATH}")
    exclude_attr = mount_joint.GetAttribute("physics:excludeFromArticulation")
    _require(exclude_attr.IsValid() and exclude_attr.Get() is False, "mount joint is excluded from articulation")
    panda_root_joint = stage.GetPrimAtPath(f"{PANDA_PRIM}/root_joint")
    enabled_attr = panda_root_joint.GetAttribute("physics:jointEnabled")
    _require(enabled_attr.IsValid() and enabled_attr.Get() is False, "Panda root_joint must remain disabled")

    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT_PRIM))
    _require(stage.GetRootLayer().Export(str(combined_usd)), f"failed to export combined asset: {combined_usd}")
    _validate_serialized_asset(combined_usd)
    return combined_usd


if __name__ == "__main__":
    try:
        print(build_asset(args.asset_root))
    finally:
        simulation_app.close()
