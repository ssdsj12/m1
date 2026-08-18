from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--force-panda-conversion", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

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
MOUNT_PATCH_HALF_EXTENTS_M = (0.11, 0.10)
EXPECTED_ARTICULATION_ROOT = f"{ROOT_PRIM}/BASE_LINK"
EXPECTED_MOUNT_BODY0 = f"{ROOT_PRIM}/BASE_LINK"
EXPECTED_MOUNT_BODY1 = f"{PANDA_PRIM}/panda_link0"
EXPECTED_MOUNT_CHILD_LOCAL_POS = (0.0, 0.0, 0.0)
EXPECTED_MOUNT_CHILD_LOCAL_ROT = (1.0, 0.0, 0.0, 0.0)


def _mount_patch_top_z(
    stage: Usd.Stage,
    prim_path: str,
    half_extents_xy: tuple[float, float],
) -> float:
    """Return the visible top surface near the mount center, not the base-wide maximum."""
    prim = stage.GetPrimAtPath(prim_path)
    _require(prim.IsValid(), f"invalid mount parent prim: {prim_path}")
    origin = (
        UsdGeom.Xformable(prim)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()
    )
    half_x, half_y = half_extents_xy
    candidates: list[float] = []
    for mesh_prim in Usd.PrimRange.Stage(
        stage, Usd.TraverseInstanceProxies()
    ):
        if not mesh_prim.IsA(UsdGeom.Mesh):
            continue
        mesh_path = str(mesh_prim.GetPath())
        if not mesh_path.startswith(f"{prim_path}/"):
            continue
        if "/visuals/" not in mesh_path:
            continue
        transform = UsdGeom.Xformable(
            mesh_prim
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get() or ():
            world = transform.Transform(point)
            if (
                abs(float(world[0] - origin[0])) <= half_x
                and abs(float(world[1] - origin[1])) <= half_y
            ):
                candidates.append(float(world[2]))
    _require(candidates, f"no visible mount-patch vertices under {prim_path}")
    return max(candidates)


def mount_offset_z(
    base_top_z: float, base_origin_z: float, clearance_m: float
) -> float:
    values = (base_top_z, base_origin_z, clearance_m)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("mount offset inputs must be finite")
    if clearance_m < 0.0:
        raise ValueError("mount clearance must be nonnegative")
    return float(base_top_z - base_origin_z + clearance_m)


def _assemble_m1_panda(
    assembler: RobotAssembler,
    stage: Usd.Stage,
    mount_offset: np.ndarray,
):
    if hasattr(assembler, "assemble_articulations"):
        return assembler.assemble_articulations(
            ROOT_PRIM,
            PANDA_PRIM,
            BASE_MOUNT_FRAME,
            PANDA_MOUNT_FRAME,
            fixed_joint_offset=mount_offset,
            fixed_joint_orient=np.array([1.0, 0.0, 0.0, 0.0]),
            mask_all_collisions=True,
            single_robot=True,
        )

    panda_prim = stage.GetPrimAtPath(PANDA_PRIM)
    _require(panda_prim.IsValid(), f"invalid Panda prim: {PANDA_PRIM}")
    panda_transform = Gf.Matrix4d(1.0)
    panda_transform.SetTranslate(
        Gf.Vec3d(*(float(value) for value in mount_offset))
    )
    UsdGeom.Xformable(panda_prim).MakeMatrixXform().Set(panda_transform)
    assembled = assembler.assemble_rigid_bodies(
        ROOT_PRIM,
        PANDA_PRIM,
        EXPECTED_MOUNT_BODY0,
        EXPECTED_MOUNT_BODY1,
        mask_all_collisions=True,
        refresh_asset_paths=False,
    )
    joint = assembled.fixed_joint
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*mount_offset))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(*EXPECTED_MOUNT_CHILD_LOCAL_POS))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.GetPrim().CreateAttribute(
        "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool
    ).Set(False)
    panda_root_joint = stage.GetPrimAtPath(f"{PANDA_PRIM}/root_joint")
    _require(
        panda_root_joint.IsValid(), "Panda root_joint is unavailable"
    )
    panda_root_joint.SetActive(True)
    panda_root_joint.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    panda_root_joint.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    root_enabled_attr = panda_root_joint.GetAttribute(
        "physics:jointEnabled"
    )
    _require(
        root_enabled_attr.IsValid(),
        "Panda root_joint has no physics:jointEnabled attribute",
    )
    panda_root_joint.GetAttribute("physics:jointEnabled").Set(False)
    return assembled


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


def build_asset(
    asset_root: Path, force_panda_conversion: bool = False
) -> Path:
    asset_root = asset_root.resolve()
    panda_dir = asset_root / "panda"
    panda_dir.mkdir(parents=True, exist_ok=True)
    panda_usd = panda_dir / "panda.usd"
    combined_usd = asset_root / "m1_panda.usd"
    urdf = asset_root / "panda_source/franka_description/robots/panda_arm_hand.urdf"

    if force_panda_conversion or not panda_usd.is_file():
        converter = UrdfConverter(
            UrdfConverterCfg(
                asset_path=str(urdf),
                usd_dir=str(panda_dir),
                usd_file_name=panda_usd.name,
                fix_base=True,
                merge_fixed_joints=False,
                force_usd_conversion=True,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=80.0, damping=4.0
                    ),
                    target_type="position",
                ),
            )
        )
        _require(
            Path(converter.usd_path).is_file(),
            f"URDF conversion did not produce {panda_usd}",
        )
    _require(panda_usd.is_file(), f"Panda USD does not exist: {panda_usd}")

    stage_utils.create_new_stage()
    prim_utils.create_prim(ROOT_PRIM, usd_path=str(asset_root / "m1_floating.usda"))
    prim_utils.create_prim(PANDA_PRIM, usd_path=str(panda_usd))
    stage = stage_utils.get_current_stage()
    base_top_z = _mount_patch_top_z(
        stage,
        f"{ROOT_PRIM}/BASE_LINK",
        MOUNT_PATCH_HALF_EXTENTS_M,
    )
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

    assembled = _assemble_m1_panda(
        RobotAssembler(), stage, mount_offset
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
        print(
            build_asset(
                args.asset_root,
                force_panda_conversion=args.force_panda_conversion,
            )
        )
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
    os._exit(0)
