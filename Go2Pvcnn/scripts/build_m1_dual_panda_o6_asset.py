#!/usr/bin/env python3
"""Build the isolated M1 + yaw platform + dual Panda + dual O6 asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--force-panda-conversion", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


ROOT_PRIM = "/M1DualPandaO6"
EXPECTED_ARTICULATION_ROOT = f"{ROOT_PRIM}/BASE_LINK"
PLATFORM_PRIM = f"{ROOT_PRIM}/platform"
LEFT_ARM_PRIM = f"{ROOT_PRIM}/left_arm"
RIGHT_ARM_PRIM = f"{ROOT_PRIM}/right_arm"
LEFT_HAND_PRIM = f"{LEFT_ARM_PRIM}/left_o6"
RIGHT_HAND_PRIM = f"{RIGHT_ARM_PRIM}/right_o6"

PLATFORM_JOINT_NAME = "dual_arm_platform_yaw_joint"
PLATFORM_JOINT_PATH = f"{ROOT_PRIM}/joints/{PLATFORM_JOINT_NAME}"
PLATFORM_YAW_LIMIT_RAD = (-1.5707963267948966, 1.5707963267948966)
LEFT_ARM_MOUNT_XYZ = (0.0, 0.2, 0.0)
RIGHT_ARM_MOUNT_XYZ = (0.0, -0.2, 0.0)
IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
HAND_MOUNT_QUATERNION_WXYZ = {
    "left": (1.0, 0.0, 0.0, 0.0),
    "right": (0.0, 0.0, 0.0, 1.0),
}
PLATFORM_HALF_EXTENTS_M = (0.31, 0.29, 0.03)
PANDA_ARM_URDF = "panda_arm.urdf"
RIGHT_PALM_HOUSING_MESH_SHA256 = (
    "f7fc8dae5d375e5a33251593c46f8b882c3a1aaafe89fa9544f0ef57d657f5de"
)
RIGHT_PALM_HOUSING_BOUNDS_MIN_M = (-0.0200, -0.0392, 0.0)
RIGHT_PALM_HOUSING_BOUNDS_MAX_M = (0.0200, 0.0376, 0.1128)
RIGHT_PALM_HOUSING_MESH_RELATIVE = "o6_right/meshes/hand_base_link.STL"

EXPECTED_ACTIVE_DOF_COUNT = 43
EXPECTED_PHYSICAL_DOF_COUNT = 53
EXPECTED_ASSEMBLY_JOINT_COUNT = 5
BUILD_SCHEMA = 1

_O6_ENTRIES = {
    "left": "o6_left/O6_left.usd",
    "right": "o6_right/O6_right.usd",
}
_ARM_PRIMS = {"left": LEFT_ARM_PRIM, "right": RIGHT_ARM_PRIM}
_HAND_PRIMS = {"left": LEFT_HAND_PRIM, "right": RIGHT_HAND_PRIM}
_ARM_MOUNTS = {"left": LEFT_ARM_MOUNT_XYZ, "right": RIGHT_ARM_MOUNT_XYZ}
_O6_ACTIVE_JOINTS = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
_O6_MIMIC_JOINTS = (
    "thumb_ip",
    "index_dip",
    "middle_dip",
    "ring_dip",
    "pinky_dip",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_source_manifests(asset_root: Path) -> dict[str, Any]:
    manifest_path = asset_root / "source_manifest.json"
    _require(manifest_path.is_file(), f"missing normalized O6 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == 1, "unsupported O6 source manifest schema")
    hashes = manifest.get("sha256")
    _require(isinstance(hashes, dict) and hashes, "O6 source manifest has no hashes")
    for relative, expected in sorted(hashes.items()):
        path = asset_root / relative
        _require(path.is_file(), f"missing normalized O6 source: {path}")
        _require(_sha256(path) == expected, f"normalized O6 hash mismatch: {path}")
    for side, entry in _O6_ENTRIES.items():
        _require(
            manifest.get(side, {}).get("entry") == entry,
            f"unexpected {side} O6 entry in source manifest",
        )
    return manifest


def _aabb_corners(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> list[list[float]]:
    return [
        [x, y, z]
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]


def _right_palm_housing_support(
    asset_root: Path, source_manifest: dict[str, Any]
) -> dict[str, Any]:
    import trimesh

    mesh_path = asset_root / RIGHT_PALM_HOUSING_MESH_RELATIVE
    expected_source_hash = source_manifest["sha256"].get(
        RIGHT_PALM_HOUSING_MESH_RELATIVE
    )
    _require(
        expected_source_hash == RIGHT_PALM_HOUSING_MESH_SHA256,
        "right palm housing source hash does not match the frozen calibration",
    )
    _require(
        _sha256(mesh_path) == RIGHT_PALM_HOUSING_MESH_SHA256,
        "right palm housing mesh hash mismatch",
    )
    mesh = trimesh.load_mesh(mesh_path, process=False)
    _require(isinstance(mesh, trimesh.Trimesh), "right palm housing is not one mesh")
    _require(not mesh.is_empty, "right palm housing mesh is empty")
    raw_bounds = mesh.bounds
    _require(raw_bounds.shape == (2, 3), "right palm housing bounds have wrong shape")
    raw_min = tuple(float(value) for value in raw_bounds[0])
    raw_max = tuple(float(value) for value in raw_bounds[1])
    _require(
        all(math.isfinite(value) for value in (*raw_min, *raw_max)),
        "right palm housing bounds are non-finite",
    )
    tolerance = 1.0e-9
    _require(
        all(
            raw >= expected - tolerance
            for raw, expected in zip(
                raw_min, RIGHT_PALM_HOUSING_BOUNDS_MIN_M, strict=True
            )
        )
        and all(
            raw <= expected + tolerance
            for raw, expected in zip(
                raw_max, RIGHT_PALM_HOUSING_BOUNDS_MAX_M, strict=True
            )
        ),
        "right palm housing mesh lies outside the frozen support bounds",
    )
    return {
        "mesh_sha256": RIGHT_PALM_HOUSING_MESH_SHA256,
        "bounds_min_m": list(RIGHT_PALM_HOUSING_BOUNDS_MIN_M),
        "bounds_max_m": list(RIGHT_PALM_HOUSING_BOUNDS_MAX_M),
        "support_points_local_m": _aabb_corners(
            RIGHT_PALM_HOUSING_BOUNDS_MIN_M,
            RIGHT_PALM_HOUSING_BOUNDS_MAX_M,
        ),
        "valid": True,
    }


def ensure_arm_only_panda(asset_root: Path, force: bool = False) -> Path:
    shared_root = asset_root.parent / "m1_panda"
    urdf = shared_root / "panda_source/franka_description/robots" / PANDA_ARM_URDF
    _require(urdf.is_file(), f"missing project Panda arm-only URDF: {urdf}")
    output_dir = asset_root / "panda_arm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "panda_arm.usd"
    if force or not output.is_file():
        converter = UrdfConverter(
            UrdfConverterCfg(
                asset_path=str(urdf),
                usd_dir=str(output_dir),
                usd_file_name=output.name,
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
        _require(Path(converter.usd_path).is_file(), f"Panda conversion did not create {output}")
    _require(output.is_file(), f"missing arm-only Panda USD: {output}")
    base_layer_path = output_dir / "configuration/panda_arm_base.usd"
    base_layer = Sdf.Layer.FindOrOpen(str(base_layer_path))
    _require(base_layer is not None, f"missing Panda base layer: {base_layer_path}")
    empty_visual_path = Sdf.Path("/panda/panda_link8/visuals")
    empty_visual_spec = base_layer.GetPrimAtPath(empty_visual_path)
    if empty_visual_spec is not None:
        empty_visual_spec.referenceList.ClearEdits()
        _require(base_layer.Save(), "failed to remove Panda link8 empty visual reference")
    (output_dir / "config.yaml").unlink(missing_ok=True)
    return output


def _write_side_prefixed_asset(source: Path, output: Path, side: str) -> Path:
    source_stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    _require(source_stage is not None, f"failed to open source asset: {source}")
    flattened = source_stage.Flatten()
    # Stage.Flatten() records the source layer's absolute path in layer metadata.
    # Clear it before export so generated project assets remain relocatable.
    flattened.comment = ""
    flattened.documentation = ""
    stage = Usd.Stage.Open(flattened)
    _require(stage is not None, f"failed to open flattened source asset: {source}")
    default_prim = stage.GetDefaultPrim()
    _require(default_prim.IsValid(), f"source has no default prim: {source}")

    rename_paths = sorted(
        (
            prim.GetPath()
            for prim in stage.Traverse()
            if prim != default_prim
            and (
                prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.IsA(UsdPhysics.Joint)
            )
        ),
        key=lambda path: (path.pathElementCount, path.pathString),
        reverse=True,
    )
    for old_path in rename_paths:
        basename = old_path.name
        if basename.startswith(f"{side}_"):
            continue
        new_path = old_path.GetParentPath().AppendChild(f"{side}_{basename}")
        editor = Usd.NamespaceEditor(stage)
        _require(
            editor.MovePrimAtPath(old_path, new_path),
            f"failed to queue namespace edit: {old_path} -> {new_path}",
        )
        _require(editor.CanApplyEdits(), f"invalid namespace edit: {old_path} -> {new_path}")
        _require(editor.ApplyEdits(), f"failed to side-prefix {old_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(flattened.Export(str(output)), f"failed to export side-prefixed asset: {output}")
    return output


def ensure_side_prefixed_assets(asset_root: Path, panda_usd: Path) -> dict[str, dict[str, Path]]:
    output_root = asset_root / "prefixed"
    result: dict[str, dict[str, Path]] = {}
    for side in ("left", "right"):
        result[side] = {
            "panda": _write_side_prefixed_asset(
                panda_usd, output_root / f"{side}_panda.usd", side
            ),
            "o6": _write_side_prefixed_asset(
                asset_root / _O6_ENTRIES[side],
                output_root / f"{side}_o6.usd",
                side,
            ),
        }
    return result


def _write_platform_asset(asset_root: Path) -> Path:
    output = asset_root / "platform.usda"
    stage = Usd.Stage.CreateNew(str(output))
    platform = UsdGeom.Xform.Define(stage, "/DualArmPlatform").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(platform)
    mass = UsdPhysics.MassAPI.Apply(platform)
    mass.CreateMassAttr(8.0)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    cube = UsdGeom.Cube.Define(stage, "/DualArmPlatform/collision")
    cube.CreateSizeAttr(2.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*PLATFORM_HALF_EXTENTS_M))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    display = cube.CreateDisplayColorAttr()
    display.Set([Gf.Vec3f(0.32, 0.36, 0.40)])
    stage.SetDefaultPrim(platform)
    _require(stage.GetRootLayer().Save(), f"failed to save platform asset: {output}")
    return output


def _set_matrix(prim: Usd.Prim, matrix: Gf.Matrix4d) -> None:
    _require(prim.IsValid(), "cannot transform invalid prim")
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)


def _mount_patch_top_z(stage: Usd.Stage) -> float:
    base_path = EXPECTED_ARTICULATION_ROOT
    base = stage.GetPrimAtPath(base_path)
    _require(base.IsValid(), f"missing M1 base body: {base_path}")
    base_origin = (
        UsdGeom.Xformable(base)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()
    )
    candidates: list[float] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith(f"{base_path}/"):
            continue
        if "/visuals/" not in str(prim.GetPath()):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in UsdGeom.Mesh(prim).GetPointsAttr().Get() or ():
            world = transform.Transform(point)
            if abs(float(world[0] - base_origin[0])) <= 0.31 and abs(
                float(world[1] - base_origin[1])
            ) <= 0.29:
                candidates.append(float(world[2]))
    _require(candidates, "no M1 top-platform vertices found")
    return max(candidates)


def create_m1_and_platform_stage(asset_root: Path) -> Usd.Stage:
    output = asset_root / "m1_dual_panda_o6.usd"
    output.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    root = stage.DefinePrim(ROOT_PRIM, "Xform")
    root.GetReferences().AddReference("../m1_panda/m1_floating.usda")
    platform = stage.DefinePrim(PLATFORM_PRIM, "Xform")
    platform.GetReferences().AddReference("platform.usda")
    stage.SetDefaultPrim(root)
    stage.Load()
    top_z = _mount_patch_top_z(stage)
    platform_matrix = Gf.Matrix4d(1.0)
    platform_matrix.SetTranslate(Gf.Vec3d(0.0, 0.0, top_z + PLATFORM_HALF_EXTENTS_M[2]))
    _set_matrix(platform, platform_matrix)
    stage.GetRootLayer().customLayerData = {"platform_top_z_m": top_z}
    return stage


def _set_joint_frames(
    joint: UsdPhysics.Joint,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_rot0: tuple[float, float, float, float] = IDENTITY_QUATERNION_WXYZ,
    local_rot1: tuple[float, float, float, float] = IDENTITY_QUATERNION_WXYZ,
) -> None:
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(*local_rot0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(*local_rot1))
    joint.CreateJointEnabledAttr().Set(True)
    joint.GetPrim().CreateAttribute(
        "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool
    ).Set(False)


def author_platform_revolute_joint(
    stage: Usd.Stage, lower: float = -math.pi / 2, upper: float = math.pi / 2
) -> UsdPhysics.RevoluteJoint:
    _require((lower, upper) == PLATFORM_YAW_LIMIT_RAD, "platform limits changed")
    joint = UsdPhysics.RevoluteJoint.Define(stage, PLATFORM_JOINT_PATH)
    _set_joint_frames(joint, EXPECTED_ARTICULATION_ROOT, PLATFORM_PRIM, (0.0, 0.0, 0.0))
    joint.CreateAxisAttr().Set("Z")
    joint.CreateLowerLimitAttr().Set(math.degrees(lower))
    joint.CreateUpperLimitAttr().Set(math.degrees(upper))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(120.0)
    drive.CreateDampingAttr().Set(18.0)
    drive.CreateMaxForceAttr().Set(180.0)
    return joint


def assemble_panda(
    stage: Usd.Stage,
    side: str,
    mount: tuple[float, float, float],
    panda_usd: Path,
) -> UsdPhysics.FixedJoint:
    arm_prim_path = _ARM_PRIMS[side]
    arm = stage.DefinePrim(arm_prim_path, "Xform")
    arm.GetReferences().AddReference(str(panda_usd.relative_to(panda_usd.parent.parent)))
    stage.Load()
    platform_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(PLATFORM_PRIM)).GetLocalTransformation()
    top_z = float(platform_matrix.ExtractTranslation()[2]) + PLATFORM_HALF_EXTENTS_M[2]
    arm_matrix = Gf.Matrix4d(1.0)
    arm_matrix.SetTranslate(Gf.Vec3d(mount[0], mount[1], top_z))
    _set_matrix(arm, arm_matrix)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{ROOT_PRIM}/joints/{side}_arm_mount_joint")
    _set_joint_frames(
        joint,
        PLATFORM_PRIM,
        f"{arm_prim_path}/{side}_panda_link0",
        (mount[0], mount[1], PLATFORM_HALF_EXTENTS_M[2]),
    )
    return joint


def assemble_o6(
    stage: Usd.Stage,
    side: str,
    wrist_body: str,
    o6_usd: Path,
) -> UsdPhysics.FixedJoint:
    hand_path = _HAND_PRIMS[side]
    hand = stage.DefinePrim(hand_path, "Xform")
    hand.GetReferences().AddReference(str(o6_usd.relative_to(o6_usd.parent.parent)))
    stage.Load()
    arm = stage.GetPrimAtPath(_ARM_PRIMS[side])
    wrist = stage.GetPrimAtPath(wrist_body)
    _require(arm.IsValid() and wrist.IsValid(), f"missing {side} Panda wrist")
    relative, _ = UsdGeom.XformCache().ComputeRelativeTransform(wrist, arm)
    mount_quaternion = HAND_MOUNT_QUATERNION_WXYZ[side]
    mount_matrix = Gf.Matrix4d(1.0)
    mount_matrix.SetRotate(
        Gf.Quatd(mount_quaternion[0], Gf.Vec3d(*mount_quaternion[1:]))
    )
    _set_matrix(hand, mount_matrix * relative)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{ROOT_PRIM}/joints/{side}_hand_mount_joint")
    _set_joint_frames(
        joint,
        wrist_body,
        f"{hand_path}/{side}_hand_base_link",
        (0.0, 0.0, 0.0),
        local_rot0=mount_quaternion,
    )
    return joint


def remove_child_roots_scenes_and_root_joints(stage: Usd.Stage) -> None:
    root_joint_paths = [
        f"{ROOT_PRIM}/root_joint",
        f"{LEFT_ARM_PRIM}/left_root_joint",
        f"{RIGHT_ARM_PRIM}/right_root_joint",
        f"{LEFT_HAND_PRIM}/left_root_joint",
        f"{RIGHT_HAND_PRIM}/right_root_joint",
    ]
    for path in root_joint_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)
    for prim in list(stage.TraverseAll()):
        path = str(prim.GetPath())
        if prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
        if path != EXPECTED_ARTICULATION_ROOT and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)


def author_convex_collision_approximations(stage: Usd.Stage) -> int:
    count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not (
            path.startswith(f"{LEFT_HAND_PRIM}/") or path.startswith(f"{RIGHT_HAND_PRIM}/")
        ):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            _require(
                approximation == "convexHull",
                f"O6 collision is not convexHull: {path} ({approximation})",
            )
        else:
            _require(not prim.IsInstanceProxy(), f"cannot repair instance collision: {path}")
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set("convexHull")
        count += 1
    _require(count > 0, "no O6 meshes received convex collision approximation")
    return count


def _enabled_joint(prim: Usd.Prim) -> bool:
    enabled = UsdPhysics.Joint(prim).GetJointEnabledAttr().Get()
    return enabled is not False


def _physical_dof_paths(stage: Usd.Stage) -> list[str]:
    return sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint))
        and _enabled_joint(prim)
    )


def _active_dof_paths(stage: Usd.Stage) -> list[str]:
    physical = _physical_dof_paths(stage)
    mimic_names = set(_O6_MIMIC_JOINTS)
    result = []
    for path in physical:
        name = path.rsplit("/", 1)[-1]
        for side in ("left", "right"):
            if name.startswith(f"{side}_"):
                name = name[len(side) + 1 :]
                break
        if name not in mimic_names:
            result.append(path)
    return result


def _quaternion_wxyz(quaternion: Any) -> tuple[float, float, float, float]:
    imaginary = quaternion.GetImaginary()
    values = (
        float(quaternion.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )
    norm = math.sqrt(sum(value * value for value in values))
    _require(math.isfinite(norm) and norm > 0.0, "invalid zero or non-finite quaternion")
    return tuple(value / norm for value in values)


def _quaternion_matches(
    measured: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
    tolerance: float = 1.0e-7,
) -> bool:
    direct = max(abs(lhs - rhs) for lhs, rhs in zip(measured, expected, strict=True))
    negated = max(abs(lhs + rhs) for lhs, rhs in zip(measured, expected, strict=True))
    return min(direct, negated) <= tolerance


def _hand_mount_contract(stage: Usd.Stage) -> dict[str, dict[str, Any]]:
    cache = UsdGeom.XformCache()
    report: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        arm = stage.GetPrimAtPath(_ARM_PRIMS[side])
        wrist = stage.GetPrimAtPath(f"{_ARM_PRIMS[side]}/{side}_panda_link8")
        hand = stage.GetPrimAtPath(_HAND_PRIMS[side])
        joint_prim = stage.GetPrimAtPath(f"{ROOT_PRIM}/joints/{side}_hand_mount_joint")
        _require(
            arm.IsValid() and wrist.IsValid() and hand.IsValid() and joint_prim.IsValid(),
            f"missing {side} hand mount prim",
        )
        wrist_matrix, _ = cache.ComputeRelativeTransform(wrist, arm)
        hand_matrix, _ = cache.ComputeRelativeTransform(hand, arm)
        measured_mount = hand_matrix * wrist_matrix.GetInverse()
        translation = tuple(float(value) for value in measured_mount.ExtractTranslation())
        measured = _quaternion_wxyz(measured_mount.ExtractRotationQuat())
        joint = UsdPhysics.Joint(joint_prim)
        joint_rot0 = _quaternion_wxyz(joint.GetLocalRot0Attr().Get())
        joint_rot1 = _quaternion_wxyz(joint.GetLocalRot1Attr().Get())
        expected = HAND_MOUNT_QUATERNION_WXYZ[side]
        valid = (
            max(abs(value) for value in translation) <= 1.0e-7
            and _quaternion_matches(measured, expected)
            and _quaternion_matches(joint_rot0, expected)
            and _quaternion_matches(joint_rot1, IDENTITY_QUATERNION_WXYZ)
            and _quaternion_matches(measured, joint_rot0)
        )
        report[side] = {
            "expected_quaternion_wxyz": list(expected),
            "measured_quaternion_wxyz": list(measured),
            "joint_local_rot0_wxyz": list(joint_rot0),
            "joint_local_rot1_wxyz": list(joint_rot1),
            "measured_translation_m": list(translation),
            "valid": valid,
        }
        _require(valid, f"serialized {side} hand mount calibration mismatch: {report[side]}")
    return report


def validate_stage_contract(stage: Usd.Stage, context: str = "stage") -> dict[str, Any]:
    articulation_roots = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    _require(
        articulation_roots == [EXPECTED_ARTICULATION_ROOT],
        f"{context}: expected one articulation root, found {articulation_roots}",
    )
    assembly_paths = [
        PLATFORM_JOINT_PATH,
        f"{ROOT_PRIM}/joints/left_arm_mount_joint",
        f"{ROOT_PRIM}/joints/right_arm_mount_joint",
        f"{ROOT_PRIM}/joints/left_hand_mount_joint",
        f"{ROOT_PRIM}/joints/right_hand_mount_joint",
    ]
    _require(len(assembly_paths) == EXPECTED_ASSEMBLY_JOINT_COUNT, "assembly count changed")
    for path in assembly_paths:
        prim = stage.GetPrimAtPath(path)
        _require(prim.IsA(UsdPhysics.Joint) and _enabled_joint(prim), f"missing assembly joint: {path}")
    all_paths = [str(prim.GetPath()) for prim in stage.Traverse()]
    _require(
        not any("panda_finger_joint" in path for path in all_paths),
        "arm-only asset contains panda_finger_joint",
    )
    physical = _physical_dof_paths(stage)
    active = _active_dof_paths(stage)
    _require(
        len(physical) == EXPECTED_PHYSICAL_DOF_COUNT,
        f"{context}: expected {EXPECTED_PHYSICAL_DOF_COUNT} physical DOF, found {len(physical)}",
    )
    _require(
        len(active) == EXPECTED_ACTIVE_DOF_COUNT,
        f"{context}: expected {EXPECTED_ACTIVE_DOF_COUNT} active DOF, found {len(active)}",
    )
    hand_mounts = _hand_mount_contract(stage)
    return {
        "articulation_roots": articulation_roots,
        "assembly_joints": assembly_paths,
        "physical_dof_paths": physical,
        "active_dof_paths": active,
        "hand_mounts": hand_mounts,
    }


def export_reopen_validate_and_manifest(
    stage: Usd.Stage,
    asset_root: Path,
    source_manifest: dict[str, Any],
    convex_mesh_count: int,
) -> Path:
    output = asset_root / "m1_dual_panda_o6.usd"
    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT_PRIM))
    _require(stage.GetRootLayer().Save(), f"failed to save combined asset: {output}")
    reopened = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    _require(reopened is not None, f"failed to reopen combined asset: {output}")
    contract = validate_stage_contract(reopened, "serialized reopen")
    manifest = {
        "schema": BUILD_SCHEMA,
        "asset": output.name,
        "asset_sha256": _sha256(output),
        "source_manifest_sha256": _sha256(asset_root / "source_manifest.json"),
        "source_file_count": len(source_manifest["sha256"]),
        "articulation_root": EXPECTED_ARTICULATION_ROOT,
        "physical_dof_count": EXPECTED_PHYSICAL_DOF_COUNT,
        "active_dof_count": EXPECTED_ACTIVE_DOF_COUNT,
        "active_dof_paths": contract["active_dof_paths"],
        "assembly_joints": contract["assembly_joints"],
        "platform_yaw_limit_rad": list(PLATFORM_YAW_LIMIT_RAD),
        "platform_velocity_limit_rad_s": 0.25,
        "arm_mounts": {
            side: {
                "translation": list(_ARM_MOUNTS[side]),
                "quaternion_wxyz": list(IDENTITY_QUATERNION_WXYZ),
            }
            for side in ("left", "right")
        },
        "hand_mounts": contract["hand_mounts"],
        "o6_active_joint_order": list(_O6_ACTIVE_JOINTS),
        "o6_mimic_joints": list(_O6_MIMIC_JOINTS),
        "o6_collision_approximation": "convexHull",
        "o6_convex_mesh_count": convex_mesh_count,
        "right_palm_housing_support": _right_palm_housing_support(
            asset_root, source_manifest
        ),
    }
    _atomic_json(asset_root / "asset_manifest.json", manifest)
    return output


def build_asset(asset_root: Path, force_panda_conversion: bool = False) -> Path:
    asset_root = Path(asset_root).resolve(strict=True)
    source_manifest = validate_source_manifests(asset_root)
    panda_usd = ensure_arm_only_panda(asset_root, force=force_panda_conversion)
    prefixed = ensure_side_prefixed_assets(asset_root, panda_usd)
    _write_platform_asset(asset_root)
    stage = create_m1_and_platform_stage(asset_root)
    author_platform_revolute_joint(stage, lower=-math.pi / 2, upper=math.pi / 2)
    assemble_panda(
        stage, side="left", mount=LEFT_ARM_MOUNT_XYZ, panda_usd=prefixed["left"]["panda"]
    )
    assemble_panda(
        stage, side="right", mount=RIGHT_ARM_MOUNT_XYZ, panda_usd=prefixed["right"]["panda"]
    )
    assemble_o6(
        stage,
        side="left",
        wrist_body=f"{LEFT_ARM_PRIM}/left_panda_link8",
        o6_usd=prefixed["left"]["o6"],
    )
    assemble_o6(
        stage,
        side="right",
        wrist_body=f"{RIGHT_ARM_PRIM}/right_panda_link8",
        o6_usd=prefixed["right"]["o6"],
    )
    remove_child_roots_scenes_and_root_joints(stage)
    convex_mesh_count = author_convex_collision_approximations(stage)
    validate_stage_contract(stage, "pre-export stage")
    return export_reopen_validate_and_manifest(
        stage, asset_root, source_manifest, convex_mesh_count
    )


if __name__ == "__main__":
    try:
        built = build_asset(
            args.asset_root, force_panda_conversion=args.force_panda_conversion
        )
        print(built)
        print((args.asset_root / "asset_manifest.json").read_text(encoding="utf-8"))
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
    os._exit(0)
