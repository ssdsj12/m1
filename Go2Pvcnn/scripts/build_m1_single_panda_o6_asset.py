#!/usr/bin/env python3
"""Build the isolated M1 + Panda arm + right O6 articulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--o6-source-root", type=Path, required=True)
parser.add_argument("--force-panda-conversion", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


ROOT_PRIM = "/M1SinglePandaO6"
PANDA_PRIM = f"{ROOT_PRIM}/Panda"
RIGHT_HAND_PRIM = f"{PANDA_PRIM}/right_o6"
EXPECTED_ARTICULATION_ROOT = f"{ROOT_PRIM}/BASE_LINK"
PANDA_MOUNT_JOINT = f"{ROOT_PRIM}/joints/panda_mount_joint"
RIGHT_HAND_MOUNT_JOINT = f"{ROOT_PRIM}/joints/right_hand_mount_joint"
EXPECTED_ACTIVE_DOF_COUNT = 29
EXPECTED_ASSEMBLY_JOINT_COUNT = 2
PANDA_ARM_URDF = "panda_arm.urdf"
RIGHT_O6_ENTRY = "o6_right/O6_right.usd"
MOUNT_CLEARANCE_M = 0.0
MOUNT_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
MOUNT_PATCH_HALF_EXTENTS_M = (0.11, 0.10)
BUILD_SCHEMA = 1

_O6_ACTIVE_JOINTS = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
_O6_MIMIC_MAP = {
    "thumb_ip": ("thumb_cmc_pitch", 1.86, 0.0),
    "index_dip": ("index_mcp_pitch", 0.89, 0.0),
    "middle_dip": ("middle_mcp_pitch", 0.89, 0.0),
    "ring_dip": ("ring_mcp_pitch", 0.89, 0.0),
    "pinky_dip": ("pinky_mcp_pitch", 0.89, 0.0),
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def validate_o6_source(o6_source_root: Path) -> dict[str, Any]:
    manifest_path = o6_source_root / "source_manifest.json"
    _require(manifest_path.is_file(), f"missing normalized O6 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == 1, "unsupported O6 source manifest schema")
    _require(
        manifest.get("right", {}).get("entry") == RIGHT_O6_ENTRY,
        "unexpected right O6 entry in source manifest",
    )
    hashes = manifest.get("sha256")
    _require(isinstance(hashes, dict) and hashes, "O6 source manifest has no hashes")
    for relative, expected in sorted(hashes.items()):
        path = o6_source_root / relative
        _require(path.is_file(), f"missing normalized O6 source: {path}")
        _require(_sha256(path) == expected, f"normalized O6 hash mismatch: {path}")
    return manifest


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
    empty_visual = base_layer.GetPrimAtPath(Sdf.Path("/panda/panda_link8/visuals"))
    if empty_visual is not None:
        empty_visual.referenceList.ClearEdits()
        _require(base_layer.Save(), "failed to remove Panda link8 empty visual reference")
    (output_dir / "config.yaml").unlink(missing_ok=True)
    return output


def write_right_prefixed_o6(source: Path, output: Path) -> Path:
    source_stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    _require(source_stage is not None, f"failed to open right O6 source: {source}")
    flattened = source_stage.Flatten()
    flattened.comment = ""
    flattened.documentation = ""
    stage = Usd.Stage.Open(flattened)
    _require(stage is not None, f"failed to open flattened right O6 source: {source}")
    default_prim = stage.GetDefaultPrim()
    _require(default_prim.IsValid(), f"right O6 source has no default prim: {source}")
    rename_paths = sorted(
        (
            prim.GetPath()
            for prim in stage.Traverse()
            if prim != default_prim
            and (prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint))
        ),
        key=lambda path: (path.pathElementCount, path.pathString),
        reverse=True,
    )
    for old_path in rename_paths:
        if old_path.name.startswith("right_"):
            continue
        new_path = old_path.GetParentPath().AppendChild(f"right_{old_path.name}")
        editor = Usd.NamespaceEditor(stage)
        _require(
            editor.MovePrimAtPath(old_path, new_path),
            f"failed to queue namespace edit: {old_path} -> {new_path}",
        )
        _require(editor.CanApplyEdits(), f"invalid namespace edit: {old_path} -> {new_path}")
        _require(editor.ApplyEdits(), f"failed to prefix O6 prim: {old_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(flattened.Export(str(output)), f"failed to export right O6 asset: {output}")
    return output


def _set_matrix(prim: Usd.Prim, matrix: Gf.Matrix4d) -> None:
    _require(prim.IsValid(), "cannot transform invalid prim")
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)


def _mount_patch(stage: Usd.Stage) -> tuple[float, float]:
    base = stage.GetPrimAtPath(EXPECTED_ARTICULATION_ROOT)
    _require(base.IsValid(), f"missing M1 base body: {EXPECTED_ARTICULATION_ROOT}")
    base_origin = (
        UsdGeom.Xformable(base)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()
    )
    half_x, half_y = MOUNT_PATCH_HALF_EXTENTS_M
    candidates: list[float] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or not path.startswith(f"{EXPECTED_ARTICULATION_ROOT}/"):
            continue
        if "/visuals/" not in path:
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in UsdGeom.Mesh(prim).GetPointsAttr().Get() or ():
            world = transform.Transform(point)
            if (
                abs(float(world[0] - base_origin[0])) <= half_x
                and abs(float(world[1] - base_origin[1])) <= half_y
            ):
                candidates.append(float(world[2]))
    _require(candidates, "no visible M1 mount-patch vertices found")
    return max(candidates), float(base_origin[2])


def create_stage(asset_root: Path) -> Usd.Stage:
    output = asset_root / "m1_single_panda_o6.usd"
    output.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    root = stage.DefinePrim(ROOT_PRIM, "Xform")
    root.GetReferences().AddReference("../m1_panda/m1_floating.usda")
    stage.SetDefaultPrim(root)
    stage.Load()
    top_z, base_origin_z = _mount_patch(stage)
    stage.GetRootLayer().customLayerData = {
        "base_origin_z_m": base_origin_z,
        "mount_patch_top_z_m": top_z,
    }
    return stage


def _set_joint_frames(
    joint: UsdPhysics.Joint,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
) -> None:
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(*MOUNT_QUATERNION_WXYZ))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(*MOUNT_QUATERNION_WXYZ))
    joint.CreateJointEnabledAttr().Set(True)
    joint.GetPrim().CreateAttribute(
        "physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool
    ).Set(False)


def assemble_panda(stage: Usd.Stage, panda_usd: Path) -> UsdPhysics.FixedJoint:
    panda = stage.DefinePrim(PANDA_PRIM, "Xform")
    panda.GetReferences().AddReference(
        str(panda_usd.relative_to(panda_usd.parent.parent))
    )
    stage.Load()
    metadata = stage.GetRootLayer().customLayerData
    top_z = float(metadata["mount_patch_top_z_m"])
    base_origin_z = float(metadata["base_origin_z_m"])
    panda_matrix = Gf.Matrix4d(1.0)
    panda_matrix.SetTranslate(Gf.Vec3d(0.0, 0.0, top_z + MOUNT_CLEARANCE_M))
    _set_matrix(panda, panda_matrix)
    joint = UsdPhysics.FixedJoint.Define(stage, PANDA_MOUNT_JOINT)
    _set_joint_frames(
        joint,
        EXPECTED_ARTICULATION_ROOT,
        f"{PANDA_PRIM}/panda_link0",
        (0.0, 0.0, top_z - base_origin_z + MOUNT_CLEARANCE_M),
    )
    return joint


def assemble_right_o6(stage: Usd.Stage, right_o6_usd: Path) -> UsdPhysics.FixedJoint:
    hand = stage.DefinePrim(RIGHT_HAND_PRIM, "Xform")
    hand.GetReferences().AddReference(
        str(right_o6_usd.relative_to(right_o6_usd.parent.parent))
    )
    stage.Load()
    panda = stage.GetPrimAtPath(PANDA_PRIM)
    wrist = stage.GetPrimAtPath(f"{PANDA_PRIM}/panda_link8")
    _require(panda.IsValid() and wrist.IsValid(), "missing Panda wrist for O6 mount")
    relative, _ = UsdGeom.XformCache().ComputeRelativeTransform(wrist, panda)
    _set_matrix(hand, relative)
    joint = UsdPhysics.FixedJoint.Define(stage, RIGHT_HAND_MOUNT_JOINT)
    _set_joint_frames(
        joint,
        f"{PANDA_PRIM}/panda_link8",
        f"{RIGHT_HAND_PRIM}/right_hand_base_link",
        (0.0, 0.0, 0.0),
    )
    return joint


def remove_child_roots_scenes_and_root_joints(stage: Usd.Stage) -> None:
    for path in (
        f"{ROOT_PRIM}/root_joint",
        f"{PANDA_PRIM}/root_joint",
        f"{RIGHT_HAND_PRIM}/right_root_joint",
    ):
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
        if not path.startswith(f"{RIGHT_HAND_PRIM}/") or not prim.HasAPI(UsdPhysics.CollisionAPI):
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
    return UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False


def _physical_dof_paths(stage: Usd.Stage) -> list[str]:
    return sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint))
        and _enabled_joint(prim)
    )


def _active_dof_paths(stage: Usd.Stage) -> list[str]:
    mimic_names = {f"right_{name}" for name in _O6_MIMIC_MAP}
    return [
        path
        for path in _physical_dof_paths(stage)
        if path.rsplit("/", 1)[-1] not in mimic_names
    ]


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
    assembly_joints = [PANDA_MOUNT_JOINT, RIGHT_HAND_MOUNT_JOINT]
    _require(
        len(assembly_joints) == EXPECTED_ASSEMBLY_JOINT_COUNT,
        "assembly joint count changed",
    )
    for path in assembly_joints:
        prim = stage.GetPrimAtPath(path)
        _require(prim.IsA(UsdPhysics.FixedJoint) and _enabled_joint(prim), f"missing assembly joint: {path}")
    all_paths = [str(prim.GetPath()) for prim in stage.Traverse()]
    _require(
        not any("panda_finger_joint" in path for path in all_paths),
        "arm-only asset contains panda_finger_joint",
    )
    physical = _physical_dof_paths(stage)
    active = _active_dof_paths(stage)
    _require(
        len(active) == EXPECTED_ACTIVE_DOF_COUNT,
        f"{context}: expected {EXPECTED_ACTIVE_DOF_COUNT} active DOF, found {len(active)}",
    )
    return {
        "articulation_roots": articulation_roots,
        "assembly_joints": assembly_joints,
        "physical_dof_paths": physical,
        "active_dof_paths": active,
    }


def export_reopen_validate_and_manifest(
    stage: Usd.Stage,
    asset_root: Path,
    source: dict[str, Any],
    source_manifest_path: Path,
    convex_mesh_count: int,
) -> Path:
    output = asset_root / "m1_single_panda_o6.usd"
    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT_PRIM))
    _require(stage.GetRootLayer().Save(), f"failed to save combined asset: {output}")
    reopened = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    _require(reopened is not None, f"failed to reopen combined asset: {output}")
    contract = validate_stage_contract(reopened, "serialized reopen")
    manifest = {
        "schema": BUILD_SCHEMA,
        "asset": output.name,
        "asset_sha256": _sha256(output),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_file_count": len(source["sha256"]),
        "articulation_root": EXPECTED_ARTICULATION_ROOT,
        "physical_dof_count": len(contract["physical_dof_paths"]),
        "physical_dof_paths": contract["physical_dof_paths"],
        "active_dof_count": EXPECTED_ACTIVE_DOF_COUNT,
        "active_dof_paths": contract["active_dof_paths"],
        "assembly_joints": contract["assembly_joints"],
        "mounts": {
            "panda": {
                "joint": PANDA_MOUNT_JOINT,
                "quaternion_wxyz": list(MOUNT_QUATERNION_WXYZ),
            },
            "right_o6": {
                "joint": RIGHT_HAND_MOUNT_JOINT,
                "quaternion_wxyz": list(MOUNT_QUATERNION_WXYZ),
            },
        },
        "right_o6_active_joint_order": [f"right_{name}" for name in _O6_ACTIVE_JOINTS],
        "right_o6_mimic_map": {
            f"right_{name}": [f"right_{driver}", multiplier, offset]
            for name, (driver, multiplier, offset) in _O6_MIMIC_MAP.items()
        },
        "o6_collision_approximation": "convexHull",
        "o6_convex_mesh_count": convex_mesh_count,
    }
    _atomic_json(asset_root / "asset_manifest.json", manifest)
    return output


def build_asset(
    asset_root: Path,
    o6_source_root: Path,
    force_panda_conversion: bool = False,
) -> Path:
    asset_root = Path(asset_root).resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    o6_source_root = Path(o6_source_root).resolve(strict=True)
    source = validate_o6_source(o6_source_root)
    panda_usd = ensure_arm_only_panda(asset_root, force=force_panda_conversion)
    right_o6_usd = write_right_prefixed_o6(
        o6_source_root / RIGHT_O6_ENTRY,
        asset_root / "prefixed/right_o6.usd",
    )
    stage = create_stage(asset_root)
    assemble_panda(stage, panda_usd)
    assemble_right_o6(stage, right_o6_usd)
    remove_child_roots_scenes_and_root_joints(stage)
    convex_mesh_count = author_convex_collision_approximations(stage)
    validate_stage_contract(stage, "pre-export")
    return export_reopen_validate_and_manifest(
        stage,
        asset_root,
        source,
        o6_source_root / "source_manifest.json",
        convex_mesh_count,
    )


if __name__ == "__main__":
    try:
        built = build_asset(
            args.asset_root,
            args.o6_source_root,
            force_panda_conversion=args.force_panda_conversion,
        )
        print(built)
        print((args.asset_root / "asset_manifest.json").read_text(encoding="utf-8"))
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
    os._exit(0)
