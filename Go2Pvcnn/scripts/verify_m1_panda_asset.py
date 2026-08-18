#!/usr/bin/env python3
"""Verify that the generated M1 + Panda asset is offline and simulation-ready."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


EXPECTED_DOF_COUNT = 25
MAX_MOUNT_RELATIVE_STEP_DELTA_M = 1.0e-4
MOUNT_PLANE_TOLERANCE_M = 1.0e-6
MOUNT_SURFACE_TOLERANCE_M = 1.0e-6
MOUNT_PATCH_HALF_EXTENTS_M = (0.11, 0.10)
REMOTE_PREFIXES = ("omniverse://", "http://", "https://")
BUILTIN_MDL_ALLOWLIST = {"OmniPBR.mdl"}
EXPECTED_ARTICULATION_ROOT = "/M1Panda/BASE_LINK"
EXPECTED_MOUNT_BODY0 = "/M1Panda/BASE_LINK"
EXPECTED_MOUNT_BODY1 = "/M1Panda/Panda/panda_link0"
EXPECTED_MOUNT_CHILD_LOCAL_POS = (0.0, 0.0, 0.0)
EXPECTED_MOUNT_CHILD_LOCAL_ROT = (1.0, 0.0, 0.0, 0.0)
MOUNT_JOINT_PATH = "/M1Panda/Panda/panda_link0/AssemblerFixedJoint"
REQUIRED_BODY_NAMES = ("BASE_LINK", "panda_link0", "panda_hand")
PANDA_HOME_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _asset_path_text(asset: Any) -> str:
    resolved = str(getattr(asset, "resolvedPath", ""))
    authored = str(getattr(asset, "path", asset))
    return resolved or authored


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _classify_unresolved_dependencies(unresolved):
    builtin_mdl_dependencies = []
    unresolved_dependencies = []
    for unresolved_asset in unresolved:
        item = str(unresolved_asset)
        if item in BUILTIN_MDL_ALLOWLIST:
            builtin_mdl_dependencies.append(item)
        else:
            unresolved_dependencies.append(item)
    return builtin_mdl_dependencies, unresolved_dependencies


def _articulation_root_errors(articulation_roots):
    if articulation_roots == [EXPECTED_ARTICULATION_ROOT]:
        return []
    return [
        f"expected articulation roots [{EXPECTED_ARTICULATION_ROOT!r}], found {articulation_roots}"
    ]


def _mount_joint_contract_errors(
    body0_targets, body1_targets, joint_enabled, excluded, child_local_pos, child_local_rot
):
    errors = []
    if body0_targets != [EXPECTED_MOUNT_BODY0]:
        errors.append(f"expected mount body0 [{EXPECTED_MOUNT_BODY0!r}], found {body0_targets}")
    if body1_targets != [EXPECTED_MOUNT_BODY1]:
        errors.append(f"expected mount body1 [{EXPECTED_MOUNT_BODY1!r}], found {body1_targets}")
    if joint_enabled is not True:
        errors.append(f"expected enabled mount joint, found jointEnabled={joint_enabled}")
    if excluded is not False:
        errors.append(f"expected mount joint in articulation, found excludeFromArticulation={excluded}")
    if child_local_pos != EXPECTED_MOUNT_CHILD_LOCAL_POS:
        errors.append(
            f"expected mount child local position {EXPECTED_MOUNT_CHILD_LOCAL_POS}, found {child_local_pos}"
        )
    if child_local_rot != EXPECTED_MOUNT_CHILD_LOCAL_ROT:
        errors.append(
            f"expected mount child local rotation {EXPECTED_MOUNT_CHILD_LOCAL_ROT}, found {child_local_rot}"
        )
    return errors


def _mount_plane_errors(
    parent_local_pos, expected_parent_local_pos, tolerance_m
):
    if parent_local_pos is None:
        return ["mount parent local position is unavailable"]
    error = max(
        abs(float(actual) - float(expected))
        for actual, expected in zip(
            parent_local_pos, expected_parent_local_pos, strict=True
        )
    )
    if error <= tolerance_m:
        return []
    return [
        f"mount parent plane error {error} m exceeds {tolerance_m} m"
    ]


def _surface_gap_errors(surface_gap_m, tolerance_m):
    if surface_gap_m is None:
        return ["mount visible surface gap is unavailable"]
    if abs(float(surface_gap_m)) <= tolerance_m:
        return []
    relation = "gap" if surface_gap_m > 0.0 else "penetration"
    return [
        f"mount visible surface {relation} {surface_gap_m} m exceeds +/-{tolerance_m} m"
    ]


def _visible_mesh_z_values(stage, prim_path, Usd, UsdGeom):
    values = []
    for mesh_prim in Usd.PrimRange.Stage(
        stage, Usd.TraverseInstanceProxies()
    ):
        path = str(mesh_prim.GetPath())
        if (
            mesh_prim.IsA(UsdGeom.Mesh)
            and path.startswith(f"{prim_path}/")
            and "/visuals/" in path
        ):
            transform = UsdGeom.Xformable(
                mesh_prim
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            values.extend(
                float(transform.Transform(point)[2])
                for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get() or ()
            )
    _require(values, f"no visible mesh vertices under {prim_path}")
    return values


def _mount_patch_top_z(stage, prim_path, Usd, UsdGeom):
    prim = stage.GetPrimAtPath(prim_path)
    _require(prim.IsValid(), f"invalid mount parent prim: {prim_path}")
    origin = (
        UsdGeom.Xformable(prim)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()
    )
    half_x, half_y = MOUNT_PATCH_HALF_EXTENTS_M
    candidates = []
    for mesh_prim in Usd.PrimRange.Stage(
        stage, Usd.TraverseInstanceProxies()
    ):
        path = str(mesh_prim.GetPath())
        if not (
            mesh_prim.IsA(UsdGeom.Mesh)
            and path.startswith(f"{prim_path}/")
            and "/visuals/" in path
        ):
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


def _independent_mount_parent_local_pos(asset_root, Usd, UsdGeom):
    m1_stage = Usd.Stage.Open(str(asset_root / "m1_floating.usda"))
    _require(m1_stage is not None, "failed to open independent M1 asset")
    base_prim = m1_stage.GetPrimAtPath(
        "/ZJ_V3_URDF_V1_0/BASE_LINK"
    )
    _require(base_prim.IsValid(), "independent M1 BASE_LINK is invalid")
    top_z = _mount_patch_top_z(
        m1_stage,
        "/ZJ_V3_URDF_V1_0/BASE_LINK",
        Usd,
        UsdGeom,
    )
    origin_z = float(
        UsdGeom.Xformable(base_prim)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()[2]
    )
    return (0.0, 0.0, top_z - origin_z)


def _inspect_dependencies(asset_path: Path, asset_root: Path, UsdUtils: Any) -> dict[str, list[str]]:
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(asset_path))
    dependencies = [str(layer.identifier) for layer in layers] + [_asset_path_text(asset) for asset in assets]
    remote_dependencies = [item for item in dependencies if item.startswith(REMOTE_PREFIXES)]
    outside_root = [
        item
        for item in dependencies
        if not item.startswith("anon:") and not _is_within(Path(item), asset_root)
    ]
    builtin_mdl_dependencies, unresolved_dependencies = _classify_unresolved_dependencies(unresolved)
    return {
        "dependencies": dependencies,
        "remote_dependencies": remote_dependencies,
        "unresolved_dependencies": unresolved_dependencies,
        "builtin_mdl_dependencies": builtin_mdl_dependencies,
        "outside_root_dependencies": outside_root,
    }


def _verify_asset(asset_path: Path, asset_root: Path, device: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg

    asset_root = asset_root.resolve()
    asset_path = asset_path.resolve()
    _require(asset_path.is_file(), f"asset does not exist: {asset_path}")
    _require(asset_root.is_dir(), f"asset root does not exist: {asset_root}")
    _require(_is_within(asset_path, asset_root), f"asset is outside asset root: {asset_path}")

    stage = Usd.Stage.Open(str(asset_path))
    _require(stage is not None, f"failed to open USD stage: {asset_path}")
    dependency_result = _inspect_dependencies(asset_path, asset_root, UsdUtils)
    validation_errors: list[str] = []
    mount_surface_top_z = _mount_patch_top_z(
        stage, "/M1Panda/BASE_LINK", Usd, UsdGeom
    )
    panda_visible_bottom_z = min(
        _visible_mesh_z_values(
            stage, "/M1Panda/Panda/panda_link0", Usd, UsdGeom
        )
    )
    mount_surface_gap_m = panda_visible_bottom_z - mount_surface_top_z
    validation_errors.extend(
        _surface_gap_errors(
            mount_surface_gap_m, MOUNT_SURFACE_TOLERANCE_M
        )
    )
    if dependency_result["unresolved_dependencies"]:
        validation_errors.append(f"unresolved dependencies: {dependency_result['unresolved_dependencies']}")
    if dependency_result["remote_dependencies"]:
        validation_errors.append(f"remote dependencies: {dependency_result['remote_dependencies']}")
    if dependency_result["outside_root_dependencies"]:
        validation_errors.append(
            f"dependencies outside asset root: {dependency_result['outside_root_dependencies']}"
        )

    articulation_roots = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    validation_errors.extend(_articulation_root_errors(articulation_roots))
    mount_joint = stage.GetPrimAtPath(MOUNT_JOINT_PATH)
    mount_body0_targets: list[str] = []
    mount_body1_targets: list[str] = []
    mount_joint_enabled: bool | None = None
    mount_joint_exclude_from_articulation: bool | None = None
    mount_parent_local_pos: tuple[float, ...] | None = None
    expected_mount_parent_local_pos = _independent_mount_parent_local_pos(
        asset_root, Usd, UsdGeom
    )
    mount_plane_error_m: float | None = None
    mount_child_local_pos: tuple[float, ...] | None = None
    mount_child_local_rot: tuple[float, ...] | None = None
    if not mount_joint.IsValid():
        validation_errors.append(f"mount joint is invalid: {MOUNT_JOINT_PATH}")
    elif not mount_joint.IsA(UsdPhysics.FixedJoint):
        validation_errors.append(f"mount joint is not a FixedJoint: {MOUNT_JOINT_PATH}")
    else:
        mount_joint_schema = UsdPhysics.Joint(mount_joint)
        mount_body0_targets = [str(path) for path in mount_joint_schema.GetBody0Rel().GetTargets()]
        mount_body1_targets = [str(path) for path in mount_joint_schema.GetBody1Rel().GetTargets()]
        mount_joint_enabled = mount_joint_schema.GetJointEnabledAttr().Get()
        exclude_attr = mount_joint.GetAttribute("physics:excludeFromArticulation")
        mount_joint_exclude_from_articulation = exclude_attr.Get() if exclude_attr.IsValid() else None
        mount_parent_local_pos = tuple(
            mount_joint_schema.GetLocalPos0Attr().Get()
        )
        mount_plane_error_m = max(
            abs(float(actual) - float(expected))
            for actual, expected in zip(
                mount_parent_local_pos,
                expected_mount_parent_local_pos,
                strict=True,
            )
        )
        mount_child_local_pos = tuple(mount_joint_schema.GetLocalPos1Attr().Get())
        child_local_rot_value = mount_joint_schema.GetLocalRot1Attr().Get()
        mount_child_local_rot = (
            float(child_local_rot_value.GetReal()),
            *tuple(child_local_rot_value.GetImaginary()),
        )
        validation_errors.extend(
            _mount_joint_contract_errors(
                mount_body0_targets,
                mount_body1_targets,
                mount_joint_enabled,
                mount_joint_exclude_from_articulation,
                mount_child_local_pos,
                mount_child_local_rot,
            )
        )
        validation_errors.extend(
            _mount_plane_errors(
                mount_parent_local_pos,
                expected_mount_parent_local_pos,
                MOUNT_PLANE_TOLERANCE_M,
            )
        )
    panda_root_joint = stage.GetPrimAtPath("/M1Panda/Panda/root_joint")
    panda_root_enabled_attr = panda_root_joint.GetAttribute("physics:jointEnabled")
    panda_root_joint_enabled = (
        bool(panda_root_enabled_attr.Get()) if panda_root_enabled_attr.IsValid() else None
    )
    if panda_root_joint_enabled is not False:
        validation_errors.append(
            f"expected disabled Panda root_joint, found jointEnabled={panda_root_joint_enabled}"
        )

    joint_names: list[str] = []
    body_names: list[str] = []
    dof_count: int | None = None
    physics_steps = 0
    runtime_initialized = False
    mount_relative_step_delta_m: float | None = None
    try:
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=device, dt=0.01))
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/M1Panda",
                spawn=sim_utils.UsdFileCfg(usd_path=str(asset_path)),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos=PANDA_HOME_JOINT_POS),
                actuators={},
            )
        )
        sim.reset()
        runtime_initialized = bool(robot.is_initialized)
        _require(runtime_initialized, "Isaac Lab articulation initialization did not complete")
        robot.reset()
        robot.write_data_to_sim()
        robot.update(0.0)
        base_index = robot.body_names.index("BASE_LINK")
        panda_mount_index = robot.body_names.index("panda_link0")
        relative_position_before = (
            robot.data.body_link_pos_w[:, panda_mount_index]
            - robot.data.body_link_pos_w[:, base_index]
        ).clone()
        sim.step()
        physics_steps = 1
        robot.update(sim.get_physics_dt())
        relative_position_after = (
            robot.data.body_link_pos_w[:, panda_mount_index]
            - robot.data.body_link_pos_w[:, base_index]
        )
        mount_relative_step_delta_m = float(
            (relative_position_after - relative_position_before).norm(dim=-1).max().item()
        )
        if mount_relative_step_delta_m > MAX_MOUNT_RELATIVE_STEP_DELTA_M:
            validation_errors.append(
                "mount relative position changed across one physics step: "
                f"{mount_relative_step_delta_m} m > {MAX_MOUNT_RELATIVE_STEP_DELTA_M} m"
            )

        joint_names = list(robot.joint_names)
        body_names = list(robot.body_names)
        dof_count = int(robot.num_joints)
        if dof_count != EXPECTED_DOF_COUNT:
            validation_errors.append(f"expected {EXPECTED_DOF_COUNT} DOF, found {dof_count}")
        for body_name in REQUIRED_BODY_NAMES:
            _, matches = robot.find_bodies(body_name)
            if matches != [body_name]:
                validation_errors.append(f"expected body {body_name!r} exactly once, found {matches}")
    except Exception as exc:
        validation_errors.append(f"runtime topology check failed: {type(exc).__name__}: {exc}")

    return {
        "root": str(asset_path),
        **dependency_result,
        "joint_names": joint_names,
        "body_names": body_names,
        "dof_count": dof_count,
        "articulation_roots": articulation_roots,
        "mount_joint_is_fixed": bool(mount_joint.IsValid() and mount_joint.IsA(UsdPhysics.FixedJoint)),
        "mount_body0_targets": mount_body0_targets,
        "mount_body1_targets": mount_body1_targets,
        "mount_joint_enabled": mount_joint_enabled,
        "mount_joint_exclude_from_articulation": mount_joint_exclude_from_articulation,
        "mount_parent_local_pos": mount_parent_local_pos,
        "expected_mount_parent_local_pos": expected_mount_parent_local_pos,
        "mount_plane_error_m": mount_plane_error_m,
        "mount_surface_top_z": mount_surface_top_z,
        "panda_visible_bottom_z": panda_visible_bottom_z,
        "mount_surface_gap_m": mount_surface_gap_m,
        "mount_child_local_pos": mount_child_local_pos,
        "mount_child_local_rot": mount_child_local_rot,
        "panda_root_joint_enabled": panda_root_joint_enabled,
        "runtime_initialized": runtime_initialized,
        "mount_relative_step_delta_m": mount_relative_step_delta_m,
        "physics_steps": 1 if physics_steps == 1 else 0,
        "validation_errors": validation_errors,
    }


def main() -> int:
    args = _build_parser().parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    result: dict[str, Any]
    exit_code = 0
    try:
        result = _verify_asset(args.asset, args.asset_root, args.device)
        if result["validation_errors"]:
            exit_code = 1
    except Exception as exc:
        exit_code = 1
        result = {
            "root": str(args.asset.resolve()),
            "dependencies": [],
            "joint_names": [],
            "body_names": [],
            "dof_count": None,
            "remote_dependencies": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(result, sort_keys=True))
    sys.stdout.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
