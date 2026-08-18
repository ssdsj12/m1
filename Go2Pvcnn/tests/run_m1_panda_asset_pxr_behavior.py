#!/usr/bin/env python3
"""PXR behavior checks for the Task 2/3 serialized asset contract."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


app = AppLauncher({"headless": True}).app

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_m1_panda_asset.py"
ASSET = ROOT / "assets" / "m1_panda" / "m1_panda.usd"
M1_ASSET = ROOT / "assets" / "m1_panda" / "m1_floating.usda"
MOUNT_PLANE_TOLERANCE_M = 1.0e-6
MOUNT_SURFACE_TOLERANCE_M = 1.0e-6
MOUNT_PATCH_HALF_EXTENTS_M = (0.11, 0.10)


def _load_cleanup_helper():
    tree = ast.parse(BUILDER.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "_remove_refresh_asset_edits" not in functions:
        raise AssertionError("builder is missing _remove_refresh_asset_edits")
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"ROOT_PRIM", "PANDA_PRIM"}
    ]
    namespace = {"Sdf": Sdf}
    exec(
        compile(
            ast.Module(
                body=constants + [functions["_require"], functions["_remove_refresh_asset_edits"]],
                type_ignores=[],
            ),
            str(BUILDER),
            "exec",
        ),
        namespace,
    )
    return namespace["_remove_refresh_asset_edits"]


def _all_asset_paths(proxy):
    fields = (proxy.prependedItems, proxy.appendedItems, proxy.addedItems, proxy.deletedItems, proxy.explicitItems)
    return [item.assetPath for field in fields for item in field]


def _mount_patch_top_z(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid()
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
    assert candidates
    return max(candidates)


def _visible_bottom_z(stage, prim_path):
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
        candidates.extend(
            float(transform.Transform(point)[2])
            for point in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get() or ()
        )
    assert candidates
    return min(candidates)


def _independent_mount_parent_local_pos():
    m1_stage = Usd.Stage.Open(str(M1_ASSET))
    assert m1_stage is not None
    base = m1_stage.GetPrimAtPath("/ZJ_V3_URDF_V1_0/BASE_LINK")
    assert base.IsValid()
    top_z = _mount_patch_top_z(
        m1_stage, "/ZJ_V3_URDF_V1_0/BASE_LINK"
    )
    origin_z = float(
        UsdGeom.Xformable(base)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()[2]
    )
    return (0.0, 0.0, top_z - origin_z)


def main() -> None:
    cleanup = _load_cleanup_helper()
    layer = Sdf.Layer.CreateAnonymous("cleanup-test.usda")
    m1 = Sdf.CreatePrimInLayer(layer, "/M1Panda")
    panda = Sdf.CreatePrimInLayer(layer, "/M1Panda/Panda")
    m1.referenceList.prependedItems = [Sdf.Reference("keep-m1.usd"), Sdf.Reference("/M1Panda")]
    m1.referenceList.deletedItems = [Sdf.Reference("/M1Panda")]
    panda.referenceList.prependedItems = [Sdf.Reference("keep-panda.usd"), Sdf.Reference("/M1Panda/Panda")]
    panda.referenceList.deletedItems = [Sdf.Reference("/M1Panda/Panda")]
    panda.payloadList.prependedItems = [Sdf.Payload("keep-payload.usd"), Sdf.Payload("/M1Panda/Panda")]
    panda.payloadList.deletedItems = [Sdf.Payload("/M1Panda/Panda")]

    cleanup(layer)
    assert _all_asset_paths(m1.referenceList) == ["keep-m1.usd"]
    assert _all_asset_paths(panda.referenceList) == ["keep-panda.usd"]
    assert _all_asset_paths(panda.payloadList) == ["keep-payload.usd"]

    stage = Usd.Stage.Open(str(ASSET))
    assert stage is not None
    roots = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    assert roots == ["/M1Panda/BASE_LINK"]
    mount_surface_top_z = _mount_patch_top_z(
        stage, "/M1Panda/BASE_LINK"
    )
    panda_visible_bottom_z = _visible_bottom_z(
        stage, "/M1Panda/Panda/panda_link0"
    )
    mount_surface_gap_m = panda_visible_bottom_z - mount_surface_top_z
    assert abs(mount_surface_gap_m) <= MOUNT_SURFACE_TOLERANCE_M
    mount = stage.GetPrimAtPath("/M1Panda/Panda/panda_link0/AssemblerFixedJoint")
    assert mount.IsA(UsdPhysics.FixedJoint)
    joint = UsdPhysics.Joint(mount)
    assert [str(path) for path in joint.GetBody0Rel().GetTargets()] == ["/M1Panda/BASE_LINK"]
    assert [str(path) for path in joint.GetBody1Rel().GetTargets()] == ["/M1Panda/Panda/panda_link0"]
    assert joint.GetJointEnabledAttr().Get() is True
    assert mount.GetAttribute("physics:excludeFromArticulation").Get() is False
    parent_local_pos = tuple(joint.GetLocalPos0Attr().Get())
    expected_parent_local_pos = _independent_mount_parent_local_pos()
    mount_plane_error_m = max(
        abs(float(actual) - float(expected))
        for actual, expected in zip(
            parent_local_pos, expected_parent_local_pos, strict=True
        )
    )
    assert mount_plane_error_m <= MOUNT_PLANE_TOLERANCE_M
    assert tuple(joint.GetLocalPos1Attr().Get()) == (0.0, 0.0, 0.0)
    local_rot1 = joint.GetLocalRot1Attr().Get()
    assert (float(local_rot1.GetReal()), *tuple(local_rot1.GetImaginary())) == (1.0, 0.0, 0.0, 0.0)
    print(
        json.dumps(
            {
                "cleanup": "pass",
                "roots": roots,
                "mount": "pass",
                "mount_parent_local_pos": parent_local_pos,
                "expected_mount_parent_local_pos": expected_parent_local_pos,
                "mount_plane_error_m": mount_plane_error_m,
                "mount_surface_top_z": mount_surface_top_z,
                "panda_visible_bottom_z": panda_visible_bottom_z,
                "mount_surface_gap_m": mount_surface_gap_m,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
