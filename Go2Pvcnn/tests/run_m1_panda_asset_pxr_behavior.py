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

from pxr import Sdf, Usd, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_m1_panda_asset.py"
ASSET = ROOT / "assets" / "m1_panda" / "m1_panda.usd"


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
    mount = stage.GetPrimAtPath("/M1Panda/Panda/panda_link0/AssemblerFixedJoint")
    assert mount.IsA(UsdPhysics.FixedJoint)
    joint = UsdPhysics.Joint(mount)
    assert [str(path) for path in joint.GetBody0Rel().GetTargets()] == ["/M1Panda/BASE_LINK"]
    assert [str(path) for path in joint.GetBody1Rel().GetTargets()] == ["/M1Panda/Panda/panda_link0"]
    assert joint.GetJointEnabledAttr().Get() is True
    assert mount.GetAttribute("physics:excludeFromArticulation").Get() is False
    assert tuple(joint.GetLocalPos1Attr().Get()) == (0.0, 0.0, 0.0)
    local_rot1 = joint.GetLocalRot1Attr().Get()
    assert (float(local_rot1.GetReal()), *tuple(local_rot1.GetImaginary())) == (1.0, 0.0, 0.0, 0.0)
    print(json.dumps({"cleanup": "pass", "roots": roots, "mount": "pass"}, sort_keys=True), flush=True)


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
