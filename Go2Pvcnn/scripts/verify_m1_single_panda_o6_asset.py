#!/usr/bin/env python3
"""Verify offline topology and a 2000-step hold for the single-O6 asset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


EXPECTED_ACTIVE_DOF_COUNT = 29
EXPECTED_PHYSICS_STEPS = 2000
EXPECTED_ARTICULATION_ROOT = "/M1SinglePandaO6/BASE_LINK"
MAX_MOUNT_POSITION_DRIFT_M = 1.0e-3
MAX_MOUNT_ORIENTATION_DRIFT_RAD = 1.0e-3
CONTACT_FORCE_THRESHOLD_N = 5.0
REMOTE_PREFIXES = ("omniverse://", "http://", "https://")
BUILTIN_MDL_ALLOWLIST = {"OmniPBR.mdl"}
FOOT_BODY_NAMES = (
    "FAR_FOOT_LINK",
    "FBL_FOOT_LINK",
    "RAR_FOOT_LINK",
    "RBL_FOOT_LINK",
)
MOUNT_BODY_PAIRS = (
    ("BASE_LINK", "panda_link0"),
    ("panda_link8", "right_hand_base_link"),
)
ASSEMBLY_JOINT_PATHS = (
    "/M1SinglePandaO6/joints/panda_mount_joint",
    "/M1SinglePandaO6/joints/right_hand_mount_joint",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=EXPECTED_PHYSICS_STEPS)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _dependency_report(
    asset: Path, project_assets_root: Path, usd_utils: Any
) -> dict[str, Any]:
    layers, assets, unresolved = usd_utils.ComputeAllDependencies(str(asset))
    dependencies = sorted(
        {str(layer.identifier) for layer in layers}
        | {
            str(getattr(item, "resolvedPath", "") or getattr(item, "path", item))
            for item in assets
        }
    )
    unresolved_items = sorted(str(item) for item in unresolved)
    builtin = [item for item in unresolved_items if item in BUILTIN_MDL_ALLOWLIST]
    unresolved_external = [
        item for item in unresolved_items if item not in BUILTIN_MDL_ALLOWLIST
    ]
    remote = [item for item in dependencies if item.startswith(REMOTE_PREFIXES)]
    outside_project = [
        item
        for item in dependencies
        if item
        and not item.startswith("anon:")
        and not item.endswith(tuple(BUILTIN_MDL_ALLOWLIST))
        and not _is_within(Path(item), project_assets_root)
    ]
    return {
        "dependencies": dependencies,
        "builtin_mdl_dependencies": builtin,
        "unresolved_dependencies": unresolved_external,
        "remote_dependencies": remote,
        "outside_project_dependencies": outside_project,
    }


def _enabled_joint(prim: Any, usd_physics: Any) -> bool:
    return usd_physics.Joint(prim).GetJointEnabledAttr().Get() is not False


def _value_is_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return _value_is_finite(value.GetReal()) and _value_is_finite(
            value.GetImaginary()
        )
    try:
        return all(_value_is_finite(item) for item in value)
    except TypeError:
        return True


def _authored_nonfinite_attributes(stage: Any, usd: Any) -> list[str]:
    checked_fragments = (
        "mass",
        "centerofmass",
        "diagonalinertia",
        "principalaxes",
        "lowerlimit",
        "upperlimit",
        "xformop:",
    )
    errors = []
    for prim in usd.PrimRange.Stage(stage, usd.TraverseInstanceProxies()):
        for attribute in prim.GetAttributes():
            name = attribute.GetName().lower()
            if not any(fragment in name for fragment in checked_fragments):
                continue
            if attribute.HasAuthoredValueOpinion() and not _value_is_finite(attribute.Get()):
                errors.append(f"{prim.GetPath()}.{attribute.GetName()}")
    return sorted(errors)


def _offline_report(
    asset: Path, asset_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics, UsdUtils

    stage = Usd.Stage.Open(str(asset), load=Usd.Stage.LoadAll)
    _require(stage is not None, f"failed to open asset: {asset}")
    dependencies = _dependency_report(asset, asset_root.parent, UsdUtils)
    articulation_roots = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    physical_dof_paths = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint))
        and _enabled_joint(prim, UsdPhysics)
    )
    body_paths = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    joint_paths = sorted(
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
    )
    body_counts = Counter(path.rsplit("/", 1)[-1] for path in body_paths)
    joint_counts = Counter(path.rsplit("/", 1)[-1] for path in joint_paths)
    duplicate_body_names = sorted(
        name for name, count in body_counts.items() if count > 1
    )
    duplicate_joint_names = sorted(
        name for name, count in joint_counts.items() if count > 1
    )
    inactive_assembly_joints = []
    for path in ASSEMBLY_JOINT_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsA(UsdPhysics.FixedJoint) or not _enabled_joint(prim, UsdPhysics):
            inactive_assembly_joints.append(path)

    nonconvex_o6_collisions = []
    o6_collision_count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not path.startswith("/M1SinglePandaO6/Panda/right_o6/"):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        o6_collision_count += 1
        approximation = (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else None
        )
        if approximation != "convexHull":
            nonconvex_o6_collisions.append(path)

    nonfinite_attributes = _authored_nonfinite_attributes(stage, Usd)
    errors = []
    if dependencies["unresolved_dependencies"]:
        errors.append(f"unresolved dependencies: {dependencies['unresolved_dependencies']}")
    if dependencies["remote_dependencies"]:
        errors.append(f"remote dependencies: {dependencies['remote_dependencies']}")
    if dependencies["outside_project_dependencies"]:
        errors.append(
            "dependencies outside project assets: "
            f"{dependencies['outside_project_dependencies']}"
        )
    if articulation_roots != [EXPECTED_ARTICULATION_ROOT]:
        errors.append(f"unexpected articulation roots: {articulation_roots}")
    if duplicate_body_names:
        errors.append(f"duplicate body names: {duplicate_body_names}")
    if duplicate_joint_names:
        errors.append(f"duplicate joint names: {duplicate_joint_names}")
    if inactive_assembly_joints:
        errors.append(f"inactive assembly joints: {inactive_assembly_joints}")
    if any("panda_finger_joint" in path for path in joint_paths):
        errors.append("arm-only asset contains panda_finger_joint")
    if nonfinite_attributes:
        errors.append(f"non-finite authored attributes: {nonfinite_attributes}")
    if o6_collision_count <= 0 or nonconvex_o6_collisions:
        errors.append(f"non-convex O6 collisions: {nonconvex_o6_collisions}")
    if manifest.get("schema") != 1:
        errors.append(f"unsupported manifest schema: {manifest.get('schema')}")
    if manifest.get("asset") != asset.name:
        errors.append(f"manifest asset mismatch: {manifest.get('asset')}")
    if manifest.get("asset_sha256") != _sha256(asset):
        errors.append("manifest asset SHA256 mismatch")
    source_manifest_path = asset_root.parent / "m1_dual_panda_o6/source_manifest.json"
    if not source_manifest_path.is_file():
        errors.append(f"missing source manifest: {source_manifest_path}")
    elif manifest.get("source_manifest_sha256") != _sha256(source_manifest_path):
        errors.append("manifest source SHA256 mismatch")
    if manifest.get("articulation_root") != EXPECTED_ARTICULATION_ROOT:
        errors.append("manifest articulation root mismatch")
    if manifest.get("assembly_joints") != list(ASSEMBLY_JOINT_PATHS):
        errors.append("manifest assembly joints mismatch")
    if manifest.get("physical_dof_count") != len(physical_dof_paths):
        errors.append("manifest physical DOF count mismatch")
    if manifest.get("active_dof_count") != EXPECTED_ACTIVE_DOF_COUNT:
        errors.append("manifest active DOF count mismatch")
    if manifest.get("o6_collision_approximation") != "convexHull":
        errors.append("manifest O6 collision approximation mismatch")
    return {
        **dependencies,
        "articulation_roots": articulation_roots,
        "physical_dof_paths": physical_dof_paths,
        "body_paths": body_paths,
        "joint_paths": joint_paths,
        "duplicate_body_names": duplicate_body_names,
        "duplicate_joint_names": duplicate_joint_names,
        "inactive_assembly_joints": inactive_assembly_joints,
        "nonfinite_authored_attributes": nonfinite_attributes,
        "o6_collision_count": o6_collision_count,
        "nonconvex_o6_collisions": nonconvex_o6_collisions,
        "offline_errors": errors,
    }


def _robot_cfg(asset: Path):
    import isaaclab.sim as sim_utils

    from go2_pvcnn.assets.m1_single_panda_o6 import M1_SINGLE_PANDA_O6_CFG

    cfg = M1_SINGLE_PANDA_O6_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=str(asset),
        activate_contact_sensors=True,
        visible=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    )
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, 0.6115))
    return cfg


def _relative_mount_poses(robot: Any, math_utils: Any) -> list[tuple[Any, Any]]:
    result = []
    for parent_name, child_name in MOUNT_BODY_PAIRS:
        parent_id = robot.body_names.index(parent_name)
        child_id = robot.body_names.index(child_name)
        position, quaternion = math_utils.subtract_frame_transforms(
            robot.data.body_link_pos_w[:, parent_id],
            robot.data.body_link_quat_w[:, parent_id],
            robot.data.body_link_pos_w[:, child_id],
            robot.data.body_link_quat_w[:, child_id],
        )
        result.append((position, quaternion))
    return result


def _orientation_error(current: Any, initial: Any) -> Any:
    import torch

    current = torch.nn.functional.normalize(current.to(torch.float64), dim=-1)
    initial = torch.nn.functional.normalize(initial.to(torch.float64), dim=-1)
    cosine = torch.abs(torch.sum(current * initial, dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(cosine)


def _runtime_report(asset: Path, steps: int, device: str) -> dict[str, Any]:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils import math as math_utils

    from go2_pvcnn.assets.m1_single_panda_o6 import resolve_active_joint_ids

    _require(steps > 0, "steps must be positive")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=device, dt=0.005))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)
    robot = Articulation(_robot_cfg(asset))
    contact_sensors = [
        ContactSensor(ContactSensorCfg(prim_path=prim_path, update_period=0.0, debug_vis=False))
        for prim_path in (
            "/World/Robot/.*",
            "/World/Robot/Panda/.*",
            "/World/Robot/Panda/right_o6/.*",
        )
    ]
    sim.reset()
    _require(robot.is_initialized, "Isaac Lab articulation did not initialize")
    robot.reset()
    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos, robot.data.default_joint_vel
    )
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    robot.update(0.0)

    active_ids = resolve_active_joint_ids(tuple(robot.joint_names))
    active_names = [robot.joint_names[index] for index in active_ids]
    initial_mounts = [
        (position.clone(), quaternion.clone())
        for position, quaternion in _relative_mount_poses(robot, math_utils)
    ]
    initial_root_position = robot.data.root_link_pos_w.clone()
    max_position_drift = 0.0
    max_orientation_drift = 0.0
    nonfinite_count = 0
    hard_joint_limit_count = 0
    unexpected_contact_count = 0
    unexpected_reset_count = 0
    base_instability_count = 0
    all_four_contact_steps = 0
    contact_measurement_available = all(sensor.is_initialized for sensor in contact_sensors)
    limits = getattr(robot.data, "joint_pos_limits", robot.data.soft_joint_pos_limits)
    max_joint_limit_violation = torch.zeros(
        robot.num_joints, device=robot.device, dtype=robot.data.joint_pos.dtype
    )

    for _ in range(steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        current_mounts = _relative_mount_poses(robot, math_utils)
        for (position, quaternion), (initial_position, initial_quaternion) in zip(
            current_mounts, initial_mounts, strict=True
        ):
            max_position_drift = max(
                max_position_drift,
                float(
                    torch.linalg.vector_norm(
                        position - initial_position, dim=-1
                    ).max().item()
                ),
            )
            max_orientation_drift = max(
                max_orientation_drift,
                float(_orientation_error(quaternion, initial_quaternion).max().item()),
            )

        finite = all(
            bool(torch.isfinite(value).all())
            for value in (
                robot.data.root_link_state_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
                robot.data.body_link_state_w,
            )
        )
        nonfinite_count += int(not finite)
        below = robot.data.joint_pos < limits[..., 0] - 1.0e-4
        above = robot.data.joint_pos > limits[..., 1] + 1.0e-4
        hard_joint_limit_count += int(bool((below | above).any()))
        violation = torch.maximum(
            limits[..., 0] - robot.data.joint_pos,
            robot.data.joint_pos - limits[..., 1],
        ).clamp_min(0.0)
        max_joint_limit_violation = torch.maximum(
            max_joint_limit_violation, violation.max(dim=0).values
        )
        root_displacement = torch.linalg.vector_norm(
            robot.data.root_link_pos_w - initial_root_position, dim=-1
        )
        unexpected_reset_count += int(bool((root_displacement > 5.0).any()))
        root_height = robot.data.root_link_pos_w[:, 2]
        root_up_z = math_utils.quat_apply(
            robot.data.root_link_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=robot.device).expand(
                robot.num_instances, -1
            ),
        )[:, 2]
        base_instability_count += int(
            bool(((root_height < 0.30) | (root_up_z < 0.75)).any())
        )

        if contact_measurement_available:
            foot_forces = {name: 0.0 for name in FOOT_BODY_NAMES}
            unexpected_this_step = False
            for sensor in contact_sensors:
                magnitudes = torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1)
                for body_id, name in enumerate(sensor.body_names):
                    force = float(magnitudes[:, body_id].max().item())
                    if name in foot_forces:
                        foot_forces[name] = max(foot_forces[name], force)
                    elif force > CONTACT_FORCE_THRESHOLD_N:
                        unexpected_this_step = True
            all_four_contact_steps += int(
                all(force > CONTACT_FORCE_THRESHOLD_N for force in foot_forces.values())
            )
            unexpected_contact_count += int(unexpected_this_step)

    joint_limit_violations = {
        name: float(value)
        for name, value in zip(
            robot.joint_names,
            max_joint_limit_violation.detach().cpu().tolist(),
            strict=True,
        )
        if value > 0.0
    }
    return {
        "physics_steps": steps,
        "measured_physical_dof_count": int(robot.num_joints),
        "runtime_joint_names": list(robot.joint_names),
        "runtime_body_names": list(robot.body_names),
        "active_control_names": active_names,
        "active_control_count": len(active_names),
        "four_wheel_contact_ratio": all_four_contact_steps / steps,
        "max_mount_position_drift_m": max_position_drift,
        "max_mount_orientation_drift_rad": max_orientation_drift,
        "nonfinite_count": nonfinite_count,
        "hard_joint_limit_count": hard_joint_limit_count,
        "max_joint_limit_violation_by_name": joint_limit_violations,
        "unexpected_contact_count": unexpected_contact_count,
        "unexpected_reset_count": unexpected_reset_count,
        "base_instability_count": base_instability_count,
        "contact_measurement_available": contact_measurement_available,
    }


def _atomic_update_manifest(path: Path, runtime: dict[str, Any]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "measured_physical_dof_count": runtime["measured_physical_dof_count"],
            "verified_physics_steps": runtime["physics_steps"],
            "runtime_joint_names": runtime["runtime_joint_names"],
            "runtime_body_names": runtime["runtime_body_names"],
            "verified_limits": {
                "contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
                "max_mount_position_drift_m": MAX_MOUNT_POSITION_DRIFT_M,
                "max_mount_orientation_drift_rad": MAX_MOUNT_ORIENTATION_DRIFT_RAD,
                "required_four_wheel_contact_ratio": 1.0,
            },
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify(asset_root: Path, steps: int, device: str) -> dict[str, Any]:
    asset_root = asset_root.resolve(strict=True)
    asset = asset_root / "m1_single_panda_o6.usd"
    manifest_path = asset_root / "asset_manifest.json"
    _require(asset.is_file(), f"missing combined asset: {asset}")
    _require(manifest_path.is_file(), f"missing asset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    offline = _offline_report(asset, asset_root, manifest)
    runtime = _runtime_report(asset, steps, device)
    hard_gates_passed = (
        not offline["offline_errors"]
        and runtime["measured_physical_dof_count"] == manifest["physical_dof_count"]
        and runtime["active_control_count"] == EXPECTED_ACTIVE_DOF_COUNT
        and runtime["physics_steps"] == EXPECTED_PHYSICS_STEPS
        and offline["articulation_roots"] == [EXPECTED_ARTICULATION_ROOT]
        and runtime["four_wheel_contact_ratio"] == 1.0
        and runtime["max_mount_position_drift_m"] <= MAX_MOUNT_POSITION_DRIFT_M
        and runtime["max_mount_orientation_drift_rad"]
        <= MAX_MOUNT_ORIENTATION_DRIFT_RAD
        and runtime["nonfinite_count"] == 0
        and runtime["hard_joint_limit_count"] == 0
        and runtime["unexpected_contact_count"] == 0
        and runtime["unexpected_reset_count"] == 0
        and runtime["base_instability_count"] == 0
        and runtime["contact_measurement_available"]
    )
    result = {
        "asset": str(asset),
        **offline,
        **runtime,
        "single_articulation_root": offline["articulation_roots"]
        == [EXPECTED_ARTICULATION_ROOT],
        "hard_gates_passed": hard_gates_passed,
    }
    if hard_gates_passed:
        _atomic_update_manifest(manifest_path, runtime)
    return result


def main() -> int:
    cli_args = _build_parser().parse_args()
    app_launcher = AppLauncher(cli_args)
    simulation_app = app_launcher.app
    try:
        result = _verify(cli_args.asset_root, cli_args.steps, cli_args.device)
        exit_code = 0 if result["hard_gates_passed"] else 1
    except Exception as exc:
        result = {
            "asset_root": str(cli_args.asset_root.resolve()),
            "hard_gates_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        exit_code = 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    sys.stdout.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
