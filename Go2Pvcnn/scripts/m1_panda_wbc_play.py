#!/usr/bin/env python3
"""Run the deterministic C0 whole-body Teacher on the combined M1 + Panda."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import traceback

import torch


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from go2_pvcnn.control.m1_panda_coordination.rolling_contact import (
    RollingContactCfg,
    build_wheel_contact_jacobian,
)
from go2_pvcnn.control.m1_panda_coordination import runtime_adapter as _runtime_adapter

TASK_ID = "Isaac-M1-Panda-Wbc-Teacher-C0-v0"
PHYSICS_DT = 0.005
SETTLE_STEPS = 100
WHEEL_RADIUS_M = 0.0959
MAX_CONTINUOUS_ARM_TARGET_STEP_RAD = 0.050001
WHEEL_BODY_NAMES = (
    "FAR_FOOT_LINK",
    "FBL_FOOT_LINK",
    "RAR_FOOT_LINK",
    "RBL_FOOT_LINK",
)


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Play the deterministic M1 + Panda prioritized WBC Teacher."
    )
    parser.add_argument("--steps", type=int, default=0, help="Physics steps; zero runs until the app closes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--disable-target-motion", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--stats-interval", type=int, default=100)
    # AppLauncher supplies --headless and --device.
    AppLauncher.add_app_launcher_args(parser)
    return parser


def validate_args(args) -> None:
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.num_envs != 1:
        raise ValueError("C0 supports exactly one environment")
    if args.stats_interval <= 0:
        raise ValueError("--stats-interval must be positive")


def atomic_write_summary(path: Path, payload: dict) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, path)


@dataclass
class C0Summary:
    seed: int
    steps: int = 0
    finite: bool = True
    qp_feasible_count: int = 0
    max_ee_position_error_m: float = 0.0
    min_singular_value: float = math.inf
    max_abs_roll_rad: float = 0.0
    max_abs_pitch_rad: float = 0.0
    max_lateral_slip_mps: float = 0.0
    joint_limit_violations: int = 0
    base_contacts: int = 0
    self_collisions: int = 0
    safety_state_counts: dict[str, int] = field(default_factory=dict)
    safety_reason_counts: dict[str, int] = field(default_factory=dict)
    base_activation_count: int = 0
    first_base_activation_step: int | None = None
    first_singularity_crossing_step: int | None = None
    max_arm_target_step_rad: float = 0.0
    arm_snap_count: int = 0
    _previous_arm_target: torch.Tensor | None = field(default=None, repr=False)
    reset_count: int = 0
    exit_reason: str = "not_started"

    def to_dict(self) -> dict:
        denominator = max(self.steps, 1)
        sigma = self.min_singular_value
        if not math.isfinite(sigma):
            sigma = 0.0
        return {
            "seed": self.seed,
            "steps": self.steps,
            "finite": self.finite,
            "qp_feasible_count": self.qp_feasible_count,
            "qp_feasible_rate": self.qp_feasible_count / denominator,
            "max_ee_position_error_m": self.max_ee_position_error_m,
            "min_singular_value": sigma,
            "max_abs_roll_rad": self.max_abs_roll_rad,
            "max_abs_pitch_rad": self.max_abs_pitch_rad,
            "max_lateral_slip_mps": self.max_lateral_slip_mps,
            "joint_limit_violations": self.joint_limit_violations,
            "base_contacts": self.base_contacts,
            "self_collisions": self.self_collisions,
            "safety_state_counts": dict(sorted(self.safety_state_counts.items())),
            "safety_reason_counts": dict(sorted(self.safety_reason_counts.items())),
            "base_activation_count": self.base_activation_count,
            "first_base_activation_step": self.first_base_activation_step,
            "first_singularity_crossing_step": self.first_singularity_crossing_step,
            "max_arm_target_step_rad": self.max_arm_target_step_rad,
            "arm_snap_count": self.arm_snap_count,
            "reset_count": self.reset_count,
            "exit_reason": self.exit_reason,
        }


class StaticPoseTrajectory:
    """Trajectory-compatible zero-motion target used by the smoke command."""

    def reset(self, initial_pose: torch.Tensor, *, seed: int) -> None:
        del seed
        self._pose = initial_pose.detach().clone()

    def sample(self, time_s: float):
        del time_s
        from go2_pvcnn.control.m1_panda_coordination.trajectory import TrajectorySample

        return TrajectorySample(
            pose=self._pose.clone(),
            twist=torch.zeros_like(self._pose),
            acceleration=torch.zeros_like(self._pose),
        )


def _exact_id(names, expected: str, kind: str) -> int:
    matches = [index for index, name in enumerate(names) if name == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {kind} named {expected!r}; got {matches}")
    return matches[0]


def _cpu64(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64).clone()


def build_teacher_gains() -> tuple[torch.Tensor, torch.Tensor]:
    """Return impedance gains in the reference QP dtype and canonical order."""

    kp = torch.cat(
        (
            torch.full((12,), 120.0, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
            torch.full((7,), 80.0, dtype=torch.float64),
        )
    )
    kd = torch.cat(
        (
            torch.full((12,), 20.0, dtype=torch.float64),
            torch.full((4,), 2.0, dtype=torch.float64),
            torch.tensor([4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0], dtype=torch.float64),
        )
    )
    return kp, kd


def read_generalized_bias_force(root_view, *, generalized_dof: int) -> torch.Tensor:
    """Read C(q, qd)+g(q), upgrading legacy joint-only floating-base APIs."""

    coriolis_force = _cpu64(root_view.get_coriolis_and_centrifugal_forces()[0])
    gravity_force = _cpu64(root_view.get_generalized_gravity_forces()[0])
    if coriolis_force.shape != (generalized_dof,) or gravity_force.shape != (
        generalized_dof,
    ):
        coriolis_force = _cpu64(
            root_view.get_coriolis_and_centrifugal_compensation_forces()[0]
        )
        gravity_force = _cpu64(root_view.get_gravity_compensation_forces()[0])
    if coriolis_force.shape != (generalized_dof,) or gravity_force.shape != (
        generalized_dof,
    ):
        raise RuntimeError(
            "PhysX generalized bias forces do not match the mass-matrix dimension"
        )
    bias_force = coriolis_force + gravity_force
    return bias_force


def contact_point_linear_jacobian(
    body_jacobian: torch.Tensor, point_offset_w: torch.Tensor
) -> torch.Tensor:
    """Map generalized velocity to a rigid body's offset-point linear velocity."""

    x, y, z = point_offset_w
    skew = point_offset_w.new_zeros((3, 3))
    skew[0, 1], skew[0, 2] = -z, y
    skew[1, 0], skew[1, 2] = z, -x
    skew[2, 0], skew[2, 1] = -y, x
    return body_jacobian[:3] - skew @ body_jacobian[3:]


def relative_axis_angle(math_utils, current_quat: torch.Tensor, reference_quat: torch.Tensor) -> torch.Tensor:
    """Express current orientation as a rotation from a fixed reference."""

    relative_quat = math_utils.quat_mul(current_quat, math_utils.quat_inv(reference_quat))
    return math_utils.axis_angle_from_quat(relative_quat)


class PhysxTeacherAdapter:
    """Translate one live Isaac articulation into explicit C0 Teacher tensors."""

    def __init__(
        self,
        env,
        math_utils,
        wheel_radius_m: float = WHEEL_RADIUS_M,
    ):
        from go2_pvcnn.control.m1_panda_coordination.contracts import WbcJointMap

        self.env = env
        self.robot = env.scene["robot"]
        self.contact_sensor = env.scene["contact_forces"]
        self.math_utils = math_utils
        self.wheel_radius_m = float(wheel_radius_m)
        self.rolling_contact_cfg = RollingContactCfg(
            wheel_radius_m=self.wheel_radius_m
        )
        robot = self.robot
        if robot.num_instances != 1:
            raise RuntimeError("C0 supports exactly one articulation instance")
        self.joint_map = WbcJointMap.resolve(robot.joint_names)
        action_term = env.action_manager.get_term("joint_effort")
        expected_action_names = [robot.joint_names[index] for index in self.joint_map.controlled.tolist()]
        if list(action_term._joint_names) != expected_action_names:
            raise RuntimeError(
                "joint-effort action order does not match canonical WBC order: "
                f"{action_term._joint_names} != {expected_action_names}"
            )
        self.controlled_joint_ids_device = self.joint_map.controlled.to(robot.device)
        # This expression is intentionally explicit: PhysX generalized columns
        # prepend the six floating-base coordinates to articulation joint order.
        self.controlled_generalized_indices = self.joint_map.controlled + 6
        self.leg_generalized_indices = self.joint_map.legs + 6
        self.wheel_generalized_indices = self.joint_map.wheels + 6
        self.arm_generalized_indices = self.joint_map.panda_arm + 6
        self.hand_body_id = _exact_id(robot.body_names, "panda_hand", "body")
        self.mount_body_id = _exact_id(robot.body_names, "panda_link0", "body")
        self.base_body_id = _exact_id(robot.body_names, "BASE_LINK", "body")
        self.wheel_body_ids = torch.tensor(
            [_exact_id(robot.body_names, name, "body") for name in WHEEL_BODY_NAMES],
            dtype=torch.long,
            device=robot.device,
        )
        sensor_ids, sensor_names = self.contact_sensor.find_bodies(
            list(WHEEL_BODY_NAMES), preserve_order=True
        )
        if tuple(sensor_names) != WHEEL_BODY_NAMES:
            raise RuntimeError(f"wheel contact sensor order mismatch: {sensor_names}")
        self.wheel_sensor_ids = torch.tensor(sensor_ids, dtype=torch.long, device=robot.device)
        base_sensor_ids, _ = self.contact_sensor.find_bodies("BASE_LINK", preserve_order=True)
        if len(base_sensor_ids) != 1:
            raise RuntimeError("BASE_LINK contact sensor did not resolve uniquely")
        self.base_sensor_id = int(base_sensor_ids[0])
        self._previous_contact_jacobian = None
        self._initial_root_pos = _cpu64(robot.data.root_pos_w[0])
        self._initial_root_quat = robot.data.root_quat_w[0].detach().clone()
        self._initial_hand_quat = robot.data.body_quat_w[
            0, self.hand_body_id
        ].detach().clone()
        self._initial_arm_q = _cpu64(
            robot.data.joint_pos[0].index_select(
                0, self.joint_map.panda_arm.to(robot.device)
            )
        )
        initial_euler = self.math_utils.euler_xyz_from_quat(
            self._initial_root_quat.unsqueeze(0)
        )
        self._initial_rpy = _cpu64(torch.stack(initial_euler, dim=-1)[0])
        self.latest_root_height = float(robot.data.root_pos_w[0, 2].item())
        self.latest_wheel_heights = [
            float(value)
            for value in robot.data.body_pos_w[0]
            .index_select(0, self.wheel_body_ids)[:, 2]
            .detach()
            .cpu()
            .tolist()
        ]

    @staticmethod
    def _body_jacobian(
        jacobians: torch.Tensor, body_id: int, body_count: int
    ) -> torch.Tensor:
        """Resolve PhysX body rows across root-inclusive and legacy layouts."""

        jacobian_body_count = jacobians.shape[1]
        if jacobian_body_count == body_count:
            row = body_id
        elif jacobian_body_count == body_count - 1:
            if body_id <= 0:
                raise ValueError(
                    "the floating root link has no PhysX body Jacobian row"
                )
            row = body_id - 1
        else:
            raise ValueError(
                "PhysX Jacobian body count does not match articulation bodies"
            )
        return jacobians[0, row]

    def _generalized_velocity(self) -> torch.Tensor:
        data = self.robot.data
        return _cpu64(
            torch.cat((data.root_lin_vel_w[0], data.root_ang_vel_w[0], data.joint_vel[0]))
        )

    def _contact_metrics(self) -> tuple[int, float, int]:
        forces = self.contact_sensor.data.net_forces_w[0]
        wheel_forces = forces.index_select(0, self.wheel_sensor_ids)
        self.latest_measured_wheel_forces = _cpu64(wheel_forces)
        wheel_contact = torch.linalg.vector_norm(wheel_forces, dim=-1) > 1.0
        wheel_vel_w = self.robot.data.body_lin_vel_w[0].index_select(0, self.wheel_body_ids)
        root_quat = self.robot.data.root_quat_w[0].expand(4, -1)
        wheel_vel_b = self.math_utils.quat_apply_inverse(root_quat, wheel_vel_w)
        lateral = torch.where(wheel_contact, wheel_vel_b[:, 1].abs(), torch.zeros_like(wheel_vel_b[:, 1]))
        base_contact = int(torch.linalg.vector_norm(forces[self.base_sensor_id]).item() > 10.0)
        return int(wheel_contact.sum().item()), float(lateral.max().item()), base_contact

    def _pose_and_orientation(self) -> tuple[torch.Tensor, float, float, float]:
        data = self.robot.data
        hand_position = _cpu64(data.body_pos_w[0, self.hand_body_id])
        hand_quat = data.body_quat_w[0, self.hand_body_id].unsqueeze(0)
        hand_axis_angle = _cpu64(
            relative_axis_angle(
                self.math_utils,
                hand_quat,
                self._initial_hand_quat.unsqueeze(0),
            )[0]
        )
        root_euler = self.math_utils.euler_xyz_from_quat(data.root_quat_w[0].unsqueeze(0))
        roll, pitch, yaw = (float(component[0].item()) for component in root_euler)
        return torch.cat((hand_position, hand_axis_angle)), roll, pitch, yaw

    def build_state(self, physics_step: int):
        from go2_pvcnn.control.m1_panda_coordination.kinematics import singularity_metrics
        from go2_pvcnn.control.m1_panda_coordination.standing_wbc import StandingWbcInput
        from go2_pvcnn.control.m1_panda_coordination.teacher import TeacherState

        robot = self.robot
        data = robot.data
        self.latest_root_height = float(data.root_pos_w[0, 2].item())
        self.latest_wheel_heights = [
            float(value)
            for value in data.body_pos_w[0]
            .index_select(0, self.wheel_body_ids)[:, 2]
            .detach()
            .cpu()
            .tolist()
        ]
        root_view = robot.root_physx_view
        mass_matrix = _cpu64(root_view.get_generalized_mass_matrices()[0])
        bias_force = read_generalized_bias_force(
            root_view, generalized_dof=mass_matrix.shape[0]
        )
        jacobians_device = root_view.get_jacobians()
        jacobians = _cpu64(jacobians_device)
        generalized_velocity = self._generalized_velocity()

        wheel_body_jacobians = torch.stack(
            [
                self._body_jacobian(
                    jacobians, int(index), len(robot.body_names)
                )
                for index in self.wheel_body_ids.cpu()
            ],
            dim=0,
        )
        contact_jacobian = build_wheel_contact_jacobian(
            wheel_body_jacobians,
            self.rolling_contact_cfg,
        )
        if self._previous_contact_jacobian is None:
            contact_jacobian_dot_qd = torch.zeros(12, dtype=torch.float64)
        else:
            contact_jacobian_dot_qd = (
                (contact_jacobian - self._previous_contact_jacobian) / PHYSICS_DT
            ) @ generalized_velocity
        self._previous_contact_jacobian = contact_jacobian.clone()

        hand_jacobian = self._body_jacobian(
            jacobians, self.hand_body_id, len(robot.body_names)
        )
        self.latest_hand_base_jacobian = hand_jacobian[:, :6].clone()
        mount_jacobian = self._body_jacobian(
            jacobians, self.mount_body_id, len(robot.body_names)
        )
        panda_jacobian = hand_jacobian.index_select(1, self.arm_generalized_indices)
        coordinated_columns = torch.cat(
            (torch.tensor([0, 1, 5], dtype=torch.long), self.arm_generalized_indices)
        )
        coordinated_jacobian = hand_jacobian.index_select(1, coordinated_columns)
        sigma_min, _ = singularity_metrics(panda_jacobian)
        self.latest_arm_jacobian_twist = panda_jacobian @ _cpu64(
            data.joint_vel[0].index_select(0, self.joint_map.panda_arm.to(robot.device))
        )
        hand_offset_w = _cpu64(
            data.body_pos_w[0, self.hand_body_id] - data.root_pos_w[0]
        )
        base_linear_at_hand = _cpu64(data.root_lin_vel_w[0]) + torch.linalg.cross(
            _cpu64(data.root_ang_vel_w[0]), hand_offset_w
        )
        self.latest_measured_arm_twist = torch.cat(
            (
                _cpu64(data.body_lin_vel_w[0, self.hand_body_id]) - base_linear_at_hand,
                _cpu64(data.body_ang_vel_w[0, self.hand_body_id])
                - _cpu64(data.root_ang_vel_w[0]),
            )
        )

        ee_pose, roll, pitch, yaw = self._pose_and_orientation()
        root_position = _cpu64(data.root_pos_w[0])
        joint_position = _cpu64(data.joint_pos[0])
        joint_velocity = _cpu64(data.joint_vel[0])
        controlled_q = joint_position.index_select(0, self.joint_map.controlled)
        controlled_qd = joint_velocity.index_select(0, self.joint_map.controlled)
        arm_q = joint_position.index_select(0, self.joint_map.panda_arm)
        arm_qd = joint_velocity.index_select(0, self.joint_map.panda_arm)
        root_linear_velocity = _cpu64(data.root_lin_vel_w[0])
        root_angular_velocity = _cpu64(data.root_ang_vel_w[0])
        self.latest_root_xy_yaw = torch.cat(
            (root_position[:2], root_position.new_tensor((yaw,)))
        )
        self.latest_root_vxy_yawrate = torch.cat(
            (
                root_linear_velocity[:2],
                root_angular_velocity[2:3],
            )
        )
        self.latest_generalized_velocity = generalized_velocity.clone()
        self.latest_contact_jacobian = contact_jacobian.clone()
        self.latest_wheel_velocity = controlled_qd[12:16].clone()
        coord_q = torch.cat(
            (root_position[:2], torch.tensor([yaw], dtype=torch.float64), arm_q)
        )
        coord_qd = torch.cat((root_linear_velocity[:2], root_angular_velocity[2:3], arm_qd))

        joint_limits = _cpu64(data.soft_joint_pos_limits[0])
        arm_limits = joint_limits.index_select(0, self.joint_map.panda_arm)
        coord_q_min = torch.cat((self._initial_root_pos[:2] - 0.25, self._initial_rpy[2:3] - 0.5, arm_limits[:, 0]))
        coord_q_max = torch.cat((self._initial_root_pos[:2] + 0.25, self._initial_rpy[2:3] + 0.5, arm_limits[:, 1]))
        velocity_limits = _cpu64(data.joint_vel_limits[0]).index_select(0, self.joint_map.panda_arm)
        coord_v_max = torch.cat(
            (
                torch.tensor([0.5, 0.5, 0.75], dtype=torch.float64),
                velocity_limits.clamp(max=2.5),
            )
        )
        coord_a_max = torch.cat(
            (
                torch.tensor([3.0, 3.0, 4.0], dtype=torch.float64),
                torch.full((7,), 100.0, dtype=torch.float64),
            )
        )

        base_jacobian = torch.zeros((6, 31), dtype=torch.float64)
        base_jacobian[:, :6] = torch.eye(6, dtype=torch.float64)
        balance_jacobian = base_jacobian.index_select(0, torch.tensor([2, 3, 4]))
        position_error = self._initial_root_pos - root_position
        rpy_error = self._initial_rpy - torch.tensor([roll, pitch, yaw], dtype=torch.float64)
        base_acceleration = torch.cat((
            40.0 * position_error - 12.0 * root_linear_velocity,
            40.0 * rpy_error - 12.0 * root_angular_velocity,
        ))
        balance_acceleration = base_acceleration.index_select(0, torch.tensor([2, 3, 4]))
        default_joint_position = _cpu64(data.default_joint_pos[0])
        leg_q = joint_position.index_select(0, self.joint_map.legs)
        leg_qd = joint_velocity.index_select(0, self.joint_map.legs)
        leg_default = default_joint_position.index_select(0, self.joint_map.legs)
        leg_acceleration = 80.0 * (leg_default - leg_q) - 12.0 * leg_qd
        wheel_acceleration = -8.0 * joint_velocity.index_select(0, self.joint_map.wheels)

        qdd_lower = torch.full((31,), -100.0, dtype=torch.float64)
        qdd_upper = torch.full((31,), 100.0, dtype=torch.float64)
        effort_limit = _cpu64(data.joint_effort_limits[0]).index_select(0, self.joint_map.controlled)
        effort_limit = effort_limit.abs().clamp_min(1.0)
        wheel_contact_count, max_lateral_slip, base_contact = self._contact_metrics()
        finite_tensors = (
            mass_matrix, bias_force, jacobians, ee_pose, controlled_q, controlled_qd
        )
        signals_finite = all(bool(torch.isfinite(value).all().item()) for value in finite_tensors)

        wbc_input = StandingWbcInput(
            mass_matrix=mass_matrix,
            bias_force=bias_force,
            contact_jacobian=contact_jacobian,
            contact_jacobian_dot_qd=contact_jacobian_dot_qd,
            mount_wrench_jacobian=mount_jacobian,
            external_wrench=torch.zeros(6, dtype=torch.float64),
            balance_jacobian=balance_jacobian,
            balance_acceleration=balance_acceleration,
            base_jacobian=base_jacobian,
            base_acceleration=base_acceleration,
            leg_generalized_indices=self.leg_generalized_indices,
            wheel_generalized_indices=self.wheel_generalized_indices,
            arm_generalized_indices=self.arm_generalized_indices,
            leg_acceleration=leg_acceleration,
            wheel_acceleration=wheel_acceleration,
            arm_acceleration=torch.zeros(7, dtype=torch.float64),
            qdd_lower=qdd_lower,
            qdd_upper=qdd_upper,
            effort_limit=effort_limit,
            friction_coefficient=1.0,
        )
        state = TeacherState(
            physics_step=physics_step,
            time_s=physics_step * PHYSICS_DT,
            ee_pose=ee_pose,
            coordinated_jacobian=coordinated_jacobian,
            coord_q=coord_q,
            coord_qd=coord_qd,
            coord_q_min=coord_q_min,
            coord_q_max=coord_q_max,
            coord_v_max=coord_v_max,
            coord_a_max=coord_a_max,
            manipulability_gradient=torch.cat(
                (
                    torch.zeros(3, dtype=torch.float64),
                    self._initial_arm_q - arm_q,
                )
            ),
            sigma_min=sigma_min,
            wbc_input=wbc_input,
            controlled_q=controlled_q,
            controlled_qd=controlled_qd,
            roll=roll,
            pitch=pitch,
            wheel_contact_count=wheel_contact_count,
            max_lateral_slip=max_lateral_slip,
            signals_finite=signals_finite,
        )
        return state, base_contact

    def joint_limit_violations(self) -> int:
        positions = self.robot.data.joint_pos[0].index_select(
            0, self.controlled_joint_ids_device
        )
        limits = self.robot.data.soft_joint_pos_limits[0].index_select(
            0, self.controlled_joint_ids_device
        )
        return int(((positions < limits[:, 0]) | (positions > limits[:, 1])).sum().item())


PhysxTeacherAdapter = _runtime_adapter.PhysxTeacherAdapter
build_teacher_gains = _runtime_adapter.build_teacher_gains
contact_point_linear_jacobian = _runtime_adapter.contact_point_linear_jacobian
read_generalized_bias_force = _runtime_adapter.read_generalized_bias_force
relative_axis_angle = _runtime_adapter.relative_axis_angle


def _termination_cause(env, terminated: torch.Tensor, truncated: torch.Tensor) -> str:
    manager = env.termination_manager
    active = []
    for name in ("base_contact", "bad_orientation", "time_out"):
        try:
            value = manager.get_term(name)
        except Exception:
            value = None
        if value is not None and bool(torch.as_tensor(value).any().item()):
            active.append(name)
    if active:
        return "+".join(active)
    if bool(torch.as_tensor(terminated).any().item()):
        return "terminated"
    if bool(torch.as_tensor(truncated).any().item()):
        return "truncated"
    return "none"


def _update_summary(summary: C0Summary, state, command, base_contact: int, adapter) -> None:
    ee_error = float(torch.linalg.vector_norm(command.target_pose[:3] - state.ee_pose[:3]).item())
    sigma_min = float(state.sigma_min.item())
    summary.steps += 1
    summary.finite = summary.finite and state.signals_finite and bool(torch.isfinite(command.effort).all().item())
    summary.qp_feasible_count += int(command.qp_result.success)
    summary.max_ee_position_error_m = max(summary.max_ee_position_error_m, ee_error)
    summary.min_singular_value = min(summary.min_singular_value, sigma_min)
    summary.max_abs_roll_rad = max(summary.max_abs_roll_rad, abs(state.roll))
    summary.max_abs_pitch_rad = max(summary.max_abs_pitch_rad, abs(state.pitch))
    summary.max_lateral_slip_mps = max(summary.max_lateral_slip_mps, state.max_lateral_slip)
    summary.joint_limit_violations += adapter.joint_limit_violations()
    summary.base_contacts += base_contact
    safety = command.safety_state.name
    summary.safety_state_counts[safety] = summary.safety_state_counts.get(safety, 0) + 1
    reason = command.safety_reason
    summary.safety_reason_counts[reason] = summary.safety_reason_counts.get(reason, 0) + 1
    if bool(command.motion_distribution.base_active.item()):
        summary.base_activation_count += 1
        if summary.first_base_activation_step is None:
            summary.first_base_activation_step = summary.steps
    if sigma_min < 0.1 and summary.first_singularity_crossing_step is None:
        summary.first_singularity_crossing_step = summary.steps
    arm_target = command.q_des[-7:].detach().cpu()
    if summary._previous_arm_target is not None:
        target_step = float(
            torch.max(torch.abs(arm_target - summary._previous_arm_target)).item()
        )
        summary.max_arm_target_step_rad = max(
            summary.max_arm_target_step_rad, target_step
        )
        summary.arm_snap_count += int(
            target_step > MAX_CONTINUOUS_ARM_TARGET_STEP_RAD
        )
    summary._previous_arm_target = arm_target.clone()


def _format_diagnostics(step: int, state, command, reset_cause: str, adapter) -> str:
    ee_error = float(torch.linalg.vector_norm(command.target_pose[:3] - state.ee_pose[:3]).item())
    fastest_joint = int(torch.argmax(state.controlled_qd.abs()).item())
    effort_peak = int(torch.argmax(command.effort.abs()).item())
    target_xyz = command.target_pose[:3].detach().cpu().tolist()
    actual_xyz = state.ee_pose[:3].detach().cpu().tolist()
    target_rotation = command.target_pose[3:].detach().cpu().tolist()
    actual_rotation = state.ee_pose[3:].detach().cpu().tolist()
    predicted_twist = (
        state.coordinated_jacobian @ command.motion_distribution.qd_coord
    ).detach().cpu().tolist()
    measured_twist = (
        state.coordinated_jacobian @ state.coord_qd
    ).detach().cpu().tolist()
    return (
        f"[WBC C0] step={step} ee_error={ee_error:.5f} "
        f"sigma_min={float(state.sigma_min.item()):.5f} "
        f"qp_feasible={bool(command.qp_result.success)} "
        f"qp_eq={command.qp_result.max_equality_residual:.3e} "
        f"qp_ineq={command.qp_result.max_inequality_violation:.3e} "
        f"qp_iter={command.qp_result.iterations} "
        f"joint_speed={float(state.controlled_qd.abs().max().item()):.3e}@{fastest_joint} "
        f"effort={float(command.effort.abs().max().item()):.3e}@{effort_peak} "
        f"target_xyz={[round(value, 4) for value in target_xyz]} "
        f"actual_xyz={[round(value, 4) for value in actual_xyz]} "
        f"target_rotation={[round(value, 4) for value in target_rotation]} "
        f"actual_rotation={[round(value, 4) for value in actual_rotation]} "
        f"predicted_twist={[round(value, 4) for value in predicted_twist]} "
        f"measured_twist={[round(value, 4) for value in measured_twist]} "
        f"root_z={adapter.latest_root_height:.5f} "
        f"roll={state.roll:.5f} pitch={state.pitch:.5f} "
        f"wheel_contacts={state.wheel_contact_count} "
        f"lateral_slip={state.max_lateral_slip:.5f} "
        f"base_active={bool(command.motion_distribution.base_active.item())} "
        f"safety={command.safety_state.name} safety_reason={command.safety_reason} "
        f"motion_failure={command.motion_failure_reason or 'none'} "
        f"reset_cause={reset_cause}"
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    summary = C0Summary(seed=args.seed)
    simulation_app = None
    env = None
    try:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(args).app

        import gymnasium as gym
        from isaaclab.utils import math as math_utils
        from isaaclab_tasks.utils import parse_env_cfg

        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.control.m1_panda_coordination.teacher import M1PandaWbcTeacher, TeacherCfg
        from go2_pvcnn.control.m1_panda_coordination.trajectory import (
            BandLimitedPoseTrajectory,
            BandLimitedTrajectoryCfg,
        )

        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
        env.reset(seed=args.seed)
        adapter = PhysxTeacherAdapter(env, math_utils)
        initial_state, _ = adapter.build_state(0)
        controlled_limits = initial_state.wbc_input.effort_limit
        kp, kd = build_teacher_gains()
        trajectory = (
            StaticPoseTrajectory()
            if args.disable_target_motion
            else BandLimitedPoseTrajectory(
                BandLimitedTrajectoryCfg(
                    position_amplitude=0.005,
                    orientation_amplitude=0.01,
                )
            )
        )
        teacher = M1PandaWbcTeacher(
            kp=kp,
            kd=kd,
            effort_limit=controlled_limits,
            safe_arm_target=initial_state.controlled_q[-7:],
            cfg=TeacherCfg(warmup_steps=SETTLE_STEPS),
            trajectory=trajectory,
        )
        teacher.reset(initial_state, seed=args.seed)

        physics_step = 0
        mission_step = 0
        settled = False
        previous_safety = None
        while simulation_app.is_running() and (args.steps == 0 or mission_step < args.steps):
            state, base_contact = adapter.build_state(physics_step)
            if not settled and physics_step == SETTLE_STEPS:
                # The floating composite settles onto its four wheels before
                # the mission. Re-center both the EE trajectory and safe arm
                # posture on that physically realized state.
                teacher.reset(state, seed=args.seed)
                settled = True
            command = teacher.step(state)
            effort_action = command.effort.to(device=env.device, dtype=torch.float32).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(effort_action)
            reset_cause = _termination_cause(env, terminated, truncated)
            physics_step += 1
            if settled:
                _update_summary(summary, state, command, base_contact, adapter)
                mission_step += 1
            display_step = mission_step if settled else physics_step
            safety_changed = command.safety_state != previous_safety
            if (
                display_step % args.stats_interval == 0
                or display_step == 1
                or safety_changed
                or command.motion_failure_reason is not None
                or reset_cause != "none"
            ):
                print(
                    _format_diagnostics(
                        display_step, state, command, reset_cause, adapter
                    ),
                    flush=True,
                )
            previous_safety = command.safety_state
            did_reset = bool(torch.as_tensor(terminated | truncated).any().item())
            if did_reset:
                if settled:
                    summary.reset_count += 1
                reset_state, _ = adapter.build_state(physics_step)
                teacher.reset(reset_state, seed=args.seed + summary.reset_count)
            if command.terminate:
                summary.exit_reason = "safety_terminate"
                break
        else:
            summary.exit_reason = (
                "steps_complete"
                if args.steps > 0 and mission_step >= args.steps
                else "app_closed"
            )
        if summary.exit_reason == "not_started":
            summary.exit_reason = "steps_complete"
        return 0 if summary.finite else 1
    except Exception:
        summary.finite = False
        summary.exit_reason = "exception"
        traceback.print_exc()
        return 1
    finally:
        if args.summary_json is not None:
            atomic_write_summary(args.summary_json, summary.to_dict())
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
