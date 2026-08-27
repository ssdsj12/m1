"""PhysX-to-Teacher tensor adapter shared by C0, C1a, and Student S1."""

from __future__ import annotations

import torch

from .rolling_contact import RollingContactCfg, build_wheel_contact_jacobian


PHYSICS_DT = 0.005
WHEEL_RADIUS_M = 0.0959
WHEEL_BODY_NAMES = (
    "FAR_FOOT_LINK",
    "FBL_FOOT_LINK",
    "RAR_FOOT_LINK",
    "RBL_FOOT_LINK",
)


def _exact_id(names, expected: str, kind: str) -> int:
    matches = [index for index, name in enumerate(names) if name == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {kind} named {expected!r}; got {matches}")
    return matches[0]


def _cpu64(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64).clone()


def build_arm_mpc_input_from_teacher_state(
    state,
    target_pose_b: torch.Tensor,
    target_twist_b: torch.Tensor,
    *,
    ee_pose_b: torch.Tensor | None = None,
    arm_jacobian_b: torch.Tensor | None = None,
):
    """Build one ArmMpcInput from an already-atomic TeacherState snapshot."""

    from go2_pvcnn.control.m1_panda_coordination.arm_mpc import ArmMpcInput

    arm_indices = torch.as_tensor(
        state.wbc_input.arm_generalized_indices, dtype=torch.long, device="cpu"
    ).clone()
    if arm_indices.shape != (7,):
        raise ValueError(
            f"arm_generalized_indices must have shape (7,); got {tuple(arm_indices.shape)}"
        )
    mass_matrix = _cpu64(state.wbc_input.mass_matrix)
    bias_force = _cpu64(state.wbc_input.bias_force)
    if mass_matrix.shape != (31, 31):
        raise ValueError(
            f"mass_matrix must have shape (31, 31); got {tuple(mass_matrix.shape)}"
        )
    if bias_force.shape != (31,):
        raise ValueError(
            f"bias_force must have shape (31,); got {tuple(bias_force.shape)}"
        )
    arm_mass_matrix = mass_matrix.index_select(0, arm_indices).index_select(
        1, arm_indices
    )
    base_arm_coupling = mass_matrix[:6].index_select(1, arm_indices)
    arm_bias = bias_force.index_select(0, arm_indices)
    arm_jacobian = _cpu64(
        state.coordinated_jacobian[:, -7:]
        if arm_jacobian_b is None
        else arm_jacobian_b
    )
    arm_q = _cpu64(state.coord_q)[-7:]
    arm_qd = _cpu64(state.coord_qd)[-7:]
    return ArmMpcInput(
        q=arm_q,
        qd=arm_qd,
        ee_pose_b=_cpu64(state.ee_pose if ee_pose_b is None else ee_pose_b),
        ee_twist_b=arm_jacobian @ arm_qd,
        target_pose_b=_cpu64(target_pose_b),
        target_twist_b=_cpu64(target_twist_b),
        jacobian_b=arm_jacobian,
        arm_mass_matrix=arm_mass_matrix,
        arm_bias=arm_bias,
        base_arm_coupling=base_arm_coupling,
        q_min=_cpu64(state.coord_q_min)[-7:],
        q_max=_cpu64(state.coord_q_max)[-7:],
        qd_max=_cpu64(state.coord_v_max)[-7:],
        qdd_max=_cpu64(state.coord_a_max)[-7:],
        effort_max=_cpu64(state.wbc_input.effort_limit)[-7:],
    )


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


def read_generalized_bias_force(
    root_view, *, generalized_dof: int, env_index: int = 0
) -> torch.Tensor:
    """Read C(q, qd)+g(q), upgrading legacy joint-only floating-base APIs."""

    coriolis_force = _cpu64(
        root_view.get_coriolis_and_centrifugal_forces()[env_index]
    )
    gravity_force = _cpu64(root_view.get_generalized_gravity_forces()[env_index])
    if coriolis_force.shape != (generalized_dof,) or gravity_force.shape != (
        generalized_dof,
    ):
        coriolis_force = _cpu64(
            root_view.get_coriolis_and_centrifugal_compensation_forces()[env_index]
        )
        gravity_force = _cpu64(
            root_view.get_gravity_compensation_forces()[env_index]
        )
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


def arm_kinematics_in_base_frame(
    math_utils,
    *,
    root_position_w: torch.Tensor,
    root_quaternion_w: torch.Tensor,
    hand_position_w: torch.Tensor,
    hand_quaternion_w: torch.Tensor,
    reference_root_quaternion_w: torch.Tensor,
    reference_hand_quaternion_w: torch.Tensor,
    arm_jacobian_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express Panda pose and arm-only spatial Jacobian at the base origin."""

    root_position = _cpu64(root_position_w)
    root_quaternion = _cpu64(root_quaternion_w)
    hand_position = _cpu64(hand_position_w)
    hand_quaternion = _cpu64(hand_quaternion_w)
    reference_root_quaternion = _cpu64(reference_root_quaternion_w)
    reference_hand_quaternion = _cpu64(reference_hand_quaternion_w)
    jacobian_w = _cpu64(arm_jacobian_w)
    if jacobian_w.shape != (6, 7):
        raise ValueError(
            f"arm_jacobian_w must have shape (6, 7); got {tuple(jacobian_w.shape)}"
        )

    position_b = math_utils.quat_apply_inverse(
        root_quaternion.unsqueeze(0),
        (hand_position - root_position).unsqueeze(0),
    )[0]
    hand_quaternion_b = math_utils.quat_mul(
        math_utils.quat_inv(root_quaternion.unsqueeze(0)),
        hand_quaternion.unsqueeze(0),
    )
    reference_hand_quaternion_b = math_utils.quat_mul(
        math_utils.quat_inv(reference_root_quaternion.unsqueeze(0)),
        reference_hand_quaternion.unsqueeze(0),
    )
    orientation_b = relative_axis_angle(
        math_utils, hand_quaternion_b, reference_hand_quaternion_b
    )[0]
    root_quaternions = root_quaternion.unsqueeze(0).expand(7, -1)
    linear_b = math_utils.quat_apply_inverse(
        root_quaternions, jacobian_w[:3].transpose(0, 1)
    ).transpose(0, 1)
    angular_b = math_utils.quat_apply_inverse(
        root_quaternions, jacobian_w[3:].transpose(0, 1)
    ).transpose(0, 1)
    pose_b = _cpu64(torch.cat((position_b, orientation_b)))
    jacobian_b = _cpu64(torch.cat((linear_b, angular_b), dim=0))
    if not torch.isfinite(pose_b).all().item() or not torch.isfinite(
        jacobian_b
    ).all().item():
        raise ValueError("base-frame arm kinematics must contain only finite values")
    return pose_b, jacobian_b


class PhysxTeacherAdapter:
    """Translate one live Isaac articulation into explicit C0 Teacher tensors."""

    def __init__(
        self,
        env,
        math_utils=None,
        wheel_radius_m: float = WHEEL_RADIUS_M,
        *,
        env_index: int = 0,
    ):
        from go2_pvcnn.control.m1_panda_coordination.contracts import WbcJointMap

        self.env = env
        self.robot = env.scene["robot"]
        self.contact_sensor = env.scene["contact_forces"]
        if math_utils is None:
            math_utils = getattr(env, "math_utils", None)
        if math_utils is None:
            from isaaclab.utils import math as math_utils
        self.math_utils = math_utils
        self.wheel_radius_m = float(wheel_radius_m)
        self.rolling_contact_cfg = RollingContactCfg(
            wheel_radius_m=self.wheel_radius_m
        )
        robot = self.robot
        if (
            not isinstance(env_index, int)
            or isinstance(env_index, bool)
            or env_index < 0
            or env_index >= robot.num_instances
        ):
            raise IndexError(
                f"env_index must be in [0,{robot.num_instances}), got {env_index!r}"
            )
        self.env_index = env_index
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
        self.rebase_reference()
        self.latest_root_height = float(
            robot.data.root_pos_w[self.env_index, 2].item()
        )
        self.latest_wheel_heights = [
            float(value)
            for value in robot.data.body_pos_w[self.env_index]
            .index_select(0, self.wheel_body_ids)[:, 2]
            .detach()
            .cpu()
            .tolist()
        ]

    def rebase_reference(self) -> None:
        """Capture reset-relative state used by pose, balance, and null-space tasks."""

        data = self.robot.data
        self._initial_root_pos = _cpu64(data.root_pos_w[self.env_index])
        self._initial_root_quat = data.root_quat_w[self.env_index].detach().clone()
        self._initial_hand_quat = data.body_quat_w[
            self.env_index, self.hand_body_id
        ].detach().clone()
        self._initial_arm_q = _cpu64(
            data.joint_pos[self.env_index].index_select(
                0, self.joint_map.panda_arm.to(self.robot.device)
            )
        )
        initial_euler = self.math_utils.euler_xyz_from_quat(
            self._initial_root_quat.unsqueeze(0)
        )
        self._initial_rpy = _cpu64(torch.stack(initial_euler, dim=-1)[0])
        self._previous_contact_jacobian = None

    @staticmethod
    def _body_jacobian(
        jacobians: torch.Tensor,
        body_id: int,
        body_count: int,
        env_index: int = 0,
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
        return jacobians[env_index, row]

    def _generalized_velocity(self) -> torch.Tensor:
        data = self.robot.data
        return _cpu64(
            torch.cat(
                (
                    data.root_lin_vel_w[self.env_index],
                    data.root_ang_vel_w[self.env_index],
                    data.joint_vel[self.env_index],
                )
            )
        )

    def read_mount_wrench_b(self) -> torch.Tensor:
        """Read one canonical base-frame mount wrench as a CPU float64 clone."""

        from go2_pvcnn.mdp.m1_panda_wrench import shift_rotate_wrench_to_base

        data = self.robot.data
        incoming = self.robot.root_physx_view.get_link_incoming_joint_force()[
            self.env_index, self.mount_body_id
        ]
        mount_quat_w = data.body_quat_w[
            self.env_index, self.mount_body_id
        ]
        force_w = self.math_utils.quat_apply(
            mount_quat_w.unsqueeze(0), incoming[:3].unsqueeze(0)
        )
        torque_w = self.math_utils.quat_apply(
            mount_quat_w.unsqueeze(0), incoming[3:].unsqueeze(0)
        )
        wrench = shift_rotate_wrench_to_base(
            force_w,
            torque_w,
            data.body_pos_w[
                self.env_index, self.mount_body_id
            ].unsqueeze(0),
            data.body_pos_w[
                self.env_index, self.base_body_id
            ].unsqueeze(0),
            data.body_quat_w[
                self.env_index, self.base_body_id
            ].unsqueeze(0),
        )
        return _cpu64(wrench[0])

    def leg_soft_limits(self) -> torch.Tensor:
        """Return canonical 12-leg soft limits for residual stance clipping."""

        return _cpu64(
            self.robot.data.soft_joint_pos_limits[self.env_index].index_select(
                0, self.joint_map.legs.to(self.robot.device)
            )
        )

    def build_arm_mpc_input(
        self,
        state,
        target_pose_b: torch.Tensor,
        target_twist_b: torch.Tensor,
    ):
        """Convert the same TeacherState used by WBC into an ArmMpcInput."""

        ee_pose_b, arm_jacobian_b = self.arm_mpc_kinematics_b(state)
        return build_arm_mpc_input_from_teacher_state(
            state,
            target_pose_b,
            target_twist_b,
            ee_pose_b=ee_pose_b,
            arm_jacobian_b=arm_jacobian_b,
        )

    def arm_mpc_kinematics_b(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the MPC-only arm kinematics in the canonical base frame."""

        data = self.robot.data
        return arm_kinematics_in_base_frame(
            self.math_utils,
            root_position_w=data.root_pos_w[self.env_index],
            root_quaternion_w=data.root_quat_w[self.env_index],
            hand_position_w=data.body_pos_w[self.env_index, self.hand_body_id],
            hand_quaternion_w=data.body_quat_w[self.env_index, self.hand_body_id],
            reference_root_quaternion_w=self._initial_root_quat,
            reference_hand_quaternion_w=self._initial_hand_quat,
            arm_jacobian_w=state.coordinated_jacobian[:, -7:],
        )

    def _contact_metrics(self) -> tuple[int, float, int]:
        forces = self.contact_sensor.data.net_forces_w[self.env_index]
        wheel_forces = forces.index_select(0, self.wheel_sensor_ids)
        self.latest_measured_wheel_forces = _cpu64(wheel_forces)
        wheel_contact = torch.linalg.vector_norm(wheel_forces, dim=-1) > 1.0
        wheel_vel_w = self.robot.data.body_lin_vel_w[self.env_index].index_select(
            0, self.wheel_body_ids
        )
        root_quat = self.robot.data.root_quat_w[self.env_index].expand(4, -1)
        wheel_vel_b = self.math_utils.quat_apply_inverse(root_quat, wheel_vel_w)
        lateral = torch.where(wheel_contact, wheel_vel_b[:, 1].abs(), torch.zeros_like(wheel_vel_b[:, 1]))
        base_contact = int(torch.linalg.vector_norm(forces[self.base_sensor_id]).item() > 10.0)
        return int(wheel_contact.sum().item()), float(lateral.max().item()), base_contact

    def _pose_and_orientation(self) -> tuple[torch.Tensor, float, float, float]:
        data = self.robot.data
        hand_position = _cpu64(data.body_pos_w[self.env_index, self.hand_body_id])
        hand_quat = data.body_quat_w[
            self.env_index, self.hand_body_id
        ].unsqueeze(0)
        hand_axis_angle = _cpu64(
            relative_axis_angle(
                self.math_utils,
                hand_quat,
                self._initial_hand_quat.unsqueeze(0),
            )[0]
        )
        root_euler = self.math_utils.euler_xyz_from_quat(
            data.root_quat_w[self.env_index].unsqueeze(0)
        )
        roll, pitch, yaw = (float(component[0].item()) for component in root_euler)
        return torch.cat((hand_position, hand_axis_angle)), roll, pitch, yaw

    def build_state(self, physics_step: int):
        from go2_pvcnn.control.m1_panda_coordination.kinematics import singularity_metrics
        from go2_pvcnn.control.m1_panda_coordination.standing_wbc import StandingWbcInput
        from go2_pvcnn.control.m1_panda_coordination.teacher import TeacherState

        robot = self.robot
        data = robot.data
        self.latest_root_height = float(data.root_pos_w[self.env_index, 2].item())
        self.latest_wheel_heights = [
            float(value)
            for value in data.body_pos_w[self.env_index]
            .index_select(0, self.wheel_body_ids)[:, 2]
            .detach()
            .cpu()
            .tolist()
        ]
        root_view = robot.root_physx_view
        mass_matrix = _cpu64(
            root_view.get_generalized_mass_matrices()[self.env_index]
        )
        bias_force = read_generalized_bias_force(
            root_view,
            generalized_dof=mass_matrix.shape[0],
            env_index=self.env_index,
        )
        jacobians_device = root_view.get_jacobians()
        jacobians = _cpu64(jacobians_device)
        generalized_velocity = self._generalized_velocity()

        wheel_body_jacobians = torch.stack(
            [
                self._body_jacobian(
                    jacobians,
                    int(index),
                    len(robot.body_names),
                    self.env_index,
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
            jacobians, self.hand_body_id, len(robot.body_names), self.env_index
        )
        self.latest_hand_base_jacobian = hand_jacobian[:, :6].clone()
        mount_jacobian = self._body_jacobian(
            jacobians, self.mount_body_id, len(robot.body_names), self.env_index
        )
        panda_jacobian = hand_jacobian.index_select(1, self.arm_generalized_indices)
        coordinated_columns = torch.cat(
            (torch.tensor([0, 1, 5], dtype=torch.long), self.arm_generalized_indices)
        )
        coordinated_jacobian = hand_jacobian.index_select(1, coordinated_columns)
        sigma_min, _ = singularity_metrics(panda_jacobian)
        self.latest_arm_jacobian_twist = panda_jacobian @ _cpu64(
            data.joint_vel[self.env_index].index_select(
                0, self.joint_map.panda_arm.to(robot.device)
            )
        )
        hand_offset_w = _cpu64(
            data.body_pos_w[self.env_index, self.hand_body_id]
            - data.root_pos_w[self.env_index]
        )
        base_linear_at_hand = _cpu64(
            data.root_lin_vel_w[self.env_index]
        ) + torch.linalg.cross(
            _cpu64(data.root_ang_vel_w[self.env_index]), hand_offset_w
        )
        self.latest_measured_arm_twist = torch.cat(
            (
                _cpu64(data.body_lin_vel_w[self.env_index, self.hand_body_id])
                - base_linear_at_hand,
                _cpu64(data.body_ang_vel_w[self.env_index, self.hand_body_id])
                - _cpu64(data.root_ang_vel_w[self.env_index]),
            )
        )

        ee_pose, roll, pitch, yaw = self._pose_and_orientation()
        root_position = _cpu64(data.root_pos_w[self.env_index])
        joint_position = _cpu64(data.joint_pos[self.env_index])
        joint_velocity = _cpu64(data.joint_vel[self.env_index])
        controlled_q = joint_position.index_select(0, self.joint_map.controlled)
        controlled_qd = joint_velocity.index_select(0, self.joint_map.controlled)
        arm_q = joint_position.index_select(0, self.joint_map.panda_arm)
        arm_qd = joint_velocity.index_select(0, self.joint_map.panda_arm)
        root_linear_velocity = _cpu64(data.root_lin_vel_w[self.env_index])
        root_angular_velocity = _cpu64(data.root_ang_vel_w[self.env_index])
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

        joint_limits = _cpu64(data.soft_joint_pos_limits[self.env_index])
        arm_limits = joint_limits.index_select(0, self.joint_map.panda_arm)
        coord_q_min = torch.cat((self._initial_root_pos[:2] - 0.25, self._initial_rpy[2:3] - 0.5, arm_limits[:, 0]))
        coord_q_max = torch.cat((self._initial_root_pos[:2] + 0.25, self._initial_rpy[2:3] + 0.5, arm_limits[:, 1]))
        velocity_limits = _cpu64(
            data.joint_vel_limits[self.env_index]
        ).index_select(0, self.joint_map.panda_arm)
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
        default_joint_position = _cpu64(data.default_joint_pos[self.env_index])
        leg_q = joint_position.index_select(0, self.joint_map.legs)
        leg_qd = joint_velocity.index_select(0, self.joint_map.legs)
        leg_default = default_joint_position.index_select(0, self.joint_map.legs)
        leg_acceleration = 80.0 * (leg_default - leg_q) - 12.0 * leg_qd
        wheel_acceleration = -8.0 * joint_velocity.index_select(0, self.joint_map.wheels)

        qdd_lower = torch.full((31,), -100.0, dtype=torch.float64)
        qdd_upper = torch.full((31,), 100.0, dtype=torch.float64)
        effort_limit = _cpu64(
            data.joint_effort_limits[self.env_index]
        ).index_select(0, self.joint_map.controlled)
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
        positions = self.robot.data.joint_pos[self.env_index].index_select(
            0, self.controlled_joint_ids_device
        )
        limits = self.robot.data.soft_joint_pos_limits[self.env_index].index_select(
            0, self.controlled_joint_ids_device
        )
        return int(((positions < limits[:, 0]) | (positions > limits[:, 1])).sum().item())



__all__ = [
    "PhysxTeacherAdapter",
    "arm_kinematics_in_base_frame",
    "build_arm_mpc_input_from_teacher_state",
    "build_teacher_gains",
    "contact_point_linear_jacobian",
    "read_generalized_bias_force",
    "relative_axis_angle",
]
