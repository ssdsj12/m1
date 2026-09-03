"""IsaacLab snapshot adapter and inline bimanual MPC execution wrapper."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch

from go2_pvcnn.assets.m1_dual_panda_o6 import (
    LEFT_O6_ACTIVE_JOINT_NAMES,
    LEFT_O6_FINGERTIP_BODY_NAMES,
    LEFT_O6_PALM_BODY_NAME,
    LEFT_PANDA_ACTIVE_JOINT_NAMES,
    LEFT_PANDA_WRIST_BODY_NAME,
    M1_BASE_ACTIVE_JOINT_NAMES,
    M1_DUAL_PANDA_O6_BASE_BODY_NAME,
    M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME,
    RIGHT_O6_ACTIVE_JOINT_NAMES,
    RIGHT_O6_FINGERTIP_BODY_NAMES,
    RIGHT_O6_PALM_BODY_NAME,
    RIGHT_PANDA_ACTIVE_JOINT_NAMES,
    RIGHT_PANDA_WRIST_BODY_NAME,
    resolve_active_joint_ids,
)
from go2_pvcnn.control.m1_bimanual_coordination import (
    BimanualRuntime,
    BimanualSnapshot,
    BoxState,
    SideArmState,
    SideHandState,
)


def _exact_body_id(names: Sequence[str], expected: str) -> int:
    matches = [index for index, name in enumerate(names) if name == expected]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one body named {expected!r}; found {len(matches)}"
        )
    return matches[0]


def _exact_sensor_body_id(names: Sequence[str], expected: str) -> int:
    """Contact sensors may expose a relative prim path instead of its leaf name."""

    matches = [
        index
        for index, name in enumerate(names)
        if name == expected or name.endswith(f"/{expected}")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one sensor body ending in {expected!r}; "
            f"found {len(matches)} in {tuple(names)!r}"
        )
    return matches[0]


def _cpu64(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64).clone()


def _quat_to_rotvec(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    quat = _cpu64(quaternion_wxyz)
    quat = quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-12)
    if quat[0] < 0.0:
        quat = -quat
    vector_norm = torch.linalg.vector_norm(quat[1:])
    if vector_norm <= 1.0e-10:
        return 2.0 * quat[1:]
    angle = 2.0 * torch.atan2(vector_norm, quat[0].clamp_min(1.0e-12))
    return angle * quat[1:] / vector_norm


class M1DualPandaO6SnapshotAdapter:
    """Resolve the articulation layout once, then build atomic CPU snapshots."""

    def __init__(self, env, *, env_index: int = 0) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.env_index = int(env_index)
        self.robot = self.env.scene["robot"]
        self.box = self.env.scene["box"]
        names = tuple(self.robot.joint_names)
        self.active_joint_ids = resolve_active_joint_ids(names)
        name_to_id = {name: index for index, name in enumerate(names)}
        self.m1_ids = tuple(name_to_id[name] for name in M1_BASE_ACTIVE_JOINT_NAMES)
        self.platform_id = name_to_id[M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME]
        self.left_arm_ids = tuple(name_to_id[name] for name in LEFT_PANDA_ACTIVE_JOINT_NAMES)
        self.right_arm_ids = tuple(name_to_id[name] for name in RIGHT_PANDA_ACTIVE_JOINT_NAMES)
        self.left_hand_ids = tuple(name_to_id[name] for name in LEFT_O6_ACTIVE_JOINT_NAMES)
        self.right_hand_ids = tuple(name_to_id[name] for name in RIGHT_O6_ACTIVE_JOINT_NAMES)

        body_names = tuple(self.robot.body_names)
        self.base_body_id = _exact_body_id(body_names, M1_DUAL_PANDA_O6_BASE_BODY_NAME)
        self.left_palm_id = _exact_body_id(body_names, LEFT_O6_PALM_BODY_NAME)
        self.right_palm_id = _exact_body_id(body_names, RIGHT_O6_PALM_BODY_NAME)
        self.left_wrist_id = _exact_body_id(body_names, LEFT_PANDA_WRIST_BODY_NAME)
        self.right_wrist_id = _exact_body_id(body_names, RIGHT_PANDA_WRIST_BODY_NAME)
        self.left_tip_ids = tuple(_exact_body_id(body_names, name) for name in LEFT_O6_FINGERTIP_BODY_NAMES)
        self.right_tip_ids = tuple(_exact_body_id(body_names, name) for name in RIGHT_O6_FINGERTIP_BODY_NAMES)

        left_sensor_names = tuple(self.env.scene["o6_contacts"].body_names)
        right_sensor_names = tuple(self.env.scene["right_o6_contacts"].body_names)
        self.left_tip_sensor_ids = tuple(_exact_sensor_body_id(left_sensor_names, name) for name in LEFT_O6_FINGERTIP_BODY_NAMES)
        self.right_tip_sensor_ids = tuple(_exact_sensor_body_id(right_sensor_names, name) for name in RIGHT_O6_FINGERTIP_BODY_NAMES)
        self._sequence = 0
        self.arm_dynamics_diagnostics: dict[str, dict[str, object]] = {}

    def _arm_state(self, side: str, joint_ids: tuple[int, ...], palm_id: int) -> SideArmState:
        data = self.robot.data
        env = self.env_index
        base_position = data.root_pos_w[env]
        palm_position = data.body_pos_w[env, palm_id]
        palm_pose = torch.cat((_cpu64(palm_position - base_position), _quat_to_rotvec(data.body_quat_w[env, palm_id])))
        palm_twist = torch.cat((_cpu64(data.body_lin_vel_w[env, palm_id]), _cpu64(data.body_ang_vel_w[env, palm_id])))

        jacobian = torch.zeros((6, 7), dtype=torch.float64)
        mass = torch.eye(7, dtype=torch.float64)
        bias = torch.zeros(7, dtype=torch.float64)
        diagnostics: dict[str, object] = {"physx_used": False, "fallback_reason": None}
        try:
            all_jacobians = self.robot.root_physx_view.get_jacobians()
            jacobian_body_count = all_jacobians.shape[1]
            body_count = len(self.robot.body_names)
            if jacobian_body_count == body_count:
                jacobian_body_id = palm_id
            elif jacobian_body_count == body_count - 1:
                if palm_id <= 0:
                    raise RuntimeError("floating root body has no legacy Jacobian row")
                jacobian_body_id = palm_id - 1
            else:
                raise RuntimeError(
                    "PhysX Jacobian body count does not match articulation bodies"
                )
            generalized_ids = tuple(joint_id + 6 for joint_id in joint_ids)
            full_body_jacobian = all_jacobians[env, jacobian_body_id]
            jacobian = _cpu64(full_body_jacobian[:, list(generalized_ids)])
            all_mass = self.robot.root_physx_view.get_generalized_mass_matrices()
            index = torch.tensor(generalized_ids, device=all_mass.device)
            mass = _cpu64(all_mass[env].index_select(0, index).index_select(1, index))
            gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[env]
            coriolis = self.robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()[env]
            bias = _cpu64((gravity + coriolis)[list(generalized_ids)])
            diagnostics.update(
                physx_used=True,
                jacobian_shape=list(all_jacobians.shape),
                mass_matrix_shape=list(all_mass.shape),
                body_count=len(self.robot.body_names),
                palm_body_id=palm_id,
                jacobian_body_id=jacobian_body_id,
                full_body_jacobian_norm=float(torch.linalg.vector_norm(full_body_jacobian)),
                joint_id_columns_norm=float(
                    torch.linalg.vector_norm(full_body_jacobian[:, list(joint_ids)])
                ),
                generalized_id_columns=list(generalized_ids),
                jacobian_norm=float(torch.linalg.vector_norm(jacobian)),
                mass_condition=float(torch.linalg.cond(mass)),
            )
        except (AttributeError, IndexError, RuntimeError) as error:
            diagnostics["fallback_reason"] = f"{type(error).__name__}: {error}"
        self.arm_dynamics_diagnostics[side] = diagnostics
        return SideArmState(
            q=_cpu64(data.joint_pos[env, list(joint_ids)]),
            qd=_cpu64(data.joint_vel[env, list(joint_ids)]),
            palm_pose_b=palm_pose,
            palm_twist_b=palm_twist,
            jacobian_b=jacobian,
            mass_matrix=mass,
            bias=bias,
        )

    def _hand_state(
        self,
        joint_ids: tuple[int, ...],
        fingertip_ids: tuple[int, ...],
        sensor_ids: tuple[int, ...],
        sensor_name: str,
    ) -> SideHandState:
        data = self.robot.data
        env = self.env_index
        base_position = data.root_pos_w[env]
        positions = _cpu64(data.body_pos_w[env, list(fingertip_ids)] - base_position)
        forces = _cpu64(self.env.scene[sensor_name].data.net_forces_w[env, list(sensor_ids)])
        return SideHandState(
            q=_cpu64(data.joint_pos[env, list(joint_ids)]),
            qd=_cpu64(data.joint_vel[env, list(joint_ids)]),
            fingertip_forces_b=forces,
            fingertip_positions_b=positions,
            contact_mask=torch.linalg.vector_norm(forces, dim=1) > 0.2,
        )

    def snapshot(self) -> BimanualSnapshot:
        data = self.robot.data
        env = self.env_index
        self._sequence += 1
        timestamp_ns = max(time.monotonic_ns(), self._sequence)
        base_state = _cpu64(data.root_state_w[env])
        joint_pos = data.joint_pos[env]
        joint_vel = data.joint_vel[env]
        box_position = self.box.data.root_pos_w[env]
        box_pose = torch.cat(
            (
                _cpu64(box_position - data.root_pos_w[env]),
                _quat_to_rotvec(self.box.data.root_quat_w[env]),
            )
        )
        box_twist = torch.cat(
            (
                _cpu64(self.box.data.root_lin_vel_w[env]),
                _cpu64(self.box.data.root_ang_vel_w[env]),
            )
        )
        x, y, z = (float(value) for value in (0.12, 0.18, 0.10))
        inertia = (0.5 / 12.0) * torch.diag(
            torch.tensor((y * y + z * z, x * x + z * z, x * x + y * y), dtype=torch.float64)
        )
        return BimanualSnapshot(
            timestamp_ns=timestamp_ns,
            base_state=base_state,
            m1_q=_cpu64(joint_pos[list(self.m1_ids)]),
            m1_qd=_cpu64(joint_vel[list(self.m1_ids)]),
            platform_q_qd=_cpu64(torch.stack((joint_pos[self.platform_id], joint_vel[self.platform_id]))),
            left_arm=self._arm_state("left", self.left_arm_ids, self.left_palm_id),
            right_arm=self._arm_state("right", self.right_arm_ids, self.right_palm_id),
            left_hand=self._hand_state(self.left_hand_ids, self.left_tip_ids, self.left_tip_sensor_ids, "o6_contacts"),
            right_hand=self._hand_state(self.right_hand_ids, self.right_tip_ids, self.right_tip_sensor_ids, "right_o6_contacts"),
            box=BoxState(
                pose_b=box_pose,
                twist_b=box_twist,
                mass=torch.tensor(0.5, dtype=torch.float64),
                inertia_b=inertia,
                supported=bool(box_position[2] <= 1.005),
            ),
        )


class M1DualPandaO6BimanualWrapper:
    """Compute one MPC command and apply it atomically on every physics step."""

    def __init__(self, env, runtime: BimanualRuntime | None = None) -> None:
        self.env = env
        self.adapter = M1DualPandaO6SnapshotAdapter(env)
        self.runtime = BimanualRuntime() if runtime is None else runtime
        self.last_snapshot: BimanualSnapshot | None = None
        self.last_command = None

    def step(self):
        snapshot = self.adapter.snapshot()
        command = self.runtime.compute(snapshot)
        action = command.effort.to(device=self.env.unwrapped.device, dtype=torch.float32)
        action = action.unsqueeze(0).repeat(self.env.unwrapped.num_envs, 1)
        self.last_snapshot = snapshot
        self.last_command = command
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()
