"""IsaacLab snapshot adapter and inline bimanual MPC execution wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import time
from collections.abc import Sequence

import torch

from go2_pvcnn.assets import M1_FOOT_BODY_NAMES
from go2_pvcnn.assets.m1_dual_panda_o6 import (
    LEFT_O6_ACTIVE_JOINT_NAMES,
    LEFT_O6_FINGERTIP_BODY_NAMES,
    LEFT_O6_PALM_BODY_NAME,
    LEFT_PANDA_ACTIVE_JOINT_NAMES,
    LEFT_PANDA_WRIST_BODY_NAME,
    M1_BASE_ACTIVE_JOINT_NAMES,
    M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES,
    M1_DUAL_PANDA_O6_BASE_BODY_NAME,
    M1_DUAL_PANDA_O6_PLATFORM_JOINT_NAME,
    O6_MIMIC_MAP,
    RIGHT_O6_ACTIVE_JOINT_NAMES,
    RIGHT_O6_FINGERTIP_BODY_NAMES,
    RIGHT_O6_PALM_BODY_NAME,
    RIGHT_PANDA_ACTIVE_JOINT_NAMES,
    RIGHT_PANDA_WRIST_BODY_NAME,
    resolve_active_joint_ids,
)
from go2_pvcnn.control.m1_bimanual_coordination import (
    BimanualRuntime,
    BimanualCommand,
    BimanualSnapshot,
    BoxState,
    FullActionTeacher,
    FullDynamicsState,
    LatentRuntime,
    SafetyInput,
    SafetyProjection,
    SideArmState,
    SideHandState,
    build_actuation_matrix,
    build_teacher_input,
    fold_o6_fingertip_jacobians,
    stack_stationary_wheel_jacobians,
)
from go2_pvcnn.control.m1_bimanual_coordination.constraints import effort_limits
from go2_pvcnn.control.m1_bimanual_coordination.frame_kinematics import (
    physx_jacobian_body_row,
    pose_in_base,
    spatial_jacobian_in_base,
    twist_in_base,
    vectors_in_base,
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
        self.left_hand_generalized_ids = tuple(index + 6 for index in self.left_hand_ids)
        self.right_hand_generalized_ids = tuple(index + 6 for index in self.right_hand_ids)
        self.left_mimic_specs = tuple(
            (
                name_to_id[f"left_{mimic_name}"] + 6,
                LEFT_O6_ACTIVE_JOINT_NAMES.index(f"left_{master_name}"),
                multiplier,
            )
            for mimic_name, (master_name, multiplier, _offset) in O6_MIMIC_MAP.items()
        )
        self.right_mimic_specs = tuple(
            (
                name_to_id[f"right_{mimic_name}"] + 6,
                RIGHT_O6_ACTIVE_JOINT_NAMES.index(f"right_{master_name}"),
                multiplier,
            )
            for mimic_name, (master_name, multiplier, _offset) in O6_MIMIC_MAP.items()
        )

        body_names = tuple(self.robot.body_names)
        self.base_body_id = _exact_body_id(body_names, M1_DUAL_PANDA_O6_BASE_BODY_NAME)
        self.left_palm_id = _exact_body_id(body_names, LEFT_O6_PALM_BODY_NAME)
        self.right_palm_id = _exact_body_id(body_names, RIGHT_O6_PALM_BODY_NAME)
        self.left_wrist_id = _exact_body_id(body_names, LEFT_PANDA_WRIST_BODY_NAME)
        self.right_wrist_id = _exact_body_id(body_names, RIGHT_PANDA_WRIST_BODY_NAME)
        self.left_tip_ids = tuple(_exact_body_id(body_names, name) for name in LEFT_O6_FINGERTIP_BODY_NAMES)
        self.right_tip_ids = tuple(_exact_body_id(body_names, name) for name in RIGHT_O6_FINGERTIP_BODY_NAMES)
        self.wheel_body_ids = tuple(
            _exact_body_id(body_names, name) for name in M1_FOOT_BODY_NAMES
        )

        left_sensor_names = tuple(self.env.scene["o6_contacts"].body_names)
        right_sensor_names = tuple(self.env.scene["right_o6_contacts"].body_names)
        self.left_tip_sensor_ids = tuple(_exact_sensor_body_id(left_sensor_names, name) for name in LEFT_O6_FINGERTIP_BODY_NAMES)
        self.right_tip_sensor_ids = tuple(_exact_sensor_body_id(right_sensor_names, name) for name in RIGHT_O6_FINGERTIP_BODY_NAMES)
        self._sequence = 0
        self._previous_wheel_contact_jacobian: torch.Tensor | None = None
        self.arm_dynamics_diagnostics: dict[str, dict[str, object]] = {}

    def dynamics(self) -> FullDynamicsState:
        """Read full PhysX dynamics and four fixed-wheel contact constraints."""

        env = self.env_index
        base_quaternion = _cpu64(self.robot.data.root_quat_w[env])
        physx = self.robot.root_physx_view
        mass_matrix = _cpu64(physx.get_generalized_mass_matrices()[env])
        gravity = physx.get_gravity_compensation_forces()[env]
        coriolis = physx.get_coriolis_and_centrifugal_compensation_forces()[env]
        bias = _cpu64(gravity + coriolis)
        all_jacobians = physx.get_jacobians()
        body_count = len(self.robot.body_names)
        wheel_jacobians = torch.stack(
            tuple(
                spatial_jacobian_in_base(
                    base_quaternion,
                    _cpu64(
                        all_jacobians[
                            env,
                            physx_jacobian_body_row(
                                body_id, body_count, all_jacobians.shape[1]
                            ),
                        ]
                    ),
                )
                for body_id in self.wheel_body_ids
            )
        )
        wheel_contact_jacobian = stack_stationary_wheel_jacobians(
            wheel_jacobians
        )
        wheel_contact_bias = torch.zeros(12, dtype=torch.float64)
        if self._previous_wheel_contact_jacobian is not None:
            base_linear_b = vectors_in_base(
                base_quaternion, _cpu64(self.robot.data.root_lin_vel_w[env])
            )
            base_angular_b = vectors_in_base(
                base_quaternion, _cpu64(self.robot.data.root_ang_vel_w[env])
            )
            generalized_velocity = torch.cat(
                (
                    base_linear_b,
                    base_angular_b,
                    _cpu64(self.robot.data.joint_vel[env]),
                )
            )
            dt = float(self.env.physics_dt)
            jacobian_rate = (
                wheel_contact_jacobian - self._previous_wheel_contact_jacobian
            ) / dt
            wheel_contact_bias = jacobian_rate @ generalized_velocity
        self._previous_wheel_contact_jacobian = wheel_contact_jacobian.clone()
        return FullDynamicsState(
            mass_matrix=mass_matrix,
            bias=bias,
            actuation_matrix=build_actuation_matrix(self.active_joint_ids),
            wheel_contact_jacobian=wheel_contact_jacobian,
            wheel_contact_bias=wheel_contact_bias,
        )

    def teacher_task_jacobian(self) -> torch.Tensor:
        """Return base plus both palm spatial Jacobians in full 59 columns."""

        env = self.env_index
        base_quaternion = _cpu64(self.robot.data.root_quat_w[env])
        all_jacobians = self.robot.root_physx_view.get_jacobians()
        body_count = len(self.robot.body_names)
        result = torch.zeros((18, 59), dtype=torch.float64)
        result[:6, :6] = torch.eye(6, dtype=torch.float64)
        for rows, body_id in (
            (slice(6, 12), self.left_palm_id),
            (slice(12, 18), self.right_palm_id),
        ):
            result[rows] = spatial_jacobian_in_base(
                base_quaternion,
                _cpu64(
                    all_jacobians[
                        env,
                        physx_jacobian_body_row(
                            body_id, body_count, all_jacobians.shape[1]
                        ),
                    ]
                ),
            )
        return result

    def _arm_state(self, side: str, joint_ids: tuple[int, ...], palm_id: int) -> SideArmState:
        data = self.robot.data
        env = self.env_index
        base_position = _cpu64(data.root_pos_w[env])
        base_quaternion = _cpu64(data.root_quat_w[env])
        palm_position = _cpu64(data.body_pos_w[env, palm_id])
        palm_pose = pose_in_base(
            base_position,
            base_quaternion,
            palm_position,
            _cpu64(data.body_quat_w[env, palm_id]),
        )
        palm_twist = twist_in_base(
            base_position,
            base_quaternion,
            _cpu64(data.root_lin_vel_w[env]),
            _cpu64(data.root_ang_vel_w[env]),
            palm_position,
            _cpu64(data.body_lin_vel_w[env, palm_id]),
            _cpu64(data.body_ang_vel_w[env, palm_id]),
        )

        jacobian = torch.zeros((6, 7), dtype=torch.float64)
        mass = torch.eye(7, dtype=torch.float64)
        bias = torch.zeros(7, dtype=torch.float64)
        diagnostics: dict[str, object] = {"physx_used": False, "fallback_reason": None}
        try:
            all_jacobians = self.robot.root_physx_view.get_jacobians()
            jacobian_body_count = all_jacobians.shape[1]
            body_count = len(self.robot.body_names)
            jacobian_body_id = physx_jacobian_body_row(
                palm_id, body_count, jacobian_body_count
            )
            generalized_ids = tuple(joint_id + 6 for joint_id in joint_ids)
            full_body_jacobian = all_jacobians[env, jacobian_body_id]
            jacobian = spatial_jacobian_in_base(
                base_quaternion,
                _cpu64(full_body_jacobian[:, list(generalized_ids)]),
            )
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
        active_generalized_ids: tuple[int, ...],
        mimic_specs: tuple[tuple[int, int, float], ...],
    ) -> SideHandState:
        data = self.robot.data
        env = self.env_index
        base_position = _cpu64(data.root_pos_w[env])
        base_quaternion = _cpu64(data.root_quat_w[env])
        positions = vectors_in_base(
            base_quaternion,
            _cpu64(data.body_pos_w[env, list(fingertip_ids)]) - base_position,
        )
        forces = vectors_in_base(
            base_quaternion,
            _cpu64(
                self.env.scene[sensor_name].data.net_forces_w[
                    env, list(sensor_ids)
                ]
            ),
        )
        all_jacobians = self.robot.root_physx_view.get_jacobians()
        body_count = len(self.robot.body_names)
        fingertip_spatial_jacobians = torch.stack(
            tuple(
                spatial_jacobian_in_base(
                    base_quaternion,
                    _cpu64(
                        all_jacobians[
                            env,
                            physx_jacobian_body_row(
                                body_id, body_count, all_jacobians.shape[1]
                            ),
                        ]
                    ),
                )
                for body_id in fingertip_ids
            )
        )
        fingertip_jacobian = fold_o6_fingertip_jacobians(
            fingertip_spatial_jacobians,
            active_generalized_ids,
            mimic_specs,
        )
        return SideHandState(
            q=_cpu64(data.joint_pos[env, list(joint_ids)]),
            qd=_cpu64(data.joint_vel[env, list(joint_ids)]),
            fingertip_forces_b=forces,
            fingertip_positions_b=positions,
            fingertip_jacobian_b=fingertip_jacobian,
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
        base_position = _cpu64(data.root_pos_w[env])
        base_quaternion = _cpu64(data.root_quat_w[env])
        box_position = _cpu64(self.box.data.root_pos_w[env])
        box_pose = pose_in_base(
            base_position,
            base_quaternion,
            box_position,
            _cpu64(self.box.data.root_quat_w[env]),
        )
        box_twist = twist_in_base(
            base_position,
            base_quaternion,
            _cpu64(data.root_lin_vel_w[env]),
            _cpu64(data.root_ang_vel_w[env]),
            box_position,
            _cpu64(self.box.data.root_lin_vel_w[env]),
            _cpu64(self.box.data.root_ang_vel_w[env]),
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
            left_hand=self._hand_state(
                self.left_hand_ids,
                self.left_tip_ids,
                self.left_tip_sensor_ids,
                "o6_contacts",
                self.left_hand_generalized_ids,
                self.left_mimic_specs,
            ),
            right_hand=self._hand_state(
                self.right_hand_ids,
                self.right_tip_ids,
                self.right_tip_sensor_ids,
                "right_o6_contacts",
                self.right_hand_generalized_ids,
                self.right_mimic_specs,
            ),
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

    def __init__(
        self,
        env,
        runtime: BimanualRuntime | None = None,
        *,
        mode: str = "teacher",
        latent_artifact: str | Path | None = None,
    ) -> None:
        if mode not in {"teacher", "latent"}:
            raise ValueError("mode must be 'teacher' or 'latent'")
        self.env = env
        self.adapter = M1DualPandaO6SnapshotAdapter(env)
        self.runtime = BimanualRuntime() if runtime is None else runtime
        self.mode = mode
        self.teacher = FullActionTeacher()
        self.safety = SafetyProjection()
        self._effort_limits = effort_limits()
        self._baseline_command: BimanualCommand | None = None
        self.latent_runtime: LatentRuntime | None = None
        if mode == "latent":
            selected_artifact = latent_artifact or os.environ.get(
                "M1_BIMANUAL_LATENT_ARTIFACT"
            )
            if selected_artifact is None:
                raise FileNotFoundError(
                    "latent model path is required through M1_BIMANUAL_LATENT_ARTIFACT"
                )
            self.latent_runtime = LatentRuntime.from_artifact(
                Path(selected_artifact),
                action_order=tuple(M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES),
                teacher=self.teacher,
                safety=self.safety,
                effort_limits=self._effort_limits,
                safety_input_provider=self._safety_input,
            )
        self.last_snapshot: BimanualSnapshot | None = None
        self.last_dynamics: FullDynamicsState | None = None
        self.last_teacher_solution = None
        self.last_command = None
        self.startup_complete = False
        self.base_reference_w: torch.Tensor | None = None
        self._step = 0

    def _active_q_qd(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.adapter.robot.data
        ids = list(self.adapter.active_joint_ids)
        return _cpu64(data.joint_pos[0, ids]), _cpu64(data.joint_vel[0, ids])

    def _safety_input(
        self,
        snapshot: BimanualSnapshot,
        dynamics: FullDynamicsState,
        candidate: torch.Tensor,
    ) -> SafetyInput:
        if self._baseline_command is None or self.base_reference_w is None:
            raise RuntimeError("baseline command and base reference must exist")
        active_q, active_qd = self._active_q_qd()
        data = self.adapter.robot.data
        ids = list(self.adapter.active_joint_ids)
        limits = _cpu64(data.soft_joint_pos_limits[0, ids])
        velocity_limits = _cpu64(data.soft_joint_vel_limits[0, ids])
        current_root = _cpu64(data.root_state_w[0])
        base_error = pose_in_base(
            self.base_reference_w[:3],
            self.base_reference_w[3:7],
            current_root[:3],
            current_root[3:7],
        )
        latest_wbc = self.runtime.latest_solutions["wbc"]
        closure = (
            1.0
            if latest_wbc is None
            else latest_wbc.diagnostics.force_closure_margin
        )
        return SafetyInput(
            candidate_effort=candidate,
            safe_effort=self._baseline_command.effort,
            dynamics=dynamics,
            active_generalized_ids=torch.tensor(
                [index + 6 for index in self.adapter.active_joint_ids],
                dtype=torch.int64,
            ),
            active_q=active_q,
            active_qd=active_qd,
            q_min=limits[:, 0],
            q_max=limits[:, 1],
            qd_max=velocity_limits,
            effort_limits=self._effort_limits,
            collision_distances=torch.ones(1, dtype=torch.float64),
            collision_jacobian=torch.zeros((1, 43), dtype=torch.float64),
            base_error=base_error,
            force_closure_margin=float(closure),
            phase=self.runtime.mission.phase,
        )

    def _write_default_physics_state(self) -> None:
        raw = self.env.unwrapped
        robot = raw.scene["robot"]
        box = raw.scene["box"]
        robot_root = robot.data.default_root_state.clone()
        robot_root[:, 7:13] = 0.0
        robot.write_root_state_to_sim(robot_root)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(),
            torch.zeros_like(robot.data.default_joint_vel),
        )
        box_root = box.data.default_root_state.clone()
        box_root[:, 7:13] = 0.0
        box.write_root_state_to_sim(box_root)

    def reset(self, *, seed: int) -> BimanualSnapshot:
        """Reset, physically synchronize once, then expose an exact zero-velocity state."""

        self.startup_complete = False
        self.env.reset(seed=int(seed))
        raw = self.env.unwrapped
        self._write_default_physics_state()
        raw.scene.reset()

        zero_action = torch.zeros(
            (raw.num_envs, raw.action_manager.total_action_dim),
            device=raw.device,
            dtype=torch.float32,
        )
        raw.action_manager.process_action(zero_action)
        raw.action_manager.apply_action()
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        raw.scene.update(dt=float(raw.physics_dt))

        self._write_default_physics_state()
        raw.scene.reset()
        raw.sim.forward()
        raw.scene.update(dt=0.0)

        self.adapter = M1DualPandaO6SnapshotAdapter(self.env)
        self.runtime.reset()
        if self.latent_runtime is not None:
            self.latent_runtime.reset()
        self.last_command = None
        self.last_teacher_solution = None
        self.last_snapshot = self.adapter.snapshot()
        self.last_dynamics = self.adapter.dynamics()
        self.base_reference_w = _cpu64(
            raw.scene["robot"].data.default_root_state[0, :7]
        )
        self.startup_complete = True
        self._step = 0
        return self.last_snapshot

    def step(self):
        if not self.startup_complete:
            raise RuntimeError("wrapper.reset(seed=...) must complete before step()")
        snapshot = self.adapter.snapshot()
        self.last_dynamics = self.adapter.dynamics()
        self._baseline_command = self.runtime.compute(snapshot)
        teacher_input = build_teacher_input(
            snapshot=snapshot,
            dynamics=self.last_dynamics,
            task_jacobian=self.adapter.teacher_task_jacobian(),
            baseline_command=self._baseline_command,
            latest_solutions=self.runtime.latest_solutions,
            effort_limits=self._effort_limits,
        )
        if self.mode == "teacher":
            if self._step % 8 == 0 or self.last_teacher_solution is None:
                self.last_teacher_solution = self.teacher.plan(teacher_input)
            teacher_solution = self.last_teacher_solution
            if teacher_solution.diagnostics.feasible:
                command = BimanualCommand(
                    timestamp_ns=snapshot.timestamp_ns,
                    effort=teacher_solution.action_trajectory[0],
                    feasible=True,
                    fallback_reasons=(),
                )
            else:
                command = BimanualCommand(
                    timestamp_ns=snapshot.timestamp_ns,
                    effort=self._baseline_command.effort,
                    feasible=False,
                    fallback_reasons=(
                        teacher_solution.diagnostics.fallback_reason
                        or "teacher_infeasible",
                    ),
                )
        else:
            if self.latent_runtime is None:
                raise RuntimeError("latent runtime was not initialized")
            command = self.latent_runtime.compute(
                snapshot, self.last_dynamics, teacher_input
            )
            self.last_teacher_solution = self.latent_runtime.last_teacher_solution
        action = command.effort.to(device=self.env.unwrapped.device, dtype=torch.float32)
        action = action.unsqueeze(0).repeat(self.env.unwrapped.num_envs, 1)
        self.last_snapshot = snapshot
        self.last_command = command
        self._step += 1
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()
