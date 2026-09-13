"""IsaacLab snapshot adapter and inline bimanual MPC execution wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from collections.abc import Sequence
import weakref

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
    BimanualPhase,
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
    latch_contact_joint_targets,
    stack_stationary_wheel_jacobians,
)
from go2_pvcnn.control.m1_bimanual_coordination.constraints import effort_limits
from go2_pvcnn.control.m1_bimanual_coordination.contact_summary import (
    ContactSummary,
    summarize_contacts,
)
from go2_pvcnn.control.m1_bimanual_coordination.frame_kinematics import (
    damped_cartesian_joint_delta,
    embed_fixed_base_jacobian,
    embed_fixed_base_mass_matrix,
    embed_fixed_base_vector,
    physx_jacobian_body_row,
    pose_in_base,
    spatial_jacobian_in_base,
    twist_in_base,
    vectors_in_base,
)
from go2_pvcnn.control.m1_bimanual_coordination.vector_control import (
    stack_lane_actions,
)


def _close_fingertip_priors(priors: Sequence[object]) -> None:
    """Best-effort, idempotent worker cleanup for wrapper-owned priors."""

    for prior in priors:
        try:
            prior.close()
        except BaseException:
            pass


def _construct_fingertip_prior(path: str | Path) -> object:
    """Keep the optional runtime package outside the default import path."""

    from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.runtime import (
        FrozenO6FingertipPrior,
    )

    return FrozenO6FingertipPrior.from_artifact(path)


def _build_fingertip_priors(path: str | Path) -> tuple[object, object]:
    """Strictly construct independent left/right workers or leave none alive."""

    created: list[object] = []
    try:
        for _side in ("left", "right"):
            created.append(_construct_fingertip_prior(path))
    except BaseException:
        _close_fingertip_priors(created)
        raise
    return created[0], created[1]


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


def _o6_contact_body_candidates(side: str) -> tuple[tuple[str, ...], ...]:
    return (
        (f"{side}_thumb_distal",),
        (f"{side}_index_proximal", f"{side}_index_distal"),
        (f"{side}_middle_proximal", f"{side}_middle_distal"),
        (f"{side}_ring_proximal", f"{side}_ring_distal"),
        (f"{side}_pinky_proximal", f"{side}_pinky_distal"),
    )


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
        self.left_contact_body_id_candidates = tuple(
            tuple(_exact_body_id(body_names, name) for name in candidates)
            for candidates in _o6_contact_body_candidates("left")
        )
        self.right_contact_body_id_candidates = tuple(
            tuple(_exact_body_id(body_names, name) for name in candidates)
            for candidates in _o6_contact_body_candidates("right")
        )
        self.wheel_body_ids = tuple(
            _exact_body_id(body_names, name) for name in M1_FOOT_BODY_NAMES
        )

        left_sensor_names = tuple(self.env.scene["o6_contacts"].body_names)
        right_sensor_names = tuple(self.env.scene["right_o6_contacts"].body_names)
        self.left_tip_sensor_ids = tuple(_exact_sensor_body_id(left_sensor_names, name) for name in LEFT_O6_FINGERTIP_BODY_NAMES)
        self.right_tip_sensor_ids = tuple(_exact_sensor_body_id(right_sensor_names, name) for name in RIGHT_O6_FINGERTIP_BODY_NAMES)
        self.left_contact_sensor_id_candidates = tuple(
            tuple(_exact_sensor_body_id(left_sensor_names, name) for name in candidates)
            for candidates in _o6_contact_body_candidates("left")
        )
        self.right_contact_sensor_id_candidates = tuple(
            tuple(_exact_sensor_body_id(right_sensor_names, name) for name in candidates)
            for candidates in _o6_contact_body_candidates("right")
        )
        self.left_filtered_sensor_names = tuple(
            tuple(f"{body_name}_box_contact" for body_name in candidates)
            for candidates in _o6_contact_body_candidates("left")
        )
        self.right_filtered_sensor_names = tuple(
            tuple(f"{body_name}_box_contact" for body_name in candidates)
            for candidates in _o6_contact_body_candidates("right")
        )
        self._sequence = 0
        self._previous_wheel_contact_jacobian: torch.Tensor | None = None
        self.arm_dynamics_diagnostics: dict[str, dict[str, object]] = {}
        self.contact_summaries: dict[str, ContactSummary] = {}

    def dynamics(self) -> FullDynamicsState:
        """Read full PhysX dynamics and four fixed-wheel contact constraints."""

        env = self.env_index
        base_quaternion = _cpu64(self.robot.data.root_quat_w[env])
        physx = self.robot.root_physx_view
        mass_matrix = _cpu64(
            embed_fixed_base_mass_matrix(
                physx.get_generalized_mass_matrices()[env]
            )
        )
        gravity = physx.get_gravity_compensation_forces()[env]
        coriolis = physx.get_coriolis_and_centrifugal_compensation_forces()[env]
        bias = _cpu64(embed_fixed_base_vector(gravity + coriolis))
        all_jacobians = embed_fixed_base_jacobian(physx.get_jacobians())
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
        all_jacobians = embed_fixed_base_jacobian(
            self.robot.root_physx_view.get_jacobians()
        )
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
            all_jacobians = embed_fixed_base_jacobian(
                self.robot.root_physx_view.get_jacobians()
            )
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
            all_mass = embed_fixed_base_mass_matrix(
                self.robot.root_physx_view.get_generalized_mass_matrices()
            )
            index = torch.tensor(generalized_ids, device=all_mass.device)
            mass = _cpu64(all_mass[env].index_select(0, index).index_select(1, index))
            gravity = embed_fixed_base_vector(
                self.robot.root_physx_view.get_gravity_compensation_forces()[env]
            )
            coriolis = embed_fixed_base_vector(
                self.robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()[env]
            )
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
                    torch.linalg.vector_norm(full_body_jacobian[:, list(generalized_ids)])
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
        side: str,
        joint_ids: tuple[int, ...],
        contact_body_id_candidates: tuple[tuple[int, ...], ...],
        contact_sensor_id_candidates: tuple[tuple[int, ...], ...],
        sensor_name: str,
        filtered_sensor_names: tuple[tuple[str, ...], ...],
        active_generalized_ids: tuple[int, ...],
        mimic_specs: tuple[tuple[int, int, float], ...],
    ) -> SideHandState:
        data = self.robot.data
        env = self.env_index
        base_position = _cpu64(data.root_pos_w[env])
        base_quaternion = _cpu64(data.root_quat_w[env])
        raw_sensor_forces_w = self.env.scene[sensor_name].data.net_forces_w[env]
        flat_sensor_ids = tuple(
            sensor_id
            for candidates in contact_sensor_id_candidates
            for sensor_id in candidates
        )
        flat_filtered_sensor_names = tuple(
            name for candidates in filtered_sensor_names for name in candidates
        )
        filtered_forces_w = []
        for filtered_sensor_name in flat_filtered_sensor_names:
            force_matrix_w = self.env.scene[filtered_sensor_name].data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError(
                    f"{filtered_sensor_name} must filter contacts against the box"
                )
            filtered_forces_w.append(force_matrix_w[env, 0, 0])
        summary = summarize_contacts(
            side=side,
            candidate_names=_o6_contact_body_candidates(side),
            filtered_forces_w=_cpu64(torch.stack(filtered_forces_w)),
            raw_forces_w=_cpu64(raw_sensor_forces_w[list(flat_sensor_ids)]),
        )
        self.contact_summaries[side] = summary
        body_names = tuple(self.robot.body_names)
        selected_body_ids = [
            _exact_body_id(body_names, name) for name in summary.selected_names
        ]
        positions = vectors_in_base(
            base_quaternion,
            _cpu64(data.body_pos_w[env, selected_body_ids]) - base_position,
        )
        forces = vectors_in_base(
            base_quaternion,
            summary.filtered_forces_w,
        )
        all_jacobians = embed_fixed_base_jacobian(
            self.robot.root_physx_view.get_jacobians()
        )
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
                for body_id in selected_body_ids
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
            contact_mask=summary.contact_mask,
            contact_consistent=summary.consistency_reason is None,
        )

    def snapshot(self) -> BimanualSnapshot:
        data = self.robot.data
        env = self.env_index
        self._sequence += 1
        timestamp_ns = max(
            1, round(self._sequence * float(self.env.physics_dt) * 1e9)
        )
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
        x, y, z = (float(value) for value in (0.20, 0.18, 0.10))
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
                "left",
                self.left_hand_ids,
                self.left_contact_body_id_candidates,
                self.left_contact_sensor_id_candidates,
                "o6_contacts",
                self.left_filtered_sensor_names,
                self.left_hand_generalized_ids,
                self.left_mimic_specs,
            ),
            right_hand=self._hand_state(
                "right",
                self.right_hand_ids,
                self.right_contact_body_id_candidates,
                self.right_contact_sensor_id_candidates,
                "right_o6_contacts",
                self.right_filtered_sensor_names,
                self.right_hand_generalized_ids,
                self.right_mimic_specs,
            ),
            box=BoxState(
                pose_b=box_pose,
                twist_b=box_twist,
                mass=torch.tensor(0.5, dtype=torch.float64),
                inertia_b=inertia,
                supported=bool(box_position[2] <= 1.205),
            ),
        )


@dataclass
class BimanualLaneController:
    env_index: int
    adapter: M1DualPandaO6SnapshotAdapter
    runtime: BimanualRuntime
    teacher: FullActionTeacher
    safety: SafetyProjection
    latent_runtime: LatentRuntime | None = None
    baseline_command: BimanualCommand | None = None
    last_snapshot: BimanualSnapshot | None = None
    last_dynamics: FullDynamicsState | None = None
    last_teacher_solution: object | None = None
    last_safety_result: object | None = None
    last_command: BimanualCommand | None = None
    base_reference_w: torch.Tensor | None = None
    preload_arm_q: torch.Tensor | None = None
    left_hand_contact_q: torch.Tensor = field(
        default_factory=lambda: torch.full((6,), torch.nan, dtype=torch.float64)
    )
    right_hand_contact_q: torch.Tensor = field(
        default_factory=lambda: torch.full((6,), torch.nan, dtype=torch.float64)
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
        fingertip_prior_artifact: str | Path | None = None,
    ) -> None:
        if mode not in {"teacher", "latent"}:
            raise ValueError("mode must be 'teacher' or 'latent'")
        self.env = env
        self.mode = mode
        self._effort_limits = effort_limits()
        self._closed = False
        self._fingertip_priors: tuple[object, ...] = ()
        self._prior_finalizer: weakref.finalize | None = None
        # Artifact loading is intentionally first: a bad artifact must not touch
        # the Isaac scene, reset it, or advance physics.  There is one isolated
        # worker per physical side; vector lanes share only that side's worker.
        if fingertip_prior_artifact is not None:
            self._fingertip_priors = _build_fingertip_priors(
                fingertip_prior_artifact
            )
            self._prior_finalizer = weakref.finalize(
                self, _close_fingertip_priors, self._fingertip_priors
            )
        try:
            raw = self.env.unwrapped
            self.lanes = [
                BimanualLaneController(
                    env_index=index,
                    adapter=M1DualPandaO6SnapshotAdapter(env, env_index=index),
                    runtime=self._runtime_for_lane(
                        runtime if index == 0 else None
                    ),
                    teacher=FullActionTeacher(fixed_base=True),
                    safety=SafetyProjection(
                        joint_position_margin_rad=0.02,
                        joint_velocity_margin_fraction=0.1,
                        externally_servoed_hand=True,
                    ),
                )
                for index in range(raw.num_envs)
            ]
        except BaseException:
            self._close_prior_workers()
            raise
        selected_artifact = latent_artifact or os.environ.get(
            "M1_BIMANUAL_LATENT_ARTIFACT"
        )
        if mode == "latent":
            if selected_artifact is None:
                raise FileNotFoundError(
                    "latent model path is required through M1_BIMANUAL_LATENT_ARTIFACT"
                )
            for lane in self.lanes:
                lane.latent_runtime = LatentRuntime.from_artifact(
                    Path(selected_artifact),
                    action_order=tuple(M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES),
                    teacher=lane.teacher,
                    safety=lane.safety,
                    effort_limits=self._effort_limits,
                    safety_input_provider=lambda snapshot, dynamics, candidate,
                    index=lane.env_index: self._safety_input(
                        self.lanes[index], snapshot, dynamics, candidate
                    ),
                )
        self.startup_complete = False
        self.last_actions: torch.Tensor | None = None
        self._step = 0
        self._sync_legacy_aliases()

    def _runtime_for_lane(
        self, supplied_runtime: BimanualRuntime | None
    ) -> BimanualRuntime:
        """Attach the two approved workers without a second artifact load."""

        if not self._fingertip_priors:
            return BimanualRuntime() if supplied_runtime is None else supplied_runtime
        from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import O6HandMpc

        left_prior, right_prior = self._fingertip_priors
        if supplied_runtime is None:
            return BimanualRuntime(
                left_hand_mpc=O6HandMpc(expert_prior=left_prior),
                right_hand_mpc=O6HandMpc(expert_prior=right_prior),
            )
        controllers: list[tuple[object, object]] = []
        for side, prior in (("left", left_prior), ("right", right_prior)):
            controller = getattr(supplied_runtime, f"{side}_hand_mpc", None)
            if not isinstance(controller, O6HandMpc):
                raise TypeError(f"{side}_hand_mpc must be O6HandMpc for fingertip prior")
            if controller.expert_prior is not None:
                raise ValueError(f"{side}_hand_mpc already has an expert prior")
            controllers.append((controller, prior))
        for controller, prior in controllers:
            controller.expert_prior = prior
        return supplied_runtime

    def _close_prior_workers(self) -> None:
        finalizer = self._prior_finalizer
        if finalizer is not None:
            finalizer()
            self._prior_finalizer = None
        else:
            _close_fingertip_priors(self._fingertip_priors)

    def _sync_legacy_aliases(self) -> None:
        """Keep the single-lane probe API mapped to lane zero."""

        lane = self.lanes[0]
        self.adapter = lane.adapter
        self.runtime = lane.runtime
        self.teacher = lane.teacher
        self.safety = lane.safety
        self.latent_runtime = lane.latent_runtime
        self._baseline_command = lane.baseline_command
        self.last_snapshot = lane.last_snapshot
        self.last_dynamics = lane.last_dynamics
        self.last_teacher_solution = lane.last_teacher_solution
        self.last_safety_result = lane.last_safety_result
        self.last_command = lane.last_command
        self.base_reference_w = lane.base_reference_w
        self._preload_arm_q = lane.preload_arm_q
        self._left_hand_contact_q = lane.left_hand_contact_q
        self._right_hand_contact_q = lane.right_hand_contact_q

    def _stabilize_preload_arms(
        self,
        lane: BimanualLaneController,
        snapshot: BimanualSnapshot,
        command: BimanualCommand,
    ) -> BimanualCommand:
        if lane.runtime.mission.phase is not BimanualPhase.PRELOAD:
            lane.preload_arm_q = None
            return command
        if lane.preload_arm_q is None:
            lane.preload_arm_q = torch.cat(
                (snapshot.left_arm.q, snapshot.right_arm.q)
            )
        object_solution = lane.runtime.latest_solutions["object"]
        if object_solution is None:
            return command
        effort = command.effort.clone()
        for output, target_slice, state, hand, palm_target in (
            (slice(17, 24), slice(0, 7), snapshot.left_arm, snapshot.left_hand, object_solution.left_palm_pose[-1]),
            (slice(24, 31), slice(7, 14), snapshot.right_arm, snapshot.right_hand, object_solution.right_palm_pose[-1]),
        ):
            target = lane.preload_arm_q[target_slice]
            if not bool(hand.contact_mask.any()):
                cartesian_error = palm_target - state.palm_pose_b
                target = state.q + damped_cartesian_joint_delta(
                    state.jacobian_b,
                    cartesian_error,
                    damping=0.05,
                    max_abs_joint_delta=0.12,
                )
                lane.preload_arm_q[target_slice] = target
            elif bool(hand.contact_mask.any()):
                target = state.q.clone()
                lane.preload_arm_q[target_slice] = target
            effort[output] = (
                state.bias + 20.0 * (target - state.q) - 8.0 * state.qd
            )
        return BimanualCommand(
            timestamp_ns=command.timestamp_ns,
            effort=effort,
            feasible=command.feasible,
            fallback_reasons=command.fallback_reasons,
        )

    def _active_q_qd(
        self, lane: BimanualLaneController
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = lane.adapter.robot.data
        ids = list(lane.adapter.active_joint_ids)
        env = lane.env_index
        return _cpu64(data.joint_pos[env, ids]), _cpu64(data.joint_vel[env, ids])

    def _safety_input(
        self,
        lane: BimanualLaneController,
        snapshot: BimanualSnapshot,
        dynamics: FullDynamicsState,
        candidate: torch.Tensor,
    ) -> SafetyInput:
        if lane.baseline_command is None or lane.base_reference_w is None:
            raise RuntimeError("baseline command and base reference must exist")
        active_q, active_qd = self._active_q_qd(lane)
        data = lane.adapter.robot.data
        ids = list(lane.adapter.active_joint_ids)
        env = lane.env_index
        limits = _cpu64(data.soft_joint_pos_limits[env, ids])
        velocity_limits = _cpu64(data.soft_joint_vel_limits[env, ids])
        current_root = _cpu64(data.root_state_w[env])
        base_error = pose_in_base(
            lane.base_reference_w[:3],
            lane.base_reference_w[3:7],
            current_root[:3],
            current_root[3:7],
        )
        latest_wbc = lane.runtime.latest_solutions["wbc"]
        closure = (
            1.0
            if latest_wbc is None
            else latest_wbc.diagnostics.force_closure_margin
        )
        return SafetyInput(
            candidate_effort=candidate,
            safe_effort=lane.baseline_command.effort,
            dynamics=dynamics,
            active_generalized_ids=torch.tensor(
                [index + 6 for index in lane.adapter.active_joint_ids],
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
            phase=lane.runtime.mission.phase,
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

        for lane in self.lanes:
            lane.adapter = M1DualPandaO6SnapshotAdapter(
                self.env, env_index=lane.env_index
            )
            lane.runtime.reset()
            if lane.latent_runtime is not None:
                lane.latent_runtime.reset()
            lane.baseline_command = None
            lane.last_command = None
            lane.last_teacher_solution = None
            lane.last_safety_result = None
            lane.last_snapshot = lane.adapter.snapshot()
            lane.last_dynamics = lane.adapter.dynamics()
            lane.base_reference_w = _cpu64(
                raw.scene["robot"].data.default_root_state[lane.env_index, :7]
            )
            lane.preload_arm_q = None
            lane.left_hand_contact_q.fill_(torch.nan)
            lane.right_hand_contact_q.fill_(torch.nan)
        self.startup_complete = True
        self._step = 0
        self.last_actions = None
        self._sync_legacy_aliases()
        return self.lanes[0].last_snapshot

    def reset_lanes(
        self, done_mask: torch.Tensor, *, seeds: Sequence[int] | None = None
    ) -> None:
        """Clear temporal state only for lanes reset by the IsaacLab environment."""

        if done_mask.dtype != torch.bool or done_mask.shape != (len(self.lanes),):
            raise ValueError("done_mask must be bool with shape (num_envs,)")
        if seeds is not None and len(seeds) != len(self.lanes):
            raise ValueError("seeds must contain one value per lane")
        raw = self.env.unwrapped
        for lane, done in zip(self.lanes, done_mask.tolist(), strict=True):
            if not done:
                continue
            lane.adapter = M1DualPandaO6SnapshotAdapter(
                self.env, env_index=lane.env_index
            )
            lane.runtime.reset()
            if lane.latent_runtime is not None:
                lane.latent_runtime.reset()
            lane.baseline_command = None
            lane.last_snapshot = None
            lane.last_dynamics = None
            lane.last_teacher_solution = None
            lane.last_safety_result = None
            lane.last_command = None
            lane.base_reference_w = _cpu64(
                raw.scene["robot"].data.root_state_w[lane.env_index, :7]
            )
            lane.preload_arm_q = None
            lane.left_hand_contact_q.fill_(torch.nan)
            lane.right_hand_contact_q.fill_(torch.nan)
        self._sync_legacy_aliases()

    def _legacy_step(self):
        if not self.startup_complete:
            raise RuntimeError("wrapper.reset(seed=...) must complete before step()")
        snapshot = self.adapter.snapshot()
        self.last_dynamics = self.adapter.dynamics()
        self._baseline_command = self._stabilize_preload_arms(
            snapshot, self.runtime.compute(snapshot)
        )
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
            if self.teacher.fixed_base:
                candidate = self._baseline_command.effort
            else:
                candidate = (
                    teacher_solution.action_trajectory[0]
                    if teacher_solution.diagnostics.feasible
                    else self._baseline_command.effort
                )
            self.last_safety_result = self.safety.project(self._safety_input(
                snapshot, self.last_dynamics, candidate
            ))
            fallback_reasons = []
            if not teacher_solution.diagnostics.feasible:
                fallback_reasons.append(
                    teacher_solution.diagnostics.fallback_reason
                    or "teacher_infeasible"
                )
            if not self.last_safety_result.feasible:
                fallback_reasons.append(
                    self.last_safety_result.fallback_reason
                    or "safety_infeasible"
                )
            projected_effort = self.last_safety_result.effort.clone()
            projected_effort[31:43] = 0.0
            command = BimanualCommand(
                timestamp_ns=snapshot.timestamp_ns,
                effort=projected_effort,
                feasible=bool(
                    teacher_solution.diagnostics.feasible
                    and self.last_safety_result.feasible
                ),
                fallback_reasons=tuple(fallback_reasons),
            )
        else:
            if self.latent_runtime is None:
                raise RuntimeError("latent runtime was not initialized")
            command = self.latent_runtime.compute(
                snapshot, self.last_dynamics, teacher_input
            )
            self.last_teacher_solution = self.latent_runtime.last_teacher_solution
            self.last_safety_result = self.latent_runtime.last_safety_result
        action = command.effort.to(device=self.env.unwrapped.device, dtype=torch.float32)
        action = action.unsqueeze(0).expand(self.env.unwrapped.num_envs, -1)
        left_hand_solution = self.runtime.latest_solutions["left_hand"]
        right_hand_solution = self.runtime.latest_solutions["right_hand"]
        left_hand_target = left_hand_solution.q_ref
        right_hand_target = right_hand_solution.q_ref
        if self.runtime.mission.phase is BimanualPhase.APPROACH:
            self._left_hand_contact_q.fill_(torch.nan)
            self._right_hand_contact_q.fill_(torch.nan)
        elif self.runtime.mission.phase is BimanualPhase.PRELOAD:
            self._left_hand_contact_q, left_hand_target = latch_contact_joint_targets(
                snapshot.left_hand.q,
                snapshot.left_hand.contact_mask,
                self._left_hand_contact_q,
                left_hand_target,
            )
            self._right_hand_contact_q, right_hand_target = latch_contact_joint_targets(
                snapshot.right_hand.q,
                snapshot.right_hand.contact_mask,
                self._right_hand_contact_q,
                right_hand_target,
            )
        self.adapter.robot.set_joint_position_target(
            left_hand_target.to(
                device=self.env.unwrapped.device, dtype=torch.float32
            ).unsqueeze(0).expand(self.env.unwrapped.num_envs, -1),
            joint_ids=list(self.adapter.left_hand_ids),
        )
        self.adapter.robot.set_joint_position_target(
            right_hand_target.to(
                device=self.env.unwrapped.device, dtype=torch.float32
            ).unsqueeze(0).expand(self.env.unwrapped.num_envs, -1),
            joint_ids=list(self.adapter.right_hand_ids),
        )
        self.last_snapshot = snapshot
        self.last_command = command
        self._step += 1
        return self.env.step(action)

    def _compute_lane_command(
        self,
        lane: BimanualLaneController,
        snapshot: BimanualSnapshot,
        dynamics: FullDynamicsState,
        teacher_input,
    ) -> BimanualCommand:
        if self.mode == "latent":
            if lane.latent_runtime is None:
                raise RuntimeError("latent runtime was not initialized")
            command = lane.latent_runtime.compute(snapshot, dynamics, teacher_input)
            lane.last_teacher_solution = lane.latent_runtime.last_teacher_solution
            lane.last_safety_result = lane.latent_runtime.last_safety_result
            return command
        if self._step % 8 == 0 or lane.last_teacher_solution is None:
            lane.last_teacher_solution = lane.teacher.plan(teacher_input)
        teacher_solution = lane.last_teacher_solution
        candidate = (
            lane.baseline_command.effort
            if lane.teacher.fixed_base
            else teacher_solution.action_trajectory[0]
            if teacher_solution.diagnostics.feasible
            else lane.baseline_command.effort
        )
        lane.last_safety_result = lane.safety.project(
            self._safety_input(lane, snapshot, dynamics, candidate)
        )
        fallback_reasons: list[str] = []
        if not teacher_solution.diagnostics.feasible:
            fallback_reasons.append(
                teacher_solution.diagnostics.fallback_reason or "teacher_infeasible"
            )
        if not lane.last_safety_result.feasible:
            fallback_reasons.append(
                lane.last_safety_result.fallback_reason or "safety_infeasible"
            )
        projected_effort = lane.last_safety_result.effort.clone()
        projected_effort[31:43] = 0.0
        return BimanualCommand(
            timestamp_ns=snapshot.timestamp_ns,
            effort=projected_effort,
            feasible=bool(
                teacher_solution.diagnostics.feasible
                and lane.last_safety_result.feasible
            ),
            fallback_reasons=tuple(fallback_reasons),
        )

    def _lane_hand_targets(
        self, lane: BimanualLaneController, snapshot: BimanualSnapshot
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_solution = lane.runtime.latest_solutions["left_hand"]
        right_solution = lane.runtime.latest_solutions["right_hand"]
        left_target = left_solution.q_ref
        right_target = right_solution.q_ref
        if lane.runtime.mission.phase is BimanualPhase.APPROACH:
            lane.left_hand_contact_q.fill_(torch.nan)
            lane.right_hand_contact_q.fill_(torch.nan)
        elif lane.runtime.mission.phase is BimanualPhase.PRELOAD:
            lane.left_hand_contact_q, left_target = latch_contact_joint_targets(
                snapshot.left_hand.q,
                snapshot.left_hand.contact_mask,
                lane.left_hand_contact_q,
                left_target,
            )
            lane.right_hand_contact_q, right_target = latch_contact_joint_targets(
                snapshot.right_hand.q,
                snapshot.right_hand.contact_mask,
                lane.right_hand_contact_q,
                right_target,
            )
        return left_target, right_target

    def step(self):
        if not self.startup_complete:
            raise RuntimeError("wrapper.reset(seed=...) must complete before step()")
        commands: list[BimanualCommand] = []
        left_targets: list[torch.Tensor] = []
        right_targets: list[torch.Tensor] = []
        for lane in self.lanes:
            snapshot = lane.adapter.snapshot()
            dynamics = lane.adapter.dynamics()
            lane.baseline_command = self._stabilize_preload_arms(
                lane, snapshot, lane.runtime.compute(snapshot)
            )
            teacher_input = build_teacher_input(
                snapshot=snapshot,
                dynamics=dynamics,
                task_jacobian=lane.adapter.teacher_task_jacobian(),
                baseline_command=lane.baseline_command,
                latest_solutions=lane.runtime.latest_solutions,
                effort_limits=self._effort_limits,
            )
            command = self._compute_lane_command(
                lane, snapshot, dynamics, teacher_input
            )
            left_target, right_target = self._lane_hand_targets(lane, snapshot)
            commands.append(command)
            left_targets.append(left_target)
            right_targets.append(right_target)
            lane.last_snapshot = snapshot
            lane.last_dynamics = dynamics
            lane.last_command = command
        raw = self.env.unwrapped
        self.last_actions = stack_lane_actions(commands, device=raw.device)
        robot = self.lanes[0].adapter.robot
        robot.set_joint_position_target(
            torch.stack(left_targets).to(device=raw.device, dtype=torch.float32),
            joint_ids=list(self.lanes[0].adapter.left_hand_ids),
        )
        robot.set_joint_position_target(
            torch.stack(right_targets).to(device=raw.device, dtype=torch.float32),
            joint_ids=list(self.lanes[0].adapter.right_hand_ids),
        )
        result = self.env.step(self.last_actions)
        self._step += 1
        self._sync_legacy_aliases()
        return result

    def close(self, *, close_env: bool = True) -> None:
        """Close owned prior workers exactly once, then optionally close Isaac."""

        if self._closed:
            return
        self._closed = True
        self._close_prior_workers()
        if close_env:
            self.env.close()

    def __enter__(self) -> "M1DualPandaO6BimanualWrapper":
        return self

    def __exit__(self, *_unused: object) -> bool:
        self.close()
        return False
