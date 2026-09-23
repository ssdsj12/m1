"""Atomic 25/50/100/200 Hz orchestration for the bimanual control stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from go2_pvcnn.control.m1_panda_coordination.arm_mpc import ArmMpcInput

from .constraints import BimanualWbcRequest
from .grasp_goal import BimanualGraspGoal
from .contracts import (
    BimanualCommand,
    BimanualPhase,
    BimanualSnapshot,
    validate_monotonic_snapshot,
)
from .dual_arm_mpc import DualArmMpcCoordinator, DualArmMpcInput, DualArmMpcSolution
from .hand_mpc import HandMpcInput, HandMpcSolution, O6HandMpc
from .motion_primitives import BimanualMotionPrimitive, ManipulationTarget
from .object_mpc import (
    BimanualObjectMpc,
    OBJECT_MPC_HORIZON_STEPS,
    ObjectMpcInput,
    ObjectMpcSolution,
)
from .state_machine import (
    BimanualMission,
    BimanualMissionDiagnostics,
)
from .whole_body_qp import BimanualWbcSolution, BimanualWholeBodyQp


ObjectInputProvider = Callable[
    [BimanualSnapshot, BimanualPhase, ObjectMpcSolution | None], ObjectMpcInput
]
GraspGoalProvider = Callable[[BimanualSnapshot], BimanualGraspGoal | None]
ArmInputProvider = Callable[[BimanualSnapshot], tuple[ArmMpcInput, ArmMpcInput]]
HandInputProvider = Callable[[BimanualSnapshot, str, torch.Tensor], HandMpcInput]
CollisionProvider = Callable[
    [BimanualSnapshot], tuple[torch.Tensor, torch.Tensor]
]


def _default_arm_input(snapshot: BimanualSnapshot, side: str) -> ArmMpcInput:
    state = snapshot.left_arm if side == "left" else snapshot.right_arm
    q_min = torch.tensor(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        dtype=torch.float64,
    )
    q_max = torch.tensor(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        dtype=torch.float64,
    )
    horizon = 20
    return ArmMpcInput(
        q=state.q,
        qd=state.qd,
        ee_pose_b=state.palm_pose_b,
        ee_twist_b=state.palm_twist_b,
        target_pose_b=state.palm_pose_b.repeat(horizon, 1),
        target_twist_b=torch.zeros((horizon, 6), dtype=torch.float64),
        jacobian_b=state.jacobian_b,
        arm_mass_matrix=state.mass_matrix,
        arm_bias=state.bias,
        base_arm_coupling=torch.zeros((6, 7), dtype=torch.float64),
        q_min=q_min,
        q_max=q_max,
        qd_max=torch.tensor([2.175] * 4 + [2.61] * 3, dtype=torch.float64),
        qdd_max=8.0 * torch.ones(7, dtype=torch.float64),
        effort_max=torch.tensor([87.0] * 4 + [12.0] * 3, dtype=torch.float64),
    )


def _wrench_map(positions: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((6, 15), dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    for fingertip, position in enumerate(positions):
        columns = slice(3 * fingertip, 3 * (fingertip + 1))
        result[:3, columns] = identity
        x, y, z = position.tolist()
        result[3:, columns] = torch.tensor(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=torch.float64,
        )
    return result


def _default_hand_input(
    snapshot: BimanualSnapshot, side: str, target_wrench: torch.Tensor
) -> HandMpcInput:
    state = snapshot.left_hand if side == "left" else snapshot.right_hand
    arm_state = snapshot.left_arm if side == "left" else snapshot.right_arm
    return HandMpcInput(
        q=state.q,
        qd=state.qd,
        fingertip_forces_b=state.fingertip_forces_b,
        fingertip_positions_b=state.fingertip_positions_b,
        contact_mask=state.contact_mask,
        contact_jacobian=state.fingertip_jacobian_b,
        wrench_map=_wrench_map(state.fingertip_positions_b),
        target_wrench_b=target_wrench,
        q_min=torch.zeros(6, dtype=torch.float64),
        q_max=torch.tensor([0.58, 1.36, 1.60, 1.60, 1.60, 1.60], dtype=torch.float64),
        qd_max=torch.ones(6, dtype=torch.float64),
        phase=BimanualPhase.GRASP,
        palm_pose_b=arm_state.palm_pose_b,
    )


class BimanualRuntime:
    """Run due layers and commit their outputs only after a feasible WBC cycle."""

    OBJECT_PERIOD = 8
    ARM_PERIOD = 4
    HAND_PERIOD = 2

    def __init__(
        self,
        *,
        object_mpc=None,
        arm_mpc=None,
        left_hand_mpc=None,
        right_hand_mpc=None,
        wbc=None,
        mission: BimanualMission | None = None,
        object_input_provider: ObjectInputProvider | None = None,
        arm_input_provider: ArmInputProvider | None = None,
        hand_input_provider: HandInputProvider | None = None,
        collision_provider: CollisionProvider | None = None,
        motion_primitive: BimanualMotionPrimitive | None = None,
        grasp_goal: BimanualGraspGoal | None = None,
        grasp_goal_provider: GraspGoalProvider | None = None,
    ) -> None:
        if grasp_goal is not None and grasp_goal_provider is not None:
            raise ValueError("grasp_goal and grasp_goal_provider are mutually exclusive")
        if grasp_goal is not None and not isinstance(grasp_goal, BimanualGraspGoal):
            raise TypeError("grasp_goal must be BimanualGraspGoal or None")
        if grasp_goal_provider is not None and not callable(grasp_goal_provider):
            raise TypeError("grasp_goal_provider must be callable or None")
        self.object_mpc = BimanualObjectMpc() if object_mpc is None else object_mpc
        self.arm_mpc = DualArmMpcCoordinator() if arm_mpc is None else arm_mpc
        self.left_hand_mpc = O6HandMpc() if left_hand_mpc is None else left_hand_mpc
        self.right_hand_mpc = O6HandMpc() if right_hand_mpc is None else right_hand_mpc
        self.wbc = BimanualWholeBodyQp() if wbc is None else wbc
        self.mission = (
            BimanualMission(grasp_goal=grasp_goal)
            if mission is None
            else mission
        )
        if grasp_goal is not None:
            self.mission.set_grasp_goal(grasp_goal)
        self.motion_primitive = (
            BimanualMotionPrimitive() if motion_primitive is None else motion_primitive
        )
        for name in ("object_mpc", "arm_mpc", "left_hand_mpc", "right_hand_mpc"):
            if not callable(getattr(getattr(self, name), "plan", None)):
                raise TypeError(f"{name} must expose plan()")
        if not callable(getattr(self.wbc, "solve", None)):
            raise TypeError("wbc must expose solve()")
        self._object_input_provider = object_input_provider
        self._arm_input_provider = arm_input_provider
        self._hand_input_provider = hand_input_provider
        self._collision_provider = collision_provider
        self._grasp_goal = (
            grasp_goal if grasp_goal is not None else self.mission.grasp_goal
        )
        self._grasp_goal_provider = grasp_goal_provider
        self._step = 0
        self._counts = {"object": 0, "arm": 0, "hand": 0, "wbc": 0}
        self._last_snapshot: BimanualSnapshot | None = None
        self._last_object: ObjectMpcSolution | None = None
        self._last_arm: DualArmMpcSolution | None = None
        self._last_left_hand: HandMpcSolution | None = None
        self._last_right_hand: HandMpcSolution | None = None
        self._last_command: BimanualCommand | None = None
        self._initial_box_pose: torch.Tensor | None = None
        self._latest_motion_target: ManipulationTarget | None = None
        self._latest_solutions = {
            "object": None,
            "arm": None,
            "left_hand": None,
            "right_hand": None,
            "wbc": None,
        }

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def latest_solutions(self) -> dict[str, object | None]:
        """Return the most recent attempted layer outputs, including fallbacks."""

        return dict(self._latest_solutions)

    @property
    def latest_motion_target(self) -> ManipulationTarget | None:
        return self._latest_motion_target

    @property
    def latest_grasp_goal(self) -> BimanualGraspGoal | None:
        """Return the immutable geometry/contact goal used by the last target."""

        return None if self._latest_motion_target is None else self._latest_motion_target.grasp_goal

    @property
    def phase(self) -> BimanualPhase:
        """Current closed-loop phase (convenient for headless probes)."""

        return self.mission.phase

    def reset(self) -> None:
        """Clear temporal caches and object-planner state for deterministic replay."""
        reset_object = getattr(self.object_mpc, 'reset', None)
        if callable(reset_object):
            reset_object()
        self.motion_primitive.reset()

        self.mission = BimanualMission(
            cfg=self.mission.cfg,
            grasp_goal=self._grasp_goal,
        )
        self._step = 0
        self._counts = {"object": 0, "arm": 0, "hand": 0, "wbc": 0}
        self._last_snapshot = None
        self._last_object = None
        self._last_arm = None
        self._last_left_hand = None
        self._last_right_hand = None
        self._last_command = None
        self._initial_box_pose = None
        self._latest_motion_target = None
        self._latest_solutions = {
            "object": None,
            "arm": None,
            "left_hand": None,
            "right_hand": None,
            "wbc": None,
        }

    def _object_input(self, snapshot: BimanualSnapshot) -> ObjectMpcInput:
        if self._object_input_provider is not None:
            return self._object_input_provider(snapshot, self.mission.phase, self._last_object)
        # A caller may provide a preconfigured mission instead of the runtime
        # shorthand.  Preserve that goal while keeping the Box default None.
        grasp_goal = self._grasp_goal
        if grasp_goal is None:
            grasp_goal = self.mission.grasp_goal
        if self._grasp_goal_provider is not None:
            grasp_goal = self._grasp_goal_provider(snapshot)
            if grasp_goal is not None and not isinstance(grasp_goal, BimanualGraspGoal):
                raise TypeError("grasp_goal_provider must return BimanualGraspGoal or None")
        self.mission.set_grasp_goal(grasp_goal)
        if grasp_goal is None:
            # Keep the legacy motion-primitive call shape untouched for Box
            # users and third-party primitive implementations.
            self._latest_motion_target = self.motion_primitive.target(
                self.mission.phase, snapshot
            )
        else:
            self._latest_motion_target = self.motion_primitive.target(
                self.mission.phase, snapshot, grasp_goal
            )
        return ObjectMpcInput(
            snapshot=snapshot,
            target_box_pose_b=self._latest_motion_target.box_pose_b,
            target_object_pose_b=self._latest_motion_target.box_pose_b,
            obstacle_object_poses_b=tuple(
                obstacle.pose_b.repeat(OBJECT_MPC_HORIZON_STEPS, 1)
                for obstacle in snapshot.obstacle_objects
            ),
            phase=self.mission.phase,
            previous_solution=self._last_object,
            grasp_goal=grasp_goal,
        )

    def _arm_inputs(self, snapshot: BimanualSnapshot) -> tuple[ArmMpcInput, ArmMpcInput]:
        if self._arm_input_provider is not None:
            return self._arm_input_provider(snapshot)
        return _default_arm_input(snapshot, "left"), _default_arm_input(snapshot, "right")

    def _hand_input(
        self, snapshot: BimanualSnapshot, side: str, target: torch.Tensor
    ) -> HandMpcInput:
        if self._hand_input_provider is not None:
            return self._hand_input_provider(snapshot, side, target)
        sample = _default_hand_input(snapshot, side, target)
        return HandMpcInput(
            q=sample.q,
            qd=sample.qd,
            fingertip_forces_b=sample.fingertip_forces_b,
            fingertip_positions_b=sample.fingertip_positions_b,
            contact_mask=sample.contact_mask,
            contact_jacobian=sample.contact_jacobian,
            wrench_map=sample.wrench_map,
            target_wrench_b=sample.target_wrench_b,
            q_min=sample.q_min,
            q_max=sample.q_max,
            qd_max=sample.qd_max,
            phase=self.mission.phase,
            palm_pose_b=sample.palm_pose_b,
        )

    def _collision(self, snapshot: BimanualSnapshot) -> tuple[torch.Tensor, torch.Tensor]:
        if self._collision_provider is not None:
            return self._collision_provider(snapshot)
        return (
            torch.ones(1, dtype=torch.float64),
            torch.zeros((1, 43), dtype=torch.float64),
        )

    def _mission_diagnostics(
        self,
        snapshot: BimanualSnapshot,
        object_solution: ObjectMpcSolution,
        wbc_solution: BimanualWbcSolution,
    ) -> BimanualMissionDiagnostics:
        palm_errors = (
            torch.linalg.vector_norm(
                snapshot.left_arm.palm_pose_b[:3] - object_solution.left_palm_pose[-1, :3]
            ),
            torch.linalg.vector_norm(
                snapshot.right_arm.palm_pose_b[:3] - object_solution.right_palm_pose[-1, :3]
            ),
        )
        left_contact = bool(snapshot.left_hand.contact_mask.any())
        right_contact = bool(snapshot.right_hand.contact_mask.any())
        bilateral_contact = left_contact and right_contact
        slip = 0.0
        if self._last_snapshot is not None and bilateral_contact:
            for side in ("left", "right"):
                current_arm = getattr(snapshot, f"{side}_arm")
                previous_arm = getattr(self._last_snapshot, f"{side}_arm")
                current_offset = current_arm.palm_pose_b[:3] - snapshot.box.pose_b[:3]
                previous_offset = previous_arm.palm_pose_b[:3] - self._last_snapshot.box.pose_b[:3]
                slip = max(
                    slip,
                    float(torch.linalg.vector_norm(current_offset - previous_offset).item()),
                )
        reason = wbc_solution.diagnostics.fallback_reason
        left_forces = (
            torch.zeros((5, 3), dtype=torch.float64)
            if self._last_left_hand is None
            else self._last_left_hand.predicted_forces_b
        )
        right_forces = (
            torch.zeros((5, 3), dtype=torch.float64)
            if self._last_right_hand is None
            else self._last_right_hand.predicted_forces_b
        )
        left_normal_force = float(left_forces[:, 2].sum().item())
        right_normal_force = float(right_forces[:, 2].sum().item())
        vertical_force = left_normal_force + right_normal_force
        # BoxState stores a rotation vector in base coordinates.  Roll/pitch
        # are the relevant lift stability components; yaw does not tilt the
        # object and is intentionally excluded from this gate.
        object_tilt = float(torch.linalg.vector_norm(snapshot.box.pose_b[3:5]).item())
        return BimanualMissionDiagnostics(
            command_accepted=wbc_solution.feasible,
            palms_reached=bool(
                max(float(value) for value in palm_errors)
                <= self.mission.cfg.palm_position_tolerance_m
            ),
            left_palm_reached=bool(
                float(palm_errors[0]) <= self.mission.cfg.palm_position_tolerance_m
            ),
            right_palm_reached=bool(
                float(palm_errors[1]) <= self.mission.cfg.palm_position_tolerance_m
            ),
            left_contact=left_contact,
            right_contact=right_contact,
            contact_consistent=bool(
                snapshot.left_hand.contact_consistent
                and snapshot.right_hand.contact_consistent
            ),
            bilateral_contact=bilateral_contact,
            force_closure_margin=wbc_solution.diagnostics.force_closure_margin,
            relative_palm_slip_m=slip,
            box_twist_norm=float(torch.linalg.vector_norm(snapshot.box.twist_b).item()),
            box_supported=snapshot.box.supported,
            hands_open=bool(
                snapshot.left_hand.q.max() <= 0.15
                and snapshot.right_hand.q.max() <= 0.15
            ),
            collision_margin_m=wbc_solution.diagnostics.min_collision_distance - 0.02,
            support_margin_m=wbc_solution.diagnostics.support_margin,
            subsystem_failure=None if wbc_solution.feasible else (reason or "wbc_infeasible"),
            left_contact_count=int(snapshot.left_hand.contact_mask.sum().item()),
            right_contact_count=int(snapshot.right_hand.contact_mask.sum().item()),
            left_normal_force_n=left_normal_force,
            right_normal_force_n=right_normal_force,
            vertical_force_n=vertical_force,
            object_tilt_rad=object_tilt,
        )

    def compute(self, snapshot: BimanualSnapshot) -> BimanualCommand:
        if not isinstance(snapshot, BimanualSnapshot):
            raise TypeError("snapshot must be BimanualSnapshot")
        if self._last_snapshot is not None:
            validate_monotonic_snapshot(self._last_snapshot, snapshot)

        object_solution = self._last_object
        arm_solution = self._last_arm
        left_hand_solution = self._last_left_hand
        right_hand_solution = self._last_right_hand
        if self._step % self.OBJECT_PERIOD == 0 or object_solution is None:
            object_solution = self.object_mpc.plan(self._object_input(snapshot))
            self._counts["object"] += 1
        if self._step % self.ARM_PERIOD == 0 or arm_solution is None:
            left_input, right_input = self._arm_inputs(snapshot)
            arm_solution = self.arm_mpc.plan(
                DualArmMpcInput(left_input, right_input, object_solution)
            )
            self._counts["arm"] += 1
        if self._step % self.HAND_PERIOD == 0 or left_hand_solution is None or right_hand_solution is None:
            left_hand_solution = self.left_hand_mpc.plan(
                self._hand_input(snapshot, "left", object_solution.left_wrench[0])
            )
            right_hand_solution = self.right_hand_mpc.plan(
                self._hand_input(snapshot, "right", object_solution.right_wrench[0])
            )
            self._counts["hand"] += 1
        distances, collision_jacobian = self._collision(snapshot)
        request = BimanualWbcRequest(
            snapshot=snapshot,
            object_solution=object_solution,
            arm_solution=arm_solution,
            left_hand_solution=left_hand_solution,
            right_hand_solution=right_hand_solution,
            collision_distances=distances,
            collision_jacobian=collision_jacobian,
        )
        wbc_solution = self.wbc.solve(request)
        self._counts["wbc"] += 1
        self._latest_solutions = {
            "object": object_solution,
            "arm": arm_solution,
            "left_hand": left_hand_solution,
            "right_hand": right_hand_solution,
            "wbc": wbc_solution,
        }
        if wbc_solution.feasible:
            command = BimanualCommand(
                timestamp_ns=snapshot.timestamp_ns,
                effort=wbc_solution.effort,
                feasible=True,
                fallback_reasons=(),
            )
            self._last_object = object_solution
            self._last_arm = arm_solution
            self._last_left_hand = left_hand_solution
            self._last_right_hand = right_hand_solution
            self._last_command = command
        else:
            reason = wbc_solution.diagnostics.fallback_reason or "wbc_infeasible"
            effort = (
                wbc_solution.effort
                if self._last_command is None
                else self._last_command.effort
            )
            command = BimanualCommand(
                timestamp_ns=snapshot.timestamp_ns,
                effort=effort,
                feasible=False,
                fallback_reasons=(reason,),
            )
        diagnostics = self._mission_diagnostics(snapshot, object_solution, wbc_solution)
        self.mission.update(snapshot, diagnostics)
        self._last_snapshot = snapshot
        self._step += 1
        return command
