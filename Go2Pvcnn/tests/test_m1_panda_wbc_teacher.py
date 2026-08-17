from dataclasses import replace
import math

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.motion_distribution import (
    MotionDistributionResult,
)
from go2_pvcnn.control.m1_panda_coordination.qp_backend import DenseQpResult
from go2_pvcnn.control.m1_panda_coordination.safety import SafetyState
from go2_pvcnn.control.m1_panda_coordination.standing_wbc import (
    StandingWbcInput,
    StandingWbcResult,
)
from go2_pvcnn.control.m1_panda_coordination.teacher import (
    M1PandaWbcTeacher,
    TeacherCfg,
    TeacherState,
)
from go2_pvcnn.control.m1_panda_coordination.trajectory import (
    BandLimitedPoseTrajectory,
    BandLimitedTrajectoryCfg,
)


def _selector(rows, dimension=31):
    result = torch.zeros(len(rows), dimension, dtype=torch.float64)
    for output, source in enumerate(rows):
        result[output, source] = 1.0
    return result


def _wbc_input():
    return StandingWbcInput(
        mass_matrix=torch.eye(31, dtype=torch.float64),
        bias_force=torch.zeros(31, dtype=torch.float64),
        contact_jacobian=torch.zeros(12, 31, dtype=torch.float64),
        contact_jacobian_dot_qd=torch.zeros(12, dtype=torch.float64),
        mount_wrench_jacobian=torch.zeros(6, 31, dtype=torch.float64),
        external_wrench=torch.zeros(6, dtype=torch.float64),
        balance_jacobian=_selector((2, 3, 4)),
        balance_acceleration=torch.zeros(3, dtype=torch.float64),
        base_jacobian=_selector(tuple(range(6))),
        base_acceleration=torch.zeros(6, dtype=torch.float64),
        leg_generalized_indices=torch.arange(6, 18, dtype=torch.long),
        wheel_generalized_indices=torch.arange(18, 22, dtype=torch.long),
        arm_generalized_indices=torch.arange(22, 29, dtype=torch.long),
        leg_acceleration=torch.zeros(12, dtype=torch.float64),
        wheel_acceleration=torch.zeros(4, dtype=torch.float64),
        arm_acceleration=torch.zeros(7, dtype=torch.float64),
        qdd_lower=torch.full((31,), -10.0, dtype=torch.float64),
        qdd_upper=torch.full((31,), 10.0, dtype=torch.float64),
        effort_limit=torch.full((23,), 100.0, dtype=torch.float64),
        friction_coefficient=0.7,
    )


def _teacher_state(step=0):
    jacobian = torch.zeros(6, 10, dtype=torch.float64)
    jacobian[:, 3:9] = torch.eye(6, dtype=torch.float64)
    return TeacherState(
        physics_step=step,
        time_s=0.005 * step,
        ee_pose=torch.zeros(6, dtype=torch.float64),
        coordinated_jacobian=jacobian,
        coord_q=torch.zeros(10, dtype=torch.float64),
        coord_qd=torch.zeros(10, dtype=torch.float64),
        coord_q_min=torch.full((10,), -10.0, dtype=torch.float64),
        coord_q_max=torch.full((10,), 10.0, dtype=torch.float64),
        coord_v_max=torch.full((10,), 2.0, dtype=torch.float64),
        coord_a_max=torch.full((10,), 20.0, dtype=torch.float64),
        manipulability_gradient=torch.zeros(10, dtype=torch.float64),
        sigma_min=torch.tensor(1.0, dtype=torch.float64),
        wbc_input=_wbc_input(),
        controlled_q=torch.zeros(23, dtype=torch.float64),
        controlled_qd=torch.zeros(23, dtype=torch.float64),
        roll=0.0,
        pitch=0.0,
        wheel_contact_count=4,
        max_lateral_slip=0.0,
        signals_finite=True,
    )


def test_band_limited_trajectory_seed_is_repeatable_and_pose_starts_at_center():
    cfg = BandLimitedTrajectoryCfg(component_count=4)
    first = BandLimitedPoseTrajectory(cfg)
    second = BandLimitedPoseTrajectory(cfg)
    center = torch.tensor([0.4, -0.2, 0.7, 0.1, 0.0, -0.1], dtype=torch.float64)

    first.reset(center, seed=42)
    second.reset(center, seed=42)

    assert torch.equal(first.sample(0.0).pose, center)
    assert torch.equal(first.sample(0.0).twist, torch.zeros_like(center))
    for time_s in (0.0, 0.125, 1.0, 7.5):
        sample_a = first.sample(time_s)
        sample_b = second.sample(time_s)
        assert torch.equal(sample_a.pose, sample_b.pose)
        assert torch.equal(sample_a.twist, sample_b.twist)
        assert torch.equal(sample_a.acceleration, sample_b.acceleration)


def test_band_limited_trajectory_frequency_and_amplitude_contracts():
    trajectory = BandLimitedPoseTrajectory(BandLimitedTrajectoryCfg(component_count=5))
    center = torch.zeros(6, dtype=torch.float64)
    trajectory.reset(center, seed=7)

    assert torch.all(trajectory.frequencies_hz >= 0.05)
    assert torch.all(trajectory.frequencies_hz <= 0.25)
    samples = torch.stack([trajectory.sample(index * 0.02).pose for index in range(2000)])
    error = torch.abs(samples - center)
    assert torch.max(error[:, :3]).item() <= 0.08 + 1.0e-12
    assert torch.max(error[:, 3:]).item() <= 0.15 + 1.0e-12


def test_band_limited_trajectory_analytic_derivatives_match_finite_difference():
    trajectory = BandLimitedPoseTrajectory(BandLimitedTrajectoryCfg(component_count=3))
    trajectory.reset(torch.zeros(6, dtype=torch.float64), seed=9)
    time_s = 1.25
    epsilon = 1.0e-5

    sample = trajectory.sample(time_s)
    positive = trajectory.sample(time_s + epsilon)
    negative = trajectory.sample(time_s - epsilon)
    numerical_twist = (positive.pose - negative.pose) / (2.0 * epsilon)
    numerical_acceleration = (positive.twist - negative.twist) / (2.0 * epsilon)

    assert torch.allclose(sample.twist, numerical_twist, atol=1.0e-9, rtol=1.0e-7)
    assert torch.allclose(
        sample.acceleration,
        numerical_acceleration,
        atol=1.0e-9,
        rtol=1.0e-7,
    )


def test_band_limited_trajectory_rejects_sampling_before_reset_and_bad_time():
    trajectory = BandLimitedPoseTrajectory()
    with pytest.raises(RuntimeError, match="trajectory must be reset before sampling"):
        trajectory.sample(0.0)
    trajectory.reset(torch.zeros(6), seed=1)
    with pytest.raises(ValueError, match="time_s must be finite and non-negative"):
        trajectory.sample(float("nan"))


class _RecordingMotionDistributor:
    def __init__(self):
        self.steps = []

    def __call__(self, **kwargs):
        self.steps.append(kwargs.pop("physics_step"))
        qd = torch.zeros(10, dtype=torch.float64)
        qd[3:] = 0.1
        return MotionDistributionResult(
            qd_coord=qd,
            base_active=torch.tensor(False),
            sigma_min=kwargs["sigma_min"].clone(),
            phi=torch.tensor(1.0, dtype=torch.float64),
            psi=torch.tensor(1.0, dtype=torch.float64),
            saturated=torch.zeros(10, dtype=torch.bool),
        )


class _RecordingWbcSolver:
    def __init__(self, fail_on_calls=()):
        self.inputs = []
        self.fail_on_calls = set(fail_on_calls)

    def __call__(self, state):
        self.inputs.append(state)
        call = len(self.inputs)
        success = call not in self.fail_on_calls
        qp = DenseQpResult(
            solution=torch.zeros(43, dtype=torch.float64),
            success=success,
            iterations=1,
            max_equality_residual=0.0,
            max_inequality_violation=0.0 if success else 1.0,
            active_set=(),
        )
        return StandingWbcResult(
            qdd=torch.zeros(31, dtype=torch.float64),
            contact_force=torch.zeros(4, 3, dtype=torch.float64),
            effort=torch.full((23,), 0.2, dtype=torch.float64) if success else None,
            qp_result=qp,
            task_residuals={},
        )


def _teacher(motion=None, wbc=None):
    return M1PandaWbcTeacher(
        kp=torch.full((23,), 2.0, dtype=torch.float64),
        kd=torch.full((23,), 0.2, dtype=torch.float64),
        effort_limit=torch.full((23,), 100.0, dtype=torch.float64),
        safe_arm_target=torch.zeros(7, dtype=torch.float64),
        motion_distribution_fn=motion,
        wbc_solver_fn=wbc,
    )


def test_teacher_updates_distribution_at_steps_0_4_8_and_wbc_every_step():
    motion = _RecordingMotionDistributor()
    wbc = _RecordingWbcSolver()
    teacher = _teacher(motion, wbc)
    teacher.reset(_teacher_state(), seed=42)

    commands = [teacher.step(_teacher_state(step)) for step in range(9)]

    assert motion.steps == [0, 4, 8]
    assert len(wbc.inputs) == 9
    assert all(command.effort.shape == (23,) for command in commands)
    assert all(torch.isfinite(command.effort).all() for command in commands)


def test_velocity_command_becomes_acceleration_over_distribution_horizon():
    motion = _RecordingMotionDistributor()
    wbc = _RecordingWbcSolver()
    teacher = _teacher(motion, wbc)
    state = _teacher_state()
    teacher.reset(state, seed=42)

    teacher.step(state)

    assert torch.allclose(
        wbc.inputs[0].arm_acceleration,
        torch.full((7,), 5.0, dtype=torch.float64),
    )


def test_optional_warmup_holds_arm_then_starts_distribution_from_zero_time():
    motion = _RecordingMotionDistributor()
    wbc = _RecordingWbcSolver()
    teacher = M1PandaWbcTeacher(
        kp=torch.full((23,), 2.0, dtype=torch.float64),
        kd=torch.full((23,), 0.2, dtype=torch.float64),
        effort_limit=torch.full((23,), 100.0, dtype=torch.float64),
        safe_arm_target=torch.zeros(7, dtype=torch.float64),
        cfg=TeacherCfg(warmup_steps=2),
        motion_distribution_fn=motion,
        wbc_solver_fn=wbc,
    )
    teacher.reset(_teacher_state(), seed=42)

    commands = [teacher.step(_teacher_state(step)) for step in range(3)]

    assert motion.steps == [2]
    assert torch.equal(commands[0].motion_distribution.qd_coord, torch.zeros(10, dtype=torch.float64))
    assert torch.equal(commands[1].q_des[-7:], torch.zeros(7, dtype=torch.float64))


def test_teacher_uses_finite_lookahead_from_measured_arm_state_between_updates():
    motion = _RecordingMotionDistributor()
    teacher = _teacher(motion, _RecordingWbcSolver())
    teacher.reset(_teacher_state(), seed=3)

    commands = []
    for step in range(4):
        state = _teacher_state(step)
        controlled_q = state.controlled_q.clone()
        controlled_q[-7:] = 0.01 * step
        commands.append(teacher.step(replace(state, controlled_q=controlled_q)))

    expected = torch.stack(
        [torch.full((7,), 0.01 * step + 0.002, dtype=torch.float64) for step in range(4)]
    )
    assert torch.allclose(
        torch.stack([command.q_des[-7:] for command in commands]), expected
    )
    assert torch.equal(commands[-1].qd_des[-7:], torch.full((7,), 0.1, dtype=torch.float64))
    assert motion.steps == [0]


def test_c0_teacher_holds_zero_wheel_speed_through_impedance_boundary():
    motion = _RecordingMotionDistributor()
    teacher = _teacher(motion, _RecordingWbcSolver())
    state = _teacher_state()
    controlled_qd = state.controlled_qd.clone()
    controlled_qd[12:16] = 1.0
    state = replace(state, controlled_qd=controlled_qd)
    teacher.reset(state, seed=12)

    command = teacher.step(state)
    second = teacher.step(replace(state, physics_step=1, time_s=0.005))

    assert torch.equal(command.qd_des[12:16], torch.zeros(4, dtype=torch.float64))
    assert torch.all(command.effort[12:16] < 0.0)
    assert torch.all(second.effort[12:16] < command.effort[12:16])


def test_c0_teacher_anchors_legs_to_reset_posture_with_zero_speed():
    teacher = _teacher(_RecordingMotionDistributor(), _RecordingWbcSolver())
    reset_state = _teacher_state()
    teacher.reset(reset_state, seed=13)
    moved_q = reset_state.controlled_q.clone()
    moved_q[:12] = 0.1
    moved_qd = reset_state.controlled_qd.clone()
    moved_qd[:12] = 0.2

    command = teacher.step(
        replace(reset_state, controlled_q=moved_q, controlled_qd=moved_qd)
    )

    assert torch.equal(command.q_des[:12], reset_state.controlled_q[:12])
    assert torch.equal(command.qd_des[:12], torch.zeros(12, dtype=torch.float64))


def test_teacher_reset_with_same_seed_is_deterministic_and_clears_history():
    motion = _RecordingMotionDistributor()
    teacher = _teacher(motion, _RecordingWbcSolver())
    state = _teacher_state()

    teacher.reset(state, seed=77)
    first = teacher.step(state)
    teacher.reset(state, seed=77)
    second = teacher.step(state)

    assert torch.equal(first.target_pose, second.target_pose)
    assert torch.equal(first.target_twist, second.target_twist)
    assert torch.equal(first.effort, second.effort)
    assert first.safety_state == SafetyState.TRACK
    assert second.safety_state == SafetyState.TRACK


def test_safety_hold_overrides_arm_tracking_but_command_still_passes_through_wbc():
    motion = _RecordingMotionDistributor()
    wbc = _RecordingWbcSolver()
    teacher = _teacher(motion, wbc)
    teacher.reset(_teacher_state(), seed=4)

    commands = []
    for step in range(5):
        state = replace(_teacher_state(step), roll=math.radians(8.0))
        commands.append(teacher.step(state))

    assert commands[-1].safety_state == SafetyState.HOLD
    assert commands[-1].safety_reason == "warning_orientation"
    assert len(wbc.inputs) >= 5
    assert torch.equal(commands[-1].q_des[-7:], commands[-2].q_des[-7:])
    assert torch.equal(commands[-1].target_pose, torch.zeros(6, dtype=torch.float64))
    assert motion.steps == [0]


def test_recovery_from_hold_recenters_pose_and_arm_target_on_measured_state():
    teacher = _teacher(_RecordingMotionDistributor(), _RecordingWbcSolver())
    teacher.reset(_teacher_state(), seed=14)
    for step in range(4):
        teacher.step(replace(_teacher_state(step), roll=math.radians(8.0)))

    recovered = None
    for step in range(4, 24):
        state = _teacher_state(step)
        controlled_q = state.controlled_q.clone()
        controlled_q[-7:] = 0.1
        recovered = teacher.step(
            replace(
                state,
                ee_pose=torch.full((6,), 0.2, dtype=torch.float64),
                controlled_q=controlled_q,
            )
        )

    assert recovered is not None
    assert recovered.safety_state == SafetyState.SCALE
    assert torch.equal(recovered.target_pose, torch.full((6,), 0.2, dtype=torch.float64))
    assert torch.equal(recovered.q_des[-7:], torch.full((7,), 0.1, dtype=torch.float64))


def test_failed_wbc_never_reuses_unverified_effort_and_advances_safety():
    motion = _RecordingMotionDistributor()
    wbc = _RecordingWbcSolver(fail_on_calls=(2, 3))
    teacher = _teacher(motion, wbc)
    teacher.reset(_teacher_state(), seed=5)

    first = teacher.step(_teacher_state(0))
    second = teacher.step(_teacher_state(1))
    third = teacher.step(_teacher_state(2))

    assert torch.allclose(first.effort[:16], torch.full((16,), 0.2, dtype=torch.float64))
    assert torch.allclose(first.effort[-7:], torch.full((7,), 0.024, dtype=torch.float64))
    assert torch.isfinite(second.effort).all()
    assert torch.isfinite(third.effort).all()
    assert not torch.equal(second.effort, torch.full((23,), 0.2, dtype=torch.float64))
    assert third.safety_state >= SafetyState.SCALE


def test_failed_wbc_fallback_keeps_current_arm_bias_compensation():
    wbc = _RecordingWbcSolver(fail_on_calls=(2,))
    teacher = _teacher(_RecordingMotionDistributor(), wbc)
    state = _teacher_state()
    biased_wbc = replace(
        state.wbc_input,
        bias_force=torch.cat(
            (
                torch.zeros(22, dtype=torch.float64),
                torch.ones(7, dtype=torch.float64),
                torch.zeros(2, dtype=torch.float64),
            )
        ),
    )
    state = replace(state, wbc_input=biased_wbc)
    teacher.reset(state, seed=15)

    teacher.step(state)
    fallback = teacher.step(replace(state, physics_step=1, time_s=0.005))

    assert torch.allclose(
        fallback.effort[-7:], torch.full((7,), 1.024, dtype=torch.float64)
    )


def test_teacher_rejects_step_before_reset():
    teacher = _teacher(_RecordingMotionDistributor(), _RecordingWbcSolver())
    with pytest.raises(RuntimeError, match="teacher must be reset before step"):
        teacher.step(_teacher_state())
