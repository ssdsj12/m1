from dataclasses import replace
import math

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.motion_distribution import (
    MotionDistributionResult,
)
from go2_pvcnn.control.m1_panda_coordination.qp_backend import DenseQpResult
from go2_pvcnn.control.m1_panda_coordination.rolling_teacher import (
    LongitudinalCommandSchedule,
    LongitudinalScheduleCfg,
    M1PandaRollingWbcTeacher,
    PlanarBodyFrameTrajectory,
    RollingTeacherState,
)
from go2_pvcnn.control.m1_panda_coordination.whole_body_residual import (
    WholeBodyResidualCommand,
)
from go2_pvcnn.control.m1_panda_coordination.safety import SafetyState
from go2_pvcnn.control.m1_panda_coordination.student_contracts import (
    StudentNominalCommand,
)
from go2_pvcnn.control.m1_panda_coordination.student_mission import (
    StudentMissionSample,
)
from go2_pvcnn.control.m1_panda_coordination.standing_wbc import (
    StandingWbcInput,
    StandingWbcResult,
)
from go2_pvcnn.control.m1_panda_coordination.teacher import TeacherState
from go2_pvcnn.control.m1_panda_coordination.trajectory import (
    BandLimitedTrajectoryCfg,
    TrajectorySample,
)


def test_schedule_has_five_800_step_phases_and_rate_limits_boundaries():
    schedule = LongitudinalCommandSchedule(LongitudinalScheduleCfg())
    schedule.reset()

    for step in range(800):
        command = schedule.sample(step)

    assert command.phase == 0
    assert command.raw_target_mps == pytest.approx(0.0)
    first_forward = schedule.sample(800)
    assert first_forward.phase == 1
    assert first_forward.raw_target_mps == pytest.approx(0.05)
    assert first_forward.shaped_target_mps == pytest.approx(0.0005)
    for step in range(801, 4000):
        command = schedule.sample(step)
    assert command.phase == 4
    assert command.raw_target_mps == pytest.approx(-0.05)


def test_hold_scale_requests_rate_limited_stop_instead_of_locking_wheels():
    schedule = LongitudinalCommandSchedule(LongitudinalScheduleCfg())
    schedule.reset()
    for step in range(1000):
        command = schedule.sample(step)

    stopped = schedule.sample(1000, safety_scale=0.0)

    assert 0.0 < stopped.shaped_target_mps < command.shaped_target_mps
    assert command.shaped_target_mps - stopped.shaped_target_mps == pytest.approx(
        0.0005
    )


def test_schedule_rejects_skipped_or_repeated_steps():
    schedule = LongitudinalCommandSchedule()
    schedule.sample(0)
    with pytest.raises(ValueError, match="mission_step must advance exactly once"):
        schedule.sample(0)
    with pytest.raises(ValueError, match="mission_step must advance exactly once"):
        schedule.sample(2)


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan")])
def test_schedule_rejects_invalid_safety_scale(scale):
    schedule = LongitudinalCommandSchedule()
    with pytest.raises(ValueError, match="safety_scale must be finite and in"):
        schedule.sample(0, safety_scale=scale)


def test_schedule_reset_replays_the_same_commands():
    schedule = LongitudinalCommandSchedule()
    first = [schedule.sample(step) for step in range(900)]
    schedule.reset()
    second = [schedule.sample(step) for step in range(900)]
    assert first == second


def test_body_frame_center_advects_with_root_without_arm_extension():
    trajectory = PlanarBodyFrameTrajectory(
        BandLimitedTrajectoryCfg(
            position_amplitude=0.0,
            orientation_amplitude=0.0,
        )
    )
    ee = torch.tensor(
        [1.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    root = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    trajectory.reset(ee, root, seed=42)

    moved = trajectory.sample(
        1.0,
        torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
    )

    assert moved.pose[0].item() == pytest.approx(1.2)
    assert moved.pose[1].item() == pytest.approx(0.0)
    assert moved.twist[0].item() == pytest.approx(0.1)


def test_body_frame_center_rotates_with_heading_and_includes_yaw_twist():
    trajectory = PlanarBodyFrameTrajectory(
        BandLimitedTrajectoryCfg(
            position_amplitude=0.0,
            orientation_amplitude=0.0,
        )
    )
    trajectory.reset(
        torch.tensor([1.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        seed=7,
    )

    sample = trajectory.sample(
        0.0,
        torch.tensor([0.0, 0.0, torch.pi / 2.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64),
    )

    assert sample.pose[0].item() == pytest.approx(0.0, abs=1.0e-12)
    assert sample.pose[1].item() == pytest.approx(1.0)
    assert sample.pose[5].item() == pytest.approx(torch.pi / 2.0)
    assert sample.twist[0].item() == pytest.approx(-0.2)
    assert sample.twist[1].item() == pytest.approx(0.0, abs=1.0e-12)
    assert sample.twist[5].item() == pytest.approx(0.2)


def test_body_frame_trajectory_is_seed_repeatable_after_reset():
    cfg = BandLimitedTrajectoryCfg(
        position_amplitude=0.005,
        orientation_amplitude=0.01,
    )
    trajectory = PlanarBodyFrameTrajectory(cfg)
    ee = torch.tensor(
        [0.4, -0.1, 0.8, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    root = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    root_velocity = torch.zeros(3, dtype=torch.float64)

    trajectory.reset(ee, root, seed=21)
    first = trajectory.sample(1.25, root, root_velocity)
    trajectory.reset(ee, root, seed=21)
    second = trajectory.sample(1.25, root, root_velocity)

    assert torch.equal(first.pose, second.pose)
    assert torch.equal(first.twist, second.twist)
    assert torch.equal(first.acceleration, second.acceleration)


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
        qdd_lower=torch.full((31,), -100.0, dtype=torch.float64),
        qdd_upper=torch.full((31,), 100.0, dtype=torch.float64),
        effort_limit=torch.full((23,), 100.0, dtype=torch.float64),
        friction_coefficient=0.7,
    )


def _teacher_state(step=0):
    jacobian = torch.zeros(6, 10, dtype=torch.float64)
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    jacobian[5, 2] = 1.0
    jacobian[:, 3:9] = torch.eye(6, dtype=torch.float64)
    return TeacherState(
        physics_step=step,
        time_s=step * 0.005,
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


def _rolling_state(step=0):
    return RollingTeacherState(
        mission_step=step,
        teacher_state=_teacher_state(step),
        root_xy_yaw=torch.zeros(3, dtype=torch.float64),
        root_vxy_yawrate=torch.zeros(3, dtype=torch.float64),
        max_rolling_residual_mps=0.0,
    )


class _RecordingMotionDistributor:
    def __init__(self):
        self.inputs = []

    def __call__(self, **kwargs):
        self.inputs.append(kwargs)
        qd = torch.zeros(10, dtype=torch.float64)
        qd[:3] = kwargs["prescribed_base_velocity"]
        qd[3:] = 0.1
        return MotionDistributionResult(
            qd_coord=qd,
            base_active=torch.tensor(bool(torch.any(qd[:3] != 0.0))),
            base_participation=torch.tensor(
                float(torch.any(qd[:3] != 0.0).item()), dtype=torch.float64
            ),
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
        success = len(self.inputs) not in self.fail_on_calls
        return StandingWbcResult(
            qdd=torch.zeros(31, dtype=torch.float64),
            contact_force=torch.zeros(4, 3, dtype=torch.float64),
            effort=(
                torch.full((23,), 0.2, dtype=torch.float64)
                if success
                else None
            ),
            qp_result=DenseQpResult(
                solution=torch.zeros(43, dtype=torch.float64),
                success=success,
                iterations=1,
                max_equality_residual=0.0,
                max_inequality_violation=0.0 if success else 1.0,
                active_set=(),
            ),
            task_residuals={},
        )


class _RecordingTrajectory:
    def __init__(self):
        self.root_velocities = []

    def reset(self, center_pose, root_xy_yaw, *, seed):
        return None

    def sample(self, time_s, root_xy_yaw, root_vxy_yawrate):
        self.root_velocities.append(root_vxy_yawrate.clone())
        return TrajectorySample(
            pose=torch.zeros(6, dtype=torch.float64),
            twist=torch.zeros(6, dtype=torch.float64),
            acceleration=torch.zeros(6, dtype=torch.float64),
        )


def _rolling_teacher(motion=None, wbc=None, trajectory=None):
    return M1PandaRollingWbcTeacher(
        kp=torch.full((23,), 2.0, dtype=torch.float64),
        kd=torch.full((23,), 0.2, dtype=torch.float64),
        effort_limit=torch.full((23,), 100.0, dtype=torch.float64),
        safe_arm_target=torch.zeros(7, dtype=torch.float64),
        trajectory=trajectory,
        motion_distribution_fn=motion,
        wbc_solver_fn=wbc,
    )


def test_injected_student_mission_drives_teacher_without_advancing_private_schedule():
    teacher = _rolling_teacher(
        _RecordingMotionDistributor(), _RecordingWbcSolver()
    )
    state = _rolling_state(0)
    teacher.reset(state, seed=42)
    target_pose = torch.tensor(
        [0.4, 0.0, 0.8, 0.01, 0.0, 0.0], dtype=torch.float64
    )
    target_twist = torch.tensor(
        [0.02, 0.0, 0.0, 0.0, 0.01, 0.0], dtype=torch.float64
    )
    injected = StudentMissionSample(
        phase=3,
        shaped_vx=0.02,
        target_pose=target_pose,
        target_twist=target_twist,
        nominal=StudentNominalCommand(
            position=torch.zeros(1, 23, dtype=torch.float64),
            velocity=torch.zeros(1, 23, dtype=torch.float64),
        ),
    )

    command = teacher.step(state, mission_sample=injected)

    assert command.phase == 3
    assert command.shaped_base_velocity_mps == pytest.approx(0.02)
    torch.testing.assert_close(command.target_pose, target_pose)
    torch.testing.assert_close(command.target_twist, target_twist)

    private = teacher.step(state)
    assert private.phase == 0


def test_rolling_teacher_advects_body_trajectory_with_commanded_not_measured_base_velocity():
    trajectory = _RecordingTrajectory()
    teacher = _rolling_teacher(
        _RecordingMotionDistributor(),
        _RecordingWbcSolver(),
        trajectory,
    )
    state = replace(
        _rolling_state(0),
        root_vxy_yawrate=torch.tensor(
            [0.2, -0.1, 0.3], dtype=torch.float64
        ),
    )
    teacher.reset(state, seed=42)

    teacher.step(state)

    assert torch.equal(
        trajectory.root_velocities[-1],
        torch.zeros(3, dtype=torch.float64),
    )


def test_restart_mission_preserves_settled_low_level_state_and_restarts_schedule():
    teacher = _rolling_teacher(
        _RecordingMotionDistributor(), _RecordingWbcSolver()
    )
    state = _rolling_state(0)
    state = replace(
        state,
        teacher_state=replace(
            state.teacher_state,
            controlled_qd=torch.cat(
                (
                    torch.zeros(12, dtype=torch.float64),
                    torch.ones(4, dtype=torch.float64),
                    torch.zeros(7, dtype=torch.float64),
                )
            ),
        ),
    )
    teacher.reset(state, seed=42)
    teacher.step(state)
    settled_integral = teacher._wheel_velocity_integral.clone()
    settled_leg_target = teacher._leg_target.clone()

    teacher.restart_mission(state, seed=42)

    assert torch.equal(teacher._wheel_velocity_integral, settled_integral)
    assert torch.equal(teacher._leg_target, settled_leg_target)
    command = teacher.step(state)
    assert command.phase == 0
    assert command.shaped_base_velocity_mps == pytest.approx(0.0)


def test_rolling_teacher_uses_50hz_distribution_and_200hz_wbc():
    motion = _RecordingMotionDistributor()
    wbc = _RecordingWbcSolver()
    teacher = _rolling_teacher(motion, wbc)
    teacher.reset(_rolling_state(0), seed=42)

    commands = [teacher.step(_rolling_state(step)) for step in range(9)]

    assert len(motion.inputs) == 3
    assert len(wbc.inputs) == 9
    assert all(command.effort.shape == (23,) for command in commands)


def test_rolling_teacher_prescribes_base_and_nonzero_wheel_speed_after_phase_boundary():
    motion = _RecordingMotionDistributor()
    wbc = _RecordingWbcSolver()
    teacher = _rolling_teacher(motion, wbc)
    teacher.reset(_rolling_state(0), seed=42)

    for step in range(801):
        command = teacher.step(_rolling_state(step))

    assert command.phase == 1
    assert command.shaped_base_velocity_mps == pytest.approx(0.0005)
    assert torch.all(command.wheel_velocity_target > 0.0)
    assert motion.inputs[-1]["prescribed_base_velocity"][0].item() > 0.0
    assert torch.equal(command.qd_des[12:16], command.wheel_velocity_target)


def test_rolling_teacher_expands_only_longitudinal_position_bounds():
    motion = _RecordingMotionDistributor()
    teacher = _rolling_teacher(motion, _RecordingWbcSolver())
    state = _rolling_state(0)
    state = replace(
        state,
        teacher_state=replace(
            state.teacher_state,
            coord_q_min=torch.tensor(
                [-0.25, -0.25, -0.5] + [-2.0] * 7,
                dtype=torch.float64,
            ),
            coord_q_max=torch.tensor(
                [0.25, 0.25, 0.5] + [2.0] * 7,
                dtype=torch.float64,
            ),
        ),
    )
    teacher.reset(state, seed=42)

    teacher.step(state)

    inputs = motion.inputs[-1]
    assert inputs["q_min"][0].item() == pytest.approx(-2.0)
    assert inputs["q_max"][0].item() == pytest.approx(2.0)
    assert torch.equal(inputs["q_min"][1:], state.teacher_state.coord_q_min[1:])
    assert torch.equal(inputs["q_max"][1:], state.teacher_state.coord_q_max[1:])


def test_hold_ramps_wheel_target_toward_zero_and_freezes_arm_target():
    teacher = _rolling_teacher(
        _RecordingMotionDistributor(), _RecordingWbcSolver()
    )
    teacher.reset(_rolling_state(0), seed=42)
    commands = []
    for step in range(806):
        state = _rolling_state(step)
        if step >= 801:
            state = replace(
                state,
                teacher_state=replace(
                    state.teacher_state,
                    roll=math.radians(8.0),
                ),
            )
        commands.append(teacher.step(state))

    assert commands[-1].safety_state >= SafetyState.HOLD
    assert abs(commands[-1].shaped_base_velocity_mps) < abs(
        commands[-2].shaped_base_velocity_mps
    )
    assert torch.equal(commands[-1].q_des[-7:], commands[-2].q_des[-7:])


def test_failed_rolling_wbc_uses_finite_fallback_and_advances_safety():
    wbc = _RecordingWbcSolver(fail_on_calls=(2, 3))
    teacher = _rolling_teacher(_RecordingMotionDistributor(), wbc)
    teacher.reset(_rolling_state(0), seed=5)

    first = teacher.step(_rolling_state(0))
    second = teacher.step(_rolling_state(1))
    third = teacher.step(_rolling_state(2))

    assert torch.isfinite(first.effort).all()
    assert torch.isfinite(second.effort).all()
    assert torch.isfinite(third.effort).all()
    assert third.safety_state >= SafetyState.SCALE


def test_rolling_teacher_reset_clears_schedule_and_is_deterministic():
    teacher = _rolling_teacher(
        _RecordingMotionDistributor(), _RecordingWbcSolver()
    )
    state = _rolling_state(0)

    teacher.reset(state, seed=77)
    first = teacher.step(state)
    teacher.reset(state, seed=77)
    second = teacher.step(state)

    assert first.phase == second.phase == 0
    assert first.shaped_base_velocity_mps == second.shaped_base_velocity_mps
    assert torch.equal(first.target_pose, second.target_pose)
    assert torch.equal(first.effort, second.effort)
def _rolling_residual_command():
    physical = torch.tensor(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.01, 0.02],
        dtype=torch.float64,
    )
    return WholeBodyResidualCommand(
        physical=physical,
        wrench_b=physical[:6].clone(),
        delta_height=physical[6].clone(),
        delta_stance=physical[7].clone(),
    )


def test_rolling_teacher_accepts_same_residual_wbc_contract():
    wbc = _RecordingWbcSolver()
    teacher = _rolling_teacher(_RecordingMotionDistributor(), wbc)
    state = _rolling_state(0)
    teacher.reset(state, seed=5)

    command = teacher.step(
        state,
        residual_command=_rolling_residual_command(),
        leg_soft_limits=torch.tensor([[-1.0, 1.0]] * 12, dtype=torch.float64),
    )

    assert torch.equal(
        wbc.inputs[0].external_wrench,
        torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0], dtype=torch.float64),
    )
    assert command.q_des[0].item() == pytest.approx(0.02)
    assert command.q_des[3].item() == pytest.approx(-0.02)
