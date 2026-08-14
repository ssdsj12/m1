from pathlib import Path

import pytest
import go2_pvcnn.tasks.m1_curriculum as m1_curriculum

from go2_pvcnn.tasks.m1_curriculum import discover_latest_checkpoint, evaluate_obstacle_gate, evaluate_roll_gate
from go2_pvcnn.tasks.m1_curriculum import (
    build_wave_reference_actions,
    build_semantic_spatial_wave_reference,
    update_wave_encounter_phase,
    spatial_pair_lift_score,
    spatial_wheel_lift_score,
    expand_checkpoint_observations,
    prepare_wave_checkpoint,
    update_semantic_crossing_tracker,
    semantic_obstacle_ahead_mask,
    build_teacher_student_residual,
    smooth_wave_reference_actions,
    build_spatial_axle_wheel_targets,
    build_temporal_axle_wheel_targets,
    build_lateral_steering_correction,
    update_wheel_obstacle_clearance,
    update_clearance_drive_release,
    wheel_crossbar_collision_mask,
    wheel_x_from_fixed_obstacle,
    update_sequential_wheel_crossing_reference,
    sequential_wheel_crossing_progress_score,
    build_task_space_wheel_joint_actions,
    resolve_m1_leg_joint_ids_by_wheel,
    build_stabilized_task_space_wheel_actions,
    compose_sequential_leg_actions,
    scheduled_student_rollout_weight,
    build_sequential_phase_observation,
    blend_policy_wave_gate,
    apply_fixed_course_gate_safety_window,
)


def test_fixed_crossbar_progress_uses_world_x_not_robot_heading():
    import torch

    wheel_x = wheel_x_from_fixed_obstacle(
        wheel_pos_w=torch.tensor([[[1.20, 5.0, 0.2], [0.80, -4.0, 0.2]]]),
        root_pos_w=torch.tensor([[1.00, 0.0, 0.6]]),
        obstacle_x_from_root=torch.tensor([0.10]),
    )

    torch.testing.assert_close(wheel_x, torch.tensor([[0.10, -0.30]]))


def test_spatial_pair_lift_score_requires_both_wheels_in_active_pair():
    import torch

    score = spatial_pair_lift_score(
        wheel_heights=torch.tensor(
            [
                [0.26, 0.26, 0.096, 0.096],
                [0.26, 0.10, 0.096, 0.096],
                [0.26, 0.26, 0.26, 0.26],
            ]
        ),
        obstacle_x=torch.tensor([0.30, 0.30, -0.30]),
        active=torch.tensor([True, True, True]),
        baseline_height=0.096,
        target_height=0.26,
    )

    assert score[0].item() == pytest.approx(1.0)
    assert score[1].item() < 0.05
    assert score[2].item() == pytest.approx(1.0)


def test_spatial_wheel_lift_score_rewards_one_wheel_without_requiring_pair_lift():
    import torch

    score = spatial_wheel_lift_score(
        wheel_heights=torch.tensor([[0.26, 0.096, 0.096, 0.096]]),
        obstacle_x=torch.tensor([0.30]),
        active=torch.tensor([True]),
        baseline_height=0.096,
        target_height=0.26,
    )

    assert score.item() == pytest.approx(0.5)


def test_sequential_teacher_mode_adds_scaled_policy_residual():
    import torch

    composed = compose_sequential_leg_actions(
        policy_actions=torch.tensor([[2.0, -4.0]]),
        teacher_actions=torch.tensor([[0.5, 1.0]]),
        residual_scale=torch.tensor([[0.25, 0.50]]),
        policy_control=False,
    )

    torch.testing.assert_close(composed, torch.tensor([[1.0, -1.0]]))


def test_sequential_policy_control_replaces_teacher_reference():
    import torch

    composed = compose_sequential_leg_actions(
        policy_actions=torch.tensor([[0.2, -0.4]]),
        teacher_actions=torch.tensor([[9.0, 9.0]]),
        residual_scale=torch.zeros(1, 2),
        policy_control=True,
        policy_weight=1.0,
    )

    torch.testing.assert_close(composed, torch.tensor([[0.2, -0.4]]))


def test_sequential_policy_control_blends_current_teacher_and_student():
    import torch

    composed = compose_sequential_leg_actions(
        policy_actions=torch.tensor([[1.0, -1.0]]),
        teacher_actions=torch.tensor([[0.0, 1.0]]),
        residual_scale=torch.zeros(1, 2),
        policy_control=True,
        policy_weight=0.25,
    )

    torch.testing.assert_close(composed, torch.tensor([[0.25, 0.5]]))


def test_student_rollout_schedule_holds_teacher_before_ramping():
    assert scheduled_student_rollout_weight(
        update=200,
        total_updates=1000,
        final_weight=0.8,
        teacher_forcing_fraction=0.25,
    ) == pytest.approx(0.0)
    assert scheduled_student_rollout_weight(
        update=525,
        total_updates=1000,
        final_weight=0.8,
        teacher_forcing_fraction=0.25,
    ) == pytest.approx(0.4)
    assert scheduled_student_rollout_weight(
        update=900,
        total_updates=1000,
        final_weight=0.8,
        teacher_forcing_fraction=0.25,
    ) == pytest.approx(0.8)


def test_sequential_phase_observation_disambiguates_wave_stage_and_progress():
    import torch

    features = build_sequential_phase_observation(
        phase=torch.tensor([-1, 0, 7, 11]),
        phase_steps=torch.tensor([0, 5, 25, 100]),
        progress_steps=50,
    )

    assert features.shape == (4, 14)
    torch.testing.assert_close(features[:, :13].sum(dim=1), torch.ones(4))
    assert features[0, 0].item() == pytest.approx(1.0)
    assert features[1, 1].item() == pytest.approx(1.0)
    assert features[2, 8].item() == pytest.approx(1.0)
    assert features[3, 12].item() == pytest.approx(1.0)
    torch.testing.assert_close(
        features[:, 13], torch.tensor([0.0, 0.1, 0.5, 1.0])
    )


def test_policy_wave_gate_blends_oracle_teacher_and_student_score():
    import torch

    oracle = torch.tensor([True, False, True, False])
    score = torch.tensor([-2.0, 2.0, 2.0, -2.0])

    teacher_gate, _ = blend_policy_wave_gate(
        oracle_gate=oracle, policy_score=score, policy_weight=0.0
    )
    student_gate, blended = blend_policy_wave_gate(
        oracle_gate=oracle, policy_score=score, policy_weight=1.0
    )

    assert torch.equal(teacher_gate, oracle)
    assert torch.equal(student_gate, torch.tensor([False, True, True, False]))
    torch.testing.assert_close(blended, score)


def test_fixed_course_gate_window_blocks_early_trigger_and_recovers_late_miss():
    import torch

    gate, fallback = apply_fixed_course_gate_safety_window(
        policy_gate=torch.tensor([True, True, False, False]),
        oracle_gate=torch.tensor([True, True, True, False]),
        root_local_x=torch.tensor([0.10, 0.12, 0.15, 0.15]),
        minimum_root_x=0.115,
        fallback_root_x=0.14,
    )

    assert torch.equal(gate, torch.tensor([False, True, True, False]))
    assert torch.equal(fallback, torch.tensor([False, False, True, False]))


def test_wheel_clearance_rejects_contact_rollover_and_accepts_active_lift():
    import torch

    prelift = torch.zeros((1, 4), dtype=torch.bool)
    overbar = torch.zeros((1, 4), dtype=torch.bool)
    prelift, overbar, _ = update_wheel_obstacle_clearance(
        wheel_pos_local=torch.tensor(
            [[[0.50, 0.0, 0.165], [0.50, 0.0, 0.165], [0.50, 0.0, 0.150], [0.50, 0.0, 0.150]]]
        ),
        wheel_contact_force=torch.zeros((1, 4)),
        prelift_seen=prelift,
        overbar_clear_seen=overbar,
        obstacle_center_x=0.65,
        obstacle_size_x=0.06,
        obstacle_height=0.06,
        wheel_radius=0.095,
        clearance_margin=0.005,
        contact_force_limit=1.0,
    )
    assert prelift.tolist() == [[True, True, False, False]]

    prelift, overbar, required_height = update_wheel_obstacle_clearance(
        wheel_pos_local=torch.tensor(
            [[[0.65, 0.0, 0.170], [0.65, 0.0, 0.170], [0.65, 0.0, 0.155], [0.65, 0.0, 0.155]]]
        ),
        wheel_contact_force=torch.tensor([[0.0, 0.0, 40.0, 40.0]]),
        prelift_seen=prelift,
        overbar_clear_seen=overbar,
        obstacle_center_x=0.65,
        obstacle_size_x=0.06,
        obstacle_height=0.06,
        wheel_radius=0.095,
        clearance_margin=0.005,
        contact_force_limit=1.0,
    )

    assert required_height == pytest.approx(0.16)
    assert overbar.tolist() == [[True, True, False, False]]
    assert (prelift & overbar).tolist() == [[True, True, False, False]]


def test_wheel_clearance_ignores_wheels_outside_narrow_crossbar_track():
    import torch

    prelift, overbar, _ = update_wheel_obstacle_clearance(
        wheel_pos_local=torch.tensor(
            [[[0.50, -0.20, 0.17], [0.50, 0.20, 0.17], [0.65, -0.20, 0.17], [0.65, 0.20, 0.17]]]
        ),
        wheel_contact_force=torch.zeros((1, 4)),
        prelift_seen=torch.zeros((1, 4), dtype=torch.bool),
        overbar_clear_seen=torch.zeros((1, 4), dtype=torch.bool),
        obstacle_center_x=0.65,
        obstacle_size_x=0.06,
        obstacle_center_y=-0.20,
        obstacle_size_y=0.16,
        obstacle_height=0.06,
        wheel_radius=0.095,
        clearance_margin=0.005,
        contact_force_limit=1.0,
    )

    assert prelift.tolist() == [[True, False, False, False]]
    assert overbar.tolist() == [[False, False, True, False]]


def test_clearance_drive_release_holds_then_releases_each_axle_with_hysteresis():
    import torch

    axle = torch.full((1,), -1, dtype=torch.long)
    released = torch.zeros(1, dtype=torch.bool)
    axle, released, drive = update_clearance_drive_release(
        obstacle_x=torch.tensor([0.40]),
        wave_gate=torch.tensor([True]),
        wheel_heights=torch.tensor([[0.10, 0.10, 0.10, 0.10]]),
        previous_axle=axle,
        previous_released=released,
        required_height=0.16,
        axle_switch_x=0.05,
    )
    assert axle.tolist() == [0]
    assert released.tolist() == [False]
    assert drive.tolist() == [False]

    axle, released, drive = update_clearance_drive_release(
        obstacle_x=torch.tensor([0.35]),
        wave_gate=torch.tensor([True]),
        wheel_heights=torch.tensor([[0.17, 0.17, 0.10, 0.10]]),
        previous_axle=axle,
        previous_released=released,
        required_height=0.16,
        axle_switch_x=0.05,
    )
    assert released.tolist() == [True]
    assert drive.tolist() == [True]

    axle, released, drive = update_clearance_drive_release(
        obstacle_x=torch.tensor([0.20]),
        wave_gate=torch.tensor([True]),
        wheel_heights=torch.tensor([[0.14, 0.14, 0.10, 0.10]]),
        previous_axle=axle,
        previous_released=released,
        required_height=0.16,
        axle_switch_x=0.05,
    )
    assert released.tolist() == [True]
    assert drive.tolist() == [True]

    axle, released, drive = update_clearance_drive_release(
        obstacle_x=torch.tensor([-0.10]),
        wave_gate=torch.tensor([True]),
        wheel_heights=torch.tensor([[0.10, 0.10, 0.17, 0.15]]),
        previous_axle=axle,
        previous_released=released,
        required_height=0.16,
        axle_switch_x=0.05,
    )
    assert axle.tolist() == [1]
    assert released.tolist() == [False]
    assert drive.tolist() == [False]

    axle, released, drive = update_clearance_drive_release(
        obstacle_x=torch.tensor([-0.10]),
        wave_gate=torch.tensor([False]),
        wheel_heights=torch.tensor([[0.10, 0.10, 0.10, 0.10]]),
        previous_axle=axle,
        previous_released=released,
        required_height=0.16,
        axle_switch_x=0.05,
    )
    assert axle.tolist() == [-1]
    assert released.tolist() == [False]
    assert drive.tolist() == [True]


def test_sequential_wheel_crossing_reference_releases_only_crossing_phases():
    import torch

    phase, steps, reference, drive, leg_active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.70]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.21, -0.21, -1.0, -1.0]]),
        wheel_heights=torch.tensor([[0.10, 0.10, 0.10, 0.10]]),
        previous_phase=torch.tensor([-1]),
        previous_phase_steps=torch.tensor([0]),
        required_height=0.16,
        past_bar_x=0.125,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
        front_start_x=-0.20,
    )
    assert phase.tolist() == [-1]
    assert drive.tolist() == [True]
    assert leg_active.tolist() == [False]
    torch.testing.assert_close(reference, torch.zeros_like(reference))

    phase, steps, reference, drive, leg_active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.60]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.20, -0.20, -1.0, -1.0]]),
        wheel_heights=torch.tensor([[0.10, 0.10, 0.10, 0.10]]),
        previous_phase=torch.tensor([-1]),
        previous_phase_steps=torch.tensor([0]),
        required_height=0.16,
        past_bar_x=0.125,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
    )
    assert phase.tolist() == [0]
    assert drive.tolist() == [False]
    assert leg_active.tolist() == [True]
    torch.testing.assert_close(reference[0, 1:3], torch.tensor([-0.15, -0.11]))
    torch.testing.assert_close(reference[0, 3:], torch.zeros(9))

    phase, steps, reference, drive, leg_active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.55]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.15, -0.20, -1.0, -1.0]]),
        wheel_heights=torch.tensor([[0.18, 0.10, 0.10, 0.10]]),
        previous_phase=torch.tensor([0]),
        previous_phase_steps=torch.tensor([10]),
        required_height=0.16,
        past_bar_x=0.125,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
    )
    assert phase.tolist() == [1]
    torch.testing.assert_close(reference[0, 1:3], torch.tensor([-1.50, -0.96]))
    assert drive.tolist() == [True]

    phase, steps, reference, drive, leg_active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.40]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.13, -0.20, -1.0, -1.0]]),
        wheel_heights=torch.tensor([[0.18, 0.10, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([50]),
        required_height=0.16,
        past_bar_x=0.125,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
    )
    assert phase.tolist() == [2]
    assert drive.tolist() == [False]
    # The crossed wheel stays clear while the second wheel ramps directly
    # toward the forward-clear hold pose.
    torch.testing.assert_close(reference[0, 1:3], torch.tensor([-1.50, -0.96]))
    torch.testing.assert_close(reference[0, 4:6], torch.tensor([0.10, -0.20]))

    phase, steps, reference, drive, leg_active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.38]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.15, -0.18, -1.0, -1.0]]),
        wheel_heights=torch.tensor([[0.10, 0.20, 0.10, 0.10]]),
        previous_phase=torch.tensor([2]),
        previous_phase_steps=torch.tensor([10]),
        required_height=0.16,
        past_bar_x=0.125,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
    )
    assert phase.tolist() == [3]
    assert drive.tolist() == [False]
    torch.testing.assert_close(reference[0, 1:3], torch.tensor([-1.50, -1.10]))
    torch.testing.assert_close(reference[0, 4:6], torch.tensor([0.94, -1.98]))

    phase, steps, reference, drive, leg_active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.25]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.full((1, 4), 0.20),
        wheel_heights=torch.full((1, 4), 0.20),
        previous_phase=torch.tensor([4]),
        previous_phase_steps=torch.tensor([10]),
        required_height=0.16,
        past_bar_x=0.125,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
    )
    assert phase.tolist() == [4]
    assert drive.tolist() == [False]
    assert leg_active.tolist() == [True]
    torch.testing.assert_close(reference[0, 1:3], torch.tensor([-0.75, -0.40]))
    torch.testing.assert_close(reference[0, 4:6], torch.tensor([-1.0, -0.5]))

    phase, steps, reference, drive, leg_active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.25]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.full((1, 4), 0.20),
        wheel_heights=torch.full((1, 4), 0.20),
        previous_phase=torch.tensor([4]),
        previous_phase_steps=torch.tensor([20]),
        required_height=0.16,
        past_bar_x=0.125,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
    )
    assert phase.tolist() == [5]
    assert drive.tolist() == [True]
    assert leg_active.tolist() == [False]
    torch.testing.assert_close(reference, torch.zeros_like(reference))

    phase, steps, reference, drive, leg_active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.0]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.zeros((1, 4)),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([5]),
        previous_phase_steps=torch.tensor([10]),
        required_height=0.16,
        past_bar_x=0.125,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
    )
    assert phase.tolist() == [6]
    assert drive.tolist() == [False]
    assert leg_active.tolist() == [True]
    torch.testing.assert_close(reference[0, 7:9], torch.tensor([0.15, 0.11]))


def test_sequential_leg_mask_releases_exactly_one_wheel_leg():
    import torch

    assert hasattr(m1_curriculum, "sequential_active_leg_mask")
    mask = m1_curriculum.sequential_active_leg_mask(
        torch.tensor([-1, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    )
    assert mask.shape == (11, 12)
    assert mask[0].sum().item() == 0
    assert mask[1, 0:3].all() and mask[1].sum().item() == 3
    assert mask[2, 0:3].all() and mask[2].sum().item() == 3
    assert mask[3, 3:6].all() and mask[3].sum().item() == 3
    assert mask[4, 3:6].all() and mask[4].sum().item() == 3
    assert mask[5].sum().item() == 0
    assert mask[6, 6:9].all() and mask[6].sum().item() == 3
    assert mask[7, 6:9].all() and mask[7].sum().item() == 3
    assert mask[8, 9:12].all() and mask[8].sum().item() == 3
    assert mask[9, 9:12].all() and mask[9].sum().item() == 3
    assert mask[10].sum().item() == 0


def test_sequential_residual_scale_unlocks_small_support_motion_only_during_wave():
    import torch

    scale = m1_curriculum.sequential_leg_residual_scale(
        torch.tensor([-1, 0, 4, 5, 6, 10, 11]),
        support_scale=0.75,
        crossing_scale=0.25,
        support_abduction_scale=0.15,
    )

    torch.testing.assert_close(scale[0], torch.zeros(12))
    torch.testing.assert_close(scale[1, :3], torch.tensor([0.0, 0.25, 0.25]))
    torch.testing.assert_close(
        scale[1, 3:], torch.tensor([0.15, 0.75, 0.75] * 3)
    )
    torch.testing.assert_close(scale[1, (3, 6, 9)], torch.full((3,), 0.15))
    torch.testing.assert_close(scale[2], torch.tensor([0.15, 0.75, 0.75] * 4))
    torch.testing.assert_close(scale[3], torch.zeros(12))
    torch.testing.assert_close(scale[4, 6:9], torch.tensor([0.0, 0.25, 0.25]))
    torch.testing.assert_close(
        scale[4, :6], torch.tensor([0.15, 0.75, 0.75] * 2)
    )
    torch.testing.assert_close(scale[4, (0, 3, 9)], torch.full((3,), 0.15))
    torch.testing.assert_close(scale[5], torch.tensor([0.15, 0.75, 0.75] * 4))
    torch.testing.assert_close(scale[6], torch.zeros(12))


def test_strict_sequential_success_requires_finished_phase_and_finish_line():
    import torch

    success = m1_curriculum.strict_sequential_crossing_success(
        phase=torch.tensor([11, 10, 11, -1]),
        root_x=torch.tensor([1.60, 1.60, 1.40, 2.00]),
        finish_x=1.50,
    )

    assert success.tolist() == [True, False, False, False]


def test_strict_crossing_success_accepts_completed_axle_pair_phase():
    import torch

    success = m1_curriculum.strict_sequential_crossing_success(
        phase=torch.tensor([5, 4, 5]),
        root_x=torch.tensor([1.60, 1.60, 1.40]),
        finish_x=1.50,
        required_phase=5,
    )

    assert success.tolist() == [True, False, False]


def test_axle_pair_crossing_releases_only_the_axle_at_the_bar():
    import torch

    assert hasattr(m1_curriculum, "update_axle_pair_crossing_reference")
    update = m1_curriculum.update_axle_pair_crossing_reference
    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.60]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.26, -0.26, -0.90, -0.90]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([-1]),
        previous_phase_steps=torch.tensor([0]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
        front_start_x=-0.25,
    )
    assert phase.tolist() == [-1]
    assert drive.tolist() == [True]
    assert not mask.any()
    torch.testing.assert_close(ref[0, 6:], torch.zeros(6))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.24, -0.24, -0.80, -0.80]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([-1]),
        previous_phase_steps=torch.tensor([0]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
        front_start_x=-0.25,
    )
    assert phase.tolist() == [0]
    assert mask[:, :6].all()
    assert not mask[:, 6:].any()
    assert drive.tolist() == [False]

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.24, -0.24, -0.80, -0.80]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([-1]),
        previous_phase_steps=torch.tensor([0]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=10,
        front_start_x=-0.25,
    )
    assert phase.tolist() == [0]
    torch.testing.assert_close(ref, torch.zeros_like(ref))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.24, -0.24, -0.80, -0.80]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([0]),
        previous_phase_steps=torch.tensor([20]),
        required_height=0.20,
        past_bar_x=0.25,
        ramp_steps=10,
        support_steps=20,
        front_start_x=-0.25,
    )
    assert phase.tolist() == [0]
    assert steps.tolist() == [21]
    torch.testing.assert_close(ref[0, :6], torch.tensor([0.0, -0.3, 0.0, 0.0, 0.2, -0.4]))
    torch.testing.assert_close(ref[0, 6:], torch.zeros(6))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.10, -0.10, -0.80, -0.80]]),
        wheel_heights=torch.tensor([[0.18, 0.18, 0.10, 0.10]]),
        previous_phase=torch.tensor([0]),
        previous_phase_steps=torch.tensor([50]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
    )
    assert phase.tolist() == [1]
    assert drive.tolist() == [True]
    torch.testing.assert_close(ref[0, :6], torch.tensor([0.0, -1.5, 0.0, 0.0, 1.0, -2.0]))
    torch.testing.assert_close(ref[0, 6:], torch.zeros(6))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.45]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.05, -0.05, -0.75, -0.75]]),
        wheel_heights=torch.tensor([[0.18, 0.18, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([24]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
    )
    assert phase.tolist() == [1]
    torch.testing.assert_close(ref[0, :6], torch.tensor([0.0, -1.5, 0.0, 0.0, -0.5, -1.5]))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.40]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.10, 0.10, -0.60, -0.60]]),
        wheel_heights=torch.tensor([[0.20, 0.20, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([100]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
    )
    assert phase.tolist() == [1]
    torch.testing.assert_close(ref[0, 1], torch.tensor(-1.50))
    torch.testing.assert_close(ref[0, 4], torch.tensor(-2.00))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.40]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.00, 0.00, -0.60, -0.60]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([50]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
        curriculum_swing_timeout_steps=20,
    )
    assert phase.tolist() == [2]

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.20]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.26, 0.27, -0.50, -0.50]]),
        wheel_heights=torch.tensor([[0.18, 0.18, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([50]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
        restore_steps=20,
    )
    assert phase.tolist() == [2]
    assert mask[:, :6].all()
    assert not mask[:, 6:].any()
    torch.testing.assert_close(ref[0, :6], torch.tensor([0.0, -1.5, 0.0, 0.0, -2.0, -1.0]))
    torch.testing.assert_close(ref[0, 6:], torch.zeros(6))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.20]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.26, 0.27, -0.50, -0.50]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([2]),
        previous_phase_steps=torch.tensor([9]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
        restore_steps=20,
    )
    assert phase.tolist() == [2]
    assert steps.tolist() == [10]
    assert mask[:, :6].all()
    assert not mask[:, 6:].any()
    torch.testing.assert_close(ref[0, :6], torch.tensor([0.0, -0.75, 0.0, 0.0, -1.0, -0.5]))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.20]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.26, 0.27, -0.50, -0.50]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([2]),
        previous_phase_steps=torch.tensor([19]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
        restore_steps=20,
    )
    assert phase.tolist() == [2]
    assert steps.tolist() == [20]
    assert not mask.any()
    torch.testing.assert_close(ref, torch.zeros_like(ref))

    phase, steps, ref, drive, mask = update(
        obstacle_x=torch.tensor([0.04]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.40, 0.40, -0.27, -0.27]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([2]),
        previous_phase_steps=torch.tensor([20]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=50,
        restore_steps=20,
    )
    assert phase.tolist() == [3]
    assert not mask[:, :6].any()
    assert mask[:, 6:].all()
    assert drive.tolist() == [False]
    torch.testing.assert_close(ref[0, :6], torch.zeros(6))


def test_axle_pair_wave_uses_mirrored_targets_and_restores_rear_pair():
    import torch

    update = m1_curriculum.update_axle_pair_crossing_reference
    common = dict(
        obstacle_x=torch.tensor([0.20]),
        wave_gate=torch.tensor([True]),
        required_height=0.16,
        past_bar_x=0.25,
        ramp_steps=10,
        swing_steps=50,
        restore_steps=20,
        support_steps=0,
    )

    phase, _, ref, drive, _ = update(
        **common,
        wheel_x_from_obstacle=torch.tensor([[-0.10, -0.10, -0.70, -0.70]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([0]),
        previous_phase_steps=torch.tensor([20]),
    )
    assert phase.tolist() == [0]
    assert drive.tolist() == [False]
    torch.testing.assert_close(ref[0, :6], torch.tensor([0.0, -1.5, 0.0, 0.0, 1.0, -2.0]))
    torch.testing.assert_close(ref[0, 6:], torch.zeros(6))

    phase, _, ref, drive, _ = update(
        **common,
        wheel_x_from_obstacle=torch.tensor([[0.10, 0.10, -0.60, -0.60]]),
        wheel_heights=torch.tensor([[0.20, 0.20, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([24]),
    )
    assert phase.tolist() == [1]
    assert drive.tolist() == [True]
    torch.testing.assert_close(ref[0, :6], torch.tensor([0.0, -1.5, 0.0, 0.0, -0.5, -1.5]))
    torch.testing.assert_close(ref[0, 6:], torch.zeros(6))

    phase, _, ref, drive, _ = update(
        **common,
        wheel_x_from_obstacle=torch.tensor([[0.40, 0.40, -0.10, -0.10]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([3]),
        previous_phase_steps=torch.tensor([20]),
    )
    assert phase.tolist() == [3]
    assert drive.tolist() == [False]
    torch.testing.assert_close(ref[0, :6], torch.zeros(6))
    torch.testing.assert_close(ref[0, 6:], torch.tensor([0.0, 1.5, 0.0, 0.0, -1.0, 2.0]))

    phase, _, ref, drive, _ = update(
        **common,
        wheel_x_from_obstacle=torch.tensor([[0.50, 0.50, 0.10, 0.10]]),
        wheel_heights=torch.tensor([[0.10, 0.10, 0.20, 0.20]]),
        previous_phase=torch.tensor([4]),
        previous_phase_steps=torch.tensor([24]),
    )
    assert phase.tolist() == [4]
    assert drive.tolist() == [True]
    torch.testing.assert_close(ref[0, 6:], torch.tensor([0.0, 1.5, 0.0, 0.0, 0.5, 1.5]))

    phase, steps, ref, drive, mask = update(
        **common,
        wheel_x_from_obstacle=torch.tensor([[0.60, 0.60, 0.30, 0.30]]),
        wheel_heights=torch.tensor([[0.10, 0.10, 0.18, 0.18]]),
        previous_phase=torch.tensor([4]),
        previous_phase_steps=torch.tensor([50]),
    )
    assert phase.tolist() == [5]
    assert steps.tolist() == [0]
    assert drive.tolist() == [False]
    assert not mask[:, :6].any()
    assert mask[:, 6:].all()
    torch.testing.assert_close(ref[0, 6:], torch.tensor([0.0, 1.5, 0.0, 0.0, 2.0, 1.0]))

    phase, _, ref, drive, mask = update(
        **common,
        wheel_x_from_obstacle=torch.tensor([[0.60, 0.60, 0.30, 0.30]]),
        wheel_heights=torch.tensor([[0.10, 0.10, 0.18, 0.18]]),
        previous_phase=torch.tensor([5]),
        previous_phase_steps=torch.tensor([19]),
    )
    assert phase.tolist() == [6]
    assert drive.tolist() == [True]
    assert not mask.any()
    torch.testing.assert_close(ref, torch.zeros_like(ref))


def test_axle_pair_progress_requires_both_wheels_to_lift_and_pass():
    import torch

    assert hasattr(m1_curriculum, "axle_pair_crossing_progress_score")
    score = m1_curriculum.axle_pair_crossing_progress_score(
        phase=torch.tensor([1, 1, 4, -1]),
        wheel_x_from_obstacle=torch.tensor(
            [
                [0.10, -0.20, -0.60, -0.60],
                [0.10, 0.10, -0.60, -0.60],
                [0.40, 0.40, 0.30, 0.30],
                [0.00, 0.00, 0.00, 0.00],
            ]
        ),
        wheel_heights=torch.tensor(
            [
                [0.18, 0.10, 0.10, 0.10],
                [0.18, 0.18, 0.10, 0.10],
                [0.10, 0.10, 0.18, 0.18],
                [0.10, 0.10, 0.10, 0.10],
            ]
        ),
        baseline_height=0.10,
        required_height=0.16,
        swing_start_x=-0.20,
        past_bar_x=0.25,
    )

    assert score[0] < score[1]
    assert score[1] < score[2]
    assert score[3] == 0.0


def test_progress_potential_rewards_only_new_progress_and_resets_cleanly():
    import torch

    assert hasattr(m1_curriculum, "progress_potential_delta")
    delta, next_potential = m1_curriculum.progress_potential_delta(
        current=torch.tensor([1.5, 1.0, 0.0]),
        previous=torch.tensor([1.0, 1.5, 4.0]),
        reset=torch.tensor([False, False, True]),
    )

    torch.testing.assert_close(delta, torch.tensor([0.5, -0.5, 0.0]))
    torch.testing.assert_close(next_potential, torch.tensor([1.5, 1.0, 0.0]))


def test_sequential_wheel_progress_rewards_clearance_and_forward_swing():
    import torch

    score = sequential_wheel_crossing_progress_score(
        phase=torch.tensor([1, 1, -1]),
        wheel_x_from_obstacle=torch.tensor(
            [[-0.20, 0.0, 0.0, 0.0], [0.10, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        wheel_heights=torch.tensor(
            [[0.18, 0.10, 0.10, 0.10], [0.18, 0.10, 0.10, 0.10], [0.10, 0.10, 0.10, 0.10]]
        ),
        baseline_height=0.10,
        required_height=0.16,
        swing_start_x=-0.20,
        past_bar_x=0.125,
    )

    assert score[1] > score[0]
    assert score[0] > score[2]


def test_right_track_sequence_skips_left_wheels_and_restores_between_axles():
    import torch

    common = dict(
        obstacle_x=torch.tensor([0.0]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.20, 0.20, 0.20, 0.20]]),
        wheel_heights=torch.tensor([[0.20, 0.10, 0.20, 0.10]]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        right_track_only=True,
    )

    phase, _, _, _, _ = update_sequential_wheel_crossing_reference(
        **common, previous_phase=torch.tensor([1]), previous_phase_steps=torch.tensor([10])
    )
    assert phase.tolist() == [4]

    phase, _, _, _, _ = update_sequential_wheel_crossing_reference(
        **common, previous_phase=torch.tensor([4]), previous_phase_steps=torch.tensor([20])
    )
    assert phase.tolist() == [5]

    phase, _, _, _, _ = update_sequential_wheel_crossing_reference(
        **common, previous_phase=torch.tensor([5]), previous_phase_steps=torch.tensor([10])
    )
    assert phase.tolist() == [6]

    phase, _, _, _, _ = update_sequential_wheel_crossing_reference(
        **{**common, "obstacle_x": torch.tensor([-0.60])},
        previous_phase=torch.tensor([7]),
        previous_phase_steps=torch.tensor([10]),
    )
    assert phase.tolist() == [10]

    phase, _, _, _, _ = update_sequential_wheel_crossing_reference(
        **common, previous_phase=torch.tensor([10]), previous_phase_steps=torch.tensor([20])
    )
    assert phase.tolist() == [11]


def test_right_track_restore_phases_do_not_move_left_legs():
    import torch

    common = dict(
        obstacle_x=torch.tensor([0.0]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.20, 0.20, 0.20, 0.20]]),
        wheel_heights=torch.tensor([[0.20, 0.10, 0.20, 0.10]]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        right_track_only=True,
    )

    front_phase, _, front_reference, _, _ = update_sequential_wheel_crossing_reference(
        **common,
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([10]),
    )
    rear_phase, _, rear_reference, _, _ = update_sequential_wheel_crossing_reference(
        **{**common, "obstacle_x": torch.tensor([-0.60])},
        previous_phase=torch.tensor([7]),
        previous_phase_steps=torch.tensor([10]),
    )

    assert front_phase.tolist() == [4]
    assert rear_phase.tolist() == [10]
    assert front_reference[0, 1:3].abs().sum() > 0.0
    assert rear_reference[0, 7:9].abs().sum() > 0.0
    torch.testing.assert_close(front_reference[0, 3:6], torch.zeros(3))
    torch.testing.assert_close(rear_reference[0, 9:12], torch.zeros(3))


def test_right_track_prebalances_supports_before_lifting_active_wheel():
    import torch

    common = dict(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.20, -0.20, -0.80, -0.80]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([0]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        support_extension=-0.20,
        opposite_abduction=-0.10,
        balance_steps=10,
        right_track_only=True,
    )

    _, _, prebalance, _, _ = update_sequential_wheel_crossing_reference(
        **common, previous_phase_steps=torch.tensor([9])
    )
    _, _, first_lift, _, _ = update_sequential_wheel_crossing_reference(
        **common, previous_phase_steps=torch.tensor([10])
    )

    torch.testing.assert_close(prebalance[0, 0:3], torch.zeros(3))
    torch.testing.assert_close(prebalance[0, 3], torch.tensor(-0.10))
    torch.testing.assert_close(prebalance[0, 5], torch.tensor(-0.20))
    torch.testing.assert_close(prebalance[0, 8], torch.tensor(0.20))
    torch.testing.assert_close(prebalance[0, 9], torch.tensor(-0.10))
    torch.testing.assert_close(prebalance[0, 11], torch.tensor(0.20))
    torch.testing.assert_close(first_lift[0, 1], torch.tensor(-0.15))


def test_right_track_knee_feedback_retracts_low_front_and_rear_wheels():
    import torch

    common = dict(
        obstacle_x=torch.tensor([0.40, 0.40]),
        wave_gate=torch.tensor([True, True]),
        wheel_x_from_obstacle=torch.tensor(
            [[-0.10, -0.20, -0.60, -0.70], [-0.10, -0.20, -0.60, -0.70]]
        ),
        previous_phase_steps=torch.tensor([20, 20]),
        required_height=0.25,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        clearance_target_height=0.20,
        right_track_only=True,
    )

    _, _, front_reference, _, _ = update_sequential_wheel_crossing_reference(
        **common,
        wheel_heights=torch.tensor(
            [[0.10, 0.10, 0.10, 0.10], [0.19, 0.10, 0.10, 0.10]]
        ),
        previous_phase=torch.tensor([0, 0]),
    )
    _, _, rear_reference, _, _ = update_sequential_wheel_crossing_reference(
        **common,
        wheel_heights=torch.tensor(
            [[0.10, 0.10, 0.10, 0.10], [0.10, 0.10, 0.19, 0.10]]
        ),
        previous_phase=torch.tensor([6, 6]),
    )

    assert front_reference[0, 2] < front_reference[1, 2] < 0.0
    assert rear_reference[0, 8] > rear_reference[1, 8] > 0.0


def test_right_track_uses_configured_active_leg_jointspace_actions():
    import torch

    common = dict(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.10, -0.20, -0.60, -0.70]]),
        wheel_heights=torch.full((1, 4), 0.20),
        previous_phase_steps=torch.tensor([5]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        front_hip_action=-0.30,
        front_knee_action=-0.60,
        rear_hip_action=0.30,
        rear_knee_action=0.60,
        right_track_only=True,
    )

    _, _, front_reference, _, _ = update_sequential_wheel_crossing_reference(
        **common, previous_phase=torch.tensor([1])
    )
    _, _, rear_reference, _, _ = update_sequential_wheel_crossing_reference(
        **common, previous_phase=torch.tensor([7])
    )

    torch.testing.assert_close(front_reference[0, 1:3], torch.tensor([-0.30, -0.60]))
    torch.testing.assert_close(rear_reference[0, 7:9], torch.tensor([0.30, 0.60]))


def test_right_track_latches_wave_after_trigger_when_semantic_gate_flickers():
    import torch

    phase, steps, reference, drive, active = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.60]),
        wave_gate=torch.tensor([False]),
        wheel_x_from_obstacle=torch.tensor([[-0.20, -0.30, -0.80, -0.90]]),
        wheel_heights=torch.full((1, 4), 0.10),
        previous_phase=torch.tensor([0]),
        previous_phase_steps=torch.tensor([6]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        right_track_only=True,
    )

    assert phase.tolist() == [0]
    assert steps.tolist() == [7]
    assert drive.tolist() == [False]
    assert active.tolist() == [True]
    assert reference[0, 1:3].abs().sum() > 0.0


def test_right_track_swing_uses_latched_clearance_when_wheel_is_past_bar():
    import torch

    phase, _, _, _, _ = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.20]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.04, -0.10, -0.60, -0.60]]),
        wheel_heights=torch.tensor([[0.12, 0.10, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([3]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        right_track_only=True,
    )

    assert phase.tolist() == [4]


def test_right_track_holds_planted_front_wheel_until_body_catches_up():
    import torch

    phase, _, _, drive_allowed, _ = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[0.04, -0.10, -0.60, -0.60]]),
        wheel_heights=torch.tensor([[0.18, 0.10, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([3]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        front_restore_obstacle_x=0.30,
        right_track_only=True,
    )

    assert phase.tolist() == [1]
    assert drive_allowed.tolist() == [True]


def test_right_track_stops_drive_until_active_wheel_is_swung_past_bar():
    import torch

    phase, _, _, drive_allowed, _ = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.05, -0.10, -0.60, -0.60]]),
        wheel_heights=torch.tensor([[0.18, 0.10, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([3]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        right_track_only=True,
    )

    assert phase.tolist() == [1]
    assert drive_allowed.tolist() == [False]


def test_right_track_can_keep_equal_wheel_drive_during_wave():
    import torch

    phase, _, _, drive_allowed, _ = update_sequential_wheel_crossing_reference(
        obstacle_x=torch.tensor([0.50]),
        wave_gate=torch.tensor([True]),
        wheel_x_from_obstacle=torch.tensor([[-0.05, -0.10, -0.60, -0.60]]),
        wheel_heights=torch.tensor([[0.18, 0.10, 0.10, 0.10]]),
        previous_phase=torch.tensor([1]),
        previous_phase_steps=torch.tensor([3]),
        required_height=0.16,
        past_bar_x=0.03,
        swing_steps=50,
        min_lift_steps=5,
        ramp_steps=10,
        restore_steps=20,
        right_track_only=True,
        keep_drive_during_wave=True,
    )

    assert phase.tolist() == [1]
    assert drive_allowed.tolist() == [True]


def test_task_space_wheel_actions_close_lift_and_swing_errors_for_selected_wheel():
    import torch

    jacobians = torch.eye(2).reshape(1, 1, 2, 2).repeat(2, 4, 1, 1)
    actions = build_task_space_wheel_joint_actions(
        phase=torch.tensor([0, 3]),
        wheel_x_w=torch.zeros((2, 4)),
        wheel_heights=torch.full((2, 4), 0.10),
        wheel_x_obstacle_w=torch.tensor([0.20, 0.20]),
        leg_joint_pos=torch.zeros((2, 4, 2)),
        default_leg_joint_pos=torch.zeros((2, 4, 2)),
        wheel_xz_jacobians=jacobians,
        lift_height=0.20,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
    )

    torch.testing.assert_close(actions[0, 1:3], torch.tensor([0.0, 0.10]))
    torch.testing.assert_close(actions[0, (0, 3, 4, 5, 6, 7, 8, 9, 10, 11)], torch.zeros(10))
    torch.testing.assert_close(actions[1, 4:6], torch.tensor([0.30, 0.10]))
    torch.testing.assert_close(actions[1, (0, 1, 2, 3, 6, 7, 8, 9, 10, 11)], torch.zeros(10))


def test_task_space_wheel_actions_remain_finite_near_singular_jacobian():
    import torch

    actions = build_task_space_wheel_joint_actions(
        phase=torch.tensor([1]),
        wheel_x_w=torch.zeros((1, 4)),
        wheel_heights=torch.full((1, 4), 0.10),
        wheel_x_obstacle_w=torch.tensor([0.20]),
        leg_joint_pos=torch.zeros((1, 4, 2)),
        default_leg_joint_pos=torch.zeros((1, 4, 2)),
        wheel_xz_jacobians=torch.zeros((1, 4, 2, 2)),
        lift_height=0.20,
        past_bar_x=0.10,
        action_scale=0.8,
        damping=0.05,
        max_joint_step=0.15,
    )

    assert torch.isfinite(actions).all()
    torch.testing.assert_close(actions, torch.zeros_like(actions))


def test_task_space_wheel_actions_hold_three_support_wheels_during_wave():
    import torch

    jacobians = torch.eye(2).reshape(1, 1, 2, 2).repeat(1, 4, 1, 1)
    actions = build_task_space_wheel_joint_actions(
        phase=torch.tensor([0]),
        wheel_x_w=torch.tensor([[0.0, 0.10, 0.0, 0.0]]),
        wheel_heights=torch.full((1, 4), 0.10),
        wheel_x_obstacle_w=torch.tensor([0.20]),
        leg_joint_pos=torch.zeros((1, 4, 2)),
        default_leg_joint_pos=torch.zeros((1, 4, 2)),
        wheel_xz_jacobians=jacobians,
        lift_height=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        nominal_wheel_x=torch.zeros((1, 4)),
        nominal_wheel_heights=torch.full((1, 4), 0.10),
        hold_supports=True,
    )

    torch.testing.assert_close(actions[0, 1:3], torch.tensor([0.0, 0.10]))
    torch.testing.assert_close(actions[0, 4:6], torch.tensor([-0.10, 0.0]))
    torch.testing.assert_close(actions[0, 7:9], torch.zeros(2))
    torch.testing.assert_close(actions[0, 10:12], torch.zeros(2))


def test_task_space_wheel_actions_can_carry_lifted_wheel_with_body_motion():
    import torch

    actions = build_task_space_wheel_joint_actions(
        phase=torch.tensor([1]),
        wheel_x_w=torch.zeros((1, 4)),
        wheel_heights=torch.full((1, 4), 0.10),
        wheel_x_obstacle_w=torch.tensor([0.40]),
        leg_joint_pos=torch.zeros((1, 4, 2)),
        default_leg_joint_pos=torch.zeros((1, 4, 2)),
        wheel_xz_jacobians=torch.eye(2).reshape(1, 1, 2, 2).repeat(1, 4, 1, 1),
        lift_height=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        nominal_wheel_x=torch.zeros((1, 4)),
        nominal_wheel_heights=torch.full((1, 4), 0.10),
        hold_supports=True,
        swing_with_body=True,
    )

    torch.testing.assert_close(actions[0, 1:3], torch.tensor([0.0, 0.10]))


def test_m1_leg_joint_ids_are_grouped_by_name_not_physx_storage_order():
    joint_names = [
        "FAR_ABAD_JOINT", "FBL_ABAD_JOINT", "RAR_ABAD_JOINT", "RBL_ABAD_JOINT",
        "FAR_HIP_JOINT", "FBL_HIP_JOINT", "RAR_HIP_JOINT", "RBL_HIP_JOINT",
        "FAR_KNEE_JOINT", "FBL_KNEE_JOINT", "RAR_KNEE_JOINT", "RBL_KNEE_JOINT",
    ]

    assert resolve_m1_leg_joint_ids_by_wheel(joint_names) == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
    ]


def test_single_wheel_task_space_actions_lock_all_support_legs():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0, 2]),
        wheel_pos_b=torch.zeros((2, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((2, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(2),
        leg_joint_pos=torch.zeros((2, 4, 3)),
        default_leg_joint_pos=torch.zeros((2, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 4, 1, 1),
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.05,
        swing_with_body=True,
    )

    torch.testing.assert_close(actions[0, :3], torch.tensor([0.0, -0.05, 0.10]))
    torch.testing.assert_close(actions[0, 3:], torch.zeros(9))
    torch.testing.assert_close(actions[1, 3:6], torch.tensor([0.0, 0.05, 0.10]))
    torch.testing.assert_close(actions[1, :3], torch.zeros(3))
    torch.testing.assert_close(actions[1, 6:], torch.zeros(6))


def test_single_wheel_swing_can_solve_xz_without_abduction_coupling():
    import torch

    jacobian = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    jacobians = jacobian.reshape(1, 1, 3, 3).repeat(1, 4, 1, 1)
    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([1]),
        wheel_pos_b=torch.zeros((1, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((1, 4, 3)),
        wheel_x_obstacle_b=torch.tensor([0.20]),
        leg_joint_pos=torch.zeros((1, 4, 3)),
        default_leg_joint_pos=torch.zeros((1, 4, 3)),
        wheel_xyz_jacobians=jacobians,
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.05,
        swing_with_body=False,
        active_swing_xz_only=True,
    ).reshape(1, 4, 3)

    torch.testing.assert_close(actions[0, 0], torch.tensor([0.0, 0.30, 0.10]))


def test_task_space_can_use_lower_rear_lift_than_front_lift():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0, 6]),
        wheel_pos_b=torch.zeros((2, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((2, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(2),
        leg_joint_pos=torch.zeros((2, 4, 3)),
        default_leg_joint_pos=torch.zeros((2, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 4, 1, 1),
        lift_delta=0.22,
        rear_lift_delta=0.20,
        past_bar_x=0.15,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.0,
        swing_with_body=True,
    ).reshape(2, 4, 3)

    torch.testing.assert_close(actions[0, 0], torch.tensor([0.0, 0.0, 0.22]))
    torch.testing.assert_close(actions[1, 2], torch.tensor([0.0, 0.0, 0.20]))


def test_task_space_restore_lowers_only_rear_right_wheel_in_place():
    import torch

    jacobians = torch.eye(3).reshape(1, 1, 3, 3).repeat(4, 4, 1, 1)
    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([4, 10, 10, 10]),
        phase_steps=torch.tensor([0, 0, 10, 20]),
        wheel_pos_b=torch.zeros((4, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((4, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(4),
        leg_joint_pos=torch.zeros((4, 4, 3)),
        default_leg_joint_pos=torch.zeros((4, 4, 3)),
        wheel_xyz_jacobians=jacobians,
        lift_delta=0.20,
        past_bar_x=0.15,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.0,
        swing_with_body=True,
        rear_restore_steps=20,
        rear_restore_forward_offset=0.12,
        stabilize_supports=True,
    ).reshape(4, 4, 3)

    torch.testing.assert_close(actions[0], torch.zeros((4, 3)))
    torch.testing.assert_close(actions[1, 2], torch.tensor([0.12, 0.0, 0.20]))
    torch.testing.assert_close(actions[2, 2], torch.tensor([0.12, 0.0, 0.10]))
    torch.testing.assert_close(actions[3, 2], torch.tensor([0.12, 0.0, 0.0]))


def test_required_axle_lift_ignores_wheels_not_selected_to_cross():
    import torch

    assert hasattr(m1_curriculum, "required_axle_lift_passed")
    front_passed, rear_passed = m1_curriculum.required_axle_lift_passed(
        wheel_height_max=torch.tensor([0.17, 0.10, 0.18, 0.10]),
        wheel_clearance_required=torch.tensor(
            [[True, False, True, False], [True, False, True, False]]
        ),
        min_front_height=0.13,
        min_rear_height=0.14,
    )

    assert front_passed
    assert rear_passed


def test_hybrid_reference_uses_jointspace_only_for_the_active_wheel_leg():
    import torch

    assert hasattr(m1_curriculum, "merge_task_space_support_with_jointspace_active")
    task_space = torch.full((3, 12), 2.0)
    joint_space = torch.arange(36, dtype=torch.float32).reshape(3, 12)
    merged = m1_curriculum.merge_task_space_support_with_jointspace_active(
        task_space_reference=task_space,
        joint_space_reference=joint_space,
        phase=torch.tensor([0, 6, -1]),
    )

    torch.testing.assert_close(merged[0, :3], joint_space[0, :3])
    torch.testing.assert_close(merged[0, 3:], task_space[0, 3:])
    torch.testing.assert_close(merged[1, 6:9], joint_space[1, 6:9])
    torch.testing.assert_close(merged[1, :6], task_space[1, :6])
    torch.testing.assert_close(merged[1, 9:], task_space[1, 9:])
    torch.testing.assert_close(merged[2], task_space[2])


def test_hybrid_reference_uses_taskspace_xz_but_locks_active_abduction_during_swing():
    import torch

    task_space = torch.full((2, 12), 2.0)
    joint_space = torch.full((2, 12), 3.0)
    merged = m1_curriculum.merge_task_space_support_with_jointspace_active(
        task_space_reference=task_space,
        joint_space_reference=joint_space,
        phase=torch.tensor([1, 7]),
    )

    torch.testing.assert_close(merged[0, 0:1], joint_space[0, 0:1])
    torch.testing.assert_close(merged[0, 1:3], task_space[0, 1:3])
    torch.testing.assert_close(merged[0, 3:], task_space[0, 3:])
    torch.testing.assert_close(merged[1, 6:7], joint_space[1, 6:7])
    torch.testing.assert_close(merged[1, 7:9], task_space[1, 7:9])
    torch.testing.assert_close(merged[1, :6], task_space[1, :6])
    torch.testing.assert_close(merged[1, 9:], task_space[1, 9:])


def test_single_wheel_task_space_waits_before_lifting_without_moving_supports():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0, 0]),
        phase_steps=torch.tensor([10, 30]),
        wheel_pos_b=torch.zeros((2, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((2, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(2),
        leg_joint_pos=torch.zeros((2, 4, 3)),
        default_leg_joint_pos=torch.zeros((2, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 4, 1, 1),
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.05,
        swing_with_body=True,
        balance_steps=20,
        lift_ramp_steps=10,
    ).reshape(2, 4, 3)

    torch.testing.assert_close(actions[0, 0], torch.tensor([0.0, -0.025, 0.0]))
    torch.testing.assert_close(actions[0, 1:], torch.zeros((3, 3)))
    torch.testing.assert_close(actions[1, 0, 1], torch.tensor(-0.05))
    torch.testing.assert_close(actions[1, 1:, 1], torch.zeros(3))
    torch.testing.assert_close(actions[1, 0, 2], torch.tensor(0.10))
    torch.testing.assert_close(actions[1, 1:, 2], torch.zeros(3))


def test_single_wheel_task_space_prebalances_body_with_all_grounded_legs():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0]),
        phase_steps=torch.tensor([10]),
        wheel_pos_b=torch.zeros((1, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((1, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(1),
        leg_joint_pos=torch.zeros((1, 4, 3)),
        default_leg_joint_pos=torch.zeros((1, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1),
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.05,
        swing_with_body=True,
        balance_steps=20,
        lift_ramp_steps=10,
        stabilize_supports=True,
        balance_supports=True,
    ).reshape(1, 4, 3)

    torch.testing.assert_close(actions[0, :, 1], torch.full((4,), -0.025))
    torch.testing.assert_close(actions[0, :, 2], torch.zeros(4))


def test_single_wheel_task_space_prebalances_away_from_front_and_rear_right_wheels():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0, 6]),
        phase_steps=torch.tensor([10, 10]),
        wheel_pos_b=torch.zeros((2, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((2, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(2),
        leg_joint_pos=torch.zeros((2, 4, 3)),
        default_leg_joint_pos=torch.zeros((2, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 4, 1, 1),
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.04,
        longitudinal_body_shift=0.06,
        swing_with_body=True,
        balance_steps=20,
        lift_ramp_steps=10,
        balance_supports=True,
    ).reshape(2, 4, 3)

    torch.testing.assert_close(actions[0, :, 0], torch.full((4,), 0.03))
    torch.testing.assert_close(actions[1, :, 0], torch.full((4,), -0.03))
    torch.testing.assert_close(actions[:, :, 1], torch.full((2, 4), -0.02))
    torch.testing.assert_close(actions[:, :, 2], torch.zeros((2, 4)))


def test_single_wheel_task_space_uses_only_support_abad_for_lateral_recovery():
    import torch

    jacobian = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1)
    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0]),
        wheel_pos_b=torch.zeros((1, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((1, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(1),
        leg_joint_pos=torch.zeros((1, 4, 3)),
        default_leg_joint_pos=torch.zeros((1, 4, 3)),
        wheel_xyz_jacobians=jacobian,
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.0,
        swing_with_body=True,
        base_lateral_offset=torch.tensor([-0.10]),
        lateral_recovery_gain=0.5,
        lateral_recovery_max=0.08,
    ).reshape(1, 4, 3)

    torch.testing.assert_close(actions[0, 0], torch.tensor([0.0, 0.0, 0.10]))
    torch.testing.assert_close(actions[0, 1:, 0], torch.full((3,), -0.05))
    torch.testing.assert_close(actions[0, 1:, 1:], torch.zeros((3, 2)))


def test_single_wheel_task_space_can_stabilize_supports_at_nominal_pose():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0]),
        wheel_pos_b=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.02, -0.03, -0.04], [0.02, -0.03, -0.04], [0.02, -0.03, -0.04]]]
        ),
        nominal_wheel_pos_b=torch.zeros((1, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(1),
        leg_joint_pos=torch.zeros((1, 4, 3)),
        default_leg_joint_pos=torch.zeros((1, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1),
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.0,
        swing_with_body=True,
        stabilize_supports=True,
    ).reshape(1, 4, 3)

    torch.testing.assert_close(actions[0, 0], torch.tensor([0.0, 0.0, 0.10]))
    torch.testing.assert_close(
        actions[0, 1:], torch.tensor([[-0.02, 0.03, 0.04]]).repeat(3, 1)
    )


def test_single_wheel_task_space_ramps_swing_target_without_dropping_lift():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([1]),
        phase_steps=torch.tensor([4]),
        wheel_pos_b=torch.zeros((1, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((1, 4, 3)),
        wheel_x_obstacle_b=torch.tensor([0.40]),
        leg_joint_pos=torch.zeros((1, 4, 3)),
        default_leg_joint_pos=torch.zeros((1, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1),
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.0,
        swing_with_body=False,
        swing_ramp_steps=10,
    ).reshape(1, 4, 3)

    torch.testing.assert_close(actions[0, 0], torch.tensor([0.25, 0.0, 0.10]))
    torch.testing.assert_close(actions[0, 1:], torch.zeros((3, 3)))


def test_stabilized_task_space_actions_lift_both_wheels_of_active_axle():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0, 3]),
        wheel_pos_b=torch.zeros((2, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((2, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(2),
        leg_joint_pos=torch.zeros((2, 4, 3)),
        default_leg_joint_pos=torch.zeros((2, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 4, 1, 1),
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.05,
        swing_with_body=True,
        axle_pair_mode=True,
        pair_support_extension=0.08,
    ).reshape(2, 4, 3)

    torch.testing.assert_close(actions[0, :, 2], torch.tensor([0.10, 0.10, -0.08, -0.08]))
    torch.testing.assert_close(actions[1, :, 2], torch.tensor([-0.08, -0.08, 0.10, 0.10]))
    torch.testing.assert_close(actions[..., 1], torch.zeros((2, 4)))


def test_axle_pair_task_space_prebalances_com_before_lifting():
    import torch

    actions = build_stabilized_task_space_wheel_actions(
        phase=torch.tensor([0, 0, 3, 3]),
        phase_steps=torch.tensor([10, 30, 10, 30]),
        wheel_pos_b=torch.zeros((4, 4, 3)),
        nominal_wheel_pos_b=torch.zeros((4, 4, 3)),
        wheel_x_obstacle_b=torch.zeros(4),
        leg_joint_pos=torch.zeros((4, 4, 3)),
        default_leg_joint_pos=torch.zeros((4, 4, 3)),
        wheel_xyz_jacobians=torch.eye(3).reshape(1, 1, 3, 3).repeat(4, 4, 1, 1),
        lift_delta=0.10,
        past_bar_x=0.10,
        action_scale=1.0,
        damping=0.0,
        max_joint_step=1.0,
        lateral_body_shift=0.0,
        swing_with_body=True,
        axle_pair_mode=True,
        pair_support_extension=0.08,
        pair_body_shift_x=0.15,
        balanced_wheel_x_b=torch.tensor(
            [[0.0] * 4, [0.06] * 4, [0.0] * 4, [-0.06] * 4]
        ),
        balance_steps=20,
        lift_ramp_steps=10,
    ).reshape(4, 4, 3)

    torch.testing.assert_close(actions[0, :, 0], torch.full((4,), 0.075))
    torch.testing.assert_close(actions[0, :2, 2], torch.zeros(2))
    torch.testing.assert_close(actions[1, :, 0], torch.full((4,), 0.06))
    torch.testing.assert_close(actions[1, :2, 2], torch.full((2,), 0.10))
    torch.testing.assert_close(actions[2, :, 0], torch.full((4,), -0.075))
    torch.testing.assert_close(actions[2, 2:, 2], torch.zeros(2))
    torch.testing.assert_close(actions[3, :, 0], torch.full((4,), -0.06))
    torch.testing.assert_close(actions[3, 2:, 2], torch.full((2,), 0.10))

def test_crossbar_collision_mask_rejects_low_forced_wheel_in_swept_region():
    import torch

    collision = wheel_crossbar_collision_mask(
        wheel_pos_local=torch.tensor(
            [[[0.525, 0.0, 0.096], [0.65, 0.0, 0.170], [0.20, 0.0, 0.096], [0.20, 0.0, 0.096]]]
        ),
        wheel_contact_force=torch.tensor([[80.0, 0.0, 80.0, 80.0]]),
        obstacle_center_x=0.65,
        obstacle_size_x=0.06,
        obstacle_height=0.06,
        wheel_radius=0.095,
        clearance_margin=0.005,
        contact_force_threshold=1.0,
    )

    assert collision.tolist() == [[True, False, False, False]]


def test_crossbar_collision_mask_ignores_loaded_wheel_outside_narrow_track():
    import torch

    collision = wheel_crossbar_collision_mask(
        wheel_pos_local=torch.tensor(
            [[[0.65, -0.20, 0.096], [0.65, 0.20, 0.096]]]
        ),
        wheel_contact_force=torch.tensor([[80.0, 80.0]]),
        obstacle_center_x=0.65,
        obstacle_size_x=0.06,
        obstacle_center_y=-0.20,
        obstacle_size_y=0.16,
        obstacle_height=0.06,
        wheel_radius=0.095,
        clearance_margin=0.005,
        contact_force_threshold=1.0,
    )

    assert collision.tolist() == [[True, False]]


def test_spatial_axle_wheel_targets_switch_load_assist_between_axles():
    import torch

    targets = build_spatial_axle_wheel_targets(
        obstacle_x=torch.tensor([0.40, -0.20, 0.40]),
        active=torch.tensor([True, True, False]),
        base_action=1.30,
        assist_action=0.40,
    )

    torch.testing.assert_close(targets[0], torch.tensor([1.30, 1.30, 1.70, 1.70]))
    torch.testing.assert_close(targets[1], torch.tensor([1.70, 1.70, 1.30, 1.30]))
    torch.testing.assert_close(targets[2], torch.full((4,), 1.30))


def test_temporal_axle_wheel_targets_follow_front_then_rear_wave_pulse():
    import torch

    targets = build_temporal_axle_wheel_targets(
        episode_time_s=torch.tensor([0.5, 1.5, 0.5]),
        active=torch.tensor([True, True, False]),
        frequency=0.5,
        base_action=1.30,
        assist_action=0.40,
    )

    torch.testing.assert_close(targets[0], torch.tensor([1.30, 1.30, 1.70, 1.70]))
    torch.testing.assert_close(targets[1], torch.tensor([1.70, 1.70, 1.30, 1.30]))
    torch.testing.assert_close(targets[2], torch.full((4,), 1.30))


def test_lateral_steering_correction_turns_back_toward_course_center():
    import torch

    correction = build_lateral_steering_correction(
        lateral_y=torch.tensor([0.20, -0.20]),
        yaw_rate=torch.zeros(2),
        lateral_gain=2.0,
        yaw_damping_gain=0.5,
        max_correction=0.5,
    )

    assert correction[0, 0].item() > 0.0
    assert correction[0, 1].item() < 0.0
    assert correction[1, 0].item() < 0.0
    assert correction[1, 1].item() > 0.0
    torch.testing.assert_close(correction[:, 0], correction[:, 2])
    torch.testing.assert_close(correction[:, 1], correction[:, 3])


def test_semantic_spatial_wave_tracks_obstacle_from_front_to_rear_axle():
    import torch

    root_pos = torch.zeros((3, 3))
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    ray_hits = torch.tensor(
        [
            [[0.30, 0.0, 0.16], [0.32, 0.02, 0.16]],
            [[-0.30, 0.0, 0.16], [-0.28, -0.02, 0.16]],
            [[0.30, 0.40, 0.16], [0.32, 0.42, 0.16]],
        ]
    )
    semantic = torch.ones((3, 2), dtype=torch.long)

    actions, active, obstacle_x = build_semantic_spatial_wave_reference(
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        ray_hits_w=ray_hits,
        semantic_map=semantic,
        amplitude=0.5,
        knee_ratio=1.0,
        corridor_half_width_m=0.25,
        rear_amplitude_scale=2.0,
        front_overlap_scale=2.0,
        front_support_ratio=0.5,
        rear_support_ratio=0.25,
    )

    assert active.tolist() == [True, True, False]
    assert 0.30 <= obstacle_x[0].item() <= 0.32
    assert -0.30 <= obstacle_x[1].item() <= -0.28
    assert torch.count_nonzero(actions[0, :6]) == 4
    assert torch.count_nonzero(actions[0, 6:]) == 4
    # While the front axle lifts, the rear hips move toward their load-bearing direction.
    assert actions[0, 7].item() < 0.0
    assert torch.count_nonzero(actions[1, :6]) == 4
    assert actions[1, 1].item() == pytest.approx(-0.5)
    assert torch.count_nonzero(actions[1, 6:]) == 4
    assert actions[1, 7].item() == pytest.approx(-1.0)
    assert torch.count_nonzero(actions[2]) == 0


def test_semantic_spatial_wave_releases_front_pair_when_rear_pair_takes_over():
    import torch

    actions, active, _ = build_semantic_spatial_wave_reference(
        root_pos_w=torch.zeros((1, 3)),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ray_hits_w=torch.tensor([[[0.02, 0.0, 0.16]]]),
        semantic_map=torch.ones((1, 1), dtype=torch.long),
        amplitude=0.5,
        knee_ratio=-2.0,
        rear_amplitude_scale=1.0,
        front_overlap_scale=1.0,
        front_support_ratio=0.0,
    )

    assert active.tolist() == [True]
    torch.testing.assert_close(actions[0, :6], torch.zeros(6))
    assert torch.count_nonzero(actions[0, 6:]) == 4


def test_semantic_spatial_wave_preloads_front_axle_at_initial_bar_distance():
    import torch

    actions, active, obstacle_x = build_semantic_spatial_wave_reference(
        root_pos_w=torch.zeros((1, 3)),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ray_hits_w=torch.tensor([[[0.65, 0.0, 0.16]]]),
        semantic_map=torch.ones((1, 1), dtype=torch.long),
        amplitude=0.8,
        knee_ratio=-2.0,
    )

    assert active.tolist() == [True]
    assert obstacle_x.item() == pytest.approx(0.65)
    assert actions[0, 1].item() >= 0.75
    assert actions[0, 4].item() >= 0.75


def test_semantic_spatial_wave_can_limit_motion_to_axle_crossing_windows():
    import torch

    root_pos = torch.zeros((3, 3))
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    ray_hits = torch.tensor(
        [[[0.65, 0.0, 0.06]], [[0.30, 0.0, 0.06]], [[-0.34, 0.0, 0.06]]]
    )
    actions, active, _ = build_semantic_spatial_wave_reference(
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        ray_hits_w=ray_hits,
        semantic_map=torch.ones((3, 1), dtype=torch.long),
        amplitude=1.0,
        knee_ratio=-2.0,
        rear_amplitude_scale=1.0,
        front_lift_window=(0.55, 0.45, 0.20, 0.10),
        rear_lift_window=(-0.10, -0.20, -0.42, -0.52),
    )

    assert active.tolist() == [True, True, True]
    assert torch.count_nonzero(actions[0]) == 0
    assert torch.count_nonzero(actions[1, :6]) == 4
    assert torch.count_nonzero(actions[1, 6:]) == 0
    assert torch.count_nonzero(actions[2, :6]) == 0
    assert torch.count_nonzero(actions[2, 6:]) == 4


def test_wave_reference_smoothing_softens_onset_but_preserves_unclipped_target():
    import torch

    previous = torch.zeros((1, 12))
    target = torch.full((1, 12), 2.4)
    active = torch.tensor([True])

    smoothed = smooth_wave_reference_actions(
        previous=previous, target=target, active=active, alpha=0.25
    )
    torch.testing.assert_close(smoothed, torch.full((1, 12), 0.6))

    for _ in range(80):
        smoothed = smooth_wave_reference_actions(
            previous=smoothed, target=target, active=active, alpha=0.25
        )
    torch.testing.assert_close(smoothed, target, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        smooth_wave_reference_actions(
            previous=smoothed, target=target, active=torch.tensor([False]), alpha=0.25
        ),
        torch.zeros_like(target),
    )


def test_teacher_student_residual_enables_unclipped_dagger_rollout_blend():
    import torch

    teacher = torch.tensor([[0.0, 2.4, -4.8]])
    student = torch.tensor([[0.0, 3.6, -7.2]])

    residual = build_teacher_student_residual(
        student_actions=student, teacher_actions=teacher, student_weight=0.25
    )
    executed = teacher + residual
    torch.testing.assert_close(executed, 0.75 * teacher + 0.25 * student)
    torch.testing.assert_close(
        teacher
        + build_teacher_student_residual(
            student_actions=student, teacher_actions=teacher, student_weight=1.0
        ),
        student,
    )


def test_semantic_course_cuboid_override_builds_wide_shallow_bar():
    from extension.semantic_course_profiles import resolve_cuboid_size_override

    shape_kind, shape_params = resolve_cuboid_size_override(
        semantic_class="small",
        default_shape_kind="cylinder",
        default_shape_params={"radius": 0.30, "height": 0.06, "axis": "Z"},
        cuboid_size_overrides={"small": (0.06, 0.60, 0.06)},
    )

    assert shape_kind == "cuboid"
    assert shape_params == {"size": (0.06, 0.60, 0.06)}


def test_wave_encounter_phase_holds_a_full_cycle_after_short_detection():
    import torch

    elapsed = torch.full((2,), -1.0)
    previous_active = torch.zeros(2, dtype=torch.bool)

    elapsed, previous_active, gate = update_wave_encounter_phase(
        obstacle_active=torch.tensor([True, False]),
        previous_active=previous_active,
        elapsed_s=elapsed,
        step_dt=0.5,
        minimum_duration_s=2.0,
    )
    torch.testing.assert_close(elapsed, torch.tensor([0.0, -1.0]))
    assert gate.tolist() == [True, False]

    for expected_elapsed in (0.5, 1.0, 1.5):
        elapsed, previous_active, gate = update_wave_encounter_phase(
            obstacle_active=torch.tensor([False, False]),
            previous_active=previous_active,
            elapsed_s=elapsed,
            step_dt=0.5,
            minimum_duration_s=2.0,
        )
        assert elapsed[0].item() == expected_elapsed
        assert gate.tolist() == [True, False]

    elapsed, previous_active, gate = update_wave_encounter_phase(
        obstacle_active=torch.tensor([False, True]),
        previous_active=previous_active,
        elapsed_s=elapsed,
        step_dt=0.5,
        minimum_duration_s=2.0,
    )
    torch.testing.assert_close(elapsed, torch.tensor([2.0, 0.0]))
    assert gate.tolist() == [False, True]


def test_wave_encounter_phase_ends_after_one_cycle_even_if_obstacle_remains_visible():
    import torch

    elapsed = torch.tensor([1.5])
    elapsed, previous_active, gate = update_wave_encounter_phase(
        obstacle_active=torch.tensor([True]),
        previous_active=torch.tensor([True]),
        elapsed_s=elapsed,
        step_dt=0.5,
        minimum_duration_s=2.0,
        maximum_duration_s=2.0,
    )

    assert elapsed.item() == pytest.approx(2.0)
    assert previous_active.tolist() == [True]
    assert gate.tolist() == [False]


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "m1_checkpoint_eval.py"
CONTROLLER_SCRIPT = REPO_ROOT / "scripts" / "run_m1_curriculum.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "m1_train.py"


def test_discover_latest_checkpoint_uses_numeric_iteration(tmp_path: Path):
    for name in ("model_9.pt", "model_100.pt", "model_20.pt"):
        (tmp_path / name).touch()

    assert discover_latest_checkpoint(tmp_path).name == "model_100.pt"


def test_roll_gate_accepts_stable_forward_motion():
    report = evaluate_roll_gate(
        episodes=20,
        timeout_episodes=20,
        bad_orientation_episodes=0,
        mean_dx=0.19,
        mean_dy=-0.003,
        max_tilt_rad=0.02,
    )

    assert report["passed"] is True
    assert report["timeout_rate"] == 1.0
    assert report["forward_progress"] is True


def test_semantic_crossing_tracker_counts_straight_crossing_but_not_lateral_avoidance():
    import torch

    candidate = torch.zeros((2, 2))
    heading = torch.zeros((2, 2))
    valid = torch.zeros(2, dtype=torch.bool)
    crossed = torch.zeros(2, dtype=torch.bool)
    ray_hits = torch.tensor([[[0.50, 0.00, 0.16]], [[0.50, 0.00, 0.16]]])
    semantic = torch.ones((2, 1, 1), dtype=torch.long)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    candidate, heading, valid, crossed = update_semantic_crossing_tracker(
        root_pos_w=torch.zeros((2, 3)),
        root_quat_w=root_quat,
        ray_hits_w=ray_hits,
        semantic_map=semantic,
        candidate_xy=candidate,
        candidate_heading=heading,
        candidate_valid=valid,
        crossed=crossed,
    )
    assert valid.tolist() == [True, True]
    assert crossed.tolist() == [False, False]

    _, _, _, crossed = update_semantic_crossing_tracker(
        root_pos_w=torch.tensor([[0.70, 0.00, 0.0], [0.70, 0.50, 0.0]]),
        root_quat_w=root_quat,
        ray_hits_w=ray_hits,
        semantic_map=semantic,
        candidate_xy=candidate,
        candidate_heading=heading,
        candidate_valid=valid,
        crossed=crossed,
    )
    assert crossed.tolist() == [False, False]

    _, _, _, crossed = update_semantic_crossing_tracker(
        root_pos_w=torch.tensor([[0.90, 0.00, 0.0], [0.90, 0.50, 0.0]]),
        root_quat_w=root_quat,
        ray_hits_w=ray_hits,
        semantic_map=semantic,
        candidate_xy=candidate,
        candidate_heading=heading,
        candidate_valid=valid,
        crossed=crossed,
    )
    assert crossed.tolist() == [True, False]


def test_semantic_obstacle_ahead_mask_ignores_lateral_and_behind_obstacles():
    import torch

    root_pos = torch.zeros((3, 3))
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    hits = torch.tensor([[[0.4, 0.0, 0.16]], [[0.4, 0.5, 0.16]], [[-0.4, 0.0, 0.16]]])
    semantic = torch.ones((3, 1, 1), dtype=torch.long)

    active = semantic_obstacle_ahead_mask(
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        ray_hits_w=hits,
        semantic_map=semantic,
    )

    assert active.tolist() == [True, False, False]


def test_roll_gate_rejects_backward_or_excessive_lateral_drift():
    backward = evaluate_roll_gate(
        episodes=20,
        timeout_episodes=20,
        bad_orientation_episodes=0,
        mean_dx=-0.01,
        mean_dy=0.0,
        max_tilt_rad=0.02,
    )
    drifting = evaluate_roll_gate(
        episodes=20,
        timeout_episodes=20,
        bad_orientation_episodes=0,
        mean_dx=0.19,
        mean_dy=0.06,
        max_tilt_rad=0.02,
    )

    assert backward["passed"] is False
    assert drifting["passed"] is False


def test_roll_gate_rejects_old_near_stationary_policy():
    report = evaluate_roll_gate(
        episodes=20,
        timeout_episodes=20,
        bad_orientation_episodes=0,
        mean_dx=0.03,
        mean_dy=0.0,
        max_tilt_rad=0.02,
    )

    assert report["passed"] is False


def test_obstacle_gate_requires_nearly_all_environments_to_clear():
    passed = evaluate_obstacle_gate(
        episodes=20,
        timeout_episodes=20,
        bad_orientation_episodes=0,
        crossing_episodes=20,
        mean_max_dx=1.25,
        mean_dy=0.01,
        max_tilt_rad=0.10,
    )
    failed = evaluate_obstacle_gate(
        episodes=20,
        timeout_episodes=20,
        bad_orientation_episodes=0,
        crossing_episodes=18,
        mean_max_dx=1.25,
        mean_dy=0.01,
        max_tilt_rad=0.10,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["crossing_rate"] == 0.9


def test_obstacle_gate_accepts_task_specific_wave_tilt_limit():
    report = evaluate_obstacle_gate(
        episodes=8,
        timeout_episodes=8,
        bad_orientation_episodes=0,
        crossing_episodes=8,
        mean_max_dx=1.25,
        mean_dy=0.0,
        max_tilt_rad=0.46,
        max_tilt_limit_rad=0.60,
    )

    assert report["tilt_ok"] is True
    assert report["passed"] is True


def test_obstacle_gate_accepts_early_success_without_waiting_for_timeout():
    report = evaluate_obstacle_gate(
        episodes=8,
        timeout_episodes=0,
        bad_orientation_episodes=0,
        crossing_episodes=8,
        mean_max_dx=1.50,
        mean_dy=0.02,
        max_tilt_rad=0.35,
        max_tilt_limit_rad=0.45,
    )

    assert report["timeout_rate"] == 0.0
    assert report["passed"] is True


def test_obstacle_gate_rejects_visually_excessive_wave_tilt():
    report = evaluate_obstacle_gate(
        episodes=8,
        timeout_episodes=8,
        bad_orientation_episodes=0,
        crossing_episodes=8,
        mean_max_dx=1.25,
        mean_dy=0.0,
        max_tilt_rad=0.47,
        max_tilt_limit_rad=0.45,
    )

    assert report["tilt_ok"] is False
    assert report["passed"] is False


def test_checkpoint_eval_writes_report_and_promotes_only_on_pass():
    source = EVAL_SCRIPT.read_text()

    assert "evaluate_roll_gate" in source
    assert 'parser.add_argument("--report"' in source
    assert 'parser.add_argument("--promote"' in source
    assert "root_before_step" in source
    assert "mean_abs_leg_action" in source
    assert "max_abs_leg_action" in source
    assert "leg_action_saturation_rate" in source
    assert "phase_entry_root_pos_m_by_env" in source
    assert "phase_entry_wheel_velocity_by_env" in source
    assert "phase_entry_leg_joint_pos_rad_by_env" in source
    assert "phase_entry_leg_joint_target_rad_by_env" in source
    assert "mean_leg_action_by_index" in source
    assert 'parser.add_argument("--obstacle-threshold"' in source
    assert "evaluate_obstacle_gate" in source
    assert 'if report["passed"] and args.promote' in source


def test_curriculum_controller_gates_wave_training_on_stage1_report():
    source = CONTROLLER_SCRIPT.read_text()

    assert "model_2999.pt" in source
    assert "m1_checkpoint_eval.py" in source
    assert 'report["passed"]' in source
    assert "Isaac-M1-Wave-Flat-v0" in source
    assert "m1_wave_flat_stage2a" in source
    assert "discover_latest_checkpoint" in source
    assert "m1_prepare_wave_checkpoint.py" in source
    assert '"--reset-optimizer"' in source


def test_prepare_wave_checkpoint_zeros_only_leg_output_rows():
    import torch

    actor_weight = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    actor_bias = torch.arange(16, dtype=torch.float32)
    checkpoint = {
        "model_state_dict": {
            "actor.4.weight": actor_weight.clone(),
            "actor.4.bias": actor_bias.clone(),
        }
    }

    prepared = prepare_wave_checkpoint(checkpoint)

    assert torch.count_nonzero(prepared["model_state_dict"]["actor.4.weight"][:12]) == 0
    assert torch.count_nonzero(prepared["model_state_dict"]["actor.4.bias"][:12]) == 0
    assert torch.equal(prepared["model_state_dict"]["actor.4.weight"][12:], actor_weight[12:])
    assert torch.equal(prepared["model_state_dict"]["actor.4.bias"][12:], actor_bias[12:])


def test_prepare_wave_checkpoint_can_raise_only_leg_exploration_noise():
    import torch

    checkpoint = {
        "model_state_dict": {
            "actor.4.weight": torch.ones(16, 2),
            "actor.4.bias": torch.ones(16),
            "std": torch.full((16,), 0.03),
        }
    }

    prepared = prepare_wave_checkpoint(checkpoint, leg_noise_std=0.1)

    assert torch.allclose(prepared["model_state_dict"]["std"][:12], torch.full((12,), 0.1))
    assert torch.allclose(prepared["model_state_dict"]["std"][12:], torch.full((4,), 0.03))


def test_prepare_script_can_preserve_trained_leg_outputs_for_scan_expansion():
    source = (REPO_ROOT / "scripts" / "m1_prepare_wave_checkpoint.py").read_text()

    assert '"--preserve-leg-outputs"' in source
    assert "if not args.preserve_leg_outputs" in source


def test_train_supports_resetting_optimizer_for_stage_transfer():
    source = TRAIN_SCRIPT.read_text()

    assert 'parser.add_argument("--reset-optimizer"' in source
    assert "load_optimizer=not args.reset_optimizer" in source


def test_wave_reference_lifts_front_pair_before_rear_pair():
    import torch

    at_zero = build_wave_reference_actions(
        episode_time_s=torch.tensor([0.0]), amplitude=0.04, knee_ratio=1.5, frequency=0.5
    )
    at_half_second = build_wave_reference_actions(
        episode_time_s=torch.tensor([0.5]), amplitude=0.04, knee_ratio=1.5, frequency=0.5
    )
    at_rear_half_cycle = build_wave_reference_actions(
        episode_time_s=torch.tensor([1.5]), amplitude=0.04, knee_ratio=1.5, frequency=0.5
    )

    assert torch.count_nonzero(at_zero) == 0
    assert at_half_second[0, 1] == 0.04
    assert at_half_second[0, 2] == 0.06
    assert at_half_second[0, 4] == 0.04
    assert at_half_second[0, 5] == 0.06
    assert torch.count_nonzero(at_half_second[0, 6:12]) == 0
    assert at_rear_half_cycle[0, 7] == -0.04
    assert at_rear_half_cycle[0, 8] == -0.06
    assert at_rear_half_cycle[0, 10] == -0.04
    assert at_rear_half_cycle[0, 11] == -0.06
    assert torch.count_nonzero(at_rear_half_cycle[0, :6]) == 0


def test_wave_reference_can_scale_rear_axle_pulse():
    import torch

    actions = build_wave_reference_actions(
        episode_time_s=torch.tensor([1.5]),
        amplitude=0.8,
        knee_ratio=-2.0,
        frequency=0.5,
        rear_amplitude_scale=1.2,
    )

    assert actions[0, 7].item() == pytest.approx(-0.96)
    assert actions[0, 8].item() == pytest.approx(1.92)


def test_wave_reference_coordinates_rear_support_during_front_pulse():
    import torch

    actions = build_wave_reference_actions(
        episode_time_s=torch.tensor([0.5]),
        amplitude=0.8,
        knee_ratio=-2.0,
        frequency=0.5,
        rear_support_ratio=0.5,
    )

    assert actions[0, 1].item() == pytest.approx(0.8)
    assert actions[0, 2].item() == pytest.approx(-1.6)
    assert actions[0, 7].item() == pytest.approx(-0.4)
    assert actions[0, 8].item() == pytest.approx(0.8)


def test_wave_reference_trapezoid_builds_fast_bounded_front_then_rear_pulses():
    import torch

    times = torch.tensor([0.0, 0.02, 0.12, 0.60, 0.72, 1.12])
    actions = build_wave_reference_actions(
        episode_time_s=times,
        amplitude=0.8,
        knee_ratio=-2.0,
        frequency=0.5,
        rear_amplitude_scale=1.2,
        pulse_ramp_s=0.12,
        pulse_hold_s=0.48,
    )

    assert actions[0, 1].item() == pytest.approx(0.0)
    assert actions[2, 1].item() == pytest.approx(0.8)
    assert actions[3, 1].item() == pytest.approx(0.8)
    assert actions[4, 1].item() == pytest.approx(0.0)
    assert actions[5, 7].item() == pytest.approx(-0.96)
    assert torch.abs(actions[1, 2] - actions[0, 2]).item() < 0.35


def test_expand_checkpoint_observations_preserves_state_and_zeros_scan_columns():
    import torch

    actor = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    critic = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    checkpoint = {"model_state_dict": {"actor.0.weight": actor, "critic.0.weight": critic}}

    expanded = expand_checkpoint_observations(checkpoint, new_observation_dim=7)

    actor_out = expanded["model_state_dict"]["actor.0.weight"]
    critic_out = expanded["model_state_dict"]["critic.0.weight"]
    assert actor_out.shape == (4, 7)
    assert critic_out.shape == (2, 7)
    assert torch.equal(actor_out[:, :3], actor)
    assert torch.equal(critic_out[:, :3], critic)
    assert torch.count_nonzero(actor_out[:, 3:]) == 0
    assert torch.count_nonzero(critic_out[:, 3:]) == 0
