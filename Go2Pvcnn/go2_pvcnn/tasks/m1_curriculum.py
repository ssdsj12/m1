"""Pure helpers shared by the autonomous M1 curriculum scripts."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path


_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


def update_wheel_obstacle_clearance(
    *,
    wheel_pos_local,
    wheel_contact_force,
    prelift_seen,
    overbar_clear_seen,
    obstacle_center_x: float,
    obstacle_size_x: float,
    obstacle_height: float,
    wheel_radius: float,
    clearance_margin: float,
    contact_force_limit: float,
    obstacle_center_y: float | None = None,
    obstacle_size_y: float | None = None,
):
    """Track wheel lift before contact and contact-free clearance above a crossbar."""
    import torch

    positions = torch.as_tensor(wheel_pos_local)
    forces = torch.as_tensor(
        wheel_contact_force, dtype=positions.dtype, device=positions.device
    )
    prelift = torch.as_tensor(prelift_seen, dtype=torch.bool, device=positions.device)
    overbar = torch.as_tensor(
        overbar_clear_seen, dtype=torch.bool, device=positions.device
    )
    half_size_x = 0.5 * float(obstacle_size_x)
    required_height = (
        float(obstacle_height) + float(wheel_radius) + float(clearance_margin)
    )
    wheel_x = positions[..., 0]
    wheel_y = positions[..., 1]
    wheel_z = positions[..., 2]
    if obstacle_center_y is None or obstacle_size_y is None:
        in_lateral_swept_region = torch.ones_like(wheel_x, dtype=torch.bool)
    else:
        lateral_limit = 0.5 * float(obstacle_size_y) + float(wheel_radius)
        in_lateral_swept_region = (
            torch.abs(wheel_y - float(obstacle_center_y)) <= lateral_limit
        )
    before_contact = wheel_x <= (
        float(obstacle_center_x) - half_size_x - float(wheel_radius)
    )
    above_bar = torch.abs(wheel_x - float(obstacle_center_x)) <= half_size_x
    clear_height = wheel_z >= required_height
    prelift = prelift | (in_lateral_swept_region & before_contact & clear_height)
    overbar = overbar | (
        in_lateral_swept_region
        & above_bar
        & clear_height
        & (forces <= float(contact_force_limit))
    )
    return prelift, overbar, required_height


def update_clearance_drive_release(
    *,
    obstacle_x,
    wave_gate,
    wheel_heights,
    previous_axle,
    previous_released,
    required_height: float,
    axle_switch_x: float,
):
    """Hold all wheels until the active axle reaches clearance, then latch drive on."""
    import torch

    x = torch.as_tensor(obstacle_x)
    gate = torch.as_tensor(wave_gate, dtype=torch.bool, device=x.device)
    heights = torch.as_tensor(wheel_heights, dtype=x.dtype, device=x.device)
    old_axle = torch.as_tensor(previous_axle, dtype=torch.long, device=x.device)
    old_released = torch.as_tensor(
        previous_released, dtype=torch.bool, device=x.device
    )
    selected_axle = torch.where(
        gate,
        torch.where(
            x >= float(axle_switch_x),
            torch.zeros_like(old_axle),
            torch.ones_like(old_axle),
        ),
        torch.full_like(old_axle, -1),
    )
    phase_changed = selected_axle != old_axle
    released = torch.where(phase_changed, torch.zeros_like(old_released), old_released)
    front_clear = (heights[:, :2] >= float(required_height)).all(dim=1)
    rear_clear = (heights[:, 2:] >= float(required_height)).all(dim=1)
    selected_clear = torch.where(selected_axle == 0, front_clear, rear_clear)
    released = gate & (released | selected_clear)
    drive_allowed = ~gate | released
    return selected_axle, released, drive_allowed


def wheel_x_from_fixed_obstacle(*, wheel_pos_w, root_pos_w, obstacle_x_from_root):
    """Return wheel center X relative to a world-aligned fixed crossbar."""
    import torch

    wheel_pos = torch.as_tensor(wheel_pos_w)
    root_pos = torch.as_tensor(
        root_pos_w, dtype=wheel_pos.dtype, device=wheel_pos.device
    )
    obstacle_x = torch.as_tensor(
        obstacle_x_from_root, dtype=wheel_pos.dtype, device=wheel_pos.device
    )
    return (
        wheel_pos[..., 0]
        - root_pos[:, None, 0]
        - obstacle_x.unsqueeze(-1)
    )


def update_sequential_wheel_crossing_reference(
    *,
    obstacle_x,
    wave_gate,
    wheel_x_from_obstacle,
    wheel_heights,
    previous_phase,
    previous_phase_steps,
    required_height: float,
    past_bar_x: float,
    swing_steps: int,
    min_lift_steps: int,
    ramp_steps: int = 10,
    restore_steps: int = 20,
    clearance_target_height: float = 0.20,
    clearance_feedback_gain: float = 8.0,
    clearance_feedback_max: float = 0.3,
    front_start_x: float = -0.20,
    front_restore_obstacle_x: float = 0.30,
    rear_start_obstacle_x: float = 0.05,
    rear_restore_obstacle_x: float = -0.55,
    support_extension: float = 0.0,
    opposite_abduction: float = 0.0,
    balance_steps: int = 0,
    front_hip_action: float = -1.50,
    front_knee_action: float = -0.80,
    rear_hip_action: float = 1.50,
    rear_knee_action: float = 0.80,
    right_track_only: bool = False,
    keep_drive_during_wave: bool = False,
):
    """Advance a wheel-by-wheel crossing and build its 12-leg reference."""
    import torch

    x = torch.as_tensor(obstacle_x)
    gate = torch.as_tensor(wave_gate, dtype=torch.bool, device=x.device)
    wheel_x = torch.as_tensor(wheel_x_from_obstacle, dtype=x.dtype, device=x.device)
    heights = torch.as_tensor(wheel_heights, dtype=x.dtype, device=x.device)
    old_phase = torch.as_tensor(previous_phase, dtype=torch.long, device=x.device)
    old_steps = torch.as_tensor(previous_phase_steps, dtype=torch.long, device=x.device)
    sequence_gate = gate | ((old_phase >= 0) & (old_phase <= 10))
    front_approach_ready = wheel_x[:, :2].amin(dim=1) >= float(front_start_x)
    phase = torch.where(
        gate & (old_phase < 0) & front_approach_ready,
        torch.zeros_like(old_phase),
        old_phase,
    )

    clear = heights >= float(required_height)
    past = wheel_x >= float(past_bar_x)
    lift_hold_steps = max(int(min_lift_steps), int(ramp_steps))
    lift_ready_steps = max(int(balance_steps), 0) + lift_hold_steps
    if right_track_only:
        transitions = (
            (1, (phase == 0) & (old_steps >= lift_ready_steps) & clear[:, 0]),
            (
                4,
                (phase == 1)
                & past[:, 0]
                & (x <= float(front_restore_obstacle_x)),
            ),
            (5, (phase == 4) & (old_steps >= int(restore_steps))),
            (6, (phase == 5) & (x <= float(rear_start_obstacle_x))),
            (7, (phase == 6) & (old_steps >= lift_ready_steps) & clear[:, 2]),
            (
                10,
                (phase == 7)
                & past[:, 2]
                & (x <= float(rear_restore_obstacle_x)),
            ),
            (11, (phase == 10) & (old_steps >= int(restore_steps))),
        )
    else:
        transitions = tuple(
            (source_phase + 1, transition)
            for source_phase, transition in enumerate(
                (
                    ((phase == 0) & (old_steps >= lift_hold_steps) & clear[:, 0]),
                    ((phase == 1) & past[:, 0]),
                    ((phase == 2) & (old_steps >= lift_hold_steps) & clear[:, 1]),
                    ((phase == 3) & past[:, 1]),
                    ((phase == 4) & (old_steps >= int(restore_steps))),
                    ((phase == 5) & (x <= float(rear_start_obstacle_x))),
                    ((phase == 6) & (old_steps >= lift_hold_steps) & clear[:, 2]),
                    ((phase == 7) & past[:, 2]),
                    ((phase == 8) & (old_steps >= lift_hold_steps) & clear[:, 3]),
                    ((phase == 9) & past[:, 3]),
                    ((phase == 10) & (old_steps >= int(restore_steps))),
                )
            )
        )
    for target_phase, transition in transitions:
        phase = torch.where(transition, torch.full_like(phase, target_phase), phase)

    phase = torch.where(sequence_gate, phase, torch.full_like(phase, -1))
    changed = phase != old_phase
    phase_steps = torch.where(
        ~sequence_gate | changed,
        torch.zeros_like(old_steps),
        old_steps + 1,
    )
    action_steps = torch.where(changed, torch.zeros_like(old_steps), old_steps)
    lift_blend = torch.clamp(
        (
            action_steps.to(x.dtype)
            - float(max(int(balance_steps), 0))
            + 1.0
        )
        / max(int(ramp_steps), 1),
        0.0,
        1.0,
    )
    support_ramp = torch.clamp(
        (action_steps.to(x.dtype) + 1.0) / max(int(balance_steps), 1),
        0.0,
        1.0,
    )
    support_blend = torch.where(
        (phase == 0) | (phase == 6),
        support_ramp,
        torch.ones_like(support_ramp),
    )
    swing_blend = torch.clamp(
        (action_steps.to(x.dtype) + 1.0) / max(int(swing_steps), 1), 0.0, 1.0
    )
    restore_blend = torch.clamp(
        1.0 - action_steps.to(x.dtype) / max(int(restore_steps), 1), 0.0, 1.0
    )
    reference = torch.zeros((x.shape[0], 12), dtype=x.dtype, device=x.device)

    clearance_feedback = torch.clamp(
        (float(clearance_target_height) - heights) * float(clearance_feedback_gain),
        min=0.0,
        max=float(clearance_feedback_max),
    )

    def set_leg(mask, wheel_index: int, hip, knee) -> None:
        hip_index = wheel_index * 3 + 1
        knee_index = wheel_index * 3 + 2
        hip_value = torch.as_tensor(hip, dtype=x.dtype, device=x.device)
        knee_value = torch.as_tensor(knee, dtype=x.dtype, device=x.device)
        reference[:, hip_index] = torch.where(mask, hip_value, reference[:, hip_index])
        reference[:, knee_index] = torch.where(mask, knee_value, reference[:, knee_index])

    front_hip = float(front_hip_action)
    front_left_lift_hip = 1.0
    front_left_lift_knee = -2.0
    front_left_hold_hip = -2.0
    front_left_hold_knee = -1.0
    front_vertical_hip = 1.0
    front_vertical_knee = -2.0
    front_knee = float(front_knee_action) - clearance_feedback[:, 0]
    front_swing_knee = front_knee
    rear_hip = float(rear_hip_action)
    rear_left_lift_hip = -1.0
    rear_left_lift_knee = 2.0
    rear_left_hold_hip = 2.0
    rear_left_hold_knee = 1.0
    rear_vertical_hip = -1.0
    rear_vertical_knee = 2.0
    rear_knee = float(rear_knee_action) + clearance_feedback[:, 2]
    rear_swing_knee = rear_knee

    set_leg(
        phase == 0,
        0,
        front_hip * lift_blend,
        front_swing_knee * lift_blend,
    )
    set_leg(phase == 1, 0, front_hip, front_swing_knee)
    set_leg(
        phase == 2,
        0,
        front_hip,
        front_swing_knee,
    )
    set_leg(
        phase == 2,
        1,
        front_left_lift_hip * lift_blend,
        front_left_lift_knee * lift_blend,
    )
    set_leg(phase == 3, 0, front_hip, front_swing_knee)
    set_leg(
        phase == 3,
        1,
        front_left_lift_hip
        + (front_left_hold_hip - front_left_lift_hip) * swing_blend,
        front_left_lift_knee
        + (front_left_hold_knee - front_left_lift_knee) * swing_blend,
    )
    set_leg(
        phase == 4,
        0,
        front_hip * restore_blend,
        front_swing_knee * restore_blend,
    )
    set_leg(
        (phase == 4) & (not right_track_only),
        1,
        front_left_hold_hip * restore_blend,
        front_left_hold_knee * restore_blend,
    )

    set_leg(
        phase == 6,
        2,
        rear_hip * lift_blend,
        rear_swing_knee * lift_blend,
    )
    set_leg(phase == 7, 2, rear_hip, rear_swing_knee)
    set_leg(
        phase == 8,
        2,
        rear_hip,
        rear_swing_knee,
    )
    set_leg(
        phase == 8,
        3,
        rear_left_lift_hip * lift_blend,
        rear_left_lift_knee * lift_blend,
    )
    set_leg(phase == 9, 2, rear_hip, rear_swing_knee)
    set_leg(
        phase == 9,
        3,
        rear_left_lift_hip
        + (rear_left_hold_hip - rear_left_lift_hip) * swing_blend,
        rear_left_lift_knee
        + (rear_left_hold_knee - rear_left_lift_knee) * swing_blend,
    )
    set_leg(
        phase == 10,
        2,
        rear_hip * restore_blend,
        rear_swing_knee * restore_blend,
    )
    set_leg(
        (phase == 10) & (not right_track_only),
        3,
        rear_left_hold_hip * restore_blend,
        rear_left_hold_knee * restore_blend,
    )

    leg_active = sequence_gate & (
        ((phase >= 0) & (phase <= 4)) | ((phase >= 6) & (phase <= 10))
    )
    selected_wheel = torch.full_like(phase, -1)
    for wheel_index, phase_pair in enumerate(((0, 1), (2, 3), (6, 7), (8, 9))):
        selected_wheel = torch.where(
            (phase == phase_pair[0]) | (phase == phase_pair[1]),
            torch.full_like(selected_wheel, wheel_index),
            selected_wheel,
        )
    selected_active = selected_wheel >= 0
    for support_wheel in range(4):
        support = selected_active & (selected_wheel != support_wheel)
        knee_index = support_wheel * 3 + 2
        extension_direction = 1.0 if support_wheel < 2 else -1.0
        reference[:, knee_index] += (
            support.to(x.dtype)
            * extension_direction
            * float(support_extension)
            * support_blend
        )
    selected_negative_y = selected_active & ((selected_wheel % 2) == 0)
    selected_positive_y = selected_active & ((selected_wheel % 2) == 1)
    for support_wheel in (1, 3):
        abad_index = support_wheel * 3
        reference[:, abad_index] = torch.where(
            selected_negative_y,
            torch.full_like(reference[:, abad_index], float(opposite_abduction))
            * support_blend,
            reference[:, abad_index],
        )
    for support_wheel in (0, 2):
        abad_index = support_wheel * 3
        reference[:, abad_index] = torch.where(
            selected_positive_y,
            torch.full_like(reference[:, abad_index], float(opposite_abduction))
            * support_blend,
            reference[:, abad_index],
        )
    if right_track_only:
        stopped_phase = (
            (phase == 0)
            | ((phase == 1) & ~past[:, 0])
            | (phase == 4)
            | (phase == 6)
            | ((phase == 7) & ~past[:, 2])
            | (phase == 10)
        )
    else:
        swing_in_progress = (
            ((phase == 3) | (phase == 9))
            & (action_steps < max(int(swing_steps), 1))
        )
        stopped_phase = (
            (phase == 0)
            | (phase == 2)
            | (phase == 4)
            | (phase == 6)
            | (phase == 8)
            | (phase == 10)
            | swing_in_progress
        )
    drive_allowed = ~sequence_gate | ~stopped_phase
    if keep_drive_during_wave:
        drive_allowed = torch.ones_like(drive_allowed)
    return phase, phase_steps, reference, drive_allowed, leg_active


def sequential_active_leg_mask(phase):
    """Release only the three joints belonging to the wheel crossing now."""
    import torch

    state = torch.as_tensor(phase)
    mask = torch.zeros((state.shape[0], 12), dtype=torch.bool, device=state.device)
    for wheel_index, phase_pair in enumerate(((0, 1), (2, 3), (6, 7), (8, 9))):
        active = (state == phase_pair[0]) | (state == phase_pair[1])
        start = wheel_index * 3
        mask[:, start : start + 3] = active.unsqueeze(-1)
    return mask


def merge_task_space_support_with_jointspace_active(
    *, task_space_reference, joint_space_reference, phase
):
    """Keep active ABAD locked while task-space XZ control handles wheel swing."""
    import torch

    support = torch.as_tensor(task_space_reference)
    active = torch.as_tensor(
        joint_space_reference, dtype=support.dtype, device=support.device
    )
    state = torch.as_tensor(phase, dtype=torch.long, device=support.device)
    active_mask = sequential_active_leg_mask(state)
    swing = (state == 1) | (state == 3) | (state == 7) | (state == 9)
    abad_mask = torch.zeros_like(active_mask)
    abad_mask[:, (0, 3, 6, 9)] = True
    joint_space_mask = active_mask & (~swing.unsqueeze(-1) | abad_mask)
    return torch.where(joint_space_mask, active, support)


def sequential_leg_residual_scale(
    phase,
    support_scale: float = 0.75,
    crossing_scale: float = 0.25,
    support_abduction_scale: float = 0.15,
):
    """Keep the crossing trajectory authoritative while supports balance in wave."""
    import torch

    state = torch.as_tensor(phase)
    active_wave = (
        ((state >= 0) & (state <= 4))
        | ((state >= 6) & (state <= 10))
    )
    scale = active_wave.to(torch.float32).unsqueeze(-1).expand(-1, 12).clone()
    scale *= float(support_scale)
    scale[:, (0, 3, 6, 9)] = (
        active_wave.to(scale.dtype).unsqueeze(-1) * float(support_abduction_scale)
    )
    selected = sequential_active_leg_mask(state)
    scale = torch.where(selected, torch.full_like(scale, float(crossing_scale)), scale)
    selected_abduction = selected.clone()
    selected_abduction[:, (1, 2, 4, 5, 7, 8, 10, 11)] = False
    return torch.where(selected_abduction, torch.zeros_like(scale), scale)


def compose_sequential_leg_actions(
    *,
    policy_actions,
    teacher_actions,
    residual_scale,
    policy_control: bool,
    policy_weight: float = 1.0,
):
    """Select autonomous policy control or teacher-plus-residual control."""
    if policy_control:
        weight = max(0.0, min(float(policy_weight), 1.0))
        return teacher_actions * (1.0 - weight) + policy_actions * weight
    return teacher_actions + policy_actions * residual_scale


def scheduled_student_rollout_weight(
    *,
    update: int,
    total_updates: int,
    final_weight: float,
    teacher_forcing_fraction: float,
    full_weight_fraction: float = 0.80,
) -> float:
    """Hold exact teacher control first, then linearly hand control to the student."""
    total = max(int(total_updates), 1)
    progress = max(0.0, min(float(update) / total, 1.0))
    start = max(0.0, min(float(teacher_forcing_fraction), 1.0))
    end = max(start + 1.0e-6, min(float(full_weight_fraction), 1.0))
    ramp = max(0.0, min((progress - start) / (end - start), 1.0))
    return max(0.0, min(float(final_weight), 1.0)) * ramp


def build_sequential_phase_observation(
    *, phase, phase_steps, progress_steps: int = 50
):
    """Encode non-wave/phase identity and bounded within-phase progress."""
    import torch

    state = torch.as_tensor(phase, dtype=torch.long)
    steps = torch.as_tensor(phase_steps, device=state.device)
    phase_index = torch.clamp(state + 1, min=0, max=12)
    one_hot = torch.nn.functional.one_hot(phase_index, num_classes=13).to(
        torch.float32
    )
    progress = torch.clamp(
        steps.to(torch.float32) / max(int(progress_steps), 1), 0.0, 1.0
    ).unsqueeze(-1)
    progress = torch.where(
        (state >= 0).unsqueeze(-1), progress, torch.zeros_like(progress)
    )
    return torch.cat((one_hot, progress), dim=1)


def blend_policy_wave_gate(
    *, oracle_gate, policy_score, policy_weight: float, threshold: float = 0.0
):
    """Blend an oracle boolean gate with a learned signed gate score."""
    import torch

    oracle = torch.as_tensor(oracle_gate, dtype=torch.bool)
    score = torch.as_tensor(policy_score, device=oracle.device)
    teacher_score = oracle.to(score.dtype) * 2.0 - 1.0
    weight = max(0.0, min(float(policy_weight), 1.0))
    blended = teacher_score * (1.0 - weight) + score * weight
    return blended > float(threshold), blended


def apply_fixed_course_gate_safety_window(
    *,
    policy_gate,
    oracle_gate,
    root_local_x,
    minimum_root_x: float,
    fallback_root_x: float,
):
    """Constrain a learned gate to the validated fixed-course trigger window."""
    import torch

    policy = torch.as_tensor(policy_gate, dtype=torch.bool)
    oracle = torch.as_tensor(oracle_gate, dtype=torch.bool, device=policy.device)
    root_x = torch.as_tensor(root_local_x, device=policy.device)
    allowed = root_x >= float(minimum_root_x)
    fallback = oracle & (root_x >= float(fallback_root_x))
    return (policy & allowed) | fallback, fallback


def required_axle_lift_passed(
    *,
    wheel_height_max,
    wheel_clearance_required,
    min_front_height: float,
    min_rear_height: float,
):
    """Check lift thresholds only for wheels selected by the crossing task."""
    import torch

    heights = torch.as_tensor(wheel_height_max)
    required = torch.as_tensor(
        wheel_clearance_required, dtype=torch.bool, device=heights.device
    ).any(dim=0)
    front_passed = bool(
        ((heights[:2] >= float(min_front_height)) | ~required[:2]).all().item()
    )
    rear_passed = bool(
        ((heights[2:] >= float(min_rear_height)) | ~required[2:]).all().item()
    )
    return front_passed, rear_passed


def build_task_space_wheel_joint_actions(
    *,
    phase,
    wheel_x_w,
    wheel_heights,
    wheel_x_obstacle_w,
    leg_joint_pos,
    default_leg_joint_pos,
    wheel_xz_jacobians,
    lift_height: float,
    past_bar_x: float,
    action_scale: float,
    damping: float,
    max_joint_step: float,
    nominal_wheel_x=None,
    nominal_wheel_heights=None,
    hold_supports: bool = False,
    swing_with_body: bool = False,
):
    """Convert active-wheel x/z errors into hip/knee position actions."""
    import torch

    state = torch.as_tensor(phase, dtype=torch.long)
    wheel_x = torch.as_tensor(wheel_x_w)
    heights = torch.as_tensor(
        wheel_heights, dtype=wheel_x.dtype, device=wheel_x.device
    )
    obstacle_x = torch.as_tensor(
        wheel_x_obstacle_w, dtype=wheel_x.dtype, device=wheel_x.device
    )
    joint_pos = torch.as_tensor(
        leg_joint_pos, dtype=wheel_x.dtype, device=wheel_x.device
    )
    default_pos = torch.as_tensor(
        default_leg_joint_pos, dtype=wheel_x.dtype, device=wheel_x.device
    )
    jacobians = torch.as_tensor(
        wheel_xz_jacobians, dtype=wheel_x.dtype, device=wheel_x.device
    )
    state = state.to(device=wheel_x.device)

    selected = torch.full_like(state, -1)
    for wheel_index, phase_pair in enumerate(((0, 1), (2, 3), (6, 7), (8, 9))):
        selected = torch.where(
            (state == phase_pair[0]) | (state == phase_pair[1]),
            torch.full_like(selected, wheel_index),
            selected,
        )
    valid = selected >= 0
    selected_safe = torch.clamp(selected, min=0)
    row = torch.arange(state.shape[0], device=wheel_x.device)
    swing = (state == 1) | (state == 3) | (state == 7) | (state == 9)

    def solve_joint_delta(jacobian, position_error):
        if float(damping) > 0.0:
            identity = torch.eye(2, dtype=wheel_x.dtype, device=wheel_x.device)
            system = jacobian @ jacobian.transpose(1, 2)
            system = system + float(damping) ** 2 * identity.unsqueeze(0)
            joint_delta = jacobian.transpose(1, 2) @ torch.linalg.solve(
                system, position_error.unsqueeze(-1)
            )
            joint_delta = joint_delta.squeeze(-1)
        else:
            joint_delta = (
                torch.linalg.pinv(jacobian) @ position_error.unsqueeze(-1)
            ).squeeze(-1)
        delta_norm = torch.linalg.vector_norm(joint_delta, dim=1, keepdim=True)
        delta_scale = torch.clamp(
            float(max_joint_step) / torch.clamp_min(delta_norm, 1.0e-6), max=1.0
        )
        return joint_delta * delta_scale

    if hold_supports:
        nominal_x = torch.as_tensor(
            nominal_wheel_x, dtype=wheel_x.dtype, device=wheel_x.device
        )
        nominal_z = torch.as_tensor(
            nominal_wheel_heights, dtype=wheel_x.dtype, device=wheel_x.device
        )
        target_x = nominal_x.clone()
        target_z = nominal_z.clone()
        target_z[row, selected_safe] += valid.to(wheel_x.dtype) * float(lift_height)
        if not swing_with_body:
            target_x[row, selected_safe] = torch.where(
                swing,
                obstacle_x + float(past_bar_x),
                target_x[row, selected_safe],
            )
        position_error = torch.stack(
            (target_x - wheel_x, target_z - heights), dim=-1
        )
        position_error = position_error * valid.to(wheel_x.dtype).view(-1, 1, 1)
        flat_jacobian = jacobians.reshape(-1, 2, 2)
        flat_error = position_error.reshape(-1, 2)
        joint_delta = solve_joint_delta(flat_jacobian, flat_error).reshape(-1, 4, 2)
        joint_actions = (
            joint_pos + joint_delta - default_pos
        ) / float(action_scale)
        joint_actions = joint_actions * valid.to(wheel_x.dtype).view(-1, 1, 1)
        actions = torch.zeros(
            (state.shape[0], 12), dtype=wheel_x.dtype, device=wheel_x.device
        )
        for wheel_index in range(4):
            start = wheel_index * 3 + 1
            actions[:, start : start + 2] = joint_actions[:, wheel_index]
        return actions

    x_error = torch.where(
        swing,
        obstacle_x + float(past_bar_x) - wheel_x[row, selected_safe],
        torch.zeros_like(obstacle_x),
    )
    z_error = float(lift_height) - heights[row, selected_safe]
    position_error = torch.stack((x_error, z_error), dim=1)
    position_error = torch.where(
        valid.unsqueeze(-1), position_error, torch.zeros_like(position_error)
    )
    jacobian = jacobians[row, selected_safe]
    joint_delta = solve_joint_delta(jacobian, position_error)
    selected_joint_pos = joint_pos[row, selected_safe]
    selected_default_pos = default_pos[row, selected_safe]
    selected_actions = (
        selected_joint_pos + joint_delta - selected_default_pos
    ) / float(action_scale)
    selected_actions = torch.where(
        valid.unsqueeze(-1), selected_actions, torch.zeros_like(selected_actions)
    )

    actions = torch.zeros((state.shape[0], 12), dtype=wheel_x.dtype, device=wheel_x.device)
    for wheel_index in range(4):
        wheel_mask = selected == wheel_index
        start = wheel_index * 3 + 1
        actions[:, start : start + 2] = torch.where(
            wheel_mask.unsqueeze(-1), selected_actions, actions[:, start : start + 2]
        )
    return actions


def resolve_m1_leg_joint_ids_by_wheel(joint_names):
    """Resolve M1 leg joints by semantic name despite PhysX depth ordering."""
    name_to_id = {name: index for index, name in enumerate(joint_names)}
    return [
        [
            name_to_id[f"{prefix}_ABAD_JOINT"],
            name_to_id[f"{prefix}_HIP_JOINT"],
            name_to_id[f"{prefix}_KNEE_JOINT"],
        ]
        for prefix in ("FAR", "FBL", "RAR", "RBL")
    ]


def build_stabilized_task_space_wheel_actions(
    *,
    phase,
    phase_steps=None,
    wheel_pos_b,
    nominal_wheel_pos_b,
    wheel_x_obstacle_b,
    leg_joint_pos,
    default_leg_joint_pos,
    wheel_xyz_jacobians,
    lift_delta: float,
    past_bar_x: float,
    action_scale: float,
    damping: float,
    max_joint_step: float,
    lateral_body_shift: float,
    swing_with_body: bool,
    rear_lift_delta: float | None = None,
    axle_pair_mode: bool = False,
    pair_support_extension: float = 0.0,
    pair_body_shift_x: float = 0.0,
    balanced_wheel_x_b=None,
    balance_steps: int = 0,
    lift_ramp_steps: int = 1,
    rear_restore_steps: int = 20,
    rear_restore_forward_offset: float = 0.0,
    base_lateral_offset=None,
    lateral_recovery_gain: float = 0.0,
    lateral_recovery_max: float = 0.0,
    stabilize_supports: bool = False,
    swing_ramp_steps: int = 1,
    longitudinal_body_shift: float = 0.0,
    balance_supports: bool = False,
    active_swing_xz_only: bool = False,
):
    """Track one lifted wheel while support legs stabilize body height and lateral COM."""
    import torch

    state = torch.as_tensor(phase, dtype=torch.long)
    wheel_pos = torch.as_tensor(wheel_pos_b)
    nominal_pos = torch.as_tensor(
        nominal_wheel_pos_b, dtype=wheel_pos.dtype, device=wheel_pos.device
    )
    obstacle_x = torch.as_tensor(
        wheel_x_obstacle_b, dtype=wheel_pos.dtype, device=wheel_pos.device
    )
    joint_pos = torch.as_tensor(
        leg_joint_pos, dtype=wheel_pos.dtype, device=wheel_pos.device
    )
    default_pos = torch.as_tensor(
        default_leg_joint_pos, dtype=wheel_pos.dtype, device=wheel_pos.device
    )
    jacobians = torch.as_tensor(
        wheel_xyz_jacobians, dtype=wheel_pos.dtype, device=wheel_pos.device
    )
    state = state.to(device=wheel_pos.device)
    if phase_steps is None:
        steps = torch.full_like(
            state, int(balance_steps) + max(int(lift_ramp_steps), 1)
        )
    else:
        steps = torch.as_tensor(
            phase_steps, dtype=torch.long, device=wheel_pos.device
        )

    row = torch.arange(state.shape[0], device=wheel_pos.device)
    target_pos = nominal_pos.clone()
    if axle_pair_mode:
        valid = (state >= 0) & (state <= 5)
        selected_mask = torch.zeros(
            (state.shape[0], 4), dtype=torch.bool, device=wheel_pos.device
        )
        front_active = (state == 0) | (state == 1)
        rear_active = (state == 3) | (state == 4)
        pair_swing = (state == 1) | (state == 4)
        selected_mask[:, :2] = front_active.unsqueeze(-1)
        selected_mask[:, 2:] = rear_active.unsqueeze(-1)
        if int(balance_steps) > 0:
            balance_ramp = torch.clamp(
                steps.to(wheel_pos.dtype) / float(balance_steps), 0.0, 1.0
            )
        else:
            balance_ramp = torch.ones_like(obstacle_x)
        lift_ramp = torch.clamp(
            (steps.to(wheel_pos.dtype) - float(balance_steps))
            / max(float(lift_ramp_steps), 1.0),
            0.0,
            1.0,
        )
        lift_scale = torch.where(pair_swing, torch.ones_like(lift_ramp), lift_ramp)
        target_pos[..., 2] += (
            selected_mask.to(wheel_pos.dtype)
            * lift_scale.unsqueeze(-1)
            * float(lift_delta)
        )
        pair_support_mask = selected_mask.any(dim=1, keepdim=True) & ~selected_mask
        target_pos[..., 2] -= (
            pair_support_mask.to(wheel_pos.dtype)
            * lift_scale.unsqueeze(-1)
            * float(pair_support_extension)
        )
        pair_body_shift = torch.where(
            front_active,
            torch.full_like(obstacle_x, float(pair_body_shift_x)),
            torch.where(
                rear_active,
                torch.full_like(obstacle_x, -float(pair_body_shift_x)),
                torch.zeros_like(obstacle_x),
            ),
        )
        prebalancing = (front_active | rear_active) & (steps < int(balance_steps))
        prebalance_target_x = nominal_pos[..., 0] + (
            pair_body_shift * balance_ramp
        ).unsqueeze(-1)
        if balanced_wheel_x_b is None:
            held_pair_x = wheel_pos[..., 0]
        else:
            held_pair_x = torch.as_tensor(
                balanced_wheel_x_b,
                dtype=wheel_pos.dtype,
                device=wheel_pos.device,
            )
        target_pos[..., 0] = torch.where(
            prebalancing.unsqueeze(-1),
            prebalance_target_x,
            torch.where(
                (front_active | rear_active).unsqueeze(-1),
                held_pair_x,
                target_pos[..., 0],
            ),
        )
        if not swing_with_body:
            target_pos[..., 0] = torch.where(
                selected_mask & pair_swing.unsqueeze(-1),
                obstacle_x.unsqueeze(-1) + float(past_bar_x),
                target_pos[..., 0],
            )
        support_mask = valid.unsqueeze(-1) & ~selected_mask
        position_error_mask = valid.view(-1, 1, 1).expand(-1, 4, 3)
        joint_action_mask = position_error_mask
    else:
        selected = torch.full_like(state, -1)
        for wheel_index, phase_pair in enumerate(((0, 1), (2, 3), (6, 7), (8, 9))):
            selected = torch.where(
                (state == phase_pair[0]) | (state == phase_pair[1]),
                torch.full_like(selected, wheel_index),
                selected,
            )
        selected = torch.where(state == 10, torch.full_like(selected, 2), selected)
        valid = selected >= 0
        selected_safe = torch.clamp(selected, min=0)
        swing = (state == 1) | (state == 3) | (state == 7) | (state == 9)
        rear_restoring = state == 10
        lift_phase = (state == 0) | (state == 2) | (state == 6) | (state == 8)
        if int(balance_steps) > 0:
            balance_ramp = torch.clamp(
                steps.to(wheel_pos.dtype) / float(balance_steps), 0.0, 1.0
            )
        else:
            balance_ramp = torch.ones_like(obstacle_x)
        lift_ramp = torch.clamp(
            (steps.to(wheel_pos.dtype) - float(balance_steps))
            / max(float(lift_ramp_steps), 1.0),
            0.0,
            1.0,
        )
        lift_scale = torch.where(swing, torch.ones_like(lift_ramp), lift_ramp)
        restore_scale = torch.clamp(
            1.0 - steps.to(wheel_pos.dtype) / max(float(rear_restore_steps), 1.0),
            0.0,
            1.0,
        )
        lift_scale = torch.where(rear_restoring, restore_scale, lift_scale)
        lift_scale = torch.where(
            lift_phase | swing | rear_restoring,
            lift_scale,
            torch.zeros_like(lift_scale),
        )
        rear_lift = float(lift_delta) if rear_lift_delta is None else float(rear_lift_delta)
        selected_lift_delta = torch.where(
            selected_safe < 2,
            torch.full_like(obstacle_x, float(lift_delta)),
            torch.full_like(obstacle_x, rear_lift),
        )
        target_pos[row, selected_safe, 2] += (
            valid.to(wheel_pos.dtype) * lift_scale * selected_lift_delta
        )
        if not swing_with_body:
            swing_ramp = torch.clamp(
                (steps.to(wheel_pos.dtype) + 1.0)
                / max(float(swing_ramp_steps), 1.0),
                0.0,
                1.0,
            )
            swing_target_x = nominal_pos[row, selected_safe, 0] + swing_ramp * (
                obstacle_x
                + float(past_bar_x)
                - nominal_pos[row, selected_safe, 0]
            )
            target_pos[row, selected_safe, 0] = torch.where(
                swing,
                swing_target_x,
                target_pos[row, selected_safe, 0],
            )
        lateral_shift = torch.where(
            (selected_safe % 2) == 0,
            torch.full_like(obstacle_x, float(lateral_body_shift)),
            torch.full_like(obstacle_x, -float(lateral_body_shift)),
        )
        shift_ramp = torch.where(swing, torch.ones_like(balance_ramp), balance_ramp)
        shift_ramp = torch.where(rear_restoring, restore_scale, shift_ramp)
        lateral_shift *= shift_ramp
        longitudinal_shift = torch.where(
            selected_safe < 2,
            torch.full_like(obstacle_x, -float(longitudinal_body_shift)),
            torch.full_like(obstacle_x, float(longitudinal_body_shift)),
        ) * shift_ramp
        if balance_supports:
            target_pos[..., 0] = torch.where(
                valid.unsqueeze(-1),
                nominal_pos[..., 0] - longitudinal_shift.unsqueeze(-1),
                target_pos[..., 0],
            )
            target_pos[..., 1] = torch.where(
                valid.unsqueeze(-1),
                nominal_pos[..., 1] - lateral_shift.unsqueeze(-1),
                target_pos[..., 1],
            )
        else:
            target_pos[row, selected_safe, 1] = torch.where(
                valid,
                nominal_pos[row, selected_safe, 1] - lateral_shift,
                target_pos[row, selected_safe, 1],
            )
        target_pos[row, selected_safe, 0] = torch.where(
            rear_restoring,
            nominal_pos[row, selected_safe, 0]
            + float(rear_restore_forward_offset),
            target_pos[row, selected_safe, 0],
        )
        support_mask = torch.ones(
            (state.shape[0], 4), dtype=torch.bool, device=wheel_pos.device
        )
        support_mask[row, selected_safe] = False
        support_mask &= valid.unsqueeze(-1)
        support_balance_mask = support_mask & bool(balance_supports)
        selected_mask = torch.nn.functional.one_hot(
            selected_safe, num_classes=4
        ).to(dtype=torch.bool)
        selected_mask &= valid.unsqueeze(-1)
        recovery_enabled = (
            base_lateral_offset is not None
            and float(lateral_recovery_gain) > 0.0
            and float(lateral_recovery_max) > 0.0
        )
        support_control_mask = torch.zeros_like(support_mask)
        if recovery_enabled:
            lateral_offset = torch.as_tensor(
                base_lateral_offset,
                dtype=wheel_pos.dtype,
                device=wheel_pos.device,
            )
            recovery_shift = torch.clamp(
                -lateral_offset * float(lateral_recovery_gain),
                min=-float(lateral_recovery_max),
                max=float(lateral_recovery_max),
            )
            target_pos[..., 1] = torch.where(
                support_mask,
                target_pos[..., 1] - recovery_shift.unsqueeze(-1),
                target_pos[..., 1],
            )
            support_control_mask = support_mask
        if stabilize_supports:
            controlled_wheels = selected_mask | support_mask
            position_error_mask = controlled_wheels.unsqueeze(-1).expand(-1, -1, 3)
            joint_action_mask = position_error_mask
        else:
            position_error_mask = selected_mask.unsqueeze(-1).expand(-1, -1, 3).clone()
            position_error_mask[..., :2] |= support_balance_mask.unsqueeze(-1)
            position_error_mask[..., 1] |= support_control_mask
            joint_action_mask = selected_mask.unsqueeze(-1).expand(-1, -1, 3).clone()
            joint_action_mask |= support_balance_mask.unsqueeze(-1)
            joint_action_mask[..., 0] |= support_control_mask
    position_error = (
        (target_pos - wheel_pos)
        * position_error_mask.to(wheel_pos.dtype)
    )

    flat_jacobian = jacobians.reshape(-1, 3, 3)
    flat_error = position_error.reshape(-1, 3)
    if float(damping) > 0.0:
        identity = torch.eye(3, dtype=wheel_pos.dtype, device=wheel_pos.device)
        system = flat_jacobian @ flat_jacobian.transpose(1, 2)
        system = system + float(damping) ** 2 * identity.unsqueeze(0)
        joint_delta = flat_jacobian.transpose(1, 2) @ torch.linalg.solve(
            system, flat_error.unsqueeze(-1)
        )
        joint_delta = joint_delta.squeeze(-1)
    else:
        joint_delta = (
            torch.linalg.pinv(flat_jacobian) @ flat_error.unsqueeze(-1)
        ).squeeze(-1)
    delta_norm = torch.linalg.vector_norm(joint_delta, dim=1, keepdim=True)
    delta_scale = torch.clamp(
        float(max_joint_step) / torch.clamp_min(delta_norm, 1.0e-6), max=1.0
    )
    joint_delta = (joint_delta * delta_scale).reshape(-1, 4, 3)
    actions = (joint_pos + joint_delta - default_pos) / float(action_scale)
    actions *= joint_action_mask.to(wheel_pos.dtype)
    if active_swing_xz_only and not axle_pair_mode:
        xz_jacobians = jacobians[..., [0, 2], :][..., 1:3]
        xz_error = position_error[..., [0, 2]]
        flat_xz_jacobian = xz_jacobians.reshape(-1, 2, 2)
        flat_xz_error = xz_error.reshape(-1, 2)
        if float(damping) > 0.0:
            identity = torch.eye(2, dtype=wheel_pos.dtype, device=wheel_pos.device)
            system = flat_xz_jacobian @ flat_xz_jacobian.transpose(1, 2)
            system = system + float(damping) ** 2 * identity.unsqueeze(0)
            xz_joint_delta = flat_xz_jacobian.transpose(1, 2) @ torch.linalg.solve(
                system, flat_xz_error.unsqueeze(-1)
            )
            xz_joint_delta = xz_joint_delta.squeeze(-1)
        else:
            xz_joint_delta = (
                torch.linalg.pinv(flat_xz_jacobian)
                @ flat_xz_error.unsqueeze(-1)
            ).squeeze(-1)
        xz_delta_norm = torch.linalg.vector_norm(
            xz_joint_delta, dim=1, keepdim=True
        )
        xz_delta_scale = torch.clamp(
            float(max_joint_step) / torch.clamp_min(xz_delta_norm, 1.0e-6),
            max=1.0,
        )
        xz_joint_delta = (xz_joint_delta * xz_delta_scale).reshape(-1, 4, 2)
        xz_actions = (
            joint_pos[..., 1:3] + xz_joint_delta - default_pos[..., 1:3]
        ) / float(action_scale)
        swing_selected = selected_mask & swing.unsqueeze(-1)
        actions[..., 0] = torch.where(
            swing_selected, torch.zeros_like(actions[..., 0]), actions[..., 0]
        )
        actions[..., 1:3] = torch.where(
            swing_selected.unsqueeze(-1), xz_actions, actions[..., 1:3]
        )
    return actions.reshape(-1, 12)


def update_axle_pair_crossing_reference(
    *,
    obstacle_x,
    wave_gate,
    wheel_x_from_obstacle,
    wheel_heights,
    previous_phase,
    previous_phase_steps,
    required_height: float,
    past_bar_x: float,
    ramp_steps: int,
    swing_steps: int = 50,
    restore_steps: int = 20,
    support_steps: int = 20,
    curriculum_swing_timeout_steps: int | None = None,
    front_start_x: float = -0.25,
    rear_start_obstacle_x: float = 0.05,
    clearance_target_height: float = 0.20,
    clearance_feedback_gain: float = 8.0,
    clearance_feedback_max: float = 0.8,
    front_lift_hip: float = -0.75,
    front_swing_hip: float = -1.50,
    front_preload_hip: float = -0.375,
    front_preload_knee: float = 0.75,
    front_rear_support_hip: float = 0.375,
    front_rear_support_knee: float = -0.75,
):
    """Lift only the axle at the bar while all four wheels keep rolling."""
    import torch

    x = torch.as_tensor(obstacle_x)
    gate = torch.as_tensor(wave_gate, dtype=torch.bool, device=x.device)
    wheel_x = torch.as_tensor(wheel_x_from_obstacle, dtype=x.dtype, device=x.device)
    heights = torch.as_tensor(wheel_heights, dtype=x.dtype, device=x.device)
    old_phase = torch.as_tensor(previous_phase, dtype=torch.long, device=x.device)
    old_steps = torch.as_tensor(previous_phase_steps, dtype=torch.long, device=x.device)
    front_approach_ready = wheel_x[:, :2].amin(dim=1) >= float(front_start_x)
    phase = torch.where(
        gate & (old_phase < 0) & front_approach_ready,
        torch.zeros_like(old_phase),
        old_phase,
    )

    front_clear = (heights[:, :2] >= float(required_height)).all(dim=1)
    rear_clear = (heights[:, 2:] >= float(required_height)).all(dim=1)
    front_past = (
        (wheel_x[:, :2] >= float(past_bar_x))
        & (heights[:, :2] >= float(required_height))
    ).all(dim=1)
    rear_past = (
        (wheel_x[:, 2:] >= float(past_bar_x))
        & (heights[:, 2:] >= float(required_height))
    ).all(dim=1)
    if curriculum_swing_timeout_steps is None:
        swing_timeout = torch.zeros_like(gate)
    else:
        swing_timeout = old_steps >= int(curriculum_swing_timeout_steps)
    transitions = (
        (phase == 0) & (old_steps >= int(support_steps) + 5) & front_clear,
        (phase == 1)
        & (old_steps >= int(swing_steps))
        & (front_past | swing_timeout),
        (phase == 2)
        & (old_steps >= int(restore_steps))
        & (x <= float(rear_start_obstacle_x)),
        (phase == 3) & (old_steps >= 5) & rear_clear,
        (phase == 4)
        & (old_steps >= int(swing_steps))
        & (rear_past | swing_timeout),
        (phase == 5) & (old_steps >= max(int(restore_steps) - 1, 0)),
    )
    for source_phase, transition in enumerate(transitions):
        phase = torch.where(transition, torch.full_like(phase, source_phase + 1), phase)
    phase = torch.where(gate, phase, torch.full_like(phase, -1))
    changed = phase != old_phase
    phase_steps = torch.where(~gate | changed, torch.zeros_like(old_steps), old_steps + 1)

    base_lift_ramp = torch.clamp(
        (phase_steps.to(x.dtype) + 1.0) / max(int(ramp_steps), 1), 0.0, 1.0
    )
    front_lift_ramp = torch.clamp(
        (phase_steps.to(x.dtype) + 1.0 - float(support_steps))
        / max(int(ramp_steps), 1),
        0.0,
        1.0,
    )
    ramp = torch.where(phase == 0, front_lift_ramp, base_lift_ramp)
    reference = torch.zeros((x.shape[0], 12), dtype=x.dtype, device=x.device)
    swing_blend = torch.clamp(
        phase_steps.to(x.dtype) / max(int(swing_steps), 1), 0.0, 1.0
    )
    front_restore_blend = torch.clamp(
        1.0 - phase_steps.to(x.dtype) / max(int(restore_steps), 1), 0.0, 1.0
    )
    front_lift = phase == 0
    front_swing = phase == 1
    front_restore = phase == 2
    rear_lift = phase == 3
    rear_swing = phase == 4
    rear_restore = phase == 5
    rear_restore_blend = torch.clamp(
        1.0 - phase_steps.to(x.dtype) / max(int(restore_steps), 1), 0.0, 1.0
    )

    reference[:, 1] = torch.where(
        front_lift,
        -1.5 * ramp,
        torch.where(
            front_swing,
            torch.full_like(ramp, -1.5),
            torch.where(front_restore, -1.5 * front_restore_blend, 0.0),
        ),
    )
    reference[:, 4] = torch.where(
        front_lift,
        ramp,
        torch.where(
            front_swing,
            1.0 - 3.0 * swing_blend,
            torch.where(front_restore, -2.0 * front_restore_blend, 0.0),
        ),
    )
    reference[:, 5] = torch.where(
        front_lift,
        -2.0 * ramp,
        torch.where(
            front_swing,
            -2.0 + swing_blend,
            torch.where(front_restore, -1.0 * front_restore_blend, 0.0),
        ),
    )

    reference[:, 7] = torch.where(
        rear_lift,
        1.5 * ramp,
        torch.where(
            rear_swing,
            torch.full_like(ramp, 1.5),
            torch.where(rear_restore, 1.5 * rear_restore_blend, 0.0),
        ),
    )
    reference[:, 10] = torch.where(
        rear_lift,
        -1.0 * ramp,
        torch.where(
            rear_swing,
            -1.0 + 3.0 * swing_blend,
            torch.where(rear_restore, 2.0 * rear_restore_blend, 0.0),
        ),
    )
    reference[:, 11] = torch.where(
        rear_lift,
        2.0 * ramp,
        torch.where(
            rear_swing,
            2.0 - swing_blend,
            torch.where(rear_restore, rear_restore_blend, 0.0),
        ),
    )

    active_mask = torch.zeros_like(reference, dtype=torch.bool)
    front_active = (
        (phase == 0)
        | (phase == 1)
        | ((phase == 2) & (phase_steps < int(restore_steps)))
    )
    rear_active = (
        (phase == 3)
        | (phase == 4)
        | ((phase == 5) & (phase_steps < int(restore_steps)))
    )
    active_mask[:, :6] = front_active.unsqueeze(-1)
    active_mask[:, 6:] = rear_active.unsqueeze(-1)
    drive_allowed = ~(
        (phase == 0) | (phase == 2) | (phase == 3) | (phase == 5)
    )
    return phase, phase_steps, reference, drive_allowed, active_mask


def sequential_wheel_crossing_progress_score(
    *,
    phase,
    wheel_x_from_obstacle,
    wheel_heights,
    baseline_height: float,
    required_height: float,
    swing_start_x: float,
    past_bar_x: float,
):
    """Return dense stage, lift, and forward-swing progress for the active wheel."""
    import torch

    phases = torch.as_tensor(phase, dtype=torch.long)
    wheel_x = torch.as_tensor(wheel_x_from_obstacle)
    heights = torch.as_tensor(
        wheel_heights, dtype=wheel_x.dtype, device=wheel_x.device
    )
    phases = phases.to(device=wheel_x.device)
    selected = torch.full_like(phases, -1)
    for wheel_index, phase_pair in enumerate(((0, 1), (2, 3), (6, 7), (8, 9))):
        selected = torch.where(
            (phases == phase_pair[0]) | (phases == phase_pair[1]),
            torch.full_like(selected, wheel_index),
            selected,
        )
    valid = selected >= 0
    selected_safe = torch.clamp(selected, min=0)
    row = torch.arange(phases.shape[0], device=wheel_x.device)
    selected_x = wheel_x[row, selected_safe]
    selected_height = heights[row, selected_safe]
    lift = torch.clamp(
        (selected_height - float(baseline_height))
        / max(float(required_height) - float(baseline_height), 1.0e-6),
        0.0,
        1.0,
    )
    forward = torch.clamp(
        (selected_x - float(swing_start_x))
        / max(float(past_bar_x) - float(swing_start_x), 1.0e-6),
        0.0,
        1.0,
    )
    swing_phase = (phases == 1) | (phases == 3) | (phases == 7) | (phases == 9)
    local = torch.where(swing_phase, 0.5 * lift + 0.5 * lift * forward, lift)
    local = torch.where(valid, local, torch.zeros_like(local))
    stage = torch.clamp(phases.to(wheel_x.dtype), min=0.0) / 11.0
    return torch.where(phases >= 0, stage + local, torch.zeros_like(local))


def strict_sequential_crossing_success(
    *, phase, root_x, finish_x: float, required_phase: int = 11
):
    """Accept crossing only after every wheel phase and the finish line."""
    import torch

    state = torch.as_tensor(phase, dtype=torch.long)
    position = torch.as_tensor(root_x, device=state.device)
    return (state >= int(required_phase)) & (position >= float(finish_x))


def axle_pair_crossing_progress_score(
    *,
    phase,
    wheel_x_from_obstacle,
    wheel_heights,
    baseline_height: float,
    required_height: float,
    swing_start_x: float,
    past_bar_x: float,
):
    """Reward progress only when both wheels on the active axle clear the bar."""
    import torch

    phases = torch.as_tensor(phase, dtype=torch.long)
    wheel_x = torch.as_tensor(wheel_x_from_obstacle)
    heights = torch.as_tensor(
        wheel_heights, dtype=wheel_x.dtype, device=wheel_x.device
    )
    phases = phases.to(device=wheel_x.device)
    front_active = (phases == 0) | (phases == 1)
    rear_active = (phases == 3) | (phases == 4)
    pair_x = torch.where(
        front_active,
        wheel_x[:, :2].amin(dim=1),
        wheel_x[:, 2:].amin(dim=1),
    )
    pair_height = torch.where(
        front_active,
        heights[:, :2].amin(dim=1),
        heights[:, 2:].amin(dim=1),
    )
    lift = torch.clamp(
        (pair_height - float(baseline_height))
        / max(float(required_height) - float(baseline_height), 1.0e-6),
        0.0,
        1.0,
    )
    forward = torch.clamp(
        (pair_x - float(swing_start_x))
        / max(float(past_bar_x) - float(swing_start_x), 1.0e-6),
        0.0,
        1.0,
    )
    swing = (phases == 1) | (phases == 4)
    local = torch.where(swing, 0.5 * lift + 0.5 * lift * forward, lift)
    active = front_active | rear_active
    local = torch.where(active, local, torch.zeros_like(local))
    score = torch.clamp(phases.to(wheel_x.dtype), min=0.0) + local
    return torch.where(phases >= 0, score, torch.zeros_like(score))


def progress_potential_delta(*, current, previous, reset):
    """Return signed new progress while clearing episode-boundary potential."""
    import torch

    value = torch.as_tensor(current)
    old = torch.as_tensor(previous, dtype=value.dtype, device=value.device)
    reset_mask = torch.as_tensor(reset, dtype=torch.bool, device=value.device)
    delta = torch.where(reset_mask, torch.zeros_like(value), value - old)
    next_potential = torch.where(reset_mask, torch.zeros_like(value), value)
    return delta, next_potential


def wheel_crossbar_collision_mask(
    *,
    wheel_pos_local,
    wheel_contact_force,
    obstacle_center_x: float,
    obstacle_size_x: float,
    obstacle_height: float,
    wheel_radius: float,
    clearance_margin: float,
    contact_force_threshold: float,
    obstacle_center_y: float | None = None,
    obstacle_size_y: float | None = None,
):
    """Return wheels that enter a crossbar's swept volume without enough clearance."""
    import torch

    positions = torch.as_tensor(wheel_pos_local)
    forces = torch.as_tensor(
        wheel_contact_force, dtype=positions.dtype, device=positions.device
    )
    horizontal_limit = 0.5 * float(obstacle_size_x) + float(wheel_radius)
    required_height = (
        float(obstacle_height) + float(wheel_radius) + float(clearance_margin)
    )
    in_swept_region = (
        torch.abs(positions[..., 0] - float(obstacle_center_x))
        <= horizontal_limit
    )
    if obstacle_center_y is not None and obstacle_size_y is not None:
        lateral_limit = 0.5 * float(obstacle_size_y) + float(wheel_radius)
        in_swept_region &= (
            torch.abs(positions[..., 1] - float(obstacle_center_y))
            <= lateral_limit
        )
    insufficient_clearance = positions[..., 2] < required_height
    loaded = forces > float(contact_force_threshold)
    return in_swept_region & insufficient_clearance & loaded


def smooth_wave_reference_actions(*, previous, target, active, alpha: float):
    """Low-pass a wave teacher target without clipping its steady-state amplitude."""
    import torch

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    target_tensor = torch.as_tensor(target)
    previous_tensor = torch.as_tensor(
        previous, dtype=target_tensor.dtype, device=target_tensor.device
    )
    active_mask = torch.as_tensor(
        active, dtype=torch.bool, device=target_tensor.device
    ).reshape(-1, 1)
    smoothed = previous_tensor + float(alpha) * (target_tensor - previous_tensor)
    return torch.where(active_mask, smoothed, torch.zeros_like(smoothed))


def build_teacher_student_residual(
    *, student_actions, teacher_actions, student_weight: float
):
    """Return the residual that makes teacher+residual a DAgger rollout blend."""
    import torch

    weight = float(student_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("student_weight must be in [0, 1]")
    student = torch.as_tensor(student_actions)
    teacher = torch.as_tensor(
        teacher_actions, dtype=student.dtype, device=student.device
    )
    return weight * (student - teacher)


def update_wave_encounter_phase(
    *,
    obstacle_active,
    previous_active,
    elapsed_s,
    step_dt: float,
    minimum_duration_s: float,
    maximum_duration_s: float | None = None,
):
    """Start wave phase on obstacle entry and hold its gate through one full cycle."""
    import torch

    active = torch.as_tensor(obstacle_active, dtype=torch.bool)
    previous = torch.as_tensor(previous_active, dtype=torch.bool, device=active.device)
    elapsed = torch.as_tensor(elapsed_s, device=active.device).clone()
    rising = active & ~previous
    elapsed = torch.where(rising, torch.zeros_like(elapsed), elapsed)
    advancing = (elapsed >= 0.0) & ~rising
    elapsed = torch.where(advancing, elapsed + step_dt, elapsed)
    gate = active | ((elapsed >= 0.0) & (elapsed < minimum_duration_s))
    if maximum_duration_s is not None:
        gate = gate & (elapsed < float(maximum_duration_s))
    return elapsed, active.clone(), gate


def prepare_wave_checkpoint(checkpoint: dict, leg_noise_std: float | None = None) -> dict:
    """Zero the unused M1 leg actor rows while preserving wheel outputs."""
    prepared = deepcopy(checkpoint)
    state = prepared.get("model_state_dict", {})
    weight = state.get("actor.4.weight")
    bias = state.get("actor.4.bias")
    if weight is None or bias is None or weight.shape[0] != 16 or bias.shape[0] != 16:
        raise ValueError("Expected a 16-output actor.4 layer in the M1 checkpoint")
    weight[:12].zero_()
    bias[:12].zero_()
    if leg_noise_std is not None:
        std = state.get("std")
        if std is None or std.shape[0] != 16:
            raise ValueError("Expected a 16-output std tensor in the M1 checkpoint")
        std[:12].fill_(leg_noise_std)
    return prepared


def expand_checkpoint_observations(checkpoint: dict, new_observation_dim: int) -> dict:
    """Expand actor and critic first-layer inputs with zero-initialized columns."""
    import torch

    prepared = deepcopy(checkpoint)
    state = prepared.get("model_state_dict", {})
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = state.get(key)
        if weight is None or weight.ndim != 2:
            raise ValueError(f"Missing two-dimensional checkpoint tensor: {key}")
        old_dim = int(weight.shape[1])
        if new_observation_dim < old_dim:
            raise ValueError(f"Cannot shrink {key} from {old_dim} to {new_observation_dim}")
        expanded = torch.zeros(
            (weight.shape[0], new_observation_dim), dtype=weight.dtype, device=weight.device
        )
        expanded[:, :old_dim] = weight
        state[key] = expanded
    return prepared


def build_wave_reference_actions(
    *,
    episode_time_s,
    amplitude: float,
    knee_ratio: float,
    frequency: float,
    rear_amplitude_scale: float = 1.0,
    front_support_ratio: float = 0.0,
    rear_support_ratio: float = 0.0,
    pulse_ramp_s: float | None = None,
    pulse_hold_s: float | None = None,
):
    """Build raw 12-leg-action diagonal wave references for each environment."""
    import torch

    times = torch.as_tensor(episode_time_s)
    actions = torch.zeros((times.numel(), 12), dtype=times.dtype, device=times.device)
    if pulse_ramp_s is None and pulse_hold_s is None:
        front_pulse = torch.clamp(
            torch.sin(2.0 * math.pi * float(frequency) * times), min=0.0
        )
        rear_pulse = torch.clamp(
            torch.sin(2.0 * math.pi * (float(frequency) * times + 0.5)), min=0.0
        )
    else:
        if pulse_ramp_s is None or pulse_hold_s is None:
            raise ValueError("pulse_ramp_s and pulse_hold_s must be configured together")
        ramp = float(pulse_ramp_s)
        hold = float(pulse_hold_s)
        if ramp <= 0.0 or hold < 0.0 or frequency <= 0.0:
            raise ValueError("trapezoid pulse timing and frequency must be positive")
        period = 1.0 / float(frequency)

        def trapezoid(local_time):
            rising = torch.clamp(local_time / ramp, 0.0, 1.0)
            falling = torch.clamp((2.0 * ramp + hold - local_time) / ramp, 0.0, 1.0)
            return torch.minimum(rising, falling)

        front_pulse = trapezoid(torch.remainder(times, period))
        rear_pulse = trapezoid(torch.remainder(times - 0.5 * period, period))
    front_lift = float(amplitude) * front_pulse
    rear_lift = float(amplitude) * float(rear_amplitude_scale) * rear_pulse
    front_command = front_lift - float(front_support_ratio) * rear_lift
    rear_command = rear_lift + float(rear_support_ratio) * front_lift
    for base in (0, 3):
        actions[:, base + 1] = front_command
        actions[:, base + 2] = float(knee_ratio) * front_command
    for base in (6, 9):
        actions[:, base + 1] = -rear_command
        actions[:, base + 2] = -float(knee_ratio) * rear_command
    return actions


def build_semantic_spatial_wave_reference(
    *,
    root_pos_w,
    root_quat_w,
    ray_hits_w,
    semantic_map,
    amplitude: float,
    knee_ratio: float,
    corridor_half_width_m: float = 0.25,
    rear_amplitude_scale: float = 1.0,
    front_overlap_scale: float = 1.0,
    front_support_ratio: float = 0.0,
    rear_support_ratio: float = 0.0,
    front_lift_window: tuple[float, float, float, float] = (0.80, 0.65, 0.05, -0.10),
    rear_lift_window: tuple[float, float, float, float] = (0.15, 0.0, -0.40, -0.60),
):
    """Build front/rear lift references from obstacle position along the M1 body."""
    import torch

    root_pos = torch.as_tensor(root_pos_w)
    quat = torch.as_tensor(root_quat_w, dtype=root_pos.dtype, device=root_pos.device)
    hits = torch.as_tensor(ray_hits_w, dtype=root_pos.dtype, device=root_pos.device).reshape(
        root_pos.shape[0], -1, 3
    )
    semantic = torch.as_tensor(semantic_map, device=root_pos.device).reshape(root_pos.shape[0], -1)
    w, x, y, z = quat.unbind(dim=-1)
    heading = torch.stack(
        (1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z)), dim=-1
    )
    heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1.0e-6)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    delta = hits[..., :2] - root_pos[:, None, :2]
    along = (delta * heading[:, None, :]).sum(dim=-1)
    lateral = (delta * left[:, None, :]).sum(dim=-1)
    selectable = (
        (semantic == 1)
        & torch.isfinite(hits).all(dim=-1)
        & (along >= -0.60)
        & (along <= 0.70)
        & (torch.abs(lateral) <= float(corridor_half_width_m))
    )
    lower = torch.where(selectable, along, torch.full_like(along, float("inf"))).amin(dim=1)
    upper = torch.where(selectable, along, torch.full_like(along, -float("inf"))).amax(dim=1)
    active = selectable.any(dim=1)
    obstacle_x = torch.where(active, 0.5 * (lower + upper), torch.zeros_like(lower))

    def descending_trapezoid(position, window):
        start, full_start, full_end, end = (float(value) for value in window)
        if not start > full_start >= full_end > end:
            raise ValueError(
                "spatial lift windows must satisfy start > full_start >= full_end > end"
            )
        rising = torch.clamp(
            (start - position) / (start - full_start), 0.0, 1.0
        )
        falling = torch.clamp(
            (position - end) / (full_end - end), 0.0, 1.0
        )
        return torch.minimum(rising, falling)

    front_lift = descending_trapezoid(obstacle_x, front_lift_window)
    rear_lift = descending_trapezoid(obstacle_x, rear_lift_window)
    front_lift = torch.clamp(
        front_lift - float(front_overlap_scale) * rear_lift, min=0.0
    )
    front_lift = front_lift * active.to(root_pos.dtype) * float(amplitude)
    rear_lift = (
        rear_lift * active.to(root_pos.dtype) * float(amplitude) * float(rear_amplitude_scale)
    )
    front_command = front_lift - float(front_support_ratio) * rear_lift
    rear_command = rear_lift + float(rear_support_ratio) * front_lift
    actions = torch.zeros((root_pos.shape[0], 12), dtype=root_pos.dtype, device=root_pos.device)
    for base in (0, 3):
        actions[:, base + 1] = front_command
        actions[:, base + 2] = float(knee_ratio) * front_command
    for base in (6, 9):
        actions[:, base + 1] = -rear_command
        actions[:, base + 2] = -float(knee_ratio) * rear_command
    return actions, active, obstacle_x


def build_spatial_axle_wheel_targets(
    *, obstacle_x, active, base_action: float, assist_action: float
):
    """Shift drive assist to the axle opposite the obstacle-crossing axle."""
    import torch

    x = torch.as_tensor(obstacle_x)
    active_scale = torch.as_tensor(
        active, dtype=x.dtype, device=x.device
    )
    front_phase = torch.minimum(
        torch.clamp((x + 0.10) / 0.15, 0.0, 1.0),
        torch.clamp((0.80 - x) / 0.15, 0.0, 1.0),
    ) * active_scale
    rear_phase = torch.minimum(
        torch.clamp((x + 0.60) / 0.20, 0.0, 1.0),
        torch.clamp((0.15 - x) / 0.15, 0.0, 1.0),
    ) * active_scale
    front_target = float(base_action) + float(assist_action) * rear_phase
    rear_target = float(base_action) + float(assist_action) * front_phase
    return torch.stack(
        (front_target, front_target, rear_target, rear_target), dim=1
    )


def build_temporal_axle_wheel_targets(
    *,
    episode_time_s,
    active,
    frequency: float,
    base_action: float,
    assist_action: float,
):
    """Move drive assist to the axle supporting the active temporal wave pulse."""
    import torch

    times = torch.as_tensor(episode_time_s)
    active_scale = torch.as_tensor(
        active, dtype=times.dtype, device=times.device
    )
    front_pulse = torch.clamp(
        torch.sin(2.0 * math.pi * float(frequency) * times), min=0.0
    ) * active_scale
    rear_pulse = torch.clamp(
        torch.sin(2.0 * math.pi * (float(frequency) * times + 0.5)), min=0.0
    ) * active_scale
    front_target = float(base_action) + float(assist_action) * rear_pulse
    rear_target = float(base_action) + float(assist_action) * front_pulse
    return torch.stack(
        (front_target, front_target, rear_target, rear_target), dim=1
    )


def build_lateral_steering_correction(
    *,
    lateral_y,
    yaw_rate,
    lateral_gain: float,
    yaw_damping_gain: float,
    max_correction: float,
):
    """Return right/left wheel corrections that steer toward the course center."""
    import torch

    y = torch.as_tensor(lateral_y)
    yaw = torch.as_tensor(yaw_rate, dtype=y.dtype, device=y.device)
    differential = -float(lateral_gain) * y - float(yaw_damping_gain) * yaw
    differential = torch.clamp(
        differential, -float(max_correction), float(max_correction)
    )
    return torch.stack(
        (-differential, differential, -differential, differential), dim=1
    )


def spatial_pair_lift_score(
    *, wheel_heights, obstacle_x, active, baseline_height: float, target_height: float
):
    """Score clearance using the lower wheel of the pair that is crossing now."""
    import torch

    heights = torch.as_tensor(wheel_heights)
    x = torch.as_tensor(obstacle_x, dtype=heights.dtype, device=heights.device)
    active_mask = torch.as_tensor(active, dtype=torch.bool, device=heights.device)
    if heights.ndim != 2 or heights.shape[1] != 4:
        raise ValueError(f"wheel_heights must have shape [N,4], got {tuple(heights.shape)}")
    denominator = max(float(target_height) - float(baseline_height), 1.0e-6)
    pair_lift = torch.clamp(
        (torch.stack((heights[:, :2].amin(dim=1), heights[:, 2:].amin(dim=1)), dim=1)
         - float(baseline_height))
        / denominator,
        0.0,
        1.0,
    )
    front_factor = torch.minimum(
        torch.clamp((x + 0.10) / 0.15, 0.0, 1.0),
        torch.clamp((0.80 - x) / 0.15, 0.0, 1.0),
    )
    rear_factor = torch.minimum(
        torch.clamp((x + 0.60) / 0.20, 0.0, 1.0),
        torch.clamp((0.15 - x) / 0.15, 0.0, 1.0),
    )
    factors = torch.stack((front_factor, rear_factor), dim=1)
    score = (pair_lift * factors).sum(dim=1) / factors.sum(dim=1).clamp_min(1.0e-6)
    return score * active_mask.to(dtype=heights.dtype)


def spatial_wheel_lift_score(
    *, wheel_heights, obstacle_x, active, baseline_height: float, target_height: float
):
    """Score each wheel independently on the axle currently crossing an obstacle."""
    import torch

    heights = torch.as_tensor(wheel_heights)
    x = torch.as_tensor(obstacle_x, dtype=heights.dtype, device=heights.device)
    active_mask = torch.as_tensor(active, dtype=torch.bool, device=heights.device)
    if heights.ndim != 2 or heights.shape[1] != 4:
        raise ValueError(f"wheel_heights must have shape [N,4], got {tuple(heights.shape)}")
    denominator = max(float(target_height) - float(baseline_height), 1.0e-6)
    wheel_lift = torch.clamp(
        (heights - float(baseline_height)) / denominator, 0.0, 1.0
    )
    axle_lift = torch.stack(
        (wheel_lift[:, :2].mean(dim=1), wheel_lift[:, 2:].mean(dim=1)), dim=1
    )
    front_factor = torch.minimum(
        torch.clamp((x + 0.10) / 0.15, 0.0, 1.0),
        torch.clamp((0.80 - x) / 0.15, 0.0, 1.0),
    )
    rear_factor = torch.minimum(
        torch.clamp((x + 0.60) / 0.20, 0.0, 1.0),
        torch.clamp((0.15 - x) / 0.15, 0.0, 1.0),
    )
    factors = torch.stack((front_factor, rear_factor), dim=1)
    score = (axle_lift * factors).sum(dim=1) / factors.sum(dim=1).clamp_min(1.0e-6)
    return score * active_mask.to(dtype=heights.dtype)


def update_semantic_crossing_tracker(
    *,
    root_pos_w,
    root_quat_w,
    ray_hits_w,
    semantic_map,
    candidate_xy,
    candidate_heading,
    candidate_valid,
    crossed,
    approach_min_m: float = 0.15,
    approach_max_m: float = 0.75,
    corridor_half_width_m: float = 0.25,
    pass_margin_m: float = 0.38,
):
    """Track straight-over semantic small-obstacle crossings without counting avoidance."""
    import torch

    root_pos = torch.as_tensor(root_pos_w)
    quat = torch.as_tensor(root_quat_w, dtype=root_pos.dtype, device=root_pos.device)
    hits = torch.as_tensor(ray_hits_w, dtype=root_pos.dtype, device=root_pos.device).reshape(
        root_pos.shape[0], -1, 3
    )
    semantic = torch.as_tensor(semantic_map, device=root_pos.device).reshape(root_pos.shape[0], -1)
    candidate = torch.as_tensor(candidate_xy, dtype=root_pos.dtype, device=root_pos.device).clone()
    stored_heading = torch.as_tensor(
        candidate_heading, dtype=root_pos.dtype, device=root_pos.device
    ).clone()
    valid = torch.as_tensor(candidate_valid, dtype=torch.bool, device=root_pos.device).clone()
    crossed_out = torch.as_tensor(crossed, dtype=torch.bool, device=root_pos.device).clone()

    w, x, y, z = quat.unbind(dim=-1)
    current_heading = torch.stack(
        (1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z)), dim=-1
    )
    current_heading = current_heading / torch.linalg.vector_norm(
        current_heading, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    current_left = torch.stack((-current_heading[:, 1], current_heading[:, 0]), dim=-1)
    ray_delta = hits[..., :2] - root_pos[:, None, :2]
    ray_along = (ray_delta * current_heading[:, None, :]).sum(dim=-1)
    ray_lateral = (ray_delta * current_left[:, None, :]).sum(dim=-1)
    selectable = (
        (semantic == 1)
        & torch.isfinite(hits).all(dim=-1)
        & (ray_along >= float(approach_min_m))
        & (ray_along <= float(approach_max_m))
        & (torch.abs(ray_lateral) <= float(corridor_half_width_m))
    )
    score = torch.where(selectable, ray_along, torch.full_like(ray_along, float("inf")))
    has_candidate = selectable.any(dim=1) & ~valid
    nearest_surface = score.amin(dim=1)
    nearest_cluster = selectable & (ray_along <= nearest_surface[:, None] + 0.15)
    cluster_weight = nearest_cluster.to(dtype=root_pos.dtype)
    cluster_center_xy = (
        torch.where(nearest_cluster[..., None], hits[..., :2], torch.zeros_like(hits[..., :2])).sum(dim=1)
        / cluster_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    )
    candidate = torch.where(has_candidate[:, None], cluster_center_xy, candidate)
    stored_heading = torch.where(has_candidate[:, None], current_heading, stored_heading)
    valid |= has_candidate

    stored_left = torch.stack((-stored_heading[:, 1], stored_heading[:, 0]), dim=-1)
    root_from_candidate = root_pos[:, :2] - candidate
    passed_along = (root_from_candidate * stored_heading).sum(dim=-1)
    passed_lateral = torch.abs((root_from_candidate * stored_left).sum(dim=-1))
    crossed_out |= (
        valid
        & (passed_along >= float(pass_margin_m))
        & (passed_lateral <= float(corridor_half_width_m))
    )
    return candidate, stored_heading, valid, crossed_out


def semantic_obstacle_ahead_mask(
    *,
    root_pos_w,
    root_quat_w,
    ray_hits_w,
    semantic_map,
    approach_min_m: float = 0.10,
    approach_max_m: float = 0.80,
    corridor_half_width_m: float = 0.25,
):
    """Return environments with a semantic small obstacle in the forward corridor."""
    import torch

    root_pos = torch.as_tensor(root_pos_w)
    quat = torch.as_tensor(root_quat_w, dtype=root_pos.dtype, device=root_pos.device)
    hits = torch.as_tensor(ray_hits_w, dtype=root_pos.dtype, device=root_pos.device).reshape(
        root_pos.shape[0], -1, 3
    )
    semantic = torch.as_tensor(semantic_map, device=root_pos.device).reshape(root_pos.shape[0], -1)
    w, x, y, z = quat.unbind(dim=-1)
    heading = torch.stack(
        (1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z)), dim=-1
    )
    heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1.0e-6)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    delta = hits[..., :2] - root_pos[:, None, :2]
    along = (delta * heading[:, None, :]).sum(dim=-1)
    lateral = (delta * left[:, None, :]).sum(dim=-1)
    return (
        (semantic == 1)
        & torch.isfinite(hits).all(dim=-1)
        & (along >= float(approach_min_m))
        & (along <= float(approach_max_m))
        & (torch.abs(lateral) <= float(corridor_half_width_m))
    ).any(dim=1)


def discover_latest_checkpoint(directory: str | Path) -> Path:
    """Return the checkpoint with the largest numeric iteration suffix."""
    candidates: list[tuple[int, Path]] = []
    for path in Path(directory).glob("model_*.pt"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {directory}")
    return max(candidates, key=lambda item: item[0])[1]


def evaluate_roll_gate(
    *,
    episodes: int,
    timeout_episodes: int,
    bad_orientation_episodes: int,
    mean_dx: float,
    mean_dy: float,
    max_tilt_rad: float,
) -> dict[str, float | int | bool]:
    """Evaluate the deterministic Stage 1 rolling acceptance gate."""
    finite = all(math.isfinite(value) for value in (mean_dx, mean_dy, max_tilt_rad))
    timeout_rate = timeout_episodes / episodes if episodes > 0 else 0.0
    forward_progress = finite and mean_dx >= 0.15
    lateral_drift_ok = forward_progress and abs(mean_dy) <= 0.25 * mean_dx
    tilt_ok = finite and max_tilt_rad <= 0.20
    orientation_ok = bad_orientation_episodes == 0
    passed = bool(
        episodes > 0
        and timeout_rate >= 0.95
        and forward_progress
        and lateral_drift_ok
        and tilt_ok
        and orientation_ok
    )
    return {
        "passed": passed,
        "episodes": episodes,
        "timeout_episodes": timeout_episodes,
        "timeout_rate": timeout_rate,
        "bad_orientation_episodes": bad_orientation_episodes,
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "max_tilt_rad": max_tilt_rad,
        "forward_progress": forward_progress,
        "lateral_drift_ok": lateral_drift_ok,
        "tilt_ok": tilt_ok,
        "orientation_ok": orientation_ok,
    }


def evaluate_obstacle_gate(
    *,
    episodes: int,
    timeout_episodes: int,
    bad_orientation_episodes: int,
    crossing_episodes: int,
    mean_max_dx: float,
    mean_dy: float,
    max_tilt_rad: float,
    max_tilt_limit_rad: float = 0.35,
) -> dict[str, float | int | bool]:
    """Evaluate deterministic whole-robot obstacle clearance."""
    finite = all(math.isfinite(value) for value in (mean_max_dx, mean_dy, max_tilt_rad))
    timeout_rate = timeout_episodes / episodes if episodes > 0 else 0.0
    crossing_rate = crossing_episodes / episodes if episodes > 0 else 0.0
    crossing_ok = finite and crossing_rate >= 0.95
    lateral_drift_ok = crossing_ok and abs(mean_dy) <= 0.20 * mean_max_dx
    tilt_ok = finite and max_tilt_rad <= max_tilt_limit_rad
    orientation_ok = bad_orientation_episodes == 0
    passed = bool(
        episodes > 0
        and crossing_ok
        and lateral_drift_ok
        and tilt_ok
        and orientation_ok
    )
    return {
        "passed": passed,
        "episodes": episodes,
        "timeout_episodes": timeout_episodes,
        "timeout_rate": timeout_rate,
        "bad_orientation_episodes": bad_orientation_episodes,
        "crossing_episodes": crossing_episodes,
        "crossing_rate": crossing_rate,
        "mean_max_dx": mean_max_dx,
        "mean_dy": mean_dy,
        "max_tilt_rad": max_tilt_rad,
        "max_tilt_limit_rad": max_tilt_limit_rad,
        "crossing_ok": crossing_ok,
        "lateral_drift_ok": lateral_drift_ok,
        "tilt_ok": tilt_ok,
        "orientation_ok": orientation_ok,
    }
