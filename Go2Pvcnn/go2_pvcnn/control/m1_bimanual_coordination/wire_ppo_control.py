"""Policy-owned task references; low-level MPC/QP enforces feasibility only."""

from dataclasses import replace

import torch

from .contracts import BimanualPhase
from .object_mpc import ObjectMpcSolution, ObjectMpcDiagnostics, OBJECT_MPC_HORIZON_STEPS
from .wire_ppo_contracts import decode_wire_latent


def ppo_hand_reference(q, action, side):
    """Three synergies per side: thumb pitch, thumb yaw, four-finger flexion.

    Increment bounds are per physics step; both the O6 mechanical limits and
    the live Isaac limits at submission constrain the resulting reference.
    """
    values = action[6:9] if side == "left" else action[9:12]
    delta = values[[0, 1, 2, 2, 2, 2]].to(q) * .005
    lower = q.new_zeros(6)
    upper = q.new_tensor([.58, 1.36, 1.6, 1.6, 1.6, 1.6])
    return torch.maximum(lower, torch.minimum(upper, q + delta))


def ppo_phase(action):
    phases = (BimanualPhase.APPROACH, BimanualPhase.PRELOAD, BimanualPhase.GRASP,
              BimanualPhase.LIFT, BimanualPhase.HOLD)
    index = min(4, int((float(action[12]) + 1.) * 2.5))
    return phases[index]


def project_ppo_hand_reference(controller, sample, action, side):
    """Use the existing compliant contact QP to project a policy joint goal.

    The same force friction cone, contact compliance, position and speed
    constraints apply in every phase. No automatic PRELOAD closing baseline
    is mixed into the PPO reference.
    """
    from .hand_mpc import O6HandMpc, build_hand_contact_qp
    from go2_pvcnn.control.m1_panda_coordination.qp_backend import solve_reference_qp
    if not isinstance(controller, O6HandMpc):
        return controller.plan(sample)
    reference = ppo_hand_reference(sample.q, action, side)
    rate = (reference - sample.q) / controller.cfg.dt
    problem = build_hand_contact_qp(sample, controller.cfg)
    hessian, gradient = problem.hessian.clone(), problem.gradient.clone()
    weight = controller.cfg.precontact_tracking_weight
    hessian[:6, :6] += 2. * weight * torch.eye(6, dtype=torch.float64)
    gradient[:6] -= 2. * weight * rate
    result = solve_reference_qp(replace(problem, hessian=hessian, gradient=gradient),
        tolerance=controller.cfg.qp_tolerance, max_iterations=controller.cfg.qp_max_iterations)
    if not result.success or not torch.isfinite(result.solution).all():
        return controller._fallback(sample, "ppo_hand_projection_infeasible")
    return controller._accept(controller._solution(sample, result))


def ppo_object_reference(snapshot, action):
    """6D common palm twist with independent force split and no teacher plan."""
    command = decode_wire_latent(action, device="cpu", dtype=torch.float64)
    horizon = OBJECT_MPC_HORIZON_STEPS
    delta = command.palm_delta * torch.tensor([.025, .025, .025, .08, .08, .08], dtype=torch.float64)
    ramp = torch.linspace(1. / horizon, 1., horizon, dtype=torch.float64)[:, None]
    left = snapshot.left_arm.palm_pose_b[None] + ramp * delta
    right = snapshot.right_arm.palm_pose_b[None] + ramp * delta
    # Split a bounded normal force budget across opposing measured palm axes.
    total = 2. * (float(command.phase_force[1]) + 1.)
    fraction = .5 + .4 * float(command.phase_force[2])
    axis = snapshot.right_arm.palm_pose_b[:3] - snapshot.left_arm.palm_pose_b[:3]
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
    wrench_l = torch.zeros((horizon, 6), dtype=torch.float64)
    wrench_r = wrench_l.clone()
    wrench_l[:, :3] = total * fraction * axis
    wrench_r[:, :3] = -total * (1. - fraction) * axis
    margins = []
    for hand in (snapshot.left_hand, snapshot.right_hand):
        active = torch.linalg.vector_norm(hand.fingertip_forces_b[hand.contact_mask], dim=1)
        margins.append(max(0., float(active.min()) - .2)
            if hand.contact_consistent and active.numel() >= 2 else 0.)
    return ObjectMpcSolution(
        box_pose=snapshot.box.pose_b.repeat(horizon, 1),
        box_twist=snapshot.box.twist_b.repeat(horizon, 1),
        platform_yaw=snapshot.platform_q_qd[0].repeat(horizon),
        left_palm_pose=left, right_palm_pose=right,
        left_wrench=wrench_l, right_wrench=wrench_r,
        # These are a policy proposal. Arm, hand, WBC, then SafetyProjection
        # must all accept it; this flag is not a physical success certificate.
        diagnostics=ObjectMpcDiagnostics(True, False, None, min(margins), 0., 0),
    )
