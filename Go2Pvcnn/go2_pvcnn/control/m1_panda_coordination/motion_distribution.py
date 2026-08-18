"""Bounded prioritized motion distribution for the M1 base and Panda arm."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

import torch

from .constraints import compute_velocity_bounds
from .contracts import COORD_DOF, require_tensor
from .kinematics import damped_pseudoinverse


@dataclass(frozen=True)
class MotionDistributionCfg:
    pose_gain: float = 10.0
    damping: float = 1.0e-4
    singularity_threshold: float = 0.1
    null_gain: float = 5.0
    null_damping: float = 0.5
    max_saturation_passes: int = 10


@dataclass(frozen=True)
class MotionDistributionResult:
    qd_coord: torch.Tensor
    base_active: torch.Tensor
    sigma_min: torch.Tensor
    phi: torch.Tensor
    psi: torch.Tensor
    saturated: torch.Tensor


@dataclass(frozen=True)
class _BoundedSolve:
    velocity: torch.Tensor
    saturated: torch.Tensor
    feasible: bool


def _solve_bounded_task(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    allowed: torch.Tensor,
    *,
    damping: float,
    max_passes: int,
) -> _BoundedSolve:
    velocity = torch.zeros_like(lower).clamp(min=lower, max=upper)
    saturated = torch.zeros_like(allowed)
    fixed = ~allowed

    for _ in range(max_passes):
        free = allowed & ~saturated
        fixed_now = ~free
        residual = target - jacobian[:, fixed_now] @ velocity[fixed_now]
        if free.any().item():
            velocity[free] = damped_pseudoinverse(
                jacobian[:, free], damping=damping
            ) @ residual

        below = free & (velocity < lower)
        above = free & (velocity > upper)
        violated = below | above
        if not violated.any().item():
            break

        violation = torch.maximum(lower - velocity, velocity - upper)
        violation = torch.where(
            violated,
            violation,
            torch.full_like(violation, -1.0),
        )
        index = int(torch.argmax(violation).item())
        velocity[index] = torch.clamp(velocity[index], lower[index], upper[index])
        saturated[index] = True

    velocity = torch.clamp(velocity, min=lower, max=upper)
    residual_norm = torch.linalg.vector_norm(jacobian @ velocity - target)
    tolerance = 1.0e-6 + 1.0e-5 * torch.linalg.vector_norm(target)
    feasible = bool((residual_norm <= tolerance).item())
    saturated = saturated | torch.isclose(velocity, lower, atol=1.0e-9, rtol=0.0)
    saturated = saturated | torch.isclose(velocity, upper, atol=1.0e-9, rtol=0.0)
    saturated = saturated & ~fixed
    return _BoundedSolve(velocity=velocity, saturated=saturated, feasible=feasible)


def _maximum_null_scale(
    velocity: torch.Tensor,
    direction: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    positive = direction > 0.0
    negative = direction < 0.0
    candidates = torch.ones_like(direction)
    candidates = torch.where(
        positive,
        (upper - velocity) / torch.where(positive, direction, torch.ones_like(direction)),
        candidates,
    )
    candidates = torch.where(
        negative,
        (lower - velocity) / torch.where(negative, direction, torch.ones_like(direction)),
        candidates,
    )
    return candidates.min().clamp(0.0, 1.0)


def _distribute_single(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    gradient: torch.Tensor,
    sigma_min: torch.Tensor,
    cfg: MotionDistributionCfg,
) -> tuple[torch.Tensor, bool, torch.Tensor, torch.Tensor, torch.Tensor]:
    arm_allowed = torch.zeros(COORD_DOF, dtype=torch.bool, device=jacobian.device)
    arm_allowed[3:] = True
    all_allowed = torch.ones_like(arm_allowed)

    arm_rank = int(torch.linalg.matrix_rank(jacobian[:, 3:]).item())
    zero_base_unreachable = bool(
        ((lower[:3] > 0.0) | (upper[:3] < 0.0)).any().item()
    )
    base_active = bool(
        arm_rank < jacobian.shape[-2]
        or sigma_min.item() < cfg.singularity_threshold
        or zero_base_unreachable
    )
    solve = _solve_bounded_task(
        jacobian,
        target,
        lower,
        upper,
        all_allowed if base_active else arm_allowed,
        damping=cfg.damping,
        max_passes=cfg.max_saturation_passes,
    )
    if not solve.feasible and not base_active:
        base_active = True
        solve = _solve_bounded_task(
            jacobian,
            target,
            lower,
            upper,
            all_allowed,
            damping=cfg.damping,
            max_passes=cfg.max_saturation_passes,
        )

    phi = target.new_tensor(1.0)
    psi = target.new_tensor(1.0)
    if not solve.feasible:
        psi = target.new_tensor(0.0)
        low = 0.0
        high = 1.0
        best = _solve_bounded_task(
            jacobian,
            target * 0.0,
            lower,
            upper,
            all_allowed,
            damping=cfg.damping,
            max_passes=cfg.max_saturation_passes,
        )
        for _ in range(24):
            middle = 0.5 * (low + high)
            candidate = _solve_bounded_task(
                jacobian,
                target * middle,
                lower,
                upper,
                all_allowed,
                damping=cfg.damping,
                max_passes=cfg.max_saturation_passes,
            )
            if candidate.feasible:
                low = middle
                best = candidate
            else:
                high = middle
        phi = target.new_tensor(low)
        solve = best

    velocity = solve.velocity
    saturated = solve.saturated
    if phi.item() == 1.0:
        allowed = all_allowed if base_active else arm_allowed
        active_jacobian = jacobian[:, allowed]
        projector = torch.eye(
            int(allowed.sum().item()), dtype=jacobian.dtype, device=jacobian.device
        ) - damped_pseudoinverse(active_jacobian, damping=cfg.damping) @ active_jacobian
        projected = projector @ gradient[allowed]
        projected_norm = torch.linalg.vector_norm(projected)
        direction_active = (
            cfg.null_gain * projected / (1.0 + cfg.null_damping * projected_norm)
        )
        direction = torch.zeros_like(velocity)
        direction[allowed] = direction_active
        psi = _maximum_null_scale(velocity, direction, lower, upper)
        velocity = velocity + psi * direction
        velocity = torch.clamp(velocity, min=lower, max=upper)
        saturated = saturated | torch.isclose(
            velocity, lower, atol=1.0e-9, rtol=0.0
        )
        saturated = saturated | torch.isclose(
            velocity, upper, atol=1.0e-9, rtol=0.0
        )
    return velocity, base_active, phi, psi, saturated


def _validate_inputs(
    coordinated_jacobian: torch.Tensor,
    pose_error: torch.Tensor,
    desired_twist: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    v_max: torch.Tensor,
    a_max: torch.Tensor,
    manipulability_gradient: torch.Tensor,
    sigma_min: torch.Tensor,
) -> tuple[int, ...]:
    require_tensor(
        "coordinated_jacobian", coordinated_jacobian, trailing_shape=(6, COORD_DOF)
    )
    batch_shape = coordinated_jacobian.shape[:-2]
    for name, value, trailing_shape in (
        ("pose_error", pose_error, (6,)),
        ("desired_twist", desired_twist, (6,)),
        ("q", q, (COORD_DOF,)),
        ("qd", qd, (COORD_DOF,)),
        ("q_min", q_min, (COORD_DOF,)),
        ("q_max", q_max, (COORD_DOF,)),
        ("v_max", v_max, (COORD_DOF,)),
        ("a_max", a_max, (COORD_DOF,)),
        ("manipulability_gradient", manipulability_gradient, (COORD_DOF,)),
    ):
        require_tensor(name, value, trailing_shape=trailing_shape)
        if value.shape[: -len(trailing_shape)] != batch_shape:
            raise ValueError(f"{name} batch dimensions must match coordinated_jacobian")
        if value.dtype != coordinated_jacobian.dtype:
            raise TypeError(f"{name} dtype must match coordinated_jacobian")
        if value.device != coordinated_jacobian.device:
            raise ValueError(f"{name} device must match coordinated_jacobian")
    require_tensor("sigma_min", sigma_min, trailing_shape=())
    if sigma_min.shape != batch_shape:
        raise ValueError("sigma_min batch dimensions must match coordinated_jacobian")
    if sigma_min.dtype != coordinated_jacobian.dtype:
        raise TypeError("sigma_min dtype must match coordinated_jacobian")
    if sigma_min.device != coordinated_jacobian.device:
        raise ValueError("sigma_min device must match coordinated_jacobian")
    return batch_shape


def distribute_motion(
    coordinated_jacobian: torch.Tensor,
    pose_error: torch.Tensor,
    desired_twist: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    v_max: torch.Tensor,
    a_max: torch.Tensor,
    manipulability_gradient: torch.Tensor,
    sigma_min: torch.Tensor,
    dt: float,
    cfg: MotionDistributionCfg | None = None,
    prescribed_base_velocity: torch.Tensor | None = None,
) -> MotionDistributionResult:
    """Distribute a Cartesian target with optional fixed M1 planar velocity."""

    cfg = cfg or MotionDistributionCfg()
    batch_shape = _validate_inputs(
        coordinated_jacobian,
        pose_error,
        desired_twist,
        q,
        qd,
        q_min,
        q_max,
        v_max,
        a_max,
        manipulability_gradient,
        sigma_min,
    )
    lower, upper = compute_velocity_bounds(
        q, qd, q_min, q_max, v_max, a_max, dt
    )
    has_prescribed_base = prescribed_base_velocity is not None
    if not has_prescribed_base:
        prescribed = torch.zeros(
            batch_shape + (3,), dtype=q.dtype, device=q.device
        )
    else:
        require_tensor(
            "prescribed_base_velocity",
            prescribed_base_velocity,
            trailing_shape=(3,),
        )
        if prescribed_base_velocity.shape[:-1] != batch_shape:
            raise ValueError(
                "prescribed_base_velocity batch dimensions must match "
                "coordinated_jacobian"
            )
        if prescribed_base_velocity.dtype != coordinated_jacobian.dtype:
            raise TypeError(
                "prescribed_base_velocity dtype must match coordinated_jacobian"
            )
        if prescribed_base_velocity.device != coordinated_jacobian.device:
            raise ValueError(
                "prescribed_base_velocity device must match coordinated_jacobian"
            )
        prescribed = prescribed_base_velocity

    target = desired_twist + cfg.pose_gain * pose_error
    if has_prescribed_base:
        if (
            (prescribed < lower[..., :3]) | (prescribed > upper[..., :3])
        ).any().item():
            raise ValueError(
                "prescribed_base_velocity violates computed velocity bounds"
            )
        base_twist = torch.matmul(
            coordinated_jacobian[..., :, :3], prescribed.unsqueeze(-1)
        ).squeeze(-1)
        target = target - base_twist
        lower = lower.clone()
        upper = upper.clone()
        lower[..., :3] = 0.0
        upper[..., :3] = 0.0

    count = reduce(mul, batch_shape, 1)
    flat_jacobian = coordinated_jacobian.reshape(count, 6, COORD_DOF)
    flat_target = target.reshape(count, 6)
    flat_lower = lower.reshape(count, COORD_DOF)
    flat_upper = upper.reshape(count, COORD_DOF)
    flat_gradient = manipulability_gradient.reshape(count, COORD_DOF)
    flat_sigma = sigma_min.reshape(count)
    flat_prescribed = prescribed.reshape(count, 3)

    velocities = []
    base_active = []
    phi = []
    psi = []
    saturated = []
    for index in range(count):
        values = _distribute_single(
            flat_jacobian[index],
            flat_target[index],
            flat_lower[index],
            flat_upper[index],
            flat_gradient[index],
            flat_sigma[index],
            cfg,
        )
        velocity_i, base_i, phi_i, psi_i, saturated_i = values
        if has_prescribed_base:
            velocity_i = velocity_i.clone()
            velocity_i[:3] += flat_prescribed[index]
            saturated_i = saturated_i.clone()
            saturated_i[:3] = False
            base_i = base_i or bool(
                (flat_prescribed[index] != 0.0).any().item()
            )
        velocities.append(velocity_i)
        base_active.append(base_i)
        phi.append(phi_i)
        psi.append(psi_i)
        saturated.append(saturated_i)

    return MotionDistributionResult(
        qd_coord=torch.stack(velocities).reshape(batch_shape + (COORD_DOF,)),
        base_active=torch.tensor(
            base_active, dtype=torch.bool, device=coordinated_jacobian.device
        ).reshape(batch_shape),
        sigma_min=sigma_min.clone(),
        phi=torch.stack(phi).reshape(batch_shape),
        psi=torch.stack(psi).reshape(batch_shape),
        saturated=torch.stack(saturated).reshape(batch_shape + (COORD_DOF,)),
    )
