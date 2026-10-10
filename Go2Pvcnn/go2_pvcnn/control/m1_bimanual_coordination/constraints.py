"""Explicit 43-channel effort, collision, and grasp constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .contracts import ACTIVE_CONTROL_DOF, BimanualPhase, BimanualSnapshot
from .dual_arm_mpc import DualArmMpcSolution
from .hand_mpc import HandMpcSolution
from .object_mpc import ObjectMpcSolution
from .diagnostics import WirePipelineDiagnostics


@dataclass(frozen=True)
class WireGateConfig:
    """Hard gates for the reduced-wire command boundary."""

    min_contact_force_n: float = 0.2
    min_contacts_per_hand: int = 2
    max_slip_m: float = 0.02
    min_clamp_force_n: float = 0.2
    max_clamp_force_n: float | None = None
    min_normal_alignment: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.min_contact_force_n, bool) or not math.isfinite(float(self.min_contact_force_n)) or self.min_contact_force_n <= 0.0:
            raise ValueError("min_contact_force_n must be finite and positive")
        if isinstance(self.max_slip_m, bool) or not math.isfinite(float(self.max_slip_m)) or self.max_slip_m <= 0.0:
            raise ValueError("max_slip_m must be finite and positive")
        if isinstance(self.min_contacts_per_hand, bool) or not isinstance(self.min_contacts_per_hand, int) or not 1 <= self.min_contacts_per_hand <= 5:
            raise ValueError("min_contacts_per_hand must be in [1, 5]")
        if not math.isfinite(self.min_clamp_force_n) or self.min_clamp_force_n <= 0.0:
            raise ValueError("min_clamp_force_n must be finite and positive")
        if self.max_clamp_force_n is not None and (
            not math.isfinite(self.max_clamp_force_n) or self.max_clamp_force_n < self.min_clamp_force_n
        ):
            raise ValueError("max_clamp_force_n must be finite and >= min_clamp_force_n")
        if not math.isfinite(self.min_normal_alignment) or not 0.0 <= self.min_normal_alignment <= 1.0:
            raise ValueError("min_normal_alignment must be in [0, 1]")


@dataclass(frozen=True)
class WireGateResult:
    accepted: bool
    reason: str | None


def evaluate_wire_gate(
    diagnostics: WirePipelineDiagnostics,
    cfg: WireGateConfig | None = None,
) -> WireGateResult:
    """Evaluate one wire observation in stable, fail-closed order.

    ``APPROACH`` and ``PRELOAD`` allow bounded commands before contact.
    GRASP/LIFT/HOLD require actual bilateral masks; LIFT/HOLD require dynamics,
    recoverability, clamp force, and slip margins.
    """

    if not isinstance(diagnostics, WirePipelineDiagnostics):
        raise TypeError("diagnostics must be WirePipelineDiagnostics")
    cfg = WireGateConfig() if cfg is None else cfg
    if not isinstance(cfg, WireGateConfig):
        raise TypeError("cfg must be WireGateConfig")
    phase = diagnostics.phase
    if not diagnostics.command_accepted:
        return WireGateResult(False, diagnostics.terminal_reason or "command_rejected")
    if phase not in {"PRELOAD", "GRASP", "LIFT", "HOLD"}:
        return WireGateResult(True, None)
    if phase == "PRELOAD":
        if cfg.max_clamp_force_n is not None and max(diagnostics.left_clamp_force_n, diagnostics.right_clamp_force_n) > cfg.max_clamp_force_n:
            return WireGateResult(False, "clamp_force_excessive")
        return WireGateResult(True, None)
    if not diagnostics.bilateral_contact:
        return WireGateResult(False, "contact_lost_during_lift" if phase in {"LIFT", "HOLD"} else "bilateral_contact_required")
    if phase in {"GRASP", "LIFT", "HOLD"}:
        if diagnostics.left_contact_count < cfg.min_contacts_per_hand or diagnostics.right_contact_count < cfg.min_contacts_per_hand:
            return WireGateResult(False, "per_digit_contact_latch_required")
        if not diagnostics.left_latched or not diagnostics.right_latched:
            return WireGateResult(False, "per_digit_contact_latch_required")
        if min(diagnostics.left_clamp_force_n, diagnostics.right_clamp_force_n) < max(cfg.min_contact_force_n, cfg.min_clamp_force_n):
            return WireGateResult(False, "clamp_force_insufficient")
        if cfg.max_clamp_force_n is not None and max(diagnostics.left_clamp_force_n, diagnostics.right_clamp_force_n) > cfg.max_clamp_force_n:
            return WireGateResult(False, "clamp_force_excessive")
        if min(diagnostics.left_normal_alignment, diagnostics.right_normal_alignment) < cfg.min_normal_alignment:
            return WireGateResult(False, "contact_normal_misaligned")
        if diagnostics.slip_m > cfg.max_slip_m:
            return WireGateResult(False, "wire_slip_limit")
        if diagnostics.force_closure_margin <= 0.0:
            return WireGateResult(False, "force_closure_lost")
    if phase in {"LIFT", "HOLD"}:
        if not diagnostics.wire_feasible:
            return WireGateResult(False, diagnostics.wire_failure_reason or "wire_dynamics_infeasible")
        if not diagnostics.arm_recoverability:
            return WireGateResult(False, "arm_mpc_recoverability")
    return WireGateResult(True, None)


@dataclass(frozen=True)
class BimanualConstraintCfg:
    minimum_collision_distance_m: float = 0.02
    control_dt: float = 0.005

    def __post_init__(self) -> None:
        for name in ("minimum_collision_distance_m", "control_dt"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class BimanualWbcRequest:
    snapshot: BimanualSnapshot
    object_solution: ObjectMpcSolution
    arm_solution: DualArmMpcSolution
    left_hand_solution: HandMpcSolution
    right_hand_solution: HandMpcSolution
    collision_distances: torch.Tensor
    collision_jacobian: torch.Tensor
    wire_gate_config: WireGateConfig = field(default_factory=WireGateConfig)
    # Omitted phase retains the conservative contact-required legacy contract.
    phase: BimanualPhase | None = None

    def __post_init__(self) -> None:
        if self.phase is not None and not isinstance(self.phase, BimanualPhase):
            raise TypeError("phase must be BimanualPhase or None")
        if not isinstance(self.wire_gate_config, WireGateConfig):
            raise TypeError("wire_gate_config must be WireGateConfig")
        for name, expected in (
            ("snapshot", BimanualSnapshot),
            ("object_solution", ObjectMpcSolution),
            ("arm_solution", DualArmMpcSolution),
            ("left_hand_solution", HandMpcSolution),
            ("right_hand_solution", HandMpcSolution),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if not isinstance(self.collision_distances, torch.Tensor):
            raise TypeError("collision_distances must be a torch.Tensor")
        if not isinstance(self.collision_jacobian, torch.Tensor):
            raise TypeError("collision_jacobian must be a torch.Tensor")
        for name, value in (
            ("collision_distances", self.collision_distances),
            ("collision_jacobian", self.collision_jacobian),
        ):
            if value.dtype != torch.float64:
                raise TypeError(f"{name} must have dtype torch.float64")
            if value.device.type != "cpu":
                raise ValueError(f"{name} must be on device cpu")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if self.collision_distances.ndim != 1 or self.collision_distances.numel() == 0:
            raise ValueError("collision_distances must have shape (pairs,) with pairs > 0")
        expected_shape = (self.collision_distances.numel(), ACTIVE_CONTROL_DOF)
        if tuple(self.collision_jacobian.shape) != expected_shape:
            raise ValueError(
                f"collision_jacobian must have shape {expected_shape}; got {tuple(self.collision_jacobian.shape)}"
            )
        object.__setattr__(
            self, "collision_distances", self.collision_distances.clone()
        )
        object.__setattr__(
            self, "collision_jacobian", self.collision_jacobian.clone()
        )


@dataclass(frozen=True)
class BimanualConstraintSet:
    lower_effort: torch.Tensor
    upper_effort: torch.Tensor
    inequality_matrix: torch.Tensor
    inequality_upper: torch.Tensor
    min_collision_distance: float
    force_closure_margin: float

    def __post_init__(self) -> None:
        if self.lower_effort.shape != (ACTIVE_CONTROL_DOF,) or self.upper_effort.shape != (
            ACTIVE_CONTROL_DOF,
        ):
            raise ValueError("effort limits must have shape (43,)")
        if self.inequality_matrix.ndim != 2 or self.inequality_matrix.shape[1] != ACTIVE_CONTROL_DOF:
            raise ValueError("inequality_matrix must have shape (rows, 43)")
        if self.inequality_upper.shape != (self.inequality_matrix.shape[0],):
            raise ValueError("inequality_upper must match inequality rows")


def effort_limits() -> torch.Tensor:
    """Return positive limits in the frozen 16+1+14+12 channel order."""

    panda = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    return torch.tensor(
        (
            *(150.0 for _ in range(12)),
            *(200.0 for _ in range(4)),
            50.0,
            *panda,
            *panda,
            *(10.0 for _ in range(12)),
        ),
        dtype=torch.float64,
    )


def _grasp_loaded(request: BimanualWbcRequest) -> bool:
    # APPROACH commands intentionally carry no grasp wrench. Force closure is
    # a hard gate only once both object-MPC normal forces reach grasp preload.
    left_normal = -float(request.object_solution.left_wrench[0, 1].item())
    right_normal = float(request.object_solution.right_wrench[0, 1].item())
    return min(left_normal, right_normal) >= 8.0 - 1.0e-6


def subsolution_failure_reason(request: BimanualWbcRequest) -> str | None:
    if not request.object_solution.diagnostics.feasible:
        return "object_mpc_infeasible"
    if not request.arm_solution.both_feasible:
        return "dual_arm_mpc_infeasible"
    if not request.arm_solution.left.diagnostics.feasible:
        return "left_arm_mpc_infeasible"
    if not request.arm_solution.right.diagnostics.feasible:
        return "right_arm_mpc_infeasible"
    if not request.left_hand_solution.diagnostics.feasible:
        return "left_hand_mpc_infeasible"
    if not request.right_hand_solution.diagnostics.feasible:
        return "right_hand_mpc_infeasible"
    # Wire diagnostics are optional: the legacy rigid-object path has all
    # values set to ``None`` and therefore retains its historical behavior.
    wire_diag = request.object_solution.diagnostics
    wire_mode = any(
        value is not None
        for value in (
            wire_diag.wire_stretch_max,
            wire_diag.wire_slip_max_m,
            wire_diag.wire_friction_feasible,
            wire_diag.wire_lift_feasible,
        )
    )
    if wire_mode:
        if wire_diag.wire_friction_feasible is False:
            return "wire_dynamics_infeasible"
        if wire_diag.wire_slip_max_m is not None and wire_diag.wire_slip_max_m > request.wire_gate_config.max_slip_m:
            return "wire_slip_limit"
        if wire_diag.wire_lift_feasible is False:
            return "wire_dynamics_infeasible"
        if request.phase not in {BimanualPhase.APPROACH, BimanualPhase.PRELOAD}:
            if not bool(request.snapshot.left_hand.contact_mask.any()) or not bool(request.snapshot.right_hand.contact_mask.any()):
                return "bilateral_contact_required"
            for hand in (request.snapshot.left_hand, request.snapshot.right_hand):
                valid = hand.contact_mask & (torch.linalg.vector_norm(hand.fingertip_forces_b, dim=1)
                                            > request.wire_gate_config.min_contact_force_n)
                if not hand.contact_consistent or int(valid.sum()) < request.wire_gate_config.min_contacts_per_hand:
                    return "per_digit_contact_latch_required"
    if _grasp_loaded(request) and request.object_solution.diagnostics.force_closure_margin <= 0.0:
        return "force_closure_lost"
    return None


def build_bimanual_constraints(
    request: BimanualWbcRequest,
    cfg: BimanualConstraintCfg | None = None,
) -> BimanualConstraintSet:
    """Linearize pair distances against one 200 Hz effort command."""

    if not isinstance(request, BimanualWbcRequest):
        raise TypeError("request must be BimanualWbcRequest")
    cfg = BimanualConstraintCfg() if cfg is None else cfg
    if not isinstance(cfg, BimanualConstraintCfg):
        raise TypeError("cfg must be BimanualConstraintCfg")
    limits = effort_limits()
    # d_next ~= d + dt^2 J tau; enforce d_next >= d_min.
    collision_matrix = -(cfg.control_dt**2) * request.collision_jacobian
    collision_upper = (
        request.collision_distances - cfg.minimum_collision_distance_m
    )
    force_closure_margin = min(
        request.object_solution.diagnostics.force_closure_margin,
        request.left_hand_solution.diagnostics.slip_margin,
        request.right_hand_solution.diagnostics.slip_margin,
    )
    inequality_matrix = collision_matrix
    inequality_upper = collision_upper
    if _grasp_loaded(request):
        closure_row = torch.zeros((1, ACTIVE_CONTROL_DOF), dtype=torch.float64)
        inequality_matrix = torch.cat((collision_matrix, closure_row), dim=0)
        inequality_upper = torch.cat(
            (collision_upper, torch.tensor([force_closure_margin], dtype=torch.float64))
        )
    return BimanualConstraintSet(
        lower_effort=-limits,
        upper_effort=limits,
        inequality_matrix=inequality_matrix,
        inequality_upper=inequality_upper,
        min_collision_distance=float(request.collision_distances.min().item()),
        force_closure_margin=float(force_closure_margin),
    )
