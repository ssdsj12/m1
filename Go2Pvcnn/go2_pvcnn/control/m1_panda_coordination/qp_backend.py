"""Deterministic CPU float64 active-set QP backend for C0 reference control."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseQpProblem:
    hessian: torch.Tensor
    gradient: torch.Tensor
    equality_matrix: torch.Tensor
    equality_rhs: torch.Tensor
    inequality_matrix: torch.Tensor
    inequality_upper: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor


@dataclass(frozen=True)
class DenseQpResult:
    solution: torch.Tensor
    success: bool
    iterations: int
    max_equality_residual: float
    max_inequality_violation: float
    active_set: tuple[int, ...]


@dataclass(frozen=True)
class _CanonicalProblem:
    hessian: torch.Tensor
    gradient: torch.Tensor
    equality_matrix: torch.Tensor
    equality_rhs: torch.Tensor
    inequality_matrix: torch.Tensor
    inequality_upper: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor


def _as_cpu_float64(name: str, value: torch.Tensor, *, allow_infinity=False):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    converted = value.detach().to(device="cpu", dtype=torch.float64)
    valid = ~torch.isnan(converted) if allow_infinity else torch.isfinite(converted)
    if not valid.all().item():
        qualifier = "must not contain NaN" if allow_infinity else "must contain only finite values"
        raise ValueError(f"{name} {qualifier}")
    return converted


def _canonicalize(problem: DenseQpProblem) -> _CanonicalProblem:
    if not isinstance(problem, DenseQpProblem):
        raise TypeError("problem must be a DenseQpProblem")
    hessian = _as_cpu_float64("hessian", problem.hessian)
    gradient = _as_cpu_float64("gradient", problem.gradient)
    if gradient.ndim != 1 or gradient.numel() == 0:
        raise ValueError("gradient must have shape (n,) with n > 0")
    dimension = gradient.numel()
    if hessian.shape != (dimension, dimension):
        raise ValueError(f"hessian must have shape ({dimension}, {dimension})")

    equality_matrix = _as_cpu_float64(
        "equality_matrix", problem.equality_matrix
    )
    equality_rhs = _as_cpu_float64("equality_rhs", problem.equality_rhs)
    if equality_matrix.ndim != 2 or equality_matrix.shape[1:] != (dimension,):
        raise ValueError(f"equality_matrix must have shape (m, {dimension})")
    if equality_rhs.shape != (equality_matrix.shape[0],):
        raise ValueError(
            f"equality_rhs must have shape ({equality_matrix.shape[0]},)"
        )

    inequality_matrix = _as_cpu_float64(
        "inequality_matrix", problem.inequality_matrix
    )
    inequality_upper = _as_cpu_float64(
        "inequality_upper", problem.inequality_upper
    )
    if inequality_matrix.ndim != 2 or inequality_matrix.shape[1:] != (dimension,):
        raise ValueError(f"inequality_matrix must have shape (p, {dimension})")
    if inequality_upper.shape != (inequality_matrix.shape[0],):
        raise ValueError(
            f"inequality_upper must have shape ({inequality_matrix.shape[0]},)"
        )

    lower_bound = _as_cpu_float64(
        "lower_bound", problem.lower_bound, allow_infinity=True
    )
    upper_bound = _as_cpu_float64(
        "upper_bound", problem.upper_bound, allow_infinity=True
    )
    if lower_bound.shape != (dimension,):
        raise ValueError(f"lower_bound must have shape ({dimension},)")
    if upper_bound.shape != (dimension,):
        raise ValueError(f"upper_bound must have shape ({dimension},)")

    return _CanonicalProblem(
        hessian=0.5 * (hessian + hessian.transpose(0, 1)),
        gradient=gradient,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix,
        inequality_upper=inequality_upper,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def _with_bounds(problem: _CanonicalProblem) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = problem.gradient.numel()
    rows = [problem.inequality_matrix]
    upper = [problem.inequality_upper]
    identity = torch.eye(dimension, dtype=torch.float64)
    for index in range(dimension):
        if torch.isfinite(problem.upper_bound[index]).item():
            rows.append(identity[index : index + 1])
            upper.append(problem.upper_bound[index : index + 1])
    for index in range(dimension):
        if torch.isfinite(problem.lower_bound[index]).item():
            rows.append(-identity[index : index + 1])
            upper.append(-problem.lower_bound[index : index + 1])
    return torch.cat(rows, dim=0), torch.cat(upper, dim=0)


def _diagnostic_seed(problem: _CanonicalProblem) -> torch.Tensor:
    lower_finite = torch.isfinite(problem.lower_bound)
    upper_finite = torch.isfinite(problem.upper_bound)
    both = lower_finite & upper_finite
    solution = torch.zeros_like(problem.gradient)
    solution = torch.where(
        both,
        0.5 * (problem.lower_bound + problem.upper_bound),
        solution,
    )
    solution = torch.where(
        lower_finite & ~upper_finite,
        torch.maximum(solution, problem.lower_bound),
        solution,
    )
    solution = torch.where(
        upper_finite & ~lower_finite,
        torch.minimum(solution, problem.upper_bound),
        solution,
    )
    return torch.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)


def _solve_kkt(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_rhs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = gradient.numel()
    count = constraint_matrix.shape[0]
    # A single positive objective scale leaves the primal minimizer and the
    # signs of inequality multipliers unchanged.  It prevents the C0 priority
    # curvature (up to 1e6) from numerically overwhelming KKT constraint rows.
    objective_scale = max(float(hessian.abs().max().item()), 1.0)
    scaled_hessian = hessian / objective_scale
    scaled_gradient = gradient / objective_scale
    if count == 0:
        solution = torch.linalg.lstsq(
            scaled_hessian, -scaled_gradient, driver="gelsd"
        ).solution
        return solution, torch.empty(0, dtype=torch.float64)

    constraint_scale = torch.linalg.vector_norm(
        constraint_matrix, dim=1
    ).clamp_min(1.0)
    scaled_constraint_matrix = constraint_matrix / constraint_scale.unsqueeze(1)
    scaled_constraint_rhs = constraint_rhs / constraint_scale
    kkt = torch.zeros((dimension + count, dimension + count), dtype=torch.float64)
    kkt[:dimension, :dimension] = scaled_hessian
    kkt[:dimension, dimension:] = scaled_constraint_matrix.transpose(0, 1)
    kkt[dimension:, :dimension] = scaled_constraint_matrix
    rhs = torch.cat((-scaled_gradient, scaled_constraint_rhs))
    solved = torch.linalg.lstsq(kkt, rhs, driver="gelsd").solution
    return solved[:dimension], solved[dimension:]


def _metrics(
    solution: torch.Tensor,
    equality_matrix: torch.Tensor,
    equality_rhs: torch.Tensor,
    inequality_matrix: torch.Tensor,
    inequality_upper: torch.Tensor,
) -> tuple[float, float]:
    if equality_matrix.shape[0]:
        equality_residual = float(
            torch.max(torch.abs(equality_matrix @ solution - equality_rhs)).item()
        )
    else:
        equality_residual = 0.0
    if inequality_matrix.shape[0]:
        inequality_violation = float(
            torch.clamp(
                inequality_matrix @ solution - inequality_upper, min=0.0
            ).max().item()
        )
    else:
        inequality_violation = 0.0
    return equality_residual, inequality_violation


def _primal_feasible(
    equality_residual: float,
    inequality_violation: float,
    tolerance: float,
) -> bool:
    return equality_residual <= tolerance and inequality_violation <= tolerance


def solve_reference_qp(
    problem: DenseQpProblem,
    *,
    tolerance: float = 1.0e-9,
    max_iterations: int = 128,
) -> DenseQpResult:
    """Solve one dense convex QP with a deterministic primal active set."""

    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError("tolerance must be a real number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")

    canonical = _canonicalize(problem)
    inequality_matrix, inequality_upper = _with_bounds(canonical)
    invalid_infinite_bound = torch.isposinf(canonical.lower_bound) | torch.isneginf(
        canonical.upper_bound
    )
    inconsistent_bound = canonical.lower_bound > canonical.upper_bound
    if invalid_infinite_bound.any().item() or inconsistent_bound.any().item():
        solution = _diagnostic_seed(canonical)
        equality_residual, inequality_violation = _metrics(
            solution,
            canonical.equality_matrix,
            canonical.equality_rhs,
            inequality_matrix,
            inequality_upper,
        )
        if inconsistent_bound.any().item():
            bound_gap = float(
                (canonical.lower_bound - canonical.upper_bound)[inconsistent_bound]
                .max()
                .item()
            )
            inequality_violation = max(inequality_violation, 0.5 * bound_gap)
        return DenseQpResult(
            solution=solution,
            success=False,
            iterations=0,
            max_equality_residual=equality_residual,
            max_inequality_violation=inequality_violation,
            active_set=(),
        )

    active: list[int] = []
    solution = _diagnostic_seed(canonical)
    equality_count = canonical.equality_matrix.shape[0]

    for iteration in range(1, max_iterations + 1):
        if active:
            active_indices = torch.tensor(active, dtype=torch.long)
            active_matrix = inequality_matrix.index_select(0, active_indices)
            active_rhs = inequality_upper.index_select(0, active_indices)
            constraint_matrix = torch.cat(
                (canonical.equality_matrix, active_matrix), dim=0
            )
            constraint_rhs = torch.cat((canonical.equality_rhs, active_rhs), dim=0)
        else:
            constraint_matrix = canonical.equality_matrix
            constraint_rhs = canonical.equality_rhs

        solution, multipliers = _solve_kkt(
            canonical.hessian,
            canonical.gradient,
            constraint_matrix,
            constraint_rhs,
        )
        if not torch.isfinite(solution).all().item() or not torch.isfinite(
            multipliers
        ).all().item():
            solution = _diagnostic_seed(canonical)
            equality_residual, inequality_violation = _metrics(
                solution,
                canonical.equality_matrix,
                canonical.equality_rhs,
                inequality_matrix,
                inequality_upper,
            )
            return DenseQpResult(
                solution=solution,
                success=False,
                iterations=iteration,
                max_equality_residual=equality_residual,
                max_inequality_violation=inequality_violation,
                active_set=tuple(active),
            )

        active_residual = (
            torch.max(torch.abs(constraint_matrix @ solution - constraint_rhs)).item()
            if constraint_matrix.shape[0]
            else 0.0
        )
        if active_residual > tolerance:
            equality_residual, inequality_violation = _metrics(
                solution,
                canonical.equality_matrix,
                canonical.equality_rhs,
                inequality_matrix,
                inequality_upper,
            )
            return DenseQpResult(
                solution=solution,
                success=False,
                iterations=iteration,
                max_equality_residual=equality_residual,
                max_inequality_violation=inequality_violation,
                active_set=tuple(active),
            )

        if inequality_matrix.shape[0]:
            violations = inequality_matrix @ solution - inequality_upper
            if active:
                violations[torch.tensor(active, dtype=torch.long)] = -torch.inf
            maximum_violation, violated_index = torch.max(violations, dim=0)
            if maximum_violation.item() > tolerance:
                active.append(int(violated_index.item()))
                active.sort()
                continue

        if active:
            active_multipliers = multipliers[equality_count:]
            minimum_multiplier, multiplier_index = torch.min(
                active_multipliers, dim=0
            )
            if minimum_multiplier.item() < -tolerance:
                del active[int(multiplier_index.item())]
                continue

        equality_residual, inequality_violation = _metrics(
            solution,
            canonical.equality_matrix,
            canonical.equality_rhs,
            inequality_matrix,
            inequality_upper,
        )
        return DenseQpResult(
            solution=solution,
            success=_primal_feasible(
                equality_residual, inequality_violation, tolerance
            ),
            iterations=iteration,
            max_equality_residual=equality_residual,
            max_inequality_violation=inequality_violation,
            active_set=tuple(active),
        )

    equality_residual, inequality_violation = _metrics(
        solution,
        canonical.equality_matrix,
        canonical.equality_rhs,
        inequality_matrix,
        inequality_upper,
    )
    return DenseQpResult(
        solution=solution,
        success=_primal_feasible(
            equality_residual, inequality_violation, tolerance
        ),
        iterations=max_iterations,
        max_equality_residual=equality_residual,
        max_inequality_violation=inequality_violation,
        active_set=tuple(active),
    )
