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

    # Express the working set in an independent orthonormal row basis.  Real
    # WBC equalities contain zero/repeated rows; retaining those makes the full
    # KKT matrix singular and its least-squares solution can amplify roundoff
    # into a spurious direction along weak objective curvature.
    left, singular_values, right = torch.linalg.svd(
        scaled_constraint_matrix, full_matrices=False
    )
    rank_threshold = (
        max(scaled_constraint_matrix.shape)
        * torch.finfo(torch.float64).eps
        * singular_values[0]
    )
    rank = int((singular_values > rank_threshold).sum().item())
    constraint_basis = right[:rank]
    basis_rhs = (
        left[:, :rank].transpose(0, 1) @ scaled_constraint_rhs
    ) / singular_values[:rank]
    kkt = torch.zeros((dimension + rank, dimension + rank), dtype=torch.float64)
    kkt[:dimension, :dimension] = scaled_hessian
    kkt[:dimension, dimension:] = constraint_basis.transpose(0, 1)
    kkt[dimension:, :dimension] = constraint_basis
    rhs = torch.cat((-scaled_gradient, basis_rhs))
    if torch.linalg.cholesky_ex(scaled_hessian).info.item() == 0:
        solved, info = torch.linalg.solve_ex(kkt, rhs)
        if info.item() != 0 or not torch.isfinite(solved).all().item():
            solved = torch.linalg.lstsq(kkt, rhs, driver="gelsd").solution
    else:
        # Phase I has deliberately zero curvature in x.  Its minimum-norm
        # direction is well-defined even when the augmented KKT is singular.
        solved = torch.linalg.lstsq(kkt, rhs, driver="gelsd").solution
    multipliers = left[:, :rank] @ (solved[dimension:] / singular_values[:rank])
    return solved[:dimension], multipliers


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


def _constraint_rank(matrix: torch.Tensor) -> int:
    """Return a deterministic numerical rank for an active-row candidate."""
    if matrix.shape[0] == 0:
        return 0
    return int(torch.linalg.matrix_rank(matrix, hermitian=False).item())


def _active_set_qp(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    equality_matrix: torch.Tensor,
    equality_rhs: torch.Tensor,
    inequality_matrix: torch.Tensor,
    inequality_upper: torch.Tensor,
    initial_solution: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
) -> DenseQpResult:
    """Solve a QP from a feasible point with a primal blocking-step loop.

    The iterate remains feasible throughout.  A violated inactive row is only
    imposed after moving to its first blocking point, and active rows are
    released when their KKT multiplier has the wrong sign.  This avoids
    treating an inconsistent collection of working-set equalities as a proof
    that the original inequality problem is infeasible.
    """
    equality_count = equality_matrix.shape[0]
    row_count = inequality_matrix.shape[0]
    solution = initial_solution.clone()
    active: list[int] = []
    iterations = 0
    direction_threshold = max(
        1.0e-12,
        0.1 * tolerance,
        tolerance * 1.0e-4 * max(1.0, float(torch.linalg.vector_norm(solution).item())),
    )

    # With no inequalities there is no blocking or multiplier-release phase;
    # solve the equality-constrained problem once.  This also avoids taking
    # repeated correction steps in deliberately ill-conditioned equality-only
    # diagnostics where the minimizer is finite but very large.
    if row_count == 0:
        solution, _ = _solve_kkt(
            hessian, gradient, equality_matrix, equality_rhs
        )
        if equality_matrix.shape[0]:
            # A very ill-conditioned objective can leave a large-magnitude
            # KKT solution with a few ulps of equality drift.  One minimum-
            # norm equality correction restores the requested residual without
            # changing the active-set or objective logic.
            residual = equality_rhs - equality_matrix @ solution
            before = float(torch.max(torch.abs(residual)).item())
            if before > tolerance:
                correction = torch.linalg.lstsq(
                    equality_matrix, residual, driver="gelsd"
                ).solution
                candidate = solution + correction
                after = float(
                    torch.max(torch.abs(equality_matrix @ candidate - equality_rhs)).item()
                )
                if torch.isfinite(candidate).all().item() and after < before:
                    solution = candidate
        eq_residual, inequality_violation = _metrics(
            solution, equality_matrix, equality_rhs,
            inequality_matrix, inequality_upper,
        )
        return DenseQpResult(
            solution=solution,
            success=(
                torch.isfinite(solution).all().item()
                and _primal_feasible(eq_residual, inequality_violation, tolerance)
            ),
            iterations=1,
            max_equality_residual=eq_residual,
            max_inequality_violation=inequality_violation,
            active_set=(),
        )

    while iterations < max_iterations:
        iterations += 1
        if active:
            active_indices = torch.tensor(active, dtype=torch.long)
            active_matrix = inequality_matrix.index_select(0, active_indices)
            constraint_matrix = torch.cat((equality_matrix, active_matrix), dim=0)
        else:
            constraint_matrix = equality_matrix

        # Compute the minimizer direction in the null space of the current
        # working set.  The right-hand side is zero because ``solution`` is
        # already feasible for every active row.
        direction, multipliers = _solve_kkt(
            hessian,
            hessian @ solution + gradient,
            constraint_matrix,
            torch.zeros(constraint_matrix.shape[0], dtype=torch.float64),
        )
        if not torch.isfinite(direction).all().item() or not torch.isfinite(
            multipliers
        ).all().item():
            eq_residual, inequality_violation = _metrics(
                solution, equality_matrix, equality_rhs,
                inequality_matrix, inequality_upper,
            )
            return DenseQpResult(
                solution=solution,
                success=False,
                iterations=iterations,
                max_equality_residual=eq_residual,
                max_inequality_violation=inequality_violation,
                active_set=tuple(active),
            )

        direction_norm = float(torch.linalg.vector_norm(direction).item())
        if direction_norm <= direction_threshold:
            if active:
                active_multipliers = multipliers[equality_count:]
                minimum_multiplier, multiplier_index = torch.min(
                    active_multipliers, dim=0
                )
                if minimum_multiplier.item() < -tolerance:
                    # A negative multiplier means this bound is obstructing
                    # the constrained minimizer.  Remove the most negative
                    # row, with list order providing deterministic tie-breaks.
                    del active[int(multiplier_index.item())]
                    continue
            eq_residual, inequality_violation = _metrics(
                solution, equality_matrix, equality_rhs,
                inequality_matrix, inequality_upper,
            )
            return DenseQpResult(
                solution=solution,
                success=_primal_feasible(
                    eq_residual, inequality_violation, tolerance
                ),
                iterations=iterations,
                max_equality_residual=eq_residual,
                max_inequality_violation=inequality_violation,
                active_set=tuple(active),
            )

        # Find the first inactive row hit by the direction.  ``slack`` is
        # nonnegative up to floating point noise because the iterate is kept
        # feasible by construction.
        alpha = 1.0
        blocking_index: int | None = None
        if row_count:
            products = inequality_matrix @ direction
            values = inequality_matrix @ solution
            slack = inequality_upper - values
            active_mask = torch.zeros(row_count, dtype=torch.bool)
            if active:
                active_mask[torch.tensor(active, dtype=torch.long)] = True
            candidate_indices = torch.nonzero(
                (~active_mask) & (products > direction_threshold),
                as_tuple=False,
            ).flatten()
            if candidate_indices.numel():
                candidate_ratios = torch.clamp(
                    slack.index_select(0, candidate_indices), min=0.0
                ) / products.index_select(0, candidate_indices)
                working_rank = _constraint_rank(constraint_matrix)
                while True:
                    minimum_ratio, local_index = torch.min(candidate_ratios, dim=0)
                    if minimum_ratio.item() >= 1.0:
                        break
                    index = int(candidate_indices[local_index].item())
                    candidate_matrix = torch.cat(
                        (constraint_matrix, inequality_matrix[index : index + 1]),
                        dim=0,
                    )
                    if _constraint_rank(candidate_matrix) > working_rank:
                        alpha = float(minimum_ratio.item())
                        # torch.min picks the first row on an exact tie; the
                        # candidate indices are in ascending original order.
                        blocking_index = index
                        break
                    # A dependent row is constant along every working-set
                    # direction.  A tiny positive dot product is roundoff,
                    # not a blocker or a stationarity certificate.  Ignore it
                    # before moving, then consider the next blocking row.
                    candidate_ratios[local_index] = float("inf")

        solution = solution + alpha * direction
        if blocking_index is not None:
            active.append(blocking_index)
            active.sort()

    eq_residual, inequality_violation = _metrics(
        solution, equality_matrix, equality_rhs,
        inequality_matrix, inequality_upper,
    )
    return DenseQpResult(
        solution=solution,
        success=False,
        iterations=iterations,
        max_equality_residual=eq_residual,
        max_inequality_violation=inequality_violation,
        active_set=tuple(active),
    )


def _phase_one(
    canonical: _CanonicalProblem,
    inequality_matrix: torch.Tensor,
    inequality_upper: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, int, bool]:
    """Find a feasible point by minimizing one common violation slack."""
    dimension = canonical.gradient.numel()
    seed = _diagnostic_seed(canonical)
    inequality_residual = inequality_matrix @ seed - inequality_upper
    equality_residual = (
        torch.abs(canonical.equality_matrix @ seed - canonical.equality_rhs)
        if canonical.equality_matrix.shape[0]
        else torch.empty(0, dtype=torch.float64)
    )
    initial_violation = 0.0
    if inequality_residual.numel():
        initial_violation = max(initial_violation, float(inequality_residual.max().item()))
    if equality_residual.numel():
        initial_violation = max(initial_violation, float(equality_residual.max().item()))
    initial_slack = max(1.0, initial_violation + 1.0)

    # Original inequalities and both sides of each equality are relaxed by a
    # shared nonnegative scalar t.  Only t is penalized: any x regularizer
    # would trade feasibility for a smaller displacement, especially for small
    # constraint coefficients.  The deterministic least-squares KKT solve
    # handles the semidefinite objective without changing this feasibility QP.
    augmented_dimension = dimension + 1
    augmented_hessian = torch.zeros(
        (augmented_dimension, augmented_dimension), dtype=torch.float64
    )
    augmented_hessian[-1, -1] = 1.0
    augmented_gradient = torch.zeros(augmented_dimension, dtype=torch.float64)

    rows = []
    rhs = []
    if inequality_matrix.shape[0]:
        rows.append(torch.cat(
            (inequality_matrix, -torch.ones((inequality_matrix.shape[0], 1), dtype=torch.float64)),
            dim=1,
        ))
        rhs.append(inequality_upper)
    if canonical.equality_matrix.shape[0]:
        rows.append(torch.cat(
            (canonical.equality_matrix, -torch.ones((canonical.equality_matrix.shape[0], 1), dtype=torch.float64)),
            dim=1,
        ))
        rhs.append(canonical.equality_rhs)
        rows.append(torch.cat(
            (-canonical.equality_matrix, -torch.ones((canonical.equality_matrix.shape[0], 1), dtype=torch.float64)),
            dim=1,
        ))
        rhs.append(-canonical.equality_rhs)
    rows.append(torch.cat(
        (torch.zeros((1, dimension), dtype=torch.float64), -torch.ones((1, 1), dtype=torch.float64)),
        dim=1,
    ))
    rhs.append(torch.zeros(1, dtype=torch.float64))
    augmented_matrix = torch.cat(rows, dim=0)
    augmented_upper = torch.cat(rhs, dim=0)
    augmented_seed = torch.cat((seed, torch.tensor([initial_slack], dtype=torch.float64)))
    phase = _active_set_qp(
        augmented_hessian,
        augmented_gradient,
        torch.empty((0, augmented_dimension), dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        augmented_matrix,
        augmented_upper,
        augmented_seed,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    phase_solution = phase.solution[:dimension]
    phase_slack = float(phase.solution[-1].item())
    equality_residual, inequality_violation = _metrics(
        phase_solution,
        canonical.equality_matrix,
        canonical.equality_rhs,
        inequality_matrix,
        inequality_upper,
    )
    feasible = (
        phase.success
        and phase_slack <= tolerance
        and _primal_feasible(equality_residual, inequality_violation, tolerance)
    )
    return phase_solution, phase.iterations, feasible


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

    solution = _diagnostic_seed(canonical)
    equality_residual, inequality_violation = _metrics(
        solution,
        canonical.equality_matrix,
        canonical.equality_rhs,
        inequality_matrix,
        inequality_upper,
    )
    iterations_used = 0
    if not _primal_feasible(equality_residual, inequality_violation, tolerance):
        solution, phase_iterations, phase_feasible = _phase_one(
            canonical,
            inequality_matrix,
            inequality_upper,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        iterations_used = phase_iterations
        equality_residual, inequality_violation = _metrics(
            solution,
            canonical.equality_matrix,
            canonical.equality_rhs,
            inequality_matrix,
            inequality_upper,
        )
        if not phase_feasible:
            return DenseQpResult(
                solution=solution,
                success=False,
                iterations=iterations_used,
                max_equality_residual=equality_residual,
                max_inequality_violation=inequality_violation,
                active_set=(),
            )

    remaining_iterations = max_iterations - iterations_used
    if remaining_iterations <= 0:
        return DenseQpResult(
            solution=solution,
            success=False,
            iterations=iterations_used,
            max_equality_residual=equality_residual,
            max_inequality_violation=inequality_violation,
            active_set=(),
        )

    result = _active_set_qp(
        canonical.hessian,
        canonical.gradient,
        canonical.equality_matrix,
        canonical.equality_rhs,
        inequality_matrix,
        inequality_upper,
        solution,
        tolerance=tolerance,
        max_iterations=remaining_iterations,
    )
    return DenseQpResult(
        solution=result.solution,
        success=result.success,
        iterations=iterations_used + result.iterations,
        max_equality_residual=result.max_equality_residual,
        max_inequality_violation=result.max_inequality_violation,
        active_set=result.active_set,
    )
