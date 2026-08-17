import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.qp_backend import (
    DenseQpProblem,
    DenseQpResult,
    solve_reference_qp,
)


def _problem(
    hessian,
    gradient,
    *,
    equality_matrix=None,
    equality_rhs=None,
    inequality_matrix=None,
    inequality_upper=None,
    lower_bound=None,
    upper_bound=None,
):
    hessian = torch.as_tensor(hessian)
    gradient = torch.as_tensor(gradient)
    dimension = gradient.numel()
    dtype = hessian.dtype
    return DenseQpProblem(
        hessian=hessian,
        gradient=gradient,
        equality_matrix=(
            torch.empty((0, dimension), dtype=dtype)
            if equality_matrix is None
            else torch.as_tensor(equality_matrix, dtype=dtype)
        ),
        equality_rhs=(
            torch.empty(0, dtype=dtype)
            if equality_rhs is None
            else torch.as_tensor(equality_rhs, dtype=dtype)
        ),
        inequality_matrix=(
            torch.empty((0, dimension), dtype=dtype)
            if inequality_matrix is None
            else torch.as_tensor(inequality_matrix, dtype=dtype)
        ),
        inequality_upper=(
            torch.empty(0, dtype=dtype)
            if inequality_upper is None
            else torch.as_tensor(inequality_upper, dtype=dtype)
        ),
        lower_bound=(
            torch.full((dimension,), -float("inf"), dtype=dtype)
            if lower_bound is None
            else torch.as_tensor(lower_bound, dtype=dtype)
        ),
        upper_bound=(
            torch.full((dimension,), float("inf"), dtype=dtype)
            if upper_bound is None
            else torch.as_tensor(upper_bound, dtype=dtype)
        ),
    )


def test_unconstrained_quadratic_returns_float64_cpu_minimum():
    problem = _problem(
        torch.diag(torch.tensor([2.0, 2.0], dtype=torch.float32)),
        torch.tensor([-2.0, -4.0], dtype=torch.float32),
    )

    result = solve_reference_qp(problem)

    assert isinstance(result, DenseQpResult)
    assert result.success
    assert result.solution.dtype == torch.float64
    assert result.solution.device.type == "cpu"
    assert torch.allclose(
        result.solution, torch.tensor([1.0, 2.0], dtype=torch.float64), atol=1.0e-10
    )
    assert result.max_equality_residual == pytest.approx(0.0)
    assert result.max_inequality_violation == pytest.approx(0.0)
    assert result.active_set == ()


def test_equality_only_kkt_solution_satisfies_constraint():
    problem = _problem(
        torch.eye(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        equality_matrix=[[1.0, 1.0]],
        equality_rhs=[1.0],
    )

    result = solve_reference_qp(problem)

    assert result.success
    assert torch.allclose(
        result.solution, torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1.0e-10
    )
    assert result.max_equality_residual <= 1.0e-9


def test_box_bounds_clip_both_sides_at_optimum():
    problem = _problem(
        torch.eye(2, dtype=torch.float64),
        torch.tensor([-3.0, 3.0], dtype=torch.float64),
        lower_bound=[-1.0, -1.0],
        upper_bound=[1.0, 1.0],
    )

    result = solve_reference_qp(problem)

    assert result.success
    assert torch.allclose(
        result.solution, torch.tensor([1.0, -1.0], dtype=torch.float64), atol=1.0e-9
    )
    assert len(result.active_set) == 2
    assert result.max_inequality_violation <= 1.0e-9


def test_single_active_inequality_matches_projection_solution():
    problem = _problem(
        torch.eye(2, dtype=torch.float64),
        torch.tensor([-2.0, 0.0], dtype=torch.float64),
        inequality_matrix=[[1.0, 1.0]],
        inequality_upper=[1.0],
    )

    result = solve_reference_qp(problem)

    assert result.success
    assert torch.allclose(
        result.solution, torch.tensor([1.5, -0.5], dtype=torch.float64), atol=1.0e-9
    )
    assert result.active_set == (0,)


def test_redundant_inequalities_converge_without_duplicate_active_indices():
    problem = _problem(
        torch.ones((1, 1), dtype=torch.float64),
        torch.tensor([-2.0], dtype=torch.float64),
        inequality_matrix=[[1.0], [2.0]],
        inequality_upper=[1.0, 2.0],
    )

    result = solve_reference_qp(problem)

    assert result.success
    assert result.solution.item() == pytest.approx(1.0, abs=1.0e-9)
    assert len(result.active_set) == len(set(result.active_set))
    assert result.max_inequality_violation <= 1.0e-9


def test_mixed_equality_inequality_and_bounds_are_all_enforced():
    problem = _problem(
        torch.eye(3, dtype=torch.float64),
        torch.tensor([-2.0, -2.0, 0.0], dtype=torch.float64),
        equality_matrix=[[0.0, 0.0, 1.0]],
        equality_rhs=[0.25],
        inequality_matrix=[[1.0, 1.0, 0.0]],
        inequality_upper=[1.0],
        lower_bound=[0.0, 0.0, -1.0],
        upper_bound=[2.0, 2.0, 1.0],
    )

    result = solve_reference_qp(problem)

    assert result.success
    assert torch.allclose(
        result.solution,
        torch.tensor([0.5, 0.5, 0.25], dtype=torch.float64),
        atol=1.0e-9,
    )
    assert result.max_equality_residual <= 1.0e-9
    assert result.max_inequality_violation <= 1.0e-9


def test_repeated_solves_are_bitwise_deterministic():
    problem = _problem(
        torch.tensor([[3.0, 0.2], [0.2, 1.0]], dtype=torch.float64),
        torch.tensor([-2.0, -1.0], dtype=torch.float64),
        inequality_matrix=[[1.0, 1.0], [-1.0, 0.0]],
        inequality_upper=[0.8, 0.0],
    )

    results = [solve_reference_qp(problem) for _ in range(5)]

    assert all(result.success for result in results)
    assert all(torch.equal(results[0].solution, result.solution) for result in results[1:])
    assert all(results[0].active_set == result.active_set for result in results[1:])
    assert all(results[0].iterations == result.iterations for result in results[1:])


def test_infeasible_bounds_return_finite_unsuccessful_result():
    problem = _problem(
        torch.ones((1, 1), dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        lower_bound=[1.0],
        upper_bound=[0.0],
    )

    result = solve_reference_qp(problem)

    assert not result.success
    assert torch.isfinite(result.solution).all()
    assert result.max_inequality_violation > 0.0


def test_contradictory_inequalities_return_finite_unsuccessful_result():
    problem = _problem(
        torch.ones((1, 1), dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        inequality_matrix=[[1.0], [-1.0]],
        inequality_upper=[0.0, -1.0],
    )

    result = solve_reference_qp(problem)

    assert not result.success
    assert torch.isfinite(result.solution).all()
    assert result.max_inequality_violation > 0.0


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_solver_rejects_invalid_tolerance(tolerance):
    problem = _problem(torch.eye(1), torch.zeros(1))
    with pytest.raises(ValueError, match="tolerance must be finite and positive"):
        solve_reference_qp(problem, tolerance=tolerance)


@pytest.mark.parametrize("max_iterations", [0, -1, True, 1.5])
def test_solver_rejects_invalid_iteration_limit(max_iterations):
    problem = _problem(torch.eye(1), torch.zeros(1))
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        solve_reference_qp(problem, max_iterations=max_iterations)


def test_solver_rejects_malformed_shapes_and_non_finite_coefficients():
    malformed = _problem(torch.eye(2), torch.zeros(2))
    malformed = DenseQpProblem(
        hessian=malformed.hessian,
        gradient=malformed.gradient,
        equality_matrix=torch.zeros(1, 3),
        equality_rhs=torch.zeros(1),
        inequality_matrix=malformed.inequality_matrix,
        inequality_upper=malformed.inequality_upper,
        lower_bound=malformed.lower_bound,
        upper_bound=malformed.upper_bound,
    )
    with pytest.raises(ValueError, match=r"equality_matrix must have shape \(m, 2\)"):
        solve_reference_qp(malformed)

    non_finite = _problem(torch.tensor([[float("nan")]]), torch.zeros(1))
    with pytest.raises(ValueError, match="hessian must contain only finite values"):
        solve_reference_qp(non_finite)
