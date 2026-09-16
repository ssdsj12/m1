"""Persistent complete-horizon line search for the coupled GPU RTI problem."""

from dataclasses import dataclass

import torch

from .lq_problem import CoupledLqWorkspace
from .state_adapter import _workspace_device


@dataclass
class LineSearchResult:
    control: torch.Tensor
    state: torch.Tensor
    accepted: torch.Tensor
    alpha_index: torch.Tensor
    merit: torch.Tensor


class ParallelLineSearch:
    """Evaluate four fixed candidates and select one complete trajectory per row."""

    def __init__(
        self,
        *,
        batch: int,
        max_wrench_constraints: int,
        max_hard_constraints: int,
        device="cuda:0",
    ):
        if type(max_wrench_constraints) is not int or max_wrench_constraints < 0:
            raise ValueError("max_wrench_constraints must be a nonnegative integer")
        if type(max_hard_constraints) is not int or max_hard_constraints < 0:
            raise ValueError("max_hard_constraints must be a nonnegative integer")
        self.batch = batch
        self.device = _workspace_device(batch, device)
        self.max_wrench_constraints = max_wrench_constraints
        self.max_hard_constraints = max_hard_constraints
        b, candidates, horizon, state_dim, control_dim = batch, 4, 25, 110, 55
        f32 = torch.float32

        self.alphas = torch.tensor((1.0, 0.5, 0.25, 0.125), dtype=f32, device=self.device)
        self._alpha_view = self.alphas.view(1, candidates, 1, 1)
        self.direction = torch.empty(b, horizon, control_dim, dtype=f32, device=self.device)
        self._direction_flat = self.direction.view(b, horizon * control_dim)
        self._direction_view = self.direction.view(b, 1, horizon, control_dim)
        self._candidate_delta = torch.empty(b, candidates, horizon, control_dim, dtype=f32, device=self.device)
        self.candidate_control = torch.empty_like(self._candidate_delta)
        self._equality_elements = torch.empty(b, horizon, control_dim, dtype=torch.bool, device=self.device)

        self._candidate_state_residual = torch.empty(
            b, candidates, horizon, state_dim, dtype=f32, device=self.device
        )
        self._candidate_terminal_residual = torch.empty(b, candidates, state_dim, dtype=f32, device=self.device)
        self._candidate_control_square = torch.empty_like(self.candidate_control)
        self._candidate_palm_residual = torch.empty(b, candidates, horizon, 12, dtype=f32, device=self.device)
        self.merit = torch.empty(b, candidates, dtype=f32, device=self.device)
        self._merit_add = torch.empty_like(self.merit)
        self.masked_merit = torch.empty_like(self.merit)
        self.candidate_valid = torch.empty(b, candidates, dtype=torch.bool, device=self.device)
        self._candidate_merit_finite = torch.empty_like(self.candidate_valid)
        self._invalid_candidate = torch.empty_like(self.candidate_valid)
        self._candidate_merit_absolute = torch.empty_like(self.merit)

        self._nominal_state_residual = torch.empty(b, horizon, state_dim, dtype=f32, device=self.device)
        self._nominal_terminal_residual = torch.empty(b, state_dim, dtype=f32, device=self.device)
        self._nominal_control_square = torch.empty(b, horizon, control_dim, dtype=f32, device=self.device)
        self._nominal_palm_residual = torch.empty(b, horizon, 12, dtype=f32, device=self.device)
        self.nominal_merit = torch.empty(b, dtype=f32, device=self.device)
        self._nominal_add = torch.empty_like(self.nominal_merit)
        self.nominal_valid = torch.empty(b, dtype=torch.bool, device=self.device)
        self._nominal_merit_finite = torch.empty_like(self.nominal_valid)
        self._nominal_merit_absolute = torch.empty_like(self.nominal_merit)
        self._nominal_eligible = torch.empty_like(self.nominal_valid)

        self.best_merit = torch.empty(b, dtype=f32, device=self.device)
        self.best_index = torch.empty(b, dtype=torch.int64, device=self.device)
        self.accepted = torch.empty(b, dtype=torch.bool, device=self.device)
        self._normal_accepted = torch.empty_like(self.accepted)
        self._equality_accepted = torch.empty_like(self.accepted)
        self._not_equality = torch.empty_like(self.accepted)
        self._candidate_mask = torch.empty_like(self.accepted)
        self.alpha_index = torch.empty(b, dtype=torch.int64, device=self.device)
        self._zero_index = torch.zeros_like(self.alpha_index)

        self.result_control = torch.empty(b, horizon, control_dim, dtype=f32, device=self.device)
        self.result_state = torch.empty(b, horizon + 1, state_dim, dtype=f32, device=self.device)
        self.result_merit = torch.empty(b, dtype=f32, device=self.device)
        self._result = LineSearchResult(
            control=self.result_control,
            state=self.result_state,
            accepted=self.accepted,
            alpha_index=self.alpha_index,
            merit=self.result_merit,
        )

    def _check_workspace(self, workspace):
        if not isinstance(workspace, CoupledLqWorkspace):
            raise ValueError("workspace must be CoupledLqWorkspace")
        if (
            workspace.batch != self.batch
            or workspace.device != self.device
            or workspace.max_wrench_constraints != self.max_wrench_constraints
            or workspace.max_hard_constraints != self.max_hard_constraints
        ):
            raise ValueError("workspace does not match line-search batch/device/constraint capacities")

    def _candidate_merits(self, workspace, state):
        torch.sub(
            state[:, :, 1:],
            workspace.state_target[:, None, 1:],
            out=self._candidate_state_residual,
        )
        self._candidate_state_residual.square_().mul_(workspace.state_weight[None, None, None])
        torch.sum(self._candidate_state_residual, dim=(2, 3), out=self.merit)

        torch.sub(
            state[:, :, -1],
            workspace.state_target[:, None, -1],
            out=self._candidate_terminal_residual,
        )
        self._candidate_terminal_residual.square_().mul_(workspace.terminal_weight[None, None])
        torch.sum(self._candidate_terminal_residual, dim=2, out=self._merit_add)
        self.merit.add_(self._merit_add)

        torch.square(self.candidate_control, out=self._candidate_control_square)
        self._candidate_control_square.mul_(workspace.control_weight[None, None, None])
        torch.sum(self._candidate_control_square, dim=(2, 3), out=self._merit_add)
        self.merit.add_(self._merit_add)

        for candidate in range(4):
            for node in range(25):
                torch.bmm(
                    workspace.palm_state_jacobian[:, node],
                    state[:, candidate, node + 1, :, None],
                    out=self._candidate_palm_residual[:, candidate, node, :, None],
                )
                self._candidate_palm_residual[:, candidate, node].add_(
                    workspace.palm_residual_offset[:, node]
                )
        self._candidate_palm_residual.square_().mul_(workspace.palm_weight[None, None, None])
        torch.sum(self._candidate_palm_residual, dim=(2, 3), out=self._merit_add)
        self.merit.add_(self._merit_add)

    def _nominal_merits(self, workspace, state):
        torch.sub(state[:, 1:], workspace.state_target[:, 1:], out=self._nominal_state_residual)
        self._nominal_state_residual.square_().mul_(workspace.state_weight[None, None])
        torch.sum(self._nominal_state_residual, dim=(1, 2), out=self.nominal_merit)

        torch.sub(state[:, -1], workspace.state_target[:, -1], out=self._nominal_terminal_residual)
        self._nominal_terminal_residual.square_().mul_(workspace.terminal_weight[None])
        torch.sum(self._nominal_terminal_residual, dim=1, out=self._nominal_add)
        self.nominal_merit.add_(self._nominal_add)

        torch.square(workspace.nominal_control, out=self._nominal_control_square)
        self._nominal_control_square.mul_(workspace.control_weight[None, None])
        torch.sum(self._nominal_control_square, dim=(1, 2), out=self._nominal_add)
        self.nominal_merit.add_(self._nominal_add)

        for node in range(25):
            torch.bmm(
                workspace.palm_state_jacobian[:, node],
                state[:, node + 1, :, None],
                out=self._nominal_palm_residual[:, node, :, None],
            )
            self._nominal_palm_residual[:, node].add_(workspace.palm_residual_offset[:, node])
        self._nominal_palm_residual.square_().mul_(workspace.palm_weight[None, None])
        torch.sum(self._nominal_palm_residual, dim=(1, 2), out=self._nominal_add)
        self.nominal_merit.add_(self._nominal_add)

    def step(self, workspace: CoupledLqWorkspace) -> LineSearchResult:
        self._check_workspace(workspace)
        self._direction_flat.copy_(workspace.direction)
        torch.mul(self._direction_view, self._alpha_view, out=self._candidate_delta)
        self.candidate_control.copy_(workspace.nominal_control[:, None])
        self.candidate_control.add_(self._candidate_delta)
        torch.maximum(
            self.candidate_control,
            workspace.control_lower[:, None],
            out=self.candidate_control,
        )
        torch.minimum(
            self.candidate_control,
            workspace.control_upper[:, None],
            out=self.candidate_control,
        )
        torch.eq(workspace.control_lower, workspace.control_upper, out=self._equality_elements)
        torch.where(
            self._equality_elements[:, None],
            workspace.nominal_control[:, None],
            self.candidate_control,
            out=self.candidate_control,
        )

        nominal_state = workspace.rollout(workspace.nominal_control)
        self.nominal_valid.copy_(workspace.validate(nominal_state, workspace.nominal_control))
        self._nominal_merits(workspace, nominal_state)
        torch.abs(self.nominal_merit, out=self._nominal_merit_absolute)
        torch.le(
            self._nominal_merit_absolute,
            torch.finfo(torch.float32).max,
            out=self._nominal_merit_finite,
        )
        self._nominal_eligible.copy_(self.nominal_valid).logical_and_(self._nominal_merit_finite)

        candidate_state = workspace.rollout_candidates(self.candidate_control)
        self.candidate_valid.copy_(workspace.validate_candidates(candidate_state, self.candidate_control))
        self._candidate_merits(workspace, candidate_state)
        torch.abs(self.merit, out=self._candidate_merit_absolute)
        torch.le(
            self._candidate_merit_absolute,
            torch.finfo(torch.float32).max,
            out=self._candidate_merit_finite,
        )
        self.candidate_valid.logical_and_(self._candidate_merit_finite)
        self.masked_merit.copy_(self.merit)
        torch.logical_not(self.candidate_valid, out=self._invalid_candidate)
        self.masked_merit.masked_fill_(self._invalid_candidate, float("inf"))
        torch.min(self.masked_merit, dim=1, out=(self.best_merit, self.best_index))

        torch.abs(self.best_merit, out=self._nominal_merit_absolute)
        torch.le(
            self._nominal_merit_absolute,
            torch.finfo(torch.float32).max,
            out=self._normal_accepted,
        )
        self._normal_accepted.logical_and_(self._nominal_eligible)
        torch.lt(self.best_merit, self.nominal_merit, out=self._candidate_mask)
        self._normal_accepted.logical_and_(self._candidate_mask)
        torch.logical_not(workspace.equality_bound, out=self._not_equality)
        self._normal_accepted.logical_and_(self._not_equality)
        self._equality_accepted.copy_(workspace.equality_bound).logical_and_(self._nominal_eligible)
        self.accepted.copy_(self._normal_accepted).logical_or_(self._equality_accepted)

        self.result_control.copy_(workspace.nominal_control)
        self.result_state.copy_(nominal_state)
        for candidate in range(4):
            torch.eq(self.best_index, candidate, out=self._candidate_mask)
            self._candidate_mask.logical_and_(self._normal_accepted)
            torch.where(
                self._candidate_mask[:, None, None],
                self.candidate_control[:, candidate],
                self.result_control,
                out=self.result_control,
            )
            torch.where(
                self._candidate_mask[:, None, None],
                candidate_state[:, candidate],
                self.result_state,
                out=self.result_state,
            )

        self.alpha_index.fill_(-1)
        torch.where(
            self._normal_accepted,
            self.best_index,
            self.alpha_index,
            out=self.alpha_index,
        )
        torch.where(
            self._equality_accepted,
            self._zero_index,
            self.alpha_index,
            out=self.alpha_index,
        )
        self.result_merit.fill_(float("inf"))
        torch.where(
            self._normal_accepted,
            self.best_merit,
            self.result_merit,
            out=self.result_merit,
        )
        torch.where(
            self._equality_accepted,
            self.nominal_merit,
            self.result_merit,
            out=self.result_merit,
        )
        return self._result
