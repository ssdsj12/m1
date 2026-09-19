"""One eager coupled RTI update with row-local last-safe publication."""

from dataclasses import dataclass

import torch

from .line_search import ParallelLineSearch
from .lq_problem import CoupledLqWorkspace, CoupledRtiInput
from .reduced_dynamics import GpuReducedDynamics
from .state_adapter import _FiniteRows, _workspace_device
from .warm_start import GpuWarmStart


REASON_ACCEPTED = 0
REASON_NO_SAFE_HORIZON = 1
REASON_INPUT_INVALID = 2
REASON_DYNAMICS_INVALID = 3
REASON_NOMINAL_INVALID = 4
REASON_LINE_SEARCH_REJECTED = 5
REASON_NONFINITE_RESULT = 6


@dataclass
class GpuRtiPlanResult:
    """Borrowed fixed-address result of one RTI iteration."""

    action: torch.Tensor
    accepted: torch.Tensor
    safe_available: torch.Tensor
    merit: torch.Tensor
    reason_code: torch.Tensor
    rti_iterations: int = 1


class EagerBimanualRtiPlanner:
    """Compose warm shifting, one dense direction, and one four-alpha search."""

    def __init__(
        self,
        *,
        batch: int,
        max_wrench_constraints: int,
        max_hard_constraints: int,
        device="cuda:0",
    ):
        self.batch = batch
        self.device = _workspace_device(batch, device)
        self.lq = CoupledLqWorkspace(
            batch=batch,
            max_wrench_constraints=max_wrench_constraints,
            max_hard_constraints=max_hard_constraints,
            device=self.device,
        )
        self.line_search = ParallelLineSearch(
            batch=batch,
            max_wrench_constraints=max_wrench_constraints,
            max_hard_constraints=max_hard_constraints,
            device=self.device,
        )
        self.warm = GpuWarmStart(batch=batch, state_dim=111, device=self.device)

        b = batch
        self.reporting_state = torch.zeros(b, 26, 111, dtype=torch.float32, device=self.device)
        self._fallback_action = torch.zeros(b, 25, 43, dtype=torch.float32, device=self.device)
        self._fallback_state = torch.zeros(b, 26, 111, dtype=torch.float32, device=self.device)
        self._storage_action = torch.zeros_like(self._fallback_action)
        self._storage_state = torch.zeros_like(self._fallback_state)
        self.action = torch.zeros_like(self._fallback_action)
        self.accepted = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.safe_available = torch.zeros_like(self.accepted)
        self.merit = torch.zeros(b, dtype=torch.float32, device=self.device)
        self.reason_code = torch.full(
            (b,), REASON_NO_SAFE_HORIZON, dtype=torch.int64, device=self.device
        )

        self._prior_safe = torch.zeros_like(self.accepted)
        self._nominal_valid = torch.zeros_like(self.accepted)
        self._reporting_finite = torch.zeros_like(self.accepted)
        self._arithmetic_finite = torch.zeros_like(self.accepted)
        self._publishable = torch.zeros_like(self.accepted)
        self._storage_mask = torch.zeros_like(self.accepted)
        self._mask = torch.zeros_like(self.accepted)
        self._mask2 = torch.zeros_like(self.accepted)
        self._q0_valid = torch.zeros_like(self.accepted)
        self._quat_valid = torch.zeros_like(self.accepted)

        self._q0 = torch.empty(b, 4, dtype=torch.float32, device=self.device)
        self._q0_norm = torch.empty(b, 1, dtype=torch.float32, device=self.device)
        self._q0_divisor = torch.empty_like(self._q0_norm)
        self._quat_exp = torch.empty(b, 26, 4, dtype=torch.float32, device=self.device)
        self._quat = torch.empty_like(self._quat_exp)
        self._quat_negative = torch.empty_like(self._quat)
        self._quat_product = torch.empty_like(self._quat)
        self._angle = torch.empty(b, 26, 1, dtype=torch.float32, device=self.device)
        self._angle_divisor = torch.empty_like(self._angle)
        self._half_angle = torch.empty_like(self._angle)
        self._scale = torch.empty_like(self._angle)
        self._quat_norm = torch.empty_like(self._angle)
        self._quat_divisor = torch.empty_like(self._angle)
        self._hemisphere_dot = torch.empty_like(self._angle)
        self._small_angle = torch.empty(b, 26, 1, dtype=torch.bool, device=self.device)
        self._negative_hemisphere = torch.empty_like(self._small_angle)
        self._quat_term = torch.empty(b, 26, dtype=torch.float32, device=self.device)
        self._finite = _FiniteRows(batch, 4 * 26 * 110, self.device)

        self._result = GpuRtiPlanResult(
            action=self.action,
            accepted=self.accepted,
            safe_available=self.safe_available,
            merit=self.merit,
            reason_code=self.reason_code,
        )

    def _reconstruct_quaternion(self, optimizer_state):
        tangent = optimizer_state[:, :, 3:6]
        torch.linalg.vector_norm(self.lq.base_quaternion0, dim=1, keepdim=True, out=self._q0_norm)
        self._q0_valid.copy_(self._finite.check(self.lq.base_quaternion0))
        torch.gt(self._q0_norm[:, 0], torch.finfo(torch.float32).eps, out=self._mask)
        self._q0_valid.logical_and_(self._mask)
        self._q0_divisor.copy_(self._q0_norm).clamp_min_(torch.finfo(torch.float32).eps)
        torch.div(self.lq.base_quaternion0, self._q0_divisor, out=self._q0)

        torch.linalg.vector_norm(tangent, dim=2, keepdim=True, out=self._angle)
        self._half_angle.copy_(self._angle).mul_(0.5)
        torch.cos(self._half_angle[:, :, 0], out=self._quat_exp[:, :, 0])
        torch.sin(self._half_angle, out=self._scale)
        self._angle_divisor.copy_(self._angle).clamp_min_(1.0e-10)
        self._scale.div_(self._angle_divisor)
        torch.le(self._angle, 1.0e-10, out=self._small_angle)
        self._scale.masked_fill_(self._small_angle, 0.5)
        torch.mul(tangent, self._scale, out=self._quat_exp[:, :, 1:4])

        q0 = self._q0
        delta = self._quat_exp
        quat = self._quat
        torch.mul(q0[:, None, 0], delta[:, :, 0], out=quat[:, :, 0])
        for component in range(1, 4):
            quat[:, :, 0].addcmul_(q0[:, None, component], delta[:, :, component], value=-1.0)

        torch.mul(q0[:, None, 0], delta[:, :, 1], out=quat[:, :, 1])
        quat[:, :, 1].addcmul_(q0[:, None, 1], delta[:, :, 0])
        quat[:, :, 1].addcmul_(q0[:, None, 2], delta[:, :, 3])
        quat[:, :, 1].addcmul_(q0[:, None, 3], delta[:, :, 2], value=-1.0)
        torch.mul(q0[:, None, 0], delta[:, :, 2], out=quat[:, :, 2])
        quat[:, :, 2].addcmul_(q0[:, None, 2], delta[:, :, 0])
        quat[:, :, 2].addcmul_(q0[:, None, 3], delta[:, :, 1])
        quat[:, :, 2].addcmul_(q0[:, None, 1], delta[:, :, 3], value=-1.0)
        torch.mul(q0[:, None, 0], delta[:, :, 3], out=quat[:, :, 3])
        quat[:, :, 3].addcmul_(q0[:, None, 3], delta[:, :, 0])
        quat[:, :, 3].addcmul_(q0[:, None, 1], delta[:, :, 2])
        quat[:, :, 3].addcmul_(q0[:, None, 2], delta[:, :, 1], value=-1.0)

        torch.linalg.vector_norm(quat, dim=2, keepdim=True, out=self._quat_norm)
        self._quat_valid.copy_(self._q0_valid)
        self._quat_valid.logical_and_(self._finite.check(quat))
        torch.gt(self._quat_norm[:, :, 0], torch.finfo(torch.float32).eps, out=self._small_angle[:, :, 0])
        torch.all(self._small_angle[:, :, 0], dim=1, out=self._mask)
        self._quat_valid.logical_and_(self._mask)
        self._quat_divisor.copy_(self._quat_norm).clamp_min_(torch.finfo(torch.float32).eps)
        quat.div_(self._quat_divisor)

        torch.mul(quat, q0[:, None], out=self._quat_product)
        torch.sum(self._quat_product, dim=2, keepdim=True, out=self._hemisphere_dot)
        torch.lt(self._hemisphere_dot, 0.0, out=self._negative_hemisphere)
        torch.neg(quat, out=self._quat_negative)
        torch.where(self._negative_hemisphere, self._quat_negative, quat, out=quat)
        self.reporting_state[:, :, 3:7].copy_(quat)

    def _build_reporting_state(self, optimizer_state):
        report = self.reporting_state
        report[:, :, 0:3].copy_(optimizer_state[:, :, 0:3])
        self._reconstruct_quaternion(optimizer_state)
        report[:, :, 7:13].copy_(optimizer_state[:, :, 6:12])
        report[:, :, 13:29].copy_(optimizer_state[:, :, 12:28])
        report[:, :, 29:45].copy_(optimizer_state[:, :, 55:71])
        report[:, :, 45].copy_(optimizer_state[:, :, 28])
        report[:, :, 46].copy_(optimizer_state[:, :, 71])
        report[:, :, 47:54].copy_(optimizer_state[:, :, 29:36])
        report[:, :, 54:61].copy_(optimizer_state[:, :, 72:79])
        report[:, :, 61:68].copy_(optimizer_state[:, :, 36:43])
        report[:, :, 68:75].copy_(optimizer_state[:, :, 79:86])
        report[:, :, 75:81].copy_(optimizer_state[:, :, 43:49])
        report[:, :, 81:87].copy_(optimizer_state[:, :, 86:92])
        report[:, :, 87:93].copy_(optimizer_state[:, :, 49:55])
        report[:, :, 93:99].copy_(optimizer_state[:, :, 92:98])
        report[:, :, 99:111].copy_(optimizer_state[:, :, 98:110])
        self._reporting_finite.copy_(self._quat_valid)
        self._reporting_finite.logical_and_(self._finite.check(report))

    def _check_arithmetic(self):
        self._arithmetic_finite.fill_(True)
        for value in (
            self.lq.state,
            self.lq.direction,
            self.line_search.candidate_control,
            self.lq.candidate_state,
            self.line_search.merit,
            self.line_search.nominal_merit,
        ):
            self._arithmetic_finite.logical_and_(self._finite.check(value))

    def _set_reasons(self, problem, dynamics):
        self.reason_code.fill_(REASON_LINE_SEARCH_REJECTED)
        torch.logical_not(self._nominal_valid, out=self._mask)
        self.reason_code.masked_fill_(self._mask, REASON_NOMINAL_INVALID)
        torch.logical_not(self._arithmetic_finite, out=self._mask)
        torch.logical_not(self._reporting_finite, out=self._mask2)
        self._mask.logical_or_(self._mask2)
        torch.logical_not(self.lq.valid, out=self._mask2)
        self._mask.logical_or_(self._mask2)
        self._mask.logical_and_(problem.input_valid)
        self._mask.logical_and_(dynamics.valid)
        self.reason_code.masked_fill_(self._mask, REASON_NONFINITE_RESULT)
        torch.logical_not(dynamics.valid, out=self._mask)
        self.reason_code.masked_fill_(self._mask, REASON_DYNAMICS_INVALID)
        torch.logical_not(problem.input_valid, out=self._mask)
        self.reason_code.masked_fill_(self._mask, REASON_INPUT_INVALID)
        self.reason_code.masked_fill_(self.accepted, REASON_ACCEPTED)
        torch.logical_not(self.safe_available, out=self._mask)
        torch.logical_not(self.accepted, out=self._mask2)
        self._mask.logical_and_(self._mask2)
        self.reason_code.masked_fill_(self._mask, REASON_NO_SAFE_HORIZON)

    def step(
        self,
        problem: CoupledRtiInput,
        dynamics: GpuReducedDynamics,
        identity: torch.Tensor,
        reset: torch.Tensor,
    ) -> GpuRtiPlanResult:
        self.warm.step(problem.measured_state, identity, reset=reset)
        self._prior_safe.copy_(self.warm.valid)
        self._fallback_action.copy_(self.warm.shifted_action)
        self._fallback_state.copy_(self.warm.shifted_state)

        self.lq.assemble(problem, dynamics)
        self.lq.rollout(self.lq.nominal_control)
        self._nominal_valid.copy_(self.lq.validate(self.lq.state, self.lq.nominal_control))
        self.lq.linearize()
        self.lq.solve_direction()
        candidate = self.line_search.step(self.lq)
        self._build_reporting_state(candidate.state)
        self._check_arithmetic()

        self._publishable.copy_(candidate.accepted)
        self._publishable.logical_and_(self._nominal_valid)
        self._publishable.logical_and_(self.lq.valid)
        self._publishable.logical_and_(self._reporting_finite)
        self._publishable.logical_and_(self._arithmetic_finite)
        self.accepted.copy_(self._publishable)

        self._storage_action.zero_()
        self._storage_state.zero_()
        torch.where(
            self._prior_safe[:, None, None],
            self._fallback_action,
            self._storage_action,
            out=self._storage_action,
        )
        torch.where(
            self._prior_safe[:, None, None],
            self._fallback_state,
            self._storage_state,
            out=self._storage_state,
        )
        torch.where(
            self._publishable[:, None, None],
            candidate.control[:, :, :43],
            self._storage_action,
            out=self._storage_action,
        )
        torch.where(
            self._publishable[:, None, None],
            self.reporting_state,
            self._storage_state,
            out=self._storage_state,
        )
        self._storage_mask.copy_(self._prior_safe).logical_or_(self._publishable)
        self.warm.accept(self._storage_action, self._storage_state, self._storage_mask, identity)

        self.action.copy_(self._storage_action)
        self.safe_available.copy_(self._storage_mask)
        self.merit.copy_(candidate.merit)
        self._set_reasons(problem, dynamics)
        return self._result


__all__ = ["EagerBimanualRtiPlanner", "GpuRtiPlanResult"]
