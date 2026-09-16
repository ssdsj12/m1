"""Fixed-shape coupled robot/box rollout and dense correctness LQ backend."""

from dataclasses import dataclass

import torch

from .reduced_dynamics import GpuReducedDynamics
from .state_adapter import _FiniteRows, _require, _workspace_device


DT = 0.04
HALF_DT_SQUARED = 0.5 * DT * DT
CONTROL_VECTOR_DIM = 25 * 55


@dataclass(frozen=True)
class CoupledRtiDims:
    horizon: int = 25
    state_dim: int = 110
    effort_dim: int = 43
    wrench_dim: int = 12
    control_dim: int = 55


@dataclass
class CoupledRtiInput:
    measured_state: torch.Tensor
    base_pose0: torch.Tensor
    base_twist0: torch.Tensor
    base_quaternion0: torch.Tensor
    active_q0: torch.Tensor
    active_qd0: torch.Tensor
    box_pose0: torch.Tensor
    box_twist0: torch.Tensor
    active_selector: torch.Tensor
    palm_jacobian: torch.Tensor
    box_wrench_map: torch.Tensor
    box_gravity: torch.Tensor
    nominal_control: torch.Tensor
    control_lower: torch.Tensor
    control_upper: torch.Tensor
    state_target: torch.Tensor
    state_weight: torch.Tensor
    control_weight: torch.Tensor
    terminal_weight: torch.Tensor
    palm_residual_offset: torch.Tensor
    palm_state_jacobian: torch.Tensor
    palm_weight: torch.Tensor
    state_lower: torch.Tensor
    state_upper: torch.Tensor
    wrench_inequality_matrix: torch.Tensor
    wrench_inequality_upper: torch.Tensor
    hard_inequality_matrix: torch.Tensor
    hard_inequality_upper: torch.Tensor
    input_valid: torch.Tensor


class CoupledLqWorkspace:
    """Persistent eager CUDA workspace for the exact dense H25 LQ problem.

    All returned tensors are borrowed.  Resident methods use preallocated
    outputs; CUDA solver scratch is expected to be warmed before leak checks.
    """

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
        self.dims = CoupledRtiDims()
        self.max_wrench_constraints = max_wrench_constraints
        self.max_hard_constraints = max_hard_constraints
        b, h, nx, nu = batch, 25, 110, 55
        f32 = torch.float32

        # Staged input contract.
        row_shapes = {
            "measured_state": (111,),
            "base_pose0": (6,),
            "base_twist0": (6,),
            "base_quaternion0": (4,),
            "active_q0": (43,),
            "active_qd0": (43,),
            "box_pose0": (6,),
            "box_twist0": (6,),
            "active_selector": (43, 59),
            "palm_jacobian": (2, 6, 59),
            "box_wrench_map": (6, 12),
            "box_gravity": (6,),
            "nominal_control": (h, nu),
            "control_lower": (h, nu),
            "control_upper": (h, nu),
            "state_target": (h + 1, nx),
            "palm_residual_offset": (h, 12),
            "palm_state_jacobian": (h, 12, nx),
            "state_lower": (h + 1, nx),
            "state_upper": (h + 1, nx),
            "wrench_inequality_matrix": (h, max_wrench_constraints, 12),
            "wrench_inequality_upper": (h, max_wrench_constraints),
            "hard_inequality_matrix": (h, max_hard_constraints, nu),
            "hard_inequality_upper": (h, max_hard_constraints),
        }
        self._row_inputs = {}
        for name, shape in row_shapes.items():
            value = torch.empty((b, *shape), dtype=f32, device=self.device)
            self._row_inputs[name] = value
            setattr(self, name, value)
        self.input_valid = torch.empty(b, dtype=torch.bool, device=self.device)
        self.state_weight = torch.empty(nx, dtype=f32, device=self.device)
        self.control_weight = torch.empty(nu, dtype=f32, device=self.device)
        self.terminal_weight = torch.empty(nx, dtype=f32, device=self.device)
        self.palm_weight = torch.empty(12, dtype=f32, device=self.device)

        # Borrowed-dynamics augmentation.
        self.wrench_rhs = torch.zeros(b, 71, 12, dtype=f32, device=self.device)
        self.wrench_solution = torch.zeros_like(self.wrench_rhs)
        self.qdd_wrench = torch.zeros(b, 59, 12, dtype=f32, device=self.device)
        self.qdd_offset = torch.zeros(b, 59, dtype=f32, device=self.device)
        self.qdd_from_effort = torch.zeros(b, 59, 43, dtype=f32, device=self.device)
        self._wrench_lu = torch.empty(b, 71, 71, dtype=f32, device=self.device)
        self._wrench_pivots = torch.empty(b, 71, dtype=torch.int32, device=self.device)
        self._identity71 = torch.eye(71, dtype=f32, device=self.device).expand(b, -1, -1)
        self._identity_pivots = torch.arange(1, 72, dtype=torch.int32, device=self.device).expand(b, -1)

        # One-step affine dynamics and public rollouts.
        self.A = torch.zeros(b, h, nx, nx, dtype=f32, device=self.device)
        self.B = torch.zeros(b, h, nx, nu, dtype=f32, device=self.device)
        self.c = torch.zeros(b, h, nx, dtype=f32, device=self.device)
        self.state = torch.zeros(b, h + 1, nx, dtype=f32, device=self.device)
        self.control = torch.zeros(b, h, nu, dtype=f32, device=self.device)
        self.arm_substep_state = torch.zeros(b, h, 2, 98, dtype=f32, device=self.device)
        self.hand_substep_state = torch.zeros(b, h, 4, 24, dtype=f32, device=self.device)
        self.candidate_state = torch.zeros(b, 4, h + 1, nx, dtype=f32, device=self.device)
        self._candidate_control = torch.zeros(b, 4, h, nu, dtype=f32, device=self.device)
        self._candidate_arm = torch.zeros(b, 4, h, 2, 98, dtype=f32, device=self.device)
        self._candidate_hand = torch.zeros(b, 4, h, 4, 24, dtype=f32, device=self.device)
        self._validation_state = torch.zeros_like(self.state)
        self._validation_arm = torch.zeros_like(self.arm_substep_state)
        self._validation_hand = torch.zeros_like(self.hand_substep_state)

        # Transition construction and rollout scratch.
        self._identity6 = torch.eye(6, dtype=f32, device=self.device)
        self._dt_identity6 = self._identity6 * DT
        self._identity43 = torch.eye(43, dtype=f32, device=self.device)
        self._dt_identity43 = self._identity43 * DT
        self._active_effort = torch.empty(b, 43, 43, dtype=f32, device=self.device)
        self._active_wrench = torch.empty(b, 43, 12, dtype=f32, device=self.device)
        self._active_offset = torch.empty(b, 43, dtype=f32, device=self.device)
        self._base_effort = torch.empty(b, 6, 43, dtype=f32, device=self.device)
        self._base_wrench = torch.empty(b, 6, 12, dtype=f32, device=self.device)
        self._base_offset = torch.empty(b, 6, dtype=f32, device=self.device)
        self._x0 = torch.empty(b, nx, dtype=f32, device=self.device)
        self._qdd = torch.empty(b, 59, dtype=f32, device=self.device)
        self._qdd_add = torch.empty_like(self._qdd)
        self._active_qdd = torch.empty(b, 43, dtype=f32, device=self.device)
        self._box_accel = torch.empty(b, 6, dtype=f32, device=self.device)
        self._next_state = torch.empty(b, nx, dtype=f32, device=self.device)
        self._state_add = torch.empty_like(self._next_state)

        # Dense condensed Gauss-Newton storage.
        self.state_control_jacobian = torch.zeros(b, h, nx, CONTROL_VECTOR_DIM, dtype=f32, device=self.device)
        self._weighted_jacobian = torch.empty(b, nx, CONTROL_VECTOR_DIM, dtype=f32, device=self.device)
        self._palm_control_jacobian = torch.empty(b, 12, CONTROL_VECTOR_DIM, dtype=f32, device=self.device)
        self._weighted_palm_jacobian = torch.empty_like(self._palm_control_jacobian)
        self._state_residual = torch.empty(b, h, nx, dtype=f32, device=self.device)
        self._weighted_state_residual = torch.empty(b, nx, dtype=f32, device=self.device)
        self._palm_residual = torch.empty(b, h, 12, dtype=f32, device=self.device)
        self._weighted_palm_residual = torch.empty(b, 12, dtype=f32, device=self.device)
        self.H = torch.zeros(b, CONTROL_VECTOR_DIM, CONTROL_VECTOR_DIM, dtype=f32, device=self.device)
        self.gradient = torch.zeros(b, CONTROL_VECTOR_DIM, dtype=f32, device=self.device)
        self.direction = torch.zeros_like(self.gradient)
        self._h_increment = torch.empty_like(self.H)
        self._g_increment = torch.empty(b, CONTROL_VECTOR_DIM, 1, dtype=f32, device=self.device)
        self._solve_matrix = torch.empty_like(self.H)
        self._factor = torch.empty_like(self.H)
        self._pivots = torch.empty(b, CONTROL_VECTOR_DIM, dtype=torch.int32, device=self.device)
        self._factor_info = torch.empty(b, dtype=torch.int32, device=self.device)
        self._solve_rhs = torch.empty(b, CONTROL_VECTOR_DIM, 1, dtype=f32, device=self.device)
        self._solve_output = torch.empty_like(self._solve_rhs)
        self._zero_direction = torch.zeros_like(self.direction)
        self._identity_h = torch.eye(CONTROL_VECTOR_DIM, dtype=f32, device=self.device).expand(b, -1, -1)
        self._identity_h_pivots = torch.arange(1, CONTROL_VECTOR_DIM + 1, dtype=torch.int32, device=self.device).expand(b, -1)
        self._combined_state_weight = torch.empty(nx, dtype=f32, device=self.device)

        # Validation and invariant scratch.
        self.valid = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.validation_valid = torch.zeros_like(self.valid)
        self.candidate_valid = torch.zeros(b, 4, dtype=torch.bool, device=self.device)
        self._mask = torch.empty_like(self.valid)
        self._invalid = torch.empty_like(self.valid)
        self._weight_valid = torch.empty((), dtype=torch.bool, device=self.device)
        self._weight_invalid = torch.empty_like(self._weight_valid)
        self._weight_absolute = torch.empty(nx, dtype=f32, device=self.device)
        self._weight_elements = torch.empty(nx, dtype=torch.bool, device=self.device)
        self._finite = _FiniteRows(b, h * 12 * nx, self.device)
        self._state_elements = torch.empty_like(self.state, dtype=torch.bool)
        self._control_elements = torch.empty_like(self.control, dtype=torch.bool)
        self._arm_elements = torch.empty_like(self.arm_substep_state, dtype=torch.bool)
        self._hand_elements = torch.empty_like(self.hand_substep_state, dtype=torch.bool)
        self._hand_lower = torch.empty(b, h, 24, dtype=f32, device=self.device)
        self._hand_upper = torch.empty_like(self._hand_lower)
        self._wrench_lhs = torch.empty(b, h, max_wrench_constraints, dtype=f32, device=self.device)
        self._wrench_elements = torch.empty_like(self._wrench_lhs, dtype=torch.bool)
        self._hard_lhs = torch.empty(b, h, max_hard_constraints, dtype=f32, device=self.device)
        self._hard_elements = torch.empty_like(self._hard_lhs, dtype=torch.bool)
        self._equality_elements = torch.empty_like(self.control, dtype=torch.bool)
        self.equality_bound = torch.empty(b, dtype=torch.bool, device=self.device)
        self.rejected_count = torch.zeros(b, dtype=torch.int64, device=self.device)

    def _validate_metadata(self, problem):
        if not isinstance(problem, CoupledRtiInput):
            raise ValueError("problem must be CoupledRtiInput")
        for name, destination in self._row_inputs.items():
            _require(getattr(problem, name), destination.shape, self.device)
        _require(problem.input_valid, (self.batch,), self.device, torch.bool)
        for name, width in (
            ("state_weight", 110),
            ("control_weight", 55),
            ("terminal_weight", 110),
            ("palm_weight", 12),
        ):
            _require(getattr(problem, name), (width,), self.device)

    def _check_weight(self, weight):
        width = weight.shape[0]
        absolute = self._weight_absolute[:width]
        elements = self._weight_elements[:width]
        torch.abs(weight, out=absolute)
        torch.le(absolute, torch.finfo(torch.float32).max, out=elements)
        torch.all(elements, out=self._weight_valid)
        torch.ge(weight, 0.0, out=elements)
        torch.all(elements, out=self._weight_invalid)
        self._weight_valid.logical_and_(self._weight_invalid)
        return self._weight_valid

    def assemble(self, problem: CoupledRtiInput, dynamics: GpuReducedDynamics):
        self._validate_metadata(problem)
        if not isinstance(dynamics, GpuReducedDynamics) or dynamics.batch != self.batch or dynamics.device != self.device:
            raise ValueError("dynamics workspace does not match batch/device")
        for name, destination in self._row_inputs.items():
            destination.copy_(getattr(problem, name))
        self.input_valid.copy_(problem.input_valid)
        self.state_weight.copy_(problem.state_weight)
        self.control_weight.copy_(problem.control_weight)
        self.terminal_weight.copy_(problem.terminal_weight)
        self.palm_weight.copy_(problem.palm_weight)

        self.valid.copy_(self.input_valid).logical_and_(dynamics.valid)
        for value in self._row_inputs.values():
            self.valid.logical_and_(self._finite.check(value))
        for weight in (self.state_weight, self.control_weight, self.terminal_weight, self.palm_weight):
            self.valid.logical_and_(self._check_weight(weight))
        torch.le(self.control_lower, self.control_upper, out=self._control_elements)
        torch.all(self._control_elements, dim=(1, 2), out=self._mask)
        self.valid.logical_and_(self._mask)
        torch.le(self.state_lower, self.state_upper, out=self._state_elements)
        torch.all(self._state_elements, dim=(1, 2), out=self._mask)
        self.valid.logical_and_(self._mask)

        # Stage the validated borrowed factorization; never mutate Task3 buffers.
        torch.logical_not(self.valid, out=self._invalid)
        torch.where(self.valid[:, None, None], dynamics.lu, self._identity71, out=self._wrench_lu)
        torch.where(self.valid[:, None], dynamics.pivots, self._identity_pivots, out=self._wrench_pivots)
        self.wrench_rhs.zero_()
        torch.neg(self.palm_jacobian[:, 0].transpose(1, 2), out=self.wrench_rhs[:, :59, :6])
        torch.neg(self.palm_jacobian[:, 1].transpose(1, 2), out=self.wrench_rhs[:, :59, 6:])
        self.wrench_rhs.masked_fill_(self._invalid[:, None, None], 0.0)
        torch.linalg.lu_solve(self._wrench_lu, self._wrench_pivots, self.wrench_rhs, out=self.wrench_solution)
        self.valid.logical_and_(self._finite.check(self.wrench_solution))
        torch.where(self.valid[:, None, None], self.wrench_solution[:, :59], self.qdd_wrench, out=self.qdd_wrench)
        torch.where(self.valid[:, None], dynamics.qdd_offset, self.qdd_offset, out=self.qdd_offset)
        torch.where(self.valid[:, None, None], dynamics.qdd_from_effort, self.qdd_from_effort, out=self.qdd_from_effort)

        torch.bmm(self.active_selector, self.qdd_from_effort, out=self._active_effort)
        torch.bmm(self.active_selector, self.qdd_wrench, out=self._active_wrench)
        torch.bmm(self.active_selector, self.qdd_offset[:, :, None], out=self._active_offset[:, :, None])
        self._base_effort.copy_(self.qdd_from_effort[:, :6])
        self._base_wrench.copy_(self.qdd_wrench[:, :6])
        self._base_offset.copy_(self.qdd_offset[:, :6])

        self.A.zero_()
        self.A.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        self.A[:, :, 0:6, 6:12].copy_(self._dt_identity6)
        self.A[:, :, 12:55, 55:98].copy_(self._dt_identity43)
        self.A[:, :, 98:104, 104:110].copy_(self._dt_identity6)
        self.B.zero_()
        for k in range(25):
            self.B[:, k, 0:6, :43].copy_(self._base_effort).mul_(HALF_DT_SQUARED)
            self.B[:, k, 0:6, 43:].copy_(self._base_wrench).mul_(HALF_DT_SQUARED)
            self.B[:, k, 6:12, :43].copy_(self._base_effort).mul_(DT)
            self.B[:, k, 6:12, 43:].copy_(self._base_wrench).mul_(DT)
            self.B[:, k, 12:55, :43].copy_(self._active_effort).mul_(HALF_DT_SQUARED)
            self.B[:, k, 12:55, 43:].copy_(self._active_wrench).mul_(HALF_DT_SQUARED)
            self.B[:, k, 55:98, :43].copy_(self._active_effort).mul_(DT)
            self.B[:, k, 55:98, 43:].copy_(self._active_wrench).mul_(DT)
            self.B[:, k, 98:104, 43:].copy_(self.box_wrench_map).mul_(HALF_DT_SQUARED)
            self.B[:, k, 104:110, 43:].copy_(self.box_wrench_map).mul_(DT)
        self.c.zero_()
        self.c[:, :, 0:6].copy_(self._base_offset[:, None]).mul_(HALF_DT_SQUARED)
        self.c[:, :, 6:12].copy_(self._base_offset[:, None]).mul_(DT)
        self.c[:, :, 12:55].copy_(self._active_offset[:, None]).mul_(HALF_DT_SQUARED)
        self.c[:, :, 55:98].copy_(self._active_offset[:, None]).mul_(DT)
        self.c[:, :, 98:104].copy_(self.box_gravity[:, None]).mul_(HALF_DT_SQUARED)
        self.c[:, :, 104:110].copy_(self.box_gravity[:, None]).mul_(DT)
        self.A.masked_fill_(self._invalid[:, None, None, None], 0.0)
        self.B.masked_fill_(self._invalid[:, None, None, None], 0.0)
        self.c.masked_fill_(self._invalid[:, None, None], 0.0)

        self._x0[:, 0:6].copy_(self.base_pose0)
        self._x0[:, 6:12].copy_(self.base_twist0)
        self._x0[:, 12:55].copy_(self.active_q0)
        self._x0[:, 55:98].copy_(self.active_qd0)
        self._x0[:, 98:104].copy_(self.box_pose0)
        self._x0[:, 104:110].copy_(self.box_twist0)
        self._x0.masked_fill_(self._invalid[:, None], 0.0)
        self._hand_lower[:, :, :12].copy_(self.state_lower[:, 1:, 43:55])
        self._hand_lower[:, :, 12:].copy_(self.state_lower[:, 1:, 86:98])
        self._hand_upper[:, :, :12].copy_(self.state_upper[:, 1:, 43:55])
        self._hand_upper[:, :, 12:].copy_(self.state_upper[:, 1:, 86:98])
        torch.eq(self.control_lower, self.control_upper, out=self._equality_elements)
        torch.all(self._equality_elements, dim=(1, 2), out=self.equality_bound)
        torch.logical_not(self.valid, out=self._mask)
        self.rejected_count.add_(self._mask)
        return self.valid

    def _rollout_core(self, control, state, arm, hand):
        state.zero_()
        state[:, 0].copy_(self._x0)
        for k in range(25):
            torch.bmm(self.qdd_from_effort, control[:, k, :43, None], out=self._qdd[:, :, None])
            self._qdd.add_(self.qdd_offset)
            torch.bmm(self.qdd_wrench, control[:, k, 43:, None], out=self._qdd_add[:, :, None])
            self._qdd.add_(self._qdd_add)
            torch.bmm(self.active_selector, self._qdd[:, :, None], out=self._active_qdd[:, :, None])
            torch.bmm(self.box_wrench_map, control[:, k, 43:, None], out=self._box_accel[:, :, None])
            self._box_accel.add_(self.box_gravity)

            torch.bmm(self.A[:, k], state[:, k, :, None], out=self._next_state[:, :, None])
            torch.bmm(self.B[:, k], control[:, k, :, None], out=self._state_add[:, :, None])
            self._next_state.add_(self._state_add).add_(self.c[:, k])
            state[:, k + 1].copy_(self._next_state)

            for substep, elapsed in enumerate((0.02, 0.04)):
                arm[:, k, substep, 0:6].copy_(state[:, k, 0:6])
                arm[:, k, substep, 0:6].add_(state[:, k, 6:12], alpha=elapsed)
                arm[:, k, substep, 0:6].add_(self._qdd[:, :6], alpha=0.5 * elapsed * elapsed)
                arm[:, k, substep, 6:12].copy_(state[:, k, 6:12])
                arm[:, k, substep, 6:12].add_(self._qdd[:, :6], alpha=elapsed)
                arm[:, k, substep, 12:55].copy_(state[:, k, 12:55])
                arm[:, k, substep, 12:55].add_(state[:, k, 55:98], alpha=elapsed)
                arm[:, k, substep, 12:55].add_(self._active_qdd, alpha=0.5 * elapsed * elapsed)
                arm[:, k, substep, 55:98].copy_(state[:, k, 55:98])
                arm[:, k, substep, 55:98].add_(self._active_qdd, alpha=elapsed)
            for substep, elapsed in enumerate((0.01, 0.02, 0.03, 0.04)):
                hand[:, k, substep, :12].copy_(state[:, k, 43:55])
                hand[:, k, substep, :12].add_(state[:, k, 86:98], alpha=elapsed)
                hand[:, k, substep, :12].add_(self._active_qdd[:, 31:], alpha=0.5 * elapsed * elapsed)
                hand[:, k, substep, 12:].copy_(state[:, k, 86:98])
                hand[:, k, substep, 12:].add_(self._active_qdd[:, 31:], alpha=elapsed)
        return state

    def rollout(self, control):
        _require(control, (self.batch, 25, 55), self.device)
        self.control.copy_(control)
        return self._rollout_core(self.control, self.state, self.arm_substep_state, self.hand_substep_state)

    def rollout_candidates(self, control):
        _require(control, (self.batch, 4, 25, 55), self.device)
        self._candidate_control.copy_(control)
        for candidate in range(4):
            self._rollout_core(
                self._candidate_control[:, candidate],
                self.candidate_state[:, candidate],
                self._candidate_arm[:, candidate],
                self._candidate_hand[:, candidate],
            )
        return self.candidate_state

    def linearize(self):
        self.state_control_jacobian.zero_()
        for k in range(25):
            if k:
                torch.bmm(
                    self.A[:, k],
                    self.state_control_jacobian[:, k - 1],
                    out=self.state_control_jacobian[:, k],
                )
            self.state_control_jacobian[:, k, :, k * 55 : (k + 1) * 55].add_(self.B[:, k])
        return self.A, self.B

    def solve_direction(self):
        self.H.zero_()
        self.gradient.zero_()
        torch.sub(self.state[:, 1:], self.state_target[:, 1:], out=self._state_residual)
        for k in range(25):
            jacobian = self.state_control_jacobian[:, k]
            weight = self.state_weight
            if k == 24:
                torch.add(self.state_weight, self.terminal_weight, out=self._combined_state_weight)
                weight = self._combined_state_weight
            torch.mul(jacobian, weight[None, :, None], out=self._weighted_jacobian)
            torch.bmm(jacobian.transpose(1, 2), self._weighted_jacobian, out=self._h_increment)
            self.H.add_(self._h_increment)
            torch.mul(self._state_residual[:, k], weight[None], out=self._weighted_state_residual)
            torch.bmm(jacobian.transpose(1, 2), self._weighted_state_residual[:, :, None], out=self._g_increment)
            self.gradient.add_(self._g_increment[:, :, 0])

            torch.bmm(self.palm_state_jacobian[:, k], self.state[:, k + 1, :, None], out=self._palm_residual[:, k, :, None])
            self._palm_residual[:, k].add_(self.palm_residual_offset[:, k])
            torch.bmm(self.palm_state_jacobian[:, k], jacobian, out=self._palm_control_jacobian)
            torch.mul(self._palm_control_jacobian, self.palm_weight[None, :, None], out=self._weighted_palm_jacobian)
            torch.bmm(self._palm_control_jacobian.transpose(1, 2), self._weighted_palm_jacobian, out=self._h_increment)
            self.H.add_(self._h_increment)
            torch.mul(self._palm_residual[:, k], self.palm_weight[None], out=self._weighted_palm_residual)
            torch.bmm(self._palm_control_jacobian.transpose(1, 2), self._weighted_palm_residual[:, :, None], out=self._g_increment)
            self.gradient.add_(self._g_increment[:, :, 0])

            start = k * 55
            stop = start + 55
            self.H[:, start:stop, start:stop].diagonal(dim1=-2, dim2=-1).add_(self.control_weight)
            self.gradient[:, start:stop].addcmul_(self.control_weight[None], self.nominal_control[:, k])
        self.H.mul_(2.0)
        self.gradient.mul_(2.0)
        self.H.diagonal(dim1=-2, dim2=-1).add_(2.0e-6)

        nominal_valid = self.validate(self.state, self.nominal_control)
        torch.logical_not(nominal_valid, out=self._invalid)
        torch.where(nominal_valid[:, None, None], self.H, self._identity_h, out=self._solve_matrix)
        torch.neg(self.gradient, out=self._solve_rhs[:, :, 0])
        self._solve_rhs.masked_fill_(self._invalid[:, None, None], 0.0)
        torch.linalg.lu_factor_ex(self._solve_matrix, check_errors=False, out=(self._factor, self._pivots, self._factor_info))
        torch.eq(self._factor_info, 0, out=self._mask)
        self._mask.logical_and_(nominal_valid)
        torch.where(self._mask[:, None, None], self._factor, self._identity_h, out=self._factor)
        torch.where(self._mask[:, None], self._pivots, self._identity_h_pivots, out=self._pivots)
        torch.linalg.lu_solve(self._factor, self._pivots, self._solve_rhs, out=self._solve_output)
        self._mask.logical_and_(self._finite.check(self._solve_output))
        torch.logical_not(self._mask, out=self._invalid)
        self._invalid.logical_or_(self.equality_bound)
        torch.where(self._invalid[:, None], self._zero_direction, self._solve_output[:, :, 0], out=self.direction)
        return self.direction

    def _validate_into(self, state, control, expected_state, arm, hand, output):
        output.copy_(self.valid)
        output.logical_and_(self._finite.check(state))
        output.logical_and_(self._finite.check(control))
        torch.eq(state, expected_state, out=self._state_elements)
        torch.all(self._state_elements, dim=(1, 2), out=self._mask)
        output.logical_and_(self._mask)
        torch.ge(state, self.state_lower, out=self._state_elements)
        torch.all(self._state_elements, dim=(1, 2), out=self._mask)
        output.logical_and_(self._mask)
        torch.le(state, self.state_upper, out=self._state_elements)
        torch.all(self._state_elements, dim=(1, 2), out=self._mask)
        output.logical_and_(self._mask)
        torch.ge(control, self.control_lower, out=self._control_elements)
        torch.all(self._control_elements, dim=(1, 2), out=self._mask)
        output.logical_and_(self._mask)
        torch.le(control, self.control_upper, out=self._control_elements)
        torch.all(self._control_elements, dim=(1, 2), out=self._mask)
        output.logical_and_(self._mask)

        torch.ge(arm, self.state_lower[:, 1:, None, :98], out=self._arm_elements)
        torch.all(self._arm_elements, dim=(1, 2, 3), out=self._mask)
        output.logical_and_(self._mask)
        torch.le(arm, self.state_upper[:, 1:, None, :98], out=self._arm_elements)
        torch.all(self._arm_elements, dim=(1, 2, 3), out=self._mask)
        output.logical_and_(self._mask)
        torch.ge(hand, self._hand_lower[:, :, None], out=self._hand_elements)
        torch.all(self._hand_elements, dim=(1, 2, 3), out=self._mask)
        output.logical_and_(self._mask)
        torch.le(hand, self._hand_upper[:, :, None], out=self._hand_elements)
        torch.all(self._hand_elements, dim=(1, 2, 3), out=self._mask)
        output.logical_and_(self._mask)

        for k in range(25):
            torch.bmm(
                self.palm_state_jacobian[:, k],
                state[:, k + 1, :, None],
                out=self._palm_residual[:, k, :, None],
            )
            self._palm_residual[:, k].add_(self.palm_residual_offset[:, k])
            torch.bmm(
                self.wrench_inequality_matrix[:, k],
                control[:, k, 43:, None],
                out=self._wrench_lhs[:, k, :, None],
            )
            torch.bmm(
                self.hard_inequality_matrix[:, k],
                control[:, k, :, None],
                out=self._hard_lhs[:, k, :, None],
            )
        output.logical_and_(self._finite.check(self._palm_residual))
        torch.le(self._wrench_lhs, self.wrench_inequality_upper, out=self._wrench_elements)
        torch.all(self._wrench_elements, dim=(1, 2), out=self._mask)
        output.logical_and_(self._mask)
        torch.le(self._hard_lhs, self.hard_inequality_upper, out=self._hard_elements)
        torch.all(self._hard_elements, dim=(1, 2), out=self._mask)
        output.logical_and_(self._mask)
        return output

    def validate(self, state, control):
        _require(state, (self.batch, 26, 110), self.device)
        _require(control, (self.batch, 25, 55), self.device)
        self._rollout_core(control, self._validation_state, self._validation_arm, self._validation_hand)
        return self._validate_into(
            state,
            control,
            self._validation_state,
            self._validation_arm,
            self._validation_hand,
            self.validation_valid,
        )

    def validate_candidates(self, state, control):
        _require(state, (self.batch, 4, 26, 110), self.device)
        _require(control, (self.batch, 4, 25, 55), self.device)
        for candidate in range(4):
            self._rollout_core(
                control[:, candidate],
                self._validation_state,
                self._validation_arm,
                self._validation_hand,
            )
            self._validate_into(
                state[:, candidate],
                control[:, candidate],
                self._validation_state,
                self._validation_arm,
                self._validation_hand,
                self.candidate_valid[:, candidate],
            )
        return self.candidate_valid
