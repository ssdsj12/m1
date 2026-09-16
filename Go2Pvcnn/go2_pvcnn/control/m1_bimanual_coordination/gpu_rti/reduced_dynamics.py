"""Batched 71x71 KKT condensation retaining all passive/mimic inertia."""

import torch

from .state_adapter import DYNAMICS_SHAPES, _FiniteRows, _require, _workspace_device


# Existing CPU FullDynamicsState symmetry invariant, not a physical safety gate.
MASS_SYMMETRY_ATOL = 1.e-10
MASS_SYMMETRY_RTOL = 1.e-10
# Match torch.linalg.matrix_rank's default for the CUDA float32 representation.
# This is intentionally conservative versus the float64 CPU rank decision.
ACTUATION_RANK_RTOL = 59 * torch.finfo(torch.float32).eps


class GpuReducedDynamics:
    """Persistent affine 59-acceleration/12-contact maps of 43 active efforts.

    Factorization uses preallocated LU/pivot/info/solution outputs. CUDA library
    scratch is warmed up before memory measurements; no tensor storage is
    constructed by ``step`` or ``reconstruct``. A rejected row preserves latest
    maps but clears validity, so it cannot claim a usable reconstructed output.
    """

    def __init__(self, *, batch, device="cuda:0"):
        self.batch = batch
        self.device = _workspace_device(batch, device)
        self._inputs = {name: torch.empty((batch, *shape), dtype=torch.float32, device=self.device) for name, shape in DYNAMICS_SHAPES.items()}
        self.actuation_matrix = torch.zeros(batch, 59, 43, dtype=torch.float32, device=self.device)
        self.kkt = torch.zeros(batch, 71, 71, dtype=torch.float32, device=self.device)
        self.rhs = torch.zeros(batch, 71, 44, dtype=torch.float32, device=self.device)
        self.lu = torch.empty_like(self.kkt)
        self.pivots = torch.empty(batch, 71, dtype=torch.int32, device=self.device)
        self.info = torch.empty(batch, dtype=torch.int32, device=self.device)
        self.solution = torch.empty_like(self.rhs)
        self._identity = torch.eye(71, dtype=torch.float32, device=self.device).expand(batch, -1, -1)
        self._identity_pivots = torch.arange(1, 72, dtype=torch.int32, device=self.device).expand(batch, -1)
        self._mass_identity = self._identity[:, :59, :59]
        self._mass_difference = torch.empty(batch, 59, 59, dtype=torch.float32, device=self.device)
        self._mass_tolerance = torch.empty_like(self._mass_difference)
        self._mass_elements = torch.empty(batch, 59, 59, dtype=torch.bool, device=self.device)
        self._mass_cholesky = torch.empty_like(self._mass_difference)
        self.mass_info = torch.empty(batch, dtype=torch.int32, device=self.device)
        self._rank_u = torch.empty(batch, 59, 43, dtype=torch.float32, device=self.device)
        self._rank_input = torch.empty_like(self._rank_u)
        self._rank_s = torch.empty(batch, 43, dtype=torch.float32, device=self.device)
        self._rank_vh = torch.empty(batch, 43, 43, dtype=torch.float32, device=self.device)
        self._rank_threshold = torch.empty(batch, dtype=torch.float32, device=self.device)
        self._rank_elements = torch.empty(batch, 43, dtype=torch.bool, device=self.device)
        self._zero_selection = torch.zeros(batch, 59, 43, dtype=torch.float32, device=self.device)
        self._maps = torch.zeros_like(self.rhs)
        self.qdd_offset = self._maps[:, :59, 0]
        self.qdd_from_effort = self._maps[:, :59, 1:]
        self.contact_offset = self._maps[:, 59:, 0]
        self.contact_from_effort = self._maps[:, 59:, 1:]
        self.qdd = torch.zeros(batch, 59, dtype=torch.float32, device=self.device)
        self.contact = torch.zeros(batch, 12, dtype=torch.float32, device=self.device)
        self.generalized_effort = torch.zeros(batch, 59, dtype=torch.float32, device=self.device)
        self._effort = torch.empty(batch, 43, dtype=torch.float32, device=self.device)
        self._qdd = torch.empty_like(self.qdd)
        self._contact = torch.empty_like(self.contact)
        self._generalized_effort = torch.empty_like(self.generalized_effort)
        self.valid = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.output_valid = torch.zeros_like(self.valid)
        self._mask = torch.empty_like(self.valid)
        self.rejected_count = torch.zeros(batch, dtype=torch.int64, device=self.device)
        self._finite = _FiniteRows(batch, 59 * 59, self.device)

    def step(self, *, input_valid=None, **dynamics):
        if set(dynamics) != set(DYNAMICS_SHAPES):
            raise ValueError("exactly the five full dynamics fields are required")
        for name, shape in DYNAMICS_SHAPES.items():
            _require(dynamics[name], (self.batch, *shape), self.device)
        if input_valid is not None:
            _require(input_valid, (self.batch,), self.device, torch.bool)
        # Mask is borrowed too: stage before valid may overwrite an alias.
        if input_valid is None:
            self.valid.fill_(True)
        else:
            self.valid.copy_(input_valid)
        for name, value in dynamics.items():
            self._inputs[name].copy_(value)
            self.valid.logical_and_(self._finite.check(self._inputs[name]))
        mass = self._inputs["mass_matrix"]
        torch.sub(mass, mass.transpose(1, 2), out=self._mass_difference)
        self._mass_difference.abs_()
        torch.abs(mass.transpose(1, 2), out=self._mass_tolerance)
        self._mass_tolerance.mul_(MASS_SYMMETRY_RTOL).add_(MASS_SYMMETRY_ATOL)
        torch.le(self._mass_difference, self._mass_tolerance, out=self._mass_elements)
        torch.all(self._mass_elements, dim=(1, 2), out=self._mask)
        self.valid.logical_and_(self._mask)
        torch.where(self.valid[:, None, None], mass, self._mass_identity, out=self._mass_difference)
        torch.linalg.cholesky_ex(self._mass_difference, check_errors=False, out=(self._mass_cholesky, self.mass_info))
        torch.eq(self.mass_info, 0, out=self._mask)
        self.valid.logical_and_(self._mask)
        # Direct CUDA inputs cannot inherit CPU dataclass provenance. Revalidate
        # numerical full column rank in float32 with persistent SVD outputs.
        # SVD input and all three outputs have separate fixed addresses.
        torch.where(self.valid[:, None, None], self._inputs["actuation_matrix"], self._zero_selection, out=self._rank_input)
        torch.linalg.svd(self._rank_input, full_matrices=False, out=(self._rank_u, self._rank_s, self._rank_vh))
        torch.amax(self._rank_s, dim=1, out=self._rank_threshold)
        self._rank_threshold.mul_(ACTUATION_RANK_RTOL)
        torch.gt(self._rank_s, self._rank_threshold[:, None], out=self._rank_elements)
        torch.all(self._rank_elements, dim=1, out=self._mask)
        self.valid.logical_and_(self._mask)
        self.kkt.zero_()
        self.kkt[:, :59, :59].copy_(self._inputs["mass_matrix"])
        torch.neg(self._inputs["wheel_contact_jacobian"].transpose(1, 2), out=self.kkt[:, :59, 59:])
        self.kkt[:, 59:, :59].copy_(self._inputs["wheel_contact_jacobian"])
        self.rhs.zero_()
        torch.neg(self._inputs["bias"], out=self.rhs[:, :59, 0])
        torch.neg(self._inputs["wheel_contact_bias"], out=self.rhs[:, 59:, 0])
        self.rhs[:, :59, 1:].copy_(self._inputs["actuation_matrix"])
        # Sanitize invalid numeric rows before invoking a batched factorization.
        torch.where(self.valid[:, None, None], self.kkt, self._identity, out=self.kkt)
        torch.logical_not(self.valid, out=self._mask)
        self.rhs.masked_fill_(self._mask[:, None, None], 0)
        torch.linalg.lu_factor_ex(self.kkt, check_errors=False, out=(self.lu, self.pivots, self.info))
        torch.eq(self.info, 0, out=self._mask)
        self.valid.logical_and_(self._mask)
        # Singular LU rows become identity before solve, protecting other rows.
        torch.where(self.valid[:, None, None], self.lu, self._identity, out=self.lu)
        # Identity pivots are 1-based, allocated once in the constructor.
        torch.where(self.valid[:, None], self.pivots, self._identity_pivots, out=self.pivots)
        torch.linalg.lu_solve(self.lu, self.pivots, self.rhs, out=self.solution)
        self.valid.logical_and_(self._finite.check(self.solution))
        torch.where(self.valid[:, None, None], self.solution, self._maps, out=self._maps)
        torch.where(self.valid[:, None, None], self._inputs["actuation_matrix"], self.actuation_matrix, out=self.actuation_matrix)
        self.output_valid.zero_()
        torch.logical_not(self.valid, out=self._mask)
        self.rejected_count.add_(self._mask)
        return self.valid

    def reconstruct(self, effort):
        _require(effort, (self.batch, 43), self.device)
        self._effort.copy_(effort)
        self.output_valid.copy_(self.valid)
        self.output_valid.logical_and_(self._finite.check(self._effort))
        torch.bmm(self.qdd_from_effort, self._effort[:, :, None], out=self._qdd[:, :, None])
        self._qdd.add_(self.qdd_offset)
        torch.bmm(self.contact_from_effort, self._effort[:, :, None], out=self._contact[:, :, None])
        self._contact.add_(self.contact_offset)
        torch.bmm(self.actuation_matrix, self._effort[:, :, None], out=self._generalized_effort[:, :, None])
        for value in (self._qdd, self._contact, self._generalized_effort):
            self.output_valid.logical_and_(self._finite.check(value))
        for value, destination in ((self._qdd, self.qdd), (self._contact, self.contact), (self._generalized_effort, self.generalized_effort)):
            torch.where(self.output_valid[:, None], value, destination, out=destination)
        return self.output_valid
