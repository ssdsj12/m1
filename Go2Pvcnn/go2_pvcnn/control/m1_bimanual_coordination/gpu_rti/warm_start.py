"""Row-local accepted horizons and terminal-hold CUDA warm shifting."""

import torch

from .state_adapter import _FiniteRows, _require, _workspace_device


class GpuWarmStart:
    """Actions [B,25,43], states [B,26,D], identity [B,3] int64.

    Identity columns are caller-owned environment, layout and phase generation
    tokens. A mismatch/reset destroys accepted eligibility until a new ``accept``.
    Candidate buffers never replace the previous accepted horizon implicitly.
    All returned validity refers to eligibility, not physical safety acceptance.
    """

    def __init__(self, *, batch, state_dim=111, device="cuda:0"):
        self.batch = batch
        self.device = _workspace_device(batch, device)
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive static integer")
        self.state_dim = state_dim
        for name, shape in (("accepted_action", (batch, 25, 43)), ("accepted_state", (batch, 26, state_dim)),
                ("shifted_action", (batch, 25, 43)), ("shifted_state", (batch, 26, state_dim)),
                ("_action", (batch, 25, 43)), ("_state", (batch, 26, state_dim)), ("_measured", (batch, state_dim))):
            setattr(self, name, torch.zeros(shape, dtype=torch.float32, device=self.device))
        self.accepted_identity = torch.zeros(batch, 3, dtype=torch.int64, device=self.device)
        self._identity = torch.empty_like(self.accepted_identity)
        self._identity_match = torch.empty(batch, 3, dtype=torch.bool, device=self.device)
        self.accepted_valid = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros_like(self.accepted_valid)
        self._mask = torch.empty_like(self.valid)
        self._reset = torch.zeros_like(self.valid)
        self.invalidated_count = torch.zeros(batch, dtype=torch.int64, device=self.device)
        self.rejected_count = torch.zeros_like(self.invalidated_count)
        self._finite = _FiniteRows(batch, max(25 * 43, 26 * state_dim), self.device)

    def accept(self, action, state, accepted, identity):
        _require(action, (self.batch, 25, 43), self.device)
        _require(state, (self.batch, 26, self.state_dim), self.device)
        _require(accepted, (self.batch,), self.device, torch.bool)
        _require(identity, (self.batch, 3), self.device, torch.int64)
        self._action.copy_(action)
        self._state.copy_(state)
        self._identity.copy_(identity)
        self._mask.copy_(accepted)
        self._mask.logical_and_(self._finite.check(self._action))
        self._mask.logical_and_(self._finite.check(self._state))
        torch.where(self._mask[:, None, None], self._action, self.accepted_action, out=self.accepted_action)
        torch.where(self._mask[:, None, None], self._state, self.accepted_state, out=self.accepted_state)
        torch.where(self._mask[:, None], self._identity, self.accepted_identity, out=self.accepted_identity)
        # A rejected publication cannot re-certify a previous safe horizon.
        self.accepted_valid.copy_(self._mask)
        self.valid.zero_()
        torch.logical_not(self._mask, out=self._reset)
        self.rejected_count.add_(self._reset)
        return self.accepted_valid

    def step(self, measured_x0, identity, *, reset=None):
        _require(measured_x0, (self.batch, self.state_dim), self.device)
        _require(identity, (self.batch, 3), self.device, torch.int64)
        if reset is not None:
            _require(reset, (self.batch,), self.device, torch.bool)
        self._measured.copy_(measured_x0)
        self._identity.copy_(identity)
        if reset is None:
            self._reset.zero_()
        else:
            self._reset.copy_(reset)
        torch.eq(self._identity, self.accepted_identity, out=self._identity_match)
        torch.all(self._identity_match, dim=1, out=self._mask)
        torch.logical_not(self._reset, out=self.valid)
        self._mask.logical_and_(self.valid)
        torch.logical_not(self._mask, out=self.valid)
        self.valid.logical_and_(self.accepted_valid)
        self.invalidated_count.add_(self.valid)
        self.accepted_valid.logical_and_(self._mask)
        self.valid.copy_(self.accepted_valid)
        self.valid.logical_and_(self._finite.check(self._measured))
        # Source horizons remain separate; stage measurement before any writes
        # because it may alias the old candidate or accepted node zero.
        self._action[:, :-1].copy_(self.accepted_action[:, 1:])
        self._action[:, -1].copy_(self.accepted_action[:, -1])
        self._state[:, :-1].copy_(self.accepted_state[:, 1:])
        self._state[:, -1].copy_(self.accepted_state[:, -1])
        self._state[:, 0].copy_(self._measured)
        torch.where(self.valid[:, None, None], self._action, self.shifted_action, out=self.shifted_action)
        torch.where(self.valid[:, None, None], self._state, self.shifted_state, out=self.shifted_state)
        return self.valid
