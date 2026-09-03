from __future__ import annotations

import pytest
import torch
from pathlib import Path

from go2_pvcnn.control.m1_bimanual_coordination.contracts import FullDynamicsState
from go2_pvcnn.control.m1_bimanual_coordination.reduced_dynamics import (
    ACTIVE_DOF,
    GENERALIZED_DOF,
    WHEEL_CONSTRAINT_DOF,
    build_actuation_matrix,
    condense_constrained_dynamics,
    stack_stationary_wheel_jacobians,
)


DTYPE = torch.float64


def _state(*, mimic_coupling: float = 0.0) -> FullDynamicsState:
    mass = 2.0 * torch.eye(GENERALIZED_DOF, dtype=DTYPE)
    mass[20, 55] = mimic_coupling
    mass[55, 20] = mimic_coupling
    bias = torch.linspace(-0.2, 0.2, GENERALIZED_DOF, dtype=DTYPE)
    actuation = torch.zeros((GENERALIZED_DOF, ACTIVE_DOF), dtype=DTYPE)
    actuation[6:49] = torch.eye(ACTIVE_DOF, dtype=DTYPE)
    contact = torch.zeros(
        (WHEEL_CONSTRAINT_DOF, GENERALIZED_DOF), dtype=DTYPE
    )
    contact[:, :WHEEL_CONSTRAINT_DOF] = torch.eye(
        WHEEL_CONSTRAINT_DOF, dtype=DTYPE
    )
    return FullDynamicsState(
        mass_matrix=mass,
        bias=bias,
        actuation_matrix=actuation,
        wheel_contact_jacobian=contact,
        wheel_contact_bias=torch.linspace(
            -0.01, 0.01, WHEEL_CONSTRAINT_DOF, dtype=DTYPE
        ),
    )


def test_condensed_map_satisfies_dynamics_and_stationary_contact() -> None:
    state = _state(mimic_coupling=0.3)
    reduced = condense_constrained_dynamics(state)
    effort = torch.linspace(-1.0, 1.0, ACTIVE_DOF, dtype=DTYPE)
    qdd = reduced.qdd_offset + reduced.qdd_from_effort @ effort
    contact_force = reduced.contact_offset + reduced.contact_from_effort @ effort

    dynamics_residual = (
        state.mass_matrix @ qdd
        + state.bias
        - state.actuation_matrix @ effort
        - state.wheel_contact_jacobian.T @ contact_force
    )
    contact_residual = (
        state.wheel_contact_jacobian @ qdd + state.wheel_contact_bias
    )
    assert torch.linalg.vector_norm(dynamics_residual) < 1.0e-9
    assert torch.linalg.vector_norm(contact_residual) < 1.0e-9
    assert reduced.qdd_from_effort.shape == (59, 43)
    assert reduced.contact_from_effort.shape == (12, 43)


def test_passive_mimic_inertia_changes_active_acceleration() -> None:
    coupled = condense_constrained_dynamics(_state(mimic_coupling=0.3))
    uncoupled = condense_constrained_dynamics(_state(mimic_coupling=0.0))

    assert not torch.allclose(
        coupled.qdd_from_effort[20], uncoupled.qdd_from_effort[20]
    )


def test_contract_rejects_59_dimensional_action_matrix() -> None:
    with pytest.raises(ValueError, match=r"\(59, 43\)"):
        FullDynamicsState(
            mass_matrix=torch.eye(59, dtype=DTYPE),
            bias=torch.zeros(59, dtype=DTYPE),
            actuation_matrix=torch.eye(59, dtype=DTYPE),
            wheel_contact_jacobian=torch.zeros(12, 59, dtype=DTYPE),
            wheel_contact_bias=torch.zeros(12, dtype=DTYPE),
        )


def test_contract_rejects_non_symmetric_mass_matrix() -> None:
    state = _state()
    mass = state.mass_matrix.clone()
    mass[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        FullDynamicsState(
            mass_matrix=mass,
            bias=state.bias,
            actuation_matrix=state.actuation_matrix,
            wheel_contact_jacobian=state.wheel_contact_jacobian,
            wheel_contact_bias=state.wheel_contact_bias,
        )


def test_actuation_matrix_maps_only_43_physical_joint_columns() -> None:
    active_joint_ids = tuple(range(43))
    selection = build_actuation_matrix(active_joint_ids)

    assert selection.shape == (59, 43)
    assert torch.all(selection[:6] == 0.0)
    assert torch.equal(selection[6:49], torch.eye(43, dtype=DTYPE))


def test_wheel_spatial_jacobians_stack_only_linear_rows() -> None:
    jacobians = torch.arange(4 * 6 * 59, dtype=DTYPE).reshape(4, 6, 59)
    stacked = stack_stationary_wheel_jacobians(jacobians)

    assert stacked.shape == (12, 59)
    assert torch.equal(stacked, jacobians[:, :3].reshape(12, 59))


def test_isaac_adapter_exposes_full_dynamics_alongside_snapshot() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
    ).read_text(encoding="utf-8")

    assert "def dynamics(self) -> FullDynamicsState:" in wrapper
    assert "get_generalized_mass_matrices()" in wrapper
    assert "get_gravity_compensation_forces()" in wrapper
    assert "get_coriolis_and_centrifugal_compensation_forces()" in wrapper
    assert "build_actuation_matrix(self.active_joint_ids)" in wrapper
    assert "stack_stationary_wheel_jacobians(" in wrapper
    assert "self.last_dynamics = self.adapter.dynamics()" in wrapper
