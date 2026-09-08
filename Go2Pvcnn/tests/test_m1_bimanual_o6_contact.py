from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from go2_pvcnn.control.m1_bimanual_coordination import BimanualPhase
from go2_pvcnn.control.m1_bimanual_coordination.hand_mpc import O6HandMpc
from go2_pvcnn.control.m1_bimanual_coordination.o6_contact_kinematics import (
    PrecontactHandController,
    fold_o6_fingertip_jacobians,
)
from tests.test_m1_bimanual_hand_mpc import _input
from go2_pvcnn.control.m1_bimanual_coordination.contracts import SideHandState


DTYPE = torch.float64


def test_mimic_columns_fold_into_six_active_columns() -> None:
    full = torch.zeros((5, 6, 59), dtype=DTYPE)
    full[:, :3, 40] = 1.0
    full[:, :3, 50] = 2.0

    folded = fold_o6_fingertip_jacobians(
        full,
        active_generalized_ids=(40, 41, 42, 43, 44, 45),
        mimic_specs=(
            (50, 0, 1.86),
            (51, 2, 0.89),
            (52, 3, 0.89),
            (53, 4, 0.89),
            (54, 5, 0.89),
        ),
    )

    assert folded.shape == (15, 6)
    expected_thumb = full[:, :3, 40] + 1.86 * full[:, :3, 50]
    assert torch.allclose(folded[:, 0].reshape(5, 3), expected_thumb)
    assert torch.linalg.vector_norm(folded) > 0.0


def test_preload_closes_uncontacted_fingers_and_freezes_contacted_digits() -> None:
    controller = PrecontactHandController()
    assert torch.allclose(
        controller.preload_q[2:],
        1.20 * torch.ones(4, dtype=DTYPE),
    )
    q = torch.zeros(6, dtype=DTYPE)
    contact_mask = torch.tensor([True, True, False, False, False])

    q_ref, qd_ref = controller.reference(q, contact_mask, BimanualPhase.PRELOAD)

    assert torch.equal(q_ref[:3], q[:3])
    assert torch.equal(qd_ref[:3], torch.zeros(3, dtype=DTYPE))
    assert torch.all(q_ref[3:] > q[3:])
    assert torch.all(qd_ref[3:] > 0.0)


def test_preload_latches_first_contact_joint_position_after_contact_loss() -> None:
    controller = PrecontactHandController()
    contact_q = 0.85 * torch.ones(6, dtype=DTYPE)
    pinky_contact = torch.tensor([False, False, False, False, True])
    controller.reference(contact_q, pinky_contact, BimanualPhase.PRELOAD)

    drifted_q = contact_q.clone()
    drifted_q[5] = 1.10
    q_ref, qd_ref = controller.reference(
        drifted_q,
        torch.zeros(5, dtype=torch.bool),
        BimanualPhase.PRELOAD,
    )

    assert q_ref[5].item() == contact_q[5].item()
    assert qd_ref[5].item() == 0.0


def test_approach_holds_the_open_pregrasp_reference() -> None:
    controller = PrecontactHandController()
    assert torch.allclose(
        controller.open_q,
        0.25 * torch.ones(6, dtype=DTYPE),
    )
    q_ref, qd_ref = controller.reference(
        torch.ones(6, dtype=DTYPE),
        torch.zeros(5, dtype=torch.bool),
        BimanualPhase.APPROACH,
    )

    assert torch.allclose(q_ref, controller.open_q)
    assert torch.equal(qd_ref, torch.zeros(6, dtype=DTYPE))


def test_hand_mpc_uses_precontact_reference_before_contact() -> None:
    sample = replace(
        _input(),
        q=torch.zeros(6, dtype=DTYPE),
        contact_mask=torch.zeros(5, dtype=torch.bool),
        phase=BimanualPhase.PRELOAD,
    )

    solution = O6HandMpc().plan(sample)

    assert solution.diagnostics.feasible
    assert torch.all(solution.qd_ref > 0.0)
    assert torch.all(solution.predicted_forces_b == 0.0)


def test_side_hand_state_requires_real_fingertip_jacobian_shape() -> None:
    state = SideHandState(
        q=torch.zeros(6, dtype=DTYPE),
        qd=torch.zeros(6, dtype=DTYPE),
        fingertip_forces_b=torch.zeros(5, 3, dtype=DTYPE),
        fingertip_positions_b=torch.zeros(5, 3, dtype=DTYPE),
        fingertip_jacobian_b=torch.ones(15, 6, dtype=DTYPE),
        contact_mask=torch.zeros(5, dtype=torch.bool),
    )

    assert state.fingertip_jacobian_b.shape == (15, 6)


def test_isaac_adapter_folds_left_and_right_mimic_jacobians() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
    ).read_text(encoding="utf-8")

    assert "fold_o6_fingertip_jacobians(" in wrapper
    assert "fingertip_jacobian_b=fingertip_jacobian" in wrapper
    assert "contact_jacobian=state.fingertip_jacobian_b" in (
        Path(__file__).resolve().parents[1]
        / "go2_pvcnn/control/m1_bimanual_coordination/runtime.py"
    ).read_text(encoding="utf-8")
