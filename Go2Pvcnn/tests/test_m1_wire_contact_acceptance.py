from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.wire_contact_acceptance import (
    WireAcceptanceConfig,
    WireAcceptancePhase,
    WireAcceptanceTracker,
    WireContactObservation,
    evaluate_wire_contact,
)


def _observation() -> WireContactObservation:
    endpoints = torch.tensor(
        [[0.40, 0.10, 1.20], [0.60, -0.10, 1.20]], dtype=torch.float64
    )
    return WireContactObservation(
        left_contact_mask=torch.tensor([True, True, False, False, False]),
        right_contact_mask=torch.tensor([True, True, False, False, False]),
        left_fingertip_forces_b=torch.tensor(
            [[0.21, 0.0, 0.0], [0.22, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        right_fingertip_forces_b=torch.tensor(
            [[0.21, 0.0, 0.0], [0.22, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        left_fingertip_positions_b=torch.tensor(
            [[0.40, 0.10, 1.20]] * 5, dtype=torch.float64
        ),
        right_fingertip_positions_b=torch.tensor(
            [[0.60, -0.10, 1.20]] * 5, dtype=torch.float64
        ),
        left_palm_position_b=endpoints[0],
        right_palm_position_b=endpoints[1],
        wire_endpoint_positions_b=endpoints,
        dt_s=0.005,
    )


def test_contacts_below_minimum_force_do_not_count() -> None:
    observation = _observation()
    evaluation = evaluate_wire_contact(
        replace(
            observation,
            left_fingertip_forces_b=torch.tensor(
                [[0.199, 0.0, 0.0]] * 5, dtype=torch.float64
            ),
            right_fingertip_forces_b=torch.tensor(
                [[0.199, 0.0, 0.0]] * 5, dtype=torch.float64
            ),
        ),
        WireAcceptanceConfig(),
    )

    assert evaluation.left_valid_contacts == 0
    assert evaluation.right_valid_contacts == 0
    assert not evaluation.bilateral_contact
    assert evaluation.failure_reason == "insufficient_valid_contacts"


def test_contacts_equal_to_minimum_force_do_not_count() -> None:
    observation = _observation()
    evaluation = evaluate_wire_contact(
        replace(
            observation,
            left_fingertip_forces_b=torch.tensor(
                [[0.2, 0.0, 0.0]] * 5, dtype=torch.float64
            ),
            right_fingertip_forces_b=torch.tensor(
                [[0.2, 0.0, 0.0]] * 5, dtype=torch.float64
            ),
        ),
        WireAcceptanceConfig(),
    )

    assert evaluation.left_valid_contacts == 0
    assert evaluation.right_valid_contacts == 0
    assert not evaluation.bilateral_contact
    assert evaluation.failure_reason == "insufficient_valid_contacts"


def test_contacts_above_minimum_force_count() -> None:
    observation = _observation()
    evaluation = evaluate_wire_contact(
        replace(
            observation,
            left_fingertip_forces_b=torch.tensor(
                [[0.2001, 0.0, 0.0]] * 5, dtype=torch.float64
            ),
            right_fingertip_forces_b=torch.tensor(
                [[0.2001, 0.0, 0.0]] * 5, dtype=torch.float64
            ),
        ),
        WireAcceptanceConfig(),
    )

    assert evaluation.left_valid_contacts == 2
    assert evaluation.right_valid_contacts == 2
    assert evaluation.bilateral_contact


def test_two_qualified_contacts_per_hand_are_accepted() -> None:
    evaluation = evaluate_wire_contact(_observation(), WireAcceptanceConfig())

    assert evaluation.left_valid_contacts == 2
    assert evaluation.right_valid_contacts == 2
    assert evaluation.left_slip_m == pytest.approx(0.0)
    assert evaluation.right_slip_m == pytest.approx(0.0)
    assert evaluation.bilateral_contact
    assert evaluation.finite
    assert evaluation.failure_reason is None


def test_endpoint_slip_is_relative_to_palm_not_fingertip_contact_centroid() -> None:
    observation = _observation()
    shifted_positions = observation.left_fingertip_positions_b.clone()
    shifted_positions[:2] += torch.tensor([0.50, 0.0, 0.0], dtype=torch.float64)
    evaluation = evaluate_wire_contact(
        replace(
            observation,
            left_fingertip_positions_b=shifted_positions,
        ),
        WireAcceptanceConfig(),
    )

    assert evaluation.left_slip_m == pytest.approx(0.0)
    assert evaluation.bilateral_contact


def test_endpoint_slip_above_limit_is_rejected_when_palm_moves() -> None:
    observation = _observation()
    evaluation = evaluate_wire_contact(
        replace(
            observation,
            left_palm_position_b=observation.left_palm_position_b
            + torch.tensor([0.0201, 0.0, 0.0], dtype=torch.float64),
        ),
        WireAcceptanceConfig(),
    )

    assert evaluation.left_slip_m == pytest.approx(0.0201)
    assert not evaluation.bilateral_contact
    assert evaluation.failure_reason == "endpoint_slip_exceeds_0.02_m"


def test_non_finite_input_is_rejected_without_raising() -> None:
    observation = _observation()
    evaluation = evaluate_wire_contact(
        replace(
            observation,
            left_fingertip_forces_b=torch.tensor(
                [[float("nan"), 0.0, 0.0]] * 5, dtype=torch.float64
            ),
        ),
        WireAcceptanceConfig(),
    )

    assert not evaluation.finite
    assert not evaluation.bilateral_contact
    assert evaluation.failure_reason == "non_finite_contact_observation"


def test_observation_requires_cpu_float64_positions_and_forces() -> None:
    observation = _observation()

    with pytest.raises(TypeError, match="left_fingertip_forces_b must have dtype torch.float64"):
        replace(
            observation,
            left_fingertip_forces_b=observation.left_fingertip_forces_b.float(),
        )
    with pytest.raises(ValueError, match="right_palm_position_b must have exact shape"):
        replace(
            observation,
            right_palm_position_b=torch.zeros(2, dtype=torch.float64),
        )


def test_observation_is_frozen_and_detached_from_input_tensors() -> None:
    endpoint = torch.tensor([0.40, 0.10, 1.20], dtype=torch.float64)
    observation = _observation()
    detached = replace(observation, left_palm_position_b=endpoint)
    endpoint.add_(1.0)

    assert detached.left_palm_position_b.tolist() == [0.4, 0.1, 1.2]
    with pytest.raises(FrozenInstanceError):
        detached.dt_s = 0.01


def _accepted_evaluation():
    return evaluate_wire_contact(_observation(), WireAcceptanceConfig())


def _slipping_evaluation():
    observation = _observation()
    return evaluate_wire_contact(
        replace(
            observation,
            left_palm_position_b=observation.left_palm_position_b
            + torch.tensor([0.0201, 0.0, 0.0], dtype=torch.float64),
        ),
        WireAcceptanceConfig(),
    )


def _advance_contact(
    tracker: WireAcceptanceTracker,
    duration_s: float,
    dt_s: float = 0.1,
    wire_center_height_m: float = 1.20,
):
    steps = round(duration_s / dt_s)
    state = None
    for _ in range(steps):
        state = tracker.update(
            _accepted_evaluation(),
            wire_center_height_m=wire_center_height_m,
            dt_s=dt_s,
            safety_feasible=True,
        )
    assert state is not None
    return state


def test_contact_shorter_than_half_second_cannot_permit_lift() -> None:
    tracker = WireAcceptanceTracker(WireAcceptanceConfig())
    tracker.reset(wire_center_height_m=1.20)

    state = _advance_contact(tracker, duration_s=0.4)

    assert state.phase is WireAcceptancePhase.CONTACT
    assert state.bilateral_contact_elapsed_s == pytest.approx(0.4)
    assert not state.lift_permitted
    assert not state.success


def test_stable_bilateral_contact_permits_lift_at_half_second() -> None:
    tracker = WireAcceptanceTracker(WireAcceptanceConfig())
    tracker.reset(wire_center_height_m=1.20)

    state = _advance_contact(tracker, duration_s=0.5)

    assert state.phase is WireAcceptancePhase.LIFT
    assert state.lift_permitted
    assert not state.request_reclamp
    assert state.terminal_reason is None


def test_contact_loss_during_lift_requests_reclamp() -> None:
    tracker = WireAcceptanceTracker(WireAcceptanceConfig())
    tracker.reset(wire_center_height_m=1.20)
    _advance_contact(tracker, duration_s=0.5)
    lost_contact = evaluate_wire_contact(
        replace(_observation(), left_contact_mask=torch.zeros(5, dtype=torch.bool)),
        WireAcceptanceConfig(),
    )

    state = tracker.update(
        lost_contact,
        wire_center_height_m=1.20,
        dt_s=0.1,
        safety_feasible=True,
    )

    assert state.phase is WireAcceptancePhase.CONTACT
    assert not state.lift_permitted
    assert state.request_reclamp
    assert state.bilateral_contact_elapsed_s == 0.0


def test_required_lift_followed_by_one_second_stable_hold_succeeds() -> None:
    tracker = WireAcceptanceTracker(WireAcceptanceConfig())
    tracker.reset(wire_center_height_m=1.20)
    _advance_contact(tracker, duration_s=0.5)

    reached_lift = tracker.update(
        _accepted_evaluation(),
        wire_center_height_m=1.25,
        dt_s=0.1,
        safety_feasible=True,
    )
    assert reached_lift.phase is WireAcceptancePhase.HOLD
    assert reached_lift.hold_elapsed_s == pytest.approx(0.1)

    state = _advance_contact(
        tracker, duration_s=0.9, wire_center_height_m=1.25
    )

    assert state.phase is WireAcceptancePhase.SUCCEEDED
    assert state.success
    assert state.terminal_reason is None


@pytest.mark.parametrize(
    ("evaluation", "safety_feasible", "reason"),
    [
        (_slipping_evaluation, True, "excessive_endpoint_slip"),
        (_accepted_evaluation, False, "safety_rejected"),
    ],
)
def test_slip_and_safety_rejection_terminate_with_distinct_reasons(
    evaluation, safety_feasible: bool, reason: str
) -> None:
    tracker = WireAcceptanceTracker(WireAcceptanceConfig())
    tracker.reset(wire_center_height_m=1.20)

    state = tracker.update(
        evaluation(),
        wire_center_height_m=1.20,
        dt_s=0.1,
        safety_feasible=safety_feasible,
    )

    assert state.phase is WireAcceptancePhase.TERMINATED
    assert state.terminal_reason == reason
    assert not state.success


def test_reset_clears_timers_lift_permission_and_success() -> None:
    tracker = WireAcceptanceTracker(WireAcceptanceConfig())
    tracker.reset(wire_center_height_m=1.20)
    _advance_contact(tracker, duration_s=0.5)

    tracker.reset(wire_center_height_m=0.80)
    state = tracker.state

    assert state.phase is WireAcceptancePhase.CONTACT
    assert state.bilateral_contact_elapsed_s == 0.0
    assert state.lift_elapsed_s == 0.0
    assert state.hold_elapsed_s == 0.0
    assert not state.lift_permitted
    assert not state.request_reclamp
    assert not state.success
    assert state.terminal_reason is None
