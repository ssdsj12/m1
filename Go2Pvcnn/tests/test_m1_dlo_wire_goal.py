import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import (
    DloCenterlineState,
    build_wire_line_contact_goal,
)


def test_centerline_state_computes_unit_tangents_and_midpoint():
    state = DloCenterlineState(
        positions=np.array([[0.4, 0.0, 1.2], [0.5, 0.0, 1.2], [0.6, 0.0, 1.2]]),
        radius_m=0.004,
    )

    np.testing.assert_allclose(state.tangents[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(state.midpoint, [0.5, 0.0, 1.2])


def test_wire_goal_places_four_fingertips_around_centerline_midpoint():
    state = DloCenterlineState(
        positions=np.array([[0.4, 0.0, 1.2], [0.5, 0.0, 1.2], [0.6, 0.0, 1.2]]),
        radius_m=0.004,
    )
    goal = build_wire_line_contact_goal(state)

    assert len(goal.left_contact_targets) == 2
    assert len(goal.right_contact_targets) == 2
    assert goal.object_class == "wire"
    assert goal.contact_half_gap_m > state.radius_m
    for target in (*goal.left_contact_targets, *goal.right_contact_targets):
        assert np.isfinite(target.position).all()


def test_centerline_rejects_degenerate_or_non_monotonic_input():
    with pytest.raises(ValueError, match="at least two"):
        DloCenterlineState(np.array([[0.0, 0.0, 0.0]]), radius_m=0.004)
    with pytest.raises(ValueError, match="radius"):
        DloCenterlineState(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), radius_m=0.0)
