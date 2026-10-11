import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import (
    DloCenterlineState,
    build_wire_line_contact_goal,
    build_wire_station_contact_goal,
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


def test_station_goal_centers_each_opposed_pinch_on_its_endpoint():
    from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import (
        build_wire_station_contact_goal,
    )
    state = DloCenterlineState(np.array([[0.4, 0., 1.2], [0.5, 0., 1.2], [0.6, 0., 1.2]]), 0.004)
    goal = build_wire_station_contact_goal(state)
    for targets, station in ((goal.left_contact_targets, state.positions[0]),
                             (goal.right_contact_targets, state.positions[-1])):
        np.testing.assert_allclose(np.mean([t.position for t in targets], axis=0), station)
        np.testing.assert_allclose(targets[0].normal, -np.array(targets[1].normal))
        for target in targets:
            outward = np.array(target.position) - station
            np.testing.assert_allclose(target.normal, outward / np.linalg.norm(outward))
        assert np.isclose(np.linalg.norm(np.array(targets[0].position) - station),
                          np.hypot(0.007, 0.004))
    np.testing.assert_allclose(goal.center, state.midpoint)
    np.testing.assert_allclose(goal.left_palm_approach_normal, [0., -1., 0.])
    np.testing.assert_allclose(goal.right_palm_approach_normal, [0., 1., 0.])


def test_o6_station_targets_straddle_the_5mm_wire_with_positive_normal_alignment():
    """O6 thumb/index targets must fit across the actual wire cross-section."""
    state = DloCenterlineState(
        np.array([[0.4, 0.0, 1.2], [0.5, 0.0, 1.2], [0.6, 0.0, 1.2]]),
        radius_m=0.005,
    )
    goal = build_wire_station_contact_goal(state)

    for targets, station, palm_normal in (
        (goal.left_contact_targets, state.positions[0], goal.left_palm_approach_normal),
        (goal.right_contact_targets, state.positions[-1], goal.right_palm_approach_normal),
    ):
        first = np.asarray(targets[0].position) - station
        second = np.asarray(targets[1].position) - station
        pinch_axis = first - second
        pinch_axis /= np.linalg.norm(pinch_axis)
        # The thumb/index pair brackets the station in the cross-section,
        # with both tips just outside the 5 mm-radius wire plus clearance.
        assert np.dot(first, pinch_axis) > 0.0
        assert np.dot(second, pinch_axis) < 0.0
        for target in targets:
            radial = np.asarray(target.position) - station
            radial_distance = np.linalg.norm(radial)
            assert 0.009 < radial_distance < 0.011
            radial /= radial_distance
            # Force alignment is positive and sufficiently normal to the wire;
            # this is a geometry invariant, not a relaxed acceptance threshold.
            assert float(np.dot(np.asarray(target.normal), radial)) > 0.99
            assert abs(float(np.dot(np.asarray(target.normal), np.asarray(palm_normal)))) > 0.8


def test_station_fingertip_offset_is_configurable_but_remains_bounded():
    state = DloCenterlineState(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), radius_m=0.005
    )
    with pytest.raises(ValueError, match="fingertip_offset_ratio"):
        build_wire_station_contact_goal(state, fingertip_offset_ratio=0.0)
    explicit_baseline = build_wire_station_contact_goal(state, fingertip_offset_ratio=1.5)
    assert np.isclose(
        np.linalg.norm(np.asarray(explicit_baseline.left_contact_targets[0].position)),
        np.hypot(0.008, 0.0075),
    )
    with pytest.raises(ValueError, match="fingertip_offset_ratio"):
        build_wire_station_contact_goal(state, fingertip_offset_ratio=1.51)


def test_default_station_targets_preserve_endpoint_reachability_with_intermediate_radial_offset():
    """The tuned default keeps each station reachable while restoring O6 radial placement."""
    state = DloCenterlineState(
        np.array([[0.4, 0.0, 1.2], [0.5, 0.0, 1.2], [0.6, 0.0, 1.2]]),
        radius_m=0.005,
    )
    goal = build_wire_station_contact_goal(state)
    expected_radial = np.hypot(0.008, 0.005)
    for targets, station in (
        (goal.left_contact_targets, state.positions[0]),
        (goal.right_contact_targets, state.positions[-1]),
    ):
        np.testing.assert_allclose(
            np.mean([target.position for target in targets], axis=0), station
        )
        assert np.isclose(
            np.linalg.norm(np.asarray(targets[0].position) - station),
            expected_radial,
        )


@pytest.mark.parametrize("clearance", [0., -0.1, np.nan, np.inf])
def test_station_goal_rejects_invalid_clearance(clearance):
    from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import build_wire_station_contact_goal
    with pytest.raises(ValueError, match="clearance"):
        build_wire_station_contact_goal(DloCenterlineState(np.array([[0., 0., 0.], [1., 0., 0.]]), 0.004), clearance_m=clearance)
