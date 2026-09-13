from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.sources import (
    SOURCE_HANDS,
    SourceHandSpec,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.urdf_fk import (
    UrdfKinematicTree,
)


def _write_urdf(tmp_path, body: str):
    path = tmp_path / "hand.urdf"
    path.write_text(dedent(f"""\
        <?xml version="1.0"?>
        <robot name="test_hand">
        {body}
        </robot>
        """), encoding="utf-8")
    return path


def _two_joint_urdf(*, mimic: bool = False, multiplier: float = 1.0, offset: float = 0.0):
    mimic_xml = (
        f'<mimic joint="j0" multiplier="{multiplier}" offset="{offset}"/>' if mimic else ""
    )
    return f"""
        <link name="base"/>
        <link name="middle"/>
        <link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/><child link="middle"/>
          <axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2" effort="2" velocity="4"/>
        </joint>
        <joint name="j1" type="revolute">
          <parent link="middle"/><child link="tip"/>
          <origin xyz="2 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
          {mimic_xml}
        </joint>
    """


def test_two_revolute_joint_fk_matches_hand_solution(tmp_path):
    tree = UrdfKinematicTree.from_file(_write_urdf(tmp_path, _two_joint_urdf()))

    result = tree.forward_links(
        np.array([[np.pi / 2, 0.0], [0.0, np.pi / 3]]),
        ("j0", "j1"),
        ("tip",),
    )

    assert result.shape == (2, 1, 4, 4)
    np.testing.assert_allclose(result[0, 0, :3, 3], [0.0, 2.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(result[1, 0, :3, 3], [2.0, 0.0, 0.0], atol=1e-8)


def test_mimic_joint_uses_master_multiplier_and_offset(tmp_path):
    tree = UrdfKinematicTree.from_file(
        _write_urdf(tmp_path, _two_joint_urdf(mimic=True, multiplier=0.5, offset=0.1))
    )

    expanded = tree.expanded_joint_positions({"j0": 0.4})

    assert expanded["j1"] == pytest.approx(0.3)


def test_explicit_mimic_override_uses_recorded_value_without_weakening_default(tmp_path):
    tree = UrdfKinematicTree.from_file(
        _write_urdf(tmp_path, _two_joint_urdf(mimic=True, multiplier=0.5, offset=0.1))
    )

    with pytest.raises(ValueError, match="cannot be supplied directly"):
        tree.forward_links(np.array([[0.4, 1.2]]), ("j0", "j1"), ("tip",))

    result = tree.forward_links(
        np.array([[0.4, 1.2]]),
        ("j0", "j1"),
        ("tip",),
        mimic_override_joints=("j1",),
    )

    np.testing.assert_allclose(result[0, 0, :3, 3], [2.0 * np.cos(0.4), 2.0 * np.sin(0.4), 0.0])
    assert tree.expanded_joint_positions({"j0": 0.4})["j1"] == pytest.approx(0.3)


def test_declared_override_is_a_noop_for_an_already_independent_joint(tmp_path):
    tree = UrdfKinematicTree.from_file(_write_urdf(tmp_path, _two_joint_urdf()))

    result = tree.forward_links(
        np.array([[0.4, 1.2]]),
        ("j0", "j1"),
        ("tip",),
        mimic_override_joints=("j1",),
    )

    np.testing.assert_allclose(result[0, 0, :3, 3], [2.0 * np.cos(0.4), 2.0 * np.sin(0.4), 0.0])


def test_pinned_inspire_dofs_explicitly_override_only_urdf_mimic_joints():
    root = Path(__file__).parents[1] / "data" / "external" / "dexmanipnet"
    source_root = root / "source_maniptrans"
    hdf_path = root / "extracted" / "dexmanipnet_favor" / "sequences" / "1001_rh" / "rollouts.hdf5"
    if not source_root.is_dir() or not hdf_path.is_file():
        pytest.skip("pinned DexManipNet input is not installed")
    h5py = pytest.importorskip("h5py")
    spec = SOURCE_HANDS["inspire_rh"]
    tree = UrdfKinematicTree.from_file(source_root / spec.urdf_relpath)
    urdf_mimics = tuple(name for name in spec.joint_order if tree.joints[name].mimic is not None)

    assert spec.mimic_override_joints == urdf_mimics
    with h5py.File(hdf_path, "r") as stream:
        q = np.asarray(stream["rollouts/successful/rollout_0/q_rh"][:4], dtype=np.float64)
    assert q.shape == (4, len(spec.joint_order))
    with pytest.raises(ValueError, match="cannot be supplied directly"):
        tree.forward_links(q, spec.joint_order, spec.fingertip_links)

    fingertips = tree.palm_relative_fingertips(q, spec)
    assert fingertips.shape == (4, 5, 3)
    assert np.isfinite(fingertips).all()


def test_fixed_prismatic_origin_rpy_and_joint_limits_are_parsed(tmp_path):
    tree = UrdfKinematicTree.from_file(
        _write_urdf(
            tmp_path,
            """
            <link name="base"/><link name="offset"/><link name="tip"/>
            <joint name="mount" type="fixed">
              <parent link="base"/><child link="offset"/>
              <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
            </joint>
            <joint name="slide" type="prismatic">
              <parent link="offset"/><child link="tip"/>
              <axis xyz="1 0 0"/><limit lower="-0.2" upper="0.5" effort="8" velocity="2"/>
            </joint>
            """,
        )
    )

    result = tree.forward_links(np.array([[0.25]]), ("slide",), ("tip",))

    np.testing.assert_allclose(result[0, 0, :3, 3], [1.0, 2.25, 3.0], atol=1e-8)
    assert tree.joints["slide"].limit.lower == pytest.approx(-0.2)
    assert tree.joints["slide"].limit.upper == pytest.approx(0.5)
    assert tree.joints["slide"].limit.effort == pytest.approx(8.0)
    assert tree.joints["slide"].limit.velocity == pytest.approx(2.0)


def test_palm_relative_five_tip_fk_uses_frozen_spec_order(tmp_path):
    tree = UrdfKinematicTree.from_file(
        _write_urdf(
            tmp_path,
            """
            <link name="world"/><link name="palm"/>
            <link name="thumb"/><link name="index"/><link name="middle"/>
            <link name="ring"/><link name="pinky"/>
            <joint name="palm_joint" type="continuous">
              <parent link="world"/><child link="palm"/>
              <origin xyz="10 -3 2"/><axis xyz="0 0 2"/>
            </joint>
            <joint name="z_pinky" type="fixed"><parent link="palm"/><child link="pinky"/><origin xyz="5 0 0"/></joint>
            <joint name="a_thumb" type="fixed"><parent link="palm"/><child link="thumb"/><origin xyz="1 0 0"/></joint>
            <joint name="m_middle" type="fixed"><parent link="palm"/><child link="middle"/><origin xyz="3 0 0"/></joint>
            <joint name="i_index" type="fixed"><parent link="palm"/><child link="index"/><origin xyz="2 0 0"/></joint>
            <joint name="r_ring" type="fixed"><parent link="palm"/><child link="ring"/><origin xyz="4 0 0"/></joint>
            """,
        )
    )
    spec = SourceHandSpec(
        name="fixture",
        side="rh",
        urdf_relpath="fixture.urdf",
        joint_order=("palm_joint",),
        palm_link="palm",
        fingertip_links=("thumb", "index", "middle", "ring", "pinky"),
    )

    positions = tree.palm_relative_fingertips(np.array([[0.7], [-1.2]]), spec)

    assert tree.topological_joint_names == (
        "palm_joint",
        "a_thumb",
        "i_index",
        "m_middle",
        "r_ring",
        "z_pinky",
    )
    assert positions.shape == (2, 5, 3)
    np.testing.assert_allclose(
        positions,
        np.broadcast_to(np.arange(1.0, 6.0)[None, :, None] * [1.0, 0.0, 0.0], (2, 5, 3)),
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('<link name="base"/><link name="base"/>', "duplicate link"),
        (
            '<link name="base"/><link name="a"/><link name="b"/>'
            '<joint name="j" type="fixed"><parent link="base"/><child link="a"/></joint>'
            '<joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>',
            "duplicate joint",
        ),
        (
            '<link name="base"/><link name="a"/>'
            '<joint name="j" type="floating"><parent link="base"/><child link="a"/></joint>',
            "joint type",
        ),
        (
            '<link name="base"/><link name="a"/>'
            '<joint name="j" type="fixed"><parent link="missing"/><child link="a"/></joint>',
            "unknown parent",
        ),
        (
            '<link name="base"/><link name="a"/>'
            '<joint name="j" type="fixed"><parent link="base"/><child link="missing"/></joint>',
            "unknown child",
        ),
        (
            '<link name="base"/><link name="a"/><link name="b"/>'
            '<joint name="j0" type="fixed"><parent link="base"/><child link="b"/></joint>'
            '<joint name="j1" type="fixed"><parent link="a"/><child link="b"/></joint>',
            "multiple parents",
        ),
        (
            '<link name="a"/><link name="b"/>'
            '<joint name="j0" type="fixed"><parent link="a"/><child link="b"/></joint>'
            '<joint name="j1" type="fixed"><parent link="b"/><child link="a"/></joint>',
            "root",
        ),
        ('<link name="a"/><link name="b"/>', "one root"),
        (
            '<link name="base"/><link name="a"/>'
            '<joint name="j" type="revolute"><parent link="base"/><child link="a"/>'
            '<axis xyz="0 0 0"/></joint>',
            "axis",
        ),
        (
            '<link name="base"/><link name="a"/>'
            '<joint name="j" type="fixed"><parent link="base"/><child link="a"/>'
            '<origin xyz="nan 0 0"/></joint>',
            "finite",
        ),
    ],
)
def test_parser_rejects_invalid_xml_tree_contracts(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        UrdfKinematicTree.from_file(_write_urdf(tmp_path, body))


def test_parser_rejects_mimic_cycles(tmp_path):
    body = """
        <link name="base"/><link name="a"/><link name="b"/>
        <joint name="j0" type="revolute"><parent link="base"/><child link="a"/>
          <mimic joint="j1"/></joint>
        <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
          <mimic joint="j0"/></joint>
    """

    with pytest.raises(ValueError, match="mimic cycle"):
        UrdfKinematicTree.from_file(_write_urdf(tmp_path, body))


@pytest.mark.parametrize(
    ("q", "joint_order", "links", "message"),
    [
        (np.zeros((2, 1)), ("j0", "j1"), ("tip",), "columns"),
        (np.array([[np.nan, 0.0]]), ("j0", "j1"), ("tip",), "finite"),
        (np.zeros((1, 2)), ("j0", "j1"), ("missing",), "requested link"),
        (np.zeros((1, 2)), ("j0", "j0"), ("tip",), "unique"),
        (np.zeros((1, 2)), ("j0", "unknown"), ("tip",), "unknown joint"),
    ],
)
def test_forward_links_rejects_invalid_requests(tmp_path, q, joint_order, links, message):
    tree = UrdfKinematicTree.from_file(_write_urdf(tmp_path, _two_joint_urdf()))

    with pytest.raises(ValueError, match=message):
        tree.forward_links(q, joint_order, links)
