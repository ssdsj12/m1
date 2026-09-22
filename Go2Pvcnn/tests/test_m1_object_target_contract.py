from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination import (
    BimanualPhase,
    BimanualSnapshot,
    BoxState,
    ObjectMpcInput,
    ObjectState,
    SideArmState,
    SideHandState,
)
from go2_pvcnn.control.m1_bimanual_coordination.task_goal import (
    resolve_target_object_ids,
)
from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import ObjectInstance


DTYPE = torch.float64
ROOT = Path(__file__).resolve().parents[1]


def _arm() -> SideArmState:
    return SideArmState(
        q=torch.zeros(7, dtype=DTYPE),
        qd=torch.zeros(7, dtype=DTYPE),
        palm_pose_b=torch.zeros(6, dtype=DTYPE),
        palm_twist_b=torch.zeros(6, dtype=DTYPE),
        jacobian_b=torch.eye(6, 7, dtype=DTYPE),
        mass_matrix=torch.eye(7, dtype=DTYPE),
        bias=torch.zeros(7, dtype=DTYPE),
    )


def _hand() -> SideHandState:
    return SideHandState(
        q=torch.zeros(6, dtype=DTYPE),
        qd=torch.zeros(6, dtype=DTYPE),
        fingertip_forces_b=torch.zeros(5, 3, dtype=DTYPE),
        fingertip_positions_b=torch.zeros(5, 3, dtype=DTYPE),
        fingertip_jacobian_b=torch.zeros(15, 6, dtype=DTYPE),
        contact_mask=torch.zeros(5, dtype=torch.bool),
    )


def _snapshot() -> BimanualSnapshot:
    return BimanualSnapshot(
        timestamp_ns=1,
        base_state=torch.zeros(13, dtype=DTYPE),
        m1_q=torch.zeros(16, dtype=DTYPE),
        m1_qd=torch.zeros(16, dtype=DTYPE),
        platform_q_qd=torch.zeros(2, dtype=DTYPE),
        left_arm=_arm(),
        right_arm=_arm(),
        left_hand=_hand(),
        right_hand=_hand(),
        box=BoxState(
            pose_b=torch.zeros(6, dtype=DTYPE),
            twist_b=torch.zeros(6, dtype=DTYPE),
            mass=torch.tensor(0.5, dtype=DTYPE),
            inertia_b=torch.eye(3, dtype=DTYPE),
            supported=True,
        ),
        obstacle_objects=(
            ObjectState(
                pose_b=torch.tensor(
                    [0.3, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=DTYPE
                ),
                twist_b=torch.zeros(6, dtype=DTYPE),
            ),
        ),
    )


def test_play_parser_keeps_box_default_and_accepts_explicit_object_id():
    import importlib.util

    path = ROOT / "scripts/m1_dual_panda_o6_bimanual_play.py"
    spec = importlib.util.spec_from_file_location("m1_o6_play", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._parser().parse_args([]).object_id is None
    assert module._parser().parse_args(["--object-id", "cup_000"]).object_id == "cup_000"


def test_unknown_target_id_is_rejected_before_scene_startup():
    instances = (
        ObjectInstance("cube_000", "cube", (0.0, 0.0, 0.5)),
        ObjectInstance("bottle_000", "bottle", (0.5, 0.0, 0.5)),
    )

    with pytest.raises(ValueError, match="unknown target object_id"):
        resolve_target_object_ids(instances, "missing")


def test_explicit_target_takes_precedence_and_exposes_other_enabled_obstacles():
    instances = (
        ObjectInstance("cube_000", "cube", (0.0, 0.0, 0.5)),
        ObjectInstance("bottle_000", "bottle", (0.5, 0.0, 0.5)),
        ObjectInstance("disabled", "cube", (1.0, 0.0, 0.5), enabled=False),
    )

    target_id, obstacle_ids = resolve_target_object_ids(instances, "bottle_000")

    assert target_id == "bottle_000"
    assert obstacle_ids == ("cube_000",)


def test_legacy_omission_keeps_box_target_and_exposes_catalog_objects_as_obstacles():
    instances = (ObjectInstance("cube_000", "cube", (0.0, 0.0, 0.5)),)

    target_id, obstacle_ids = resolve_target_object_ids(instances, None)

    assert target_id is None
    assert obstacle_ids == ("cube_000",)


def test_mpc_goal_boundary_carries_selected_target_and_obstacle_poses():
    snapshot = _snapshot()
    target = torch.zeros((25, 6), dtype=DTYPE)
    target[:, 2] = 0.1
    obstacle = snapshot.obstacle_objects[0].pose_b.repeat(25, 1)

    sample = ObjectMpcInput(
        snapshot=snapshot,
        target_object_pose_b=target,
        obstacle_object_poses_b=(obstacle,),
        phase=BimanualPhase.APPROACH,
    )

    assert torch.equal(sample.target_object_pose_b, target)
    assert torch.equal(sample.target_box_pose_b, target)
    assert len(sample.obstacle_object_poses_b) == 1
    assert torch.equal(sample.obstacle_object_poses_b[0], obstacle)


def test_snapshot_target_alias_remains_box_compatible():
    snapshot = _snapshot()
    assert snapshot.target_object is snapshot.box
    assert len(snapshot.obstacle_objects) == 1
