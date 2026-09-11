from __future__ import annotations

from pathlib import Path

import torch

from go2_pvcnn.control.m1_bimanual_coordination.contact_summary import summarize_contacts


ROOT = Path(__file__).resolve().parents[1]


def test_box_filtered_force_is_the_contact_source_of_truth() -> None:
    result = summarize_contacts(
        side="right",
        candidate_names=(("right_index_proximal", "right_index_distal"),),
        filtered_forces_w=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=torch.float64
        ),
        raw_forces_w=torch.tensor(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 1.2]], dtype=torch.float64
        ),
        threshold_n=0.2,
    )

    assert result.selected_names == ("right_index_distal",)
    assert result.contact_mask.tolist() == [True]
    assert result.filtered_force_max_n == 1.2
    assert result.consistency_reason is None


def test_raw_force_without_box_force_is_not_box_contact() -> None:
    result = summarize_contacts(
        side="right",
        candidate_names=(("right_thumb_distal",),),
        filtered_forces_w=torch.zeros((1, 3), dtype=torch.float64),
        raw_forces_w=torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float64),
        threshold_n=0.2,
    )

    assert not result.contact_mask.any()
    assert result.consistency_reason == "raw_contact_without_box_contact"


def test_selection_is_deterministic_when_filtered_forces_are_zero() -> None:
    result = summarize_contacts(
        side="left",
        candidate_names=(("left_index_proximal", "left_index_distal"),),
        filtered_forces_w=torch.zeros((2, 3), dtype=torch.float64),
        raw_forces_w=torch.zeros((2, 3), dtype=torch.float64),
    )

    assert result.selected_names == ("left_index_proximal",)
    assert result.consistency_reason is None


def test_isaac_adapter_uses_canonical_filtered_and_raw_contact_summary() -> None:
    source = (
        ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
    ).read_text()
    assert "summarize_contacts(" in source
    assert ".data.net_forces_w" in source
    assert "self.contact_summaries[side] = summary" in source


def test_isaac_adapter_uses_deterministic_simulation_timestamps() -> None:
    source = (
        ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
    ).read_text()
    assert "round(self._sequence * float(self.env.physics_dt) * 1e9)" in source
    assert "time.monotonic_ns()" not in source


def test_box_filter_uses_one_sensor_body_per_contact_sensor() -> None:
    env_source = (
        ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py"
    ).read_text()
    wrapper_source = (
        ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
    ).read_text()
    assert "def _box_contact_sensor(" in env_source
    assert "left_thumb_distal_box_contact" in env_source
    assert "right_pinky_distal_box_contact" in env_source
    assert "filter_prim_paths_expr=[]" in env_source
    assert "filtered_sensor_names" in wrapper_source
