from __future__ import annotations

from pathlib import Path

import torch

from go2_pvcnn.control.m1_bimanual_coordination.contracts import BimanualCommand
from go2_pvcnn.control.m1_bimanual_coordination.vector_control import (
    stack_lane_actions,
)


ROOT = Path(__file__).resolve().parents[1]


def _command(value: float, timestamp_ns: int) -> BimanualCommand:
    return BimanualCommand(
        timestamp_ns=timestamp_ns,
        effort=torch.full((43,), value, dtype=torch.float64),
        feasible=True,
        fallback_reasons=(),
    )


def test_stack_lane_actions_preserves_independent_commands() -> None:
    actions = stack_lane_actions(
        (_command(1.0, 1), _command(2.0, 1)),
        device=torch.device("cpu"),
    )
    assert actions.shape == (2, 43)
    assert torch.all(actions[0] == 1.0)
    assert torch.all(actions[1] == 2.0)
    assert actions.dtype == torch.float32


def test_wrapper_source_has_one_controller_per_lane_and_no_env0_repeat() -> None:
    source = (
        ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
    ).read_text()
    assert "self.lanes = [" in source
    assert "env_index=index" in source
    assert "stack_lane_actions(" in source
    assert ".repeat(self.env.unwrapped.num_envs, 1)" not in source
    assert "self._left_hand_contact_q = lane.left_hand_contact_q" in source
    assert "self._right_hand_contact_q = lane.right_hand_contact_q" in source


def test_probe_passes_requested_vector_environment_count() -> None:
    source = (
        ROOT / "scripts/m1_dual_panda_o6_bimanual_probe.py"
    ).read_text()
    assert "cfg.scene.num_envs = args.num_envs" in source
    assert "currently requires --num-envs 1" not in source
