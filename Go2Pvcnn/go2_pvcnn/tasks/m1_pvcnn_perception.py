"""PVCNN perception adapter for the M1 semantic crossing task."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper


def grid_elevation_to_point_cloud(
    elevation: torch.Tensor,
    x_size: float = 1.5,
    y_size: float = 1.5,
) -> torch.Tensor:
    """Convert a batched elevation grid into yaw-aligned XYZ points."""
    if elevation.ndim != 3:
        raise ValueError(f"Expected elevation [B,H,W], got {tuple(elevation.shape)}")
    batch, height, width = elevation.shape
    x = torch.linspace(-0.5 * x_size, 0.5 * x_size, height, device=elevation.device, dtype=elevation.dtype)
    y = torch.linspace(-0.5 * y_size, 0.5 * y_size, width, device=elevation.device, dtype=elevation.dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack((grid_x, grid_y), dim=-1).reshape(1, height * width, 2)
    xy = xy.expand(batch, -1, -1)
    return torch.cat((xy, -elevation.reshape(batch, height * width, 1)), dim=-1)


def logits_to_semantic_channel(logits: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert per-point class logits to the expected semantic-id channel."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,C,N], got {tuple(logits.shape)}")
    if logits.shape[2] != height * width:
        raise ValueError(f"Expected {height * width} points, got {logits.shape[2]}")
    class_ids = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype).reshape(1, -1, 1)
    expected_id = (torch.softmax(logits, dim=1) * class_ids).sum(dim=1)
    return expected_id.reshape(logits.shape[0], height, width)


def downsample_perception_maps(
    elevation: torch.Tensor,
    semantic: torch.Tensor,
    target_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample scanner maps with the same pooling contract as policy observations."""
    elevation_small = F.adaptive_avg_pool2d(elevation.unsqueeze(1), (target_size, target_size)).squeeze(1)
    semantic_small = F.adaptive_max_pool2d(
        semantic.to(torch.float32).unsqueeze(1), (target_size, target_size)
    ).squeeze(1).to(torch.long)
    return elevation_small, semantic_small


class M1PvcnnRslRlEnvWrapper(M1RslRlEnvWrapper):
    """Replace the actor semantic map with PVCNN predictions; keep critic ground truth."""

    def __init__(self, env, pvcnn_model, clip_actions: float | None = 1.0):
        self.pvcnn_model = pvcnn_model
        super().__init__(env, clip_actions=clip_actions)

    def _format_observations(self, obs_dict) -> tuple[torch.Tensor, dict]:
        raw_map = obs_dict["policy_elevation_semantic_map"]
        if raw_map.ndim != 4 or raw_map.shape[1] != 2:
            raise ValueError(f"Expected policy map [B,2,H,W], got {tuple(raw_map.shape)}")
        elevation = raw_map[:, 0]
        labels = raw_map[:, 1].round().to(torch.long).clamp(min=0, max=2)
        point_cloud = grid_elevation_to_point_cloud(elevation)
        self.pvcnn_model.eval()
        with torch.no_grad():
            logits = self.pvcnn_model(point_cloud.transpose(1, 2).contiguous())
            predicted_semantic = logits_to_semantic_channel(
                logits, height=elevation.shape[1], width=elevation.shape[2]
            )
        predicted_map = torch.stack((elevation, predicted_semantic), dim=1)

        state = obs_dict["policy"].reshape(raw_map.shape[0], -1)
        policy_obs = torch.cat((state, predicted_map.reshape(raw_map.shape[0], -1)), dim=-1)
        critic_obs = torch.cat((state, raw_map.reshape(raw_map.shape[0], -1)), dim=-1)
        if bool(getattr(self.cfg, "wave_policy_phase_observation", False)):
            phase_features = self._phase_observation(policy_obs)
            policy_obs = torch.cat((policy_obs, phase_features), dim=1)
            critic_obs = torch.cat((critic_obs, phase_features), dim=1)
        self.env.unwrapped.last_point_cloud = point_cloud.detach()
        self.env.unwrapped.last_semantic_labels = labels.reshape(labels.shape[0], -1).detach()
        return policy_obs, {"observations": {"critic": critic_obs}}
