"""Deterministic transverse-bar terrain for M1 crossing curricula."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def transverse_bar_terrain(
    difficulty: float, cfg: "M1TransverseBarTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a flat tile with one full-width low bar in front of spawn."""
    height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )
    center_x = 0.5 * cfg.size[0]
    center_y = 0.5 * cfg.size[1]
    ground = trimesh.creation.box(
        (cfg.size[0], cfg.size[1], cfg.ground_thickness),
        transform=trimesh.transformations.translation_matrix(
            (center_x, center_y, -0.5 * cfg.ground_thickness)
        ),
    )
    bar_center_x = center_x + cfg.obstacle_distance
    half_plateau = 0.5 * cfg.obstacle_depth
    xs = np.array(
        (
            bar_center_x - half_plateau - cfg.obstacle_ramp_length,
            bar_center_x - half_plateau,
            bar_center_x + half_plateau,
            bar_center_x + half_plateau + cfg.obstacle_ramp_length,
        )
    )
    zs = np.array((0.0, height, height, 0.0))
    half_width = 0.5 * cfg.obstacle_width
    top = [(x, center_y + y, z) for x, z in zip(xs, zs) for y in (-half_width, half_width)]
    bottom = [(x, center_y + y, 0.0) for x in xs for y in (-half_width, half_width)]
    vertices = np.asarray(top + bottom)
    faces: list[tuple[int, int, int]] = []
    for segment in range(3):
        a, b = 2 * segment, 2 * segment + 1
        c, d = 2 * (segment + 1), 2 * (segment + 1) + 1
        faces.extend(((a, c, d), (a, d, b)))
        faces.extend(((a + 8, d + 8, c + 8), (a + 8, b + 8, d + 8)))
        faces.extend(((a, a + 8, c + 8), (a, c + 8, c)))
        faces.extend(((b, d + 8, b + 8), (b, d, d + 8)))
    faces.extend(((0, 1, 9), (0, 9, 8), (6, 14, 15), (6, 15, 7)))
    bar = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    return [ground, bar], np.array((center_x, center_y, 0.0))


@configclass
class M1TransverseBarTerrainCfg(SubTerrainBaseCfg):
    """Parameters for a single transverse obstacle bar."""

    function = transverse_bar_terrain
    obstacle_height_range: tuple[float, float] = (0.001, 0.03)
    obstacle_distance: float = 0.55
    obstacle_depth: float = 0.04
    obstacle_ramp_length: float = 0.08
    obstacle_width: float = 0.80
    ground_thickness: float = 0.10
