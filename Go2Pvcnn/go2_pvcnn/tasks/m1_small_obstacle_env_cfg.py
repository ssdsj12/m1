"""M1 wheel-assisted wave curriculum over a deterministic low obstacle."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from go2_pvcnn.assets import M1_BASE_BODY_NAME, M1_FOOT_BODY_NAMES
from go2_pvcnn.tasks.m1_roll_env_cfg import M1RollObservationsCfg, M1RollRewardsCfg
from go2_pvcnn.tasks.m1_smoke_env_cfg import M1SmokeSceneCfg
from go2_pvcnn.tasks.m1_wave_env_cfg import M1WaveFlatEnvCfg
import go2_pvcnn.mdp as mdp


def make_obstacle_cfg(height: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.80, height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.18, 0.08)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.5 * height)),
    )


@configclass
class M1SmallObstacleSceneCfg(M1SmokeSceneCfg):
    """Generated bar terrain and a local 16x16 height-scan teacher."""

    obstacle = make_obstacle_cfg(0.001)
    height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.25, 0.0, 1.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.5, 1.5)),
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Obstacle", is_shared=True, track_mesh_transforms=False
            ),
        ],
        max_distance=3.0,
        debug_vis=False,
    )


@configclass
class M1SmallObstacleObservationsCfg(M1RollObservationsCfg):
    """Roll proprioception plus the Go2Pvcnn-compatible 16x16 teacher scan."""

    @configclass
    class PolicyCfg(M1RollObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=isaac_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class M1SmallObstacleRewardsCfg(M1RollRewardsCfg):
    """Stable progress plus a sparse whole-robot crossing signal."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.02},
    )
    base_height = RewTerm(func=mdp.base_height_l2, weight=-20.0, params={"target_height": 0.60})
    base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=M1_BASE_BODY_NAME),
        },
    )
    termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-200.0)
    obstacle_progress = RewTerm(
        func=mdp.obstacle_progress,
        weight=1.0,
        params={"obstacle_distance": 0.57, "rear_axle_offset": 0.33, "clearance_margin": 0.05},
    )
    forward_velocity = RewTerm(func=mdp.forward_velocity, weight=8.0, params={"max_velocity": 0.25})
    front_wheel_lift = RewTerm(
        func=mdp.obstacle_front_wheel_lift,
        weight=5.0,
        params={
            "approach_x": 0.05,
            "release_x": 0.40,
            "baseline_height": 0.096,
            "target_height": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=list(M1_FOOT_BODY_NAMES[:2])),
        },
    )
    front_wheels_over = RewTerm(func=mdp.ObstacleMilestoneOnce, weight=5.0, params={"threshold": 0.34})
    front_axle_clear = RewTerm(func=mdp.ObstacleMilestoneOnce, weight=8.0, params={"threshold": 0.50})
    base_over = RewTerm(func=mdp.ObstacleMilestoneOnce, weight=10.0, params={"threshold": 0.60})
    rear_axle_approach = RewTerm(func=mdp.ObstacleMilestoneOnce, weight=15.0, params={"threshold": 0.75})
    rear_axle_near_clear = RewTerm(func=mdp.ObstacleMilestoneOnce, weight=20.0, params={"threshold": 0.90})
    obstacle_passed = RewTerm(
        func=mdp.ObstacleMilestoneOnce,
        weight=30.0,
        params={"threshold": 0.95},
    )


@configclass
class M1SmallObstacleEnvCfg(M1WaveFlatEnvCfg):
    """First M1 low-obstacle teacher stage."""

    scene: M1SmallObstacleSceneCfg = M1SmallObstacleSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)
    observations: M1SmallObstacleObservationsCfg = M1SmallObstacleObservationsCfg()
    rewards: M1SmallObstacleRewardsCfg = M1SmallObstacleRewardsCfg()
    wave_leg_action_limit: float = 0.40
    wave_left_right_symmetric: bool = True
    wave_lock_abduction: bool = True
    wave_reference_raw_amplitude: float = 0.20
    wave_obstacle_wheel_boost: float = 0.08
    wave_obstacle_boost_start_x: float = 0.05
    wave_obstacle_boost_end_x: float = 0.40

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.terminations.base_contact = None
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation


@configclass
class M1SmallObstacle5mmSceneCfg(M1SmallObstacleSceneCfg):
    obstacle = make_obstacle_cfg(0.005)


@configclass
class M1SmallObstacle5mmEnvCfg(M1SmallObstacleEnvCfg):
    scene: M1SmallObstacle5mmSceneCfg = M1SmallObstacle5mmSceneCfg(
        num_envs=64, env_spacing=2.5, replicate_physics=True
    )


@configclass
class M1SmallObstacle10mmSceneCfg(M1SmallObstacleSceneCfg):
    obstacle = make_obstacle_cfg(0.010)


@configclass
class M1SmallObstacle10mmEnvCfg(M1SmallObstacleEnvCfg):
    scene: M1SmallObstacle10mmSceneCfg = M1SmallObstacle10mmSceneCfg(
        num_envs=64, env_spacing=2.5, replicate_physics=True
    )
