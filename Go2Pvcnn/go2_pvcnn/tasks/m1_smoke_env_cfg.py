"""Minimal M1 smoke environment.

This config intentionally avoids the Go2-specific MPC path. Its job is to prove
that the M1 USD articulation loads, resets, accepts 16 joint-position actions,
and exposes basic contact/body state in IsaacLab.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from go2_pvcnn.assets import (
    M1_BASE_BODY_NAME,
    M1_CFG,
    M1_FOOT_BODY_NAMES,
    M1_LEG_JOINT_NAMES,
    M1_ROLLING_MODE,
    M1_WAVE_MODE,
    M1_WHEEL_JOINT_NAMES,
)
import go2_pvcnn.mdp as mdp


@configclass
class M1SmokeSceneCfg(InteractiveSceneCfg):
    """Small flat scene for validating M1 articulation startup."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot = M1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class M1SmokeEventsCfg:
    """Startup and reset events for M1 smoke validation."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 8,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class M1SmokeActionsCfg:
    """Hybrid M1 action: 12 leg positions plus 4 wheel velocities."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(M1_LEG_JOINT_NAMES),
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(M1_WHEEL_JOINT_NAMES),
        scale=8.0,
        use_default_offset=True,
        clip={".*": (-8.0, 8.0)},
    )


@configclass
class M1SmokeObservationsCfg:
    """Basic proprioceptive observations for smoke testing."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-0.0, n_max=0.0))
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class M1SmokeRewardsCfg:
    """Small rewards that keep the env manager valid without claiming locomotion quality."""

    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(M1_FOOT_BODY_NAMES)),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES)),
        },
    )


@configclass
class M1SmokeTerminationsCfg:
    """Conservative smoke-test terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=M1_BASE_BODY_NAME), "threshold": 10.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})


@configclass
class M1SmokeCommandsCfg:
    """No command terms for the first M1 smoke environment."""

    pass


@configclass
class M1SmokeCurriculumCfg:
    """No curriculum terms for the first M1 smoke environment."""

    pass


@configclass
class M1SmokeEnvCfg(ManagerBasedRLEnvCfg):
    """M1 no-MPC smoke environment."""

    scene: M1SmokeSceneCfg = M1SmokeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: M1SmokeObservationsCfg = M1SmokeObservationsCfg()
    actions: M1SmokeActionsCfg = M1SmokeActionsCfg()
    commands: M1SmokeCommandsCfg = M1SmokeCommandsCfg()
    rewards: M1SmokeRewardsCfg = M1SmokeRewardsCfg()
    events: M1SmokeEventsCfg = M1SmokeEventsCfg()
    terminations: M1SmokeTerminationsCfg = M1SmokeTerminationsCfg()
    curriculum: M1SmokeCurriculumCfg = M1SmokeCurriculumCfg()

    planner_backend: str = "none"
    planner_owned_reference_cache: bool = False
    use_batched_reference_trajectory: bool = False
    control_mode: str = M1_ROLLING_MODE
    available_control_modes: tuple[str, str] = (M1_ROLLING_MODE, M1_WAVE_MODE)
    rolling_wheel_velocity: float = 0.5
    wave_wheel_velocity: float = 1.5
    wave_amplitude: float = 0.0
    wave_frequency: float = 1.0
    wave_phase_offsets: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 0.5)

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
