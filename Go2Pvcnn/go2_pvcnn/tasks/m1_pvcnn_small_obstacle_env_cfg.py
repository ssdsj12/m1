"""M1 adaptation of the original Go2Pvcnn semantic flat-small environment."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from extension.mdp.observations import downsampled_elevation_semantic_scan
from extension.mdp.semantic_body_part_clearance import (
    semantic_body_part_clearance_reward,
    semantic_foot_over_clearance_bonus,
)
from extension.semantic_curriculum import SemanticObstacleCount
from go2_pvcnn.assets import (
    M1_BASE_BODY_NAME,
    M1_CFG,
    M1_FOOT_BODY_NAMES,
    M1_LEG_JOINT_NAMES,
    M1_WHEEL_JOINT_NAMES,
)
import go2_pvcnn.mdp as mdp
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg
from go2_pvcnn.sensor.semantic_contacter.semantic_global_contact_sensor import (
    M1SemanticGlobalContactSensor,
)
from go2_pvcnn.tasks.m1_roll_env_cfg import M1RollActionsCfg
from go2_pvcnn.tasks.m1_curriculum import (
    build_semantic_spatial_wave_reference,
    spatial_pair_lift_score,
    spatial_wheel_lift_score,
    wheel_crossbar_collision_mask,
    sequential_wheel_crossing_progress_score,
    strict_sequential_crossing_success,
    axle_pair_crossing_progress_score,
    progress_potential_delta,
)
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
    TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
    TeacherElevationTrajectoryMpcSemanticRewardsCfg,
    TeacherElevationTrajectoryMpcSemanticSceneCfg,
)


def _m1_semantic_global_contact_sensor(semantic_root: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        class_type=M1SemanticGlobalContactSensor,
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        debug_vis=False,
        filter_prim_paths_expr=[f"{semantic_root}/.*"],
    )


def _m1_semantic_body_clearance_term() -> RewTerm:
    """Penalize M1 wheel/leg/body overlap with semantic small obstacles."""
    return RewTerm(
        func=semantic_body_part_clearance_reward,
        weight=0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=list(M1_FOOT_BODY_NAMES)
            ),
            "small_semantic_ids": (1,),
            "foot_margin_m": 0.02,
            "calf_margin_m": 0.05,
            "thigh_margin_m": 0.05,
            "base_margin_m": 0.05,
            "foot_weight": 0.5,
            "calf_weight": 2.0,
            "thigh_weight": 1.5,
            "base_weight": 2.0,
            "foot_sphere_radius_m": 0.096,
            "foot_query_radius_m": 0.25,
            "calf_capsule_radius_m": 0.045,
            "calf_query_radius_m": 0.25,
            "calf_sections": 7,
            "thigh_capsule_radius_m": 0.045,
            "thigh_query_radius_m": 0.25,
            "thigh_sections": 7,
            "base_half_extents_m": (0.30, 0.12, 0.10),
            "base_footprint_grid": (5, 3),
            "base_query_radius_m": 0.25,
            "include_base": True,
            "penalty_clip": 1.0,
            "clearance_scale": 20.0,
            "contact_collision_scale": 1.0,
            "contact_force_scale": 25.0,
            "contact_force_clip": 1.0,
        },
    )


def _m1_semantic_wheel_over_term() -> RewTerm:
    """Reward wheel centers that clear low semantic obstacles in the command corridor."""
    return RewTerm(
        func=semantic_foot_over_clearance_bonus,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
            "command_name": "base_velocity",
            "small_semantic_ids": (1,),
            "corridor_width_m": 0.42,
            "lookahead_m": 1.2,
            "low_small_max_height_m": 0.20,
            "obstacle_half_extent_m": 0.08,
            "clearance_margin_m": 0.02,
            "bonus_clip": 1.0,
            "bonus_scale": 0.5,
        },
    )


def _m1_wheel_velocity_sync_term(weight: float = -10.0) -> RewTerm:
    """Penalize front/rear actual wheel speed mismatch so stance stays level."""
    return RewTerm(
        func=mdp.paired_joint_speed_mismatch,
        weight=weight,
        params={
            "front_asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(M1_WHEEL_JOINT_NAMES[:2]), preserve_order=True
            ),
            "rear_asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(M1_WHEEL_JOINT_NAMES[2:]), preserve_order=True
            ),
        },
    )


def _m1_semantic_pair_lift_reward(
    env,
    asset_cfg: SceneEntityCfg,
    scanner_cfg: SceneEntityCfg,
    corridor_half_width_m: float,
    baseline_height_m: float,
    target_height_m: float,
):
    """Reward lifting both wheels on the axle currently crossing a small obstacle."""
    asset = env.scene[asset_cfg.name]
    scanner = env.scene[scanner_cfg.name]
    _, active, obstacle_x = build_semantic_spatial_wave_reference(
        root_pos_w=asset.data.root_pos_w,
        root_quat_w=asset.data.root_quat_w,
        ray_hits_w=scanner.data.ray_hits_w,
        semantic_map=scanner.data.semantic_map,
        amplitude=0.0,
        knee_ratio=1.0,
        rear_amplitude_scale=1.0,
        front_overlap_scale=1.0,
        front_support_ratio=0.0,
        corridor_half_width_m=corridor_half_width_m,
    )
    wheel_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return spatial_wheel_lift_score(
        wheel_heights=wheel_heights,
        obstacle_x=obstacle_x,
        active=active,
        baseline_height=baseline_height_m,
        target_height=target_height_m,
    )


def _m1_raw_leg_action_l2(env):
    """Penalize policy outputs before the M1 wrapper applies residual limits."""
    actions = getattr(env.unwrapped, "m1_raw_policy_actions", None)
    if actions is None:
        return env.scene["robot"].data.root_pos_w[:, 0] * 0.0
    return actions[:, :12].square().sum(dim=1)


def _m1_sequential_crossing_progress(env):
    phase = getattr(env.unwrapped, "m1_sequential_crossing_phase", None)
    wheel_x = getattr(env.unwrapped, "m1_sequential_wheel_x_from_obstacle", None)
    wheel_heights = getattr(env.unwrapped, "m1_sequential_wheel_heights", None)
    if phase is None or wheel_x is None or wheel_heights is None:
        return env.scene["robot"].data.root_pos_w[:, 0] * 0.0
    score_func = (
        axle_pair_crossing_progress_score
        if bool(getattr(env.unwrapped.cfg, "wave_axle_pair_crossing_reference", False))
        else sequential_wheel_crossing_progress_score
    )
    score = score_func(
        phase=phase,
        wheel_x_from_obstacle=wheel_x,
        wheel_heights=wheel_heights,
        baseline_height=0.10,
        required_height=0.16,
        swing_start_x=-0.20,
        past_bar_x=float(
            getattr(env.unwrapped.cfg, "wave_sequential_past_bar_x_m", 0.14)
        ),
    )
    previous = getattr(env.unwrapped, "m1_crossing_progress_potential", None)
    if previous is None or previous.shape != score.shape:
        previous = score.new_zeros(score.shape)
    reset = (phase < 0) | (env.unwrapped.episode_length_buf <= 1)
    delta, next_potential = progress_potential_delta(
        current=score,
        previous=previous,
        reset=reset,
    )
    env.unwrapped.m1_crossing_progress_potential = next_potential.detach()
    return delta


def _m1_lateral_position_l2(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    local_y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return local_y.square()


def _m1_phase_aware_minimum_base_height(
    env,
    asset_cfg: SceneEntityCfg,
    normal_minimum_height: float,
    wave_minimum_height: float,
):
    """Keep the normal stance strict while allowing temporary axle-wave compression."""
    import torch

    asset = env.scene[asset_cfg.name]
    phase = getattr(env.unwrapped, "m1_sequential_crossing_phase", None)
    if phase is None:
        minimum = torch.full_like(
            asset.data.root_pos_w[:, 2], float(normal_minimum_height)
        )
    else:
        active_wave = (
            (phase == 0)
            | (phase == 1)
            | (phase == 2)
            | (phase == 3)
            | (phase == 4)
        )
        minimum = torch.where(
            active_wave,
            torch.full_like(asset.data.root_pos_w[:, 2], float(wave_minimum_height)),
            torch.full_like(asset.data.root_pos_w[:, 2], float(normal_minimum_height)),
        )
    return asset.data.root_pos_w[:, 2] < minimum


def _m1_phase_aware_bad_orientation(
    env,
    asset_cfg: SceneEntityCfg,
    normal_limit_angle: float,
    wave_limit_angle: float,
):
    """Allow transient pitch only while an axle is actively crossing."""
    import torch

    asset = env.scene[asset_cfg.name]
    phase = getattr(env.unwrapped, "m1_sequential_crossing_phase", None)
    angle = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)).abs()
    if phase is None:
        limit = torch.full_like(angle, float(normal_limit_angle))
    else:
        active_wave = (
            (phase == 0)
            | (phase == 1)
            | (phase == 2)
            | (phase == 3)
            | (phase == 4)
        )
        limit = torch.where(
            active_wave,
            torch.full_like(angle, float(wave_limit_angle)),
            torch.full_like(angle, float(normal_limit_angle)),
        )
    return angle > limit


def _m1_wheel_crossbar_contact(
    env,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    obstacle_center_x: float,
    obstacle_size_x: float,
    obstacle_height: float,
    wheel_radius: float,
    clearance_margin: float,
    contact_force_threshold: float,
    obstacle_center_y: float | None = None,
    obstacle_size_y: float | None = None,
):
    """Terminate fixed-course rollovers before a wheel can press onto the crossbar."""
    import torch

    robot = env.scene[asset_cfg.name]
    sensor = env.scene[contact_sensor_cfg.name]
    wheel_pos_local = (
        robot.data.body_pos_w[:, asset_cfg.body_ids]
        - env.scene.env_origins.unsqueeze(1)
    )
    wheel_force = torch.linalg.vector_norm(
        sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :], dim=-1
    )
    collision = wheel_crossbar_collision_mask(
        wheel_pos_local=wheel_pos_local,
        wheel_contact_force=wheel_force,
        obstacle_center_x=obstacle_center_x,
        obstacle_size_x=obstacle_size_x,
        obstacle_height=obstacle_height,
        wheel_radius=wheel_radius,
        clearance_margin=clearance_margin,
        contact_force_threshold=contact_force_threshold,
        obstacle_center_y=obstacle_center_y,
        obstacle_size_y=obstacle_size_y,
    )
    unwrapped = getattr(env, "unwrapped", env)
    unwrapped.m1_crossbar_collision_mask = collision.detach()
    unwrapped.m1_crossbar_collision_wheel_pos_local = wheel_pos_local.detach()
    unwrapped.m1_crossbar_collision_wheel_force = wheel_force.detach()
    return collision.any(dim=1)


def _m1_strict_sequential_crossing_success(
    env,
    asset_cfg: SceneEntityCfg,
    finish_x: float,
):
    """Require the wrapper's full four-wheel sequence before accepting finish."""
    import torch

    asset = env.scene[asset_cfg.name]
    phase = getattr(env.unwrapped, "m1_sequential_crossing_phase", None)
    if phase is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return strict_sequential_crossing_success(
        phase=phase,
        root_x=root_x,
        finish_x=finish_x,
        required_phase=6 if bool(
            getattr(env.unwrapped.cfg, "wave_axle_pair_crossing_reference", False)
        ) else 11,
    )


def _m1_prepared_leg_action_l2(env):
    """Penalize only leg actions that passed the semantic wave gate."""
    actions = getattr(env.unwrapped, "m1_prepared_leg_actions", None)
    if actions is None:
        return env.scene["robot"].data.root_pos_w[:, 0] * 0.0
    return actions.square().sum(dim=1)


@configclass
class M1PvcnnFlatSmallSceneCfg(TeacherElevationTrajectoryMpcSemanticSceneCfg):
    """Original semantic course with the M1 articulation and scanner mount."""

    robot = M1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    semantic_contact_small = _m1_semantic_global_contact_sensor(SEMANTIC_COURSE_SMALL_ROOT)
    semantic_height_scanner = SemanticGridRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK",
        offset=SemanticGridRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground", SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT],
        mesh_semantic_ids={
            "/World/ground": 0,
            SEMANTIC_COURSE_SMALL_ROOT: 1,
            SEMANTIC_COURSE_LARGE_ROOT: 2,
        },
        height_scan_offset=0.5,
        max_update_envs_per_call=512,
    )


@configclass
class M1PvcnnFlatSmallObservationsCfg:
    """M1 proprioception plus the original 16x16 elevation/semantic scan."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=isaac_mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyElevationSemanticMapCfg(ObsGroup):
        elevation_semantic_map = ObsTerm(
            func=downsampled_elevation_semantic_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-1.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    policy_elevation_semantic_map: PolicyElevationSemanticMapCfg = PolicyElevationSemanticMapCfg()


@configclass
class M1PvcnnCrossingActionsCfg(M1RollActionsCfg):
    """Release enough leg range to lift M1 wheels over 0.16 m obstacles."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(M1_LEG_JOINT_NAMES),
        scale=0.80,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
        preserve_order=True,
    )


@configclass
class M1PvcnnUnlockedCrossingActionsCfg(M1RollActionsCfg):
    """Pass policy leg actions through to the physical joint limits."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(M1_LEG_JOINT_NAMES),
        scale=0.80,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(M1_WHEEL_JOINT_NAMES),
        scale=1.0,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class M1PvcnnFlatSmallRewardsCfg(TeacherElevationTrajectoryMpcSemanticRewardsCfg):
    """Original tracking objective with selectors and stability terms for M1."""

    reference_foot_pos = None
    reference_contact = None
    semantic_contact_collision = None
    feet_air_time = None
    air_time_variance = None
    sequential_crossing_progress = None
    crossbar_contact_penalty = None
    lateral_position_l2 = None
    semantic_body_part_clearance = _m1_semantic_body_clearance_term()
    semantic_foot_over_clearance = _m1_semantic_wheel_over_term()
    semantic_pair_lift = RewTerm(
        func=_m1_semantic_pair_lift_reward,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(M1_FOOT_BODY_NAMES), preserve_order=True
            ),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
            "corridor_half_width_m": 0.25,
            "baseline_height_m": 0.096,
            "target_height_m": 0.26,
        },
    )
    crossing_progress = RewTerm(
        func=mdp.obstacle_progress,
        weight=5.0,
        params={
            "obstacle_distance": 0.40,
            "rear_axle_offset": 0.36,
            "clearance_margin": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    crossing_success = RewTerm(
        func=mdp.ObstacleMilestoneOnce,
        weight=500.0,
        params={"threshold": 0.78, "asset_cfg": SceneEntityCfg("robot")},
    )
    raw_leg_action_l2 = RewTerm(func=_m1_raw_leg_action_l2, weight=-2.0)
    base_height_recovery = RewTerm(
        func=mdp.base_height_recovery_l2,
        weight=-80.0,
        params={"target_height": 0.55, "relax_start_x": 0.05, "recovery_start_x": 0.70},
    )
    joint_posture_recovery = RewTerm(
        func=mdp.joint_posture_recovery_l2,
        weight=-15.0,
        params={
            "relax_start_x": 0.05,
            "recovery_start_x": 0.70,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(M1_LEG_JOINT_NAMES), preserve_order=True
            ),
        },
    )
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_LEG_JOINT_NAMES)),
            "stand_still_scale": 2.0,
            "velocity_threshold": 0.1,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(M1_FOOT_BODY_NAMES)),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES)),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=M1_BASE_BODY_NAME),
        },
    )
    forward_velocity = RewTerm(func=mdp.forward_velocity, weight=50.0, params={"max_velocity": 0.25})
    wheel_action_match = None
    wheel_velocity_sync = _m1_wheel_velocity_sync_term()
    termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-1000.0)


@configclass
class M1PvcnnFlatSmallAvoidanceEnvCfg(TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg):
    """M1 stage two using the original semantic small-obstacle environment."""

    scene: M1PvcnnFlatSmallSceneCfg = M1PvcnnFlatSmallSceneCfg(
        num_envs=64, env_spacing=2.5, replicate_physics=True
    )
    observations: M1PvcnnFlatSmallObservationsCfg = M1PvcnnFlatSmallObservationsCfg()
    actions: M1PvcnnCrossingActionsCfg = M1PvcnnCrossingActionsCfg()
    rewards: M1PvcnnFlatSmallRewardsCfg = M1PvcnnFlatSmallRewardsCfg()
    planner_owned_reference_cache: bool = False
    use_batched_reference_trajectory: bool = False
    planner_backend: str = "none"
    wave_fixed_forward_wheels: bool = True
    wave_leg_action_limit: float = 0.60
    wave_reference_actions: bool = True
    wave_reference_raw_amplitude: float = 0.55
    wave_reference_knee_ratio: float = 1.00
    wave_rear_amplitude_scale: float = 1.40
    wave_front_overlap_scale: float = 1.00
    wave_front_support_ratio: float = 0.67
    wave_rear_support_ratio: float = 0.0
    wave_reference_frequency: float = 0.5
    wave_spatial_reference: bool = True
    wave_reset_phase_on_obstacle: bool = True
    wave_minimum_gate_duration_s: float = 2.0
    wave_left_right_symmetric: bool = True
    wave_lock_abduction: bool = True
    wave_front_wheel_action: float = 0.50
    wave_rear_wheel_action: float = 0.50
    wave_obstacle_wheel_action: float = 0.25
    wave_obstacle_sync_max_correction: float = 0.10
    wave_wheel_residual_scale: float = 0.05
    wave_sync_actual_wheel_velocity: bool = True
    wave_wheel_sync_gain: float = 0.50
    wave_wheel_sync_integral_gain: float = 1.0
    wave_wheel_sync_integral_limit: float = 0.50
    wave_wheel_sync_max_correction: float = 0.50
    wave_semantic_obstacle_gating: bool = True
    wave_semantic_scanner_name: str = "semantic_height_scanner"
    wave_semantic_gate_min_x: float = -0.35
    wave_semantic_gate_max_x: float = 0.80
    wave_semantic_gate_half_width: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_flat_small_avoidance"
        self.planner_owned_reference_cache = False
        self.use_batched_reference_trajectory = False
        self.planner_backend = "none"
        self.episode_length_s = 120.0
        self.events.add_base_mass.params["asset_cfg"].body_names = M1_BASE_BODY_NAME
        self.events.base_external_force_torque.params["asset_cfg"].body_names = M1_BASE_BODY_NAME
        self.terminations.base_contact.params["sensor_cfg"].body_names = M1_BASE_BODY_NAME
        self.terminations.bad_orientation.params["limit_angle"] = 1.10
        self.commands.base_velocity.vx_abs_range = (0.03, 0.08)
        self.commands.base_velocity.vy_abs_range = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.03, 0.08)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # Keep all ten curriculum rows while reducing duplicate terrain columns.
        self.scene.terrain.terrain_generator.num_cols = 2
        self.scene.terrain.semantic_course_mandatory_small_xy = (0.40, 0.0)
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.push_robot = None
        # The inherited clearance terms currently resolve Go2-style lowercase body names.
        self.rewards.reference_foot_pos = None
        self.rewards.reference_contact = None
        self.rewards.semantic_contact_collision = None
        self.rewards.semantic_body_part_clearance = _m1_semantic_body_clearance_term()
        self.rewards.semantic_foot_over_clearance = _m1_semantic_wheel_over_term()
        self.rewards.wheel_velocity_sync = None
        self.scene.semantic_contact_small = _m1_semantic_global_contact_sensor(
            SEMANTIC_COURSE_SMALL_ROOT
        )
        self.scene.semantic_height_scanner.update_period = self.sim.dt * self.decimation


@configclass
class M1PvcnnFlatSmallAvoidanceEnvCfg_PLAY(M1PvcnnFlatSmallAvoidanceEnvCfg):
    """Viewer config for the M1 semantic small-obstacle policy."""

    scene: M1PvcnnFlatSmallSceneCfg = M1PvcnnFlatSmallSceneCfg(
        num_envs=8, env_spacing=2.5, replicate_physics=True
    )

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = None
        self.curriculum.terrain_levels = None
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False
        self.observations.policy_elevation_semantic_map.enable_corruption = False


@configclass
class M1PvcnnCrossing60mmEnvCfg(M1PvcnnFlatSmallAvoidanceEnvCfg):
    """First crossing curriculum stage with 60 mm semantic obstacles."""

    wave_reference_raw_amplitude: float = 0.30
    wave_rear_amplitude_scale: float = 1.50
    wave_front_overlap_scale: float = 1.00
    wave_front_support_ratio: float = 0.40
    wave_leg_action_limit: float = 0.08
    wave_policy_leg_residual_limit: float = 0.0
    wave_obstacle_wheel_action: float = 0.50
    wave_obstacle_front_wheel_action: float = 0.20
    wave_obstacle_rear_wheel_action: float = 0.95
    wave_disable_obstacle_after_root_x: float = 1.15
    wave_obstacle_sync_max_correction: float = 0.30
    wave_wheel_equalize_gain: float = 2.0
    wave_wheel_equalize_max_correction: float = 0.50
    wave_lock_left_right_wheel_targets: bool = True
    wave_lock_all_wheel_targets: bool = True
    wave_wheel_action_signs: tuple[float, float, float, float] | None = None
    base_height_target: float = 0.57
    base_height_recovery_start_x: float = 1.10
    base_height_recovery_tolerance: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing_60mm"
        self.scene.terrain.semantic_course_mandatory_small_xy = (0.65, 0.0)
        self.scene.terrain.semantic_course_scale_profile_overrides = {"small": (0.60, 0.06)}
        self.scene.terrain.semantic_course_cuboid_size_overrides = {
            "small": (0.06, 0.60, 0.06)
        }
        self.semantic_obstacle_curriculum.plane_counts = tuple(
            SemanticObstacleCount(small=0, large=0) for _ in range(10)
        )
        self.rewards.semantic_pair_lift.params["target_height_m"] = 0.18
        self.rewards.base_height_recovery.params["target_height"] = self.base_height_target
        self.rewards.base_height_recovery.params["recovery_start_x"] = (
            self.base_height_recovery_start_x
        )
        self.rewards.joint_posture_recovery.params["recovery_start_x"] = (
            self.base_height_recovery_start_x
        )
        self.rewards.wheel_velocity_sync = _m1_wheel_velocity_sync_term(weight=-80.0)
        self.rewards.crossing_progress.params["obstacle_distance"] = 0.65
        self.rewards.crossing_progress.params["rear_axle_offset"] = 0.38
        self.rewards.crossing_success.params["threshold"] = 1.45
        self.rewards.flat_orientation_l2.weight = -20.0
        self.terminations.crossing_success = DoneTerm(
            func=mdp.root_x_above,
            time_out=True,
            params={"threshold": 1.50, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.crossbar_contact = DoneTerm(
            func=_m1_wheel_crossbar_contact,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=list(M1_FOOT_BODY_NAMES), preserve_order=True
                ),
                "contact_sensor_cfg": SceneEntityCfg(
                    "semantic_contact_small",
                    body_names=list(M1_FOOT_BODY_NAMES),
                    preserve_order=True,
                ),
                "obstacle_center_x": 0.65,
                "obstacle_size_x": 0.06,
                "obstacle_center_y": 0.0,
                "obstacle_size_y": 0.60,
                "obstacle_height": 0.06,
                "wheel_radius": 0.095,
                "clearance_margin": 0.005,
                "contact_force_threshold": 1.0,
            },
        )


@configclass
class M1PvcnnCrossing60mmPlayEnvCfg(M1PvcnnCrossing60mmEnvCfg):
    """Deterministic viewer course with one 60 mm obstacle directly ahead."""

    scene: M1PvcnnFlatSmallSceneCfg = M1PvcnnFlatSmallSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    seed: int = 20260711

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing_60mm_play"
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.semantic_course_mandatory_small_xy = (0.65, 0.0)
        self.curriculum.terrain_levels = None
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False
        self.observations.policy_elevation_semantic_map.enable_corruption = False


@configclass
class M1PvcnnCrossing60mmUnlockedEnvCfg(M1PvcnnCrossing60mmEnvCfg):
    """Train full leg actions during wave while locking the nominal stance elsewhere."""

    actions: M1PvcnnUnlockedCrossingActionsCfg = M1PvcnnUnlockedCrossingActionsCfg()
    wave_leg_action_limit: float | None = None
    wave_policy_leg_residual_limit: float | None = None
    wave_reference_actions: bool = False
    wave_unclipped_policy_legs: bool = True
    wave_obstacle_front_wheel_action: float = 0.50
    wave_obstacle_rear_wheel_action: float = 0.50

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing_60mm_unlocked"
        self.rewards.raw_leg_action_l2 = RewTerm(
            func=_m1_prepared_leg_action_l2, weight=-2.0
        )
        self.rewards.semantic_pair_lift.weight = 20.0
        self.terminations.bad_orientation.params["limit_angle"] = 0.60


@configclass
class M1PvcnnCrossing60mmGuidedEnvCfg(M1PvcnnCrossing60mmUnlockedEnvCfg):
    """Bootstrap unrestricted policy residuals with a spatial obstacle wave teacher."""

    wave_reference_actions: bool = True
    wave_reference_raw_amplitude: float = 0.80
    wave_reference_knee_ratio: float = -2.0
    wave_reference_smoothing_alpha: float = 1.0
    wave_reference_pulse_ramp_s: float = 0.02
    wave_reference_pulse_hold_s: float = 1.96
    wave_reference_time_offset_s: float = 0.02
    wave_reference_constant_phase_s: float = 0.5
    wave_spatial_reference: bool = False
    wave_single_cycle_duration_s: float = 2.0
    wave_rear_amplitude_scale: float = 0.0
    wave_front_support_ratio: float = 0.0
    wave_rear_support_ratio: float = 0.50
    acceptance_max_tilt_rad: float = 0.45
    acceptance_min_front_wheel_height_m: float = 0.13
    acceptance_min_rear_wheel_height_m: float = 0.14
    wave_max_action_delta_acceptance: float = 2.0
    wave_obstacle_front_wheel_action: float = 6.4
    wave_obstacle_rear_wheel_action: float = 8.0
    wave_phase_wheel_assist: float = 0.0
    wave_lock_left_right_wheel_targets: bool = False
    wave_lateral_steering_gain: float = 2.0
    wave_yaw_damping_gain: float = 0.5
    wave_steering_max_correction: float = 0.5
    wave_wheel_equalize_max_correction: float = 0.80

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing_60mm_guided"
        self.rewards.raw_leg_action_l2 = RewTerm(
            func=_m1_raw_leg_action_l2, weight=-2.0
        )


@configclass
class M1PvcnnCrossing60mmGuidedFixedEnvCfg(M1PvcnnCrossing60mmGuidedEnvCfg):
    """Fixed front-obstacle course used to collect clean teacher trajectories."""

    scene: M1PvcnnFlatSmallSceneCfg = M1PvcnnFlatSmallSceneCfg(
        num_envs=32, env_spacing=2.5, replicate_physics=True
    )
    seed: int = 20260711

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing_60mm_guided_fixed"
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.semantic_course_mandatory_small_xy = (0.65, 0.0)
        self.curriculum.terrain_levels = None
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False
        self.observations.policy_elevation_semantic_map.enable_corruption = False


@configclass
class M1PvcnnCrossing60mmDistilledPlayEnvCfg(M1PvcnnCrossing60mmGuidedFixedEnvCfg):
    """Fixed front-obstacle course for an autonomous distilled wave policy."""

    scene: M1PvcnnFlatSmallSceneCfg = M1PvcnnFlatSmallSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    seed: int = 20260711
    wave_reference_actions: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing_60mm_distilled_play"
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.episode_length_s = 20.0
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.semantic_course_mandatory_small_xy = (0.65, 0.0)
        self.curriculum.terrain_levels = None
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False
        self.observations.policy_elevation_semantic_map.enable_corruption = False


@configclass
class M1PvcnnCrossing60mmContactFreeTrainEnvCfg(M1PvcnnCrossing60mmGuidedFixedEnvCfg):
    """Learn wheel-by-wheel clearance with legs locked outside axle windows."""

    scene: M1PvcnnFlatSmallSceneCfg = M1PvcnnFlatSmallSceneCfg(
        num_envs=32, env_spacing=2.5, replicate_physics=True
    )
    wave_reference_actions: bool = False
    wave_spatial_reference: bool = True
    wave_gate_from_spatial_reference: bool = False
    wave_sequential_crossing_reference: bool = True
    wave_axle_pair_crossing_reference: bool = False
    wave_right_track_only: bool = True
    wave_task_space_ik: bool = True
    wave_task_space_support_only: bool = True
    wave_task_space_active_swing_xz_only: bool = True
    wave_task_space_lift_delta_m: float = 0.22
    wave_task_space_rear_lift_delta_m: float = 0.20
    wave_task_space_rear_restore_forward_offset_m: float = 0.12
    wave_task_space_action_scale: float = 0.80
    wave_task_space_ik_damping: float = 0.05
    wave_task_space_max_joint_step: float = 0.25
    wave_task_space_base_height_gain: float = 1.0
    wave_task_space_base_height_max_m: float = 0.10
    wave_task_space_lateral_body_shift_m: float = 0.06
    wave_task_space_longitudinal_body_shift_m: float = 0.04
    wave_task_space_balance_supports: bool = True
    wave_task_space_lateral_recovery_gain: float = 0.0
    wave_task_space_lateral_recovery_max_m: float = 0.0
    wave_task_space_pair_support_extension_m: float = 0.16
    wave_task_space_pair_body_shift_x_m: float = 0.08
    wave_task_space_balance_steps: int = 30
    wave_task_space_lift_ramp_steps: int = 10
    wave_task_space_stop_during_wave: bool = False
    wave_task_space_swing_with_body: bool = True
    wave_task_space_stabilize_supports: bool = True
    wave_task_space_swing_ramp_steps: int = 20
    wave_axle_pair_ramp_steps: int = 10
    wave_axle_pair_support_steps: int = 20
    wave_axle_pair_restore_steps: int = 20
    wave_axle_pair_front_start_x_m: float = -0.28
    wave_sequential_past_bar_x_m: float = 0.15
    wave_sequential_swing_steps: int = 50
    wave_sequential_min_lift_steps: int = 5
    wave_sequential_ramp_steps: int = 10
    wave_sequential_restore_steps: int = 20
    wave_sequential_clearance_target_height_m: float = 0.20
    wave_sequential_balance_steps: int = 0
    wave_sequential_keep_drive_during_wave: bool = True
    wave_sequential_support_extension: float = 0.0
    wave_sequential_opposite_abduction: float = -0.10
    wave_sequential_front_hip_action: float = -0.30
    wave_sequential_front_knee_action: float = -0.60
    wave_sequential_rear_hip_action: float = 0.30
    wave_sequential_rear_knee_action: float = 0.60
    wave_sequential_support_residual_scale: float = 0.0
    wave_sequential_crossing_residual_scale: float = 0.0
    wave_sequential_support_abduction_residual_scale: float = 0.0
    wave_sequential_front_start_x_m: float = -0.45
    wave_fixed_obstacle_center_x_m: float | None = 0.85
    wave_sequential_front_restore_x: float = 0.30
    wave_sequential_rear_start_x: float = -0.10
    wave_sequential_rear_restore_x: float = -0.55
    wave_front_lift_window: tuple[float, float, float, float] = (0.80, 0.70, 0.15, 0.05)
    wave_rear_lift_window: tuple[float, float, float, float] = (0.05, -0.05, -0.50, -0.60)
    wave_left_right_symmetric: bool = False
    wave_lock_abduction: bool = False
    wave_leg_action_limit: float | None = None
    wave_policy_leg_residual_limit: float | None = 1.0
    wave_unclipped_policy_legs: bool = True
    wave_front_wheel_action: float = 1.0
    wave_rear_wheel_action: float = 1.0
    wave_obstacle_front_wheel_action: float = 6.0
    wave_obstacle_rear_wheel_action: float = 6.0
    wave_sequential_swing_wheel_action: float = 7.0
    wave_rear_wheel_velocity_feedforward: float = 0.40
    wave_hold_wheels_until_axle_clear: bool = False
    wave_clearance_minimum_hold_s: float = 0.15
    wave_axle_clearance_height_m: float = 0.16
    wave_axle_switch_obstacle_x: float = 0.05
    wave_forward_only_wheels: bool = True
    wave_wheel_action_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    wave_lock_left_right_wheel_targets: bool = False
    wave_lock_all_wheel_targets: bool = False
    wave_wheel_equalize_gain: float = 3.0
    wave_wheel_equalize_max_correction: float = 1.0
    wave_wheel_equalize_to_slowest: bool = True
    wave_wheel_sync_gain: float = 0.0
    wave_wheel_sync_integral_gain: float = 0.0
    wave_wheel_sync_max_correction: float = 1.0
    wave_lateral_steering_gain: float = 0.0
    wave_yaw_damping_gain: float = 0.0
    acceptance_require_active_wheel_clearance: bool = True
    acceptance_wheel_radius_m: float = 0.095
    acceptance_wheel_clearance_margin_m: float = 0.005
    acceptance_clearance_contact_force_limit_n: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing60_contact_free"
        self.scene.terrain.semantic_course_mandatory_small_xy = (0.85, -0.20)
        self.scene.terrain.semantic_course_scale_profile_overrides = {
            "small": (0.16, 0.06)
        }
        self.scene.terrain.semantic_course_cuboid_size_overrides = {
            "small": (0.06, 0.16, 0.06)
        }
        zero_counts = tuple(
            SemanticObstacleCount(small=0, large=0) for _ in range(10)
        )
        self.semantic_obstacle_curriculum.plane_counts = zero_counts
        self.semantic_obstacle_curriculum.non_plane_counts = zero_counts
        self.scene.terrain.semantic_obstacle_curriculum = (
            self.semantic_obstacle_curriculum
        )
        self.rewards.crossing_progress.params["obstacle_distance"] = 0.85
        crossbar_params = dict(self.terminations.crossbar_contact.params)
        crossbar_params["obstacle_center_x"] = 0.85
        crossbar_params["obstacle_center_y"] = -0.20
        crossbar_params["obstacle_size_y"] = 0.16
        self.terminations.crossbar_contact.params = crossbar_params
        self.rewards.semantic_pair_lift = None
        self.rewards.semantic_foot_over_clearance.weight = 10.0
        self.rewards.raw_leg_action_l2.weight = -0.01
        self.rewards.action_rate.weight = -0.01
        self.rewards.wheel_velocity_sync = _m1_wheel_velocity_sync_term(weight=-2.0)
        self.rewards.flat_orientation_l2.weight = -30.0
        self.rewards.base_height_recovery.weight = -1000.0
        self.rewards.base_height_recovery.params["relax_start_x"] = 10.0
        self.rewards.sequential_crossing_progress = RewTerm(
            func=_m1_sequential_crossing_progress,
            weight=40.0,
        )
        self.rewards.lateral_position_l2 = RewTerm(
            func=_m1_lateral_position_l2,
            weight=-300.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.termination_penalty.weight = -3000.0
        self.terminations.minimum_base_height = DoneTerm(
            func=_m1_phase_aware_minimum_base_height,
            time_out=False,
            params={
                "normal_minimum_height": 0.35,
                "wave_minimum_height": 0.30,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.terminations.bad_orientation = DoneTerm(
            func=_m1_phase_aware_bad_orientation,
            time_out=False,
            params={
                "normal_limit_angle": 0.60,
                "wave_limit_angle": 0.90,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class M1PvcnnCrossing60mmContactFreePlayEnvCfg(
    M1PvcnnCrossing60mmContactFreeTrainEnvCfg
):
    """Deterministic viewer course for contact-free crossing checkpoints."""

    scene: M1PvcnnFlatSmallSceneCfg = M1PvcnnFlatSmallSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing60_contact_free_play"
        self.episode_length_s = 20.0


@configclass
class M1PvcnnCrossing60mmPolicyPlayEnvCfg(
    M1PvcnnCrossing60mmContactFreePlayEnvCfg
):
    """Strict crossing course where policy legs replace the wave teacher."""

    wave_sequential_policy_control: bool = False
    wave_gate_from_policy_action: bool = True
    wave_policy_gate_action_index: int = 15
    wave_policy_gate_threshold: float = -0.03
    wave_policy_gate_weight: float = 1.0
    wave_policy_gate_minimum_root_x_m: float = 0.120
    wave_policy_gate_fallback_root_x_m: float = 0.125

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing60_policy_play"


@configclass
class M1PvcnnCrossing60mmPairCurriculumEnvCfg(
    M1PvcnnCrossing60mmContactFreeTrainEnvCfg
):
    """Pretrain the complete axle sequence before strict contact-free finetuning."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing60_pair_curriculum"
        crossbar_params = dict(self.terminations.crossbar_contact.params)
        self.terminations.crossbar_contact = None
        self.rewards.crossbar_contact_penalty = RewTerm(
            func=_m1_wheel_crossbar_contact,
            weight=-10.0,
            params=crossbar_params,
        )
        self.rewards.sequential_crossing_progress.weight = 200.0
        self.rewards.crossing_progress = None
        self.rewards.crossing_success = RewTerm(
            func=_m1_strict_sequential_crossing_success,
            weight=500.0,
            params={"finish_x": 1.50, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.crossing_success = DoneTerm(
            func=_m1_strict_sequential_crossing_success,
            time_out=True,
            params={"finish_x": 1.50, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.wave_axle_clearance_height_m = 0.16
        self.wave_sequential_past_bar_x_m = 0.15
        self.wave_pair_curriculum_swing_timeout_steps = None
        bad_orientation_params = dict(self.terminations.bad_orientation.params)
        bad_orientation_params["wave_limit_angle"] = 0.65
        self.terminations.bad_orientation.params = bad_orientation_params


@configclass
class M1PvcnnCrossing100mmEnvCfg(M1PvcnnFlatSmallAvoidanceEnvCfg):
    """Second crossing curriculum stage with 100 mm semantic obstacles."""

    wave_reference_raw_amplitude: float = 0.40
    wave_rear_amplitude_scale: float = 1.80
    wave_front_overlap_scale: float = 1.00
    wave_leg_action_limit: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "m1_pvcnn_crossing_100mm"
        self.scene.terrain.semantic_course_scale_profile_overrides = {"small": (0.12, 0.10)}
        self.semantic_obstacle_curriculum.plane_counts = tuple(
            SemanticObstacleCount(small=2, large=0) for _ in range(10)
        )
        self.rewards.semantic_pair_lift.params["target_height_m"] = 0.22
        self.terminations.crossing_success = DoneTerm(
            func=mdp.root_x_above,
            time_out=True,
            params={"threshold": 0.78, "asset_cfg": SceneEntityCfg("robot")},
        )
