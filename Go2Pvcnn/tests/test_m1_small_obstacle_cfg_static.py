from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TERRAIN_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_small_obstacle_terrain.py"
ENV_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_small_obstacle_env_cfg.py"
REGISTER_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "register_m1_envs.py"
REWARDS_FILE = REPO_ROOT / "go2_pvcnn" / "mdp" / "rewards.py"


def test_transverse_bar_terrain_has_real_crossing_geometry():
    source = TERRAIN_FILE.read_text()

    assert "class M1TransverseBarTerrainCfg(SubTerrainBaseCfg)" in source
    assert "obstacle_height_range: tuple[float, float] = (0.001, 0.03)" in source
    assert "obstacle_distance: float = 0.55" in source
    assert "obstacle_depth: float = 0.04" in source
    assert "obstacle_ramp_length: float = 0.08" in source
    assert "trimesh.creation.box" in source
    assert "trimesh.Trimesh" in source


def test_small_obstacle_env_uses_height_scan_teacher_and_long_episode():
    source = ENV_FILE.read_text()

    assert "class M1SmallObstacleEnvCfg(M1WaveFlatEnvCfg)" in source
    assert "return AssetBaseCfg" in source
    assert "obstacle = make_obstacle_cfg(0.001)" in source
    assert "obstacle = make_obstacle_cfg(0.005)" in source
    assert "obstacle = make_obstacle_cfg(0.010)" in source
    assert "class M1SmallObstacle5mmEnvCfg(M1SmallObstacleEnvCfg)" in source
    assert "class M1SmallObstacle10mmEnvCfg(M1SmallObstacleEnvCfg)" in source
    assert "height_scanner = MultiMeshRayCasterCfg" in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK"' in source
    assert 'prim_expr="/World/envs/env_.*/Obstacle"' in source
    assert "patterns.GridPatternCfg(resolution=0.1, size=(1.5, 1.5))" in source
    assert "height_scan = ObsTerm" in source
    assert "func=isaac_mdp.height_scan" in source
    assert "obstacle_passed = RewTerm" in source
    assert "obstacle_progress = RewTerm" in source
    assert "forward_velocity = RewTerm" in source
    assert "front_wheel_lift = RewTerm" in source
    assert 'target_height": 0.60' in source
    assert "base_contact = RewTerm" in source
    assert "weight=-10.0" in source
    assert "termination_penalty = RewTerm" in source
    assert "front_wheels_over = RewTerm" in source
    assert "front_axle_clear = RewTerm" in source
    assert "base_over = RewTerm" in source
    assert "rear_axle_approach = RewTerm" in source
    assert "rear_axle_near_clear = RewTerm" in source
    assert '"threshold": 0.95' in source
    assert "wave_leg_action_limit: float = 0.40" in source
    assert "wave_left_right_symmetric: bool = True" in source
    assert "wave_lock_abduction: bool = True" in source
    assert "wave_reference_raw_amplitude: float = 0.20" in source
    assert "wave_obstacle_wheel_boost: float = 0.08" in source
    assert "wave_obstacle_boost_start_x: float = 0.05" in source
    assert "self.episode_length_s = 30.0" in source
    assert "self.terminations.base_contact = None" in source


def test_obstacle_passed_reward_requires_rear_axle_clearance():
    source = REWARDS_FILE.read_text()

    assert "def obstacle_passed" in source
    assert "def obstacle_progress" in source
    assert "def forward_velocity" in source
    assert "def obstacle_front_wheel_lift" in source
    assert "class ObstacleMilestoneOnce(ManagerTermBase)" in source
    assert "self.reached" in source
    assert "newly_reached" in source
    assert '"std": 0.02' in ENV_FILE.read_text()
    assert "obstacle_distance" in source
    assert "rear_axle_offset" in source
    assert "asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]" in source


def test_small_obstacle_task_is_registered():
    source = REGISTER_FILE.read_text()

    assert "M1SmallObstacleEnvCfg" in source
    assert 'id="Isaac-M1-Small-Obstacle-v0"' in source
    assert 'id="Isaac-M1-Small-Obstacle-5mm-v0"' in source
    assert 'id="Isaac-M1-Small-Obstacle-10mm-v0"' in source
