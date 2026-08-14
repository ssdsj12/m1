"""Register M1 environments with Gymnasium."""

import gymnasium as gym

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmEnvCfg", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmPlayEnvCfg", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-Unlocked-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmUnlockedEnvCfg", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-Guided-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmGuidedEnvCfg", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-Guided-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmGuidedFixedEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-Distilled-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmDistilledPlayEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmContactFreeTrainEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmContactFreePlayEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-Pair-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmPairCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-60mm-Policy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmPolicyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Pvcnn-Crossing-100mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing100mmEnvCfg", "rsl_rl_cfg_entry_point": None},
)


gym.register(
    id="Isaac-M1-Pvcnn-Flat-Small-Avoidance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnFlatSmallAvoidanceEnvCfg", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Pvcnn-Flat-Small-Avoidance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnFlatSmallAvoidanceEnvCfg_PLAY", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Smoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_smoke_env_cfg:M1SmokeEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Panda-Smoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_panda_smoke_env_cfg:M1PandaSmokeEnvCfg", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Panda-Teacher-A0-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_panda_teacher_env_cfg:M1PandaTeacherA0EnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Panda-Teacher-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_panda_teacher_env_cfg:M1PandaTeacherA1EnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Small-Obstacle-5mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_small_obstacle_env_cfg:M1SmallObstacle5mmEnvCfg", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Small-Obstacle-10mm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "go2_pvcnn.tasks.m1_small_obstacle_env_cfg:M1SmallObstacle10mmEnvCfg", "rsl_rl_cfg_entry_point": None},
)

gym.register(
    id="Isaac-M1-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_walk_env_cfg:M1WalkEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Roll-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_roll_env_cfg:M1RollEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Wave-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_wave_env_cfg:M1WaveFlatEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-M1-Small-Obstacle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_small_obstacle_env_cfg:M1SmallObstacleEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    },
)

print("[go2_pvcnn] Registered M1 environments:")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-Play-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-Unlocked-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-Guided-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-Guided-Fixed-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-Distilled-Play-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Train-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Play-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-Pair-Curriculum-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-60mm-Policy-Play-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Crossing-100mm-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Flat-Small-Avoidance-v0")
print("[go2_pvcnn]   - Isaac-M1-Pvcnn-Flat-Small-Avoidance-Play-v0")
print("[go2_pvcnn]   - Isaac-M1-Smoke-v0")
print("[go2_pvcnn]   - Isaac-M1-Panda-Smoke-v0")
print("[go2_pvcnn]   - Isaac-M1-Panda-Teacher-A0-v0")
print("[go2_pvcnn]   - Isaac-M1-Panda-Teacher-A1-v0")
print("[go2_pvcnn]   - Isaac-M1-Walk-v0")
print("[go2_pvcnn]   - Isaac-M1-Roll-v0")
print("[go2_pvcnn]   - Isaac-M1-Wave-Flat-v0")
print("[go2_pvcnn]   - Isaac-M1-Small-Obstacle-v0")
print("[go2_pvcnn]   - Isaac-M1-Small-Obstacle-5mm-v0")
print("[go2_pvcnn]   - Isaac-M1-Small-Obstacle-10mm-v0")
