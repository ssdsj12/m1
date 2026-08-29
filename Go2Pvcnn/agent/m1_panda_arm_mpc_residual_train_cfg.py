"""PPO configuration for the 103-observation, 8-action Arm-MPC residual task."""

from __future__ import annotations


MAX_RESIDUAL_TRAINING_ITERATIONS = 3000


def get_m1_panda_arm_mpc_residual_train_cfg() -> dict:
    """Return a fresh stability-first PPO configuration for a 200 Hz task."""
    return {
        "num_steps_per_env": 256,
        "save_interval": 25,
        "max_iterations": MAX_RESIDUAL_TRAINING_ITERATIONS,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-5,
            "min_learning_rate": 1.0e-6,
            "max_learning_rate": 1.0e-4,
            "clip_param": 0.2,
            "gamma": 0.9995,
            "lam": 0.995,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "clip_min_std": 0.005,
            "clip_max_std": 0.02,
            "max_grad_norm": 0.5,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "kl_abort_threshold": 0.015,
        },
        "policy": {
            "class_name": (
                "go2_pvcnn.control.m1_panda_coordination."
                "residual_actor_critic.ResidualActorCritic"
            ),
            "init_noise_std": 0.005,
            "noise_std_type": "scalar",
        },
    }


__all__ = [
    "MAX_RESIDUAL_TRAINING_ITERATIONS",
    "get_m1_panda_arm_mpc_residual_train_cfg",
]
