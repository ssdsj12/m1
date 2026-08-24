"""RSL-RL PPO configuration for coordinated M1 + Panda training."""

from __future__ import annotations


def get_m1_panda_coordinated_train_cfg() -> dict:
    """Return a fresh PPO config matched to the 200 Hz coordinated task."""
    return {
        "num_steps_per_env": 256,
        "save_interval": 25,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-4,
            "min_learning_rate": 1.0e-6,
            "max_learning_rate": 3.0e-4,
            "clip_param": 0.2,
            "gamma": 0.9995,
            "lam": 0.995,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "clip_min_std": 0.005,
            "clip_max_std": 0.05,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.01,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
        },
    }


__all__ = ["get_m1_panda_coordinated_train_cfg"]
