"""RSL-RL PPO configuration for M1 + Panda Teacher balance stages."""

from __future__ import annotations


def get_m1_panda_teacher_train_cfg() -> dict:
    """Return a fresh small-MLP PPO config for A0/A1 Teacher training."""
    return {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "clip_min_std": 0.001,
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
