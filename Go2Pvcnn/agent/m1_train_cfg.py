"""RSL-RL training configuration for M1 locomotion."""


def get_m1_train_cfg() -> dict:
    """Return a small MLP PPO config for `Isaac-M1-Walk-v0`."""

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
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.01,
            "noise_std_type": "log",
            "state_dependent_std": False,
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
        },
    }
