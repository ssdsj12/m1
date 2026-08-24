def _get_cfg():
    from agent import get_m1_panda_coordinated_train_cfg

    return get_m1_panda_coordinated_train_cfg()


def test_coordinated_cfg_freezes_200_hz_time_horizon_and_adaptive_ppo():
    cfg = _get_cfg()

    assert cfg["num_steps_per_env"] == 256
    assert cfg["save_interval"] == 25
    assert cfg["empirical_normalization"] is False
    assert cfg["algorithm"] == {
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
    }
    assert cfg["policy"] == {
        "class_name": "ActorCritic",
        "init_noise_std": 0.01,
        "noise_std_type": "scalar",
        "actor_hidden_dims": [256, 128],
        "critic_hidden_dims": [256, 128],
        "activation": "elu",
    }


def test_coordinated_cfg_returns_independent_objects():
    left = _get_cfg()
    left["algorithm"]["gamma"] = 0.0
    left["policy"]["actor_hidden_dims"].append(64)

    right = _get_cfg()

    assert right["algorithm"]["gamma"] == 0.9995
    assert right["policy"]["actor_hidden_dims"] == [256, 128]
