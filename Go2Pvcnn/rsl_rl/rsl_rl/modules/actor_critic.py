#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        active_action_mask=None,
        zero_actor_output=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        if isinstance(active_action_mask, (str, bytes)):
            raise TypeError("active_action_mask must be a numeric sequence")
        if active_action_mask is None:
            action_mask = torch.ones(num_actions, dtype=torch.bool)
        else:
            action_mask_values = torch.as_tensor(active_action_mask)
            if action_mask_values.ndim != 1 or action_mask_values.numel() != num_actions:
                raise ValueError("active_action_mask must contain num_actions entries")
            if not torch.all((action_mask_values == 0) | (action_mask_values == 1)):
                raise ValueError("active_action_mask entries must be 0 or 1")
            action_mask = action_mask_values.to(dtype=torch.bool)
            if not bool(action_mask.any()):
                raise ValueError("active_action_mask must contain at least one active action")
        # The mask is a runtime policy contract rather than checkpoint state.  A
        # non-persistent buffer keeps old no-mask checkpoints load-compatible.
        self.register_buffer("active_action_mask", action_mask, persistent=False)
        final_actor_layer = next(
            module for module in reversed(self.actor) if isinstance(module, nn.Linear)
        )
        with torch.no_grad():
            if zero_actor_output:
                final_actor_layer.weight.zero_()
                final_actor_layer.bias.zero_()
            else:
                final_actor_layer.weight[~action_mask] = 0.0
                final_actor_layer.bias[~action_mask] = 0.0

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise.  Keep scalar mode compatible with legacy checkpoints
        # whose state dictionaries contain a direct physical ``std`` tensor.
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                "noise_std_type must be 'scalar' or 'log', "
                f"got {noise_std_type!r}"
            )
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy()[:, self.active_action_mask].sum(dim=-1)

    @property
    def noise_parameter(self):
        if self.noise_std_type == "scalar":
            return self.std
        return self.log_std

    @property
    def effective_action_std(self):
        if self.noise_std_type == "scalar":
            return self.std
        return torch.exp(self.log_std)

    def update_distribution(self, observations):
        mean = self.actor(observations) * self.active_action_mask.to(
            device=observations.device, dtype=observations.dtype
        )
        std = self.effective_action_std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample() * self.active_action_mask.to(
            device=observations.device, dtype=observations.dtype
        )

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions)[:, self.active_action_mask].sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations) * self.active_action_mask.to(
            device=observations.device, dtype=observations.dtype
        )
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    @torch.no_grad()
    def clip_std(self, min=None, max=None):
        if self.noise_std_type == "scalar":
            self.std.copy_(self.std.clip(min=min, max=max))
            return
        log_min = None if min is None else math.log(float(min))
        log_max = None if max is None else math.log(float(max))
        self.log_std.copy_(self.log_std.clip(min=log_min, max=log_max))


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
