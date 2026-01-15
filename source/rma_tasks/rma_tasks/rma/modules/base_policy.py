# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.modules import ActorCritic

class BasePolicy(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        z_size=8,
        encoder_obs_normalization=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        encoder_hidden_dims=[256, 256,256],
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "BasePolicy.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # policy obs: current state (30) + prev action (12) --> should be 42 based on paper
        self.obs_groups = obs_groups
        num_actor_obs = 0 
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        
        # critic
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        
        # env: priv_obs 
        num_env_obs = 0 # large amount of priv obs expected
        for obs_group in obs_groups["priv_obs"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_env_obs += obs[obs_group].shape[-1]
        
        
        # encoder --> put priv_obs through MLP to get z_size(8 according to paper)
        self.z_size = z_size
        self.encoder = MLP(num_env_obs, z_size, encoder_hidden_dims, activation)
        # encoder observation normalization
        self.encoder_obs_normalization = encoder_obs_normalization
        if encoder_obs_normalization:
            self.encoder_obs_normalizer = EmpiricalNormalization(num_env_obs)
        else:
            self.encoder_obs_normalizer = torch.nn.Identity()
        print(f"Encoder MLP: {self.encoder}")
        
        # actor
        self.actor = MLP(num_actor_obs+z_size, num_actions, actor_hidden_dims, activation) # input should be x_30 state, previous a_12 action, and encoder's z_8 output
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        # Takes the current state and predicts long term reward from this state allowing RL policy to adjust
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

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
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, obs):
        # Gather all policy inputs
        actor_obs = self.get_actor_obs(obs)  # shape: (B, 42)
        env_obs = self.get_encoder_obs(obs)  # shape: (B, 17)
        env_input = self.encoder_obs_normalizer(env_obs)
        
        # Combine everything (x_30, a_t-1_12, z_8 for actor input)
        latent_space = self.encoder(env_input)  # shape: (B, 8)
        actor_input = torch.cat([actor_obs, latent_space], dim=-1)  # shape: (B, 50)

        # Normalize and pass to actor
        actor_input = self.actor_obs_normalizer(actor_input)
        mean = self.actor(actor_input)  # shape: (B, num_actions)

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    # CONFIRM: that this is only run in evaluation time, so no need to update self.distribution
    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)  # shape: (B, 42)
        env_obs = self.get_encoder_obs(obs)  # shape: (B, 17)
        env_input = self.encoder_obs_normalizer(env_obs)
        
        latent_space = self.encoder(env_input)  # shape: (B, 8)t
        actor_input = torch.cat([actor_obs, latent_space], dim=-1)  # shape: (B, 50)

        # Normalize and pass to actor
        actor_input = self.actor_obs_normalizer(actor_input)
        mean = self.actor(actor_input)  # shape: (B, num_actions)
        return mean

    def get_latents(self, obs):
        env_obs = self.get_encoder_obs(obs)  # shape: (B, 17)
        env_input = self.encoder_obs_normalizer(env_obs)
        
        latent_space = self.encoder(env_input)  # shape: (B, 8)t
        return latent_space

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)
    
    def get_encoder_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["priv_obs"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.encoder_obs_normalization:
            encoder_obs = self.get_encoder_obs(obs)
            self.encoder_obs_normalizer.update(encoder_obs)
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the base-policy model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes