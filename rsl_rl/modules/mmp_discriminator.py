# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from enum import Enum
from tensordict import TensorDict
from torch import autograd

from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


class MMPLossType(Enum):
    """Supported discriminator objectives for MMP."""

    GAN = "gan"
    LSGAN = "lsgan"
    WGAN = "wgan"


class MMPDiscriminator(nn.Module):
    """Environment-fed discriminator for multimodal motion prior training.

    The discriminator consumes history observations supplied by the environment,
    e.g. ``obs["mmp"]`` for policy rollouts and ``obs["mmp_expert"]`` for
    replayed expert motion. Each observation is expected to have shape
    ``[num_envs, history_steps, obs_dim]``.
    """

    def __init__(
        self,
        disc_obs_dim: int,
        disc_obs_steps: int,
        policy_obs_groups: list[str] | tuple[str, ...] = ("mmp",),
        expert_obs_groups: list[str] | tuple[str, ...] = ("mmp_expert",),
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "relu",
        loss_type: str = "lsgan",
        reward_coef: float = 1.0,
        device: str = "cpu",
    ) -> None:
        """Initialize the MMP discriminator."""
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("MMPDiscriminator hidden_dims must contain at least one layer.")

        self.disc_obs_dim = disc_obs_dim
        self.disc_obs_steps = disc_obs_steps
        self.input_dim = disc_obs_dim * disc_obs_steps
        self.policy_obs_groups = list(policy_obs_groups)
        self.expert_obs_groups = list(expert_obs_groups)
        self.loss_type = MMPLossType(loss_type.lower())
        self.reward_coef = reward_coef
        self.device = device

        trunk_hidden_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims
        self.trunk = MLP(self.input_dim, hidden_dims[-1], trunk_hidden_dims, activation)
        self.linear = nn.Linear(hidden_dims[-1], 1)
        self.obs_normalizer = EmpiricalNormalization(self.disc_obs_dim, until=int(1.0e8)).to(device)
        self.output_normalizer = EmpiricalNormalization(1, until=int(1.0e8)).to(device) if self.loss_type == MMPLossType.WGAN else nn.Identity()
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate discriminator logits/scores."""
        return self.linear(self.trunk(x))

    def get_policy_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        """Extract policy discriminator observations from a TensorDict."""
        return self._get_obs(obs, self.policy_obs_groups, flatten_history_dim)

    def get_expert_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        """Extract expert discriminator observations from a TensorDict."""
        return self._get_obs(obs, self.expert_obs_groups, flatten_history_dim)

    def normalize_obs(self, disc_obs: torch.Tensor) -> torch.Tensor:
        """Normalize ``[B, T, D]`` discriminator observations."""
        self._validate_disc_obs(disc_obs)
        normed = self.obs_normalizer(disc_obs.reshape(-1, self.disc_obs_dim))
        return normed.reshape(-1, self.disc_obs_steps, self.disc_obs_dim)

    def update_normalization(self, disc_obs: torch.Tensor) -> None:
        """Update discriminator observation normalization statistics."""
        self._validate_disc_obs(disc_obs)
        self.obs_normalizer.update(disc_obs.reshape(-1, self.disc_obs_dim))  # type: ignore

    def compute_style_reward(self, disc_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert discriminator predictions into pure MMP style rewards."""
        with torch.no_grad():
            was_training = self.training
            self.eval()

            normed_obs = self.normalize_obs(disc_obs)
            score = self(normed_obs.reshape(normed_obs.shape[0], -1))

            if self.loss_type == MMPLossType.GAN:
                prob = torch.sigmoid(score)
                style_reward = -torch.log(torch.clamp(1.0 - prob, min=1.0e-6))
            elif self.loss_type == MMPLossType.LSGAN:
                style_reward = torch.clamp(1.0 - 0.25 * torch.square(score - 1.0), min=0.0)
            elif self.loss_type == MMPLossType.WGAN:
                style_reward = self.output_normalizer(score)
            else:
                raise ValueError(f"Unsupported MMP loss type: {self.loss_type}.")

            style_reward = self.reward_coef * style_reward
            if was_training:
                self.train()
            return style_reward.squeeze(-1), score.squeeze(-1)

    def compute_loss(self, policy_obs: torch.Tensor, expert_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute discriminator loss on policy and expert batches."""
        policy_score = self(policy_obs.reshape(policy_obs.shape[0], -1))
        expert_score = self(expert_obs.reshape(expert_obs.shape[0], -1))

        if self.loss_type == MMPLossType.GAN:
            bce = nn.BCEWithLogitsLoss()
            policy_loss = bce(policy_score, torch.zeros_like(policy_score))
            expert_loss = bce(expert_score, torch.ones_like(expert_score))
            disc_loss = 0.5 * (policy_loss + expert_loss)
        elif self.loss_type == MMPLossType.LSGAN:
            policy_loss = nn.functional.mse_loss(policy_score, -torch.ones_like(policy_score))
            expert_loss = nn.functional.mse_loss(expert_score, torch.ones_like(expert_score))
            disc_loss = 0.5 * (policy_loss + expert_loss)
        elif self.loss_type == MMPLossType.WGAN:
            disc_loss = -expert_score.mean() + policy_score.mean()
        else:
            raise ValueError(f"Unsupported MMP loss type: {self.loss_type}.")

        return disc_loss, policy_score, expert_score

    def compute_grad_penalty(self, expert_obs: torch.Tensor, scale: float) -> torch.Tensor:
        """Compute gradient penalty on normalized expert discriminator observations."""
        expert_data = expert_obs.reshape(expert_obs.shape[0], -1).clone().detach().requires_grad_(True)
        score = self(expert_data)
        grad = autograd.grad(
            outputs=score,
            inputs=expert_data,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return scale * grad.norm(2, dim=1).pow(2).mean()

    def _get_obs(self, obs: TensorDict, obs_groups: list[str], flatten_history_dim: bool) -> torch.Tensor:
        """Extract and concatenate configured discriminator observation groups."""
        obs_list = []
        for obs_group in obs_groups:
            if obs_group not in obs:
                raise KeyError(
                    f"Observation group '{obs_group}' was not found. Available observations: {list(obs.keys())}."
                )
            obs_tensor = obs[obs_group]
            self._validate_disc_obs(obs_tensor)
            obs_list.append(obs_tensor)
        disc_obs = torch.cat(obs_list, dim=-1)
        if flatten_history_dim:
            return disc_obs.reshape(disc_obs.shape[0], -1)
        return disc_obs

    def _validate_disc_obs(self, disc_obs: torch.Tensor) -> None:
        """Validate discriminator observation shape."""
        if disc_obs.ndim != 3:
            raise ValueError(
                "MMP discriminator observations must have shape [num_envs, history_steps, obs_dim], "
                f"got {tuple(disc_obs.shape)}."
            )
        if disc_obs.shape[1] != self.disc_obs_steps:
            raise ValueError(f"Expected {self.disc_obs_steps} history steps, got {disc_obs.shape[1]}.")
        if disc_obs.shape[2] != self.disc_obs_dim:
            raise ValueError(f"Expected discriminator obs dim {self.disc_obs_dim}, got {disc_obs.shape[2]}.")
