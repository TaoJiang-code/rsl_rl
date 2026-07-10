# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import ContextVAE, ContextVAEOutput, EmpiricalNormalization, HiddenState
from rsl_rl.utils import unpad_trajectories


class ContextVAEModel(nn.Module):
    """TensorDict wrapper around :class:`ContextVAE`.

    The module reads one or more 1D observation groups, concatenates them into a
    flattened history tensor, and feeds that tensor to the context VAE. It is
    intentionally separate from actor and critic models so DWAQ-style algorithms
    can optimize the VAE with their own losses.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        code_dim: int = 19,
        velocity_dim: int = 3,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128,),
        encoder_latent_dim: int = 64,
        decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
    ) -> None:
        """Initialize the TensorDict-aware context VAE model."""
        super().__init__()

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.output_dim = output_dim
        self.obs_normalization = obs_normalization

        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.vae = ContextVAE(
            input_dim=self.obs_dim,
            output_dim=output_dim,
            code_dim=code_dim,
            velocity_dim=velocity_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_latent_dim=encoder_latent_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        deterministic: bool = False,
    ) -> ContextVAEOutput:
        """Run the VAE on observations selected by this model's observation set."""
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        return self.vae(self.get_latent(obs), deterministic=deterministic)

    def encode(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return only the context code."""
        return self.forward(obs, masks=masks, hidden_state=hidden_state, deterministic=deterministic).code

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Concatenate and normalize the selected history observations."""
        latent = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        return self.obs_normalizer(latent)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            latent = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(latent)  # type: ignore

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset recurrent state. ContextVAEModel is feed-forward, so this is a no-op."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return recurrent hidden state. ContextVAEModel has no recurrent state."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent state. ContextVAEModel has no recurrent state."""
        pass

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute the flattened input dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    "ContextVAEModel only supports flattened 1D observations, "
                    f"got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim
