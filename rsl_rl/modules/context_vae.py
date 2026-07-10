# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rsl_rl.modules.mlp import MLP


@dataclass
class ContextVAEOutput:
    """Structured output of a ContextVAE forward pass."""

    code: torch.Tensor
    """Full latent code ``[velocity | latent]`` with shape ``(B, code_dim)``."""

    code_vel: torch.Tensor
    """Velocity sub-code with shape ``(B, velocity_dim)``."""

    code_latent: torch.Tensor
    """Latent sub-code with shape ``(B, code_dim - velocity_dim)``."""

    reconstruction: torch.Tensor
    """Decoder reconstruction with shape ``(B, output_dim)``."""

    mean_vel: torch.Tensor
    """Velocity posterior mean with shape ``(B, velocity_dim)``."""

    logvar_vel: torch.Tensor
    """Velocity posterior log-variance with shape ``(B, velocity_dim)``."""

    mean_latent: torch.Tensor
    """Latent posterior mean with shape ``(B, code_dim - velocity_dim)``."""

    logvar_latent: torch.Tensor
    """Latent posterior log-variance with shape ``(B, code_dim - velocity_dim)``."""


class ContextVAE(nn.Module):
    """Context variational auto-encoder used by DWAQ-style policies.

    The encoder maps a flattened observation history to an intermediate hidden
    representation. Two Gaussian branches then predict a velocity code and a
    residual latent code. The concatenated code can be decoded back to a target
    observation, while the code itself can be appended to the policy input.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        code_dim: int = 19,
        velocity_dim: int = 3,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (128,),
        encoder_latent_dim: int = 64,
        decoder_hidden_dims: list[int] | tuple[int, ...] = (64, 128),
        activation: str = "elu",
    ) -> None:
        """Initialize the context VAE."""
        super().__init__()

        if velocity_dim <= 0:
            raise ValueError(f"velocity_dim must be positive, got {velocity_dim}.")
        if code_dim <= velocity_dim:
            raise ValueError(f"code_dim must be larger than velocity_dim, got {code_dim} and {velocity_dim}.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.code_dim = code_dim
        self.velocity_dim = velocity_dim
        self.latent_dim = code_dim - velocity_dim

        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=encoder_latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            last_activation=activation,
        )

        self.mean_vel = nn.Linear(encoder_latent_dim, velocity_dim)
        self.logvar_vel = nn.Linear(encoder_latent_dim, velocity_dim)
        self.mean_latent = nn.Linear(encoder_latent_dim, self.latent_dim)
        self.logvar_latent = nn.Linear(encoder_latent_dim, self.latent_dim)

        self.decoder = MLP(
            input_dim=code_dim,
            output_dim=output_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
        )

    def forward(self, obs_history: torch.Tensor, deterministic: bool = False) -> ContextVAEOutput:
        """Encode history, sample or take posterior means, and decode the code."""
        h = self.encoder(obs_history)

        mean_vel = self.mean_vel(h)
        logvar_vel = self.logvar_vel(h)
        mean_latent = self.mean_latent(h)
        logvar_latent = self.logvar_latent(h)

        if deterministic:
            code_vel = mean_vel
            code_latent = mean_latent
        else:
            code_vel = self.reparameterize(mean_vel, logvar_vel)
            code_latent = self.reparameterize(mean_latent, logvar_latent)

        code = torch.cat((code_vel, code_latent), dim=-1)
        reconstruction = self.decoder(code)

        return ContextVAEOutput(
            code=code,
            code_vel=code_vel,
            code_latent=code_latent,
            reconstruction=reconstruction,
            mean_vel=mean_vel,
            logvar_vel=logvar_vel,
            mean_latent=mean_latent,
            logvar_latent=logvar_latent,
        )

    def encode(self, obs_history: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return only the context code."""
        return self.forward(obs_history, deterministic=deterministic).code

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample with the reparameterization trick."""
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    reparameterise = reparameterize
