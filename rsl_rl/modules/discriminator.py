# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd

from .mlp import MLP


class Discriminator(nn.Module):
    """Discriminator used by adversarial motion priors.

    The discriminator receives the concatenation of an AMP state and the next
    AMP state and is trained to distinguish policy rollouts from expert motion
    clips. Its prediction is also converted into the AMP style reward.
    """

    def __init__(
        self,
        input_dim: int,
        reward_coef: float = 1.0,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512),
        activation: str = "relu",
        task_reward_lerp: float = 0.0,
        device: str = "cpu",
    ) -> None:
        """Initialize the discriminator network."""
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("Discriminator hidden_dims must contain at least one layer.")
        self.input_dim = input_dim
        self.reward_coef = reward_coef
        self.task_reward_lerp = task_reward_lerp
        self.device = device

        trunk_hidden_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims
        self.trunk = MLP(input_dim, hidden_dims[-1], trunk_hidden_dims, activation)
        self.amp_linear = nn.Linear(hidden_dims[-1], 1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate discriminator logits."""
        return self.amp_linear(self.trunk(x))

    def compute_grad_penalty(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        """Compute the AMP gradient penalty on expert transitions."""
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad_(True)

        disc = self(expert_data)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=torch.ones_like(disc),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert discriminator predictions into AMP rewards."""
        with torch.no_grad():
            was_training = self.training
            self.eval()
            if normalizer is not None:
                state = normalizer(state)
                next_state = normalizer(next_state)

            logits = self(torch.cat([state, next_state], dim=-1))
            amp_reward = self.reward_coef * torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)
            if self.task_reward_lerp > 0.0:
                task_reward = task_reward.unsqueeze(-1) if task_reward.ndim == 1 else task_reward
                amp_reward = (1.0 - self.task_reward_lerp) * amp_reward + self.task_reward_lerp * task_reward
            if was_training:
                self.train()
            return amp_reward.squeeze(-1), logits
