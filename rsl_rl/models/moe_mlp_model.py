# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MoELayer


class MoEMLPModel(MLPModel):
    """MLP model whose output head is a softmax-gated mixture of experts."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 8,
        gate_hidden_dims: tuple[int, ...] | list[int] = (),
        gate_activation: str | None = None,
    ) -> None:
        """Initialize the mixture-of-experts MLP model."""
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        moe_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MoELayer(
            input_dim=self._get_latent_dim(),
            output_dim=moe_output_dim,
            expert_hidden_dims=hidden_dims,
            num_experts=num_experts,
            gate_hidden_dims=gate_hidden_dims,
            activation=activation,
            gate_activation=gate_activation,
        )

        if self.distribution is not None:
            for expert in self.mlp.experts:
                self.distribution.init_mlp_weights(expert)

    @property
    def gate_weights(self) -> torch.Tensor | None:
        """Return the gate weights computed by the most recent forward pass."""
        return self.mlp.last_gate_weights
