# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.mlp import MLP
from rsl_rl.utils import resolve_nn_activation


class MoELayer(nn.Module):
    """Mixture-of-Experts MLP layer with a softmax gate."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        expert_hidden_dims: tuple[int, ...] | list[int],
        num_experts: int = 2,
        gate_hidden_dims: tuple[int, ...] | list[int] = (),
        activation: str = "elu",
        gate_activation: str | None = None,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.last_gate_weights: torch.Tensor | None = None

        self.experts = nn.ModuleList(
            [MLP(input_dim, output_dim, expert_hidden_dims, activation) for _ in range(num_experts)]
        )
        self.gate = self._build_gate(input_dim, num_experts, gate_hidden_dims, gate_activation or activation)

    @staticmethod
    def _build_gate(
        input_dim: int,
        num_experts: int,
        gate_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> nn.Module:
        if len(gate_hidden_dims) == 0:
            return nn.Linear(input_dim, num_experts)

        layers: list[nn.Module] = []
        activation_mod = resolve_nn_activation(activation)
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in gate_hidden_dims]
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)
        for idx in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[idx], hidden_dims_processed[idx + 1]))
            layers.append(activation_mod)
        layers.append(nn.Linear(hidden_dims_processed[-1], num_experts))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_gate_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gate_weights = torch.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weight_shape = (*gate_weights.shape, *((1,) * (expert_outputs.ndim - gate_weights.ndim)))
        output = (gate_weights.reshape(weight_shape) * expert_outputs).sum(dim=1)
        self.last_gate_weights = gate_weights

        if return_gate_weights:
            return output, gate_weights
        return output
