# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm used by the Stage-1 sequence retargeter."""

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class ResidualConv1dBlock(nn.Module):
    """Temporal Conv1D residual block preserving sequence length."""

    def __init__(self, channels: int, kernel_size: int = 3, activation: str = "gelu") -> None:
        super().__init__()
        padding = kernel_size // 2
        act = nn.GELU() if activation == "gelu" else nn.ELU()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            act,
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.norm = RMSNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class MotionBridgeRetargeter(nn.Module):
    """Stage-1 supervised retargeter: SMPL sequence -> GMR/G1 robot sequence.

    This is intentionally not an RL algorithm. It follows the Stage-1 direction:
    large-scale kinematic alignment with paired SMPL and GMR robot motion,
    trained with regression loss.
    """

    def __init__(
        self,
        input_dim: int = 75,
        output_dim: int = 36,
        hidden_dim: int = 512,
        num_conv_blocks: int = 4,
        num_transformer_layers: int = 6,
        num_attention_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = RMSNorm(hidden_dim)
        self.conv_blocks = nn.ModuleList([ResidualConv1dBlock(hidden_dim) for _ in range(num_conv_blocks)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.output_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.output_norm = RMSNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, smpl_sequence: torch.Tensor) -> torch.Tensor:
        """Predict robot motion for every frame.

        Args:
            smpl_sequence: Tensor with shape [batch, time, input_dim].

        Returns:
            Tensor with shape [batch, time, output_dim].
        """
        x = self.input_norm(self.input_projection(smpl_sequence))
        for block in self.conv_blocks:
            x = block(x)
        x = self.transformer(x)
        x = self.output_norm(x + self.output_conv(x.transpose(1, 2)).transpose(1, 2))
        return self.output_projection(x)


def stage1_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Paper-aligned Stage-1 objective: frame-wise L1 regression."""

    return F.l1_loss(prediction, target)
