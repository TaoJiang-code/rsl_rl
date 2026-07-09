# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from rsl_rl.modules import MoELayer


def test_moe_forward_and_gate_weights():
    layer = MoELayer(
        input_dim=6,
        output_dim=3,
        expert_hidden_dims=[8, 8],
        num_experts=4,
        gate_hidden_dims=[5],
    )
    inputs = torch.randn(10, 6, requires_grad=True)

    outputs, gate_weights = layer(inputs, return_gate_weights=True)

    assert outputs.shape == (10, 3)
    assert gate_weights.shape == (10, 4)
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(10))
    assert layer.last_gate_weights is gate_weights

    outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in layer.parameters())


def test_moe_rejects_invalid_expert_count():
    with pytest.raises(ValueError, match="num_experts must be >= 1"):
        MoELayer(input_dim=6, output_dim=3, expert_hidden_dims=[8], num_experts=0)
