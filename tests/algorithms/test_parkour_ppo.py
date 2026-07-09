# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ParkourPPO."""

import inspect
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import AMPPPO, ParkourPPO
from rsl_rl.algorithms.parkour_ppo import _EncoderMoEModel, _add_latent
from rsl_rl.models import CNNModel, MoEMLPModel

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 3
CNN_LATENT_DIM = 6


def _make_cnn_moe_actor() -> tuple[_EncoderMoEModel, TensorDict]:
    obs = TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, OBS_DIM),
            "depth_image": torch.randn(NUM_ENVS, 1, 16, 16),
        },
        batch_size=[NUM_ENVS],
    )
    obs_groups = {
        "cnn": ["depth_image"],
        "actor": ["policy"],
    }
    cnn = CNNModel(
        obs,
        obs_groups,
        "cnn",
        CNN_LATENT_DIM,
        hidden_dims=[8],
        activation="relu",
        last_activation="relu",
        cnn_cfg={
            "depth_image": {
                "output_channels": [4],
                "kernel_size": [3],
                "stride": [1],
                "padding": "zeros",
                "activation": "relu",
                "flatten": True,
            }
        },
    )

    with torch.no_grad():
        sample_obs = _add_latent(obs, "actor_cnn_latent", cnn(obs))
    actor_groups = {"actor": ["policy", "actor_cnn_latent"]}
    actor = MoEMLPModel(
        sample_obs,
        actor_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        num_experts=2,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    return _EncoderMoEModel(cnn, actor, "actor_cnn_latent"), obs


def test_parkour_ppo_explicitly_accepts_cnn() -> None:
    """ParkourPPO should expose CNN separately from actor and critic heads."""
    parameters = inspect.signature(ParkourPPO.__init__).parameters

    assert issubclass(ParkourPPO, AMPPPO)
    assert parameters["actor"].annotation == "MoEMLPModel"
    assert parameters["critic"].annotation == "MoEMLPModel"
    assert "cnn" in parameters
    assert parameters["cnn"].annotation == "CNNModel"
    assert "critic_cnn" in parameters


def test_cnn_moe_composition_backpropagates_into_cnn() -> None:
    """The MoE action loss should update the separately supplied CNN."""
    model, obs = _make_cnn_moe_actor()

    actions = model(obs)
    actions.sum().backward()

    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert isinstance(model.cnn.mlp[-1], nn.ReLU)
    assert any(parameter.grad is not None for parameter in model.cnn.parameters())
    assert any(parameter.grad is not None for parameter in model.head.parameters())
