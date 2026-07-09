# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from rsl_rl.models import MoEMLPModel
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 3
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def test_moe_actor_uses_standard_model_interface() -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    actor = MoEMLPModel(
        obs,
        OBS_GROUPS,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        num_experts=4,
        gate_hidden_dims=[8],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )

    deterministic_actions = actor(obs)
    stochastic_actions = actor(obs, stochastic_output=True)

    assert deterministic_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert stochastic_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert actor.gate_weights.shape == (NUM_ENVS, 4)
    assert torch.allclose(actor.gate_weights.sum(dim=-1), torch.ones(NUM_ENVS))
    assert actor.get_output_log_prob(stochastic_actions).shape == (NUM_ENVS,)


def test_moe_critic_is_independent_and_deterministic() -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    critic = MoEMLPModel(
        obs,
        OBS_GROUPS,
        "critic",
        1,
        hidden_dims=[16],
        num_experts=2,
    )

    values = critic(obs)

    assert values.shape == (NUM_ENVS, 1)
    assert critic.distribution is None
    assert critic.gate_weights.shape == (NUM_ENVS, 2)
