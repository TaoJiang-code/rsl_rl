# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch.nn as nn

from rsl_rl.algorithms import SonicLoRAPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class SonicLoRARunner(OnPolicyRunner):
    """On-policy runner for Sonic/Any2Any LoRA fine-tuning."""

    alg: SonicLoRAPPO
    """The Sonic LoRA PPO algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        if not isinstance(self.alg, SonicLoRAPPO):
            raise TypeError("SonicLoRARunner requires cfg['algorithm']['class_name'] to resolve to SonicLoRAPPO.")

    def get_inference_policy(self, device: str | None = None) -> nn.Module:
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)
