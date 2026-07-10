# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch.nn as nn

from rsl_rl.algorithms import DWAQPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class DWAQRunner(OnPolicyRunner):
    """On-policy runner for DWAQ PPO with a ContextVAE policy encoder."""

    alg: DWAQPPO
    """The DWAQ PPO algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the DWAQ runner and verify that the configured algorithm matches it."""
        super().__init__(env, train_cfg, log_dir, device)
        if not isinstance(self.alg, DWAQPPO):
            raise TypeError("DWAQRunner requires cfg['algorithm']['class_name'] to resolve to DWAQPPO.")

    def get_inference_policy(self, device: str | None = None) -> nn.Module:
        """Return the ContextVAE-augmented actor on the requested device for inference."""
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)
