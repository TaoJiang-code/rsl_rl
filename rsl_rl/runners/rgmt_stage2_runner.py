# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import RGMTStageII
from rsl_rl.env import VecEnv
from rsl_rl.runners.rgmt_runner import RGMTRunner


class RGMTStageIIRunner(RGMTRunner):
    """Runner that appends Stage-II acquisition/consolidation role masks."""

    alg: RGMTStageII

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        if not isinstance(self.alg, RGMTStageII):
            raise TypeError("RGMTStageIIRunner requires cfg['algorithm']['class_name'] to resolve to RGMTStageII.")

    def _get_observations(self) -> tuple[TensorDict, dict]:
        obs, extras = super()._get_observations()
        return self._append_stage2_role(obs), extras

    def _step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, extras = super()._step(actions)
        return self._append_stage2_role(obs), rewards, dones, extras

    def _append_stage2_role(self, obs: TensorDict) -> TensorDict:
        command = self.env.unwrapped.command_manager.get_term("motion")
        if "stage2_is_acquisition" not in command.metrics:
            raise RuntimeError(
                "RGMTStageIIRunner requires RGMTStageIIMotionCommand metrics['stage2_is_acquisition']."
            )
        is_acquisition = command.metrics["stage2_is_acquisition"].to(obs.device).reshape(self.env.num_envs, 1)
        obs = obs.clone()
        obs["stage2_is_acquisition"] = is_acquisition
        return obs
