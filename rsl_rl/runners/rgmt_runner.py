# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import RGMT
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class RGMTRunner(OnPolicyRunner):
    """On-policy runner for Robust and Generalized Motion Tracking."""

    alg: RGMT
    """The RGMT algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the RGMT runner and verify that the configured algorithm matches it."""
        super().__init__(env, train_cfg, log_dir, device)
        if not isinstance(self.alg, RGMT):
            raise TypeError("RGMTRunner requires cfg['algorithm']['class_name'] to resolve to RGMT.")

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save RGMT policy and adaptive motion-sampler state."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        env_state = self._rgmt_env_state_dict()
        if env_state is not None:
            saved_dict["rgmt_env_state_dict"] = env_state
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load RGMT policy and adaptive motion-sampler state when present."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        self._load_rgmt_env_state_dict(loaded_dict.get("rgmt_env_state_dict"))
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> nn.Module:
        """Return the RGMT actor on the requested device for inference."""
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)

    def _motion_command(self):
        env = getattr(self.env, "unwrapped", self.env)
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return None
        try:
            return command_manager.get_term("motion")
        except Exception:
            return None

    def _rgmt_env_state_dict(self) -> dict | None:
        command = self._motion_command()
        if command is None or not hasattr(command, "sampler_state_dict"):
            return None
        return {"motion_sampler": command.sampler_state_dict()}

    def _load_rgmt_env_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        command = self._motion_command()
        if command is None or not hasattr(command, "load_sampler_state_dict"):
            return
        sampler_state = state.get("motion_sampler")
        if sampler_state is None:
            return
        try:
            restored = command.load_sampler_state_dict(sampler_state, strict=False)
        except Exception as exc:
            print(f"[WARN] RGMT sampler state was not restored: {exc}")
            return
        if restored:
            print("[INFO] Restored RGMT adaptive sampler state from checkpoint.")
        else:
            print("[WARN] RGMT sampler state was skipped because it is incompatible with the current environment.")
