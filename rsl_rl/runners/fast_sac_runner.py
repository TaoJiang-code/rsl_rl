# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import FastSAC
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan


class FastSACRunner(OnPolicyRunner):
    """Off-policy runner for FastSAC.

    The loop follows the Holosoma FastSAC rhythm: collect one vectorized environment step, insert it into replay, then
    run replay updates once `learning_starts` has passed. `num_learning_iterations` is therefore a global-step count,
    not a PPO-style rollout-iteration count.
    """

    alg: FastSAC

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        train_cfg.setdefault("num_steps_per_env", 1)
        train_cfg.setdefault("logging_interval", 1)
        train_cfg["algorithm"].setdefault("rnd_cfg", None)
        super().__init__(env, train_cfg, log_dir, device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run FastSAC learning for `num_learning_iterations` environment steps."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, _ = self._get_observations()
        obs = obs.to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        last_loss_dict: dict[str, float] = {}

        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                actions = self.alg.act(obs)
                next_obs, rewards, dones, extras = self._step(actions.to(self.env.device))
                if self.cfg.get("check_for_nan", True):
                    check_nan(next_obs, rewards, dones)
                next_obs, rewards, dones = (
                    next_obs.to(self.device),
                    rewards.to(self.device),
                    dones.to(self.device),
                )
                self.alg.process_env_step(next_obs, rewards, dones, extras)
                self.logger.process_env_step(rewards, dones, extras)
                obs = next_obs

            collect_time = time.time() - start
            start = time.time()
            loss_dict = self.alg.update()
            if loss_dict:
                last_loss_dict = loss_dict
            learn_time = time.time() - start

            self.current_learning_iteration = it
            should_log = (it == start_it) or ((it + 1) % self.cfg["logging_interval"] == 0) or (it == total_it - 1)
            if should_log:
                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=collect_time,
                    learn_time=learn_time,
                    loss_dict=last_loss_dict,
                    learning_rate=self.alg.learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=None,
                    print_minimal=True,
                )

            if (
                self.logger.writer is not None
                and self.cfg["save_interval"] > 0
                and it > 0
                and it % self.cfg["save_interval"] == 0
            ):
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
