# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import AMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


class AmpOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for PPO with adversarial motion priors."""

    alg: AMPPPO
    """The AMP actor-critic algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the AMP runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        self._configure_multi_gpu()
        obs, extras = self._get_observations()
        self._ensure_initial_amp_obs(obs, extras)

        alg_class: type[AMPPPO] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the AMP learning loop for the specified number of iterations."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs, extras = self._get_observations()
        amp_obs = self._extract_amp_obs(obs, extras)
        obs = obs.to(self.device)
        amp_obs = amp_obs.to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, amp_obs)
                    next_obs, task_rewards, dones, extras = self._step(actions.to(self.env.device))
                    next_amp_obs = self._extract_amp_obs(next_obs, extras)

                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, task_rewards, dones)

                    next_obs = next_obs.to(self.device)
                    task_rewards = task_rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    amp_next_for_reward = next_amp_obs.clone()
                    reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if reset_env_ids.numel() > 0:
                        amp_next_for_reward[reset_env_ids] = amp_obs[reset_env_ids]

                    rewards = self.alg.compute_amp_reward(amp_obs, amp_next_for_reward, task_rewards)
                    self.alg.process_env_step(next_obs, rewards, dones, extras, amp_next_for_reward)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                    obs = next_obs
                    amp_obs = next_amp_obs

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the actor on the requested device for inference."""
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)  # type: ignore

    def _get_observations(self) -> tuple[TensorDict, dict]:
        """Get observations while accepting current and IsaacLab 2.1 wrapper formats."""
        result = self.env.get_observations()
        if isinstance(result, tuple):
            obs, extras = result
        else:
            obs, extras = result, {}
        return self._to_tensordict(obs, extras), extras

    def _step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment while accepting current and IsaacLab 2.1 wrapper formats."""
        obs, rewards, dones, extras = self.env.step(actions)
        return self._to_tensordict(obs, extras), rewards, dones, extras

    def _to_tensordict(self, obs: TensorDict | torch.Tensor | dict, extras: dict) -> TensorDict:
        """Convert legacy observations to TensorDict."""
        if isinstance(obs, TensorDict):
            return obs
        if isinstance(obs, dict):
            return TensorDict(obs, batch_size=[self.env.num_envs], device=self.env.device)

        data = {"policy": obs}
        observations = extras.get("observations", {})
        if isinstance(observations, dict):
            for key, value in observations.items():
                if key != "policy":
                    data[key] = value
        if "amp_obs" in extras:
            data.setdefault("amp", extras["amp_obs"])
        return TensorDict(data, batch_size=[self.env.num_envs], device=self.env.device)

    def _extract_amp_obs(self, obs: TensorDict, extras: dict) -> torch.Tensor:
        """Extract AMP observations from TensorDict, extras, or the unwrapped IsaacLab env."""
        try:
            return AMPPPO.extract_amp_obs(obs, extras)
        except KeyError:
            unwrapped = getattr(self.env, "unwrapped", self.env)
            env_extras = getattr(unwrapped, "extras", {})
            if isinstance(env_extras, dict) and "amp_obs" in env_extras:
                return env_extras["amp_obs"]
            raise

    def _ensure_initial_amp_obs(self, obs: TensorDict, extras: dict) -> None:
        """Fail early if the environment does not expose AMP observations."""
        self._extract_amp_obs(obs, extras)
