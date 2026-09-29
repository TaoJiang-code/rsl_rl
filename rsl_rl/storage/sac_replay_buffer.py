# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict


class SACReplayBuffer:
    """Holosoma-style vectorized replay buffer with asymmetric observations and n-step returns."""

    def __init__(
        self,
        num_envs: int,
        buffer_size: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device

        self.observations = torch.zeros(num_envs, buffer_size, actor_obs_dim, device=device)
        self.actions = torch.zeros(num_envs, buffer_size, action_dim, device=device)
        self.rewards = torch.zeros(num_envs, buffer_size, device=device)
        self.dones = torch.zeros(num_envs, buffer_size, dtype=torch.long, device=device)
        self.truncations = torch.zeros(num_envs, buffer_size, dtype=torch.long, device=device)
        self.next_observations = torch.zeros(num_envs, buffer_size, actor_obs_dim, device=device)
        self.critic_observations = torch.zeros(num_envs, buffer_size, critic_obs_dim, device=device)
        self.next_critic_observations = torch.zeros(num_envs, buffer_size, critic_obs_dim, device=device)
        self.ptr = 0

    @property
    def num_samples(self) -> int:
        return min(self.buffer_size, self.ptr) * self.num_envs

    def can_sample(self, batch_size_per_env: int) -> bool:
        return min(self.buffer_size, self.ptr) >= max(1, batch_size_per_env)

    def extend(self, transition: TensorDict) -> None:
        """Insert one vectorized environment step."""
        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = transition["observations"].detach()
        self.actions[:, ptr] = transition["actions"].detach()
        self.rewards[:, ptr] = transition["next"]["rewards"].detach()
        self.dones[:, ptr] = transition["next"]["dones"].detach().long()
        self.truncations[:, ptr] = transition["next"]["truncations"].detach().long()
        self.next_observations[:, ptr] = transition["next"]["observations"].detach()
        self.critic_observations[:, ptr] = transition["critic_observations"].detach()
        self.next_critic_observations[:, ptr] = transition["next"]["critic_observations"].detach()
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size_per_env: int) -> TensorDict:
        """Sample `num_envs * batch_size_per_env` transitions."""
        if not self.can_sample(batch_size_per_env):
            raise RuntimeError(
                f"Cannot sample {batch_size_per_env} transitions per env from replay buffer with ptr={self.ptr}."
            )
        if self.n_steps == 1:
            return self._sample_one_step(batch_size_per_env)
        return self._sample_n_step(batch_size_per_env)

    def _sample_one_step(self, batch_size_per_env: int) -> TensorDict:
        indices = torch.randint(
            0,
            min(self.buffer_size, self.ptr),
            (self.num_envs, batch_size_per_env),
            device=self.device,
        )
        observations = self._gather(self.observations, indices, self.actor_obs_dim)
        next_observations = self._gather(self.next_observations, indices, self.actor_obs_dim)
        actions = self._gather(self.actions, indices, self.action_dim)
        rewards = torch.gather(self.rewards, 1, indices).reshape(self.num_envs * batch_size_per_env)
        dones = torch.gather(self.dones, 1, indices).reshape(self.num_envs * batch_size_per_env)
        truncations = torch.gather(self.truncations, 1, indices).reshape(self.num_envs * batch_size_per_env)
        critic_observations = self._gather(self.critic_observations, indices, self.critic_obs_dim)
        next_critic_observations = self._gather(self.next_critic_observations, indices, self.critic_obs_dim)
        effective_n_steps = torch.ones_like(dones)
        return self._make_batch(
            observations,
            actions,
            rewards,
            dones,
            truncations,
            next_observations,
            effective_n_steps,
            critic_observations,
            next_critic_observations,
        )

    def _sample_n_step(self, batch_size_per_env: int) -> TensorDict:
        rollback_truncations = None
        if self.ptr >= self.buffer_size:
            current_pos = self.ptr % self.buffer_size
            rollback_truncations = (current_pos - 1, self.truncations[:, current_pos - 1].clone())
            self.truncations[:, current_pos - 1] = torch.logical_not(self.dones[:, current_pos - 1]).long()
            indices = torch.randint(0, self.buffer_size, (self.num_envs, batch_size_per_env), device=self.device)
        else:
            max_start_idx = max(1, self.ptr - self.n_steps + 1)
            indices = torch.randint(0, max_start_idx, (self.num_envs, batch_size_per_env), device=self.device)

        observations = self._gather(self.observations, indices, self.actor_obs_dim)
        actions = self._gather(self.actions, indices, self.action_dim)
        critic_observations = self._gather(self.critic_observations, indices, self.critic_obs_dim)

        seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
        all_indices = (indices.unsqueeze(-1) + seq_offsets) % self.buffer_size
        all_rewards = torch.gather(self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
        all_dones = torch.gather(self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
        all_truncations = torch.gather(
            self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
        )

        all_dones_shifted = torch.cat([torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2)
        done_masks = torch.cumprod(1.0 - all_dones_shifted.float(), dim=2)
        effective_n_steps = done_masks.sum(dim=2)

        discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))
        rewards = (all_rewards * done_masks * discounts.view(1, 1, -1)).sum(dim=2)

        first_done = torch.argmax((all_dones > 0).float(), dim=2)
        first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)
        first_done = torch.where(all_dones.sum(dim=2) == 0, self.n_steps - 1, first_done)
        first_trunc = torch.where(all_truncations.sum(dim=2) == 0, self.n_steps - 1, first_trunc)
        final_indices = torch.minimum(first_done, first_trunc)
        final_next_obs_indices = torch.gather(all_indices, 2, final_indices.unsqueeze(-1)).squeeze(-1)

        next_observations = self._gather(self.next_observations, final_next_obs_indices, self.actor_obs_dim)
        next_critic_observations = self._gather(
            self.next_critic_observations, final_next_obs_indices, self.critic_obs_dim
        )
        dones = torch.gather(self.dones, 1, final_next_obs_indices)
        truncations = torch.gather(self.truncations, 1, final_next_obs_indices)

        if rollback_truncations is not None:
            pos, values = rollback_truncations
            self.truncations[:, pos] = values

        return self._make_batch(
            observations,
            actions,
            rewards.reshape(self.num_envs * batch_size_per_env),
            dones.reshape(self.num_envs * batch_size_per_env),
            truncations.reshape(self.num_envs * batch_size_per_env),
            next_observations,
            effective_n_steps.reshape(self.num_envs * batch_size_per_env),
            critic_observations,
            next_critic_observations,
        )

    def _gather(self, data: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.gather(data, 1, indices.unsqueeze(-1).expand(-1, -1, dim)).reshape(
            self.num_envs * indices.shape[1], dim
        )

    def _make_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncations: torch.Tensor,
        next_observations: torch.Tensor,
        effective_n_steps: torch.Tensor,
        critic_observations: torch.Tensor,
        next_critic_observations: torch.Tensor,
    ) -> TensorDict:
        batch_size = observations.shape[0]
        batch = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "observations": next_observations,
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "effective_n_steps": effective_n_steps,
                },
                "critic_observations": critic_observations,
            },
            batch_size=batch_size,
        )
        batch["next"]["critic_observations"] = next_critic_observations
        return batch


SACReplayBatch = TensorDict
