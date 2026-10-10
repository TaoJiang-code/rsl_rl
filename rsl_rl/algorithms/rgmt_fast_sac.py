# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.fast_sac import FastSAC, FastSACCriticModel
from rsl_rl.algorithms.rgmt import RGMTActorModel
from rsl_rl.env import VecEnv
from rsl_rl.modules import MLP
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups


class RGMTSACReplayBuffer:
    """Vectorized SAC replay buffer that preserves RGMT actor observation groups."""

    def __init__(
        self,
        num_envs: int,
        buffer_size: int,
        actor_obs: TensorDict,
        critic_obs_dim: int,
        action_dim: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device

        self.observations = self._allocate_actor_obs(actor_obs)
        self.next_observations = self._allocate_actor_obs(actor_obs)
        self.actions = torch.zeros(num_envs, buffer_size, action_dim, device=device)
        self.rewards = torch.zeros(num_envs, buffer_size, device=device)
        self.dones = torch.zeros(num_envs, buffer_size, dtype=torch.long, device=device)
        self.truncations = torch.zeros(num_envs, buffer_size, dtype=torch.long, device=device)
        self.critic_observations = torch.zeros(num_envs, buffer_size, critic_obs_dim, device=device)
        self.next_critic_observations = torch.zeros(num_envs, buffer_size, critic_obs_dim, device=device)
        self.ptr = 0

    @property
    def num_samples(self) -> int:
        return min(self.buffer_size, self.ptr) * self.num_envs

    def can_sample(self, batch_size_per_env: int) -> bool:
        return min(self.buffer_size, self.ptr) >= max(1, batch_size_per_env)

    def extend(self, transition: TensorDict) -> None:
        ptr = self.ptr % self.buffer_size
        self._copy_actor_obs(self.observations, transition["observations"], ptr)
        self._copy_actor_obs(self.next_observations, transition["next"]["observations"], ptr)
        self.actions[:, ptr] = transition["actions"].detach()
        self.rewards[:, ptr] = transition["next"]["rewards"].detach()
        self.dones[:, ptr] = transition["next"]["dones"].detach().long()
        self.truncations[:, ptr] = transition["next"]["truncations"].detach().long()
        self.critic_observations[:, ptr] = transition["critic_observations"].detach()
        self.next_critic_observations[:, ptr] = transition["next"]["critic_observations"].detach()
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size_per_env: int) -> TensorDict:
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
        observations = self._gather_actor_obs(self.observations, indices)
        next_observations = self._gather_actor_obs(self.next_observations, indices)
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

        observations = self._gather_actor_obs(self.observations, indices)
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

        next_observations = self._gather_actor_obs(self.next_observations, final_next_obs_indices)
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

    def _allocate_actor_obs(self, obs: TensorDict) -> TensorDict:
        return TensorDict(
            {
                key: torch.zeros(
                    self.num_envs,
                    self.buffer_size,
                    *value.shape[1:],
                    dtype=value.dtype,
                    device=self.device,
                )
                for key, value in obs.items()
            },
            batch_size=[self.num_envs, self.buffer_size],
            device=self.device,
        )

    def _copy_actor_obs(self, target: TensorDict, source: TensorDict, ptr: int) -> None:
        for key in target.keys():
            target[key][:, ptr].copy_(source[key].detach())

    def _gather_actor_obs(self, data: TensorDict, indices: torch.Tensor) -> TensorDict:
        batch_size = self.num_envs * indices.shape[1]
        gathered = {}
        for key, value in data.items():
            expand_shape = (self.num_envs, indices.shape[1], *value.shape[2:])
            gather_indices = indices.reshape(self.num_envs, indices.shape[1], *([1] * (value.ndim - 2))).expand(
                expand_shape
            )
            gathered[key] = torch.gather(value, 1, gather_indices).reshape(batch_size, *value.shape[2:])
        return TensorDict(gathered, batch_size=[batch_size], device=self.device)

    def _gather(self, data: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.gather(data, 1, indices.unsqueeze(-1).expand(-1, -1, dim)).reshape(
            self.num_envs * indices.shape[1], dim
        )

    def _make_batch(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncations: torch.Tensor,
        next_observations: TensorDict,
        effective_n_steps: torch.Tensor,
        critic_observations: torch.Tensor,
        next_critic_observations: torch.Tensor,
    ) -> TensorDict:
        batch_size = actions.shape[0]
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
            batch_size=[batch_size],
            device=self.device,
        )
        batch["next"]["critic_observations"] = next_critic_observations
        return batch


class RGMTFastSACActorModel(RGMTActorModel):
    """RGMT history/command encoder with a tanh-Gaussian SAC policy head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        use_tanh: bool = True,
        action_scale: float | list[float] | tuple[float, ...] = 1.0,
        action_bias: float | list[float] | tuple[float, ...] = 0.0,
        **kwargs,
    ) -> None:
        kwargs.pop("distribution_cfg", None)
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            distribution_cfg=None,
            **kwargs,
        )
        self.action_dim = output_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_tanh = use_tanh
        self.actor_obs_groups = self._unique_actor_obs_groups()
        actor_input_dim = self.obs_dim + self.embedding_dim
        self.log_std_mlp = MLP(actor_input_dim, output_dim, hidden_dims, activation)
        self._init_sac_heads()

        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32).view(1, -1))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32).view(1, -1))
        if self.action_scale.shape[-1] == 1:
            self.action_scale = self.action_scale.expand(1, output_dim).clone()
        if self.action_bias.shape[-1] == 1:
            self.action_bias = self.action_bias.expand(1, output_dim).clone()

    def forward(self, obs: TensorDict | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_normalized(obs)

    def forward_normalized(self, obs: TensorDict | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(obs, TensorDict):
            raise TypeError("RGMTFastSACActorModel expects TensorDict actor observations.")
        latent = self.get_latent(obs)
        mean = self.mlp(latent)
        log_std = self.log_std_mlp(latent)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        action = torch.tanh(mean) * self.action_scale + self.action_bias if self.use_tanh else mean
        return action, mean, log_std

    def sample(self, obs: TensorDict | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_actions_and_log_probs_normalized(obs)

    def explore(self, obs: TensorDict | torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, mean, log_std = self.forward_normalized(obs)
        if deterministic:
            return action
        std = log_std.exp()
        raw_action = torch.distributions.Normal(mean, std).rsample()
        if self.use_tanh:
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        return raw_action

    def get_actions_and_log_probs_normalized(self, obs: TensorDict | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self.forward_normalized(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
            log_prob = dist.log_prob(raw_action)
            log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1.0e-6)
            log_prob -= torch.log(self.action_scale + 1.0e-6)
        else:
            action = raw_action
            log_prob = dist.log_prob(raw_action)
        return action, log_prob.sum(dim=1)

    def get_observation(self, obs: TensorDict) -> TensorDict:
        return TensorDict(
            {group: obs[group] for group in self.actor_obs_groups},
            batch_size=obs.batch_size,
            device=obs.device,
        )

    def _unique_actor_obs_groups(self) -> list[str]:
        groups = []
        for group in (
            *self.obs_groups,
            *self.state_history_obs_groups,
            *self.action_history_obs_groups,
            *self.command_obs_groups,
        ):
            if group not in groups:
                groups.append(group)
        return groups

    def normalize_observation(self, obs: TensorDict | torch.Tensor, update: bool) -> TensorDict | torch.Tensor:
        if update and isinstance(obs, TensorDict):
            self.update_normalization(obs)
        return obs

    @property
    def output_std(self) -> torch.Tensor:
        return self.log_std_mlp[-1].weight.new_zeros(self.action_dim)

    def _init_sac_heads(self) -> None:
        for module in (self.mlp[-1], self.log_std_mlp[-1]):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.0)
                nn.init.constant_(module.bias, 0.0)


class RGMTFastSAC(FastSAC):
    """FastSAC variant whose actor replay observations remain structured RGMT TensorDicts."""

    def _sample_and_prepare_batches(self, large_data: TensorDict, batch_size_per_env: int) -> list[TensorDict]:
        self.actor.normalize_observation(large_data["observations"], update=True)
        self.actor.normalize_observation(large_data["next"]["observations"], update=True)
        large_data["critic_observations"] = self.critic.normalize_observation(
            large_data["critic_observations"], update=True
        )
        large_data["next"]["critic_observations"] = self.critic.normalize_observation(
            large_data["next"]["critic_observations"], update=True
        )

        samples_per_update = batch_size_per_env * self.replay_buffer.num_envs
        batches = []
        for i in range(self.num_updates):
            start = i * samples_per_update
            end = (i + 1) * samples_per_update
            batch = TensorDict(
                {
                    "observations": large_data["observations"][start:end],
                    "actions": large_data["actions"][start:end],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start:end],
                        "dones": large_data["next"]["dones"][start:end],
                        "truncations": large_data["next"]["truncations"][start:end],
                        "observations": large_data["next"]["observations"][start:end],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start:end],
                    },
                    "critic_observations": large_data["critic_observations"][start:end],
                },
                batch_size=[samples_per_update],
                device=self.device,
            )
            batch["next"]["critic_observations"] = large_data["next"]["critic_observations"][start:end]
            batches.append(batch)
        return batches

    def _replace_timeouts_with_final_obs(
        self,
        next_obs,
        extras: dict,
        truncations: torch.Tensor,
        final_key: str,
        model: RGMTFastSACActorModel | FastSACCriticModel,
    ):
        final_obs = self._extract_final_obs(extras, final_key, model)
        if final_obs is None:
            return next_obs
        mask = truncations.to(dtype=torch.bool, device=self.device)
        if isinstance(next_obs, TensorDict):
            replaced = next_obs.clone()
            for key in replaced.keys():
                view_shape = (mask.shape[0],) + (1,) * (replaced[key].ndim - 1)
                replaced[key] = torch.where(mask.view(view_shape), final_obs[key].to(self.device), replaced[key])
            return replaced
        return torch.where(mask.unsqueeze(-1), final_obs.to(self.device), next_obs)

    def _extract_final_obs(
        self,
        extras: dict,
        final_key: str,
        model: RGMTFastSACActorModel | FastSACCriticModel,
    ):
        observations = extras.get("observations", {})
        top_level_final = extras.get("final_observations")
        model_groups = getattr(model, "actor_obs_groups", model.obs_groups)
        if isinstance(top_level_final, dict) and all(group in top_level_final for group in model_groups):
            final_td = TensorDict(top_level_final, batch_size=[self.replay_buffer.num_envs], device=self.device)
            return model.get_observation(final_td)
        final = observations.get("final") if isinstance(observations, dict) else None
        if final is None:
            return None
        if isinstance(final, dict) and final_key in final and not hasattr(model, "actor_obs_groups"):
            return final[final_key].to(self.device)
        if isinstance(final, TensorDict):
            return model.get_observation(final).to(self.device)
        if isinstance(final, dict) and all(group in final for group in model_groups):
            final_td = TensorDict(final, batch_size=[self.replay_buffer.num_envs], device=self.device)
            return model.get_observation(final_td)
        return None

    def compile(self, mode: str | None = None) -> None:
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        self.target_critic = compile_model(self._raw_target_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "RGMTFastSAC":
        alg_class: type[RGMTFastSAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[RGMTFastSACActorModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[FastSACCriticModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = [
            "actor",
            "critic",
            cfg["actor"].get("history_obs_set", "proprio_history"),
            cfg["actor"].get("action_history_obs_set", "action_history"),
            cfg["actor"].get("command_obs_set", "command_window"),
        ]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        action_scale = cfg["actor"].pop("action_scale", 1.0)
        action_bias = cfg["actor"].pop("action_bias", 0.0)
        actor = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            action_scale=action_scale,
            action_bias=action_bias,
            **cfg["actor"],
        ).to(device)
        print(f"RGMT FastSAC Actor Model: {actor}")
        critic = critic_class(obs, cfg["obs_groups"], "critic", env.num_actions, **cfg["critic"]).to(device)
        print(f"FastSAC Critic Model: {critic}")
        target_critic = copy.deepcopy(critic)

        gamma = cfg["algorithm"].get("gamma", 0.99)
        rnd_cfg = cfg["algorithm"].pop("rnd_cfg", None)
        actor_obs = actor.get_observation(obs.to(device))
        replay_buffer = RGMTSACReplayBuffer(
            num_envs=env.num_envs,
            buffer_size=cfg["algorithm"].pop("buffer_size"),
            actor_obs=actor_obs,
            critic_obs_dim=critic.obs_dim,
            action_dim=env.num_actions,
            n_steps=cfg["algorithm"].pop("n_steps", 1),
            gamma=gamma,
            device=device,
        )
        alg = alg_class(
            actor,
            critic,
            target_critic,
            replay_buffer,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **cfg["algorithm"],
        )
        cfg["algorithm"]["rnd_cfg"] = rnd_cfg
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
