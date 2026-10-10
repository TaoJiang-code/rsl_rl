# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import SACReplayBuffer
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


def _concat_obs(obs: TensorDict, obs_groups: list[str]) -> torch.Tensor:
    return torch.cat([obs[obs_group] for obs_group in obs_groups], dim=-1)


def _make_holosoma_mlp(input_dim: int, hidden_dim: int, use_layer_norm: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
        nn.SiLU(),
        nn.Linear(hidden_dim // 2, hidden_dim // 4),
        nn.LayerNorm(hidden_dim // 4) if use_layer_norm else nn.Identity(),
        nn.SiLU(),
    )


class FastSACActorModel(nn.Module):
    """Holosoma-style tanh-Gaussian actor."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dim: int = 512,
        use_layer_norm: bool = True,
        use_tanh: bool = True,
        obs_normalization: bool = True,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        action_scale: float | list[float] | tuple[float, ...] = 1.0,
        action_bias: float | list[float] | tuple[float, ...] = 0.0,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.action_dim = output_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_tanh = use_tanh
        self.obs_normalization = obs_normalization

        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.net = _make_holosoma_mlp(self.obs_dim, hidden_dim, use_layer_norm)
        self.fc_mu = nn.Linear(hidden_dim // 4, output_dim)
        self.fc_logstd = nn.Linear(hidden_dim // 4, output_dim)
        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32).view(1, -1))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32).view(1, -1))
        if self.action_scale.shape[-1] == 1:
            self.action_scale = self.action_scale.expand(1, output_dim).clone()
        if self.action_bias.shape[-1] == 1:
            self.action_bias = self.action_bias.expand(1, output_dim).clone()

    def forward(self, obs: TensorDict | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_obs = self.get_observation(obs) if isinstance(obs, TensorDict) else obs
        return self.forward_normalized(self.normalize_observation(raw_obs, update=False))

    def forward_normalized(self, normalized_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(normalized_obs)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        if self.use_tanh:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = mean
        return action, mean, log_std

    def sample(self, obs: TensorDict | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_obs = self.get_observation(obs) if isinstance(obs, TensorDict) else obs
        return self.get_actions_and_log_probs_normalized(self.normalize_observation(raw_obs, update=False))

    def explore(self, obs: TensorDict | torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        raw_obs = self.get_observation(obs) if isinstance(obs, TensorDict) else obs
        action, mean, log_std = self.forward_normalized(self.normalize_observation(raw_obs, update=False))
        if deterministic:
            return action
        std = log_std.exp()
        raw_action = torch.distributions.Normal(mean, std).rsample()
        if self.use_tanh:
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        return raw_action

    def get_actions_and_log_probs_normalized(self, normalized_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self.forward_normalized(normalized_obs)
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

    def get_observation(self, obs: TensorDict) -> torch.Tensor:
        return _concat_obs(obs, self.obs_groups)

    def normalize_observation(self, obs: torch.Tensor, update: bool) -> torch.Tensor:
        if self.obs_normalization and update:
            self.obs_normalizer.update(obs)  # type: ignore
        return self.obs_normalizer(obs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def get_hidden_state(self) -> None:
        return None

    @property
    def output_std(self) -> torch.Tensor:
        return self.fc_logstd.weight.new_zeros(self.action_dim)

    def as_jit(self) -> nn.Module:
        return _TorchFastSACActorModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxFastSACActorModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for group in active_obs_groups:
            if len(obs[group].shape) != 2:
                raise ValueError(f"FastSACActorModel only supports 1D observations, got {obs[group].shape}.")
            obs_dim += obs[group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchFastSACActorModel(nn.Module):
    def __init__(self, model: FastSACActorModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.net = copy.deepcopy(model.net)
        self.fc_mu = copy.deepcopy(model.fc_mu)
        self.action_scale = copy.deepcopy(model.action_scale)
        self.action_bias = copy.deepcopy(model.action_bias)
        self.use_tanh = model.use_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        mean = self.fc_mu(self.net(x))
        if self.use_tanh:
            return torch.tanh(mean) * self.action_scale + self.action_bias
        return mean

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxFastSACActorModel(_TorchFastSACActorModel):
    is_recurrent: bool = False

    def __init__(self, model: FastSACActorModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _make_holosoma_mlp(obs_dim + action_dim, hidden_dim, use_layer_norm),
            nn.Linear(hidden_dim // 4, num_atoms),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, actions], dim=1))

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=q_support.device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        max_index = proj_dist.numel() - 1
        lower_indices = torch.clamp((lower + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper + offset).view(-1), 0, max_index)
        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class FastSACCriticModel(nn.Module):
    """Holosoma-style distributional Q ensemble."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        action_dim: int,
        hidden_dim: int = 512,
        use_layer_norm: bool = True,
        obs_normalization: bool = True,
        num_q_networks: int = 2,
        num_atoms: int = 51,
        v_min: float = -100.0,
        v_max: float = 100.0,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.action_dim = action_dim
        self.num_q_networks = num_q_networks
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.qnets = nn.ModuleList(
            [
                _DistributionalQNetwork(self.obs_dim, action_dim, num_atoms, v_min, v_max, hidden_dim, use_layer_norm)
                for _ in range(num_q_networks)
            ]
        )
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms))

    def forward(self, obs: TensorDict | torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        normalized_obs = self.normalize_observation(self.get_observation(obs), update=False) if isinstance(obs, TensorDict) else obs
        return torch.stack([qnet(normalized_obs, actions) for qnet in self.qnets], dim=0)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [qnet.projection(obs, actions, rewards, bootstrap, discount, self.q_support) for qnet in self.qnets],
            dim=0,
        )

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * self.q_support, dim=-1)

    def get_observation(self, obs: TensorDict) -> torch.Tensor:
        return _concat_obs(obs, self.obs_groups)

    def normalize_observation(self, obs: torch.Tensor, update: bool) -> torch.Tensor:
        if self.obs_normalization and update:
            self.obs_normalizer.update(obs)  # type: ignore
        return self.obs_normalizer(obs)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for group in active_obs_groups:
            if len(obs[group].shape) != 2:
                raise ValueError(f"FastSACCriticModel only supports 1D observations, got {obs[group].shape}.")
            obs_dim += obs[group].shape[-1]
        return active_obs_groups, obs_dim


class FastSAC:
    """Holosoma-style FastSAC adapted to rsl_rl."""

    def __init__(
        self,
        actor: FastSACActorModel,
        critic: FastSACCriticModel,
        target_critic: FastSACCriticModel,
        replay_buffer: SACReplayBuffer,
        batch_size: int = 4096,
        learning_starts: int = 10_000,
        num_updates: int = 1,
        policy_frequency: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_learning_rate: float = 3.0e-4,
        critic_learning_rate: float = 3.0e-4,
        alpha_learning_rate: float = 3.0e-4,
        target_entropy_ratio: float = 1.0,
        max_grad_norm: float = 1.0,
        optimizer: str = "adamw",
        use_autotune: bool = True,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_critic = target_critic.to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self._raw_target_critic = self.target_critic

        optimizer_class = resolve_optimizer(optimizer)
        self.actor_optimizer = optimizer_class(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optimizer_class(self.critic.parameters(), lr=critic_learning_rate)
        self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
        self.alpha_optimizer = optimizer_class([self.log_alpha], lr=alpha_learning_rate)

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.num_updates = num_updates
        self.policy_frequency = max(1, int(policy_frequency))
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.use_autotune = use_autotune
        self.target_entropy = -float(actor.action_dim) * target_entropy_ratio
        self.learning_rate = actor_learning_rate
        self.total_env_steps = 0
        self.global_step = 0

        self._transition_actor_obs: torch.Tensor | None = None
        self._transition_critic_obs: torch.Tensor | None = None
        self._transition_actions: torch.Tensor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: TensorDict, dones: torch.Tensor | None = None) -> torch.Tensor:
        """Sample exploratory actions from raw observations."""
        raw_actor_obs = self.actor.get_observation(obs)
        raw_critic_obs = self.critic.get_observation(obs)
        actions = self.actor.explore(raw_actor_obs, deterministic=False)
        self._transition_actor_obs = raw_actor_obs.detach()
        self._transition_critic_obs = raw_critic_obs.detach()
        self._transition_actions = actions.detach()
        del dones
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self._transition_actor_obs is None or self._transition_critic_obs is None or self._transition_actions is None:
            raise RuntimeError("FastSAC.process_env_step() called before act().")

        next_actor_obs = self.actor.get_observation(obs)
        next_critic_obs = self.critic.get_observation(obs)
        truncations = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.long)).to(self.device)
        next_actor_obs = self._replace_timeouts_with_final_obs(
            next_actor_obs, extras, truncations, "actor_obs", self.actor
        )
        next_critic_obs = self._replace_timeouts_with_final_obs(
            next_critic_obs, extras, truncations, "critic_obs", self.critic
        )
        transition = TensorDict(
            {
                "observations": self._transition_actor_obs,
                "actions": self._transition_actions,
                "next": {
                    "observations": next_actor_obs,
                    "rewards": rewards.detach(),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
                "critic_observations": self._transition_critic_obs,
            },
            batch_size=(self.replay_buffer.num_envs,),
            device=self.device,
        )
        transition["next"]["critic_observations"] = next_critic_obs
        self.replay_buffer.extend(transition)
        self.total_env_steps += rewards.numel()
        self.global_step += 1
        self._transition_actor_obs = None
        self._transition_critic_obs = None
        self._transition_actions = None
        self.actor.reset(dones)

    def update(self, batch_size_per_env: int | None = None) -> dict[str, float]:
        if self.global_step <= self.learning_starts:
            return {}
        if batch_size_per_env is None:
            batch_size_per_env = max(self.batch_size // self.replay_buffer.num_envs // self.gpu_world_size, 1)
        if not self.replay_buffer.can_sample(batch_size_per_env):
            return {}

        large_batch = self.replay_buffer.sample(batch_size_per_env * self.num_updates)
        prepared_batches = self._sample_and_prepare_batches(large_batch, batch_size_per_env)

        metrics_sum: dict[str, float] = {}
        actor_metrics = {
            "actor_grad_norm": 0.0,
            "actor_loss": 0.0,
            "policy_entropy": 0.0,
            "action_std": 0.0,
        }
        for i, batch in enumerate(prepared_batches):
            main_metrics = self._update_main(batch)
            if self.num_updates > 1:
                if i % self.policy_frequency == 1:
                    actor_metrics = self._update_policy(batch)
            elif self.global_step % self.policy_frequency == 0:
                actor_metrics = self._update_policy(batch)
            self._soft_update_target_critic()

            current_metrics = {**main_metrics, **actor_metrics, "alpha_value": float(self.alpha.detach().cpu())}
            for key, value in current_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value / self.num_updates
        return metrics_sum

    def _sample_and_prepare_batches(self, large_data: TensorDict, batch_size_per_env: int) -> list[TensorDict]:
        large_data["observations"] = self.actor.normalize_observation(large_data["observations"], update=True)
        large_data["next"]["observations"] = self.actor.normalize_observation(
            large_data["next"]["observations"], update=True
        )
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
                batch_size=samples_per_update,
            )
            batch["next"]["critic_observations"] = large_data["next"]["critic_observations"][start:end]
            batches.append(batch)
        return batches

    def _update_main(self, data: TensorDict) -> dict[str, float]:
        next_observations = data["next"]["observations"]
        critic_observations = data["critic_observations"]
        next_critic_observations = data["next"]["critic_observations"]
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        with torch.no_grad():
            next_state_actions, next_state_log_probs = self.actor.get_actions_and_log_probs_normalized(next_observations)
            discount = self.gamma ** data["next"]["effective_n_steps"]
            target_distributions = self.target_critic.projection(
                next_critic_observations,
                next_state_actions,
                rewards - discount * bootstrap * self.alpha.detach() * next_state_log_probs,
                bootstrap,
                discount,
            )
            target_values = self.target_critic.get_value(target_distributions)
            target_value_max = target_values.max()
            target_value_min = target_values.min()

        q_outputs = self.critic(critic_observations, actions)
        critic_log_probs = F.log_softmax(q_outputs, dim=-1)
        critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
        qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        self.critic_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self._all_reduce_model_grads(self.critic)
        critic_grad_norm = self._clip_grad_norm(self.critic.parameters())
        self.critic_optimizer.step()

        alpha_loss = qf_loss.new_tensor(0.0)
        if self.use_autotune:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (-self.alpha * (next_state_log_probs.detach() + self.target_entropy)).mean()
            alpha_loss.backward()
            self._all_reduce_alpha_grad()
            self.alpha_optimizer.step()

        return {
            "buffer_rewards": float(rewards.mean().detach().cpu()),
            "critic_grad_norm": float(critic_grad_norm.detach().cpu()),
            "qf_loss": float(qf_loss.detach().cpu()),
            "qf_max": float(target_value_max.detach().cpu()),
            "qf_min": float(target_value_min.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
        }

    def _update_policy(self, data: TensorDict) -> dict[str, float]:
        critic_observations = data["critic_observations"]
        actions, log_probs = self.actor.get_actions_and_log_probs_normalized(data["observations"])
        with torch.no_grad():
            _, _, log_std = self.actor.forward_normalized(data["observations"])
            action_std = log_std.exp().mean()
            policy_entropy = -log_probs.mean()

        q_outputs = self.critic(critic_observations, actions)
        q_probs = F.softmax(q_outputs, dim=-1)
        q_values = self.critic.get_value(q_probs)
        qf_value = q_values.mean(dim=0)
        actor_loss = (self.alpha.detach() * log_probs - qf_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._all_reduce_model_grads(self.actor)
        actor_grad_norm = self._clip_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        return {
            "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "policy_entropy": float(policy_entropy.detach().cpu()),
            "action_std": float(action_std.detach().cpu()),
        }

    def _soft_update_target_critic(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def _clip_grad_norm(self, parameters) -> torch.Tensor:
        if self.max_grad_norm > 0:
            return nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        return torch.tensor(0.0, device=self.device)

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for param in model.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.copy_(flat[offset : offset + numel].view_as(param.grad))
                offset += numel

    def _all_reduce_alpha_grad(self) -> None:
        if self.is_multi_gpu and self.log_alpha.grad is not None:
            torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
            self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

    def _replace_timeouts_with_final_obs(
        self,
        next_obs: torch.Tensor,
        extras: dict,
        truncations: torch.Tensor,
        final_key: str,
        model: FastSACActorModel | FastSACCriticModel,
    ) -> torch.Tensor:
        final_obs = self._extract_final_obs(extras, final_key, model)
        if final_obs is None:
            return next_obs
        mask = truncations.to(dtype=torch.bool, device=next_obs.device).unsqueeze(-1)
        return torch.where(mask, final_obs.to(next_obs.device), next_obs)

    def _extract_final_obs(
        self,
        extras: dict,
        final_key: str,
        model: FastSACActorModel | FastSACCriticModel,
    ) -> torch.Tensor | None:
        observations = extras.get("observations", {})
        top_level_final = extras.get("final_observations")
        if isinstance(top_level_final, dict) and all(group in top_level_final for group in model.obs_groups):
            final_td = TensorDict(top_level_final, batch_size=[self.replay_buffer.num_envs], device=self.device)
            return model.get_observation(final_td)
        final = observations.get("final") if isinstance(observations, dict) else None
        if final is None:
            return None
        if isinstance(final, dict) and final_key in final:
            return final[final_key].to(self.device)
        if isinstance(final, TensorDict):
            return model.get_observation(final).to(self.device)
        if isinstance(final, dict) and all(group in final for group in model.obs_groups):
            final_td = TensorDict(final, batch_size=[self.replay_buffer.num_envs], device=self.device)
            return model.get_observation(final_td)
        return None

    def compute_returns(self, obs: TensorDict) -> None:
        del obs

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()
        self.target_critic.eval()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def get_policy(self) -> FastSACActorModel:
        return self._raw_actor

    def save(self) -> dict:
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "target_critic_state_dict": self._raw_target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "total_env_steps": self.total_env_steps,
            "global_step": self.global_step,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "target_critic": True,
                "optimizer": True,
                "alpha": True,
                "iteration": True,
            }
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("target_critic"):
            self._raw_target_critic.load_state_dict(loaded_dict["target_critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
        if load_cfg.get("alpha"):
            self.log_alpha.data.copy_(loaded_dict["log_alpha"].to(self.device))
        self.total_env_steps = int(loaded_dict.get("total_env_steps", self.total_env_steps))
        self.global_step = int(loaded_dict.get("global_step", self.global_step))
        return load_cfg.get("iteration", False)

    def compile(self, mode: str | None = None) -> None:
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        self.target_critic = compile_model(self._raw_target_critic, mode)  # type: ignore

    def broadcast_parameters(self) -> None:
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict(), self._raw_target_critic.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        self._raw_target_critic.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        all_params = list(chain(self.actor.parameters(), self.critic.parameters(), [self.log_alpha]))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= torch.distributed.get_world_size()
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "FastSAC":
        alg_class: type[FastSAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[FastSACActorModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[FastSACCriticModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

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
        )
        print(f"FastSAC Actor Model: {actor}")
        critic = critic_class(obs, cfg["obs_groups"], "critic", env.num_actions, **cfg["critic"])
        print(f"FastSAC Critic Model: {critic}")
        target_critic = copy.deepcopy(critic)

        gamma = cfg["algorithm"].get("gamma", 0.99)
        replay_buffer = SACReplayBuffer(
            num_envs=env.num_envs,
            actor_obs_dim=actor.obs_dim,
            critic_obs_dim=critic.obs_dim,
            action_dim=env.num_actions,
            buffer_size=cfg["algorithm"].pop("buffer_size"),
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
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
