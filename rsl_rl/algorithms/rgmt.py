# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import Distribution, EmpiricalNormalization, HiddenState, MLP
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, unpad_trajectories


class _SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding for short RGMT temporal windows."""

    def __init__(self, max_length: int, embedding_dim: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        encoding = torch.zeros(max_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class _CausalTransformerBlock(nn.Module):
    """One causal Transformer block matching the RGMT history encoder formula."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, embedding_dim, [feedforward_dim], activation)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attention_input = self.attention_norm(x)
        attention_output, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attention_output
        x = x + self.mlp(self.mlp_norm(x))
        return self.output_norm(x)


class _CommandCrossAttentionBlock(nn.Module):
    """Dynamics-conditioned command aggregation block from the RGMT paper."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, embedding_dim, [feedforward_dim], activation)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, query: torch.Tensor, command_tokens: torch.Tensor) -> torch.Tensor:
        query = query.unsqueeze(1)
        attention_output, _ = self.attention(
            self.query_norm(query),
            command_tokens,
            command_tokens,
            need_weights=False,
        )
        latent = query + attention_output
        latent = latent + self.mlp(self.mlp_norm(latent))
        return self.output_norm(latent).squeeze(1)


class RGMTActorModel(nn.Module):
    """RGMT actor with proprioceptive history encoding and command cross attention.

    The actor consumes:
    - ``actor`` obs set: current policy observation.
    - ``history_obs_set``: 10-step proprioceptive history.
    - ``command_obs_set``: reference command window ``[v_ref, w_ref, g_ref, q_ref]``.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        history_obs_set: str = "proprio_history",
        command_obs_set: str = "command_window",
        history_length: int = 10,
        command_window_size: int = 11,
        embedding_dim: int = 128,
        history_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        command_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        dynamics_hidden_dims: tuple[int, ...] | list[int] = (128,),
        transformer_num_layers: int = 1,
        transformer_num_heads: int = 4,
        transformer_feedforward_dim: int = 512,
        cross_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.history_obs_groups = obs_groups[history_obs_set]
        self.command_obs_groups = obs_groups[command_obs_set]
        self.history_length = history_length
        self.command_window_size = command_window_size
        self.embedding_dim = embedding_dim

        self.obs_dim = self._sum_flat_obs_dim(obs, self.obs_groups)
        self.history_step_dim = self._infer_step_dim(obs, self.history_obs_groups, history_length)
        self.command_step_dim = self._infer_step_dim(obs, self.command_obs_groups, command_window_size)

        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.history_normalizer = EmpiricalNormalization(self.history_step_dim) if obs_normalization else nn.Identity()
        self.command_normalizer = EmpiricalNormalization(self.command_step_dim) if obs_normalization else nn.Identity()

        dist_cfg = distribution_cfg
        if dist_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.history_step_encoder = MLP(self.history_step_dim, embedding_dim, history_hidden_dims, activation)
        self.history_position_encoding = _SinusoidalPositionEncoding(history_length, embedding_dim)
        self.history_blocks = nn.ModuleList(
            [
                _CausalTransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=transformer_num_heads,
                    feedforward_dim=transformer_feedforward_dim,
                )
                for _ in range(transformer_num_layers)
            ]
        )
        self.dynamics_query_encoder = MLP(embedding_dim, embedding_dim, dynamics_hidden_dims, activation)
        self.command_step_encoder = MLP(self.command_step_dim, embedding_dim, command_hidden_dims, activation)
        self.command_position_encoding = _SinusoidalPositionEncoding(command_window_size, embedding_dim)
        self.command_block = _CommandCrossAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=cross_attention_heads,
            feedforward_dim=transformer_feedforward_dim,
        )

        actor_input_dim = self.obs_dim + embedding_dim + embedding_dim
        self.mlp = MLP(actor_input_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        policy_obs = self.obs_normalizer(self._flatten_obs_groups(obs, self.obs_groups))
        history = self.history_normalizer(self._sequence_obs_groups(obs, self.history_obs_groups, self.history_length))
        command = self.command_normalizer(self._sequence_obs_groups(obs, self.command_obs_groups, self.command_window_size))

        history_tokens = self.history_position_encoding(self.history_step_encoder(history))
        causal_mask = torch.triu(
            torch.ones(self.history_length, self.history_length, device=history_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.history_blocks:
            history_tokens = block(history_tokens, causal_mask)
        dynamics_latent = torch.max(history_tokens, dim=1).values

        command_query = self.dynamics_query_encoder(dynamics_latent)
        command_tokens = self.command_position_encoding(self.command_step_encoder(command))
        command_latent = self.command_block(command_query, command_tokens)
        return torch.cat((policy_obs, dynamics_latent, command_latent), dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(self._flatten_obs_groups(obs, self.obs_groups))  # type: ignore
            self.history_normalizer.update(  # type: ignore
                self._sequence_obs_groups(obs, self.history_obs_groups, self.history_length).reshape(-1, self.history_step_dim)
            )
            self.command_normalizer.update(  # type: ignore
                self._sequence_obs_groups(obs, self.command_obs_groups, self.command_window_size).reshape(
                    -1,
                    self.command_step_dim,
                )
            )

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-friendly deterministic RGMT actor."""
        return _TorchRGMTActorModel(self)

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore

    @staticmethod
    def _sum_flat_obs_dim(obs: TensorDict, groups: list[str]) -> int:
        return sum(math.prod(obs[group].shape[1:]) for group in groups)

    @staticmethod
    def _flatten_obs_groups(obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[group].reshape(obs.batch_size[0], -1) for group in groups], dim=-1)

    @staticmethod
    def _infer_step_dim(obs: TensorDict, groups: list[str], sequence_length: int) -> int:
        dim = 0
        for group in groups:
            shape = obs[group].shape
            if len(shape) == 3:
                dim += shape[-1]
            elif len(shape) == 2:
                if shape[-1] % sequence_length != 0:
                    raise ValueError(
                        f"Observation '{group}' with shape {shape} cannot be reshaped into sequence length "
                        f"{sequence_length}."
                    )
                dim += shape[-1] // sequence_length
            else:
                raise ValueError(f"RGMTActorModel only supports 1D or 2D sequence obs, got {shape} for '{group}'.")
        return dim

    @staticmethod
    def _sequence_obs_groups(obs: TensorDict, groups: list[str], sequence_length: int) -> torch.Tensor:
        sequences = []
        batch_size = obs.batch_size[0]
        for group in groups:
            value = obs[group]
            if len(value.shape) == 3:
                sequences.append(value)
            else:
                sequences.append(value.reshape(batch_size, sequence_length, -1))
        return torch.cat(sequences, dim=-1)


class _TorchRGMTActorModel(nn.Module):
    """Exportable RGMT actor.

    Forward inputs:
        policy_obs: concatenated current actor observations.
        history_obs: ``[B, history_length, history_step_dim]``.
        command_obs: ``[B, command_window_size, command_step_dim]``.
    """

    def __init__(self, model: RGMTActorModel) -> None:
        super().__init__()
        self.history_length = model.history_length
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_normalizer = copy.deepcopy(model.history_normalizer)
        self.command_normalizer = copy.deepcopy(model.command_normalizer)
        self.history_step_encoder = copy.deepcopy(model.history_step_encoder)
        self.history_position_encoding = copy.deepcopy(model.history_position_encoding)
        self.history_blocks = copy.deepcopy(model.history_blocks)
        self.dynamics_query_encoder = copy.deepcopy(model.dynamics_query_encoder)
        self.command_step_encoder = copy.deepcopy(model.command_step_encoder)
        self.command_position_encoding = copy.deepcopy(model.command_position_encoding)
        self.command_block = copy.deepcopy(model.command_block)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, policy_obs: torch.Tensor, history_obs: torch.Tensor, command_obs: torch.Tensor) -> torch.Tensor:
        policy_obs = self.obs_normalizer(policy_obs)
        history = self.history_normalizer(history_obs)
        command = self.command_normalizer(command_obs)

        history_tokens = self.history_position_encoding(self.history_step_encoder(history))
        causal_mask = torch.triu(
            torch.ones(self.history_length, self.history_length, device=history_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.history_blocks:
            history_tokens = block(history_tokens, causal_mask)
        dynamics_latent = torch.max(history_tokens, dim=1).values

        command_query = self.dynamics_query_encoder(dynamics_latent)
        command_tokens = self.command_position_encoding(self.command_step_encoder(command))
        command_latent = self.command_block(command_query, command_tokens)
        out = self.mlp(torch.cat((policy_obs, dynamics_latent, command_latent), dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class RGMT(PPO):
    """PPO with RGMT actor construction and an on-policy failure-risk predictor."""

    def __init__(
        self,
        actor: RGMTActorModel,
        critic: MLPModel,
        storage: RolloutStorage,
        failure_predictor_obs_groups: list[str] | tuple[str, ...],
        failure_predictor_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        failure_predictor_activation: str = "elu",
        failure_predictor_learning_rate: float = 1.0e-3,
        failure_predictor_horizon_steps: int = 12,
        failure_predictor_num_epochs: int = 1,
        failure_predictor_num_mini_batches: int = 4,
        failure_reward_weight: float = 0.0,
        failure_reward_threshold: float = 0.3,
        device: str = "cpu",
        **ppo_kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, device=device, **ppo_kwargs)
        self.failure_predictor_obs_groups = list(failure_predictor_obs_groups)
        self.failure_predictor_horizon_steps = failure_predictor_horizon_steps
        self.failure_predictor_num_epochs = failure_predictor_num_epochs
        self.failure_predictor_num_mini_batches = failure_predictor_num_mini_batches
        self.failure_reward_weight = failure_reward_weight
        self.failure_reward_threshold = failure_reward_threshold
        self.failure_dones = torch.zeros(
            self.storage.num_transitions_per_env,
            self.storage.num_envs,
            1,
            device=self.device,
        )

        obs_dim = sum(math.prod(self.storage.observations[group].shape[2:]) for group in self.failure_predictor_obs_groups)
        input_dim = obs_dim + math.prod(self.storage.actions.shape[2:])
        self.failure_predictor = MLP(
            input_dim,
            1,
            failure_predictor_hidden_dims,
            failure_predictor_activation,
        ).to(self.device)
        self.failure_predictor_optimizer = torch.optim.Adam(
            self.failure_predictor.parameters(),
            lr=failure_predictor_learning_rate,
        )
        self.failure_predictor_loss = nn.BCEWithLogitsLoss()
        self.failure_reward_sum = 0.0
        self.failure_reward_count = 0

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        failure_dones = dones.float().view(-1, 1)
        if "time_outs" in extras:
            failure_dones = failure_dones * (1.0 - extras["time_outs"].to(self.device).float().view(-1, 1))
        self.failure_dones[self.storage.step].copy_(failure_dones)

        if self.failure_reward_weight != 0.0:
            failure_reward = self._compute_failure_reward(self.transition.observations, self.transition.actions)
            rewards = rewards + (failure_reward.squeeze(-1) if rewards.ndim == 1 else failure_reward)
            self.failure_reward_sum += failure_reward.mean().item()
            self.failure_reward_count += 1
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        failure_loss, failure_target, failure_pred = self._update_failure_predictor()
        loss_dict = super().update()
        loss_dict["failure_predictor"] = failure_loss
        loss_dict["failure_target"] = failure_target
        loss_dict["failure_pred"] = failure_pred
        loss_dict["failure_reward"] = self.failure_reward_sum / max(self.failure_reward_count, 1)
        self.failure_reward_sum = 0.0
        self.failure_reward_count = 0
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.failure_predictor.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.failure_predictor.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["failure_predictor_state_dict"] = self.failure_predictor.state_dict()
        saved_dict["failure_predictor_optimizer_state_dict"] = self.failure_predictor_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "failure_predictor_state_dict" in loaded_dict:
            self.failure_predictor.load_state_dict(loaded_dict["failure_predictor_state_dict"], strict=strict)
        if "failure_predictor_optimizer_state_dict" in loaded_dict and (load_cfg is None or load_cfg.get("optimizer", True)):
            self.failure_predictor_optimizer.load_state_dict(loaded_dict["failure_predictor_optimizer_state_dict"])
        return load_iteration

    def _update_failure_predictor(self) -> tuple[float, float, float]:
        inputs = self._failure_predictor_inputs()
        targets = self._failure_predictor_targets()
        batch_size = inputs.shape[0]
        num_mini_batches = min(self.failure_predictor_num_mini_batches, batch_size)
        mini_batch_size = batch_size // num_mini_batches

        mean_loss = 0.0
        num_updates = 0
        for _ in range(self.failure_predictor_num_epochs):
            indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
            for mini_batch_idx in range(num_mini_batches):
                start = mini_batch_idx * mini_batch_size
                stop = (mini_batch_idx + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                logits = self.failure_predictor(inputs[batch_idx])
                loss = self.failure_predictor_loss(logits, targets[batch_idx])

                self.failure_predictor_optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self._reduce_failure_predictor_parameters()
                nn.utils.clip_grad_norm_(self.failure_predictor.parameters(), self.max_grad_norm)
                self.failure_predictor_optimizer.step()

                mean_loss += loss.item()
                num_updates += 1

        with torch.inference_mode():
            pred = torch.sigmoid(self.failure_predictor(inputs)).mean().item()
        return mean_loss / max(num_updates, 1), targets.mean().item(), pred

    def _failure_predictor_inputs(self) -> torch.Tensor:
        observations = [
            self.storage.observations[group].reshape(
                self.storage.num_transitions_per_env,
                self.storage.num_envs,
                -1,
            )
            for group in self.failure_predictor_obs_groups
        ]
        obs = torch.cat(observations, dim=-1)
        inputs = torch.cat((obs, self.storage.actions), dim=-1)
        return inputs.flatten(0, 1).detach()

    def _compute_failure_reward(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        predictor_input = self._make_failure_predictor_input(obs, actions)
        risk = torch.sigmoid(self.failure_predictor(predictor_input))
        penalty = torch.clamp(risk - self.failure_reward_threshold, min=0.0)
        return self.failure_reward_weight * penalty

    def _make_failure_predictor_input(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        observations = [obs[group].reshape(obs.batch_size[0], -1) for group in self.failure_predictor_obs_groups]
        return torch.cat((*observations, actions), dim=-1).detach()

    def _failure_predictor_targets(self) -> torch.Tensor:
        dones = self.failure_dones
        horizon = max(1, min(self.failure_predictor_horizon_steps, self.storage.num_transitions_per_env))
        targets = torch.zeros_like(dones)
        for offset in range(horizon):
            risk = float(horizon - offset) / float(horizon)
            targets[: self.storage.num_transitions_per_env - offset] = torch.maximum(
                targets[: self.storage.num_transitions_per_env - offset],
                dones[offset:] * risk,
            )
        return targets.flatten(0, 1).detach()

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        model_params = [self.failure_predictor.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.failure_predictor.load_state_dict(model_params[0])

    def _reduce_failure_predictor_parameters(self) -> None:
        params = list(self.failure_predictor.parameters())
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> RGMT:
        cfg.setdefault("obs_groups", {})
        cfg.setdefault("multi_gpu", None)

        alg_class: type[RGMT] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[RGMTActorModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = [
            "actor",
            "critic",
            cfg["actor"].get("history_obs_set", "proprio_history"),
            cfg["actor"].get("command_obs_set", "command_window"),
        ]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"RGMT Actor Model: {actor}")
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        failure_predictor_obs_sets = cfg["algorithm"].pop("failure_predictor_obs_sets", ("actor", "command_window"))
        failure_predictor_obs_groups = []
        for obs_set in failure_predictor_obs_sets:
            if obs_set not in cfg["obs_groups"]:
                raise KeyError(f"cfg['obs_groups']['{obs_set}'] is required for the failure predictor.")
            failure_predictor_obs_groups.extend(cfg["obs_groups"][obs_set])
        alg = alg_class(
            actor,
            critic,
            storage,
            failure_predictor_obs_groups=failure_predictor_obs_groups,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
