# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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


class RGMT(PPO):
    """PPO with RGMT actor construction."""

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
        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
