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


class _FiniteScalarQuantizer(nn.Module):
    """Straight-through finite scalar quantization for the command latent."""

    def __init__(self, embedding_dim: int, num_tokens: int = 2, token_dim: int = 32, levels: int = 8) -> None:
        super().__init__()
        if num_tokens * token_dim != embedding_dim:
            raise ValueError(
                f"FSQ expects num_tokens * token_dim == embedding_dim, got {num_tokens} * {token_dim} != "
                f"{embedding_dim}."
            )
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.step = 2.0 / float(levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(x)
        tokens = bounded.reshape(*bounded.shape[:-1], self.num_tokens, self.token_dim)
        quantized = torch.round((tokens + 1.0) / self.step) * self.step - 1.0
        quantized = tokens + (quantized - tokens).detach()
        return quantized.reshape_as(x)


class RGMTActorModel(nn.Module):
    """RGMT actor with proprioceptive history encoding and command cross attention.

    The actor consumes:
    - ``actor`` obs set: current policy observation.
    - ``history_obs_set``: proprioceptive state history.
    - ``action_history_obs_set``: previous-action history.
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
        action_history_obs_set: str = "action_history",
        command_obs_set: str = "command_window",
        history_length: int = 10,
        command_window_size: int = 11,
        embedding_dim: int = 128,
        history_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        state_hidden_dims: tuple[int, ...] | list[int] | None = None,
        action_hidden_dims: tuple[int, ...] | list[int] = (64,),
        command_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        dynamics_hidden_dims: tuple[int, ...] | list[int] = (128,),
        transformer_num_layers: int = 1,
        transformer_num_heads: int = 4,
        transformer_feedforward_dim: int = 512,
        cross_attention_heads: int = 4,
        use_fsq: bool = True,
        fsq_num_tokens: int = 2,
        fsq_token_dim: int = 32,
        fsq_levels: int = 8,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.state_history_obs_groups = obs_groups[history_obs_set]
        self.action_history_obs_groups = obs_groups[action_history_obs_set]
        self.command_obs_groups = obs_groups[command_obs_set]
        self.history_length = history_length
        self.command_window_size = command_window_size
        self.embedding_dim = embedding_dim

        self.obs_dim = self._sum_flat_obs_dim(obs, self.obs_groups)
        self.state_step_dim = self._infer_step_dim(obs, self.state_history_obs_groups, history_length)
        self.action_step_dim = self._infer_step_dim(obs, self.action_history_obs_groups, history_length)
        self.command_step_dim = self._infer_step_dim(obs, self.command_obs_groups, command_window_size)

        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.state_normalizer = EmpiricalNormalization(self.state_step_dim) if obs_normalization else nn.Identity()
        self.action_normalizer = EmpiricalNormalization(self.action_step_dim) if obs_normalization else nn.Identity()
        self.command_normalizer = EmpiricalNormalization(self.command_step_dim) if obs_normalization else nn.Identity()

        dist_cfg = distribution_cfg
        if dist_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        if state_hidden_dims is None:
            state_hidden_dims = history_hidden_dims
        self.state_step_encoder = MLP(self.state_step_dim, embedding_dim, state_hidden_dims, activation)
        self.action_step_encoder = MLP(self.action_step_dim, embedding_dim, action_hidden_dims, activation)
        self.state_encoder_norm = nn.LayerNorm(embedding_dim)
        self.action_encoder_norm = nn.LayerNorm(embedding_dim)
        self.history_position_encoding = _SinusoidalPositionEncoding(history_length * 2, embedding_dim)
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
        self.command_encoder_norm = nn.LayerNorm(embedding_dim)
        self.command_position_encoding = _SinusoidalPositionEncoding(command_window_size, embedding_dim)
        self.command_block = _CommandCrossAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=cross_attention_heads,
            feedforward_dim=transformer_feedforward_dim,
        )
        self.command_quantizer = (
            _FiniteScalarQuantizer(
                embedding_dim=embedding_dim,
                num_tokens=fsq_num_tokens,
                token_dim=fsq_token_dim,
                levels=fsq_levels,
            )
            if use_fsq
            else nn.Identity()
        )

        actor_input_dim = self.obs_dim + embedding_dim
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
        state_history = self.state_normalizer(
            self._sequence_obs_groups(obs, self.state_history_obs_groups, self.history_length)
        )
        action_history = self.action_normalizer(
            self._sequence_obs_groups(obs, self.action_history_obs_groups, self.history_length)
        )
        command = self.command_normalizer(self._sequence_obs_groups(obs, self.command_obs_groups, self.command_window_size))

        state_tokens = self.state_encoder_norm(self.state_step_encoder(state_history))
        action_tokens = self.action_encoder_norm(self.action_step_encoder(action_history))
        history_tokens = torch.stack((action_tokens, state_tokens), dim=2).reshape(
            state_tokens.shape[0],
            self.history_length * 2,
            self.embedding_dim,
        )
        history_tokens = self.history_position_encoding(history_tokens)
        causal_mask = torch.triu(
            torch.ones(self.history_length * 2, self.history_length * 2, device=history_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.history_blocks:
            history_tokens = block(history_tokens, causal_mask)
        dynamics_latent = torch.max(history_tokens, dim=1).values

        command_query = self.dynamics_query_encoder(dynamics_latent)
        command_tokens = self.command_position_encoding(self.command_encoder_norm(self.command_step_encoder(command)))
        command_latent = self.command_block(command_query, command_tokens)
        command_latent = self.command_quantizer(command_latent)
        return torch.cat((policy_obs, command_latent), dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(self._flatten_obs_groups(obs, self.obs_groups))  # type: ignore
            self.state_normalizer.update(  # type: ignore
                self._sequence_obs_groups(obs, self.state_history_obs_groups, self.history_length).reshape(
                    -1,
                    self.state_step_dim,
                )
            )
            self.action_normalizer.update(  # type: ignore
                self._sequence_obs_groups(obs, self.action_history_obs_groups, self.history_length).reshape(
                    -1,
                    self.action_step_dim,
                )
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
        state_history_obs: ``[B, history_length, state_step_dim]``.
        action_history_obs: ``[B, history_length, action_step_dim]``.
        command_obs: ``[B, command_window_size, command_step_dim]``.
    """


    def __init__(self, model: RGMTActorModel) -> None:
        super().__init__()
        self.history_length = model.history_length
        self.embedding_dim = model.embedding_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.state_normalizer = copy.deepcopy(model.state_normalizer)
        self.action_normalizer = copy.deepcopy(model.action_normalizer)
        self.command_normalizer = copy.deepcopy(model.command_normalizer)
        self.state_step_encoder = copy.deepcopy(model.state_step_encoder)
        self.action_step_encoder = copy.deepcopy(model.action_step_encoder)
        self.state_encoder_norm = copy.deepcopy(model.state_encoder_norm)
        self.action_encoder_norm = copy.deepcopy(model.action_encoder_norm)
        self.history_position_encoding = copy.deepcopy(model.history_position_encoding)
        self.history_blocks = copy.deepcopy(model.history_blocks)
        self.dynamics_query_encoder = copy.deepcopy(model.dynamics_query_encoder)
        self.command_step_encoder = copy.deepcopy(model.command_step_encoder)
        self.command_encoder_norm = copy.deepcopy(model.command_encoder_norm)
        self.command_position_encoding = copy.deepcopy(model.command_position_encoding)
        self.command_block = copy.deepcopy(model.command_block)
        self.command_quantizer = copy.deepcopy(model.command_quantizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(
        self,
        policy_obs: torch.Tensor,
        state_history_obs: torch.Tensor,
        action_history_obs: torch.Tensor,
        command_obs: torch.Tensor,
    ) -> torch.Tensor:
        policy_obs = self.obs_normalizer(policy_obs)
        state_history = self.state_normalizer(state_history_obs)
        action_history = self.action_normalizer(action_history_obs)
        command = self.command_normalizer(command_obs)

        state_tokens = self.state_encoder_norm(self.state_step_encoder(state_history))
        action_tokens = self.action_encoder_norm(self.action_step_encoder(action_history))
        history_tokens = torch.stack((action_tokens, state_tokens), dim=2).reshape(
            state_tokens.shape[0],
            self.history_length * 2,
            self.embedding_dim,
        )
        history_tokens = self.history_position_encoding(history_tokens)
        causal_mask = torch.triu(
            torch.ones(self.history_length * 2, self.history_length * 2, device=history_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.history_blocks:
            history_tokens = block(history_tokens, causal_mask)
        dynamics_latent = torch.max(history_tokens, dim=1).values

        command_query = self.dynamics_query_encoder(dynamics_latent)
        command_tokens = self.command_position_encoding(self.command_encoder_norm(self.command_step_encoder(command)))
        command_latent = self.command_block(command_query, command_tokens)
        command_latent = self.command_quantizer(command_latent)
        out = self.mlp(torch.cat((policy_obs, command_latent), dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


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
            cfg["actor"].get("action_history_obs_set", "action_history"),
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
