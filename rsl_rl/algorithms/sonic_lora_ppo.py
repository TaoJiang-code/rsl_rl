# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Any2Any-style Sonic LoRA fine-tuning without depending on the Sonic repo.

The GlushLab environment handles Any2Any kinematic alignment:

    target robot obs/action <-> Sonic/G1 semantic space

This file handles the dynamics adaptation stage:

    frozen Sonic-style policy + LoRA residuals + PPO

Only the G1 motion encoder path is implemented here. Teleop/SMPL branches from
the Sonic release are intentionally not constructed because this fine-tuning
environment only emits motion-data tokenizer observations.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any
import pickle
import re
import sys
import types

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from vector_quantize_pytorch import FSQ

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, unpad_trajectories

from .ppo import PPO


def _activation(name: str) -> nn.Module:
    if not hasattr(nn, name):
        raise ValueError(f"Unsupported Sonic activation: {name}")
    return getattr(nn, name)()


def _make_sonic_mlp(input_dim: int, output_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_dims, output_dim]
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank update."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.base = linear
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False)

        for param in self.base.parameters():
            param.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def apply_lora_to_linear_layers(
    module: nn.Module,
    target_module_names: tuple[str, ...] | list[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Replace selected ``nn.Linear`` children with ``LoRALinear`` modules."""

    replaced: list[str] = []
    targets = tuple(target_module_names)
    for module_name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if not _matches_lora_target(module_name, targets):
            continue
        parent, child_name = _resolve_parent_module(module, module_name)
        setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(module_name)
    if not replaced:
        raise ValueError(f"No Linear layers matched LoRA targets: {targets}.")
    return replaced


def freeze_non_lora_parameters(module: nn.Module, train_distribution: bool = True) -> None:
    """Freeze everything except LoRA weights and optionally action-noise parameters."""

    for name, param in module.named_parameters():
        trainable = "lora_a" in name or "lora_b" in name
        if train_distribution and name in ("std", "log_std"):
            trainable = True
        param.requires_grad_(trainable)


def _matches_lora_target(module_name: str, targets: tuple[str, ...]) -> bool:
    return any(module_name == target or module_name.endswith(target) or target in module_name for target in targets)


def _resolve_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _flat_dim(shape: torch.Size | tuple[int, ...]) -> int:
    return int(torch.tensor(tuple(shape)).prod().item())


def _obs_dim(obs: TensorDict, key: str) -> int:
    value = obs[key]
    if isinstance(value, TensorDict):
        return sum(_flat_dim(term.shape[1:]) for term in value.values())
    return value.shape[-1]


def _tokenizer_dims(
    obs: TensorDict,
    tokenizer_obs_names: tuple[str, ...],
    tokenizer_obs_dims: dict[str, tuple[int, ...]] | None,
) -> dict[str, tuple[int, ...]]:
    if tokenizer_obs_dims is not None:
        return {name: tuple(dims) for name, dims in tokenizer_obs_dims.items()}
    tokenizer = obs["tokenizer"]
    if not isinstance(tokenizer, TensorDict):
        raise ValueError("tokenizer_obs_dims must be set when obs['tokenizer'] is already concatenated.")
    return {name: tuple(tokenizer[name].shape[1:]) for name in tokenizer_obs_names}


def _prepare_step_obs(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(1)
    return value


def _flatten_tokenizer_obs(tokenizer: TensorDict | torch.Tensor, tokenizer_obs_names: tuple[str, ...]) -> torch.Tensor:
    if isinstance(tokenizer, torch.Tensor):
        return _prepare_step_obs(tokenizer)
    terms = []
    for name in tokenizer_obs_names:
        value = tokenizer[name]
        terms.append(value.reshape(value.shape[0], -1))
    return torch.cat(terms, dim=-1).unsqueeze(1)


def _parse_flat_tokenizer(
    tokenizer: torch.Tensor,
    tokenizer_obs_names: tuple[str, ...],
    tokenizer_obs_dims: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    parsed = {}
    cursor = 0
    for name in tokenizer_obs_names:
        dims = tokenizer_obs_dims[name]
        dim = _flat_dim(dims)
        parsed[name] = tokenizer[..., cursor : cursor + dim].reshape(*tokenizer.shape[:-1], *dims)
        cursor += dim
    if cursor != tokenizer.shape[-1]:
        raise ValueError(f"Sonic tokenizer dim mismatch: expected {cursor}, got {tokenizer.shape[-1]}.")
    return parsed


class SonicBaseModule(nn.Module):
    """Sonic BaseModule-compatible MLP naming and temporal reshape behavior."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "SiLU",
        num_input_temporal_dims: int | None = None,
        num_output_temporal_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_input_temporal_dims = num_input_temporal_dims
        self.num_output_temporal_dims = num_output_temporal_dims
        mlp_input_dim = input_dim * num_input_temporal_dims if num_input_temporal_dims is not None else input_dim
        mlp_output_dim = output_dim * num_output_temporal_dims if num_output_temporal_dims is not None else output_dim
        self.module = _make_sonic_mlp(mlp_input_dim, mlp_output_dim, hidden_dims, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_input_temporal_dims is not None:
            x = x.reshape(*x.shape[:-2], self.input_dim * self.num_input_temporal_dims)
        out = self.module(x)
        if self.num_output_temporal_dims is not None:
            out = out.reshape(*out.shape[:-1], self.num_output_temporal_dims, self.output_dim)
        return out


class SonicRunningMeanStd(nn.Module):
    """Sonic-compatible running mean/std normalizer for flat observations."""

    def __init__(self, obs_dim: int, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(obs_dim, dtype=torch.float32))
        self.register_buffer("count", torch.ones((), dtype=torch.float32))
        self.frozen = False

    def forward(self, x: torch.Tensor, unnorm: bool = False) -> torch.Tensor:
        input_shape = x.shape
        if x.ndim == 3:
            x = x.reshape(-1, input_shape[-1])

        if unnorm:
            y = torch.clamp(x, min=-5.0, max=5.0)
            y = torch.sqrt(self.running_var + self.epsilon) * y + self.running_mean
        else:
            y = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
            y = torch.clamp(y, min=-5.0, max=5.0)

        if len(input_shape) == 3:
            y = y.reshape(input_shape)
        return y

    def update(self, x: torch.Tensor) -> None:
        if self.frozen:
            return
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.detach()
        with torch.inference_mode(False), torch.no_grad():
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_count = x.shape[0]
            delta = batch_mean - self.running_mean
            total_count = self.count + batch_count
            new_mean = self.running_mean + delta * batch_count / total_count
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
            self.running_mean.copy_(new_mean)
            self.running_var.copy_(m2 / total_count)
            self.count.copy_(total_count)


class SonicG1ActorModule(nn.Module):
    """Minimal Sonic UniversalTokenModule for the G1 motion path."""

    def __init__(
        self,
        actor_obs_dim: int,
        action_dim: int,
        tokenizer_obs_names: tuple[str, ...],
        tokenizer_obs_dims: dict[str, tuple[int, ...]],
        num_future_frames: int = 10,
        max_num_tokens: int = 2,
        token_dim: int = 32,
    ) -> None:
        super().__init__()
        self.tokenizer_obs_names = tokenizer_obs_names
        self.tokenizer_obs_dims = tokenizer_obs_dims
        self.num_future_frames = num_future_frames
        self.max_num_tokens = max_num_tokens
        self.token_dim = token_dim
        self.token_total_dim = max_num_tokens * token_dim
        encoder_input_dim = (
            tokenizer_obs_dims["command_multi_future_nonflat"][-1]
            + tokenizer_obs_dims["motion_anchor_ori_b_mf_nonflat"][-1]
        )
        self.encoders = nn.ModuleDict(
            {
                "g1": SonicBaseModule(
                    input_dim=encoder_input_dim,
                    output_dim=token_dim,
                    hidden_dims=[2048, 1024, 512, 512],
                    activation="SiLU",
                    num_input_temporal_dims=num_future_frames,
                    num_output_temporal_dims=max_num_tokens,
                )
            }
        )
        self.quantizer = FSQ(levels=[32] * token_dim)
        self.decoders = nn.ModuleDict(
            {
                "g1_dyn": SonicBaseModule(
                    input_dim=self.token_total_dim + actor_obs_dim,
                    output_dim=action_dim,
                    hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
                    activation="SiLU",
                )
            }
        )

    def parse_tokenizer_obs(self, input_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return _parse_flat_tokenizer(input_data["tokenizer"], self.tokenizer_obs_names, self.tokenizer_obs_dims)

    def forward(self, input_data: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = input_data["actor_obs"].shape[:2]
        tokenizer_obs = self.parse_tokenizer_obs(input_data)
        encoder_input = torch.cat(
            (tokenizer_obs["command_multi_future_nonflat"], tokenizer_obs["motion_anchor_ori_b_mf_nonflat"]), dim=-1
        )
        tokens = self.encoders["g1"](encoder_input)
        tokens, _ = self.quantizer(tokens)
        token_flat = tokens.reshape(batch_size, seq_len, self.token_total_dim)
        return self.decoders["g1_dyn"](torch.cat((token_flat, input_data["actor_obs"]), dim=-1))


class SonicActorModel(nn.Module):
    """Sonic G1 motion actor wrapped for rsl_rl PPO."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],  # noqa: ARG002
        obs_set: str,  # noqa: ARG002
        output_dim: int,
        tokenizer_obs_names: tuple[str, ...] | list[str] = (
            "encoder_index",
            "command_multi_future_nonflat",
            "command_z_multi_future_nonflat",
            "motion_anchor_ori_b_mf_nonflat",
        ),
        tokenizer_obs_dims: dict[str, tuple[int, ...]] | None = None,
        num_future_frames: int = 10,
        max_num_tokens: int = 2,
        token_dim: int = 32,
        init_noise_std: float = 0.05,
        use_log_std: bool = False,
        use_clampped_std: bool = True,
        std_clamp_min: float = 0.001,
        std_clamp_max: float = 0.5,
        freeze_noise_std: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.tokenizer_obs_names = tuple(tokenizer_obs_names)
        self.tokenizer_obs_dims = _tokenizer_dims(obs, self.tokenizer_obs_names, tokenizer_obs_dims)
        self.actor_module = SonicG1ActorModule(
            actor_obs_dim=_obs_dim(obs, "actor_obs"),
            action_dim=output_dim,
            tokenizer_obs_names=self.tokenizer_obs_names,
            tokenizer_obs_dims=self.tokenizer_obs_dims,
            num_future_frames=num_future_frames,
            max_num_tokens=max_num_tokens,
            token_dim=token_dim,
        )
        self.use_log_std = use_log_std
        self.use_clampped_std = use_clampped_std
        self.std_clamp_min = std_clamp_min
        self.std_clamp_max = std_clamp_max
        if use_log_std:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)), requires_grad=not freeze_noise_std)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(output_dim), requires_grad=not freeze_noise_std)
        self.distribution: Normal | None = None

    @property
    def get_std(self) -> torch.Tensor:
        if self.use_log_std:
            std = torch.exp(self.log_std.clamp(min=-20.0, max=2.0))
        else:
            std = self.std
        if self.use_clampped_std:
            std = std.clamp(min=self.std_clamp_min, max=self.std_clamp_max)
        return std.clamp(min=1e-6)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,  # noqa: ARG002
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        input_data = {
            "actor_obs": _prepare_step_obs(obs["actor_obs"]),
            "tokenizer": _flatten_tokenizer_obs(obs["tokenizer"], self.tokenizer_obs_names),
        }
        mean = self.actor_module(input_data)
        if mean.ndim == 3 and mean.shape[1] == 1:
            mean = mean[:, 0]
        std = self.get_std
        self.distribution = Normal(mean, (mean * 0.0 + std).clamp(min=1e-6))
        if stochastic_output:
            return self.distribution.sample()
        return self.distribution.mean

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.stddev  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return (self.output_mean, self.output_std)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs).sum(dim=-1)  # type: ignore[union-attr]

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("SonicActorModel export needs a dedicated deployment wrapper.")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("SonicActorModel ONNX export needs a dedicated deployment wrapper.")


class SonicCriticModel(nn.Module):
    """Sonic release critic MLP wrapped for rsl_rl PPO."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],  # noqa: ARG002
        obs_set: str,  # noqa: ARG002
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (2048, 2048, 1024, 1024, 512, 512),
        activation: str = "SiLU",
        running_mean_std: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        critic_dim = _obs_dim(obs, "critic")
        self.running_mean_std = SonicRunningMeanStd(critic_dim) if running_mean_std else None
        self.critic_module = SonicBaseModule(
            input_dim=critic_dim,
            output_dim=output_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,  # noqa: ARG002
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        critic_obs = obs["critic"]
        if self.running_mean_std is not None:
            with torch.no_grad():
                critic_obs = self.running_mean_std(critic_obs)
        return self.critic_module(critic_obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        if self.running_mean_std is not None:
            self.running_mean_std.update(obs["critic"])


class SonicLoRAPPO(PPO):
    """PPO with Any2Any-style LoRA dynamics adaptation."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        storage: RolloutStorage,
        source_checkpoint_path: str,
        source_load_strict: bool = False,
        load_actor_from_source: bool = True,
        load_critic_from_source: bool = True,
        freeze_actor_base: bool = True,
        freeze_critic_base: bool = True,
        train_actor_distribution: bool = True,
        lora_actor_targets: tuple[str, ...] | list[str] = ("actor_module.decoders.g1_dyn",),
        lora_critic_targets: tuple[str, ...] | list[str] = ("critic_module",),
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)

        self.source_checkpoint_path = source_checkpoint_path
        self.source_load_strict = source_load_strict
        self.load_actor_from_source = load_actor_from_source
        self.load_critic_from_source = load_critic_from_source
        self.freeze_actor_base = freeze_actor_base
        self.freeze_critic_base = freeze_critic_base
        self.train_actor_distribution = train_actor_distribution

        self._load_source_checkpoint()
        self.actor_lora_layers = apply_lora_to_linear_layers(
            self._raw_actor,
            lora_actor_targets,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        self.critic_lora_layers: list[str] = []
        if lora_critic_targets:
            self.critic_lora_layers = apply_lora_to_linear_layers(
                self._raw_critic,
                lora_critic_targets,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        if freeze_actor_base:
            freeze_non_lora_parameters(self._raw_actor, train_distribution=train_actor_distribution)
        if freeze_critic_base:
            freeze_non_lora_parameters(self._raw_critic, train_distribution=False)

        self.optimizer = self._build_optimizer()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["source_checkpoint_path"] = self.source_checkpoint_path
        saved_dict["actor_lora_layers"] = self.actor_lora_layers
        saved_dict["critic_lora_layers"] = self.critic_lora_layers
        return saved_dict

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SonicLoRAPPO:
        alg_class: type[SonicLoRAPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[nn.Module] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[nn.Module] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: nn.Module = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Sonic LoRA Actor Model: {actor}")
        cfg["algorithm"].pop("share_cnn_encoders", None)
        critic: nn.Module = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Sonic LoRA Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg: SonicLoRAPPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def _load_source_checkpoint(self) -> None:
        path = Path(self.source_checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Sonic source checkpoint does not exist: {path}")
        checkpoint = self._load_checkpoint(path)
        if self.load_actor_from_source:
            actor_state = self._extract_state_dict(
                checkpoint,
                ("policy_state_dict", "actor_model_state_dict", "actor_state_dict", "actor", "policy", "model"),
            )
            self._load_source_state_dict(self._raw_actor, actor_state, "actor")
        if self.load_critic_from_source:
            critic_state = self._extract_state_dict(
                checkpoint,
                ("value_state_dict", "critic_state_dict", "critic", "value"),
            )
            self._load_source_state_dict(self._raw_critic, critic_state, "critic")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [param for param in chain(self.actor.parameters(), self.critic.parameters()) if param.requires_grad]
        if not params:
            raise RuntimeError("SonicLoRAPPO has no trainable parameters after LoRA freezing.")
        return type(self.optimizer)(params, lr=self.learning_rate)

    @staticmethod
    def _extract_state_dict(checkpoint: dict, keys: tuple[str, ...]) -> dict:
        if not isinstance(checkpoint, dict):
            raise TypeError("Sonic checkpoint must be a dictionary.")
        for key in keys:
            if key in checkpoint:
                value = checkpoint[key]
                if isinstance(value, dict):
                    return value
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
        raise KeyError(f"Could not find any state_dict key in checkpoint. Tried: {keys}.")

    def _load_checkpoint(self, path: Path) -> dict:
        """Load a Sonic checkpoint without installing Sonic's training stack.

        Some public Sonic checkpoints include trainer-state objects from packages
        such as ``trl`` next to the model tensors.  PyTorch's weights-only loader
        cannot handle all of those objects.  We first try weights-only loading;
        if it still fails, we retry normal loading after installing lightweight
        dummy modules for known external trainer packages.
        """

        allowed_globals: set[str] = set()
        for _ in range(32):
            try:
                return torch.load(path, weights_only=True, map_location=self.device)
            except pickle.UnpicklingError as exc:
                unsupported_global = self._parse_unsupported_global(str(exc))
                if unsupported_global is None:
                    return self._load_checkpoint_with_dummy_modules(path)
                if unsupported_global in allowed_globals:
                    return self._load_checkpoint_with_dummy_modules(path)
                torch.serialization.add_safe_globals([self._make_dummy_global(unsupported_global)])
                allowed_globals.add(unsupported_global)
        raise RuntimeError(f"Too many unsupported globals while loading Sonic checkpoint: {path}")

    def _load_checkpoint_with_dummy_modules(self, path: Path) -> dict:
        for global_name in ("trl.trainer.utils.OnlineTrainerState",):
            self._make_dummy_global(global_name)
        for _ in range(32):
            try:
                return torch.load(path, weights_only=False, map_location=self.device)
            except ModuleNotFoundError as exc:
                module_name = getattr(exc, "name", None)
                if module_name is None:
                    raise
                self._ensure_dummy_module(module_name)
            except AttributeError as exc:
                missing_global = self._parse_missing_attribute_global(str(exc))
                if missing_global is None:
                    raise
                self._make_dummy_global(missing_global)
        raise RuntimeError(f"Too many missing Python globals while loading Sonic checkpoint: {path}")

    @staticmethod
    def _parse_unsupported_global(message: str) -> str | None:
        match = re.search(r"Unsupported global: GLOBAL ([\w.]+)", message)
        return match.group(1) if match else None

    @staticmethod
    def _parse_missing_attribute_global(message: str) -> str | None:
        match = re.search(r"Can't get attribute '([^']+)' on <module '([^']+)'", message)
        if match is None:
            return None
        return f"{match.group(2)}.{match.group(1)}"

    @staticmethod
    def _make_dummy_global(global_name: str) -> type:
        module_name, class_name = global_name.rsplit(".", 1)
        module = SonicLoRAPPO._ensure_dummy_module(module_name)
        if hasattr(module, class_name):
            dummy_cls = getattr(module, class_name)
            if isinstance(dummy_cls, type):
                return dummy_cls
        def __new__(cls, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            return object.__new__(cls)

        def __init__(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state):  # noqa: ANN001
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

        dummy_cls = type(
            class_name,
            (),
            {
                "__module__": module_name,
                "__new__": __new__,
                "__init__": __init__,
                "__setstate__": __setstate__,
            },
        )
        setattr(module, class_name, dummy_cls)
        return dummy_cls

    @staticmethod
    def _ensure_dummy_module(module_name: str) -> types.ModuleType:
        parts = module_name.split(".")
        parent_module = None
        current_name = ""
        for part in parts:
            current_name = part if not current_name else f"{current_name}.{part}"
            module = sys.modules.get(current_name)
            if module is None:
                module = types.ModuleType(current_name)
                sys.modules[current_name] = module
                if parent_module is not None:
                    setattr(parent_module, part, module)
            parent_module = module
        return sys.modules[module_name]

    def _load_source_state_dict(self, module: nn.Module, source_state: dict, module_name: str) -> None:
        if self.source_load_strict:
            module.load_state_dict(source_state, strict=True)
            return

        target_state = module.state_dict()
        matched_state = {}
        for key, value in source_state.items():
            normalized_keys = self._candidate_source_keys(key)
            candidate_keys = tuple(
                dict.fromkeys(
                    chain(
                        normalized_keys,
                        (f"actor.{candidate}" for candidate in normalized_keys),
                        (f"critic.{candidate}" for candidate in normalized_keys),
                    )
                )
            )
            for candidate_key in candidate_keys:
                if candidate_key in target_state and target_state[candidate_key].shape == value.shape:
                    matched_state[candidate_key] = value
                    break
        if not matched_state:
            raise RuntimeError(f"No matching {module_name} parameters were found in the Sonic source checkpoint.")
        missing, unexpected = module.load_state_dict(matched_state, strict=False)
        print(
            f"Loaded {len(matched_state)} {module_name} tensors from Sonic source checkpoint "
            f"({len(missing)} missing, {len(unexpected)} unexpected after filtering)."
        )

    @staticmethod
    def _candidate_source_keys(key: str) -> tuple[str, ...]:
        keys = [key]
        prefixes = ("_orig_mod.", "module.", "model.", "policy.", "actor.", "critic.", "value.")
        changed = True
        while changed:
            changed = False
            for current in list(keys):
                for prefix in prefixes:
                    if current.startswith(prefix):
                        stripped = current.removeprefix(prefix)
                        if stripped not in keys:
                            keys.append(stripped)
                            changed = True
        return tuple(keys)
