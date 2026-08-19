# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BFM-style DAgger-regularized PPO.

This algorithm is intended for the Light-Loco-Parkour style setup:

    L_total = ppo_loss_coef * L_PPO + L_DAgger

The teacher policy is frozen. The student actor can use ``BFMActorModel`` to
match the Kitov-style pipeline:

    expert/reference obs -> frozen backward -> z
    policy obs + z -> MLP -> action
"""

from __future__ import annotations

import copy
import json
import math
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, unpad_trajectories

from .ppo import PPO


class _KitovBoxSpec:
    """Minimal shape holder compatible with Kitov obs-space validation."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _get_filter_keys(arch_cfg: dict) -> tuple[str, ...]:
    input_filter = arch_cfg.get("input_filter", {})
    keys = input_filter.get("key")
    if isinstance(keys, str):
        return (keys,)
    if isinstance(keys, (list, tuple)):
        return tuple(str(key) for key in keys)
    raise ValueError(f"Unsupported Kitov input_filter config: {input_filter}")


class _KitovDictInputFilter(nn.Module):
    """Kitov-compatible dict observation concat filter."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        super().__init__()
        self.keys = keys

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        if len(self.keys) == 1:
            return obs[self.keys[0]]
        return torch.cat([obs[key] for key in self.keys], dim=-1)


class _KitovNorm(nn.Module):
    """Kitov latent norm: sqrt(dim) * normalized vector."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class _KitovBlock(nn.Module):
    """LayerNorm -> Linear -> optional Mish block with Kitov state-dict names."""

    def __init__(self, input_dim: int, output_dim: int, activation: bool) -> None:
        super().__init__()
        seq: list[nn.Module] = [nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim)]
        if activation:
            seq.append(nn.Mish())
        self.mlp = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _KitovResidualBlock(nn.Module):
    """Residual block used by Kitov residual actors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


def _kitov_residual_embedding(input_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    if hidden_layers < 2:
        raise ValueError("Kitov residual embedding requires at least 2 layers.")
    seq: list[nn.Module] = [_KitovBlock(input_dim, hidden_dim, True)]
    for _ in range(hidden_layers - 2):
        seq.append(_KitovResidualBlock(hidden_dim))
    seq.append(_KitovBlock(hidden_dim, hidden_dim // 2, True))
    return nn.Sequential(*seq)


class _KitovBackwardMap(nn.Module):
    """Minimal Kitov BackwardMap implementation."""

    def __init__(self, obs_dims: dict[str, int], z_dim: int, cfg: dict) -> None:
        super().__init__()
        keys = _get_filter_keys(cfg)
        self.input_filter = _KitovDictInputFilter(keys)
        input_dim = sum(obs_dims[key] for key in keys)
        hidden_dim = int(cfg.get("hidden_dim", 256))
        hidden_layers = int(cfg.get("hidden_layers", 1))

        seq: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq.append(nn.Linear(hidden_dim, z_dim))
        if bool(cfg.get("norm", True)):
            seq.append(_KitovNorm())
        self.net = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(self.input_filter(obs))


class _KitovResidualActor(nn.Module):
    """Minimal Kitov ResidualActor implementation."""

    def __init__(self, obs_dims: dict[str, int], z_dim: int, action_dim: int, cfg: dict) -> None:
        super().__init__()
        if cfg.get("model") != "residual":
            raise ValueError(f"Only Kitov residual actor is supported, got model={cfg.get('model')!r}.")
        keys = _get_filter_keys(cfg)
        self.input_filter = _KitovDictInputFilter(keys)
        input_dim = sum(obs_dims[key] for key in keys)
        hidden_dim = int(cfg.get("hidden_dim", 2048))
        hidden_layers = int(cfg.get("hidden_layers", 6))
        embedding_layers = int(cfg.get("embedding_layers", 2))

        self.embed_z = _kitov_residual_embedding(input_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = _kitov_residual_embedding(input_dim, hidden_dim, embedding_layers)
        self.policy = nn.Sequential(
            *[_KitovResidualBlock(hidden_dim) for _ in range(hidden_layers)],
            _KitovBlock(hidden_dim, action_dim, False),
        )

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        obs_tensor = self.input_filter(obs)
        z_embedding = self.embed_z(torch.cat([obs_tensor, z], dim=-1))
        s_embedding = self.embed_s(obs_tensor)
        return torch.tanh(self.policy(torch.cat([s_embedding, z_embedding], dim=-1)))


class _KitovObsNormalizer(nn.Module):
    """Kitov-compatible dict BatchNorm observation normalizer."""

    def __init__(self, obs_dims: dict[str, int], cfg: dict) -> None:
        super().__init__()
        normalizers = cfg.get("normalizers", {})
        self.allow_mismatching_keys = bool(cfg.get("allow_mismatching_keys", False))
        self._normalizers = nn.ModuleDict()
        for key, normalizer_cfg in normalizers.items():
            if normalizer_cfg.get("name") != "BatchNormNormalizerConfig":
                raise ValueError(f"Only Kitov BatchNormNormalizerConfig is supported, got {normalizer_cfg}.")
            self._normalizers[key] = nn.Module()
            self._normalizers[key]._normalizer = nn.BatchNorm1d(  # type: ignore[attr-defined]
                num_features=obs_dims[key],
                affine=False,
                momentum=float(normalizer_cfg.get("momentum", 0.01)),
            )

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized_obs = {}
        for key, normalizer in self._normalizers.items():
            if key not in obs:
                if self.allow_mismatching_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in Kitov observation.")
            normalized_obs[key] = normalizer._normalizer(obs[key])  # type: ignore[attr-defined]
        return normalized_obs


class _MinimalKitovFBModel(nn.Module):
    """Local loader for the Kitov FBcprAux inference path."""

    def __init__(self, cfg: dict, init_kwargs: dict, device: str) -> None:
        super().__init__()
        model_name = cfg.get("name")
        if model_name not in {"FBModel", "FBcprModel", "FBcprAuxModel"}:
            raise ValueError(f"Unsupported Kitov model name: {model_name}.")

        obs_spaces = init_kwargs["obs_space"]["spaces"]
        self.obs_dims = {key: len(value["low"]) for key, value in obs_spaces.items()}
        self.obs_space = {key: _KitovBoxSpec((dim,)) for key, dim in self.obs_dims.items()}
        self.action_dim = int(init_kwargs["action_dim"])
        self.device = device
        self.amp_dtype = torch.bfloat16

        arch = cfg["archi"]
        self.cfg = SimpleNamespace(
            archi=SimpleNamespace(z_dim=int(arch["z_dim"]), norm_z=bool(arch.get("norm_z", True))),
            actor_std=float(cfg.get("actor_std", 0.2)),
            amp=bool(cfg.get("amp", False)),
            inference_batch_size=int(cfg.get("inference_batch_size", 500000)),
            seq_length=int(cfg.get("seq_length", 1)),
        )

        self._backward_map = _KitovBackwardMap(self.obs_dims, self.cfg.archi.z_dim, arch["b"])
        self._actor = _KitovResidualActor(self.obs_dims, self.cfg.archi.z_dim, self.action_dim, arch["actor"])
        self._obs_normalizer = _KitovObsNormalizer(self.obs_dims, cfg["obs_normalizer"])

    def _normalize(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            was_training = self._obs_normalizer.training
            self._obs_normalizer.eval()
            normalized = self._obs_normalizer(obs)
            self._obs_normalizer.train(was_training)
            return normalized

    def backward_map(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._backward_map(self._normalize(obs))

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.archi.norm_z:
            return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def goal_inference(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.project_z(self.backward_map(obs))

    def tracking_inference(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)

    def act(self, obs: dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        mu = self._actor(self._normalize(obs), z)
        if mean:
            return mu.float()
        std = torch.ones_like(mu) * self.cfg.actor_std
        return torch.clamp(mu + torch.randn_like(mu) * std, -1.0, 1.0).float()


def _json_to_finite_obs_space(init_kwargs: dict) -> dict:
    """Keep the original checkpoint init kwargs but avoid non-finite math outside Kitov."""
    return init_kwargs


def _load_minimal_kitov_model_from_checkpoint_dir(
    checkpoint_dir: str | Path,
    device: str = "cpu",
    checkpoint_subdir: str = "checkpoint",
) -> _MinimalKitovFBModel:
    checkpoint_dir = KitovBackwardEncoderWrapper._resolve_checkpoint_dir(str(checkpoint_dir), checkpoint_subdir)
    model_dir = checkpoint_dir / "model"
    with (model_dir / "config.json").open("r") as f:
        cfg = json.load(f)
    with (model_dir / "init_kwargs.json").open("r") as f:
        init_kwargs = _json_to_finite_obs_space(json.load(f))

    model = _MinimalKitovFBModel(cfg, init_kwargs, device=device)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Loading Kitov teacher checkpoints requires 'safetensors'.") from exc

    state_dict = load_file(model_dir / "model.safetensors", device=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model


class KitovBackwardEncoderWrapper(nn.Module):
    """Frozen Kitov backward encoder exposed as a plain ``nn.Module``.

    Kitov FB/CPR checkpoints store a full model under ``checkpoint/model`` and
    the backward path includes the Kitov observation normalizer. This wrapper
    keeps that logic intact and presents a small forward API for rsl_rl:

        expert/reference obs -> Kitov model.goal_inference() -> z
    """

    def __init__(
        self,
        checkpoint_dir: str,
        kitov_root: str | None = None,
        checkpoint_subdir: str = "checkpoint",
        device: str = "cpu",
        output_mode: str = "goal_inference",
    ) -> None:
        super().__init__()

        self.checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir, checkpoint_subdir)
        self.output_mode = output_mode
        self.kitov_root = kitov_root
        self.model = _load_minimal_kitov_model_from_checkpoint_dir(
            self.checkpoint_dir,
            device=device,
            checkpoint_subdir=checkpoint_subdir,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.z_dim = int(self.model.cfg.archi.z_dim)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode expert/reference observations into a Kitov latent ``z``."""
        with torch.no_grad():
            if self.output_mode == "goal_inference":
                return self.model.goal_inference(obs).detach()
            if self.output_mode == "backward_map":
                return self.model.backward_map(obs).detach()
            raise ValueError(f"Unsupported Kitov backward output_mode: {self.output_mode}")

    @staticmethod
    def _resolve_checkpoint_dir(checkpoint_dir: str, checkpoint_subdir: str) -> Path:
        path = Path(checkpoint_dir).resolve()
        if (path / "model" / "model.safetensors").is_file():
            return path
        if (path / checkpoint_subdir / "model" / "model.safetensors").is_file():
            return path / checkpoint_subdir
        raise FileNotFoundError(
            "Could not find Kitov checkpoint model.safetensors. Expected either "
            f"'{path / 'model' / 'model.safetensors'}' or "
            f"'{path / checkpoint_subdir / 'model' / 'model.safetensors'}'."
        )


class KitovTeacherActorWrapper(nn.Module):
    """Frozen Kitov teacher exposed with the rsl_rl actor call signature.

    Kitov BFM policies are not a single rsl_rl ``MLPModel``. Their deployed
    path is:

        backward_obs -> Kitov backward/goal_inference -> z
        actor_obs + z -> Kitov actor -> action

    This wrapper keeps that path intact while allowing the student to remain a
    normal rsl_rl actor. The input maps intentionally live in config because
    the IsaacLab obs layout must match the Kitov checkpoint's training obs.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        checkpoint_dir: str,
        kitov_root: str | None = None,
        checkpoint_subdir: str = "checkpoint",
        device: str = "cpu",
        actor_obs_groups: tuple[str, ...] | list[str] | None = None,
        backward_obs_groups: tuple[str, ...] | list[str] | None = None,
        actor_input_map: dict[str, object] | None = None,
        backward_input_map: dict[str, object] | None = None,
        z_mode: str = "goal_inference",
        mean_actions: bool = True,
        action_scale: float | list[float] | tuple[float, ...] = 1.0,
        action_bias: float | list[float] | tuple[float, ...] = 0.0,
        source_action_names: tuple[str, ...] | list[str] | None = None,
        target_action_names: tuple[str, ...] | list[str] | None = None,
        source_action_scale: dict[str, float] | list[float] | tuple[float, ...] | None = None,
        target_action_scale: dict[str, float] | list[float] | tuple[float, ...] | None = None,
        source_action_pre_scale: float = 1.0,
        source_action_clip: float | None = None,
        target_action_pre_scale: float = 1.0,
        validate_action_dim: bool = True,
        **_: object,
    ) -> None:
        super().__init__()

        self.checkpoint_dir = KitovBackwardEncoderWrapper._resolve_checkpoint_dir(checkpoint_dir, checkpoint_subdir)
        self.actor_obs_groups = tuple(actor_obs_groups if actor_obs_groups is not None else (obs_set,))
        self.backward_obs_groups = tuple(
            backward_obs_groups if backward_obs_groups is not None else self.actor_obs_groups
        )
        self.actor_input_map = actor_input_map
        self.backward_input_map = backward_input_map
        self.z_mode = z_mode
        self.mean_actions = mean_actions
        self.output_dim = int(output_dim)
        self.skip_compile = True

        self.kitov_root = kitov_root
        self.model = _load_minimal_kitov_model_from_checkpoint_dir(
            self.checkpoint_dir,
            device=device,
            checkpoint_subdir=checkpoint_subdir,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.z_dim = int(self.model.cfg.archi.z_dim)
        self.teacher_loaded = True
        self.use_action_space_conversion = source_action_names is not None or target_action_names is not None
        if self.use_action_space_conversion:
            if source_action_names is None or target_action_names is None:
                raise ValueError("Action conversion requires both source_action_names and target_action_names.")
            if source_action_scale is None or target_action_scale is None:
                raise ValueError("Action conversion requires both source_action_scale and target_action_scale.")
            source_names = tuple(source_action_names)
            target_names = tuple(target_action_names)
            missing_targets = [name for name in target_names if name not in source_names]
            if missing_targets:
                raise ValueError(f"Action conversion target joints missing from source order: {missing_targets}.")
            self.source_action_names = source_names
            self.target_action_names = target_names
            self.source_action_pre_scale = float(source_action_pre_scale)
            self.source_action_clip = None if source_action_clip is None else float(source_action_clip)
            self.target_action_pre_scale = float(target_action_pre_scale)
            self.register_buffer(
                "source_to_target_action_ids",
                torch.tensor([source_names.index(name) for name in target_names], dtype=torch.long),
            )
            self.register_buffer(
                "source_action_scale",
                self._as_named_action_scale(source_action_scale, source_names, "source_action_scale"),
            )
            self.register_buffer(
                "target_action_scale",
                self._as_named_action_scale(target_action_scale, target_names, "target_action_scale"),
            )
        else:
            self.source_action_names = None
            self.target_action_names = None
            self.source_action_pre_scale = 1.0
            self.source_action_clip = None
            self.target_action_pre_scale = 1.0
            self.register_buffer("action_scale", self._as_action_transform(action_scale, "action_scale"))
            self.register_buffer("action_bias", self._as_action_transform(action_bias, "action_bias"))

        expected_model_action_dim = len(self.source_action_names) if self.use_action_space_conversion else self.output_dim
        if self.use_action_space_conversion and len(self.target_action_names) != self.output_dim:
            raise ValueError(
                f"Kitov teacher target action dim={len(self.target_action_names)} does not match "
                f"environment action_dim={self.output_dim}."
            )
        if validate_action_dim and int(getattr(self.model, "action_dim", expected_model_action_dim)) != expected_model_action_dim:
            raise ValueError(
                f"Kitov teacher action_dim={getattr(self.model, 'action_dim', None)} does not match "
                f"expected source action_dim={expected_model_action_dim}."
            )

        self._validate_obs_sources(obs)
        self._validate_kitov_obs(self._build_kitov_obs(obs, self.actor_input_map, self.actor_obs_groups), "actor")
        self._validate_kitov_obs(
            self._build_kitov_obs(obs, self.backward_input_map, self.backward_obs_groups),
            "backward",
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Compute deterministic teacher actions from Kitov backward + actor."""
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        actor_obs = self._build_kitov_obs(obs, self.actor_input_map, self.actor_obs_groups)
        backward_obs = self._build_kitov_obs(obs, self.backward_input_map, self.backward_obs_groups)
        with torch.no_grad():
            z = self._compute_z(backward_obs)
            actions = self.model.act(actor_obs, z, mean=self.mean_actions).detach()
        expected_action_dim = len(self.source_action_names) if self.use_action_space_conversion else self.output_dim
        if actions.shape[-1] != expected_action_dim:
            raise ValueError(f"Kitov teacher returned action dim {actions.shape[-1]}, expected {expected_action_dim}.")
        return self._convert_teacher_action(actions)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Match the rsl_rl actor interface."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return no recurrent state."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Match the rsl_rl recurrent interface."""
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        """Teacher normalizers are loaded from the Kitov checkpoint and frozen."""
        pass

    def _compute_z(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if self.z_mode == "goal_inference":
            return self.model.goal_inference(obs).detach()
        if self.z_mode == "tracking_inference":
            return self.model.tracking_inference(obs).detach()
        if self.z_mode == "backward_map":
            z = self.model.backward_map(obs)
            return self.model.project_z(z).detach()
        raise ValueError(f"Unsupported Kitov teacher z_mode: {self.z_mode}")

    def _validate_obs_sources(self, obs: TensorDict) -> None:
        if self.actor_input_map is None:
            self._validate_groups(obs, self.actor_obs_groups, "actor_obs_groups")
        if self.backward_input_map is None:
            self._validate_groups(obs, self.backward_obs_groups, "backward_obs_groups")

    def _validate_kitov_obs(self, kitov_obs: torch.Tensor | dict[str, torch.Tensor], obs_name: str) -> None:
        obs_space = getattr(self.model, "obs_space", None)
        spaces = getattr(obs_space, "spaces", None)
        if spaces is None and isinstance(obs_space, dict):
            spaces = obs_space
        if spaces is None:
            if not isinstance(kitov_obs, torch.Tensor):
                raise TypeError(f"Kitov {obs_name} obs must be a tensor for non-dict obs_space, got dict.")
            expected_shape = getattr(obs_space, "shape", None)
            if expected_shape is not None and kitov_obs.shape[-1] != int(expected_shape[-1]):
                raise ValueError(
                    f"Kitov {obs_name} obs dim mismatch: got {kitov_obs.shape[-1]}, expected {expected_shape[-1]}."
                )
            return
        if not isinstance(kitov_obs, dict):
            raise TypeError(
                f"Kitov checkpoint expects dict observations with keys {list(spaces.keys())}. "
                f"Set {obs_name}_input_map instead of passing a single concatenated obs group."
            )
        for key, value in kitov_obs.items():
            if key not in spaces:
                raise KeyError(f"Kitov {obs_name} input key '{key}' is not in checkpoint obs_space keys {list(spaces.keys())}.")
            expected_shape = getattr(spaces[key], "shape", None)
            if expected_shape is not None and value.shape[-1] != int(expected_shape[-1]):
                raise ValueError(
                    f"Kitov {obs_name} input '{key}' dim mismatch: got {value.shape[-1]}, expected {expected_shape[-1]}."
                )

    @staticmethod
    def _validate_groups(obs: TensorDict, obs_groups: tuple[str, ...], field_name: str) -> None:
        missing = [obs_group for obs_group in obs_groups if obs_group not in obs]
        if missing:
            raise KeyError(f"Missing {field_name} observation group(s): {missing}.")

    def _as_action_transform(
        self, value: float | list[float] | tuple[float, ...], field_name: str
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            return tensor
        if tensor.ndim == 1 and tensor.shape[0] == self.output_dim:
            return tensor
        raise ValueError(f"{field_name} must be a scalar or a vector with length {self.output_dim}, got {tuple(tensor.shape)}.")

    def _as_named_action_scale(
        self,
        value: dict[str, float] | list[float] | tuple[float, ...],
        action_names: tuple[str, ...],
        field_name: str,
    ) -> torch.Tensor:
        if isinstance(value, dict):
            missing = [name for name in action_names if name not in value]
            if missing:
                raise ValueError(f"{field_name} missing scale entries for joints: {missing}.")
            tensor = torch.tensor([value[name] for name in action_names], dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 1 and tensor.shape[0] == len(action_names):
            return tensor
        raise ValueError(
            f"{field_name} must be a vector with length {len(action_names)}, got {tuple(tensor.shape)}."
        )

    def _convert_teacher_action(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.use_action_space_conversion:
            return actions * self.action_scale + self.action_bias

        source_delta = actions * self.source_action_pre_scale
        if self.source_action_clip is not None:
            source_delta = torch.clamp(source_delta, -self.source_action_clip, self.source_action_clip)
        physical_delta = source_delta * self.source_action_scale
        physical_delta = physical_delta[:, self.source_to_target_action_ids]
        return physical_delta / (self.target_action_pre_scale * self.target_action_scale)

    def _build_kitov_obs(
        self,
        obs: TensorDict,
        input_map: dict[str, object] | None,
        fallback_obs_groups: tuple[str, ...],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if input_map is None:
            if len(fallback_obs_groups) == 1:
                return self._as_plain_obs(obs[fallback_obs_groups[0]])
            return {obs_group: obs[obs_group] for obs_group in fallback_obs_groups}
        return {kitov_key: self._resolve_input_spec(obs, spec) for kitov_key, spec in input_map.items()}

    def _as_plain_obs(self, obs_value: object) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(obs_value, torch.Tensor):
            return obs_value
        if isinstance(obs_value, dict):
            return {str(key): value for key, value in obs_value.items()}
        if hasattr(obs_value, "keys"):
            return {str(key): obs_value[key] for key in obs_value.keys()}
        raise TypeError(f"Unsupported Kitov obs container type: {type(obs_value)}.")

    def _resolve_input_spec(self, obs: TensorDict, spec: object) -> torch.Tensor:
        if isinstance(spec, str):
            return obs[spec]
        if isinstance(spec, dict):
            if "group" in spec:
                tensor = obs[str(spec["group"])]
                key = spec.get("key")
                if key is not None:
                    if not hasattr(tensor, "__getitem__"):
                        raise TypeError(f"Input map group '{spec['group']}' does not support key lookup.")
                    tensor = tensor[str(key)]
                start = spec.get("start")
                end = spec.get("end")
                tensor = tensor[..., int(start) if start is not None else None : int(end) if end is not None else None]
                if spec.get("reverse_history", False):
                    output_history_length = int(spec["history_length"])
                    input_history_length = int(spec.get("buffer_history_length", output_history_length))
                    term_dim = int(spec["term_dim"])
                    expected_input_dim = input_history_length * term_dim
                    if tensor.shape[-1] != expected_input_dim:
                        raise ValueError(
                            "Cannot reverse Kitov history input with unexpected dim: "
                            f"got {tensor.shape[-1]}, expected {expected_input_dim} "
                            f"({input_history_length} x {term_dim})."
                        )
                    leading_shape = tensor.shape[:-1]
                    tensor = tensor.reshape(*leading_shape, input_history_length, term_dim)
                    if spec.get("drop_current", False):
                        tensor = tensor[..., :-1, :]
                    if tensor.shape[-2] != output_history_length:
                        raise ValueError(
                            "Cannot build Kitov history input with unexpected length after trimming: "
                            f"got {tensor.shape[-2]}, expected {output_history_length}."
                        )
                    tensor = torch.flip(tensor, dims=(-2,))
                    tensor = tensor.reshape(*leading_shape, output_history_length * term_dim)
                return tensor
            if "concat" in spec:
                return torch.cat([self._resolve_input_spec(obs, child) for child in spec["concat"]], dim=-1)
        if isinstance(spec, (list, tuple)):
            if len(spec) == 3 and isinstance(spec[0], str) and all(isinstance(x, int) for x in spec[1:]):
                return obs[spec[0]][..., spec[1] : spec[2]]
            return torch.cat([self._resolve_input_spec(obs, child) for child in spec], dim=-1)
        raise TypeError(
            "Kitov input map values must be an obs-group string, a {'group', 'start', 'end'} slice, "
            "a {'concat': [...]} block, or a list/tuple of specs."
        )


class BFMActorModel(MLPModel):
    """MLP actor conditioned on a latent computed by a frozen backward encoder.

    This is the rsl_rl-side equivalent of the first Kitov-compatible student
    shape:

        z = backward_encoder(expert/reference obs)
        concat(policy obs, z) -> MLP -> action distribution

    The environment should provide the expert/reference observation group that
    the backward encoder consumes. It should not provide ``z`` directly.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        z_source_obs_groups: tuple[str, ...] | list[str] = ("expert",),
        z_dim: int = 100,
        backward_encoder: nn.Module | None = None,
        pass_z_source_as_dict: bool = False,
        detach_z: bool = True,
        **kwargs,
    ) -> None:
        self.z_source_obs_groups = tuple(z_source_obs_groups)
        self.z_dim = int(getattr(backward_encoder, "z_dim", z_dim))
        self.backward_encoder = backward_encoder
        self.pass_z_source_as_dict = pass_z_source_as_dict
        self.detach_z = detach_z
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = list(obs_groups[obs_set])
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The BFMActorModel only supports 1D policy observations, got shape "
                    f"{obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        for obs_group in self.z_source_obs_groups:
            if obs_group not in obs:
                raise KeyError(f"Missing z-source observation group '{obs_group}' for BFMActorModel.")
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The z-source observation group must be 1D, got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.z_dim

    def set_backward_encoder(self, backward_encoder: nn.Module) -> None:
        """Attach the frozen backward encoder used to compute ``z``."""
        self.backward_encoder = backward_encoder
        self.backward_encoder.eval()
        for param in self.backward_encoder.parameters():
            param.requires_grad_(False)

    def compute_z(self, obs: TensorDict) -> torch.Tensor:
        """Compute latent ``z`` from expert/reference observations."""
        if self.backward_encoder is None:
            raise RuntimeError(
                "BFMActorModel requires a backward_encoder. In Kitov terms, z should come from "
                "frozen backward(expert/reference obs), not from the environment."
            )
        if self.pass_z_source_as_dict:
            z_source = {obs_group: obs[obs_group] for obs_group in self.z_source_obs_groups}
        else:
            z_source = torch.cat([obs[obs_group] for obs_group in self.z_source_obs_groups], dim=-1)
        z = self.backward_encoder(z_source)
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"Backward encoder returned z dim {z.shape[-1]}, expected {self.z_dim}.")
        if self.detach_z:
            z = z.detach()
        return z

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs_latent = super().get_latent(obs, masks, hidden_state)
        z = self.compute_z(obs)
        return torch.cat([obs_latent, z], dim=-1)


class BFMDAggerPPO(PPO):
    """PPO with an additional frozen-teacher action imitation loss."""

    teacher: nn.Module
    """Frozen teacher actor."""

    teacher_loaded: bool = False
    """Whether teacher weights have been loaded from a checkpoint."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        teacher: nn.Module,
        storage: RolloutStorage,
        dagger_loss_coef: float = 1.0,
        ppo_loss_coef: float = 0.1,
        dagger_loss_type: str = "mse",
        dagger_decay: float = 1.0,
        dagger_min_coef: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize DAgger-PPO.

        Args:
            actor: Student actor trained by PPO and DAgger.
            critic: PPO critic.
            teacher: Frozen teacher actor used to generate target actions.
            storage: Rollout storage.
            dagger_loss_coef: Deprecated compatibility parameter. DAgger is the primary loss and is not scaled.
            ppo_loss_coef: PPO loss coefficient. Keep this below 1.0 when PPO is only auxiliary.
            dagger_loss_type: Imitation loss type, either ``mse`` or ``huber``.
            dagger_decay: Multiplicative coefficient decay applied after each update.
            dagger_min_coef: Lower bound for the decayed DAgger coefficient.
            **kwargs: PPO parameters.
        """
        super().__init__(actor, critic, storage, **kwargs)

        self.teacher = teacher.to(self.device)
        self._raw_teacher = self.teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.dagger_loss_coef = dagger_loss_coef
        self.ppo_loss_coef = ppo_loss_coef
        self.teacher_loaded = bool(getattr(self.teacher, "teacher_loaded", False))
        self.dagger_decay = dagger_decay
        self.dagger_min_coef = dagger_min_coef
        if dagger_loss_type == "mse":
            self.dagger_loss_fn = nn.MSELoss()
        elif dagger_loss_type == "huber":
            self.dagger_loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported dagger_loss_type: {dagger_loss_type}")

    def update(self) -> dict[str, float]:
        """Run PPO updates with an additional DAgger actor loss."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_dagger_loss = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            with torch.no_grad():
                teacher_actions = self.teacher(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=False,
                )
            student_actions = self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=False,
            )
            dagger_loss = self.dagger_loss_fn(
                student_actions[:original_batch_size],
                teacher_actions[:original_batch_size],
            )

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            loss = self.ppo_loss_coef * ppo_loss + dagger_loss

            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_dagger_loss += dagger_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_dagger_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "dagger": mean_dagger_loss,
            "dagger_coef": 1.0,
            "ppo_coef": self.ppo_loss_coef,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.dagger_loss_coef = max(self.dagger_min_coef, self.dagger_loss_coef * self.dagger_decay)
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models and keep the teacher frozen."""
        super().train_mode()
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set eval mode for all models."""
        super().eval_mode()
        self.teacher.eval()

    def save(self) -> dict:
        """Return a dict of all trainable models and optimizer states."""
        saved_dict = super().save()
        saved_dict["teacher_loaded"] = self.teacher_loaded
        saved_dict["dagger_loss_coef"] = self.dagger_loss_coef
        saved_dict["ppo_loss_coef"] = self.ppo_loss_coef
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load student actor, critic, and optimizer states."""
        loaded = super().load(loaded_dict, load_cfg, strict)
        if "dagger_loss_coef" in loaded_dict:
            self.dagger_loss_coef = float(loaded_dict["dagger_loss_coef"])
        if "ppo_loss_coef" in loaded_dict:
            self.ppo_loss_coef = float(loaded_dict["ppo_loss_coef"])
        self.teacher_loaded = bool(loaded_dict.get("teacher_loaded", self.teacher_loaded))
        return loaded

    def load_teacher(self, checkpoint_path: str, strict: bool = True) -> None:
        """Load frozen teacher actor weights from a torch checkpoint.

        Kitov checkpoints can wrap the actor under project-specific keys. This
        method accepts common actor state-dict keys and otherwise treats the
        checkpoint itself as the teacher state dict.
        """
        if getattr(self._raw_teacher, "teacher_loaded", False):
            raise RuntimeError(
                "This teacher is already loaded internally, likely through KitovTeacherActorWrapper. "
                "Do not also set algorithm.teacher_checkpoint_path for that teacher."
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            for key in ("teacher_actor_state_dict", "actor_state_dict", "model_state_dict", "state_dict"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        self._raw_teacher.load_state_dict(checkpoint, strict=strict)
        self.teacher_loaded = True

    def compile(self, mode: str | None = None) -> None:
        """Compile actor, critic, and teacher with ``torch.compile``."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        if getattr(self._raw_teacher, "skip_compile", False):
            self.teacher = self._raw_teacher
        else:
            self.teacher = compile_model(self._raw_teacher, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> BFMDAggerPPO:
        """Construct a BFMDAggerPPO algorithm.

        Config convention:
            - ``teacher_actor`` is optional. If omitted, the teacher is built
              from the same class and config as ``actor``.
            - If ``actor`` is omitted, the student actor is built from
              ``teacher_actor`` so the two networks have the same type/layers.
        """
        alg_class: type[BFMDAggerPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore

        teacher_actor_cfg = copy.deepcopy(cfg.get("teacher_actor", cfg.get("actor")))
        if teacher_actor_cfg is None:
            raise ValueError("BFMDAggerPPO requires either 'actor' or 'teacher_actor' config.")
        actor_cfg = copy.deepcopy(cfg.get("actor", teacher_actor_cfg))
        critic_cfg = cfg["critic"]

        actor_backward_encoder = BFMDAggerPPO._build_backward_encoder(actor_cfg.pop("backward_encoder_cfg", None), device)
        teacher_backward_encoder = BFMDAggerPPO._build_backward_encoder(
            teacher_actor_cfg.pop("backward_encoder_cfg", None), device
        )
        if actor_backward_encoder is not None:
            actor_cfg["backward_encoder"] = actor_backward_encoder
            actor_cfg["z_dim"] = int(getattr(actor_backward_encoder, "z_dim", actor_cfg.get("z_dim", 100)))
        if teacher_backward_encoder is not None:
            teacher_actor_cfg["backward_encoder"] = teacher_backward_encoder
            teacher_actor_cfg["z_dim"] = int(
                getattr(teacher_backward_encoder, "z_dim", teacher_actor_cfg.get("z_dim", 100))
            )

        actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore
        teacher_actor_class: type[nn.Module] = resolve_callable(teacher_actor_cfg.pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "teacher" in cfg.get("obs_groups", {}):
            default_sets.append("teacher")
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        teacher_obs_set = cfg["algorithm"].pop("teacher_obs_set", "teacher" if "teacher" in cfg["obs_groups"] else "actor")
        teacher_checkpoint_path = cfg["algorithm"].pop("teacher_checkpoint_path", None)
        teacher_checkpoint_strict = cfg["algorithm"].pop("teacher_checkpoint_strict", True)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Student Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            critic_cfg["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")
        teacher: nn.Module = teacher_actor_class(
            obs, cfg["obs_groups"], teacher_obs_set, env.num_actions, **teacher_actor_cfg
        ).to(device)
        print(f"Teacher Actor Model: {teacher}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg: BFMDAggerPPO = alg_class(
            actor, critic, teacher, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        if teacher_checkpoint_path is not None:
            alg.load_teacher(teacher_checkpoint_path, strict=teacher_checkpoint_strict)
        if not alg.teacher_loaded:
            raise RuntimeError(
                "BFMDAggerPPO teacher is not loaded. Set algorithm.teacher_checkpoint_path for an rsl_rl teacher, "
                "or use KitovTeacherActorWrapper with a valid checkpoint_dir in teacher_actor."
            )

        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    @staticmethod
    def _build_backward_encoder(backward_encoder_cfg: dict | None, device: str) -> nn.Module | None:
        """Build a frozen backward encoder from config."""
        if backward_encoder_cfg is None:
            return None
        backward_encoder_cfg = copy.deepcopy(backward_encoder_cfg)
        encoder_class: type[nn.Module] = resolve_callable(backward_encoder_cfg.pop("class_name"))  # type: ignore
        backward_encoder_cfg.setdefault("device", device)
        encoder = encoder_class(**backward_encoder_cfg).to(device)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad_(False)
        return encoder

    def broadcast_parameters(self) -> None:
        """Broadcast trainable and frozen model parameters to all GPUs."""
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict(), self._raw_teacher.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        self._raw_teacher.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        """Average gradients across GPUs for trainable modules."""
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
