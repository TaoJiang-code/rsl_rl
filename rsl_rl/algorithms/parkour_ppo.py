# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms.amp_ppo import AMPPPO, _construct_amp_data, _resolve_discriminator_cfg
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import CNNModel, MoEMLPModel
from rsl_rl.modules import Discriminator, Distribution, EmpiricalNormalization, HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import AMPLoader, resolve_callable, resolve_obs_groups, unpad_trajectories


class _EncoderMoEModel(nn.Module):
    """Connect an existing CNNModel encoder to a MoEMLPModel head."""

    is_recurrent: bool = False

    def __init__(self, cnn: CNNModel, head: MoEMLPModel, latent_key: str) -> None:
        super().__init__()
        if cnn.is_recurrent or head.is_recurrent:
            raise ValueError("ParkourPPO currently supports feed-forward CNN and MoE models only.")
        self.cnn = cnn
        self.head = head
        self.latent_key = latent_key

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Encode visual observations and evaluate the MoE head."""
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        augmented_obs = self._augment_observations(obs)
        return self.head(
            augmented_obs,
            hidden_state=hidden_state,
            stochastic_output=stochastic_output,
        )

    def _augment_observations(self, obs: TensorDict) -> TensorDict:
        """Add the differentiable CNN latent to a shallow observation dictionary."""
        latent = self.cnn(obs)
        data = {key: value for key, value in obs.items()}
        data[self.latent_key] = latent
        return TensorDict(data, batch_size=obs.batch_size, device=obs.device)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update CNN and MoE-head observation normalizers."""
        if self.cnn.obs_groups:
            self.cnn.update_normalization(obs)
        with torch.no_grad():
            self.head.update_normalization(self._augment_observations(obs))

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset model state."""
        self.cnn.reset(dones)
        self.head.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the head hidden state."""
        return self.head.get_hidden_state()

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach model hidden state."""
        self.cnn.detach_hidden_state(dones)
        self.head.detach_hidden_state(dones)

    @property
    def distribution(self) -> Distribution | None:
        """Return the head output distribution."""
        return self.head.distribution

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the current action mean."""
        return self.head.output_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the current action standard deviation."""
        return self.head.output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the current action entropy."""
        return self.head.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return the current distribution parameters."""
        return self.head.output_distribution_params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute output log probabilities."""
        return self.head.get_output_log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute distribution KL divergence."""
        return self.head.get_kl_divergence(old_params, new_params)


class ParkourPPO(AMPPPO):
    """CNN-MoE parkour policy using the existing AMPPPO discriminator pipeline."""

    def __init__(
        self,
        actor: MoEMLPModel,
        critic: MoEMLPModel,
        cnn: CNNModel,
        storage: RolloutStorage,
        discriminator: Discriminator,
        amp_data: AMPLoader,
        critic_cnn: CNNModel | None = None,
        actor_cnn_latent_key: str = "actor_cnn_latent",
        critic_cnn_latent_key: str = "critic_cnn_latent",
        **amp_ppo_kwargs: Any,
    ) -> None:
        """Initialize CNN encoders, MoE heads, and the existing AMP implementation."""
        if critic_cnn is None:
            critic_cnn = copy.deepcopy(cnn)

        actor_model = _EncoderMoEModel(cnn, actor, actor_cnn_latent_key)
        critic_model = _EncoderMoEModel(critic_cnn, critic, critic_cnn_latent_key)
        super().__init__(
            actor_model,
            critic_model,
            storage,
            discriminator,
            amp_data,
            **amp_ppo_kwargs,
        )

        self.cnn = self._raw_actor.cnn
        self.critic_cnn = self._raw_critic.cnn
        self.actor_head = self._raw_actor.head
        self.critic_head = self._raw_critic.head

    def act(self, obs: TensorDict, amp_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Sample actions through CNN and MoE while recording the AMP state."""
        return super().act(obs, amp_obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
        amp_obs: torch.Tensor | None = None,
    ) -> None:
        """Store PPO and AMP transitions using the existing AMPPPO pipeline."""
        super().process_env_step(obs, rewards, dones, extras, amp_obs)

    def compute_amp_reward(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        task_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the existing transition-based AMP reward."""
        return super().compute_amp_reward(amp_obs, next_amp_obs, task_reward)

    def update(self) -> dict[str, float]:
        """Update CNNs, MoE heads, critic, and discriminator with AMPPPO."""
        return super().update()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> ParkourPPO:
        """Construct independent CNN encoders, MoE heads, and the existing AMP components."""
        cfg.setdefault("obs_groups", {})
        cfg.setdefault("multi_gpu", None)

        alg_class: type[ParkourPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        cnn_class: type[CNNModel] = resolve_callable(cfg["cnn"].pop("class_name"))  # type: ignore
        actor_class: type[MoEMLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MoEMLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        if not issubclass(actor_class, MoEMLPModel):
            raise TypeError(f"ParkourPPO actor must be a MoEMLPModel, got {actor_class.__name__}.")
        if not issubclass(critic_class, MoEMLPModel):
            raise TypeError(f"ParkourPPO critic must be a MoEMLPModel, got {critic_class.__name__}.")

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        if "cnn" not in cfg["obs_groups"]:
            raise KeyError("ParkourPPO requires cfg['obs_groups']['cnn'] for the visual encoder.")

        cnn_output_dim = cfg["cnn"].pop("output_dim")
        cfg["cnn"].setdefault("last_activation", cfg["cnn"].get("activation", "relu"))
        cnn = cnn_class(obs, cfg["obs_groups"], "cnn", cnn_output_dim, **cfg["cnn"]).to(device)
        print(f"Actor CNN Model: {cnn}")

        critic_cnn_cfg = cfg.get("critic_cnn")
        if critic_cnn_cfg is None:
            critic_cnn = copy.deepcopy(cnn)
        else:
            critic_cnn_class: type[CNNModel] = resolve_callable(critic_cnn_cfg.pop("class_name"))  # type: ignore
            if "critic_cnn" not in cfg["obs_groups"]:
                raise KeyError("cfg['obs_groups']['critic_cnn'] is required when critic_cnn is configured.")
            critic_cnn_output_dim = critic_cnn_cfg.pop("output_dim")
            critic_cnn_cfg.setdefault("last_activation", critic_cnn_cfg.get("activation", "relu"))
            critic_cnn = critic_cnn_class(
                obs,
                cfg["obs_groups"],
                "critic_cnn",
                critic_cnn_output_dim,
                **critic_cnn_cfg,
            ).to(device)
        print(f"Critic CNN Model: {critic_cnn}")

        with torch.no_grad():
            actor_sample_obs = _add_latent(obs, "actor_cnn_latent", cnn(obs))
            critic_sample_obs = _add_latent(obs, "critic_cnn_latent", critic_cnn(obs))

        actor_obs_groups = dict(cfg["obs_groups"])
        actor_obs_groups["actor"] = [*actor_obs_groups["actor"], "actor_cnn_latent"]
        critic_obs_groups = dict(cfg["obs_groups"])
        critic_obs_groups["critic"] = [*critic_obs_groups["critic"], "critic_cnn_latent"]

        actor = actor_class(
            actor_sample_obs,
            actor_obs_groups,
            "actor",
            env.num_actions,
            **cfg["actor"],
        ).to(device)
        print(f"Actor MoE Model: {actor}")
        critic = critic_class(
            critic_sample_obs,
            critic_obs_groups,
            "critic",
            1,
            **cfg["critic"],
        ).to(device)
        print(f"Critic MoE Model: {critic}")

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        amp_data = _construct_amp_data(cfg, env, device)
        amp_normalizer = EmpiricalNormalization(amp_data.observation_dim, until=int(1.0e8)).to(device)
        if not cfg.get("amp_normalization", True):
            amp_normalizer = None

        discriminator_cfg = _resolve_discriminator_cfg(cfg, amp_data.observation_dim)
        discriminator_class: type[Discriminator] = resolve_callable(discriminator_cfg.pop("class_name"))  # type: ignore
        discriminator = discriminator_class(device=device, **discriminator_cfg)
        print(f"AMP Discriminator: {discriminator}")

        min_std = cfg.get("min_normalized_std")
        if min_std is not None and not isinstance(min_std, torch.Tensor):
            min_std = torch.tensor(min_std, device=device, dtype=torch.float32)

        alg = alg_class(
            actor,
            critic,
            cnn,
            storage,
            discriminator,
            amp_data,
            critic_cnn=critic_cnn,
            amp_normalizer=amp_normalizer,
            min_std=min_std,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


def _add_latent(obs: TensorDict, key: str, latent: torch.Tensor) -> TensorDict:
    """Return a shallow TensorDict augmented with one latent tensor."""
    data = {obs_key: value for obs_key, value in obs.items()}
    data[key] = latent
    return TensorDict(data, batch_size=obs.batch_size, device=obs.device)
