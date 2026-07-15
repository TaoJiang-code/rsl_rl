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
from rsl_rl.models import CNNModel, MLPModel, MoEMLPModel
from rsl_rl.modules import Discriminator, Distribution, EmpiricalNormalization, HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import AMPLoader, resolve_callable, resolve_obs_groups, unpad_trajectories


class _EncoderMoEModel(nn.Module):
    """Connect an existing CNNModel encoder to a MoEMLPModel head."""

    is_recurrent: bool = False

    def __init__(
        self,
        cnn: CNNModel,
        head: MoEMLPModel,
        latent_key: str,
    ) -> None:
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


class _ParkourActorModel(_EncoderMoEModel):
    """Parkour actor wrapper with CNN and velocity-estimator latents."""

    def __init__(
        self,
        cnn: CNNModel,
        velocity_estimator: MLPModel,
        head: MoEMLPModel,
        cnn_latent_key: str,
        velocity_latent_key: str,
        velocity_target_key: str,
    ) -> None:
        if velocity_estimator.is_recurrent:
            raise ValueError("ParkourPPO currently supports feed-forward velocity estimators only.")
        super().__init__(cnn, head, cnn_latent_key)
        self.velocity_estimator = velocity_estimator
        self.velocity_latent_key = velocity_latent_key
        self.velocity_target_key = velocity_target_key

    def _augment_observations(self, obs: TensorDict) -> TensorDict:
        data = {key: value for key, value in super()._augment_observations(obs).items()}
        data[self.velocity_latent_key] = self.velocity_estimator(obs)
        return TensorDict(data, batch_size=obs.batch_size, device=obs.device)

    def compute_velocity_loss(self, obs: TensorDict) -> torch.Tensor:
        predicted_velocity = self.velocity_estimator(obs)
        target_velocity = obs[self.velocity_target_key]
        return nn.functional.mse_loss(predicted_velocity, target_velocity)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.cnn.obs_groups:
            self.cnn.update_normalization(obs)
        self.velocity_estimator.update_normalization(obs)
        with torch.no_grad():
            self.head.update_normalization(self._augment_observations(obs))

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.cnn.reset(dones)
        self.velocity_estimator.reset(dones)
        self.head.reset(dones, hidden_state)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.cnn.detach_hidden_state(dones)
        self.velocity_estimator.detach_hidden_state(dones)
        self.head.detach_hidden_state(dones)


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
        velocity_estimator: MLPModel,
        critic_cnn: CNNModel | None = None,
        actor_cnn_latent_key: str = "actor_cnn_latent",
        critic_cnn_latent_key: str = "critic_cnn_latent",
        actor_velocity_latent_key: str = "actor_velocity_latent",
        velocity_target_key: str = "velocity_estimator",
        velocity_loss_coef: float = 1.0,
        **amp_ppo_kwargs: Any,
    ) -> None:
        """Initialize CNN encoders, MoE heads, and the existing AMP implementation."""
        if critic_cnn is None:
            critic_cnn = copy.deepcopy(cnn)

        actor_model = _ParkourActorModel(
            cnn,
            velocity_estimator,
            actor,
            actor_cnn_latent_key,
            actor_velocity_latent_key,
            velocity_target_key,
        )
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
        self.velocity_estimator = self._raw_actor.velocity_estimator
        self.velocity_loss_coef = velocity_loss_coef

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
        """Update CNNs, velocity estimator, MoE heads, critic, and discriminator."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_penalty = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_velocity_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            mini_batch_size,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            mini_batch_size,
        )

        for batch, policy_amp_batch, expert_amp_batch in zip(generator, amp_policy_generator, amp_expert_generator):
            original_batch_size = batch.observations.batch_size[0]  # type: ignore

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages = batch.advantages  # type: ignore
                    batch.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            velocity_loss = self._raw_actor.compute_velocity_loss(batch.observations[:original_batch_size])

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
                    kl = self.actor.get_kl_divergence(  # type: ignore
                        batch.old_distribution_params,
                        distribution_params,
                    )
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
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.velocity_loss_coef * velocity_loss
            )

            rnd_loss = (
                self.rnd.compute_loss(batch.observations[:original_batch_size])  # type: ignore
                if self.rnd
                else None
            )

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            policy_state, policy_next_state = policy_amp_batch
            expert_state, expert_next_state = expert_amp_batch
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state)
                self.amp_normalizer.update(expert_state)
                policy_state = self.amp_normalizer(policy_state)
                policy_next_state = self.amp_normalizer(policy_next_state)
                expert_state = self.amp_normalizer(expert_state)
                expert_next_state = self.amp_normalizer(expert_next_state)

            policy_logits = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_logits = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = nn.functional.mse_loss(expert_logits, torch.ones_like(expert_logits))
            policy_loss = nn.functional.mse_loss(policy_logits, -torch.ones_like(policy_logits))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_penalty = self.discriminator.compute_grad_penalty(
                expert_state,
                expert_next_state,
                self.amp_grad_penalty_coef,
            )
            loss = loss + self.amp_loss_coef * (amp_loss + grad_penalty)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._clamp_min_std()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_penalty += grad_penalty.item()
            mean_policy_pred += policy_logits.mean().item()
            mean_expert_pred += expert_logits.mean().item()
            mean_velocity_loss += velocity_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "amp": mean_amp_loss / num_updates,
            "amp_grad_penalty": mean_grad_penalty / num_updates,
            "amp_policy_pred": mean_policy_pred / num_updates,
            "amp_expert_pred": mean_expert_pred / num_updates,
            "velocity_estimator": mean_velocity_loss / num_updates,
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        self.storage.clear()
        return loss_dict

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> ParkourPPO:
        """Construct independent CNN encoders, MoE heads, and the existing AMP components."""
        cfg.setdefault("obs_groups", {})
        cfg.setdefault("multi_gpu", None)

        alg_class: type[ParkourPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        cnn_class: type[CNNModel] = resolve_callable(cfg["cnn"].pop("class_name"))  # type: ignore
        actor_class: type[MoEMLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MoEMLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        velocity_estimator_cfg = cfg["velocity_estimator"]
        velocity_estimator_class: type[MLPModel] = resolve_callable(velocity_estimator_cfg.pop("class_name"))  # type: ignore
        if not issubclass(actor_class, MoEMLPModel):
            raise TypeError(f"ParkourPPO actor must be a MoEMLPModel, got {actor_class.__name__}.")
        if not issubclass(critic_class, MoEMLPModel):
            raise TypeError(f"ParkourPPO critic must be a MoEMLPModel, got {critic_class.__name__}.")
        if not issubclass(velocity_estimator_class, MLPModel):
            raise TypeError(
                f"ParkourPPO velocity_estimator must be an MLPModel, got {velocity_estimator_class.__name__}."
            )

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

        velocity_target_key = cfg["algorithm"].get("velocity_target_key", "velocity_estimator")
        velocity_input_set = velocity_estimator_cfg.pop("obs_set", "velocity_estimator_input")
        velocity_estimator_obs_groups = dict(cfg["obs_groups"])
        velocity_estimator_obs_groups.setdefault(velocity_input_set, cfg["obs_groups"]["actor"])
        velocity_estimator_output_dim = velocity_estimator_cfg.pop("output_dim", obs[velocity_target_key].shape[-1])
        velocity_estimator = velocity_estimator_class(
            obs,
            velocity_estimator_obs_groups,
            velocity_input_set,
            velocity_estimator_output_dim,
            **velocity_estimator_cfg,
        ).to(device)
        print(f"Velocity Estimator Model: {velocity_estimator}")

        with torch.no_grad():
            actor_sample_obs = _add_latent(obs, "actor_cnn_latent", cnn(obs))
            actor_sample_obs = _add_latent(
                actor_sample_obs,
                cfg["algorithm"].get("actor_velocity_latent_key", "actor_velocity_latent"),
                velocity_estimator(obs),
            )
            critic_sample_obs = _add_latent(obs, "critic_cnn_latent", critic_cnn(obs))

        actor_obs_groups = dict(cfg["obs_groups"])
        actor_obs_groups["actor"] = [*actor_obs_groups["actor"], "actor_cnn_latent"]
        actor_obs_groups["actor"] = [
            *actor_obs_groups["actor"],
            cfg["algorithm"].get("actor_velocity_latent_key", "actor_velocity_latent"),
        ]
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
            velocity_estimator=velocity_estimator,
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
