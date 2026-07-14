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

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import CNNModel, MoEMLPModel
from rsl_rl.modules import Distribution, HiddenState, MMPDiscriminator
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer, unpad_trajectories


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


class ParkourPPO(PPO):
    """CNN-MoE parkour policy using environment-fed MMP discriminator rewards."""

    def __init__(
        self,
        actor: MoEMLPModel,
        critic: MoEMLPModel,
        cnn: CNNModel,
        storage: RolloutStorage,
        discriminators: dict[int, MMPDiscriminator],
        disc_buffers: dict[int, ReplayBuffer],
        critic_cnn: CNNModel | None = None,
        actor_cnn_latent_key: str = "actor_cnn_latent",
        critic_cnn_latent_key: str = "critic_cnn_latent",
        mmp_id_obs_group: str = "mmp_id",
        mmp_loss_coef: float = 1.0,
        mmp_grad_penalty_coef: float = 10.0,
        discriminator_learning_rate: float | None = None,
        discriminator_optimizer: str = "adam",
        discriminator_max_grad_norm: float | None = None,
        discriminator_mini_batch_size: int | None = None,
        min_std: float | list[float] | torch.Tensor | None = None,
        **ppo_kwargs: Any,
    ) -> None:
        """Initialize CNN encoders, MoE heads, and the environment-fed MMP implementation."""
        if critic_cnn is None:
            critic_cnn = copy.deepcopy(cnn)

        actor_model = _EncoderMoEModel(cnn, actor, actor_cnn_latent_key)
        critic_model = _EncoderMoEModel(critic_cnn, critic, critic_cnn_latent_key)
        super().__init__(actor_model, critic_model, storage, **ppo_kwargs)

        if len(discriminators) == 0:
            raise ValueError("ParkourPPO requires at least one MMP discriminator.")
        missing_buffers = sorted(set(discriminators) - set(disc_buffers))
        if missing_buffers:
            raise KeyError(f"Missing discriminator buffers for MMP ids: {missing_buffers}.")

        self.discriminator_ids = sorted(int(disc_id) for disc_id in discriminators)
        self.discriminators = nn.ModuleDict(
            {str(disc_id): discriminators[disc_id].to(self.device) for disc_id in self.discriminator_ids}
        )
        self._raw_discriminators = self.discriminators
        self.disc_buffers = {int(disc_id): disc_buffers[int(disc_id)] for disc_id in self.discriminator_ids}
        self.mmp_id_obs_group = mmp_id_obs_group
        self.mmp_loss_coef = mmp_loss_coef
        self.mmp_grad_penalty_coef = mmp_grad_penalty_coef
        self.discriminator_max_grad_norm = discriminator_max_grad_norm or self.max_grad_norm
        self.discriminator_mini_batch_size = discriminator_mini_batch_size
        self.min_std = min_std

        disc_lr = discriminator_learning_rate if discriminator_learning_rate is not None else self.learning_rate
        self.discriminator_optimizer = resolve_optimizer(discriminator_optimizer)(
            self.discriminators.parameters(),
            lr=disc_lr,
        )  # type: ignore

        first_discriminator = self._first_discriminator
        self.disc_obs_steps = first_discriminator.disc_obs_steps
        self.disc_obs_dim = first_discriminator.disc_obs_dim

        self.mmp_rewards: torch.Tensor | None = None
        self.mmp_scores: torch.Tensor | None = None

        self.cnn = self._raw_actor.cnn
        self.critic_cnn = self._raw_critic.cnn
        self.actor_head = self._raw_actor.head
        self.critic_head = self._raw_critic.head

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions through CNN and MoE."""
        return super().act(obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Store PPO transitions using MMP rewards computed from environment-fed expert observations."""
        policy_obs = self._first_discriminator.get_policy_obs(obs, flatten_history_dim=False)
        expert_obs = self._first_discriminator.get_expert_obs(obs, flatten_history_dim=False)
        mmp_ids = _get_mmp_ids(obs, self.mmp_id_obs_group)
        known_id_mask = torch.zeros_like(mmp_ids, dtype=torch.bool)
        for disc_id in self.discriminator_ids:
            known_id_mask |= mmp_ids == disc_id
        if torch.any(~known_id_mask):
            unknown_ids = torch.unique(mmp_ids[~known_id_mask]).detach().cpu().tolist()
            raise ValueError(f"ParkourPPO received unknown MMP discriminator ids: {unknown_ids}.")

        self.mmp_rewards = torch.zeros_like(rewards, device=self.device)
        self.mmp_scores = torch.zeros_like(rewards, device=self.device)

        for disc_id in self.discriminator_ids:
            mask = mmp_ids == disc_id
            if not torch.any(mask):
                continue

            discriminator = self.discriminators[str(disc_id)]
            disc_policy_obs = policy_obs[mask]
            disc_expert_obs = expert_obs[mask]
            disc_rewards, disc_scores = discriminator.predict_reward(disc_policy_obs, rewards[mask].to(self.device))
            self.mmp_rewards[mask] = disc_rewards
            self.mmp_scores[mask] = disc_scores
            self.disc_buffers[disc_id].insert(
                disc_policy_obs.reshape(disc_policy_obs.shape[0], -1),
                disc_expert_obs.reshape(disc_expert_obs.shape[0], -1),
            )

        super().process_env_step(obs, self.mmp_rewards, dones, extras)

    def update(self) -> dict[str, float]:  # noqa: C901
        """Update CNNs, MoE heads, critic, and MMP discriminator."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_mmp_loss = 0.0
        mean_grad_penalty = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]  # type: ignore

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages = batch.advantages  # type: ignore
                    batch.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            rnd_loss = (
                self.rnd.compute_loss(batch.observations[:original_batch_size])  # type: ignore
                if self.rnd
                else None
            )

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    ppo_loss = ppo_loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            discriminator_loss, disc_stats = self._compute_discriminator_loss(original_batch_size)

            self.optimizer.zero_grad()
            ppo_loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            self.discriminator_optimizer.zero_grad()
            if discriminator_loss is not None:
                discriminator_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()
                self._reduce_discriminator_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            if discriminator_loss is not None:
                nn.utils.clip_grad_norm_(self.discriminators.parameters(), self.discriminator_max_grad_norm)
            self.optimizer.step()
            self._clamp_min_std()
            if self.rnd:
                self.rnd.optimizer.step()
            if discriminator_loss is not None:
                self.discriminator_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_mmp_loss += disc_stats["mmp"]
            mean_grad_penalty += disc_stats["grad_penalty"]
            mean_policy_pred += disc_stats["policy_pred"]
            mean_expert_pred += disc_stats["expert_pred"]
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "mmp": mean_mmp_loss / num_updates,
            "mmp_grad_penalty": mean_grad_penalty / num_updates,
            "mmp_policy_pred": mean_policy_pred / num_updates,
            "mmp_expert_pred": mean_expert_pred / num_updates,
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        self.storage.clear()
        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        super().train_mode()
        self.discriminators.train()

    def eval_mode(self) -> None:
        """Set eval mode for learnable models."""
        super().eval_mode()
        self.discriminators.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["discriminators_state_dict"] = self._raw_discriminators.state_dict()
        saved_dict["discriminator_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
        saved_dict["discriminator_ids"] = self.discriminator_ids
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("discriminator", True):
            self._raw_discriminators.load_state_dict(loaded_dict["discriminators_state_dict"], strict=strict)
        if load_cfg is None or load_cfg.get("discriminator_optimizer", True):
            self.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
        return load_iteration

    def compile(self, mode: str | None = None) -> None:
        """Compile actor, critic, and routed discriminators if requested."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        self.discriminators = nn.ModuleDict(
            {
                str(disc_id): compile_model(self._raw_discriminators[str(disc_id)], mode)  # type: ignore
                for disc_id in self.discriminator_ids
            }
        )

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        super().broadcast_parameters()
        model_params = [self._raw_discriminators.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_discriminators.load_state_dict(model_params[0])

    def _compute_discriminator_loss(self, ppo_mini_batch_size: int) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Sample each routed discriminator buffer and compute the summed discriminator loss."""
        active_discriminator_ids = [
            disc_id for disc_id in self.discriminator_ids if self.disc_buffers[disc_id].num_samples > 0
        ]
        if len(active_discriminator_ids) == 0:
            return None, {"mmp": 0.0, "grad_penalty": 0.0, "policy_pred": 0.0, "expert_pred": 0.0}

        per_disc_batch_size = self.discriminator_mini_batch_size
        if per_disc_batch_size is None:
            per_disc_batch_size = max(1, ppo_mini_batch_size // len(active_discriminator_ids))

        total_loss: torch.Tensor | None = None
        mean_mmp_loss = 0.0
        mean_grad_penalty = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        num_losses = 0

        for disc_id in active_discriminator_ids:
            buffer = self.disc_buffers[disc_id]
            if buffer.num_samples == 0:
                continue

            batch_size = min(per_disc_batch_size, buffer.num_samples)
            sample_ids = torch.randint(buffer.num_samples, (batch_size,), device=self.device)
            policy_obs = buffer.states[sample_ids].view(batch_size, self.disc_obs_steps, self.disc_obs_dim)
            expert_obs = buffer.next_states[sample_ids].view(batch_size, self.disc_obs_steps, self.disc_obs_dim)
            discriminator = self.discriminators[str(disc_id)]

            with torch.no_grad():
                discriminator.update_normalization(policy_obs)
                discriminator.update_normalization(expert_obs)
                policy_obs = discriminator.normalize_obs(policy_obs)
                expert_obs = discriminator.normalize_obs(expert_obs)

            mmp_loss, policy_logits, expert_logits = discriminator.compute_loss(policy_obs, expert_obs)
            grad_penalty = discriminator.compute_grad_penalty(expert_obs, self.mmp_grad_penalty_coef)
            disc_loss = self.mmp_loss_coef * (mmp_loss + grad_penalty)
            total_loss = disc_loss if total_loss is None else total_loss + disc_loss

            mean_mmp_loss += mmp_loss.item()
            mean_grad_penalty += grad_penalty.item()
            mean_policy_pred += policy_logits.mean().item()
            mean_expert_pred += expert_logits.mean().item()
            num_losses += 1

        if total_loss is None or num_losses == 0:
            return None, {"mmp": 0.0, "grad_penalty": 0.0, "policy_pred": 0.0, "expert_pred": 0.0}

        return total_loss / num_losses, {
            "mmp": mean_mmp_loss / num_losses,
            "grad_penalty": mean_grad_penalty / num_losses,
            "policy_pred": mean_policy_pred / num_losses,
            "expert_pred": mean_expert_pred / num_losses,
        }

    def _reduce_discriminator_parameters(self) -> None:
        """Collect discriminator gradients from all GPUs and average them."""
        params = list(self.discriminators.parameters())
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

    def _clamp_min_std(self) -> None:
        """Clamp learnable Gaussian std parameters when a floor is configured."""
        if self.min_std is None or not hasattr(self._raw_actor, "distribution"):
            return
        distribution = self._raw_actor.distribution
        if distribution is None:
            return
        min_std = torch.as_tensor(self.min_std, device=self.device, dtype=torch.float32)
        if min_std.ndim == 0:
            min_std = min_std.repeat(self._raw_actor.output_std.shape[-1])
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                target = distribution.std_param
                floor = min_std if min_std.numel() == target.numel() else min_std.min().repeat(target.numel())
                target.clamp_(min=floor.reshape_as(target))
            elif hasattr(distribution, "log_std_param"):
                target = distribution.log_std_param
                floor = min_std if min_std.numel() == target.numel() else min_std.min().repeat(target.numel())
                target.clamp_(min=torch.log(floor.reshape_as(target).clamp_min(1.0e-6)))

    @property
    def _first_discriminator(self) -> MMPDiscriminator:
        """Return the first routed discriminator for shared observation extraction."""
        return self.discriminators[str(self.discriminator_ids[0])]  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> ParkourPPO:
        """Construct independent CNN encoders, MoE heads, and environment-fed MMP components."""
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

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        for required_obs_set in ("cnn", "mmp", "mmp_expert", "mmp_id"):
            if required_obs_set not in cfg["obs_groups"]:
                raise KeyError(f"ParkourPPO requires cfg['obs_groups']['{required_obs_set}'].")

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

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        disc_obs_steps, disc_obs_dim = _resolve_disc_obs_shape(obs, cfg["obs_groups"]["mmp"])
        expert_obs_steps, expert_obs_dim = _resolve_disc_obs_shape(obs, cfg["obs_groups"]["mmp_expert"])
        if expert_obs_steps != disc_obs_steps or expert_obs_dim != disc_obs_dim:
            raise ValueError(
                "ParkourPPO MMP policy and expert discriminator observations must have identical shapes, "
                f"got policy [T={disc_obs_steps}, D={disc_obs_dim}] and "
                f"expert [T={expert_obs_steps}, D={expert_obs_dim}]."
            )

        discriminator_cfg = dict(cfg.get("mmp_discriminator", cfg.get("discriminator", {})))
        discriminator_cfg.setdefault("class_name", "MMPDiscriminator")
        mmp_ids = discriminator_cfg.pop("mmp_ids", None)
        if mmp_ids is None:
            mmp_ids = cfg["algorithm"].pop("mmp_ids", None)
        if mmp_ids is None:
            mmp_ids = _resolve_mmp_ids(obs, cfg["obs_groups"]["mmp_id"])
        mmp_ids = sorted(int(mmp_id) for mmp_id in mmp_ids)
        if len(mmp_ids) == 0:
            raise ValueError("ParkourPPO could not resolve any MMP discriminator ids.")

        discriminator_cfg.setdefault("disc_obs_steps", disc_obs_steps)
        discriminator_cfg.setdefault("disc_obs_dim", disc_obs_dim)
        discriminator_cfg.setdefault("policy_obs_groups", cfg["obs_groups"]["mmp"])
        discriminator_cfg.setdefault("expert_obs_groups", cfg["obs_groups"]["mmp_expert"])
        discriminator_class: type[MMPDiscriminator] = resolve_callable(discriminator_cfg.pop("class_name"))  # type: ignore
        discriminators = {
            mmp_id: discriminator_class(device=device, **copy.deepcopy(discriminator_cfg)) for mmp_id in mmp_ids
        }
        print(f"MMP Discriminators: {nn.ModuleDict({str(k): v for k, v in discriminators.items()})}")

        buffer_size = cfg["algorithm"].pop("mmp_replay_buffer_size", None)
        if buffer_size is None:
            buffer_size = cfg["algorithm"].pop("mmp_buffer_size", cfg["num_steps_per_env"] * env.num_envs)
        disc_buffers = {
            mmp_id: ReplayBuffer(disc_obs_steps * disc_obs_dim, buffer_size, device=device) for mmp_id in mmp_ids
        }

        min_std = cfg.get("min_normalized_std")
        if min_std is not None and not isinstance(min_std, torch.Tensor):
            min_std = torch.tensor(min_std, device=device, dtype=torch.float32)

        alg = alg_class(
            actor,
            critic,
            cnn,
            storage,
            discriminators,
            disc_buffers,
            critic_cnn=critic_cnn,
            mmp_id_obs_group=cfg["obs_groups"]["mmp_id"][0],
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


def _get_mmp_ids(obs: TensorDict, obs_group: str) -> torch.Tensor:
    """Return discriminator routing ids as a flat int64 tensor."""
    if obs_group not in obs:
        raise KeyError(f"Observation group '{obs_group}' was not found. Available observations: {list(obs.keys())}.")
    return obs[obs_group].reshape(-1).long()


def _resolve_mmp_ids(obs: TensorDict, obs_groups: list[str]) -> list[int]:
    """Resolve routed discriminator ids from the initial observation batch."""
    if len(obs_groups) != 1:
        raise ValueError("ParkourPPO expects cfg['obs_groups']['mmp_id'] to contain exactly one observation group.")
    mmp_ids = _get_mmp_ids(obs, obs_groups[0])
    return sorted(int(mmp_id) for mmp_id in torch.unique(mmp_ids).detach().cpu().tolist())


def _resolve_disc_obs_shape(obs: TensorDict, obs_groups: list[str]) -> tuple[int, int]:
    """Return common history length and concatenated feature dim for MMP discriminator observations."""
    disc_obs_steps = -1
    disc_obs_dim = 0
    for obs_group in obs_groups:
        if obs_group not in obs:
            raise KeyError(f"Observation group '{obs_group}' was not found. Available observations: {list(obs.keys())}.")
        obs_tensor = obs[obs_group]
        if obs_tensor.ndim != 3:
            raise ValueError(
                "ParkourPPO MMP discriminator observations must have shape [num_envs, history_steps, obs_dim], "
                f"got {tuple(obs_tensor.shape)} for '{obs_group}'."
            )
        if disc_obs_steps == -1:
            disc_obs_steps = obs_tensor.shape[1]
        elif disc_obs_steps != obs_tensor.shape[1]:
            raise ValueError("All ParkourPPO MMP discriminator observation groups must have the same history length.")
        disc_obs_dim += obs_tensor.shape[-1]
    return disc_obs_steps, disc_obs_dim
