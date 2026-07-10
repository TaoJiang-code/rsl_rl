# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import ContextVAEModel, MLPModel
from rsl_rl.modules import Distribution, HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class _DWAQActorModel(nn.Module):
    """Append a ContextVAE code to observations before evaluating an actor head."""

    def __init__(
        self,
        actor: MLPModel,
        context_vae: ContextVAEModel,
        context_latent_key: str,
        detach_context_for_actor: bool = True,
    ) -> None:
        super().__init__()
        if actor.is_recurrent:
            raise ValueError("DWAQPPO currently supports feed-forward actor models only.")
        self.actor = actor
        self.context_latent_key = context_latent_key
        self.detach_context_for_actor = detach_context_for_actor
        object.__setattr__(self, "_context_vae", context_vae)

    @property
    def is_recurrent(self) -> bool:
        """Return whether the wrapped actor is recurrent."""
        return self.actor.is_recurrent

    def set_context_vae(self, context_vae: ContextVAEModel) -> None:
        """Update the context VAE reference used by this wrapper."""
        object.__setattr__(self, "_context_vae", context_vae)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Evaluate the actor on observations augmented with the context code."""
        augmented_obs = self._augment_observations(obs, deterministic=not stochastic_output)
        return self.actor(
            augmented_obs,
            masks=masks,
            hidden_state=hidden_state,
            stochastic_output=stochastic_output,
        )

    def _augment_observations(self, obs: TensorDict, deterministic: bool) -> TensorDict:
        """Return a shallow TensorDict with the ContextVAE code appended."""
        code = self._context_vae.encode(obs, deterministic=deterministic)
        if self.detach_context_for_actor:
            code = code.detach()
        data = {key: value for key, value in obs.items()}
        data[self.context_latent_key] = code
        return TensorDict(data, batch_size=obs.batch_size, device=obs.device)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update actor observation-normalization statistics on augmented observations."""
        with torch.no_grad():
            self.actor.update_normalization(self._augment_observations(obs, deterministic=True))

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset wrapped actor state."""
        self.actor.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return wrapped actor hidden state."""
        return self.actor.get_hidden_state()

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach wrapped actor hidden state."""
        self.actor.detach_hidden_state(dones)

    @property
    def distribution(self) -> Distribution | None:
        """Return the wrapped actor distribution."""
        return self.actor.distribution

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the current action mean."""
        return self.actor.output_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the current action standard deviation."""
        return self.actor.output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the current action entropy."""
        return self.actor.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return current action-distribution parameters."""
        return self.actor.output_distribution_params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute action log probabilities."""
        return self.actor.get_output_log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute action-distribution KL divergence."""
        return self.actor.get_kl_divergence(old_params, new_params)


class DWAQPPO(PPO):
    """PPO with a separately optimized DWAQ ContextVAE."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        context_vae: ContextVAEModel,
        storage: RolloutStorage,
        context_latent_key: str = "context_vae_code",
        velocity_obs_groups: list[str] | tuple[str, ...] = ("velocity",),
        reconstruction_obs_groups: list[str] | tuple[str, ...] = ("policy",),
        beta: float = 1.0,
        vae_learning_rate: float = 1.0e-3,
        vae_optimizer: str = "adam",
        detach_context_for_actor: bool = True,
        **ppo_kwargs: Any,
    ) -> None:
        """Initialize DWAQ PPO with independent actor, critic, and ContextVAE modules."""
        if critic.is_recurrent:
            raise ValueError("DWAQPPO currently supports feed-forward critic models only.")

        self.context_vae = context_vae.to(ppo_kwargs.get("device", "cpu"))
        actor_model = _DWAQActorModel(
            actor=actor,
            context_vae=self.context_vae,
            context_latent_key=context_latent_key,
            detach_context_for_actor=detach_context_for_actor,
        )
        super().__init__(actor_model, critic, storage, **ppo_kwargs)

        self.context_vae = context_vae.to(self.device)
        self._raw_context_vae = self.context_vae
        self._raw_actor.set_context_vae(self.context_vae)
        self.actor_head = self._raw_actor.actor
        self.context_latent_key = context_latent_key
        self.velocity_obs_groups = list(velocity_obs_groups)
        self.reconstruction_obs_groups = list(reconstruction_obs_groups)
        self.beta = beta
        self.vae_optimizer = resolve_optimizer(vae_optimizer)(self.context_vae.parameters(), lr=vae_learning_rate)  # type: ignore

        self._next_reconstruction_key = "_dwaq_next_reconstruction"
        self._live_mask_key = "_dwaq_live_mask"
        self._reconstruction_mask_key = "_dwaq_reconstruction_mask"

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record one step and update actor, critic, and ContextVAE normalizers."""
        self.context_vae.update_normalization(obs)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:  # noqa: C901
        """Run PPO updates and a separate ContextVAE reconstruction/velocity/KL update."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_autoencoder_loss = 0.0
        mean_velocity_loss = 0.0
        mean_reconstruction_loss = 0.0
        mean_kl_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        self._prepare_context_vae_targets()

        try:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

            for batch in generator:
                original_batch_size = batch.observations.batch_size[0]  # type: ignore
                vae_observations = batch.observations[:original_batch_size]  # type: ignore

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

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                rnd_loss = (
                    self.rnd.compute_loss(batch.observations[:original_batch_size])  # type: ignore
                    if self.rnd
                    else None
                )

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

                autoencoder_loss, velocity_loss, reconstruction_loss, kl_loss = self._compute_context_vae_loss(
                    vae_observations
                )
                self.vae_optimizer.zero_grad()
                autoencoder_loss.backward()
                if self.is_multi_gpu:
                    self._reduce_context_vae_parameters()
                nn.utils.clip_grad_norm_(self.context_vae.parameters(), self.max_grad_norm)
                self.vae_optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.mean().item()
                mean_autoencoder_loss += autoencoder_loss.item()
                mean_velocity_loss += velocity_loss.item()
                mean_reconstruction_loss += reconstruction_loss.item()
                mean_kl_loss += kl_loss.item()
                if mean_rnd_loss is not None:
                    mean_rnd_loss += rnd_loss.item()
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss += symmetry_loss.item()
        finally:
            self._clear_context_vae_targets()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "autoencoder": mean_autoencoder_loss / num_updates,
            "context_velocity": mean_velocity_loss / num_updates,
            "context_reconstruction": mean_reconstruction_loss / num_updates,
            "context_kl": mean_kl_loss / num_updates,
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        self.storage.clear()
        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for all learnable modules."""
        super().train_mode()
        self.context_vae.train()

    def eval_mode(self) -> None:
        """Set eval mode for all learnable modules."""
        super().eval_mode()
        self.context_vae.eval()

    def save(self) -> dict:
        """Return a dict of all DWAQ models and optimizers."""
        saved_dict = super().save()
        saved_dict["context_vae_state_dict"] = self._raw_context_vae.state_dict()
        saved_dict["vae_optimizer_state_dict"] = self.vae_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified DWAQ models and optimizer state."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("context_vae", True):
            self._raw_context_vae.load_state_dict(loaded_dict["context_vae_state_dict"], strict=strict)
        if load_cfg is None or load_cfg.get("vae_optimizer", True):
            self.vae_optimizer.load_state_dict(loaded_dict["vae_optimizer_state_dict"])
        return load_iteration

    def compile(self, mode: str | None = None) -> None:
        """Compile actor, critic, and context VAE if requested."""
        self.context_vae = compile_model(self._raw_context_vae, mode)  # type: ignore
        self._raw_actor.set_context_vae(self.context_vae)
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    def broadcast_parameters(self) -> None:
        """Broadcast actor, critic, context VAE, and optional RND parameters."""
        super().broadcast_parameters()
        model_params = [self._raw_context_vae.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_context_vae.load_state_dict(model_params[0])

    def _prepare_context_vae_targets(self) -> None:
        """Add next-step reconstruction targets and masks to rollout observations."""
        reconstruction = torch.cat(
            [self.storage.observations[key] for key in self.reconstruction_obs_groups],
            dim=-1,
        )
        self.storage.observations[self._next_reconstruction_key] = torch.cat(
            [reconstruction[1:], reconstruction[-1:]],
            dim=0,
        )

        live_mask = 1.0 - self.storage.dones.float()
        reconstruction_mask = live_mask.clone()
        reconstruction_mask[-1] = 0.0
        self.storage.observations[self._live_mask_key] = live_mask
        self.storage.observations[self._reconstruction_mask_key] = reconstruction_mask

    def _clear_context_vae_targets(self) -> None:
        """Remove temporary ContextVAE target tensors from rollout observations."""
        for key in (self._next_reconstruction_key, self._live_mask_key, self._reconstruction_mask_key):
            if key in self.storage.observations:
                del self.storage.observations[key]

    def _compute_context_vae_loss(
        self,
        obs: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DWAQ ContextVAE velocity, reconstruction, and KL losses."""
        vae_out = self.context_vae(obs)
        velocity_target = torch.cat([obs[key] for key in self.velocity_obs_groups], dim=-1).detach()
        reconstruction_target = obs[self._next_reconstruction_key].detach()
        live_mask = obs[self._live_mask_key].detach()
        reconstruction_mask = obs[self._reconstruction_mask_key].detach()

        velocity_loss = nn.functional.mse_loss(vae_out.code_vel * live_mask, velocity_target * live_mask)
        reconstruction_loss = nn.functional.mse_loss(
            vae_out.reconstruction * reconstruction_mask,
            reconstruction_target * reconstruction_mask,
        )
        kl_loss = -0.5 * torch.mean(
            torch.sum(
                1.0 + vae_out.logvar_latent - vae_out.mean_latent.pow(2) - vae_out.logvar_latent.exp(),
                dim=-1,
            )
            * live_mask.squeeze(-1)
        )
        autoencoder_loss = velocity_loss + reconstruction_loss + self.beta * kl_loss
        return autoencoder_loss, velocity_loss, reconstruction_loss, kl_loss

    def _reduce_context_vae_parameters(self) -> None:
        """Collect ContextVAE gradients from all GPUs and average them."""
        params = list(self.context_vae.parameters())
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
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DWAQPPO:
        """Construct DWAQPPO from separate actor, critic, and ContextVAE configs."""
        cfg.setdefault("obs_groups", {})
        cfg.setdefault("multi_gpu", None)

        _resolve_dwaq_obs_groups(obs, cfg["obs_groups"])

        alg_class: type[DWAQPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        context_vae_class: type[ContextVAEModel] = resolve_callable(cfg["context_vae"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        for required_obs_set in ("context_vae", "velocity"):
            if required_obs_set not in cfg["obs_groups"]:
                raise KeyError(
                    f"DWAQPPO requires cfg['obs_groups']['{required_obs_set}']. "
                    "For common IsaacLab-style names, use context_vae=['obs_history'] and velocity=['velocity']."
                )
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        base_actor_obs_groups = list(cfg["obs_groups"]["actor"])
        context_latent_key = cfg["algorithm"].get("context_latent_key", "context_vae_code")
        context_code_dim = cfg["context_vae"].get(
            "code_dim",
            cfg["context_vae"].get("cenet_out_dim", 19),
        )
        context_sample = torch.zeros(*obs.batch_size, context_code_dim, device=obs.device, dtype=torch.float32)
        actor_sample_obs = _add_context_latent(obs, context_latent_key, context_sample)
        actor_obs_groups = {key: list(value) for key, value in cfg["obs_groups"].items()}
        if context_latent_key not in actor_obs_groups["actor"]:
            actor_obs_groups["actor"].append(context_latent_key)

        actor: MLPModel = actor_class(
            actor_sample_obs,
            actor_obs_groups,
            "actor",
            env.num_actions,
            **cfg["actor"],
        ).to(device)
        print(f"DWAQ Actor Model: {actor}")

        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"DWAQ Critic Model: {critic}")

        context_vae_cfg = dict(cfg["context_vae"])
        if "cenet_out_dim" in context_vae_cfg and "code_dim" not in context_vae_cfg:
            context_vae_cfg["code_dim"] = context_vae_cfg.pop("cenet_out_dim")
        else:
            context_vae_cfg.pop("cenet_out_dim", None)
        context_vae_output_dim = context_vae_cfg.pop("output_dim", _sum_obs_dim(obs, base_actor_obs_groups))
        context_vae = context_vae_class(
            obs,
            cfg["obs_groups"],
            "context_vae",
            context_vae_output_dim,
            **context_vae_cfg,
        ).to(device)
        print(f"DWAQ ContextVAE Model: {context_vae}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg = alg_class(
            actor,
            critic,
            context_vae,
            storage,
            velocity_obs_groups=cfg["obs_groups"]["velocity"],
            reconstruction_obs_groups=base_actor_obs_groups,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


def _resolve_dwaq_obs_groups(obs: TensorDict, obs_groups: dict[str, list[str]]) -> None:
    """Fill common DWAQ observation groups before full validation."""
    if "context_vae" not in obs_groups and "obs_history" in obs:
        obs_groups["context_vae"] = ["obs_history"]
    if "velocity" not in obs_groups and "velocity" in obs:
        obs_groups["velocity"] = ["velocity"]


def _add_context_latent(obs: TensorDict, key: str, latent: torch.Tensor) -> TensorDict:
    """Return a shallow TensorDict augmented with one context latent tensor."""
    data = {obs_key: value for obs_key, value in obs.items()}
    data[key] = latent
    return TensorDict(data, batch_size=obs.batch_size, device=obs.device)


def _sum_obs_dim(obs: TensorDict, obs_groups: list[str]) -> int:
    """Return the concatenated feature dimension for 1D observation groups."""
    obs_dim = 0
    for obs_group in obs_groups:
        if len(obs[obs_group].shape) != 2:
            raise ValueError(
                "DWAQPPO reconstruction observations must be flattened 1D tensors, "
                f"got shape {obs[obs_group].shape} for '{obs_group}'."
            )
        obs_dim += obs[obs_group].shape[-1]
    return obs_dim
