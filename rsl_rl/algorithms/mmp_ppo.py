# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import MMPDiscriminator
from rsl_rl.storage import CircularBuffer, RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class MMPPPO(PPO):
    """PPO with an environment-fed multimodal motion prior discriminator."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: MMPDiscriminator,
        disc_obs_buffer: CircularBuffer,
        disc_expert_obs_buffer: CircularBuffer,
        mmp_loss_coef: float = 1.0,
        mmp_grad_penalty_coef: float = 10.0,
        discriminator_learning_rate: float | None = None,
        discriminator_optimizer: str = "adam",
        discriminator_max_grad_norm: float | None = None,
        min_std: float | list[float] | torch.Tensor | None = None,
        **ppo_kwargs,
    ) -> None:
        """Initialize MMPPPO."""
        super().__init__(actor, critic, storage, **ppo_kwargs)

        self.discriminator = discriminator.to(self.device)
        self._raw_discriminator = self.discriminator
        self.disc_obs_buffer = disc_obs_buffer
        self.disc_expert_obs_buffer = disc_expert_obs_buffer
        self.mmp_loss_coef = mmp_loss_coef
        self.mmp_grad_penalty_coef = mmp_grad_penalty_coef
        self.discriminator_max_grad_norm = discriminator_max_grad_norm or self.max_grad_norm
        self.min_std = min_std

        disc_lr = discriminator_learning_rate if discriminator_learning_rate is not None else self.learning_rate
        self.discriminator_optimizer = resolve_optimizer(discriminator_optimizer)(
            self.discriminator.parameters(),
            lr=disc_lr,
        )  # type: ignore

        self.mmp_rewards: torch.Tensor | None = None
        self.mmp_scores: torch.Tensor | None = None

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record one environment step using MMP rewards computed from env-fed expert observations."""
        disc_obs = self.discriminator.get_policy_obs(obs, flatten_history_dim=False)
        disc_expert_obs = self.discriminator.get_expert_obs(obs, flatten_history_dim=False)
        self.mmp_rewards, self.mmp_scores = self.discriminator.predict_reward(disc_obs, rewards.to(self.device))

        self.disc_obs_buffer.append(disc_obs.detach())
        self.disc_expert_obs_buffer.append(disc_expert_obs.detach())

        super().process_env_step(obs, self.mmp_rewards, dones, extras)

    def update(self) -> dict[str, float]:  # noqa: C901
        """Run PPO and MMP discriminator updates over stored rollouts."""
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

        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_expert_obs_generator = self.disc_expert_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        for batch, disc_obs_batch, disc_expert_obs_batch in zip(
            generator,
            disc_obs_generator,
            disc_expert_obs_generator,
        ):
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

            with torch.no_grad():
                self.discriminator.update_normalization(disc_obs_batch)
                self.discriminator.update_normalization(disc_expert_obs_batch)
                disc_obs_batch = self.discriminator.normalize_obs(disc_obs_batch)
                disc_expert_obs_batch = self.discriminator.normalize_obs(disc_expert_obs_batch)

            mmp_loss, policy_logits, expert_logits = self.discriminator.compute_loss(
                disc_obs_batch,
                disc_expert_obs_batch,
            )
            grad_penalty = self.discriminator.compute_grad_penalty(
                disc_expert_obs_batch,
                self.mmp_grad_penalty_coef,
            )
            discriminator_loss = self.mmp_loss_coef * (mmp_loss + grad_penalty)

            self.optimizer.zero_grad()
            ppo_loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()
                self._reduce_discriminator_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.discriminator_max_grad_norm)
            self.optimizer.step()
            self._clamp_min_std()
            if self.rnd:
                self.rnd.optimizer.step()
            self.discriminator_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_mmp_loss += mmp_loss.item()
            mean_grad_penalty += grad_penalty.item()
            mean_policy_pred += policy_logits.mean().item()
            mean_expert_pred += expert_logits.mean().item()
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
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Set eval mode for learnable models."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self._raw_discriminator.state_dict()
        saved_dict["discriminator_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("discriminator", True):
            self._raw_discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        if load_cfg is None or load_cfg.get("discriminator_optimizer", True):
            self.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
        return load_iteration

    def compile(self, mode: str | None = None) -> None:
        """Compile actor, critic, and discriminator if requested."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        self.discriminator = compile_model(self._raw_discriminator, mode)  # type: ignore

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        super().broadcast_parameters()
        model_params = [self._raw_discriminator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_discriminator.load_state_dict(model_params[0])

    def _reduce_discriminator_parameters(self) -> None:
        """Collect discriminator gradients from all GPUs and average them."""
        params = list(self.discriminator.parameters())
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

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MMPPPO:
        """Construct MMPPPO from environment-fed discriminator observation groups."""
        cfg.setdefault("obs_groups", {})
        cfg.setdefault("multi_gpu", None)

        alg_class: type[MMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        for required_obs_set in ("mmp", "mmp_expert"):
            if required_obs_set not in cfg["obs_groups"]:
                raise KeyError(f"MMPPPO requires cfg['obs_groups']['{required_obs_set}'].")
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        disc_obs_steps, disc_obs_dim = _resolve_disc_obs_shape(obs, cfg["obs_groups"]["mmp"])
        expert_obs_steps, expert_obs_dim = _resolve_disc_obs_shape(obs, cfg["obs_groups"]["mmp_expert"])
        if expert_obs_steps != disc_obs_steps or expert_obs_dim != disc_obs_dim:
            raise ValueError(
                "MMP policy and expert discriminator observations must have identical shapes, "
                f"got policy [T={disc_obs_steps}, D={disc_obs_dim}] and "
                f"expert [T={expert_obs_steps}, D={expert_obs_dim}]."
            )

        discriminator_cfg = dict(cfg.get("mmp_discriminator", cfg.get("discriminator", {})))
        discriminator_cfg.setdefault("class_name", "MMPDiscriminator")
        discriminator_cfg.setdefault("disc_obs_steps", disc_obs_steps)
        discriminator_cfg.setdefault("disc_obs_dim", disc_obs_dim)
        discriminator_cfg.setdefault("policy_obs_groups", cfg["obs_groups"]["mmp"])
        discriminator_cfg.setdefault("expert_obs_groups", cfg["obs_groups"]["mmp_expert"])
        discriminator_class: type[MMPDiscriminator] = resolve_callable(discriminator_cfg.pop("class_name"))  # type: ignore
        discriminator = discriminator_class(device=device, **discriminator_cfg)
        print(f"MMP Discriminator: {discriminator}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        buffer_size = cfg["algorithm"].pop("mmp_buffer_size", cfg["num_steps_per_env"])
        disc_obs_buffer = CircularBuffer(buffer_size, env.num_envs, device)
        disc_expert_obs_buffer = CircularBuffer(buffer_size, env.num_envs, device)

        min_std = cfg.get("min_normalized_std")
        if min_std is not None and not isinstance(min_std, torch.Tensor):
            min_std = torch.tensor(min_std, device=device, dtype=torch.float32)

        alg = alg_class(
            actor,
            critic,
            storage,
            discriminator,
            disc_obs_buffer,
            disc_expert_obs_buffer,
            min_std=min_std,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

def _resolve_disc_obs_shape(obs: TensorDict, obs_groups: list[str]) -> tuple[int, int]:
    """Return common history length and concatenated feature dim for discriminator observations."""
    disc_obs_steps = -1
    disc_obs_dim = 0
    for obs_group in obs_groups:
        if obs_group not in obs:
            raise KeyError(f"Observation group '{obs_group}' was not found. Available observations: {list(obs.keys())}.")
        obs_tensor = obs[obs_group]
        if obs_tensor.ndim != 3:
            raise ValueError(
                "MMP discriminator observations must have shape [num_envs, history_steps, obs_dim], "
                f"got {tuple(obs_tensor.shape)} for '{obs_group}'."
            )
        if disc_obs_steps == -1:
            disc_obs_steps = obs_tensor.shape[1]
        elif disc_obs_steps != obs_tensor.shape[1]:
            raise ValueError("All MMP discriminator observation groups must have the same history length.")
        disc_obs_dim += obs_tensor.shape[-1]
    return disc_obs_steps, disc_obs_dim
