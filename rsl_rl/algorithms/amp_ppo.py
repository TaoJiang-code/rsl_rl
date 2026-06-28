# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import Discriminator, EmpiricalNormalization
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import AMPLoader, compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class AMPPPO(PPO):
    """PPO with adversarial motion prior rewards."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: Discriminator,
        amp_data: AMPLoader,
        amp_normalizer: nn.Module | None = None,
        amp_replay_buffer_size: int = 100000,
        amp_loss_coef: float = 1.0,
        amp_grad_penalty_coef: float = 10.0,
        min_std: float | list[float] | torch.Tensor | None = None,
        **ppo_kwargs,
    ) -> None:
        """Initialize AMPPPO."""
        super().__init__(actor, critic, storage, **ppo_kwargs)

        self.discriminator = discriminator.to(self.device)
        self._raw_discriminator = self.discriminator
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer.to(self.device) if amp_normalizer is not None else None
        self.amp_storage = ReplayBuffer(amp_data.observation_dim, amp_replay_buffer_size, self.device)
        self.amp_loss_coef = amp_loss_coef
        self.amp_grad_penalty_coef = amp_grad_penalty_coef
        self.min_std = min_std
        self._amp_obs: torch.Tensor | None = None

        optimizer_name = ppo_kwargs.get("optimizer", "adam")
        learning_rate = ppo_kwargs.get("learning_rate", self.learning_rate)
        params = chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters())
        self.optimizer = resolve_optimizer(optimizer_name)(params, lr=learning_rate)  # type: ignore

    def act(self, obs: TensorDict, amp_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Sample actions and remember the AMP state belonging to this transition."""
        if amp_obs is None:
            amp_obs = self.extract_amp_obs(obs, {})
        self._amp_obs = amp_obs.detach()
        return super().act(obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
        amp_obs: torch.Tensor | None = None,
    ) -> None:
        """Record one environment step, including the AMP policy transition."""
        if amp_obs is None:
            amp_obs = self.extract_amp_obs(obs, extras)
        if self._amp_obs is None:
            raise RuntimeError("AMPPPO.process_env_step() was called before AMPPPO.act().")

        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        self.amp_storage.insert(self._amp_obs.to(self.device), amp_obs.to(self.device))
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self._amp_obs = None
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_amp_reward(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        task_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the reward used for PPO from AMP state transitions."""
        return self.discriminator.predict_reward(
            amp_obs.to(self.device),
            next_amp_obs.to(self.device),
            task_reward.to(self.device),
            normalizer=self.amp_normalizer,
        )[0]

    def update(self) -> dict[str, float]:  # noqa: C901
        """Run PPO and discriminator updates over stored batches."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_penalty = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
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
        if self.amp_normalizer is not None:
            self.amp_normalizer.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        super().eval_mode()
        self.discriminator.eval()
        if self.amp_normalizer is not None:
            self.amp_normalizer.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self._raw_discriminator.state_dict()
        if self.amp_normalizer is not None:
            saved_dict["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("discriminator", True):
            self._raw_discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        if self.amp_normalizer is not None and "amp_normalizer_state_dict" in loaded_dict:
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"], strict=strict)
        return load_iteration

    def compile(self, mode: str | None = None) -> None:
        """Compile actor, critic, and discriminator if requested."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        self.discriminator = compile_model(self._raw_discriminator, mode)  # type: ignore

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [
            self._raw_actor.state_dict(),
            self._raw_critic.state_dict(),
            self._raw_discriminator.state_dict(),
        ]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        self._raw_discriminator.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        all_params = chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters())
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
    def extract_amp_obs(obs: TensorDict | torch.Tensor, extras: dict) -> torch.Tensor:
        """Extract AMP observations from common IsaacLab/RSL-RL locations."""
        if isinstance(obs, TensorDict) and "amp" in obs:
            return obs["amp"]
        observations = extras.get("observations", {})
        if isinstance(observations, dict) and "amp" in observations:
            return observations["amp"]
        if "amp_obs" in extras:
            return extras["amp_obs"]
        raise KeyError(
            "AMP observations were not found. Provide obs['amp'], extras['observations']['amp'], or extras['amp_obs']."
        )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """Construct the AMPPPO algorithm."""
        _migrate_legacy_policy_cfg(cfg)
        cfg.setdefault("obs_groups", {})

        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

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

        alg: AMPPPO = alg_class(
            actor,
            critic,
            storage,
            discriminator,
            amp_data,
            amp_normalizer=amp_normalizer,
            min_std=min_std,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


def _migrate_legacy_policy_cfg(cfg: dict) -> None:
    """Convert IsaacLab 2.1 style policy config to current actor/critic configs."""
    if "actor" in cfg and "critic" in cfg:
        return
    policy_cfg = cfg.pop("policy", None)
    if policy_cfg is None:
        return

    actor_hidden_dims = policy_cfg.pop("actor_hidden_dims", policy_cfg.pop("hidden_dims", [256, 256, 256]))
    critic_hidden_dims = policy_cfg.pop("critic_hidden_dims", actor_hidden_dims)
    activation = policy_cfg.pop("activation", "elu")
    class_name = policy_cfg.pop("class_name", "MLPModel")
    rnn_cfg = {}
    if class_name == "ActorCriticRecurrent":
        class_name = "RNNModel"
        rnn_cfg = {
            "rnn_type": policy_cfg.pop("rnn_type", "lstm"),
            "rnn_hidden_dim": policy_cfg.pop("rnn_hidden_dim", 256),
            "rnn_num_layers": policy_cfg.pop("rnn_num_layers", 1),
        }
    elif class_name == "ActorCritic":
        class_name = "MLPModel"

    distribution_cfg = policy_cfg.pop("distribution_cfg", None)
    if distribution_cfg is None:
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": policy_cfg.pop("init_noise_std", 1.0),
            "std_type": policy_cfg.pop("noise_std_type", "scalar"),
        }

    cfg["actor"] = {
        "class_name": class_name,
        "hidden_dims": actor_hidden_dims,
        "activation": activation,
        "distribution_cfg": distribution_cfg,
        "obs_normalization": cfg.get("empirical_normalization", False),
        **rnn_cfg,
    }
    cfg["critic"] = {
        "class_name": class_name,
        "hidden_dims": critic_hidden_dims,
        "activation": activation,
        "obs_normalization": cfg.get("empirical_normalization", False),
        **rnn_cfg,
    }


def _construct_amp_data(cfg: dict, env: VecEnv, device: str) -> AMPLoader:
    """Construct the expert AMP data loader."""
    motion_file = cfg.get("amp_motion_files", cfg.get("amp_motion_file"))
    if motion_file is None:
        raise KeyError("AMP requires 'amp_motion_files' or 'amp_motion_file' in the runner config.")
    body_names = cfg["amp_body_names"]
    anchor_name = cfg["amp_anchor_name"]
    all_body_names = cfg.get("amp_all_body_names") or _resolve_env_body_names(env)
    return AMPLoader(
        motion_file=motion_file,
        body_names=body_names,
        anchor_name=anchor_name,
        all_body_names=all_body_names,
        device=device,
    )


def _resolve_env_body_names(env: VecEnv) -> list[str] | None:
    """Best-effort resolution of IsaacLab robot body names."""
    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    robot = None
    if scene is not None:
        try:
            robot = scene["robot"]
        except Exception:
            robot = getattr(scene, "articulations", {}).get("robot")
    if robot is None:
        robot = getattr(unwrapped, "robot", None)
    if robot is None:
        return None
    if hasattr(robot, "body_names"):
        return list(robot.body_names)
    data = getattr(robot, "data", None)
    if data is not None and hasattr(data, "body_names"):
        return list(data.body_names)
    return None


def _resolve_discriminator_cfg(cfg: dict, amp_obs_dim: int) -> dict:
    """Resolve discriminator configuration from current or legacy AMP keys."""
    discr_cfg = dict(cfg.get("discriminator", {}))
    discr_cfg.setdefault("class_name", "Discriminator")
    discr_cfg.setdefault("input_dim", 2 * amp_obs_dim)
    discr_cfg.setdefault("reward_coef", cfg.get("amp_reward_coef", 1.0))
    discr_cfg.setdefault("hidden_dims", cfg.get("amp_discr_hidden_dims", [1024, 512]))
    discr_cfg.setdefault("task_reward_lerp", cfg.get("amp_task_reward_lerp", 0.0))
    return discr_cfg
