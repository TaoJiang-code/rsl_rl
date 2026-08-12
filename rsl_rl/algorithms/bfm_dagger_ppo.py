# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BFM-style DAgger-regularized PPO.

This algorithm is intended for the Light-Loco-Parkour style setup:

    L_total = L_PPO + dagger_loss_coef * L_DAgger

The teacher policy is frozen. The student actor can use ``BFMActorModel`` to
match the Kitov-style pipeline:

    expert/reference obs -> frozen backward -> z
    policy obs + z -> MLP -> action
"""

from __future__ import annotations

import copy
import sys
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups

from .ppo import PPO


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

        if kitov_root is not None:
            kitov_root_path = str(Path(kitov_root).resolve())
            if kitov_root_path not in sys.path:
                sys.path.insert(0, kitov_root_path)

        from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

        self.model = load_model_from_checkpoint_dir(str(self.checkpoint_dir), device=device)
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

    teacher: MLPModel
    """Frozen teacher actor."""

    teacher_loaded: bool = False
    """Whether teacher weights have been loaded from a checkpoint."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        dagger_loss_coef: float = 1.0,
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
            dagger_loss_coef: Initial DAgger loss coefficient.
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

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.dagger_loss_coef * dagger_loss
            )

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
            "dagger_coef": self.dagger_loss_coef,
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
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load student actor, critic, and optimizer states."""
        return super().load(loaded_dict, load_cfg, strict)

    def load_teacher(self, checkpoint_path: str, strict: bool = True) -> None:
        """Load frozen teacher actor weights from a torch checkpoint.

        Kitov checkpoints can wrap the actor under project-specific keys. This
        method accepts common actor state-dict keys and otherwise treats the
        checkpoint itself as the teacher state dict.
        """
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
        teacher_actor_class: type[MLPModel] = resolve_callable(teacher_actor_cfg.pop("class_name"))  # type: ignore
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
        teacher: MLPModel = teacher_actor_class(
            obs, cfg["obs_groups"], teacher_obs_set, env.num_actions, **teacher_actor_cfg
        ).to(device)
        print(f"Teacher Actor Model: {teacher}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg: BFMDAggerPPO = alg_class(
            actor, critic, teacher, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        if teacher_checkpoint_path is not None:
            alg.load_teacher(teacher_checkpoint_path, strict=teacher_checkpoint_strict)

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
