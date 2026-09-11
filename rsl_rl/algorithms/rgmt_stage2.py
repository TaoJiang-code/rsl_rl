# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from rsl_rl.algorithms.rgmt import RGMT


class RGMTStageII(RGMT):
    """Stage-II RGMT with acquisition PPO and PACE-style consolidation.

    Acquisition samples are optimized with PPO on challenging clips. Consolidation
    samples align the policy action to the frozen Stage-I reference policy on
    mastered clips, with the dynamic consolidation weight from the paper.
    """

    def __init__(
        self,
        *args,
        lambda_base: float = 0.3,
        lambda_kappa: float = 5.0,
        lambda_rho_ref: float = 0.6,
        lambda_beta: float = 0.99,
        lambda_con_max: float = 1.0,
        consolidation_obs_key: str = "stage2_is_acquisition",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_base = lambda_base
        self.lambda_kappa = lambda_kappa
        self.lambda_rho_ref = lambda_rho_ref
        self.lambda_beta = lambda_beta
        self.lambda_con_max = lambda_con_max
        self.consolidation_obs_key = consolidation_obs_key

        self.reference_actor = copy.deepcopy(self._raw_actor).to(self.device)
        self.reference_actor.eval()
        self.reference_actor.requires_grad_(False)
        self.valid_acq_ratio_ema: float | None = None
        self.lambda_con = 0.0

    def sync_reference_from_actor(self) -> None:
        """Reset the frozen reference policy to the current actor weights."""
        self.reference_actor.load_state_dict(self._raw_actor.state_dict())
        self.reference_actor.eval()
        self.reference_actor.requires_grad_(False)

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["reference_actor_state_dict"] = self.reference_actor.state_dict()
        saved_dict["stage2_valid_acq_ratio_ema"] = self.valid_acq_ratio_ema
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "reference_actor_state_dict" in loaded_dict:
            self.reference_actor.load_state_dict(loaded_dict["reference_actor_state_dict"], strict=strict)
            self.valid_acq_ratio_ema = loaded_dict.get("stage2_valid_acq_ratio_ema")
        elif "teacher_actor_state_dict" in loaded_dict:
            self.reference_actor.load_state_dict(loaded_dict["teacher_actor_state_dict"], strict=strict)
            self.valid_acq_ratio_ema = loaded_dict.get("stage2_valid_acq_ratio_ema")
        else:
            self.sync_reference_from_actor()
        return load_iteration

    def update(self) -> dict[str, float]:
        """Run Stage-II acquisition PPO and consolidation updates."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_consolidation_loss = 0
        mean_valid_acq_ratio = self._compute_valid_acquisition_ratio()
        self.lambda_con = self._compute_lambda_con(mean_valid_acq_ratio)

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

            original_observations = batch.observations[:original_batch_size]
            is_acquisition, is_consolidation = self._stage2_masks(original_observations)

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
                    if torch.any(is_acquisition):
                        kl_mean = torch.mean(kl[is_acquisition])
                    else:
                        kl_mean = torch.zeros((), device=self.device)

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
            surrogate_loss_all = torch.max(surrogate, surrogate_clipped)
            if torch.any(is_acquisition):
                surrogate_loss = surrogate_loss_all[is_acquisition].mean()
            else:
                surrogate_loss = torch.zeros((), device=self.device)

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss_all = torch.max(value_losses, value_losses_clipped).reshape(-1)
            else:
                value_loss_all = (batch.returns - values).pow(2).reshape(-1)
            if torch.any(is_acquisition):
                value_loss = value_loss_all[is_acquisition].mean()
                entropy_loss = entropy[is_acquisition].mean()
            else:
                value_loss = torch.zeros((), device=self.device)
                entropy_loss = torch.zeros((), device=self.device)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
            consolidation_loss = self._compute_consolidation_loss(original_observations, is_consolidation)
            loss = loss + self.lambda_con * consolidation_loss

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
            mean_entropy += entropy_loss.item()
            mean_consolidation_loss += consolidation_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_consolidation_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "stage2_consolidation": mean_consolidation_loss,
            "stage2_lambda_con": self.lambda_con,
            "stage2_valid_acq_ratio": mean_valid_acq_ratio,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.storage.clear()
        return loss_dict

    def _stage2_masks(self, observations) -> tuple[torch.Tensor, torch.Tensor]:
        if self.consolidation_obs_key not in observations:
            raise KeyError(
                f"RGMTStageII requires observation key '{self.consolidation_obs_key}'. "
                "Use RGMTStageIIRunner with an RGMTStageIIMotionCommand environment."
            )
        is_acquisition = observations[self.consolidation_obs_key].reshape(-1) > 0.5
        return is_acquisition, ~is_acquisition

    def _compute_consolidation_loss(self, observations, consolidation_mask: torch.Tensor) -> torch.Tensor:
        if not torch.any(consolidation_mask):
            return torch.zeros((), device=self.device)

        consolidation_obs = observations[consolidation_mask]
        student_actions = self.actor(consolidation_obs)
        with torch.no_grad():
            reference_actions = self.reference_actor(consolidation_obs)
        return torch.mean((student_actions - reference_actions).pow(2))

    def _compute_valid_acquisition_ratio(self) -> float:
        if self.consolidation_obs_key not in self.storage.observations:
            return 1.0
        is_acquisition = self.storage.observations[self.consolidation_obs_key].reshape(-1) > 0.5
        not_done = (1.0 - self.storage.dones.reshape(-1).float()) > 0.5
        valid_acquisition = is_acquisition & not_done
        valid_consolidation = (~is_acquisition) & not_done
        valid_total = valid_acquisition.sum() + valid_consolidation.sum()
        if valid_total.item() == 0:
            return 0.0
        return float((valid_acquisition.sum().float() / valid_total.float()).item())

    def _compute_lambda_con(self, valid_acq_ratio: float) -> float:
        if self.valid_acq_ratio_ema is None:
            self.valid_acq_ratio_ema = valid_acq_ratio
        else:
            self.valid_acq_ratio_ema = (
                self.lambda_beta * self.valid_acq_ratio_ema + (1.0 - self.lambda_beta) * valid_acq_ratio
            )
        value = self.lambda_base + self.lambda_kappa * max(0.0, self.valid_acq_ratio_ema - self.lambda_rho_ref)
        return float(min(self.lambda_con_max, max(0.0, value)))
