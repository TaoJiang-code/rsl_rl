# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from collections.abc import Generator

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from tensordict import TensorDict

from rsl_rl.algorithms.rgmt import RGMT, RGMTActorModel
from rsl_rl.env import VecEnv


TRACKING_FAILURES_EXTRA = "tracking_failures"
STAR_GROUP = "star"


class StageIIBatch(RolloutStorage.Batch):
    """Mini-batch carrying the Stage-II acquisition mask."""

    def __init__(self, *args, acq_mask: torch.Tensor | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.acq_mask = acq_mask


def _wrapped_slice(perm: torch.Tensor, start: int, length: int) -> torch.Tensor:
    if length <= 0:
        return perm[:0]
    if perm.numel() == 0:
        raise ValueError("Cannot sample from an empty Stage-II index pool.")
    idx = torch.arange(start, start + length, device=perm.device) % perm.numel()
    return perm[idx]


class StageIIStarRolloutStorage(RolloutStorage):
    """Rollout storage with PACE role masks and STAR resampling."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        acquisition_ratio: float = 0.8,
        rho_topk: float = 0.05,
        rho_star: float = 0.25,
        use_star: bool = True,
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.acquisition_ratio = acquisition_ratio
        self.rho_topk = rho_topk
        self.rho_star = rho_star
        self.use_star = use_star
        self.env_split = int(round(num_envs * acquisition_ratio))

        flat = torch.arange(num_transitions_per_env * num_envs, device=device)
        env_of_flat = flat % num_envs
        self.acq_flat_idx = flat[env_of_flat < self.env_split]
        self.con_flat_idx = flat[env_of_flat >= self.env_split]
        self.raw_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.tracking_failures = torch.zeros(
            num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device
        )
        self.tracking_failure_sources = torch.zeros(num_transitions_per_env, dtype=torch.uint8, device=device)
        self.last_star_pool_size = 0

    def clear(self) -> None:
        super().clear()
        self.tracking_failures.zero_()
        self.tracking_failure_sources.zero_()

    def record_tracking_failures(self, failures: torch.Tensor) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow while recording tracking failures.")
        mask = failures.to(device=self.device, dtype=torch.bool).reshape(-1)
        if mask.shape != (self.num_envs,):
            raise ValueError(f"tracking failure mask must have shape {(self.num_envs,)}, got {tuple(mask.shape)}.")
        self.tracking_failures[self.step, :, 0].copy_(mask)
        self.tracking_failure_sources[self.step] = 1

    @staticmethod
    def _prefix_before_first_event(events: torch.Tensor) -> torch.Tensor:
        prior_events = torch.zeros_like(events, dtype=torch.long)
        prior_events[1:] = torch.cumsum(events.long(), dim=0)[:-1]
        return prior_events == 0

    def valid_sample_counts(self) -> tuple[int, int]:
        diagnostics = self.valid_sample_diagnostics()
        return int(diagnostics["valid_acq_samples"]), int(diagnostics["valid_con_samples"])

    def valid_sample_diagnostics(self) -> dict[str, int]:
        populated_steps = self.num_transitions_per_env if self.step == 0 else self.step
        sources = self.tracking_failure_sources[:populated_steps]
        if bool((sources == 0).any()):
            missing = int((sources == 0).sum().item())
            raise RuntimeError(
                f"Missing tracking failure masks for {missing}/{populated_steps} rollout steps. "
                f"Pass extras[{TRACKING_FAILURES_EXTRA!r}] from RGMTStageIIRunner."
            )

        failures = self.tracking_failures[:populated_steps].squeeze(-1)
        dones = self.dones[:populated_steps].squeeze(-1).bool()
        valid_failure = self._prefix_before_first_event(failures)
        valid_done = self._prefix_before_first_event(dones)
        non_failure_dones = dones & ~failures

        def role_counts(mask: torch.Tensor) -> tuple[int, int]:
            return (
                int(mask[:, : self.env_split].sum().item()),
                int(mask[:, self.env_split :].sum().item()),
            )

        n_acq, n_con = role_counts(valid_failure)
        done_acq, done_con = role_counts(dones)
        failure_acq, failure_con = role_counts(failures)
        non_failure_done_acq, non_failure_done_con = role_counts(non_failure_dones)
        done_prefix_acq, done_prefix_con = role_counts(valid_done)
        return {
            "valid_acq_samples": n_acq,
            "valid_con_samples": n_con,
            "failure_prefix_valid_acq_samples": n_acq,
            "failure_prefix_valid_con_samples": n_con,
            "combined_done_prefix_valid_acq_samples": done_prefix_acq,
            "combined_done_prefix_valid_con_samples": done_prefix_con,
            "valid_acq_gain_vs_combined_done": n_acq - done_prefix_acq,
            "valid_con_gain_vs_combined_done": n_con - done_prefix_con,
            "tracking_failure_acq_events": failure_acq,
            "tracking_failure_con_events": failure_con,
            "combined_done_acq_events": done_acq,
            "combined_done_con_events": done_con,
            "non_failure_done_acq_events": non_failure_done_acq,
            "non_failure_done_con_events": non_failure_done_con,
            "tracking_failure_explicit_steps": int((sources == 1).sum().item()),
            "tracking_failure_missing_steps": int((sources == 0).sum().item()),
        }

    def _star_meta(self) -> tuple[torch.Tensor, torch.Tensor]:
        if STAR_GROUP not in self.observations:
            weights = torch.ones(self.num_transitions_per_env * self.num_envs, device=self.device)
            bins = torch.zeros_like(weights, dtype=torch.long)
            return weights, bins
        meta = self.observations[STAR_GROUP].flatten(0, 1)
        return meta[:, 0], meta[:, 1].long()

    def _group_moments(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        selected = values[mask]
        count = int(selected.numel())
        if count == 0:
            zero = values.new_zeros(())
            return 0, zero, zero
        mean = selected.mean()
        if count < 2:
            return count, mean, values.new_zeros(())
        return count, mean, selected.std(unbiased=True)

    def normalize_acquisition_advantages(self, eps: float = 1e-8) -> None:
        adv = self.advantages.flatten(0, 1).squeeze(-1)
        acq = torch.zeros_like(adv, dtype=torch.bool)
        acq[self.acq_flat_idx] = True
        count, mean, std = self._group_moments(adv, acq)
        if count < 2:
            raise RuntimeError("Stage-II advantage normalization needs at least two acquisition rows.")
        normalized = torch.zeros_like(adv)
        normalized[acq] = (adv[acq] - mean) / (std + eps)
        self.advantages.copy_(normalized.view(self.num_transitions_per_env, self.num_envs, 1))

    def normalize_advantages_by_difficulty(self, eps: float = 1e-8) -> None:
        adv = self.advantages.flatten(0, 1).squeeze(-1)
        weights, _ = self._star_meta()
        acq = torch.zeros_like(adv, dtype=torch.bool)
        acq[self.acq_flat_idx] = True
        high = acq & (weights > 1.0)
        low = acq & ~high

        high_stats = self._group_moments(adv, high)
        low_stats = self._group_moments(adv, low)
        normalized = torch.zeros_like(adv)
        if high_stats[0] < 2 or low_stats[0] < 2:
            count, mean, std = self._group_moments(adv, acq)
            if count < 2:
                raise RuntimeError("Stage-II advantage normalization needs at least two acquisition rows.")
            normalized[acq] = (adv[acq] - mean) / (std + eps)
        else:
            for mask, (_, mean, std) in ((high, high_stats), (low, low_stats)):
                normalized[mask] = (adv[mask] - mean) / (std + eps)
        self.advantages.copy_(normalized.view(self.num_transitions_per_env, self.num_envs, 1))

    def fragment_ids(self) -> torch.Tensor:
        dones = self.dones.squeeze(-1).float()
        counter = torch.zeros_like(dones)
        counter[1:] = torch.cumsum(dones, dim=0)[:-1]
        env_ids = torch.arange(self.num_envs, device=self.device).expand_as(counter)
        return (env_ids * (self.num_transitions_per_env + 1) + counter.long()).flatten()

    def _build_star_pool(self) -> tuple[torch.Tensor, torch.Tensor]:
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        if not self.use_star:
            return empty, empty.float()

        weights, bin_ids = self._star_meta()
        raw_adv = self.raw_advantages.flatten(0, 1).squeeze(-1)
        fragments = self.fragment_ids()
        acq = torch.zeros_like(weights, dtype=torch.bool)
        acq[self.acq_flat_idx] = True
        high = acq & (weights > 1.0)
        if int(high.sum().item()) == 0:
            return empty, empty.float()

        stride = int(fragments.max().item()) + 1
        keys = bin_ids[high] * stride + fragments[high]
        uniq_keys, inverse = torch.unique(keys, return_inverse=True)
        counts = torch.zeros(uniq_keys.numel(), device=self.device)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float))
        sums = torch.zeros(uniq_keys.numel(), device=self.device)
        sums.index_add_(0, inverse, raw_adv[high])
        q = sums / counts.clamp_min(1.0)

        uniq_bin = torch.div(uniq_keys, stride, rounding_mode="floor")
        uniq_frag = uniq_keys % stride
        by_q = torch.argsort(q, descending=True)
        order = by_q[torch.argsort(uniq_bin[by_q], stable=True)]
        bins_sorted = uniq_bin[order]
        _, seg_counts = torch.unique_consecutive(bins_sorted, return_counts=True)
        seg_starts = torch.cumsum(seg_counts, 0) - seg_counts
        rank = torch.arange(order.numel(), device=self.device) - torch.repeat_interleave(seg_starts, seg_counts)
        k_b = torch.clamp(torch.ceil(self.rho_topk * seg_counts.float()).long(), min=1)
        keep = rank < torch.repeat_interleave(k_b, seg_counts)
        selected_frags = torch.unique(uniq_frag[order][keep])
        if selected_frags.numel() == 0:
            return empty, empty.float()

        in_pool = acq & torch.isin(fragments, selected_frags)
        pool = torch.nonzero(in_pool, as_tuple=False).squeeze(-1)
        if pool.numel() == 0:
            return empty, empty.float()

        pool_frags = fragments[pool]
        uniq_pf, pf_inv = torch.unique(pool_frags, return_inverse=True)
        pf_counts = torch.zeros(uniq_pf.numel(), device=self.device)
        pf_counts.index_add_(0, pf_inv, torch.ones_like(pf_inv, dtype=torch.float))
        pf_sums = torch.zeros(uniq_pf.numel(), device=self.device)
        pf_sums.index_add_(0, pf_inv, weights[pool])
        eta = pf_sums / pf_counts.clamp_min(1.0)
        omega = eta[pf_inv]
        if float(omega.sum().item()) <= 0.0:
            return empty, empty.float()
        return pool, omega

    def star_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[StageIIBatch, None, None]:
        if self.training_type != "rl":
            raise ValueError("STAR mini-batch generation is only available for RL training.")

        total = self.num_envs * self.num_transitions_per_env
        mini_batch_size = total // num_mini_batches
        n_acq_pool = int(self.acq_flat_idx.numel())
        n_con_pool = int(self.con_flat_idx.numel())
        if n_con_pool == 0:
            acq_per_batch, con_per_batch = mini_batch_size, 0
        else:
            acq_per_batch = min(
                max(int(round(mini_batch_size * n_acq_pool / total)), 1), mini_batch_size - 1
            )
            con_per_batch = mini_batch_size - acq_per_batch

        pool, omega = self._build_star_pool()
        self.last_star_pool_size = int(pool.numel())
        m_star = min(acq_per_batch, max(int(self.rho_star * acq_per_batch), 1)) if pool.numel() > 0 else 0

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_params = tuple(p.flatten(0, 1) for p in self.distribution_params)

        acq_mask_template = torch.zeros(acq_per_batch + con_per_batch, dtype=torch.bool, device=self.device)
        acq_mask_template[:acq_per_batch] = True

        for _ in range(num_epochs):
            acq_perm = self.acq_flat_idx[torch.randperm(n_acq_pool, device=self.device)]
            con_perm = (
                self.con_flat_idx[torch.randperm(n_con_pool, device=self.device)] if n_con_pool > 0 else None
            )
            for i in range(num_mini_batches):
                acq_idx = _wrapped_slice(acq_perm, i * acq_per_batch, acq_per_batch)
                if m_star > 0:
                    resampled = pool[torch.multinomial(omega, m_star, replacement=True)]
                    acq_idx = torch.cat([resampled, acq_idx[m_star:]])
                if con_perm is not None:
                    con_idx = _wrapped_slice(con_perm, i * con_per_batch, con_per_batch)
                    batch_idx = torch.cat([acq_idx, con_idx])
                else:
                    batch_idx = acq_idx
                yield StageIIBatch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_params),
                    acq_mask=acq_mask_template,
                )


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
        acquisition_ratio: float = 0.8,
        rho_topk: float = 0.05,
        rho_star: float = 0.25,
        use_star: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_base = lambda_base
        self.lambda_kappa = lambda_kappa
        self.lambda_rho_ref = lambda_rho_ref
        self.lambda_beta = lambda_beta
        self.lambda_con_max = lambda_con_max
        self.consolidation_obs_key = consolidation_obs_key
        self.acquisition_ratio = acquisition_ratio
        self.rho_topk = rho_topk
        self.rho_star = rho_star
        self.use_star = use_star

        self.reference_actor = copy.deepcopy(self._raw_actor).to(self.device)
        self.reference_actor.eval()
        self.reference_actor.requires_grad_(False)
        self.valid_acq_ratio_ema: float | None = None
        self.lambda_con = 0.0
        self._last_valid_sample_diagnostics: dict[str, int] = {}

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "RGMTStageII":
        cfg.setdefault("obs_groups", {})
        cfg.setdefault("multi_gpu", None)

        alg_class: type[RGMTStageII] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[RGMTActorModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = [
            "actor",
            "critic",
            cfg["actor"].get("history_obs_set", "proprio_history"),
            cfg["actor"].get("action_history_obs_set", "action_history"),
            cfg["actor"].get("command_obs_set", "command_window"),
            STAR_GROUP,
        ]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"RGMT Stage-II Actor Model: {actor}")
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"RGMT Stage-II Critic Model: {critic}")

        acquisition_ratio = float(cfg["algorithm"].get("acquisition_ratio", 0.8))
        storage = StageIIStarRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            acquisition_ratio=acquisition_ratio,
            rho_topk=float(cfg["algorithm"].get("rho_topk", 0.05)),
            rho_star=float(cfg["algorithm"].get("rho_star", 0.25)),
            use_star=bool(cfg["algorithm"].get("use_star", True)),
        )
        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

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
        is_stage2_checkpoint = (
            "reference_actor_state_dict" in loaded_dict
            or "teacher_actor_state_dict" in loaded_dict
            or "stage2_valid_acq_ratio_ema" in loaded_dict
        )
        if load_cfg is None and not is_stage2_checkpoint:
            # Stage II commonly starts from a Stage I policy checkpoint. Reuse
            # the policy/value weights, but start Stage II with a fresh optimizer
            # and learning-rate schedule; otherwise a collapsed Stage I adaptive
            # KL state can make Stage II appear frozen from the first update.
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }
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
        mean_kl = 0
        num_kl_updates = 0
        mean_valid_acq_ratio, valid_acq_samples, valid_con_samples = self._compute_valid_sample_stats()
        self.lambda_con = self._compute_lambda_con(mean_valid_acq_ratio)
        actor_grad_norm_sum = 0.0
        critic_grad_norm_sum = 0.0
        actor_grad_norm_max = 0.0
        critic_grad_norm_max = 0.0

        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.star_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    if getattr(batch, "acq_mask", None) is not None:
                        acq_adv = batch.advantages[batch.acq_mask]
                        if acq_adv.numel() > 1:
                            batch.advantages[batch.acq_mask] = (acq_adv - acq_adv.mean()) / (acq_adv.std() + 1e-8)  # type: ignore
                    else:
                        batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            original_observations = batch.observations[:original_batch_size]
            if getattr(batch, "acq_mask", None) is not None:
                is_acquisition = batch.acq_mask
                is_consolidation = ~batch.acq_mask
            else:
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
                        selected_kl = kl[is_acquisition]
                        kl_stats = torch.stack(
                            (
                                selected_kl.sum(),
                                torch.tensor(float(selected_kl.numel()), device=self.device),
                            )
                        )
                    else:
                        kl_stats = torch.zeros(2, device=self.device)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_stats, op=torch.distributed.ReduceOp.SUM)
                    kl_mean = kl_stats[0] / kl_stats[1].clamp_min(1.0)

                    mean_kl += float(kl_mean.item())
                    num_kl_updates += 1

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
            student_action_mean = self.actor.output_mean[:original_batch_size]
            consolidation_loss = self._compute_consolidation_loss(
                original_observations,
                is_consolidation,
                student_action_mean,
            )
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

            actor_grad_norm = float(nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm).item())
            critic_grad_norm = float(nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm).item())
            actor_grad_norm_sum += actor_grad_norm
            critic_grad_norm_sum += critic_grad_norm
            actor_grad_norm_max = max(actor_grad_norm_max, actor_grad_norm)
            critic_grad_norm_max = max(critic_grad_norm_max, critic_grad_norm)
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
            "stage2_consolidation_weighted": self.lambda_con * mean_consolidation_loss,
            "stage2_lambda_con": self.lambda_con,
            "stage2_valid_acq_ratio": mean_valid_acq_ratio,
            "stage2_valid_acq_ratio_ema": (
                self.lambda_rho_ref if self.valid_acq_ratio_ema is None else self.valid_acq_ratio_ema
            ),
            "stage2_valid_acq_samples": float(valid_acq_samples),
            "stage2_valid_con_samples": float(valid_con_samples),
            "stage2_star_pool_size": float(getattr(self.storage, "last_star_pool_size", 0)),
            "stage2_actor_grad_norm_mean": actor_grad_norm_sum / num_updates,
            "stage2_actor_grad_norm_max": actor_grad_norm_max,
            "stage2_critic_grad_norm_mean": critic_grad_norm_sum / num_updates,
            "stage2_critic_grad_norm_max": critic_grad_norm_max,
        }
        if num_kl_updates > 0:
            loss_dict["stage2_kl"] = mean_kl / num_kl_updates
        for key, value in getattr(self, "_last_valid_sample_diagnostics", {}).items():
            if key in {"valid_acq_samples", "valid_con_samples"}:
                continue
            loss_dict[f"stage2_{key}"] = float(value)
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

    def _compute_consolidation_loss(
        self,
        observations,
        consolidation_mask: torch.Tensor,
        student_action_mean: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.any(consolidation_mask):
            return torch.zeros((), device=self.device)

        consolidation_obs = observations[consolidation_mask]
        student_actions = student_action_mean[consolidation_mask]
        with torch.no_grad():
            reference_actions = self.reference_actor(consolidation_obs)
        return (student_actions - reference_actions).pow(2).sum(dim=-1).mean()

    def _compute_valid_sample_stats(self) -> tuple[float, int, int]:
        if hasattr(self.storage, "valid_sample_counts"):
            diagnostics = {}
            if hasattr(self.storage, "valid_sample_diagnostics"):
                diagnostics = self.storage.valid_sample_diagnostics()
                n_acq = int(diagnostics["valid_acq_samples"])
                n_con = int(diagnostics["valid_con_samples"])
            else:
                n_acq, n_con = self.storage.valid_sample_counts()
            total = n_acq + n_con
            ratio = float(n_acq / total) if total > 0 else self.lambda_rho_ref
            self._last_valid_sample_diagnostics = diagnostics
            return ratio, int(n_acq), int(n_con)
        if self.consolidation_obs_key not in self.storage.observations:
            return 1.0, 0, 0
        is_acquisition = self.storage.observations[self.consolidation_obs_key].reshape(-1) > 0.5
        not_done = (1.0 - self.storage.dones.reshape(-1).float()) > 0.5
        valid_acquisition = is_acquisition & not_done
        valid_consolidation = (~is_acquisition) & not_done
        valid_total = valid_acquisition.sum() + valid_consolidation.sum()
        if valid_total.item() == 0:
            return 0.0, 0, 0
        n_acq = int(valid_acquisition.sum().item())
        n_con = int(valid_consolidation.sum().item())
        return float((valid_acquisition.sum().float() / valid_total.float()).item()), n_acq, n_con

    def _compute_lambda_con(self, valid_acq_ratio: float) -> float:
        if self.valid_acq_ratio_ema is None:
            # GMTrack/PACE Eq. (18): initialize the smoothed progress ratio at
            # rho_ref so lambda_con starts at lambda_base instead of jumping to
            # the first rollout's acquisition ratio.
            self.valid_acq_ratio_ema = self.lambda_rho_ref
        self.valid_acq_ratio_ema = (
            self.lambda_beta * self.valid_acq_ratio_ema + (1.0 - self.lambda_beta) * valid_acq_ratio
        )
        value = self.lambda_base + self.lambda_kappa * max(0.0, self.valid_acq_ratio_ema - self.lambda_rho_ref)
        return float(min(self.lambda_con_max, max(0.0, value)))

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if hasattr(self.storage, "record_tracking_failures"):
            failures = extras.get(TRACKING_FAILURES_EXTRA)
            if failures is None:
                raise RuntimeError(
                    "RGMTStageII requires a true tracking failure mask. "
                    f"Pass extras[{TRACKING_FAILURES_EXTRA!r}] from the environment's terminated signal "
                    "via RGMTStageIIRunner; do not derive failures from done/time_outs."
                )
            failures = failures.reshape(-1).bool()
            dones_flat = dones.reshape(-1).bool()
            if failures.shape != dones_flat.shape:
                raise ValueError(
                    "tracking failure and done masks must have the same shape; "
                    f"got {tuple(failures.shape)} and {tuple(dones_flat.shape)}."
                )
            if bool((failures & ~dones_flat).any()):
                raise ValueError("A tracking failure must also be marked done on the same step.")
            self.storage.record_tracking_failures(failures)
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        self.critic.reset(hidden_state=critic_hidden_state)

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if hasattr(st, "raw_advantages"):
            st.raw_advantages.copy_(st.advantages)
        if self.normalize_advantage_per_mini_batch:
            return
        if hasattr(st, "normalize_advantages_by_difficulty") and self.use_star:
            st.normalize_advantages_by_difficulty()
        elif hasattr(st, "normalize_acquisition_advantages"):
            st.normalize_acquisition_advantages()
        else:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
