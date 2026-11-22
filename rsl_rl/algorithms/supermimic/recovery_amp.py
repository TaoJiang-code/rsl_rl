"""Training-only AMP prior for the original perturbation-based recovery window.

Expert and simulator states use identical heading-invariant features (height,
root orientation/velocities, joint positions/velocities). Transition pairs never
cross clip boundaries. The existing AMP discriminator and LSGAN objective are
reused; reward evaluation deliberately retains input gradients for SuperMimic.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rsl_rl.modules.discriminator import Discriminator
from .mdp import invariant_features, unflatten_state
from .networks import RunningNorm


class RecoveryAMP:
    def __init__(self, motion, cfg: dict, device: str):
        if motion.num_recovery_clips == 0:
            raise ValueError("Recovery AMP requires nonempty recovery_motion_dir.")
        self.batch_size = int(cfg.get("recovery_amp_batch_size", 512))
        self.reward_coef = float(cfg["amp_reward_coef"])
        self.task_reward_lerp = float(cfg["amp_task_reward_lerp"])
        self.dt = float(cfg.get("control_dt", 0.02))
        self.grad_penalty = float(cfg.get("recovery_amp_grad_penalty", 10.0))
        if abs(float(motion.cfg.fps) * self.dt - 1.0) > 1e-5:
            raise ValueError("AMP expert transitions must match the control frequency.")
        if self.batch_size <= 0 or self.reward_coef < 0:
            raise ValueError("AMP batch size must be positive and reward weight nonnegative.")
        if not 0.0 <= self.task_reward_lerp <= 1.0:
            raise ValueError("amp_task_reward_lerp must be between 0 and 1.")
        expert = torch.cat(
            (
                motion.recovery_body_pos_w[:, 0],
                motion.recovery_body_quat_w[:, 0],
                motion.recovery_body_lin_vel_w[:, 0],
                motion.recovery_body_ang_vel_w[:, 0],
                motion.recovery_joint_pos,
                motion.recovery_joint_vel,
            ),
            -1,
        )
        with torch.no_grad():
            self.expert = invariant_features(unflatten_state(expert)).detach()
        self.starts = []
        for start, end in zip(motion.recovery_clip_starts.tolist(), motion.recovery_clip_ends.tolist()):
            if end <= start:
                raise ValueError("AMP expert clips require at least two frames.")
            self.starts.append(torch.arange(start, end, device=device))
        self.norm = RunningNorm(self.expert.shape[-1]).to(device)
        self.norm.update(self.expert)
        self.discriminator = Discriminator(2 * self.expert.shape[-1], hidden_dims=(512, 256), device=device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=float(cfg.get("recovery_amp_learning_rate", 1e-4))
        )
        self.updates = 0
        self.last_loss = 0.0
        self.discriminator.requires_grad_(False)

    def sample_expert(self, count: int):
        # Sample clips uniformly, then frames within each clip; short clips remain represented.
        clips = torch.randint(len(self.starts), (count,), device=self.expert.device)
        indices = torch.empty_like(clips)
        for index, starts in enumerate(self.starts):
            mask = clips == index
            indices[mask] = starts[torch.randint(len(starts), (int(mask.sum()),), device=clips.device)]
        return self.expert[indices], self.expert[indices + 1]

    def reward(self, state, next_state, active: torch.Tensor) -> torch.Tensor:
        if self.updates == 0:
            return torch.zeros_like(active, dtype=state[0].dtype)
        first = self.norm(invariant_features(state))
        second = self.norm(invariant_features(next_state))
        # Frozen discriminator parameters, but NOT no_grad: policy needs d(reward)/d(state).
        logits = self.discriminator(torch.cat((first, second), -1)).squeeze(-1)
        return self.reward_coef * (1.0 - 0.25 * (logits - 1).square()).clamp_min(0.0) * active

    def mix_reward(self, task_reward, style_reward, active):
        """Blend only recovery transitions; preserve task rewards during AMP warmup."""
        if self.updates == 0:
            return task_reward
        return task_reward * (1.0 - active * (1.0 - self.task_reward_lerp)) + (
            1.0 - self.task_reward_lerp
        ) * style_reward

    def update(self, replay, active_slice: slice) -> None:
        if replay.size == 0:
            return
        # Bound temporary allocations; use real pre-reset transitions, including final steps.
        count = self.batch_size * 16
        age = torch.randint(replay.size, (count,), device=replay.device)
        env = torch.randint(replay.num_envs, (count,), device=replay.device)
        rows = replay._row(age)
        active = replay.data["xi"][rows, env, active_slice].squeeze(-1) > 0
        rows, env = rows[active][: self.batch_size], env[active][: self.batch_size]
        if rows.numel() == 0:
            return
        with torch.no_grad():
            first = invariant_features(unflatten_state(replay.data["s"][rows, env]))
            second = invariant_features(unflatten_state(replay.data["s1"][rows, env]))
            expert, expert_next = self.sample_expert(len(rows))
            self.norm.update(torch.cat((first, second, expert, expert_next)))
            first, second = self.norm(first), self.norm(second)
            expert, expert_next = self.norm(expert), self.norm(expert_next)
        self.discriminator.requires_grad_(True)
        try:
            policy_logits = self.discriminator(torch.cat((first, second), -1))
            expert_logits = self.discriminator(torch.cat((expert, expert_next), -1))
            loss = 0.5 * (
                F.mse_loss(policy_logits, -torch.ones_like(policy_logits))
                + F.mse_loss(expert_logits, torch.ones_like(expert_logits))
            )
            loss = loss + self.discriminator.compute_grad_penalty(expert, expert_next, self.grad_penalty)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach())
            self.updates += 1
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.discriminator.requires_grad_(False)

    def state_dict(self) -> dict:
        return dict(
            discriminator=self.discriminator.state_dict(),
            normalizer=self.norm.state_dict(),
            optimizer=self.optimizer.state_dict(),
            updates=self.updates,
        )

    def load_state_dict(self, state: dict, load_optimizers: bool = True) -> None:
        self.discriminator.load_state_dict(state["discriminator"])
        self.norm.load_state_dict(state["normalizer"])
        self.updates = int(state["updates"])
        if load_optimizers:
            self.optimizer.load_state_dict(state["optimizer"])
