"""Observation-only recovery evaluation, independent of rewards and terminations."""
import math

import torch


class RecoveryMetrics:
    def __init__(self, num_envs, device, dt, window_steps, cfg):
        self.dt, self.window_steps = dt, window_steps
        self.validation_steps = max(
            1, math.ceil(float(cfg.get("recovery_metric_validation_s", 0.5)) / dt)
        )
        self.pending = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.valid_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.interval = torch.zeros(5, device=device, dtype=torch.float32)
        self.reset_interval()

    def reset_interval(self):
        self.interval.zero_()

    @torch.no_grad()
    def update(self, state, next_state, active, episode_step, done, terminated):
        active = active.bool()
        fresh = episode_step == 0
        self.pending[fresh] = active[fresh]
        self.valid_steps[fresh] = 0
        # Count only transitions starting after the grace window. The environment's
        # normal termination checks (including reference errors) define recovery.
        checking = self.pending & ~active & (episode_step >= self.window_steps)
        self.valid_steps += (checking & ~done).long()
        failed = self.pending & terminated
        passed = self.pending & ~done & (self.valid_steps >= self.validation_steps)
        excluded = self.pending & done & ~terminated
        self.interval[0].add_((failed | passed).sum())
        self.interval[1].add_(passed.sum())
        self.interval[2].add_(excluded.sum())
        self.pending &= ~(failed | passed | excluded)
        acceleration = (next_state[-1] - state[-1]) / self.dt
        active_weight = active.to(acceleration.dtype)[:, None]
        self.interval[3].add_((acceleration.square() * active_weight).sum())
        self.interval[4].add_(active.sum() * acceleration.shape[-1])

    def flush(self):
        attempts, successes, excluded, acc_sq, acc_count = self.interval.detach().cpu().tolist()
        values = {
            "Recovery/evaluated_attempts": attempts,
            "Recovery/excluded_timeouts": excluded,
        }
        if attempts:
            values["Recovery/success_rate"] = successes / attempts
        if acc_count:
            values["Recovery/joint_acceleration_rms"] = math.sqrt(acc_sq / acc_count)
        self.reset_interval()
        return values
