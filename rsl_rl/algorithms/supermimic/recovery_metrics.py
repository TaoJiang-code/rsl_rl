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
        self.reset_interval()

    def reset_interval(self):
        self.attempts = self.successes = self.excluded = 0
        self.acc_sq = self.acc_count = 0

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
        self.attempts += int((failed | passed).sum())
        self.successes += int(passed.sum())
        self.excluded += int(excluded.sum())
        self.pending &= ~(failed | passed | excluded)
        acceleration = (next_state[-1][active] - state[-1][active]) / self.dt
        self.acc_sq += float(acceleration.square().sum())
        self.acc_count += acceleration.numel()

    def flush(self):
        values = {"Recovery/evaluated_attempts": self.attempts,
                  "Recovery/excluded_timeouts": self.excluded}
        if self.attempts:
            values["Recovery/success_rate"] = self.successes / self.attempts
        if self.acc_count:
            values["Recovery/joint_acceleration_rms"] = math.sqrt(self.acc_sq / self.acc_count)
        self.reset_interval()
        return values
