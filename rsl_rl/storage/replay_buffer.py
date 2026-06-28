# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator


class ReplayBuffer:
    """Fixed-size transition buffer for AMP policy samples."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str = "cpu") -> None:
        """Allocate the replay buffer."""
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Insert a batch of state transitions."""
        states = states.detach()
        next_states = next_states.detach()
        num_states = states.shape[0]
        if num_states >= self.buffer_size:
            self.states.copy_(states[-self.buffer_size :])
            self.next_states.copy_(next_states[-self.buffer_size :])
            self.step = 0
            self.num_samples = self.buffer_size
            return

        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            first_count = self.buffer_size - start_idx
            self.states[start_idx:] = states[:first_count]
            self.next_states[start_idx:] = next_states[:first_count]
            self.states[: end_idx - self.buffer_size] = states[first_count:]
            self.next_states[: end_idx - self.buffer_size] = next_states[first_count:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.step = end_idx % self.buffer_size
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))

    def feed_forward_generator(
        self, num_mini_batches: int, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yield random AMP transition mini-batches."""
        if self.num_samples == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        for _ in range(num_mini_batches):
            sample_idxs = torch.randint(self.num_samples, (mini_batch_size,), device=self.device)
            yield self.states[sample_idxs], self.next_states[sample_idxs]
