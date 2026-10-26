"""Fixed-size GPU replay buffer laid out as (time, environment)."""

from __future__ import annotations

import torch

from .mdp import SuperMimicRobotSpec


class ReplayBuffer:
    def __init__(
        self,
        time_steps: int,
        num_envs: int,
        device: str,
        robot_spec: SuperMimicRobotSpec,
    ):
        self.time_steps = time_steps
        self.num_envs = num_envs
        self.device = device
        fields = {
            "s": ((robot_spec.state_dim,), torch.float32),
            "s1": ((robot_spec.state_dim,), torch.float32),
            "a": ((robot_spec.num_joints,), torch.float32),
            "a_prev": ((robot_spec.num_joints,), torch.float32),
            "xi": ((robot_spec.context_dim,), torch.float32),
            "r": ((), torch.float32),
            "f": ((), torch.long),
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
        }
        self.data = {
            name: torch.zeros((time_steps, num_envs) + shape, dtype=dtype, device=device)
            for name, (shape, dtype) in fields.items()
        }
        self.pointer = 0
        self.size = 0

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        for name in self.data:
            self.data[name][self.pointer].copy_(transition[name])
        self.pointer = (self.pointer + 1) % self.time_steps
        self.size = min(self.size + 1, self.time_steps)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        return (self.pointer - self.size + age) % self.time_steps

    def sample_windows(self, batch_size: int, window: int) -> dict[str, torch.Tensor]:
        if self.size < window:
            raise RuntimeError(f"Replay contains {self.size} rows, but a {window}-step window was requested.")
        first = torch.randint(0, self.size - window + 1, (batch_size,), device=self.device)
        env = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        ages = first[:, None] + torch.arange(window, device=self.device)
        rows = self._row(ages)
        envs = env[:, None].expand_as(rows)
        output = {
            name: self.data[name][rows, envs]
            for name in ("s", "s1", "a", "a_prev", "xi", "done")
        }
        done = output["done"].float()
        output["mask"] = torch.cumprod(
            torch.cat((torch.ones(batch_size, 1, device=self.device), 1 - done[:, :-1]), -1), -1
        )
        return output

    def sample_starts(self, batch_size: int, history_length: int):
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer.")
        age = torch.randint(0, self.size, (batch_size,), device=self.device)
        env = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        return self._gather_starts(age, env, history_length)

    def _gather_starts(
        self,
        age: torch.Tensor,
        env: torch.Tensor,
        history_length: int,
    ):
        row = self._row(age)
        episode_step = self.data["ep_step"][row, env].long()
        back = torch.arange(history_length - 1, -1, -1, device=self.device)
        back = torch.minimum(back[None], episode_step[:, None])
        back = torch.minimum(back, age[:, None])
        history_rows = self._row(age[:, None] - back)
        history_envs = env[:, None].expand_as(history_rows)
        start = {
            name: self.data[name][row, env]
            for name in ("s", "a_prev", "xi", "f", "ep_step")
        }
        history = {
            name: self.data[name][history_rows, history_envs] for name in ("s", "a_prev", "xi", "f")
        }
        return start, history

    def sample_valid_starts(
        self,
        batch_size: int,
        history_length: int,
        frame_clip_ids: torch.Tensor,
        clip_ends: torch.Tensor,
        horizon: int,
    ):
        """Sample uniformly from every replay entry that can support a full imagined rollout."""
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer.")
        ages = torch.arange(self.size, device=self.device)
        rows = self._row(ages)
        frames = self.data["f"][rows]
        ends = clip_ends[frame_clip_ids[frames]]
        valid_entries = (frames + horizon <= ends).nonzero(as_tuple=False)
        valid_count = int(valid_entries.shape[0])
        total_count = self.size * self.num_envs
        if valid_count == 0:
            remaining = (ends - frames).amax().item()
            raise RuntimeError(
                "SuperMimic replay has no valid imagined-rollout starts: "
                f"horizon={horizon}, replay_entries={total_count}, "
                f"maximum_remaining_frames={remaining}."
            )
        selected = torch.randint(0, valid_count, (batch_size,), device=self.device)
        entries = valid_entries[selected]
        start, history = self._gather_starts(entries[:, 0], entries[:, 1], history_length)
        valid_ratio = valid_count / total_count
        return start, history, valid_ratio
