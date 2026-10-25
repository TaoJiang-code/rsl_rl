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
        output = {name: self.data[name][rows, envs] for name in ("s", "s1", "a", "xi", "done")}
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
