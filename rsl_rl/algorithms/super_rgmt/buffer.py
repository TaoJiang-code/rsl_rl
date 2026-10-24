"""GPU replay storage for Super-RGMT transitions."""

from __future__ import annotations

import torch

from .mdp import SuperRGMTRobotSpec


class ReplayBuffer:
    """Fixed-size replay laid out as ``[time, environment]``."""

    def __init__(self, time_steps: int, num_envs: int, device: str, spec: SuperRGMTRobotSpec):
        self.time_steps = time_steps
        self.num_envs = num_envs
        self.device = device
        fields = {
            "s": ((spec.state_dim,), torch.float32),
            "s1": ((spec.state_dim,), torch.float32),
            "joint_target": ((spec.num_joints,), torch.float32),
            "previous_action": ((spec.num_joints,), torch.float32),
            "dropout_cached_command": ((spec.command_dim,), torch.float16),
            "dropout_remaining": ((), torch.int32),
            "dropout_active": ((), torch.bool),
            "xi": ((spec.context_dim,), torch.float32),
            "r": ((), torch.float32),
            "f": ((), torch.long),
            "motion_id": ((), torch.long),
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
            "time_out": ((), torch.bool),
        }
        self.data = {
            name: torch.zeros((time_steps, num_envs) + shape, dtype=dtype, device=device)
            for name, (shape, dtype) in fields.items()
        }
        self.pointer = 0
        self.size = 0

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.data.values())

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        for name, destination in self.data.items():
            destination[self.pointer].copy_(transition[name])
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
        names = (
            "s",
            "s1",
            "joint_target",
            "xi",
            "f",
            "motion_id",
            "ep_step",
            "done",
            "time_out",
        )
        output = {name: self.data[name][rows, envs] for name in names}
        done = output["done"].float()
        state_continuous = torch.isclose(
            output["s1"][:, :-1],
            output["s"][:, 1:],
            rtol=1.0e-5,
            atol=1.0e-6,
        ).all(-1)
        trajectory_continuous = (
            (output["motion_id"][:, 1:] == output["motion_id"][:, :-1])
            & (output["f"][:, 1:] == output["f"][:, :-1] + 1)
            & (output["ep_step"][:, 1:] == output["ep_step"][:, :-1] + 1)
        )
        continuous = state_continuous & trajectory_continuous
        can_continue = (1.0 - done[:, :-1]) * continuous.to(done.dtype)
        output["mask"] = torch.cumprod(
            torch.cat((torch.ones(batch_size, 1, device=self.device), can_continue), dim=-1),
            dim=-1,
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
        start_names = (
            "s",
            "previous_action",
            "dropout_cached_command",
            "dropout_remaining",
            "dropout_active",
            "xi",
            "f",
            "motion_id",
            "ep_step",
            "time_out",
        )
        history_names = ("s", "previous_action", "f", "motion_id")
        start = {name: self.data[name][row, env] for name in start_names}
        history = {name: self.data[name][history_rows, history_envs] for name in history_names}
        return start, history

    def sample_recent_windows(self, batch_size: int, window: int) -> dict[str, torch.Tensor]:
        """Sample the latest contiguous real trajectories for model-exploitation diagnostics."""
        if self.size < window:
            raise RuntimeError(f"Replay contains {self.size} rows, but a {window}-step window was requested.")
        batch_size = min(batch_size, self.num_envs)
        first = torch.full((batch_size,), self.size - window, dtype=torch.long, device=self.device)
        env = torch.randperm(self.num_envs, device=self.device)[:batch_size]
        ages = first[:, None] + torch.arange(window, device=self.device)
        rows = self._row(ages)
        envs = env[:, None].expand_as(rows)
        names = ("s", "s1", "joint_target", "xi", "f", "motion_id", "ep_step", "done", "time_out")
        output = {name: self.data[name][rows, envs] for name in names}
        done = output["done"].float()
        state_continuous = torch.isclose(
            output["s1"][:, :-1], output["s"][:, 1:], rtol=1.0e-5, atol=1.0e-6
        ).all(-1)
        trajectory_continuous = (
            (output["motion_id"][:, 1:] == output["motion_id"][:, :-1])
            & (output["f"][:, 1:] == output["f"][:, :-1] + 1)
            & (output["ep_step"][:, 1:] == output["ep_step"][:, :-1] + 1)
        )
        can_continue = (1.0 - done[:, :-1]) * (state_continuous & trajectory_continuous).to(done.dtype)
        output["mask"] = torch.cumprod(
            torch.cat((torch.ones(batch_size, 1, device=self.device), can_continue), dim=-1), dim=-1
        )
        return output
