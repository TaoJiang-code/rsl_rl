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
        mixed_tasks: bool = False,
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
            "rf": ((), torch.long),
            "recovery_phase": ((), torch.uint8),
            "recovery_transition_step": ((), torch.int16),
            "recovery_error_step": ((), torch.int16),
            "recovery_gate_target": ((), torch.float16),
            "recovery_gate_tracking_override": ((), torch.bool),
            "reference_yaw_rotation": ((3, 3), torch.float16),
            "reference_translation": ((3,), torch.float16),
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
        }
        if mixed_tasks:
            fields.update(
                {
                    "velocity_command": ((3,), torch.float32),
                    "velocity_time_left": ((), torch.float32),
                    "teleop_task": ((), torch.bool),
                }
            )
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

    def sample_windows(
        self,
        batch_size: int,
        window: int,
        env_range: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.size < window:
            raise RuntimeError(f"Replay contains {self.size} rows, but a {window}-step window was requested.")
        first = torch.randint(0, self.size - window + 1, (batch_size,), device=self.device)
        env_low, env_high = (0, self.num_envs) if env_range is None else env_range
        if not 0 <= env_low < env_high <= self.num_envs:
            raise ValueError(f"Invalid replay environment range: {(env_low, env_high)}.")
        env = torch.randint(env_low, env_high, (batch_size,), device=self.device)
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
            for name in (
                "s",
                "a",
                "a_prev",
                "xi",
                "f",
                "rf",
                "recovery_phase",
                "recovery_transition_step",
                "recovery_error_step",
                "recovery_gate_target",
                "recovery_gate_tracking_override",
                "reference_yaw_rotation",
                "reference_translation",
                "ep_step",
            )
        }
        start["age"] = age
        start["env"] = env
        for name in (
            "velocity_command",
            "velocity_time_left",
            "teleop_task",
        ):
            if name in self.data:
                start[name] = self.data[name][row, env]
        history_names = ["s", "a_prev", "xi", "f", "reference_yaw_rotation"]
        history = {
            name: self.data[name][history_rows, history_envs] for name in history_names
        }
        return start, history

    def forward_from(
        self,
        age: torch.Tensor,
        env: torch.Tensor,
        steps: int,
        names: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Read recorded transitions forward from replay starts.

        Returned tensors are time-major ``(steps, batch, ...)``. ``complete`` marks starts whose
        entire window is still available before the replay write head.
        """
        ages = age[None] + torch.arange(steps, device=self.device)[:, None]
        rows = self._row(ages.clamp(max=self.size - 1))
        envs = env[None].expand_as(rows)
        return {name: self.data[name][rows, envs] for name in names}, ages[-1] < self.size

    def sample_valid_starts(
        self,
        batch_size: int,
        history_length: int,
        frame_clip_ids: torch.Tensor,
        clip_ends: torch.Tensor,
        horizon: int,
        teleop_task: bool | None = None,
    ):
        """Sample uniformly from every replay entry that can support a full imagined rollout."""
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer.")
        ages = torch.arange(self.size, device=self.device)
        rows = self._row(ages)
        frames = self.data["f"][rows]
        ends = clip_ends[frame_clip_ids[frames]]
        valid_entries = (frames + horizon <= ends).nonzero(as_tuple=False)
        eligible_count = self.size * self.num_envs
        if teleop_task is not None:
            if "teleop_task" not in self.data:
                raise RuntimeError("Task-filtered replay sampling requires mixed-task metadata.")
            tasks = self.data["teleop_task"][rows]
            eligible = tasks == teleop_task
            valid_entries = ((frames + horizon <= ends) & eligible).nonzero(as_tuple=False)
            eligible_count = int(eligible.sum())
        valid_count = int(valid_entries.shape[0])
        if valid_count == 0:
            remaining = (ends - frames).amax().item()
            raise RuntimeError(
                "SuperMimic replay has no valid imagined-rollout starts: "
                f"horizon={horizon}, replay_entries={eligible_count}, "
                f"maximum_remaining_frames={remaining}."
            )
        selected = torch.randint(0, valid_count, (batch_size,), device=self.device)
        entries = valid_entries[selected]
        start, history = self._gather_starts(entries[:, 0], entries[:, 1], history_length)
        valid_ratio = valid_count / max(eligible_count, 1)
        return start, history, valid_ratio
