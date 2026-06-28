# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import AmpOnPolicyRunner


NUM_ENVS = 4
OBS_DIM = 8
AMP_DIM = 15
NUM_ACTIONS = 4
MAX_EP_LEN = 8


class DummyAmpEnv(VecEnv):
    """Minimal VecEnv with AMP observations."""

    def __init__(self, device: str = "cpu") -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM, device=self.device),
                "amp": torch.randn(self.num_envs, AMP_DIM, device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return self.get_observations(), rewards, dones, extras


def _write_motion(path) -> None:
    """Create a tiny AMP expert motion file."""
    num_frames = 12
    body_pos = np.zeros((num_frames, 2, 3), dtype=np.float32)
    body_pos[:, 1, 0] = np.linspace(0.0, 0.2, num_frames)
    body_quat = np.zeros((num_frames, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin_vel = np.zeros((num_frames, 2, 3), dtype=np.float32)
    body_ang_vel = np.zeros((num_frames, 2, 3), dtype=np.float32)
    np.savez(
        path,
        fps=np.array(30),
        body_names=np.array(["root", "foot"]),
        body_positions=body_pos,
        body_rotations=body_quat,
        body_linear_velocities=body_lin_vel,
        body_angular_velocities=body_ang_vel,
    )


def _make_amp_cfg(motion_file: str) -> dict:
    """Return a minimal AMP training configuration."""
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "AMPPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "amp_replay_buffer_size": 128,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
        "discriminator": {
            "class_name": "Discriminator",
            "hidden_dims": [32, 16],
            "reward_coef": 1.0,
            "task_reward_lerp": 0.0,
        },
        "amp_motion_file": motion_file,
        "amp_body_names": ["foot"],
        "amp_anchor_name": "root",
    }


def test_amp_runner_learn_runs(tmp_path) -> None:
    """A short AMP training call should complete without raising."""
    motion_file = tmp_path / "motion.npz"
    _write_motion(motion_file)
    runner = AmpOnPolicyRunner(DummyAmpEnv(), _make_amp_cfg(str(motion_file)), log_dir=None, device="cpu")
    runner.learn(num_learning_iterations=1)
    assert runner.current_learning_iteration == 0
