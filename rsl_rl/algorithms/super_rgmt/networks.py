"""Dynamics model and critic used by Super-RGMT."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.supermimic.networks import RunningNorm, mlp, run_mlp
from rsl_rl.algorithms.supermimic.mdp import invariant_features
from rsl_rl.algorithms.supermimic.rotations import so3_exp, so3_log, yaw_mat

from .mdp import SuperRGMTRobotSpec


class WorldModel(nn.Module):
    """Predict one simulator step from Markov state, PD target, and physics context."""

    def __init__(self, spec: SuperRGMTRobotSpec):
        super().__init__()
        self.num_joints = spec.num_joints
        self.input_dim = spec.invariant_dim + spec.num_joints + spec.context_dim
        self.net = mlp(self.input_dim, [2048, 2048, 2048], spec.state_increment_dim, nn.SiLU, True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.input_dim)
        self.out_norm = RunningNorm(spec.state_increment_dim, eps=1.0e-4)

    def inputs(self, state, joint_target, normalized_context):
        return torch.cat((invariant_features(state), joint_target, normalized_context), dim=-1)

    @staticmethod
    def targets(state, next_state):
        p, rotation, v, w, q, qd = state
        p1, rotation1, v1, w1, q1, qd1 = next_state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(rotation_t @ rotation1),
                (rotation_t @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            dim=-1,
        )

    def step(self, state, joint_target, normalized_context):
        p, rotation, v, w, q, qd = state
        output = run_mlp(self.net, self.in_norm(self.inputs(state, joint_target, normalized_context)))
        delta = output * self.out_norm.std() + self.out_norm.mean
        dp, dtheta, dv, dw, dq, dqd = delta.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), dim=-1
        )
        heading = yaw_mat(rotation)
        return (
            p + (heading @ dp.unsqueeze(-1)).squeeze(-1),
            rotation @ so3_exp(dtheta),
            v + (rotation @ dv.unsqueeze(-1)).squeeze(-1),
            w + (rotation @ dw.unsqueeze(-1)).squeeze(-1),
            q + dq,
            qd + dqd,
        )

    @staticmethod
    def state_error(predicted, target):
        p, rotation, v, w, q, qd = target
        ph, rh, vh, wh, qh, qdh = predicted
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(rotation_t @ rh),
                (rotation_t @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            dim=-1,
        )


class Critic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = mlp(input_dim, [256, 128], 1)
        self.norm = RunningNorm(input_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(value)).squeeze(-1)
