"""World model, full-reference policy, and critic from SuperMimic Appendix A."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import SuperMimicRobotSpec, invariant_features
from .rotations import so3_exp, so3_log, yaw_mat

AMP_DTYPE: torch.dtype | None = None


def run_mlp(network: nn.Module, value: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None or not value.is_cuda:
        return network(value)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return network(value).float()


def mlp(input_dim: int, hidden: list[int], output_dim: int, activation=nn.ELU, layer_norm=False):
    layers: list[nn.Module] = []
    dim = input_dim
    for width in hidden:
        layers.append(nn.Linear(dim, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        dim = width
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        value = value.reshape(-1, value.shape[-1]).float()
        if self.count >= self.max_count:
            return
        count = value.shape[0]
        mean, variance = value.mean(0), value.var(0, unbiased=False)
        total = self.count + count
        delta = mean - self.mean
        self.mean += delta * count / total
        self.var = (
            self.var * self.count + variance * count + delta.square() * self.count * count / total
        ) / total
        self.count = total

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


class WorldModel(nn.Module):
    def __init__(self, robot_spec: SuperMimicRobotSpec):
        super().__init__()
        self.num_joints = robot_spec.num_joints
        self.uses_action_history = robot_spec.uses_action_history
        action_dim = self.num_joints * (2 if self.uses_action_history else 1)
        self.input_dim = robot_spec.invariant_dim + action_dim + robot_spec.context_dim
        self.net = mlp(
            self.input_dim,
            [2048, 2048, 2048],
            robot_spec.state_increment_dim,
            nn.SiLU,
            True,
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.input_dim)
        self.out_norm = RunningNorm(robot_spec.state_increment_dim, eps=1e-4)

    def inputs(self, state, action, normalized_context, previous_action=None):
        values = [invariant_features(state)]
        if self.uses_action_history:
            if previous_action is None:
                raise ValueError("Delayed-actuator world model requires previous_action.")
            values.append(previous_action)
        values.extend((action, normalized_context))
        return torch.cat(values, -1)

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
            -1,
        )

    def step(self, state, action, normalized_context, previous_action=None):
        p, rotation, v, w, q, qd = state
        output = run_mlp(
            self.net,
            self.in_norm(self.inputs(state, action, normalized_context, previous_action)),
        )
        delta = output * self.out_norm.std() + self.out_norm.mean
        dp, dtheta, dv, dw, dq, dqd = delta.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), -1
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
            -1,
        )


class Policy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        history_length: int = 10,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.encoder = mlp(goal_dim, [2048, 1024, 512, 256], latent_dim)
        self.decoder = mlp(latent_dim + obs_dim * history_length, [2048, 1024, 512, 256], action_dim)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.01)
            self.decoder[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, observation_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(run_mlp(self.encoder, self.goal_norm(goal)))
        observation = self.obs_norm(observation_history).flatten(-2)
        return run_mlp(self.decoder, torch.cat((latent, observation), -1))


class Critic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = mlp(input_dim, [256, 128], 1)
        self.norm = RunningNorm(input_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(value)).squeeze(-1)
