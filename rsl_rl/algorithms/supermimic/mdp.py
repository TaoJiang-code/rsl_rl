"""Differentiable SuperMimic observation, reward, termination, and critic heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, yaw_mat

NUM_JOINTS = 29
NUM_TRACKED_BODIES = 14
STATE_DIM = 71
OBS_DIM = 99
XI_DIM = 33
GOAL_STEP_DIM = 73
INV_DIM = 1 + 6 + 3 + 3 + NUM_JOINTS + NUM_JOINTS
STATE_INCREMENT_DIM = 3 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS
CRITIC_EXTRA_DIM = NUM_JOINTS + NUM_TRACKED_BODIES * 3 + 3 + 6 + XI_DIM
CONTROL_DT = 0.02


def flatten_state(state) -> torch.Tensor:
    p, rotation, v, w, q, qd = state
    return torch.cat((p, mat_to_quat(rotation), v, w, q, qd), -1)


def unflatten_state(value: torch.Tensor):
    return (
        value[..., 0:3],
        quat_to_mat(value[..., 3:7]),
        value[..., 7:10],
        value[..., 10:13],
        value[..., 13:42],
        value[..., 42:71],
    )


def invariant_features(state) -> torch.Tensor:
    p, rotation, v, w, q, qd = state
    heading = yaw_mat(rotation)
    rotation_t = rotation.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(heading.transpose(-1, -2) @ rotation),
            (rotation_t @ v.unsqueeze(-1)).squeeze(-1),
            (rotation_t @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


def normalize_context(context: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        ((context[..., :1] - 0.75) / 0.45, context[..., 1:30] / 0.01, context[..., 30:33] / 0.05),
        -1,
    )


@dataclass
class RewardConfig:
    body_position_std: float = 0.3
    body_orientation_std: float = 0.4
    body_linear_velocity_std: float = 1.0
    body_angular_velocity_std: float = 3.14
    root_position_std: float = 0.3
    root_orientation_std: float = 0.4
    action_rate_weight: float = -0.1
    joint_limit_weight: float = -2.0
    termination_height: float = 0.25
    termination_tilt: float = 0.8


class SuperMimicMDP:
    def __init__(self, kinematics, motion, default_joint_position, soft_limits, cfg=RewardConfig()):
        self.kinematics = kinematics
        self.motion = motion
        self.default_joint_position = default_joint_position
        self.soft_lower = soft_limits[..., 0]
        self.soft_upper = soft_limits[..., 1]
        self.cfg = cfg
        device = default_joint_position.device
        self.obs_noise = torch.cat(
            (
                torch.full((29,), 0.01),
                torch.full((29,), 0.5),
                torch.full((3,), 0.2),
                torch.full((3,), 0.05),
                torch.zeros(29),
                torch.full((6,), 0.05),
            )
        ).to(device)

    def goal(self, frame: torch.Tensor, count: int) -> torch.Tensor:
        return self.motion.goal(frame, count)

    def observation(self, state, previous_action, context, frame) -> torch.Tensor:
        _, rotation, _, w, q, qd = state
        rotation_t = rotation.transpose(-1, -2)
        calibration = context[..., 1:30]
        projected_gravity = -rotation[..., 2, :]
        relative_reference = mat_to_6d(rotation_t @ quat_to_mat(self.motion.body_quat_w[frame, 0]))
        return torch.cat(
            (
                q - self.default_joint_position - calibration,
                qd,
                (rotation_t @ w.unsqueeze(-1)).squeeze(-1),
                projected_gravity,
                previous_action,
                relative_reference,
            ),
            -1,
        )

    def add_noise(self, observation: torch.Tensor) -> torch.Tensor:
        return observation + (2 * torch.rand_like(observation) - 1) * self.obs_noise

    def reward(self, state, frame, action, previous_action, return_terms: bool = False):
        cfg = self.cfg
        p, rotation, v, w, q, qd = state
        body_p, body_r, body_v, body_w = self.kinematics.forward(p, rotation, v, w, q, qd)
        reference_p = self.motion.body_pos_w[frame, 0]
        reference_r = quat_to_mat(self.motion.body_quat_w[frame, 0])
        yaw_delta = yaw_mat(rotation @ reference_r.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], reference_p[..., 2:3]), -1)
        reference_body_p = anchor.unsqueeze(-2) + (
            yaw_delta.unsqueeze(-3)
            @ (self.motion.body_pos_w[frame] - reference_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        reference_body_r = yaw_delta.unsqueeze(-3) @ quat_to_mat(self.motion.body_quat_w[frame])
        body_position_error = (reference_body_p - body_p).square().sum(-1).mean(-1)
        body_orientation_error = rot_angle_sq(reference_body_r, body_r).mean(-1)
        body_linear_velocity_error = (self.motion.body_lin_vel_w[frame] - body_v).square().sum(-1).mean(-1)
        body_angular_velocity_error = (self.motion.body_ang_vel_w[frame] - body_w).square().sum(-1).mean(-1)
        root_position_error = (reference_p - p).square().sum(-1)
        root_orientation_error = rot_angle_sq(reference_r, rotation)
        tracking = (
            torch.exp(-body_position_error / cfg.body_position_std**2)
            + torch.exp(-body_orientation_error / cfg.body_orientation_std**2)
            + torch.exp(-body_linear_velocity_error / cfg.body_linear_velocity_std**2)
            + torch.exp(-body_angular_velocity_error / cfg.body_angular_velocity_std**2)
            + 0.5 * torch.exp(-root_position_error / cfg.root_position_std**2)
            + 0.5 * torch.exp(-root_orientation_error / cfg.root_orientation_std**2)
        )
        action_rate = (action - previous_action).square().sum(-1)
        joint_limit = (torch.relu(q - self.soft_upper) + torch.relu(self.soft_lower - q)).sum(-1)
        reward = (
            tracking + cfg.action_rate_weight * action_rate + cfg.joint_limit_weight * joint_limit
        ) * CONTROL_DT
        if return_terms:
            return reward, {
                "tracking": tracking * CONTROL_DT,
                "body_position_error": body_position_error,
                "body_orientation_error": body_orientation_error,
                "root_position_error": root_position_error,
                "action_rate": action_rate,
                "joint_limit": joint_limit,
            }
        return reward

    def soft_termination(self, state, frame, height_temperature=0.02, tilt_temperature=0.04):
        p, rotation = state[0], state[1]
        reference_p = self.motion.body_pos_w[frame, 0]
        reference_r = quat_to_mat(self.motion.body_quat_w[frame, 0])
        height_error = (reference_p[..., 2] - p[..., 2]).abs()
        tilt_error = (reference_r[..., 2, 2] - rotation[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((height_error - self.cfg.termination_height) / height_temperature)) * (
            1 - torch.sigmoid((tilt_error - self.cfg.termination_tilt) / tilt_temperature)
        )
        return 1 - alive

    def critic_features(self, state, frame, normalized_context):
        p, rotation, v, w, q, qd = state
        heading_t = yaw_mat(rotation).transpose(-1, -2)
        reference_p = self.motion.body_pos_w[frame, 0]
        reference_r = quat_to_mat(self.motion.body_quat_w[frame, 0])
        body_p = self.kinematics.forward(p, rotation, v, w, q, qd)[0]
        yaw_delta = yaw_mat(rotation @ reference_r.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], reference_p[..., 2:3]), -1)
        reference_body_p = anchor.unsqueeze(-2) + (
            yaw_delta.unsqueeze(-3)
            @ (self.motion.body_pos_w[frame] - reference_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        body_error = (
            heading_t.unsqueeze(-3) @ (reference_body_p - body_p).unsqueeze(-1)
        ).squeeze(-1).flatten(-2)
        root_error = (heading_t @ (reference_p - p).unsqueeze(-1)).squeeze(-1)
        relative_rotation = mat_to_6d(rotation.transpose(-1, -2) @ reference_r)
        return torch.cat(
            (
                invariant_features(state),
                self.motion.joint_pos[frame] - q,
                body_error,
                root_error,
                relative_rotation,
                normalized_context,
            ),
            -1,
        )
