"""Differentiable RGMT observations and tracking objective for imagined rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.supermimic.mdp import invariant_features
from rsl_rl.algorithms.supermimic.rotations import mat_to_6d, quat_to_mat, rot_angle_sq, yaw_mat

@dataclass(frozen=True)
class SuperRGMTRobotSpec:
    num_joints: int
    num_tracked_bodies: int
    context_dim: int
    history_length: int = 10
    command_window_size: int = 10
    command_step_dim: int = 73

    @property
    def state_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def state_increment_dim(self) -> int:
        return 12 + 2 * self.num_joints

    @property
    def invariant_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def policy_dim(self) -> int:
        return 9 + 3 * self.num_joints

    @property
    def proprio_dim(self) -> int:
        return 6 + 2 * self.num_joints

    @property
    def command_dim(self) -> int:
        return self.command_window_size * self.command_step_dim

    @property
    def critic_feature_dim(self) -> int:
        return self.invariant_dim + self.num_joints + 3 * self.num_tracked_bodies + 9 + self.context_dim

    def to_dict(self) -> dict[str, int]:
        return {
            "schema_version": 2,
            "num_joints": self.num_joints,
            "num_tracked_bodies": self.num_tracked_bodies,
            "context_dim": self.context_dim,
            "history_length": self.history_length,
            "command_window_size": self.command_window_size,
            "command_step_dim": self.command_step_dim,
        }


@dataclass
class RewardConfig:
    control_dt: float = 0.02
    body_position_weight: float = 1.0
    body_orientation_weight: float = 1.0
    body_linear_velocity_weight: float = 1.0
    body_angular_velocity_weight: float = 1.0
    root_position_weight: float = 0.5
    root_orientation_weight: float = 0.5
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
    termination_body_height: float = 0.25
    termination_height_temperature: float = 0.02
    termination_tilt_temperature: float = 0.04


class SuperRGMTMDP:
    """Pure-tensor counterpart of the Stage-I RGMT manager environment."""

    def __init__(
        self,
        kinematics,
        motion,
        default_joint_position: torch.Tensor,
        soft_limits: torch.Tensor,
        action_scale: torch.Tensor,
        context_layout,
        spec: SuperRGMTRobotSpec,
        frame_offsets: tuple[int, ...] = (-10, -8, -4, -2, 0, 1, 2, 3, 4, 5),
        end_effector_body_ids: tuple[int, ...] = (),
        cfg: RewardConfig | None = None,
    ):
        self.kinematics = kinematics
        self.motion = motion
        self.spec = spec
        self.default_joint_position = default_joint_position
        self.soft_lower = soft_limits[..., 0]
        self.soft_upper = soft_limits[..., 1]
        self.action_scale = action_scale
        self.context_layout = context_layout
        self.frame_offsets = torch.tensor(frame_offsets, dtype=torch.long, device=default_joint_position.device)
        self.end_effector_body_ids = end_effector_body_ids
        self.cfg = cfg or RewardConfig()
        if len(frame_offsets) != spec.command_window_size:
            raise ValueError("Super-RGMT frame offsets do not match command_window_size.")
        expected_step_dim = 15 + 2 * spec.num_joints
        if spec.command_step_dim != expected_step_dim:
            raise ValueError(
                f"Super-RGMT command token has dim {spec.command_step_dim}, expected {expected_step_dim}."
            )
        self.proprio_noise = torch.cat(
            (
                torch.full((3,), 0.05),
                torch.full((3,), 0.2),
                torch.full((spec.num_joints,), 0.01),
                torch.full((spec.num_joints,), 0.5),
            )
        ).to(default_joint_position.device)

    def _global_frames(self, motion_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.long().clamp_min(0)
        maximum = self.motion.time_step_totals[motion_id] - 1
        while maximum.ndim < frame.ndim:
            maximum = maximum.unsqueeze(-1)
        frame = torch.minimum(frame, maximum)
        offsets = self.motion.motion_frame_offsets[motion_id]
        while offsets.ndim < frame.ndim:
            offsets = offsets.unsqueeze(-1)
        return offsets + frame

    def _gather(self, source: str, name: str, motion_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        data = self.motion._command_motion_data if source == "command" else self.motion._motion_data
        return data[name][self._global_frames(motion_id, frame)]

    def proprioception(self, state) -> torch.Tensor:
        _, rotation, _, angular_velocity, joint_position, joint_velocity = state
        rotation_t = rotation.transpose(-1, -2)
        projected_gravity = -rotation[..., 2, :]
        angular_velocity_b = (rotation_t @ angular_velocity.unsqueeze(-1)).squeeze(-1)
        return torch.cat(
            (projected_gravity, angular_velocity_b, joint_position - self.default_joint_position, joint_velocity),
            dim=-1,
        )

    def add_proprio_noise(self, value: torch.Tensor) -> torch.Tensor:
        return value + (2.0 * torch.rand_like(value) - 1.0) * self.proprio_noise

    def policy_observation(self, state, previous_action: torch.Tensor, add_noise: bool) -> torch.Tensor:
        proprio = self.proprioception(state)
        if add_noise:
            proprio = self.add_proprio_noise(proprio)
        return torch.cat((proprio, previous_action), dim=-1)

    def command_window(self, state, motion_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        robot_rotation = state[1]
        local_frames = frame[:, None] + self.frame_offsets[None]
        body_position = self._gather("command", "body_pos_w", motion_id, local_frames)
        body_quaternion = self._gather("command", "body_quat_w", motion_id, local_frames)
        body_linear_velocity = self._gather("command", "body_lin_vel_w", motion_id, local_frames)
        body_angular_velocity = self._gather("command", "body_ang_vel_w", motion_id, local_frames)
        joint_position = self._gather("command", "joint_pos", motion_id, local_frames)
        joint_velocity = self._gather("command", "joint_vel", motion_id, local_frames)
        anchor = self.motion.motion_anchor_body_index

        reference_rotation = quat_to_mat(body_quaternion[..., anchor, :])
        relative_rotation = robot_rotation.transpose(-1, -2)[:, None] @ reference_rotation
        root_orientation_6d = mat_to_6d(relative_rotation)

        current_position = self._gather("command", "body_pos_w", motion_id, frame)[:, anchor]
        current_rotation = quat_to_mat(
            self._gather("command", "body_quat_w", motion_id, frame)[:, anchor]
        )
        current_position_z0 = torch.cat((current_position[:, :2], torch.zeros_like(current_position[:, 2:])), dim=-1)
        current_heading_t = yaw_mat(current_rotation).transpose(-1, -2)[:, None]
        future_position_local = (
            current_heading_t
            @ (body_position[..., anchor, :] - current_position_z0[:, None]).unsqueeze(-1)
        ).squeeze(-1)

        reference_rotation_t = reference_rotation.transpose(-1, -2)
        linear_velocity_b = (
            reference_rotation_t @ body_linear_velocity[..., anchor, :].unsqueeze(-1)
        ).squeeze(-1)
        angular_velocity_b = (
            reference_rotation_t @ body_angular_velocity[..., anchor, :].unsqueeze(-1)
        ).squeeze(-1)
        command = torch.cat(
            (
                root_orientation_6d,
                future_position_local,
                linear_velocity_b,
                angular_velocity_b,
                joint_position,
                joint_velocity,
            ),
            dim=-1,
        )
        return command

    def actor_input(
        self,
        state,
        previous_action: torch.Tensor,
        proprio_history: torch.Tensor,
        action_history: torch.Tensor,
        command_window: torch.Tensor,
        add_noise: bool,
    ) -> TensorDict:
        policy = self.policy_observation(state, previous_action, add_noise)
        return TensorDict(
            {
                "policy": policy,
                "proprio_history": proprio_history,
                "action_history": action_history,
                "command_window": command_window.reshape(command_window.shape[0], -1),
            },
            batch_size=[policy.shape[0]],
            device=policy.device,
        )

    def joint_target(self, raw_action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        calibration = context[..., self.context_layout.calibration_offset]
        return self.default_joint_position + calibration + raw_action * self.action_scale

    def advance_context(self, context: torch.Tensor, episode_step: torch.Tensor, recovery_window_steps: int):
        if recovery_window_steps <= 0:
            return context
        updated = context.clone()
        was_active = context[..., self.context_layout.recovery_active].squeeze(-1) > 0.5
        active = was_active & (episode_step < recovery_window_steps)
        updated[..., self.context_layout.recovery_active] = active[..., None].to(context.dtype)
        assist = updated[..., self.context_layout.recovery_assist]
        updated[..., self.context_layout.recovery_assist] = assist * active[..., None]
        return updated

    def _reference(self, motion_id: torch.Tensor, frame: torch.Tensor):
        names = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos")
        return tuple(self._gather("target", name, motion_id, frame) for name in names)

    def reward(
        self,
        state,
        alignment_state,
        motion_id: torch.Tensor,
        frame: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        p, rotation, v, w, q, qd = state
        body_p, body_r, body_v, body_w = self.kinematics.forward(p, rotation, v, w, q, qd)
        reference_body_p, reference_body_q, reference_body_v, reference_body_w, _ = self._reference(
            motion_id, frame
        )
        reference_body_r = quat_to_mat(reference_body_q)
        anchor_id = self.motion.motion_anchor_body_index
        reference_p = reference_body_p[..., anchor_id, :]
        reference_r = reference_body_r[..., anchor_id, :, :]
        alignment_p, alignment_rotation = alignment_state[:2]
        yaw_delta = yaw_mat(alignment_rotation @ reference_r.transpose(-1, -2))
        anchor = torch.cat((alignment_p[..., :2], reference_p[..., 2:3]), dim=-1)
        aligned_body_p = anchor.unsqueeze(-2) + (
            yaw_delta.unsqueeze(-3)
            @ (reference_body_p - reference_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        aligned_body_r = yaw_delta.unsqueeze(-3) @ reference_body_r

        body_position_error = (aligned_body_p - body_p).square().sum(-1).mean(-1)
        body_orientation_error = rot_angle_sq(aligned_body_r, body_r).mean(-1)
        body_linear_velocity_error = (reference_body_v - body_v).square().sum(-1).mean(-1)
        body_angular_velocity_error = (reference_body_w - body_w).square().sum(-1).mean(-1)
        root_position_error = (reference_p - p).square().sum(-1)
        root_orientation_error = rot_angle_sq(reference_r, rotation)
        tracking = (
            cfg.body_position_weight * torch.exp(-body_position_error / cfg.body_position_std**2)
            + cfg.body_orientation_weight * torch.exp(-body_orientation_error / cfg.body_orientation_std**2)
            + cfg.body_linear_velocity_weight
            * torch.exp(-body_linear_velocity_error / cfg.body_linear_velocity_std**2)
            + cfg.body_angular_velocity_weight
            * torch.exp(-body_angular_velocity_error / cfg.body_angular_velocity_std**2)
            + cfg.root_position_weight * torch.exp(-root_position_error / cfg.root_position_std**2)
            + cfg.root_orientation_weight * torch.exp(-root_orientation_error / cfg.root_orientation_std**2)
        )
        action_rate = (action - previous_action).square().sum(-1)
        joint_limit = (torch.relu(q - self.soft_upper) + torch.relu(self.soft_lower - q)).sum(-1)
        return (
            tracking + cfg.action_rate_weight * action_rate + cfg.joint_limit_weight * joint_limit
        ) * cfg.control_dt

    def soft_termination(
        self,
        state,
        alignment_state,
        motion_id: torch.Tensor,
        frame: torch.Tensor,
        context: torch.Tensor,
        return_terms: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p, rotation, v, w, q, qd = state
        reference_body_p, reference_body_q, _, _, _ = self._reference(motion_id, frame)
        reference_body_r = quat_to_mat(reference_body_q)
        anchor_id = self.motion.motion_anchor_body_index
        reference_p = reference_body_p[..., anchor_id, :]
        reference_r = reference_body_r[..., anchor_id, :, :]
        height_error = (reference_p[..., 2] - p[..., 2]).abs()
        gravity = torch.tensor((0.0, 0.0, -1.0), device=p.device, dtype=p.dtype)
        reference_gravity_z = (reference_r.transpose(-1, -2) @ gravity.expand_as(p).unsqueeze(-1)).squeeze(-1)[..., 2]
        robot_gravity_z = (rotation.transpose(-1, -2) @ gravity.expand_as(p).unsqueeze(-1)).squeeze(-1)[..., 2]
        tilt_error = (reference_gravity_z - robot_gravity_z).abs()
        height_temperature = self.cfg.termination_height_temperature
        tilt_temperature = self.cfg.termination_tilt_temperature
        height_failure = torch.sigmoid((height_error - self.cfg.termination_height) / height_temperature)
        tilt_failure = torch.sigmoid((tilt_error - self.cfg.termination_tilt) / tilt_temperature)
        failure = 1.0 - (1.0 - height_failure) * (1.0 - tilt_failure)
        body_failure = torch.zeros_like(failure)
        if self.end_effector_body_ids:
            body_p = self.kinematics.forward(p, rotation, v, w, q, qd)[0]
            alignment_p, alignment_rotation = alignment_state[:2]
            yaw_delta = yaw_mat(alignment_rotation @ reference_r.transpose(-1, -2))
            anchor = torch.cat((alignment_p[..., :2], reference_p[..., 2:3]), dim=-1)
            aligned_body_p = anchor.unsqueeze(-2) + (
                yaw_delta.unsqueeze(-3)
                @ (reference_body_p - reference_p.unsqueeze(-2)).unsqueeze(-1)
            ).squeeze(-1)
            body_height_error = (
                aligned_body_p[..., self.end_effector_body_ids, 2]
                - body_p[..., self.end_effector_body_ids, 2]
            ).abs().amax(-1)
            body_failure = torch.sigmoid(
                (body_height_error - self.cfg.termination_body_height) / height_temperature
            )
            failure = 1.0 - (1.0 - failure) * (1.0 - body_failure)
        recovery_active = context[..., self.context_layout.recovery_active].squeeze(-1)
        enabled = 1.0 - recovery_active
        failure = failure * enabled
        if not return_terms:
            return failure
        return failure, {
            "height": height_failure * enabled,
            "tilt": tilt_failure * enabled,
            "end_effector": body_failure * enabled,
        }

    def critic_features(
        self,
        state,
        motion_id: torch.Tensor,
        frame: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        p, rotation, v, w, q, qd = state
        heading_t = yaw_mat(rotation).transpose(-1, -2)
        reference_body_p, reference_body_q, _, _, reference_q = self._reference(motion_id, frame)
        reference_body_r = quat_to_mat(reference_body_q)
        anchor_id = self.motion.motion_anchor_body_index
        reference_p = reference_body_p[..., anchor_id, :]
        reference_r = reference_body_r[..., anchor_id, :, :]
        body_p = self.kinematics.forward(p, rotation, v, w, q, qd)[0]
        yaw_delta = yaw_mat(rotation @ reference_r.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], reference_p[..., 2:3]), dim=-1)
        aligned_body_p = anchor.unsqueeze(-2) + (
            yaw_delta.unsqueeze(-3)
            @ (reference_body_p - reference_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        body_error = (
            heading_t.unsqueeze(-3) @ (aligned_body_p - body_p).unsqueeze(-1)
        ).squeeze(-1).flatten(-2)
        root_error = (heading_t @ (reference_p - p).unsqueeze(-1)).squeeze(-1)
        relative_rotation = mat_to_6d(rotation.transpose(-1, -2) @ reference_r)
        return torch.cat(
            (
                invariant_features(state),
                reference_q - q,
                body_error,
                root_error,
                relative_rotation,
                normalized_context,
            ),
            dim=-1,
        )
