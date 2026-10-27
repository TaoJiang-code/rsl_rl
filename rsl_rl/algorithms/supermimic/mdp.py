"""Differentiable SuperMimic observation, reward, termination, and critic heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, yaw_mat

CONTROL_DT = 0.02


@dataclass(frozen=True)
class SuperMimicRobotSpec:
    """Robot-dependent tensor dimensions used by the shared SuperMimic implementation."""

    num_joints: int
    num_tracked_bodies: int
    context_dim_override: int | None = None
    actuator_delay_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.num_joints <= 0:
            raise ValueError("SuperMimic requires at least one joint.")
        if self.num_tracked_bodies <= 0:
            raise ValueError("SuperMimic requires at least one tracked body.")
        if self.context_dim_override is not None and self.context_dim_override <= 0:
            raise ValueError("SuperMimic context dimension must be positive.")

    @property
    def state_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def observation_dim(self) -> int:
        return 12 + 3 * self.num_joints

    @property
    def context_dim(self) -> int:
        return 4 + self.num_joints if self.context_dim_override is None else self.context_dim_override

    @property
    def uses_action_history(self) -> bool:
        return bool(self.actuator_delay_groups)

    @property
    def goal_step_dim(self) -> int:
        return 15 + 2 * self.num_joints

    @property
    def invariant_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def state_increment_dim(self) -> int:
        return 12 + 2 * self.num_joints

    @property
    def critic_extra_dim(self) -> int:
        return self.num_joints + 3 * self.num_tracked_bodies + 9 + self.context_dim

    def goal_dim(self, goal_frames: int) -> int:
        return self.goal_step_dim * goal_frames

    def to_dict(self) -> dict[str, int | list[str]]:
        value = {
            "num_joints": self.num_joints,
            "num_tracked_bodies": self.num_tracked_bodies,
        }
        if self.context_dim_override is not None and self.context_dim_override != 4 + self.num_joints:
            value["context_dim"] = self.context_dim_override
        if self.actuator_delay_groups:
            value["actuator_delay_groups"] = list(self.actuator_delay_groups)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, int | list[str]]) -> "SuperMimicRobotSpec":
        return cls(
            num_joints=int(value["num_joints"]),
            num_tracked_bodies=int(value["num_tracked_bodies"]),
            context_dim_override=(
                int(value["context_dim"]) if "context_dim" in value else None
            ),
            actuator_delay_groups=tuple(value.get("actuator_delay_groups", ())),
        )


# Backward-compatible constants for the existing G1 task and external imports.
G1_ROBOT_SPEC = SuperMimicRobotSpec(num_joints=29, num_tracked_bodies=14)
NUM_JOINTS = G1_ROBOT_SPEC.num_joints
NUM_TRACKED_BODIES = G1_ROBOT_SPEC.num_tracked_bodies
STATE_DIM = G1_ROBOT_SPEC.state_dim
OBS_DIM = G1_ROBOT_SPEC.observation_dim
XI_DIM = G1_ROBOT_SPEC.context_dim
GOAL_STEP_DIM = G1_ROBOT_SPEC.goal_step_dim
INV_DIM = G1_ROBOT_SPEC.invariant_dim
STATE_INCREMENT_DIM = G1_ROBOT_SPEC.state_increment_dim
CRITIC_EXTRA_DIM = G1_ROBOT_SPEC.critic_extra_dim


def flatten_state(state) -> torch.Tensor:
    p, rotation, v, w, q, qd = state
    return torch.cat((p, mat_to_quat(rotation), v, w, q, qd), -1)


def unflatten_state(value: torch.Tensor, num_joints: int | None = None):
    if num_joints is None:
        num_joints = (value.shape[-1] - 13) // 2
    expected_dim = 13 + 2 * num_joints
    if value.shape[-1] != expected_dim:
        raise ValueError(f"State has dimension {value.shape[-1]}, expected {expected_dim}.")
    joint_position_end = 13 + num_joints
    return (
        value[..., 0:3],
        quat_to_mat(value[..., 3:7]),
        value[..., 7:10],
        value[..., 10:13],
        value[..., 13:joint_position_end],
        value[..., joint_position_end:expected_dim],
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
    calibration_end = context.shape[-1] - 3
    return torch.cat(
        (
            (context[..., :1] - 0.75) / 0.45,
            context[..., 1:calibration_end] / 0.01,
            context[..., calibration_end:] / 0.05,
        ),
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
    termination_body_height: float = 0.25


class SuperMimicMDP:
    def __init__(
        self,
        kinematics,
        motion,
        default_joint_position,
        soft_limits,
        robot_spec: SuperMimicRobotSpec,
        context_layout=None,
        end_effector_body_names: tuple[str, ...] = (),
        cfg=RewardConfig(),
    ):
        self.kinematics = kinematics
        self.motion = motion
        self.robot_spec = robot_spec
        self.context_layout = context_layout
        self.default_joint_position = default_joint_position
        self.soft_lower = soft_limits[..., 0]
        self.soft_upper = soft_limits[..., 1]
        self.cfg = cfg
        tracked_body_names = tuple(motion.cfg.tracked_body_names)
        missing = set(end_effector_body_names).difference(tracked_body_names)
        if missing:
            raise ValueError(f"SuperMimic termination bodies are not tracked: {sorted(missing)}.")
        self.end_effector_body_ids = tuple(
            tracked_body_names.index(name) for name in end_effector_body_names
        )
        device = default_joint_position.device
        self.obs_noise = torch.cat(
            (
                torch.full((robot_spec.num_joints,), 0.01),
                torch.full((robot_spec.num_joints,), 0.5),
                torch.full((3,), 0.2),
                torch.full((3,), 0.05),
                torch.zeros(robot_spec.num_joints),
                torch.full((6,), 0.05),
            )
        ).to(device)

    def goal(self, frame: torch.Tensor, count: int) -> torch.Tensor:
        return self.motion.goal(frame, count)

    def observation(self, state, previous_action, context, frame) -> torch.Tensor:
        _, rotation, _, w, q, qd = state
        rotation_t = rotation.transpose(-1, -2)
        calibration_slice = (
            slice(1, 1 + self.robot_spec.num_joints)
            if self.context_layout is None
            else self.context_layout.calibration_offset
        )
        calibration = context[..., calibration_slice]
        projected_gravity = -rotation[..., 2, :]
        relative_reference = mat_to_6d(
            rotation_t
            @ quat_to_mat(self.motion.gather_command_motion("body_quat_w", frame)[..., 0, :])
        )
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

    def advance_context(
        self,
        context: torch.Tensor,
        episode_step: torch.Tensor,
        recovery_window_steps: int,
    ) -> torch.Tensor:
        if self.context_layout is None:
            return context
        updated = context.clone()
        push_velocity = getattr(self.context_layout, "push_velocity", None)
        if push_velocity is not None:
            # A velocity impulse is instantaneous. It affects only the transition carrying it.
            updated[..., push_velocity] = 0.0
        was_active = context[..., self.context_layout.recovery_active].squeeze(-1) > 0.5
        active = was_active & (episode_step < recovery_window_steps)
        updated[..., self.context_layout.recovery_active] = active[..., None].to(context.dtype)
        assist = updated[..., self.context_layout.recovery_assist]
        updated[..., self.context_layout.recovery_assist] = assist * active[..., None]
        return updated

    def soft_termination(
        self,
        state,
        frame,
        context: torch.Tensor | None = None,
        height_temperature=0.02,
        tilt_temperature=0.04,
        return_terms: bool = False,
    ):
        p, rotation, v, w, q, qd = state
        reference_p = self.motion.body_pos_w[frame, 0]
        reference_r = quat_to_mat(self.motion.body_quat_w[frame, 0])
        height_error = (reference_p[..., 2] - p[..., 2]).abs()
        tilt_error = (reference_r[..., 2, 2] - rotation[..., 2, 2]).abs()
        height_failure = torch.sigmoid(
            (height_error - self.cfg.termination_height) / height_temperature
        )
        tilt_failure = torch.sigmoid(
            (tilt_error - self.cfg.termination_tilt) / tilt_temperature
        )
        failure = 1.0 - (1.0 - height_failure) * (1.0 - tilt_failure)
        body_failure = torch.zeros_like(failure)
        if self.end_effector_body_ids:
            body_p = self.kinematics.forward(p, rotation, v, w, q, qd)[0]
            reference_body_p = self.motion.body_pos_w[frame]
            body_height_error = (
                reference_body_p[..., self.end_effector_body_ids, 2]
                - body_p[..., self.end_effector_body_ids, 2]
            ).abs().amax(-1)
            body_failure = torch.sigmoid(
                (body_height_error - self.cfg.termination_body_height) / height_temperature
            )
            failure = 1.0 - (1.0 - failure) * (1.0 - body_failure)

        enabled = torch.ones_like(failure)
        if context is not None and self.context_layout is not None:
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
