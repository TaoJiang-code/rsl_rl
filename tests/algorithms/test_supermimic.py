from pathlib import Path

import torch

from rsl_rl.algorithms.supermimic.buffer import ReplayBuffer
from rsl_rl.algorithms.supermimic.kinematics import UrdfKinematics
from rsl_rl.algorithms.supermimic.mdp import (
    G1_ROBOT_SPEC,
    SuperMimicRobotSpec,
    flatten_state,
    unflatten_state,
)
from rsl_rl.algorithms.supermimic.rotations import so3_exp, so3_log


JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
TRACKED_BODIES = [
    "base", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]


def test_state_and_rotation_round_trip():
    rotation_vector = torch.randn(32, 3) * 0.3
    rotation = so3_exp(rotation_vector)
    recovered = so3_log(rotation)
    torch.testing.assert_close(recovered, rotation_vector, atol=1e-5, rtol=1e-5)
    state = (
        torch.randn(32, 3), rotation, torch.randn(32, 3), torch.randn(32, 3),
        torch.randn(32, 29), torch.randn(32, 29),
    )
    rebuilt = unflatten_state(flatten_state(state))
    for actual, expected in zip(rebuilt, state):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_replay_window_masks_episode_boundaries():
    replay = ReplayBuffer(time_steps=8, num_envs=2, device="cpu", robot_spec=G1_ROBOT_SPEC)
    for step in range(6):
        replay.add({
            "s": torch.zeros(2, 71), "s1": torch.zeros(2, 71),
            "a": torch.zeros(2, 29), "a_prev": torch.zeros(2, 29),
            "xi": torch.zeros(2, 33), "r": torch.zeros(2),
            "f": torch.full((2,), step), "ep_step": torch.full((2,), step, dtype=torch.int32),
            "done": torch.tensor([step == 2, False]),
        })
    sample = replay.sample_windows(64, 4)
    assert sample["s"].shape == (64, 4, 71)
    assert sample["mask"].shape == (64, 4)
    assert torch.all((sample["mask"] == 0) | (sample["mask"] == 1))


def test_q1_robot_spec_dimensions():
    spec = SuperMimicRobotSpec(num_joints=22, num_tracked_bodies=12)
    assert spec.state_dim == 57
    assert spec.observation_dim == 78
    assert spec.context_dim == 26
    assert spec.goal_step_dim == 59
    assert spec.state_increment_dim == 56
    assert spec.critic_extra_dim == 93
    assert spec.goal_dim(10) == 590
    state = (
        torch.randn(4, 3),
        torch.eye(3).expand(4, -1, -1),
        torch.randn(4, 3),
        torch.randn(4, 3),
        torch.randn(4, 22),
        torch.randn(4, 22),
    )
    rebuilt = unflatten_state(flatten_state(state))
    for actual, expected in zip(rebuilt, state):
        torch.testing.assert_close(actual, expected)


def test_urdf_fk_shapes_and_joint_mapping():
    root = Path(__file__).resolve().parents[3]
    urdf = root / "Glush_Zoo/g1_description/urdf/g1_29dof_rev_1_0.urdf"
    kinematics = UrdfKinematics(urdf, JOINT_NAMES, TRACKED_BODIES, "base", "cpu")
    batch = 4
    identity = torch.eye(3).expand(batch, -1, -1)
    result = kinematics.forward(
        torch.zeros(batch, 3), identity, torch.zeros(batch, 3), torch.zeros(batch, 3),
        torch.zeros(batch, 29), torch.zeros(batch, 29),
    )
    assert result[0].shape == (batch, 14, 3)
    assert result[1].shape == (batch, 14, 3, 3)
    assert result[2].shape == (batch, 14, 3)
    assert result[3].shape == (batch, 14, 3)


def test_urdf_fk_returns_center_of_mass_linear_velocity():
    root = Path(__file__).resolve().parents[3]
    urdf = root / "Glush_Zoo/g1_description/urdf/g1_29dof_rev_1_0.urdf"
    kinematics = UrdfKinematics(urdf, JOINT_NAMES, TRACKED_BODIES, "base", "cpu")
    identity = torch.eye(3).unsqueeze(0)
    root_position = torch.zeros(1, 3)
    root_com_velocity = torch.tensor([[0.2, -0.1, 0.3]])
    root_angular_velocity = torch.tensor([[0.0, 0.0, 2.0]])
    body_position, body_rotation, body_velocity, _ = kinematics.forward(
        root_position,
        identity,
        root_com_velocity,
        root_angular_velocity,
        torch.zeros(1, 29),
        torch.zeros(1, 29),
    )
    tracked_com_offsets = kinematics.com_offsets[kinematics.tracked_indices]
    body_com_position = body_position + (
        body_rotation @ tracked_com_offsets.unsqueeze(-1)
    ).squeeze(-1)
    root_com_position = root_position + (
        identity @ kinematics.com_offsets[0].view(1, 3, 1)
    ).squeeze(-1)
    expected_velocity = root_com_velocity[:, None] + torch.cross(
        root_angular_velocity[:, None].expand_as(body_com_position),
        body_com_position - root_com_position[:, None],
        dim=-1,
    )
    torch.testing.assert_close(body_velocity, expected_velocity, atol=1e-5, rtol=1e-5)
