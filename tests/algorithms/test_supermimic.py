from pathlib import Path
from types import SimpleNamespace

import torch

from rsl_rl.algorithms.supermimic.buffer import ReplayBuffer
from rsl_rl.algorithms.supermimic.kinematics import UrdfKinematics
from rsl_rl.algorithms.supermimic.mdp import (
    G1_ROBOT_SPEC,
    RewardConfig,
    SuperMimicMDP,
    SuperMimicRobotSpec,
    flatten_state,
    unflatten_state,
)
from rsl_rl.algorithms.supermimic.networks import WorldModel
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
    assert sample["a_prev"].shape == (64, 4, 29)
    assert sample["mask"].shape == (64, 4)
    assert torch.all((sample["mask"] == 0) | (sample["mask"] == 1))


def test_replay_samples_only_full_horizon_starts():
    replay = ReplayBuffer(time_steps=4, num_envs=2, device="cpu", robot_spec=G1_ROBOT_SPEC)
    frames = (0, 3, 4, 7)
    for step, frame in enumerate(frames):
        replay.add({
            "s": torch.zeros(2, 71), "s1": torch.zeros(2, 71),
            "a": torch.zeros(2, 29), "a_prev": torch.zeros(2, 29),
            "xi": torch.zeros(2, 33), "r": torch.zeros(2),
            "f": torch.full((2,), frame), "ep_step": torch.full((2,), step, dtype=torch.int32),
            "done": torch.zeros(2, dtype=torch.bool),
        })
    frame_clip_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    clip_ends = torch.tensor([4, 9])
    start, history, valid_ratio = replay.sample_valid_starts(
        64, 2, frame_clip_ids, clip_ends, horizon=2
    )
    assert torch.all((start["f"] == 0) | (start["f"] == 7))
    assert history["f"].shape == (64, 2)
    assert valid_ratio == 0.5


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


def test_full_physics_context_spec_and_recovery_expiry():
    spec = SuperMimicRobotSpec(
        num_joints=29,
        num_tracked_bodies=14,
        context_dim_override=219,
    )
    assert spec.context_dim == 219
    assert SuperMimicRobotSpec.from_dict(spec.to_dict()) == spec

    mdp = SuperMimicMDP.__new__(SuperMimicMDP)
    mdp.context_layout = SimpleNamespace(
        push_velocity=slice(0, 6),
        recovery_assist=slice(6, 7),
        recovery_active=slice(7, 8),
    )
    context = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 80.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0],
        ]
    )
    updated = mdp.advance_context(context, torch.tensor([149, 150, 20]), 150)
    torch.testing.assert_close(updated[:, :6], torch.zeros(3, 6))
    torch.testing.assert_close(updated[:, 6], torch.tensor([80.0, 0.0, 0.0]))
    torch.testing.assert_close(updated[:, 7], torch.tensor([1.0, 0.0, 0.0]))


def test_delayed_actuator_spec_and_world_model_inputs_include_previous_action():
    spec = SuperMimicRobotSpec(
        num_joints=2,
        num_tracked_bodies=1,
        context_dim_override=11,
        actuator_delay_groups=("legs", "arms"),
    )
    assert spec.uses_action_history
    assert SuperMimicRobotSpec.from_dict(spec.to_dict()) == spec

    model = WorldModel.__new__(WorldModel)
    torch.nn.Module.__init__(model)
    model.uses_action_history = True
    state = (
        torch.zeros(3, 3),
        torch.eye(3).expand(3, -1, -1),
        torch.zeros(3, 3),
        torch.zeros(3, 3),
        torch.zeros(3, 2),
        torch.zeros(3, 2),
    )
    previous_action = torch.full((3, 2), -1.0)
    action = torch.full((3, 2), 1.0)
    context = torch.zeros(3, 11)
    inputs = model.inputs(state, action, context, previous_action)
    assert inputs.shape == (3, spec.invariant_dim + 4 + spec.context_dim)
    torch.testing.assert_close(
        inputs[:, spec.invariant_dim : spec.invariant_dim + 2], previous_action
    )
    torch.testing.assert_close(
        inputs[:, spec.invariant_dim + 2 : spec.invariant_dim + 4], action
    )


def test_soft_termination_includes_end_effector_height_and_recovery_mask():
    class StubKinematics:
        def forward(self, p, rotation, v, w, q, qd):
            del p, rotation, v, w, q, qd
            body_position = torch.zeros(3, 3, 3)
            body_position[1:, 1, 2] = 0.2
            return body_position, None, None, None

    mdp = SuperMimicMDP.__new__(SuperMimicMDP)
    mdp.kinematics = StubKinematics()
    mdp.motion = SimpleNamespace(
        body_pos_w=torch.zeros(1, 3, 3),
        body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 3]),
    )
    mdp.cfg = RewardConfig(termination_body_height=0.12)
    mdp.end_effector_body_ids = (1, 2)
    mdp.context_layout = SimpleNamespace(recovery_active=slice(0, 1))
    state = (
        torch.zeros(3, 3),
        torch.eye(3).expand(3, -1, -1),
        torch.zeros(3, 3),
        torch.zeros(3, 3),
        torch.zeros(3, 1),
        torch.zeros(3, 1),
    )
    context = torch.tensor([[0.0], [0.0], [1.0]])
    failure, terms = mdp.soft_termination(
        state,
        torch.zeros(3, dtype=torch.long),
        context,
        return_terms=True,
    )
    assert failure[0] < 0.01
    assert failure[1] > 0.95
    assert failure[2] == 0.0
    assert terms["end_effector"][1] > 0.95
    assert terms["end_effector"][2] == 0.0


def test_reward_applies_soft_termination_penalty_with_control_dt():
    class StubKinematics:
        def forward(self, p, rotation, v, w, q, qd):
            del q, qd
            return p[:, None], rotation[:, None], v[:, None], w[:, None]

    mdp = SuperMimicMDP.__new__(SuperMimicMDP)
    mdp.kinematics = StubKinematics()
    mdp.motion = SimpleNamespace(
        body_pos_w=torch.zeros(1, 1, 3),
        body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        body_lin_vel_w=torch.zeros(1, 1, 3),
        body_ang_vel_w=torch.zeros(1, 1, 3),
    )
    mdp.soft_lower = torch.full((1,), -1.0)
    mdp.soft_upper = torch.full((1,), 1.0)
    mdp.cfg = RewardConfig(termination_penalty_weight=-1.0)
    state = (
        torch.zeros(2, 3),
        torch.eye(3).expand(2, -1, -1),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
    )
    action = torch.zeros(2, 1)
    reward = mdp.reward(
        state,
        torch.zeros(2, dtype=torch.long),
        action,
        action,
        termination=torch.tensor([0.0, 1.0]),
    )
    torch.testing.assert_close(reward[0] - reward[1], torch.tensor(0.02))


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
