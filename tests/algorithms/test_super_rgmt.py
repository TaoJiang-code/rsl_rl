from types import SimpleNamespace

import torch

from rsl_rl.algorithms.super_rgmt.algorithm import SuperRGMTAlgorithm
from rsl_rl.algorithms.super_rgmt.buffer import ReplayBuffer
from rsl_rl.algorithms.super_rgmt.mdp import SuperRGMTRobotSpec
from rsl_rl.runners.super_rgmt_runner import SuperRGMTRunner


def _transition(spec: SuperRGMTRobotSpec, num_envs: int, step: int) -> dict[str, torch.Tensor]:
    return {
        "s": torch.zeros(num_envs, spec.state_dim),
        "s1": torch.zeros(num_envs, spec.state_dim),
        "joint_target": torch.zeros(num_envs, spec.num_joints),
        "previous_action": torch.zeros(num_envs, spec.num_joints),
        "dropout_cached_command": torch.full((num_envs, spec.command_dim), float(step)),
        "dropout_remaining": torch.full((num_envs,), step, dtype=torch.int32),
        "dropout_active": torch.full((num_envs,), step > 0, dtype=torch.bool),
        "xi": torch.zeros(num_envs, spec.context_dim),
        "r": torch.zeros(num_envs),
        "f": torch.full((num_envs,), step, dtype=torch.long),
        "motion_id": torch.zeros(num_envs, dtype=torch.long),
        "ep_step": torch.full((num_envs,), step, dtype=torch.int32),
        "done": torch.zeros(num_envs, dtype=torch.bool),
        "time_out": torch.zeros(num_envs, dtype=torch.bool),
    }


def test_replay_preserves_dropout_start_state():
    spec = SuperRGMTRobotSpec(
        num_joints=2,
        num_tracked_bodies=1,
        context_dim=4,
        history_length=2,
        command_window_size=2,
        command_step_dim=19,
    )
    replay = ReplayBuffer(time_steps=4, num_envs=2, device="cpu", spec=spec)
    for step in range(3):
        replay.add(_transition(spec, 2, step))
    start, _ = replay.sample_starts(32, history_length=2)
    assert start["dropout_cached_command"].dtype == torch.float16
    assert torch.equal(start["dropout_active"], start["dropout_remaining"] > 0)


def test_imagined_dropout_continues_saved_burst():
    algorithm = SuperRGMTAlgorithm.__new__(SuperRGMTAlgorithm)
    algorithm.dropout_start_probability = 0.0
    algorithm.dropout_duration_range = (5, 5)
    command = torch.ones(2, 2, 3)
    cached = torch.zeros_like(command)
    remaining = torch.tensor([2, 0])
    output, next_cached, next_remaining, active = algorithm._apply_command_dropout(
        command, cached, remaining
    )
    torch.testing.assert_close(output[0], cached[0])
    torch.testing.assert_close(output[1], command[1])
    torch.testing.assert_close(next_cached[1], command[1])
    assert torch.equal(next_remaining, torch.tensor([1, 0]))
    assert torch.equal(active, torch.tensor([True, False]))


def test_runner_restores_observation_groups_from_legacy_wrapper_output():
    runner = SuperRGMTRunner.__new__(SuperRGMTRunner)
    runner.env = SimpleNamespace(num_envs=2, device="cpu")
    policy = torch.zeros(2, 3)
    history = torch.ones(2, 4, 5)

    observation = runner._to_tensordict(
        policy,
        {"observations": {"policy": policy, "proprio_history": history}},
    )

    assert set(observation.keys()) == {"policy", "proprio_history"}
    torch.testing.assert_close(observation["policy"], policy)
    torch.testing.assert_close(observation["proprio_history"], history)
