"""Recovery AMP regressions independent of Isaac Sim."""

from types import SimpleNamespace

import pytest
import torch

from rsl_rl.algorithms.supermimic.buffer import ReplayBuffer
from rsl_rl.algorithms.supermimic.exporter import _SuperMimicExportPolicy
from rsl_rl.algorithms.supermimic.mdp import SuperMimicRobotSpec, unflatten_state
from rsl_rl.algorithms.supermimic.networks import Policy
from rsl_rl.algorithms.supermimic.recovery_amp import RecoveryAMP


def make_motion():
    n = 12
    quat = torch.zeros(n, 1, 4)
    quat[..., 0] = 1
    return SimpleNamespace(
        num_recovery_clips=2,
        cfg=SimpleNamespace(fps=50),
        recovery_body_pos_w=torch.randn(n, 1, 3) * 0.1,
        recovery_body_quat_w=quat,
        recovery_body_lin_vel_w=torch.randn(n, 1, 3) * 0.1,
        recovery_body_ang_vel_w=torch.randn(n, 1, 3) * 0.1,
        recovery_joint_pos=torch.randn(n, 2) * 0.1,
        recovery_joint_vel=torch.randn(n, 2) * 0.1,
        recovery_clip_starts=torch.tensor([0, 6]),
        recovery_clip_ends=torch.tensor([5, 11]),
    )


def make_amp():
    return RecoveryAMP(make_motion(), {"recovery_amp_batch_size": 8, "amp_reward_coef": 0.1, "amp_task_reward_lerp": 0.75}, "cpu")


def test_expert_pairs_do_not_cross_clip_boundaries():
    amp = make_amp()
    amp.expert = torch.arange(12).float()[:, None].expand(-1, 17)
    first, second = amp.sample_expert(512)
    assert not ((first[:, 0] == 5) | (first[:, 0] == 11)).any()
    torch.testing.assert_close(second, first + 1)
    assert (first[:, 0] < 6).any() and (first[:, 0] >= 6).any()


def test_discriminator_updates_and_reward_keeps_state_gradients():
    torch.manual_seed(7)
    amp = make_amp()
    replay = ReplayBuffer(4, 8, "cpu", SuperMimicRobotSpec(2, 1, context_dim_override=1))
    for data in replay.data.values():
        data.zero_()
    replay.data["s"][..., 3] = 1
    replay.data["s1"][..., 3] = 1
    replay.data["s1"][..., 13:] = 0.1
    replay.data["xi"][..., 0] = 1
    replay.size = 4
    before = [p.clone() for p in amp.discriminator.parameters()]
    amp.update(replay, slice(0, 1))
    assert amp.updates == 1 and torch.isfinite(torch.tensor(amp.last_loss))
    assert any(not torch.equal(a, b) for a, b in zip(before, amp.discriminator.parameters()))
    state = replay.data["s"][0, :2].clone()
    nxt = replay.data["s1"][0, :2].clone().requires_grad_()
    reward = amp.reward(unflatten_state(state), unflatten_state(nxt), torch.tensor([1.0, 0.0]))
    assert reward[1] == 0
    reward.sum().backward()
    assert nxt.grad[0].abs().sum() > 0
    assert nxt.grad[1].abs().sum() == 0
    assert all(p.grad is None and not p.requires_grad for p in amp.discriminator.parameters())
    restored = make_amp()
    restored.load_state_dict(amp.state_dict())
    torch.testing.assert_close(
        restored.reward(unflatten_state(state), unflatten_state(nxt.detach()), torch.tensor([1.0, 0.0])),
        reward.detach(),
    )
    assert restored.updates == 1 and restored.optimizer.state_dict()["state"]


def test_no_recovery_samples_skip_discriminator_update():
    amp = make_amp()
    replay = ReplayBuffer(2, 2, "cpu", SuperMimicRobotSpec(2, 1, context_dim_override=1))
    replay.size = 2
    amp.update(replay, slice(0, 1))
    assert amp.updates == 0


def test_single_decoder_export_has_no_gate_or_recovery_weights():
    policy = Policy(18, 38, 2, history_length=2)
    assert not any(k.startswith(("recovery_decoder.", "recovery_gate.")) for k in policy.state_dict())
    obs, goal = torch.randn(3, 2, 18), torch.randn(3, 38)
    expected = policy.decode(obs, policy.encode_motion(goal))
    torch.testing.assert_close(policy(obs, goal), expected)
    export = _SuperMimicExportPolicy(policy)
    traced = torch.jit.trace(export, (obs, goal))
    torch.testing.assert_close(traced(obs, goal), expected)


def test_expert_frequency_mismatch_rejected():
    motion = make_motion()
    motion.cfg.fps = 30
    with pytest.raises(ValueError, match="frequency"):
        RecoveryAMP(motion, {"amp_reward_coef": 0.1, "amp_task_reward_lerp": 0.75}, "cpu")


@pytest.mark.parametrize("mixed", [False, True])
def test_small_full_algorithm_update_with_amp_and_window_expiry(mixed):
    from rsl_rl.algorithms.supermimic.algorithm import SuperMimicAlgorithm
    from rsl_rl.algorithms.supermimic.mdp import SuperMimicMDP, RewardConfig

    motion = make_motion()
    motion.cfg.joint_names = ("a", "b")
    motion.cfg.tracked_body_names = ("base",)
    for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel"):
        setattr(motion, name, getattr(motion, "recovery_" + name))
    motion.clip_ends = torch.tensor([11])
    motion.frame_clip_ids = torch.zeros(12, dtype=torch.long)
    motion.gather_command_motion = lambda name, frames: getattr(motion, name)[frames]
    motion.future_frames = lambda frames, count: (frames[:, None] + torch.arange(1, count + 1)).clamp_max(11)
    motion.goal = lambda frames, count: torch.zeros(*frames.shape, count * 19)
    layout = SimpleNamespace(
        calibration_offset=slice(0, 2),
        recovery_assist=slice(2, 3),
        recovery_active=slice(3, 4),
        push_velocity=slice(4, 10),
        stiffness=slice(4, 6),
        damping=slice(6, 8),
        effort_limit=slice(8, 10),
    )
    kin = SimpleNamespace(
        forward=lambda p, r, v, w, q, qd: (p.unsqueeze(-2), r.unsqueeze(-3), v.unsqueeze(-2), w.unsqueeze(-2))
    )
    mdp = SuperMimicMDP(
        kin,
        motion,
        torch.zeros(2),
        torch.tensor([[-2.0, 2.0], [-2.0, 2.0]]),
        SuperMimicRobotSpec(2, 1, context_dim_override=10),
        context_layout=layout,
        cfg=RewardConfig(),
        teleop_body_names=("base",),
    )
    cfg = dict(
        goal_frames=2,
        history_length=2,
        imagination_horizon=2,
        world_model_window=2,
        learning_rate=1e-4,
        replay_steps=6,
        recovery_amp=True,
        amp_reward_coef=0.1,
        amp_task_reward_lerp=0.75,
        recovery_amp_batch_size=4,
        compile=False,
        amp=None,
        policy_batch_size=2,
        world_model_batch_size=2,
        gamma=0.99,
        gae_lambda=0.95,
        critic_epochs=1,
        critic_minibatches=1,
        recovery_window_steps=3,
        disable_critic=False,
        self_imitation_weight=1.0,
        value_gradient_consistency_weight=0.5,
        value_gradient_consistency_samples=2,
    )
    cfg.update(
        mixed_motion_teleop=mixed,
        velocity_warmup_updates=0,
        velocity_command_ranges=((-0.4, 0.7), (-0.4, 0.4), (-1.57, 1.57)),
        velocity_standing_probability=0.05,
        velocity_resampling_time_range=(3.0, 8.0),
        task_switch_probability=0.0,
        velocity_alignment_weight=0.1,
    )
    algorithm = SuperMimicAlgorithm(mdp, 2, cfg, "cpu")
    algorithm.motion_initialized = True
    for step in range(6):
        sample = {name: data[0].clone() for name, data in algorithm.replay.data.items()}
        sample["s"][:, 3] = sample["s1"][:, 3] = 1
        sample["xi"][:, 3] = 1
        if mixed:
            sample["teleop_task"][1] = True
            sample["velocity_time_left"][:] = 5.0
        sample["f"][:] = step
        sample["ep_step"][:] = 2  # The next state reaches the three-step test window boundary.
        sample["reference_yaw_rotation"][:] = torch.eye(3)
        algorithm.replay.add(sample)
    algorithm.recovery_amp.update(algorithm.replay, layout.recovery_active)
    loss, _ = algorithm.world_model_update()
    assert torch.isfinite(loss)
    result = algorithm.policy_update()
    assert all(torch.isfinite(v).all() for v in result.values())
    assert not any(name.startswith("gate_") for name in result)
    assert result["imagined_recovery_ratio"] < 1
    saved = algorithm.state_dict()
    algorithm.load_state_dict(saved)
    assert algorithm.recovery_amp.updates == 1
    with pytest.raises(ValueError, match="architecture"):
        algorithm.load_state_dict({**saved, "training_schema": "obsolete"})


def test_environment_window_and_reset_selection_without_hidden_reference():
    # Execute the actual command methods without importing Isaac Sim.
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3] / "source/GlushLab/GlushLab/tasks/manager_based/supermimic/mdp/commands.py"
    )
    if not path.exists():
        pytest.skip("GlushLab integration source is not present in this rsl_rl checkout")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SuperMimicMotionCommand")
    selected = {"_sample_recovery_env_ids", "in_recovery_window", "advance"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    namespace = dict(
        torch=torch,
        sample_uniform=lambda low, high, shape, device: torch.full(shape, (low + high) / 2, device=device),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    Command = type("CommandUnderTest", (), {name: namespace[name] for name in selected})
    command = Command()
    command.device = "cpu"
    command.cfg = SimpleNamespace(recovery_probability=1.0, recovery_assist_force_range=(0.0, 200.0))
    command._env = SimpleNamespace(step_dt=0.02, episode_length_buf=torch.tensor([149, 150]))
    command.recovery_mask = torch.zeros(2, dtype=torch.bool)
    command.recovery_assist_raw_n = torch.zeros(2)
    command.motion_ids = torch.zeros(2, dtype=torch.long)
    command._recovery_window_steps = 150
    command.time_steps = torch.full((2,), 99)
    command.clip_ends, command.clip_starts = torch.tensor([199]), torch.tensor([0])
    command.current_clip_ends = torch.full((2,), 199)
    command.num_recovery_clips, command.recovery_steps_elapsed = 1, 0
    ids = command._sample_recovery_env_ids(torch.arange(2))
    assert ids.tolist() == [0, 1]
    assert command.time_steps.tolist() == [48, 48]
    assert command.in_recovery_window.tolist() == [True, False]
    command.advance()  # No hidden-reference arrays were installed: recovery must not read them.
    assert command.time_steps.tolist() == [49, 49]
    assert command.recovery_steps_elapsed == 1


def test_configured_reward_blend_and_window_mask():
    amp = make_amp()
    state = torch.zeros(2, 17)
    state[:, 3] = 1
    active = torch.tensor([1.0, 0.0])
    task = torch.tensor([2.0, 2.0], requires_grad=True)
    torch.testing.assert_close(amp.mix_reward(task, torch.zeros(2), active), task)
    amp.updates = 1
    for parameter in amp.discriminator.parameters():
        parameter.data.zero_()
    # D=0 gives style score 0.75, scaled by configured coefficient 0.1 (no dt).
    style = amp.reward(unflatten_state(state), unflatten_state(state), active)
    torch.testing.assert_close(style, torch.tensor([0.075, 0.0]))
    blended = amp.mix_reward(task, style, active)
    torch.testing.assert_close(blended, torch.tensor([1.51875, 2.0]))
    blended.sum().backward()
    torch.testing.assert_close(task.grad, torch.tensor([0.75, 1.0]))
