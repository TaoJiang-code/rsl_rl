"""SuperMimic Algorithm A.1 updates, adapted to an IsaacLab transition source."""

from __future__ import annotations

import copy
import math
import os
from itertools import chain

import torch
import torch.nn as nn

from .buffer import ReplayBuffer
from .mdp import (
    SuperMimicMDP,
    invariant_features,
    normalize_context,
    unflatten_state,
)
from .networks import Critic, Policy, RunningNorm, WorldModel
from . import networks
from .rotations import mat33_mul


class SuperMimicAlgorithm:
    def __init__(self, mdp: SuperMimicMDP, num_envs: int, cfg: dict, device: str):
        self.mdp = mdp
        self.motion = mdp.motion
        self.robot_spec = mdp.robot_spec
        self.cfg = cfg
        self.device = device
        self.goal_frames = int(cfg["goal_frames"])
        self.history_length = int(cfg["history_length"])
        self.horizon = int(cfg["imagination_horizon"])
        self.world_model_window = int(cfg["world_model_window"])
        self.base_learning_rate = float(cfg["learning_rate"])
        self.mixed_tasks = bool(cfg.get("mixed_motion_teleop", False))
        self.structured_world_model = bool(cfg.get("structured_world_model", False))
        self.self_imitation_weight = float(cfg.get("self_imitation_weight", 0.0))
        self.self_imitation_weight_cap = float(cfg.get("self_imitation_weight_cap", 5.0))
        self.value_gradient_consistency_weight = float(
            cfg.get("value_gradient_consistency_weight", 0.0)
        )
        self.value_gradient_consistency_samples = int(
            cfg.get("value_gradient_consistency_samples", 4096)
        )
        self.value_gradient_consistency_epochs = int(
            cfg.get("value_gradient_consistency_epochs", 1)
        )
        self.policy_ema_decay = float(cfg.get("policy_ema_decay", 0.0))
        self.compile_enabled = bool(cfg.get("compile", True))
        self.compile_mode = str(cfg.get("compile_mode", "reduce-overhead"))
        self.compile_fullgraph = bool(cfg.get("compile_fullgraph", False))
        self.async_replay_prefetch = bool(cfg.get("async_replay_prefetch", True))
        self.fused_optimizer = bool(cfg.get("fused_optimizer", True)) and str(device).startswith("cuda")
        if str(device).startswith("cuda") and bool(cfg.get("allow_tf32", True)):
            torch.set_float32_matmul_precision("high")
        goal_dim = self.robot_spec.goal_dim(self.goal_frames)
        teleop_goal_dim = (
            self.robot_spec.teleop_goal_dim(self.goal_frames) if self.mixed_tasks else None
        )
        critic_dim = self.robot_spec.invariant_dim + self.robot_spec.critic_extra_dim + goal_dim

        networks.AMP_DTYPE = torch.bfloat16 if cfg.get("amp", "bf16") == "bf16" else None
        self.world_model = WorldModel(
            self.robot_spec,
            structured=self.structured_world_model,
            control_dt=float(cfg.get("control_dt", 0.02)),
        ).to(device)
        self.policy = Policy(
            self.robot_spec.observation_dim,
            goal_dim,
            self.robot_spec.num_joints,
            self.history_length,
            teleop_goal_dim=teleop_goal_dim,
        ).to(device)
        self.critic = Critic(critic_dim).to(device)
        self.recovery_amp = None
        if cfg.get("recovery_amp", False):
            from .recovery_amp import RecoveryAMP

            if self.mdp.context_layout is None:
                raise ValueError("Recovery AMP requires full physics context.")
            self.recovery_amp = RecoveryAMP(self.motion, cfg, device)
        self.policy_ema = None
        if self.policy_ema_decay > 0.0:
            self.policy_ema = copy.deepcopy(self.policy).to(device).eval()
            self._requires_grad(self.policy_ema, False)
        self.policy_ema_updates = 0
        self.velocity_critic = None
        if self.mixed_tasks:
            velocity_critic_dim = (
                self.robot_spec.invariant_dim
                + self.robot_spec.num_joints
                + self.robot_spec.context_dim
                + int(teleop_goal_dim)
            )
            self.velocity_critic = Critic(velocity_critic_dim).to(device)
        self.context_norm = (
            RunningNorm(self.robot_spec.context_dim).to(device)
            if self.mdp.context_layout is not None
            else None
        )
        optimizer_kwargs = {"lr": self.base_learning_rate}
        if self.fused_optimizer:
            optimizer_kwargs["fused"] = True
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), **optimizer_kwargs
        )
        if self.mixed_tasks:
            self.policy_optimizer = torch.optim.Adam(
                chain(self.policy.encoder.parameters(), self.policy.decoder.parameters()),
                betas=(0.9, 0.95),
                **optimizer_kwargs,
            )
            self.teleop_optimizer = torch.optim.Adam(
                self.policy.teleop_encoder.parameters(),
                betas=(0.9, 0.95),
                **optimizer_kwargs,
            )
        else:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), betas=(0.9, 0.95), **optimizer_kwargs
            )
            self.teleop_optimizer = None
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), **optimizer_kwargs
        )
        self.velocity_critic_optimizer = (
            None
            if self.velocity_critic is None
            else torch.optim.Adam(self.velocity_critic.parameters(), **optimizer_kwargs)
        )
        self.replay = ReplayBuffer(
            int(cfg["replay_steps"]),
            num_envs,
            device,
            self.robot_spec,
            mixed_tasks=self.mixed_tasks,
        )
        self._world_model_prefetch_stream = None
        self._world_model_prefetched = None
        if self.async_replay_prefetch and str(device).startswith("cuda"):
            self._world_model_prefetch_stream = torch.cuda.Stream(device=torch.device(device))
        action_scale = torch.as_tensor(
            cfg.get("action_scale", [1.0] * self.robot_spec.num_joints),
            device=device,
            dtype=torch.float32,
        ).flatten()
        if action_scale.numel() == 1:
            action_scale = action_scale.expand(self.robot_spec.num_joints).clone()
        if action_scale.numel() != self.robot_spec.num_joints:
            raise ValueError(
                f"SuperMimic action_scale has {action_scale.numel()} entries, expected "
                f"{self.robot_spec.num_joints}."
            )
        self.action_scale = action_scale
        self.switch_action_continuity_weight = float(
            cfg.get("switch_action_continuity_weight", 0.0)
        )
        continuity_joint_names = tuple(
            cfg.get("switch_action_continuity_joint_names", ())
        )
        joint_names = tuple(self.motion.cfg.joint_names)
        unknown_continuity_joints = sorted(
            set(continuity_joint_names) - set(joint_names)
        )
        if unknown_continuity_joints:
            raise ValueError(
                "Unknown switch-action continuity joints: "
                f"{unknown_continuity_joints}. Available joints: {joint_names}."
            )
        self.switch_action_continuity_joint_indices = torch.tensor(
            [joint_names.index(name) for name in continuity_joint_names],
            device=device,
            dtype=torch.long,
        )
        self.error_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((self.robot_spec.num_joints,), 3e-3),
                torch.full((self.robot_spec.num_joints,), 5e-2),
            )
        ).to(device)
        self.world_model_rollout = self._world_model_rollout
        self.world_model_error = self._world_model_error
        self.imagination_step = self._imagination_step
        self.imagination_rollout = self._motion_imagination_rollout
        self.reward_termination = self._reward_termination
        self.critic_input = self._critic_input
        self.teleop_critic_input = self._teleop_critic_input
        if self.compile_enabled:
            compile_threads = int(cfg.get("compile_threads", 0))
            if compile_threads > 0:
                os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
                from torch._inductor import config as inductor_config

                inductor_config.compile_threads = compile_threads
                print(f"[INFO] TorchInductor compile threads: {compile_threads}")
            compile_options = {
                "mode": self.compile_mode,
                "fullgraph": self.compile_fullgraph,
                "dynamic": False,
            }
            reusable_output_compile_options = {
                "mode": "default",
                "fullgraph": self.compile_fullgraph,
                "dynamic": False,
            }
            self.world_model_rollout = torch.compile(self._world_model_rollout, **compile_options)
            self.world_model_error = torch.compile(self._world_model_error, **compile_options)
            self.imagination_step = torch.compile(
                self._imagination_step, **reusable_output_compile_options
            )
            self.imagination_rollout = torch.compile(
                self._motion_imagination_rollout, **compile_options
            )
            self.reward_termination = torch.compile(self._reward_termination, **compile_options)
            self.critic_input = torch.compile(
                self._critic_input, **reusable_output_compile_options
            )
            if self.mixed_tasks:
                self.teleop_critic_input = torch.compile(
                    self._teleop_critic_input, **reusable_output_compile_options
                )
        if self.mixed_tasks:
            self.imagination_step = self._mixed_imagination_step
            self.reward_termination = self._mixed_reward_termination
            if self.compile_enabled:
                self.imagination_step = torch.compile(
                    self._mixed_imagination_step, **reusable_output_compile_options
                )
                self.reward_termination = torch.compile(
                    self._mixed_reward_termination, **compile_options
                )
        print(
            "[INFO] SuperMimic GPU execution: "
            f"compile={self.compile_enabled} mode={self.compile_mode} "
            f"fused_adam={self.fused_optimizer} async_prefetch="
            f"{self._world_model_prefetch_stream is not None}"
        )
        self.motion_initialized = not self.mixed_tasks
        self.velocity_warmup_completed = 0

    def learning_rate(self, iteration: int) -> float:
        decay_iterations = int(self.cfg["learning_rate_decay_iterations"])
        minimum_ratio = float(self.cfg["minimum_learning_rate_ratio"])
        if iteration >= decay_iterations:
            return self.base_learning_rate * minimum_ratio
        cosine = 0.5 * (1 + math.cos(math.pi * iteration / decay_iterations))
        return self.base_learning_rate * (minimum_ratio + (1 - minimum_ratio) * cosine)

    @property
    def in_velocity_warmup(self) -> bool:
        return self.mixed_tasks and self.velocity_warmup_completed < int(
            self.cfg.get("velocity_warmup_updates", 500)
        )

    @property
    def training_stage(self) -> str:
        if not self.mixed_tasks:
            return "motion"
        return "velocity_warmup" if self.in_velocity_warmup else "joint"

    def set_learning_rate(self, iteration: int) -> float:
        learning_rate = self.learning_rate(iteration)
        optimizers = [self.world_model_optimizer, self.policy_optimizer, self.critic_optimizer]
        if self.teleop_optimizer is not None:
            optimizers.append(self.teleop_optimizer)
        if self.velocity_critic_optimizer is not None:
            optimizers.append(self.velocity_critic_optimizer)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        return learning_rate

    @torch.no_grad()
    def update_normalizers(
        self,
        transition: dict[str, torch.Tensor],
        observation,
        goal,
        teleop_goal: torch.Tensor | None = None,
        teleop_task: torch.Tensor | None = None,
    ) -> None:
        if not self.in_velocity_warmup:
            self.policy.obs_norm.update(observation)
        if self.mixed_tasks:
            if teleop_goal is None or teleop_task is None or self.policy.teleop_goal_norm is None:
                raise ValueError("Mixed SuperMimic normalizers require teleop goal and task mask.")
            motion = ~teleop_task.bool()
            if not self.in_velocity_warmup:
                self.policy.goal_norm.update(goal, motion)
            self.policy.teleop_goal_norm.update(teleop_goal, teleop_task)
        else:
            self.policy.goal_norm.update(goal)
        state = unflatten_state(transition["s"])
        next_state = unflatten_state(transition["s1"])
        if self.context_norm is not None:
            self.context_norm.update(transition["xi"])
        context = self._normalize_context(transition["xi"])
        self.world_model.in_norm.update(
            self.world_model.inputs(
                state,
                transition["a"],
                context,
                transition["a_prev"],
            )
        )
        self.world_model.out_norm.update(self.world_model.targets(state, next_state))
        self.world_model.increment_norm.update(self.world_model.increments(state, next_state))

    def _normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        if self.context_norm is not None:
            return self.context_norm(context)
        return normalize_context(context)

    def _world_model_rollout(self, state, action, previous_action, context):
        state = unflatten_state(state[:, 0])
        predictions = []
        for step in range(self.world_model_window):
            state = self.world_model.step(
                state,
                action[:, step],
                self._normalize_context(context[:, step]),
                previous_action[:, step],
            )
            predictions.append(state)
        predicted = tuple(torch.stack([value[index] for value in predictions], 1) for index in range(6))
        return predicted

    def _world_model_error(self, predicted, next_state, mask):
        target = unflatten_state(next_state)
        std = torch.maximum(self.world_model.increment_norm.std(), self.error_floor)
        error = (self.world_model.state_error(predicted, target) / std).square().mean(-1)
        return (error * mask).sum() / next_state.shape[0], error[:, -1]

    def _schedule_world_model_prefetch(self) -> None:
        if self._world_model_prefetch_stream is None:
            return
        batch_size = int(self.cfg["world_model_batch_size"])
        with torch.cuda.stream(self._world_model_prefetch_stream):
            self._world_model_prefetched = self.replay.sample_windows(
                batch_size, self.world_model_window
            )

    def _next_world_model_sample(self) -> dict[str, torch.Tensor]:
        if self._world_model_prefetch_stream is None:
            return self.replay.sample_windows(
                int(self.cfg["world_model_batch_size"]), self.world_model_window
            )
        if self._world_model_prefetched is None:
            self._schedule_world_model_prefetch()
        current_stream = torch.cuda.current_stream(device=torch.device(self.device))
        current_stream.wait_stream(self._world_model_prefetch_stream)
        sample = self._world_model_prefetched
        for value in sample.values():
            value.record_stream(current_stream)
        self._world_model_prefetched = None
        self._schedule_world_model_prefetch()
        return sample

    def seal_replay_prefetch(self) -> None:
        """Order later replay writes after the side-stream gather without a host sync."""
        if self._world_model_prefetch_stream is not None:
            torch.cuda.current_stream(device=torch.device(self.device)).wait_stream(
                self._world_model_prefetch_stream
            )

    def world_model_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.world_model_window
        sample = self._next_world_model_sample()
        predicted = self.world_model_rollout(
            sample["s"], sample["a"], sample["a_prev"], sample["xi"]
        )
        loss, last_errors = self.world_model_error(predicted, sample["s1"], sample["mask"])
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        last_mask = sample["mask"][:, -1]
        last_error = (last_errors.detach() * last_mask).sum() / last_mask.sum().clamp_min(1)
        return loss.detach() / window, last_error

    def _sample_valid_starts(
        self,
        batch_size: int,
        teleop_task: bool | None = None,
    ):
        starts, histories = [], []
        collected = 0
        valid_candidates = 0
        total_candidates = 0
        for _ in range(8):
            start, history = self.replay.sample_starts(2 * batch_size, self.history_length)
            frame = start["f"]
            clip_end = self.motion.clip_ends[self.motion.frame_clip_ids[frame]]
            valid = frame + self.horizon <= clip_end
            eligible = torch.ones_like(valid)
            if teleop_task is not None:
                eligible = start["teleop_task"].bool() == teleop_task
                valid &= eligible
            valid_candidates += int(valid.sum())
            total_candidates += int(eligible.sum())
            ids = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - collected]
            if ids.numel() == 0:
                continue
            starts.append({key: value[ids] for key, value in start.items()})
            histories.append({key: value[ids] for key, value in history.items()})
            collected += ids.numel()
            if collected >= batch_size:
                break
        if collected < batch_size:
            start, history, valid_ratio = self.replay.sample_valid_starts(
                batch_size - collected,
                self.history_length,
                self.motion.frame_clip_ids,
                self.motion.clip_ends,
                self.horizon,
                teleop_task=teleop_task,
            )
            starts.append(start)
            histories.append(history)
            self.valid_imagined_start_ratio = torch.tensor(valid_ratio, device=self.device)
        else:
            self.valid_imagined_start_ratio = torch.tensor(
                valid_candidates / max(total_candidates, 1), device=self.device
            )
        concatenate = lambda values: {  # noqa: E731
            key: torch.cat([entry[key] for entry in values]) for key in values[0]
        }
        return concatenate(starts), concatenate(histories)

    @staticmethod
    def _requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    @torch.no_grad()
    def _update_policy_ema(self) -> None:
        if self.policy_ema is None:
            return
        if self.policy_ema_updates == 0:
            self.policy_ema.load_state_dict(self.policy.state_dict())
        else:
            online = list(self.policy.parameters()) + list(self.policy.buffers())
            averaged = list(self.policy_ema.parameters()) + list(self.policy_ema.buffers())
            for target, source in zip(averaged, online, strict=True):
                if target.is_floating_point():
                    target.lerp_(source.detach(), 1.0 - self.policy_ema_decay)
                else:
                    target.copy_(source)
        self.policy_ema_updates += 1

    def inference_policy(self) -> Policy:
        if self.policy_ema is not None and self.policy_ema_updates > 0:
            return self.policy_ema
        return self.policy

    @staticmethod
    def _rotation_vee(matrix: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                matrix[..., 2, 1] - matrix[..., 1, 2],
                matrix[..., 0, 2] - matrix[..., 2, 0],
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ),
            -1,
        )

    def _gradient_coordinates(self, gradients, rotation, scale):
        parts = (
            gradients[0],
            self._rotation_vee(mat33_mul(rotation.transpose(-1, -2), gradients[1])),
            *gradients[2:],
        )
        return torch.cat([part.flatten(1) for part in parts], -1) * scale

    def _gradient_scale(self, state_sequence) -> torch.Tensor:
        p, rotation, v, w, q, qd = state_sequence
        differences = (
            p[1:] - p[:-1],
            0.5
            * self._rotation_vee(
                mat33_mul(rotation[:-1].transpose(-1, -2), rotation[1:])
            ),
            v[1:] - v[:-1],
            w[1:] - w[:-1],
            q[1:] - q[:-1],
            qd[1:] - qd[:-1],
        )
        return (
            torch.cat([value.flatten(2) for value in differences], -1)
            .flatten(0, 1)
            .std(0)
            .clamp_min(1.0e-6)
        )

    @torch.no_grad()
    def _next_recorded_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Advance replay frame ids without crossing a motion-clip boundary."""
        clip_ids = self.motion.frame_clip_ids[frames]
        return torch.minimum(frames + 1, self.motion.clip_ends[clip_ids])

    @torch.no_grad()
    def _real_returns(self, start: dict[str, torch.Tensor]):
        names = ("s1", "a", "a_prev", "xi", "f", "ep_step", "done", "r")
        steps = self.horizon
        if self.mixed_tasks:
            names += (
                "velocity_command",
                "teleop_task",
                "reference_yaw_rotation",
                "reference_translation",
            )
            # The extra row provides the task and command active at the bootstrap state.
            steps += 1
        window, complete = self.replay.forward_from(
            start["age"],
            start["env"],
            steps,
            names,
        )
        transition = {name: value[:-1] for name, value in window.items()} if self.mixed_tasks else window
        batch_size = complete.shape[0]
        rewards = transition["r"].float()
        alive = torch.cumprod(1.0 - transition["done"].float(), 0)
        survival = torch.cat(
            (torch.ones(1, batch_size, device=self.device), alive[:-1]), 0
        )
        discount = float(self.cfg["gamma"]) ** torch.arange(
            self.horizon, device=self.device, dtype=torch.float32
        )[:, None]
        final_state = unflatten_state(transition["s1"][-1])
        final_frame = self._next_recorded_frames(transition["f"][-1])
        final_goal = self.mdp.goal(final_frame, self.goal_frames)
        final_context = self.mdp.advance_context(
            transition["xi"][-1],
            transition["ep_step"][-1].long() + 1,
            int(self.cfg["recovery_window_steps"]),
        )
        normalized_final_context = self._normalize_context(final_context)
        if self.mixed_tasks:
            final_teleop_task = window["teleop_task"][-1].bool()
            final_teleop_goal = self.mdp.teleop_goal(
                final_frame,
                self.goal_frames,
                window["velocity_command"][-1],
            )
            motion_value = self.critic(
                self.critic_input(
                    final_state,
                    final_frame,
                    normalized_final_context,
                    final_goal,
                )
            )
            velocity_value = self.velocity_critic(
                self.teleop_critic_input(
                    final_state,
                    normalized_final_context,
                    final_teleop_goal,
                    transition["a"][-1],
                )
            )
            final_value = torch.where(final_teleop_task, velocity_value, motion_value)
        else:
            final_value = self.critic(
                self.critic_input(
                    final_state,
                    final_frame,
                    normalized_final_context,
                    final_goal,
                )
            )
        returns = (discount * survival * rewards).sum(0) + (
            float(self.cfg["gamma"]) ** self.horizon
        ) * alive[-1] * final_value
        return returns, complete.float(), transition["a"][0]

    def _value_gradient_targets(
        self,
        states,
        rewards,
        terminations,
        survival,
        bootstrap,
        frames,
        actions,
        previous_actions,
        contexts,
        velocity_commands=None,
        task_modes=None,
        reference_yaw_rotations=None,
        reference_translations=None,
        amp_previous_states=None,
        amp_contexts=None,
    ):
        horizon, batch_size = rewards.shape
        count = (horizon - 1) * batch_size
        with torch.no_grad():
            continuation, returns = bootstrap.detach().float(), []
            for step in range(horizon - 1, 0, -1):
                continuation = rewards[step].detach() + float(self.cfg["gamma"]) * (
                    1.0 - terminations[step].detach()
                ) * continuation
                returns.append(continuation)
            returns = torch.stack(returns[::-1])
            discount = float(self.cfg["gamma"]) ** torch.arange(
                horizon, device=self.device, dtype=torch.float32
            )[:, None]
            direct_reward_weight = (discount * survival.detach())[:-1].flatten()
            direct_termination_weight = (
                float(self.cfg["gamma"]) * discount * survival.detach()
            )[:-1].flatten() * returns.flatten()
            continuation_weight = (discount * survival.detach())[1:]

        flat_states = tuple(
            torch.stack([state[index] for state in states[:-1]]).flatten(0, 1)
            for index in range(6)
        )
        state_leaf = tuple(value.detach().requires_grad_() for value in flat_states)
        if self.mixed_tasks:
            if any(
                value is None
                for value in (
                    velocity_commands,
                    task_modes,
                    reference_yaw_rotations,
                    reference_translations,
                )
            ):
                raise RuntimeError("Mixed gradient targets require task metadata.")
            direct_rewards, direct_terminations, _ = self.reward_termination(
                state_leaf,
                torch.stack(frames[:-1]).flatten(),
                torch.stack(velocity_commands[:-1]).flatten(0, 1).detach(),
                torch.stack(task_modes[:-1]).flatten().detach(),
                torch.stack(actions[:-1]).flatten(0, 1).detach(),
                torch.stack(previous_actions[:-1]).flatten(0, 1).detach(),
                torch.stack(contexts[:-1]).flatten(0, 1).detach(),
                torch.stack(reference_yaw_rotations[:-1]).flatten(0, 1).detach(),
                torch.stack(reference_translations[:-1]).flatten(0, 1).detach(),
            )
        else:
            direct_rewards, direct_terminations, _ = self.reward_termination(
                state_leaf,
                torch.stack(frames[:-1]).flatten(),
                torch.stack(actions[:-1]).flatten(0, 1).detach(),
                torch.stack(previous_actions[:-1]).flatten(0, 1).detach(),
                torch.stack(contexts[:-1]).flatten(0, 1).detach(),
            )
        if self.recovery_amp is not None:
            if amp_previous_states is None or amp_contexts is None:
                raise RuntimeError("AMP gradient targets require incoming transition states and contexts.")
            previous = tuple(
                torch.stack([s[i] for s in amp_previous_states[:-1]]).flatten(0, 1).detach() for i in range(6)
            )
            active = (
                torch.stack(amp_contexts[:-1]).flatten(0, 1)[..., self.mdp.context_layout.recovery_active].squeeze(-1)
            )
            direct_rewards = self.recovery_amp.mix_reward(
                direct_rewards, self.recovery_amp.reward(previous, state_leaf, active), active
            )
        local_gradients = torch.autograd.grad(
            (
                direct_reward_weight * direct_rewards
                - direct_termination_weight * direct_terminations
            ).sum(),
            state_leaf,
            allow_unused=True,
        )
        targets = []
        for index in range(6):
            policy_gradients = torch.stack(
                [
                    state[index].grad
                    if state[index].grad is not None
                    else torch.zeros_like(state[index])
                    for state in states[:-1]
                ]
            ).flatten(0, 1)
            local = local_gradients[index]
            target = -batch_size * policy_gradients - (
                local if local is not None else torch.zeros_like(policy_gradients)
            )
            divisor = continuation_weight.flatten().clamp_min(1.0e-3).view(
                -1, *([1] * (target.dim() - 1))
            )
            targets.append((target / divisor).detach())
        mask = (survival.detach()[1:] > 0.5).float().flatten()
        assert targets[0].shape[0] == count
        return tuple(targets), mask

    def _gradient_consistency_loss(
        self,
        critic,
        state,
        frame,
        normalized_context,
        goal,
        previous_action,
        teleop_task,
        targets,
        mask,
        scale,
    ):
        state = tuple(value.detach().requires_grad_() for value in state)
        critic_input = (
            self.teleop_critic_input(state, normalized_context, goal, previous_action)
            if teleop_task
            else self.critic_input(state, frame, normalized_context, goal)
        )
        input_leaf = critic_input.detach().requires_grad_()
        input_gradient = torch.autograd.grad(
            critic(input_leaf).sum(), input_leaf, create_graph=True
        )[0]
        state_gradients = torch.autograd.grad(
            critic_input,
            state,
            grad_outputs=input_gradient.detach(),
            allow_unused=True,
        )
        state_gradients = tuple(
            (
                value if value is not None else torch.zeros_like(state_value)
            ).detach().requires_grad_()
            for value, state_value in zip(state_gradients, state, strict=True)
        )
        rotation = state[1].detach()
        target_coordinates = self._gradient_coordinates(targets, rotation, scale)
        gradient_coordinates = self._gradient_coordinates(state_gradients, rotation, scale)
        difference = gradient_coordinates - target_coordinates
        mask_sum = mask.sum().clamp_min(1.0)
        denominator = (mask * target_coordinates.square().sum(-1)).sum() / mask_sum
        relative_error = (
            (mask * difference.square().sum(-1)).sum()
            / mask_sum
            / denominator.clamp_min(1.0e-12)
        )
        error_gradients = torch.autograd.grad(relative_error, state_gradients)
        with torch.no_grad():
            epsilon = 0.05 / self._gradient_coordinates(
                error_gradients, rotation, 1.0 / scale
            ).norm(dim=-1).clamp_min(1.0e-12)

            def shifted(sign):
                return tuple(
                    value
                    + sign
                    * epsilon.view(-1, *([1] * (value.dim() - 1)))
                    * gradient
                    for value, gradient in zip(state, error_gradients, strict=True)
                )

            if teleop_task:
                positive_input = self.teleop_critic_input(
                    shifted(1.0), normalized_context, goal, previous_action
                )
                negative_input = self.teleop_critic_input(
                    shifted(-1.0), normalized_context, goal, previous_action
                )
            else:
                positive_input = self.critic_input(shifted(1.0), frame, normalized_context, goal)
                negative_input = self.critic_input(shifted(-1.0), frame, normalized_context, goal)
            input_direction = (positive_input - negative_input) / (2.0 * epsilon[:, None])
            cosine = (
                mask
                * (gradient_coordinates * target_coordinates).sum(-1)
                / (
                    gradient_coordinates.norm(dim=-1)
                    * target_coordinates.norm(dim=-1)
                ).clamp_min(1.0e-12)
            ).sum() / mask_sum
        surrogate = (input_gradient * input_direction).sum()
        return surrogate, relative_error.detach(), cosine.detach()

    def _critic_input(self, state, frame, normalized_context, goal):
        return torch.cat((self.mdp.critic_features(state, frame, normalized_context), goal), -1)

    def _teleop_critic_input(
        self,
        state,
        normalized_context,
        teleop_goal,
        previous_action,
    ):
        return torch.cat(
            (invariant_features(state), previous_action, normalized_context, teleop_goal),
            -1,
        )

    def _imagination_step(
        self,
        state,
        observation_history,
        frame,
        context,
        normalized_context,
        goal,
        previous_action,
    ):
        action = self.policy(observation_history, goal)
        next_state = self.world_model.step(state, action, normalized_context, previous_action)
        next_frame = frame + 1
        observation = self.mdp.add_noise(self.mdp.observation(next_state, action, context, next_frame))
        next_history = torch.cat((observation_history[:, 1:], observation[:, None]), 1)
        next_goal = self.mdp.goal(next_frame, self.goal_frames)
        return next_state, action, next_history, next_frame, next_goal

    def _motion_imagination_rollout(
        self,
        state,
        observation_history,
        frame,
        context,
        normalized_context,
        goal,
        previous_action,
        episode_step,
    ):
        """Run the fixed motion-only horizon as one compilable GPU program."""
        state_sequence = [[] for _ in range(6)]
        actions = []
        previous_actions = []
        frames = []
        goals = []
        contexts = [context]
        for _ in range(self.horizon):
            previous_actions.append(previous_action)
            state, previous_action, observation_history, frame, goal = self._imagination_step(
                state,
                observation_history,
                frame,
                context,
                normalized_context,
                goal,
                previous_action,
            )
            for index, value in enumerate(state):
                state_sequence[index].append(value)
            actions.append(previous_action)
            frames.append(frame)
            goals.append(goal)
            episode_step = episode_step + 1
            context = self.mdp.advance_context(
                context,
                episode_step,
                int(self.cfg["recovery_window_steps"]),
            )
            normalized_context = self._normalize_context(context)
            contexts.append(context)
        return (
            tuple(torch.stack(values) for values in state_sequence),
            torch.stack(actions),
            torch.stack(previous_actions),
            torch.stack(frames),
            torch.stack(goals),
            torch.stack(contexts),
        )

    def _mixed_imagination_step(
        self,
        state,
        observation_history,
        frame,
        context,
        normalized_context,
        motion_goal,
        teleop_goal,
        velocity_command,
        velocity_time_left,
        teleop_task,
        reference_yaw_rotation,
        reference_translation,
        previous_action,
    ):
        action = self.policy.forward_mixed(
            observation_history, motion_goal, teleop_goal, teleop_task
        )
        next_state = self.world_model.step(state, action, normalized_context, previous_action)
        next_frame = frame + 1
        next_velocity_command, next_velocity_time_left = self._advance_velocity_command(
            velocity_command, velocity_time_left
        )
        next_teleop_task = self._advance_task_mode(teleop_task)
        next_reference_yaw = reference_yaw_rotation
        next_reference_translation = reference_translation
        teleop_to_motion = teleop_task.bool() & ~next_teleop_task
        switched_yaw, switched_translation = self.mdp.reanchor_reference(
            next_state, next_frame
        )
        next_reference_yaw = torch.where(
            teleop_to_motion[:, None, None], switched_yaw, reference_yaw_rotation
        )
        next_reference_translation = torch.where(
            teleop_to_motion[:, None], switched_translation, reference_translation
        )
        observation = self.mdp.add_noise(
            self.mdp.observation(
                next_state,
                action,
                context,
                next_frame,
                reference_yaw_rotation=next_reference_yaw,
            )
        )
        next_history = torch.cat((observation_history[:, 1:], observation[:, None]), 1)
        next_motion_goal = self.mdp.goal(next_frame, self.goal_frames)
        next_teleop_goal = self.mdp.teleop_goal(
            next_frame, self.goal_frames, next_velocity_command
        )
        return (
            next_state,
            action,
            next_history,
            next_frame,
            next_motion_goal,
            next_teleop_goal,
            next_velocity_command,
            next_velocity_time_left,
            next_teleop_task,
            next_reference_yaw,
            next_reference_translation,
        )

    def _advance_task_mode(self, teleop_task):
        probability = float(self.cfg.get("task_switch_probability", 0.0))
        switches = torch.rand(teleop_task.shape, device=teleop_task.device) < probability
        return teleop_task.bool() ^ switches

    def _advance_velocity_command(self, command, time_left):
        next_time = time_left - float(self.cfg.get("control_dt", 0.02))
        expired = next_time <= 0.0
        ranges = command.new_tensor(self.cfg["velocity_command_ranges"])
        sampled = ranges[:, 0] + torch.rand_like(command) * (ranges[:, 1] - ranges[:, 0])
        standing = torch.rand_like(time_left) < float(
            self.cfg["velocity_standing_probability"]
        )
        sampled = torch.where(standing[:, None], torch.zeros_like(sampled), sampled)
        low, high = self.cfg["velocity_resampling_time_range"]
        sampled_time = float(low) + (float(high) - float(low)) * torch.rand_like(time_left)
        return (
            torch.where(expired[:, None], sampled, command),
            torch.where(expired, sampled_time, next_time),
        )

    def _reward_termination(self, state, frame, action, previous_action, context):
        termination, terms = self.mdp.soft_termination(
            state, frame, context, return_terms=True
        )
        return (
            self.mdp.reward(
                state,
                frame,
                action,
                previous_action,
                termination=termination,
                include_regularization=False,
            )
            + self.mdp.regularization_reward(state, action, previous_action, context, termination=termination),
            termination,
            terms,
        )

    def _mixed_reward_termination(
        self,
        state,
        frame,
        velocity_command,
        teleop_task,
        action,
        previous_action,
        context,
        reference_yaw_rotation,
        reference_translation,
    ):
        termination, terms = self.mdp.soft_termination(
            state, frame, context, return_terms=True
        )
        return (
            self.mdp.mixed_reward(
                state,
                frame,
                velocity_command,
                teleop_task,
                action,
                previous_action,
                reference_yaw_rotation,
                reference_translation,
                termination=termination,
                include_regularization=False,
            )
            + self.mdp.regularization_reward(state, action, previous_action, context, termination=termination),
            termination,
            terms,
        )

    def _velocity_warmup_update(self) -> dict[str, torch.Tensor]:
        batch_size = max(1, int(self.cfg["policy_batch_size"]) // 2)
        start, _ = self._sample_valid_starts(batch_size, teleop_task=True)
        frame = start["f"]
        motion_goal = self.mdp.goal(frame, self.goal_frames)
        with torch.no_grad():
            actual_teleop_goal = self.mdp.teleop_goal(
                frame, self.goal_frames, start["velocity_command"]
            )
            velocity_critic_input = self.teleop_critic_input(
                unflatten_state(start["s"]),
                self._normalize_context(start["xi"]),
                actual_teleop_goal,
                start["a_prev"],
            )
            self.velocity_critic.norm.update(velocity_critic_input)
        reference_command = self.mdp.reference_velocity_command(frame)
        teleop_goal = self.mdp.teleop_goal(frame, self.goal_frames, reference_command)
        with torch.no_grad():
            target_latent = self.policy.encode_motion(motion_goal)
        predicted_latent = self.policy.encode_teleop(teleop_goal)
        alignment_loss = (predicted_latent - target_latent).square().mean()

        self.teleop_optimizer.zero_grad(set_to_none=True)
        alignment_loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.policy.teleop_encoder.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.policy.teleop_encoder.parameters(), 0.1)
        if torch.isfinite(gradient_norm):
            self.teleop_optimizer.step()
        self._update_policy_ema()
        self.velocity_warmup_completed += 1

        zero = torch.zeros((), device=self.device)
        replay_recovery = zero
        if self.mdp.context_layout is not None:
            replay_recovery = start["xi"][..., self.mdp.context_layout.recovery_active].float().mean()
        return {
            "policy_loss": alignment_loss.detach(),
            "motion_policy_loss": zero,
            "teleop_policy_loss": alignment_loss.detach(),
            "alignment_loss": alignment_loss.detach(),
            "switch_action_continuity_loss": zero,
            "self_imitation_loss": zero,
            "self_imitation_positive_ratio": zero,
            "gradient_consistency_error": zero,
            "gradient_consistency_cosine": zero,
            "critic_loss": zero,
            "motion_critic_loss": zero,
            "velocity_critic_loss": zero,
            "imagined_reward": zero,
            "imagined_motion_reward": zero,
            "imagined_teleop_reward": zero,
            "policy_gradient_norm": gradient_norm.detach(),
            "imagined_survival": zero,
            "valid_imagined_start_ratio": self.valid_imagined_start_ratio.detach(),
            "replay_recovery_ratio": replay_recovery.detach(),
            "imagined_recovery_ratio": zero,
            "soft_termination_height": zero,
            "soft_termination_tilt": zero,
            "soft_termination_end_effector": zero,
        }

    def _fit_critic(
        self,
        critic,
        optimizer,
        inputs,
        targets,
        gradient_consistency=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_targets = targets.reshape(-1)
        loss = torch.zeros((), device=self.device)
        consistency_error = torch.zeros((), device=self.device)
        consistency_cosine = torch.zeros((), device=self.device)
        num_minibatches = int(self.cfg["critic_minibatches"])
        minibatch_size = flat_inputs.shape[0] // num_minibatches
        for epoch in range(int(self.cfg["critic_epochs"])):
            permutation = torch.randperm(flat_inputs.shape[0], device=self.device)
            for minibatch in range(num_minibatches):
                ids = permutation[
                    minibatch * minibatch_size : (minibatch + 1) * minibatch_size
                ]
                loss = (critic(flat_inputs[ids]) - flat_targets[ids]).square().mean()
                total_loss = loss
                if (
                    gradient_consistency is not None
                    and epoch < self.value_gradient_consistency_epochs
                ):
                    (
                        gc_state,
                        gc_frame,
                        gc_context,
                        gc_goal,
                        gc_previous_action,
                        gc_teleop_task,
                        gc_targets,
                        gc_mask,
                        gc_scale,
                    ) = gradient_consistency
                    count = min(self.value_gradient_consistency_samples, gc_mask.shape[0])
                    if count > 0:
                        gc_ids = torch.randint(0, gc_mask.shape[0], (count,), device=self.device)
                        consistency_loss, consistency_error, consistency_cosine = (
                            self._gradient_consistency_loss(
                                critic,
                                tuple(value[gc_ids] for value in gc_state),
                                gc_frame[gc_ids],
                                gc_context[gc_ids],
                                gc_goal[gc_ids],
                                gc_previous_action[gc_ids],
                                gc_teleop_task,
                                tuple(value[gc_ids] for value in gc_targets),
                                gc_mask[gc_ids],
                                gc_scale,
                            )
                        )
                        total_loss = (
                            total_loss
                            + self.value_gradient_consistency_weight * consistency_loss
                        )
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
        return loss, consistency_error, consistency_cosine

    def policy_update(self) -> dict[str, torch.Tensor]:
        if self.in_velocity_warmup:
            return self._velocity_warmup_update()

        batch_size = int(self.cfg["policy_batch_size"])
        if self.mixed_tasks:
            motion_batch = batch_size // 2
            motion_start, motion_history = self._sample_valid_starts(
                motion_batch, teleop_task=False
            )
            motion_valid_ratio = self.valid_imagined_start_ratio
            teleop_start, teleop_history = self._sample_valid_starts(
                batch_size - motion_batch, teleop_task=True
            )
            teleop_valid_ratio = self.valid_imagined_start_ratio
            concatenate = lambda values: {  # noqa: E731
                key: torch.cat((values[0][key], values[1][key])) for key in values[0]
            }
            start = concatenate((motion_start, teleop_start))
            history = concatenate((motion_history, teleop_history))
            self.valid_imagined_start_ratio = 0.5 * (
                motion_valid_ratio + teleop_valid_ratio
            )
        else:
            start, history = self._sample_valid_starts(batch_size)
        state = unflatten_state(start["s"])
        initial_state = state
        context = start["xi"]
        normalized_context = self._normalize_context(context)
        initial_normalized_context = normalized_context
        frame = start["f"]
        episode_step = start["ep_step"].long()
        previous_action = start["a_prev"]
        velocity_command = start.get("velocity_command")
        velocity_time_left = start.get("velocity_time_left")
        teleop_task = start.get("teleop_task")
        reference_yaw_rotation = start["reference_yaw_rotation"].float()
        reference_translation = start["reference_translation"].float()
        initial_teleop_task = teleop_task
        with torch.no_grad():
            observation_history = self.mdp.observation(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
                reference_yaw_rotation=history["reference_yaw_rotation"].float(),
            )
        observation_history = self.mdp.add_noise(observation_history)

        self._requires_grad(self.world_model, False)
        self._requires_grad(self.critic, False)
        if self.velocity_critic is not None:
            self._requires_grad(self.velocity_critic, False)
        initial_goal = self.mdp.goal(frame, self.goal_frames)
        goal = initial_goal
        initial_teleop_goal = None
        teleop_goal = None
        if self.mixed_tasks:
            if (
                velocity_command is None
                or velocity_time_left is None
                or teleop_task is None
                or reference_yaw_rotation is None
                or reference_translation is None
            ):
                raise RuntimeError("Mixed SuperMimic replay is missing task metadata.")
            initial_teleop_goal = self.mdp.teleop_goal(
                frame, self.goal_frames, velocity_command
            )
            teleop_goal = initial_teleop_goal
        states, actions, previous_actions, frames, goals = [], [], [], [], []
        teleop_goals = []
        velocity_commands = []
        task_modes = []
        next_task_modes = []
        reference_yaw_rotations = []
        reference_translations = []
        if self.mixed_tasks:
            contexts = [context]
            for _ in range(self.horizon):
                previous_actions.append(previous_action)
                velocity_commands.append(velocity_command)
                task_modes.append(teleop_task)
                reference_yaw_rotations.append(reference_yaw_rotation)
                reference_translations.append(reference_translation)
                (
                    state,
                    previous_action,
                    observation_history,
                    frame,
                    goal,
                    teleop_goal,
                    velocity_command,
                    velocity_time_left,
                    teleop_task,
                    reference_yaw_rotation,
                    reference_translation,
                ) = self.imagination_step(
                    state,
                    observation_history,
                    frame,
                    context,
                    normalized_context,
                    goal,
                    teleop_goal,
                    velocity_command,
                    velocity_time_left,
                    teleop_task,
                    reference_yaw_rotation,
                    reference_translation,
                    previous_action,
                )
                teleop_goals.append(teleop_goal)
                next_task_modes.append(teleop_task)
                states.append(state)
                actions.append(previous_action)
                frames.append(frame)
                goals.append(goal)
                episode_step = episode_step + 1
                context = self.mdp.advance_context(
                    context,
                    episode_step,
                    int(self.cfg["recovery_window_steps"]),
                )
                normalized_context = self._normalize_context(context)
                contexts.append(context)
        else:
            (
                state_sequence,
                action_sequence,
                previous_action_sequence,
                frame_sequence,
                goal_sequence,
                context_sequence,
            ) = self.imagination_rollout(
                state,
                observation_history,
                frame,
                context,
                normalized_context,
                goal,
                previous_action,
                episode_step,
            )
            states = [
                tuple(value[step] for value in state_sequence)
                for step in range(self.horizon)
            ]
            actions = list(action_sequence.unbind(0))
            previous_actions = list(previous_action_sequence.unbind(0))
            frames = list(frame_sequence.unbind(0))
            goals = list(goal_sequence.unbind(0))
            contexts = list(context_sequence.unbind(0))
            state = tuple(value[-1] for value in state_sequence)
            previous_action = action_sequence[-1]
            frame = frame_sequence[-1]
            goal = goal_sequence[-1]
            context = context_sequence[-1]
            normalized_context = self._normalize_context(context)

        flatten = lambda values: torch.stack(values).flatten(0, 1)  # noqa: E731
        if self.mixed_tasks:
            rollout_contexts = torch.stack(contexts)
            all_states = tuple(
                flatten([state_at_time[index] for state_at_time in states])
                for index in range(6)
            )
            rewards, terminations, termination_terms = self.reward_termination(
                all_states,
                flatten(frames),
                flatten(velocity_commands),
                flatten(task_modes),
                flatten(actions),
                flatten(previous_actions),
                flatten(contexts[1:]),
                flatten(reference_yaw_rotations),
                flatten(reference_translations),
            )
        else:
            rollout_contexts = context_sequence
            all_states = tuple(value.flatten(0, 1) for value in state_sequence)
            rewards, terminations, termination_terms = self.reward_termination(
                all_states,
                frame_sequence.flatten(0, 1),
                action_sequence.flatten(0, 1),
                previous_action_sequence.flatten(0, 1),
                context_sequence[1:].flatten(0, 1),
            )
        if self.recovery_amp is not None:
            if self.mixed_tasks:
                prior_states = [initial_state] + states[:-1]
                prior_flat = tuple(flatten([s[i] for s in prior_states]) for i in range(6))
            else:
                prior_flat = tuple(
                    torch.cat((initial_state[index][None], state_sequence[index][:-1]), 0).flatten(0, 1)
                    for index in range(6)
                )
            active = rollout_contexts[:-1].flatten(0, 1)[
                ..., self.mdp.context_layout.recovery_active
            ].squeeze(-1)
            rewards = self.recovery_amp.mix_reward(
                rewards, self.recovery_amp.reward(prior_flat, all_states, active), active
            )
        rewards = rewards.view(self.horizon, batch_size)
        terminations = terminations.view(self.horizon, batch_size)
        alive = torch.cumprod(1 - terminations, 0)
        survival = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), 0)
        discount = float(self.cfg["gamma"]) ** torch.arange(
            self.horizon, device=self.device, dtype=torch.float32
        )[:, None]
        objective = (discount * survival * rewards).sum(0)
        final_survival = alive[-1]
        detached_reward_sum = (survival * rewards).detach().sum(0)
        if not self.cfg.get("disable_critic", False):
            if self.mixed_tasks:
                motion_bootstrap = self.critic(self.critic_input(state, frame, normalized_context, goal))
                velocity_bootstrap = self.velocity_critic(
                    self.teleop_critic_input(
                        state, normalized_context, teleop_goal, previous_action
                    )
                )
                bootstrap = torch.where(
                    teleop_task.bool(), velocity_bootstrap, motion_bootstrap
                )
            else:
                bootstrap = self.critic(self.critic_input(state, frame, normalized_context, goal))
            objective = objective + (
                float(self.cfg["gamma"]) ** self.horizon
            ) * final_survival * bootstrap

        alignment_loss = torch.zeros((), device=self.device)
        switch_action_continuity_loss = torch.zeros((), device=self.device)
        if (
            self.mixed_tasks
            and self.switch_action_continuity_weight > 0.0
            and self.switch_action_continuity_joint_indices.numel() > 0
            and self.horizon > 1
        ):
            rollout_task_modes = torch.stack(task_modes).bool()
            motion_to_teleop = (~rollout_task_modes[:-1]) & rollout_task_modes[1:]
            switch_mask = motion_to_teleop.to(rewards.dtype) * survival[1:].detach()
            imagined_actions = torch.stack(actions)
            continuity_indices = self.switch_action_continuity_joint_indices
            action_delta = (
                imagined_actions[1:, :, continuity_indices]
                - imagined_actions[:-1, :, continuity_indices].detach()
            ) * self.action_scale[continuity_indices]
            switch_action_continuity_loss = (
                switch_mask * action_delta.square().sum(-1)
            ).sum() / switch_mask.sum().clamp_min(1.0)
        self_imitation_loss = torch.zeros((), device=self.device)
        self_imitation_positive_ratio = torch.zeros((), device=self.device)
        if self.self_imitation_weight > 0.0:
            recorded_returns, recorded_complete, recorded_action = self._real_returns(start)
            with torch.no_grad():
                motion_initial_value = self.critic(
                    self.critic_input(
                        initial_state,
                        start["f"],
                        initial_normalized_context,
                        initial_goal,
                    )
                )
                if self.mixed_tasks:
                    velocity_initial_value = self.velocity_critic(
                        self.teleop_critic_input(
                            initial_state,
                            initial_normalized_context,
                            initial_teleop_goal,
                            start["a_prev"],
                        )
                    )
                    initial_value = torch.where(
                        initial_teleop_task.bool(),
                        velocity_initial_value,
                        motion_initial_value,
                    )
                else:
                    initial_value = motion_initial_value
                advantage = (recorded_returns - initial_value) * recorded_complete
                self_imitation_positive_ratio = (
                    (advantage > 0.0).float().sum() / recorded_complete.sum().clamp_min(1.0)
                )
                imitation_weight = torch.clamp(
                    advantage / (advantage.abs().mean() + 1.0e-6),
                    0.0,
                    self.self_imitation_weight_cap,
                )
            self_imitation_loss = (
                imitation_weight
                * (
                    (actions[0] - recorded_action) * self.action_scale
                ).square().mean(-1)
            ).mean()
        motion_policy_loss = torch.zeros((), device=self.device)
        teleop_policy_loss = torch.zeros((), device=self.device)
        gradient_targets = None
        consistency_enabled = (
            self.value_gradient_consistency_weight > 0.0
            and not self.cfg.get("disable_critic", False)
        )
        if consistency_enabled:
            for imagined_state in states[:-1]:
                for value in imagined_state:
                    value.retain_grad()
        if self.mixed_tasks:
            motion_mask = ~initial_teleop_task.bool()
            teleop_mask = initial_teleop_task.bool()
            motion_policy_loss = -objective[motion_mask].mean()

            alignment_frames = torch.cat((start["f"][None], torch.stack(frames)), 0)
            alignment_motion_goals = torch.cat(
                (initial_goal[None], torch.stack(goals)), 0
            )
            reference_command = self.mdp.reference_velocity_command(
                alignment_frames.flatten()
            ).view(self.horizon + 1, batch_size, -1)
            alignment_teleop_goals = self.mdp.teleop_goal(
                alignment_frames.flatten(), self.goal_frames, reference_command.flatten(0, 1)
            ).view(self.horizon + 1, batch_size, -1)
            alignment_task_modes = torch.cat(
                (initial_teleop_task[None], torch.stack(next_task_modes)), 0
            ).bool()
            alignment_mask = alignment_task_modes.flatten()
            with torch.no_grad():
                target_latent = self.policy.encode_motion(
                    alignment_motion_goals.flatten(0, 1)[alignment_mask]
                )
            predicted_latent = self.policy.encode_teleop(
                alignment_teleop_goals.flatten(0, 1)[alignment_mask]
            )
            alignment_loss = (predicted_latent - target_latent).square().mean()
            teleop_policy_loss = -objective[teleop_mask].mean() + float(
                self.cfg.get("velocity_alignment_weight", 0.1)
            ) * alignment_loss + (
                self.switch_action_continuity_weight
                * switch_action_continuity_loss
            )
            joint_policy_loss = (
                -objective.mean()
                + self.self_imitation_weight * self_imitation_loss
                + self.switch_action_continuity_weight
                * switch_action_continuity_loss
            )

            motion_parameters = list(chain(self.policy.encoder.parameters(), self.policy.decoder.parameters()))
            teleop_parameters = list(self.policy.teleop_encoder.parameters())
            self.policy_optimizer.zero_grad(set_to_none=True)
            self.teleop_optimizer.zero_grad(set_to_none=True)
            (
                joint_policy_loss
                + float(self.cfg.get("velocity_alignment_weight", 0.1)) * alignment_loss
            ).backward()
            if consistency_enabled:
                gradient_targets = self._value_gradient_targets(
                    states,
                    rewards,
                    terminations,
                    survival,
                    bootstrap,
                    frames,
                    actions,
                    previous_actions,
                    contexts[1:],
                    velocity_commands=velocity_commands,
                    task_modes=task_modes,
                    reference_yaw_rotations=reference_yaw_rotations,
                    reference_translations=reference_translations,
                    amp_previous_states=prior_states if self.recovery_amp is not None else None,
                    amp_contexts=contexts[:-1] if self.recovery_amp is not None else None,
                )
            motion_gradient_norm = nn.utils.clip_grad_norm_(motion_parameters, 1.0)
            teleop_gradient_norm = nn.utils.clip_grad_norm_(teleop_parameters, 1.0)
            nn.utils.clip_grad_value_(motion_parameters, 0.1)
            nn.utils.clip_grad_value_(teleop_parameters, 0.1)
            gradient_norm = torch.maximum(motion_gradient_norm, teleop_gradient_norm)
            if torch.isfinite(motion_gradient_norm):
                self.policy_optimizer.step()
            if torch.isfinite(teleop_gradient_norm):
                self.teleop_optimizer.step()
            policy_loss = joint_policy_loss + float(
                self.cfg.get("velocity_alignment_weight", 0.1)
            ) * alignment_loss
        else:
            policy_loss = (
                -objective.mean()
                + self.self_imitation_weight * self_imitation_loss
            )
            motion_policy_loss = policy_loss
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            if consistency_enabled:
                gradient_targets = self._value_gradient_targets(
                    states,
                    rewards,
                    terminations,
                    survival,
                    bootstrap,
                    frames,
                    actions,
                    previous_actions,
                    contexts[1:],
                    amp_previous_states=prior_states if self.recovery_amp is not None else None,
                    amp_contexts=contexts[:-1] if self.recovery_amp is not None else None,
                )
            gradient_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
            if torch.isfinite(gradient_norm):
                self.policy_optimizer.step()
        self._update_policy_ema()
        self._requires_grad(self.world_model, True)
        self._requires_grad(self.critic, True)
        if self.velocity_critic is not None:
            self._requires_grad(self.velocity_critic, True)

        critic_loss = torch.zeros((), device=self.device)
        motion_critic_loss = torch.zeros((), device=self.device)
        velocity_critic_loss = torch.zeros((), device=self.device)
        gradient_consistency_error = torch.zeros((), device=self.device)
        gradient_consistency_cosine = torch.zeros((), device=self.device)
        if not self.cfg.get("disable_critic", False):
            with torch.no_grad():
                critic_state_sequence = tuple(
                    torch.cat(
                        (initial_state[index][None], torch.stack([item[index].detach() for item in states])), 0
                    )
                    for index in range(6)
                )
                critic_states = tuple(value.flatten(0, 1) for value in critic_state_sequence)
                critic_frames = torch.cat((start["f"][None], torch.stack(frames)), 0).flatten()
                if self.mixed_tasks:
                    motion_goals = torch.cat((initial_goal[None], torch.stack(goals)), 0)
                    mixed_teleop_goals = torch.cat(
                        (initial_teleop_goal[None], torch.stack(teleop_goals)), 0
                    )
                    critic_task_modes = torch.cat(
                        (initial_teleop_task[None], torch.stack(next_task_modes)), 0
                    ).bool()
                else:
                    critic_goals = torch.cat(
                        (initial_goal[None], torch.stack(goals)), 0
                    ).flatten(0, 1)
                critic_context = self._normalize_context(torch.stack(contexts)).flatten(0, 1)
                if self.mixed_tasks:
                    motion_critic_inputs = self.critic_input(
                        critic_states,
                        critic_frames,
                        critic_context,
                        motion_goals.flatten(0, 1),
                    ).view(self.horizon + 1, batch_size, -1)
                    critic_previous_actions = torch.cat(
                        (start["a_prev"][None], torch.stack(actions)), 0
                    ).flatten(0, 1)
                    velocity_critic_inputs = self.teleop_critic_input(
                        critic_states,
                        critic_context,
                        mixed_teleop_goals.flatten(0, 1),
                        critic_previous_actions,
                    ).view(self.horizon + 1, batch_size, -1)
                else:
                    critic_inputs = self.critic_input(critic_states, critic_frames, critic_context, critic_goals).view(
                        self.horizon + 1, batch_size, -1
                    )
                gradient_consistency = None
                motion_gradient_consistency = None
                velocity_gradient_consistency = None
                if gradient_targets is not None:
                    consistency_states = tuple(
                        value[1:-1].flatten(0, 1) for value in critic_state_sequence
                    )
                    consistency_frames = torch.stack(frames[:-1]).flatten()
                    consistency_context = self._normalize_context(
                        torch.stack(contexts[1:-1])
                    ).flatten(0, 1)
                    consistency_previous_actions = torch.stack(actions[:-1]).flatten(
                        0, 1
                    )
                    consistency_scale = self._gradient_scale(critic_state_sequence)
                    if self.mixed_tasks:
                        consistency_task_modes = (
                            torch.stack(next_task_modes[:-1]).flatten().bool()
                        )
                        consistency_motion_goals = torch.stack(goals[:-1]).flatten(
                            0, 1
                        )
                        consistency_teleop_goals = torch.stack(
                            teleop_goals[:-1]
                        ).flatten(0, 1)
                        motion_consistency_mask = ~consistency_task_modes
                        velocity_consistency_mask = consistency_task_modes
                        motion_gradient_consistency = (
                            tuple(
                                value[motion_consistency_mask]
                                for value in consistency_states
                            ),
                            consistency_frames[motion_consistency_mask],
                            consistency_context[motion_consistency_mask],
                            consistency_motion_goals[motion_consistency_mask],
                            consistency_previous_actions[motion_consistency_mask],
                            False,
                            tuple(
                                value[motion_consistency_mask]
                                for value in gradient_targets[0]
                            ),
                            gradient_targets[1][motion_consistency_mask],
                            consistency_scale,
                        )
                        velocity_gradient_consistency = (
                            tuple(
                                value[velocity_consistency_mask]
                                for value in consistency_states
                            ),
                            consistency_frames[velocity_consistency_mask],
                            consistency_context[velocity_consistency_mask],
                            consistency_teleop_goals[velocity_consistency_mask],
                            consistency_previous_actions[velocity_consistency_mask],
                            True,
                            tuple(
                                value[velocity_consistency_mask]
                                for value in gradient_targets[0]
                            ),
                            gradient_targets[1][velocity_consistency_mask],
                            consistency_scale,
                        )
                    else:
                        gradient_consistency = (
                            consistency_states,
                            consistency_frames,
                            consistency_context,
                            torch.stack(goals[:-1]).flatten(0, 1),
                            consistency_previous_actions,
                            False,
                            gradient_targets[0],
                            gradient_targets[1],
                            consistency_scale,
                        )
            if self.mixed_tasks:
                self.critic.norm.update(motion_critic_inputs[~critic_task_modes])
                self.velocity_critic.norm.update(velocity_critic_inputs[critic_task_modes])
            else:
                self.critic.norm.update(critic_inputs)
            with torch.no_grad():
                if self.mixed_tasks:
                    motion_values = self.critic(motion_critic_inputs)
                    velocity_values = self.velocity_critic(velocity_critic_inputs)
                    values = torch.where(critic_task_modes, velocity_values, motion_values)
                else:
                    values = self.critic(critic_inputs)
                target = values[-1]
                targets = []
                gamma, gae_lambda = float(self.cfg["gamma"]), float(self.cfg["gae_lambda"])
                for step in range(self.horizon - 1, -1, -1):
                    target = rewards[step].detach() + gamma * (1 - terminations[step].detach()) * (
                        (1 - gae_lambda) * values[step + 1] + gae_lambda * target
                    )
                    targets.append(target)
                targets = torch.stack(targets[::-1])
            if self.mixed_tasks:
                motion_value_mask = ~critic_task_modes[:-1]
                teleop_value_mask = critic_task_modes[:-1]
                (
                    motion_critic_loss,
                    motion_consistency_error,
                    motion_consistency_cosine,
                ) = self._fit_critic(
                    self.critic,
                    self.critic_optimizer,
                    motion_critic_inputs[:-1][motion_value_mask],
                    targets[motion_value_mask],
                    gradient_consistency=motion_gradient_consistency,
                )
                (
                    velocity_critic_loss,
                    velocity_consistency_error,
                    velocity_consistency_cosine,
                ) = self._fit_critic(
                    self.velocity_critic,
                    self.velocity_critic_optimizer,
                    velocity_critic_inputs[:-1][teleop_value_mask],
                    targets[teleop_value_mask],
                    gradient_consistency=velocity_gradient_consistency,
                )
                critic_loss = 0.5 * (motion_critic_loss + velocity_critic_loss)
                motion_consistency_count = (
                    motion_gradient_consistency[7].numel() if motion_gradient_consistency is not None else 0
                )
                velocity_consistency_count = (
                    velocity_gradient_consistency[7].numel() if velocity_gradient_consistency is not None else 0
                )
                consistency_count = (
                    motion_consistency_count + velocity_consistency_count
                )
                if consistency_count > 0:
                    gradient_consistency_error = (
                        motion_consistency_count * motion_consistency_error
                        + velocity_consistency_count * velocity_consistency_error
                    ) / consistency_count
                    gradient_consistency_cosine = (
                        motion_consistency_count * motion_consistency_cosine
                        + velocity_consistency_count * velocity_consistency_cosine
                    ) / consistency_count
            else:
                (
                    critic_loss,
                    gradient_consistency_error,
                    gradient_consistency_cosine,
                ) = self._fit_critic(
                    self.critic,
                    self.critic_optimizer,
                    critic_inputs[:-1],
                    targets,
                    gradient_consistency=gradient_consistency,
                )
                motion_critic_loss = critic_loss

        result = {
            "policy_loss": policy_loss.detach(),
            "motion_policy_loss": motion_policy_loss.detach(),
            "teleop_policy_loss": teleop_policy_loss.detach(),
            "alignment_loss": alignment_loss.detach(),
            "switch_action_continuity_loss": switch_action_continuity_loss.detach(),
            "self_imitation_loss": self_imitation_loss.detach(),
            "self_imitation_positive_ratio": self_imitation_positive_ratio.detach(),
            "gradient_consistency_error": gradient_consistency_error.detach(),
            "gradient_consistency_cosine": gradient_consistency_cosine.detach(),
            "critic_loss": critic_loss.detach(),
            "motion_critic_loss": motion_critic_loss.detach(),
            "velocity_critic_loss": velocity_critic_loss.detach(),
            "imagined_reward": (detached_reward_sum / self.horizon / 0.1).mean(),
            "policy_gradient_norm": gradient_norm.detach(),
            "imagined_survival": final_survival.mean().detach(),
            "valid_imagined_start_ratio": self.valid_imagined_start_ratio.detach(),
        }
        if self.mixed_tasks:
            rollout_task_modes = torch.stack(task_modes).bool()
            weighted_rewards = (survival * rewards).detach()
            motion = ~rollout_task_modes
            teleop = rollout_task_modes
            result["imagined_motion_reward"] = (
                (weighted_rewards * motion).sum() / motion.sum().clamp_min(1) / 0.1
            )
            result["imagined_teleop_reward"] = (
                (weighted_rewards * teleop).sum() / teleop.sum().clamp_min(1) / 0.1
            )
        if self.mdp.context_layout is not None:
            result["replay_recovery_ratio"] = (
                start["xi"][..., self.mdp.context_layout.recovery_active].float().mean().detach()
            )
            result["imagined_recovery_ratio"] = (
                rollout_contexts[1:][..., self.mdp.context_layout.recovery_active]
                .float()
                .mean()
                .detach()
            )
        else:
            zero = torch.zeros((), device=self.device)
            result["replay_recovery_ratio"] = zero
            result["imagined_recovery_ratio"] = zero
        result["soft_termination_height"] = termination_terms["height"].mean().detach()
        result["soft_termination_tilt"] = termination_terms["tilt"].mean().detach()
        result["soft_termination_end_effector"] = (
            termination_terms["end_effector"].mean().detach()
        )
        return result

    def state_dict(self) -> dict:
        inference_policy = self.inference_policy()
        state = {
            "training_schema": "single_decoder_recovery_amp_v2",
            "recovery_amp": None if self.recovery_amp is None else self.recovery_amp.state_dict(),
            "robot_spec": self.robot_spec.to_dict(),
            "mixed_tasks": self.mixed_tasks,
            "world_model_structured": self.structured_world_model,
            "world_model": self.world_model.state_dict(),
            "policy": inference_policy.state_dict(),
            "policy_online": self.policy.state_dict(),
            "policy_ema_updates": self.policy_ema_updates,
            "critic": self.critic.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "context_norm": None if self.context_norm is None else self.context_norm.state_dict(),
        }
        if self.mixed_tasks:
            state.update(
                {
                    "teleop_optimizer": self.teleop_optimizer.state_dict(),
                    "velocity_critic": self.velocity_critic.state_dict(),
                    "velocity_critic_optimizer": self.velocity_critic_optimizer.state_dict(),
                    "motion_initialized": self.motion_initialized,
                    "velocity_warmup_completed": self.velocity_warmup_completed,
                }
            )
        return state

    @staticmethod
    def _module_state(policy_state: dict, prefix: str) -> dict:
        return {
            name[len(prefix) :]: value
            for name, value in policy_state.items()
            if name.startswith(prefix)
        }

    def initialize_from_motion_checkpoint(self, state: dict) -> None:
        if not self.mixed_tasks:
            raise RuntimeError("Motion checkpoint initialization is only used by mixed training.")
        if state.get("training_schema") != "single_decoder_recovery_amp_v2":
            raise ValueError("Checkpoint training architecture differs. Start a new run after the recovery cleanup.")
        saved_spec = state.get("robot_spec")
        if saved_spec is not None and saved_spec != self.robot_spec.to_dict():
            raise ValueError(
                f"Motion checkpoint robot spec {saved_spec} does not match "
                f"environment {self.robot_spec.to_dict()}."
            )
        if bool(state.get("mixed_tasks", False)):
            raise ValueError("Expected a full-reference checkpoint, but received a mixed-task checkpoint.")
        saved_structured = bool(state.get("world_model_structured", False))
        if saved_structured != self.structured_world_model:
            raise ValueError(
                "Motion checkpoint world-model type does not match this run: "
                f"checkpoint structured={saved_structured}, current={self.structured_world_model}."
            )
        missing = {"world_model", "policy", "critic"} - state.keys()
        if missing:
            raise ValueError(
                "Motion initialization requires a full training checkpoint; missing "
                f"{sorted(missing)}."
            )

        if self.recovery_amp is not None:
            if state.get("recovery_amp") is None:
                raise ValueError("Checkpoint is missing AMP discriminator state.")
            self.recovery_amp.load_state_dict(state["recovery_amp"], load_optimizers=False)
        self.world_model.load_state_dict(state["world_model"])
        policy_state = state["policy"]
        for name in ("encoder", "decoder", "obs_norm", "goal_norm"):
            module = getattr(self.policy, name)
            module.load_state_dict(self._module_state(policy_state, f"{name}."))
        if self.policy_ema is not None:
            self.policy_ema.load_state_dict(self.policy.state_dict())
            self.policy_ema_updates = 1
        self.critic.load_state_dict(state["critic"])
        saved_context_norm = state.get("context_norm")
        if self.context_norm is not None:
            if saved_context_norm is None:
                raise ValueError("Motion checkpoint is missing the physics-context normalizer.")
            self.context_norm.load_state_dict(saved_context_norm)
        self.motion_initialized = True
        self.velocity_warmup_completed = 0

    def load_state_dict(self, state: dict, load_optimizers: bool = True) -> None:
        if state.get("training_schema") != "single_decoder_recovery_amp_v2":
            raise ValueError("Checkpoint training architecture differs. Start a new run after the recovery cleanup.")
        saved_spec = state.get("robot_spec")
        if saved_spec is not None and saved_spec != self.robot_spec.to_dict():
            raise ValueError(
                f"Checkpoint robot spec {saved_spec} does not match environment {self.robot_spec.to_dict()}."
            )
        saved_mixed_tasks = bool(state.get("mixed_tasks", False))
        if saved_mixed_tasks != self.mixed_tasks:
            raise ValueError(
                "Checkpoint task interface does not match the current environment: "
                f"checkpoint mixed_tasks={saved_mixed_tasks}, current={self.mixed_tasks}."
            )
        saved_structured = bool(state.get("world_model_structured", False))
        if saved_structured != self.structured_world_model:
            raise ValueError(
                "Checkpoint world-model type does not match this run: "
                f"checkpoint structured={saved_structured}, current={self.structured_world_model}. "
                "The structured model changes the regression target and must be trained from scratch."
            )
        if self.recovery_amp is not None:
            if state.get("recovery_amp") is None:
                raise ValueError("Checkpoint is missing AMP discriminator state.")
            self.recovery_amp.load_state_dict(state["recovery_amp"], load_optimizers=load_optimizers)
        self.world_model.load_state_dict(state["world_model"])
        self.policy.load_state_dict(state.get("policy_online", state["policy"]))
        if self.policy_ema is not None:
            self.policy_ema.load_state_dict(state.get("policy", self.policy.state_dict()))
            self.policy_ema_updates = int(state.get("policy_ema_updates", 1))
        self.critic.load_state_dict(state["critic"])
        if self.mixed_tasks:
            self.velocity_critic.load_state_dict(state["velocity_critic"])
            self.motion_initialized = bool(state.get("motion_initialized", True))
            self.velocity_warmup_completed = int(state.get("velocity_warmup_completed", 0))
        saved_context_norm = state.get("context_norm")
        if self.context_norm is not None:
            if saved_context_norm is None:
                raise ValueError("Checkpoint is missing the full SuperMimic context normalizer.")
            self.context_norm.load_state_dict(saved_context_norm)
        if load_optimizers:
            self.world_model_optimizer.load_state_dict(state["world_model_optimizer"])
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            if self.mixed_tasks:
                self.teleop_optimizer.load_state_dict(state["teleop_optimizer"])
                self.velocity_critic_optimizer.load_state_dict(
                    state["velocity_critic_optimizer"]
                )
