"""SuperMimic Algorithm A.1 updates, adapted to an IsaacLab transition source."""

from __future__ import annotations

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
        goal_dim = self.robot_spec.goal_dim(self.goal_frames)
        teleop_goal_dim = (
            self.robot_spec.teleop_goal_dim(self.goal_frames) if self.mixed_tasks else None
        )
        critic_dim = self.robot_spec.invariant_dim + self.robot_spec.critic_extra_dim + goal_dim

        networks.AMP_DTYPE = torch.bfloat16 if cfg.get("amp", "bf16") == "bf16" else None
        self.world_model = WorldModel(self.robot_spec).to(device)
        self.policy = Policy(
            self.robot_spec.observation_dim,
            goal_dim,
            self.robot_spec.num_joints,
            self.history_length,
            teleop_goal_dim=teleop_goal_dim,
        ).to(device)
        self.critic = Critic(critic_dim).to(device)
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
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.base_learning_rate)
        if self.mixed_tasks:
            self.policy_optimizer = torch.optim.Adam(
                chain(self.policy.encoder.parameters(), self.policy.decoder.parameters()),
                lr=self.base_learning_rate,
                betas=(0.9, 0.95),
            )
            self.teleop_optimizer = torch.optim.Adam(
                self.policy.teleop_encoder.parameters(),
                lr=self.base_learning_rate,
                betas=(0.9, 0.95),
            )
        else:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self.base_learning_rate, betas=(0.9, 0.95)
            )
            self.teleop_optimizer = None
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.base_learning_rate)
        self.velocity_critic_optimizer = (
            None
            if self.velocity_critic is None
            else torch.optim.Adam(self.velocity_critic.parameters(), lr=self.base_learning_rate)
        )
        self.replay = ReplayBuffer(
            int(cfg["replay_steps"]),
            num_envs,
            device,
            self.robot_spec,
            mixed_tasks=self.mixed_tasks,
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
        self.reward_termination = self._reward_termination
        self.critic_input = self._critic_input
        self.teleop_critic_input = self._teleop_critic_input
        if cfg.get("compile", True):
            compile_threads = int(cfg.get("compile_threads", 0))
            if compile_threads > 0:
                os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
                from torch._inductor import config as inductor_config

                inductor_config.compile_threads = compile_threads
                print(f"[INFO] TorchInductor compile threads: {compile_threads}")
            self.world_model_rollout = torch.compile(self._world_model_rollout)
            self.world_model_error = torch.compile(self._world_model_error)
            self.imagination_step = torch.compile(self._imagination_step)
            self.reward_termination = torch.compile(self._reward_termination)
            self.critic_input = torch.compile(self._critic_input)
            if self.mixed_tasks:
                self.teleop_critic_input = torch.compile(self._teleop_critic_input)
        if self.mixed_tasks:
            self.imagination_step = self._mixed_imagination_step
            self.reward_termination = self._mixed_reward_termination
            if cfg.get("compile", True):
                self.imagination_step = torch.compile(self._mixed_imagination_step)
                self.reward_termination = torch.compile(self._mixed_reward_termination)
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
            if motion.any() and not self.in_velocity_warmup:
                self.policy.goal_norm.update(goal[motion])
            if teleop_task.any():
                self.policy.teleop_goal_norm.update(teleop_goal[teleop_task.bool()])
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
        std = torch.maximum(self.world_model.out_norm.std(), self.error_floor)
        error = (self.world_model.state_error(predicted, target) / std).square().mean(-1)
        return (error * mask).sum() / next_state.shape[0], error[:, -1]

    def world_model_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(self.cfg["world_model_batch_size"])
        window = self.world_model_window
        sample = self.replay.sample_windows(batch_size, window)
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
        previous_action,
    ):
        action = self.policy.forward_mixed(
            observation_history, motion_goal, teleop_goal, teleop_task
        )
        next_state = self.world_model.step(state, action, normalized_context, previous_action)
        next_frame = frame + 1
        observation = self.mdp.add_noise(
            self.mdp.observation(next_state, action, context, next_frame)
        )
        next_history = torch.cat((observation_history[:, 1:], observation[:, None]), 1)
        next_velocity_command, next_velocity_time_left = self._advance_velocity_command(
            velocity_command, velocity_time_left
        )
        next_motion_goal = self.mdp.goal(next_frame, self.goal_frames)
        next_teleop_goal = self.mdp.teleop_goal(
            next_frame, self.goal_frames, next_velocity_command
        )
        next_teleop_task = self._advance_task_mode(teleop_task)
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
            ),
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
                termination=termination,
            ),
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
        self.velocity_warmup_completed += 1

        zero = torch.zeros((), device=self.device)
        replay_recovery = zero
        if self.mdp.context_layout is not None:
            replay_recovery = start["xi"][
                ..., self.mdp.context_layout.recovery_active
            ].float().mean()
        return {
            "policy_loss": alignment_loss.detach(),
            "motion_policy_loss": zero,
            "teleop_policy_loss": alignment_loss.detach(),
            "alignment_loss": alignment_loss.detach(),
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

    def _fit_critic(self, critic, optimizer, inputs, targets) -> torch.Tensor:
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_targets = targets.reshape(-1)
        loss = torch.zeros((), device=self.device)
        num_minibatches = int(self.cfg["critic_minibatches"])
        minibatch_size = flat_inputs.shape[0] // num_minibatches
        for _ in range(int(self.cfg["critic_epochs"])):
            permutation = torch.randperm(flat_inputs.shape[0], device=self.device)
            for minibatch in range(num_minibatches):
                ids = permutation[
                    minibatch * minibatch_size : (minibatch + 1) * minibatch_size
                ]
                loss = (critic(flat_inputs[ids]) - flat_targets[ids]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
        return loss

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
        frame = start["f"]
        episode_step = start["ep_step"].long()
        previous_action = start["a_prev"]
        velocity_command = start.get("velocity_command")
        velocity_time_left = start.get("velocity_time_left")
        teleop_task = start.get("teleop_task")
        initial_teleop_task = teleop_task
        with torch.no_grad():
            observation_history = self.mdp.observation(
                unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"]
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
            if velocity_command is None or velocity_time_left is None or teleop_task is None:
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
        contexts = [context]
        for _ in range(self.horizon):
            previous_actions.append(previous_action)
            if self.mixed_tasks:
                velocity_commands.append(velocity_command)
                task_modes.append(teleop_task)
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
                    previous_action,
                )
                teleop_goals.append(teleop_goal)
                next_task_modes.append(teleop_task)
            else:
                state, previous_action, observation_history, frame, goal = self.imagination_step(
                    state,
                    observation_history,
                    frame,
                    context,
                    normalized_context,
                    goal,
                    previous_action,
                )
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

        flatten = lambda values: torch.stack(values).flatten(0, 1)  # noqa: E731
        all_states = tuple(flatten([state_at_time[index] for state_at_time in states]) for index in range(6))
        if self.mixed_tasks:
            rewards, terminations, termination_terms = self.reward_termination(
                all_states,
                flatten(frames),
                flatten(velocity_commands),
                flatten(task_modes),
                flatten(actions),
                flatten(previous_actions),
                flatten(contexts[1:]),
            )
        else:
            rewards, terminations, termination_terms = self.reward_termination(
                all_states,
                flatten(frames),
                flatten(actions),
                flatten(previous_actions),
                flatten(contexts[1:]),
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
                motion_bootstrap = self.critic(
                    self.critic_input(state, frame, normalized_context, goal)
                )
                velocity_bootstrap = self.velocity_critic(
                    self.teleop_critic_input(
                        state, normalized_context, teleop_goal, previous_action
                    )
                )
                bootstrap = torch.where(
                    teleop_task.bool(), velocity_bootstrap, motion_bootstrap
                )
            else:
                bootstrap = self.critic(
                    self.critic_input(state, frame, normalized_context, goal)
                )
            objective = objective + (
                float(self.cfg["gamma"]) ** self.horizon
            ) * final_survival * bootstrap

        alignment_loss = torch.zeros((), device=self.device)
        motion_policy_loss = torch.zeros((), device=self.device)
        teleop_policy_loss = torch.zeros((), device=self.device)
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
            ) * alignment_loss
            joint_policy_loss = -objective.mean()

            motion_parameters = list(
                chain(self.policy.encoder.parameters(), self.policy.decoder.parameters())
            )
            teleop_parameters = list(self.policy.teleop_encoder.parameters())
            self.policy_optimizer.zero_grad(set_to_none=True)
            self.teleop_optimizer.zero_grad(set_to_none=True)
            motion_gradients = torch.autograd.grad(
                joint_policy_loss, motion_parameters, retain_graph=True
            )
            teleop_gradients = torch.autograd.grad(
                joint_policy_loss
                + float(self.cfg.get("velocity_alignment_weight", 0.1)) * alignment_loss,
                teleop_parameters,
            )
            for parameter, gradient in zip(motion_parameters, motion_gradients):
                parameter.grad = gradient
            for parameter, gradient in zip(teleop_parameters, teleop_gradients):
                parameter.grad = gradient
            motion_gradient_norm = nn.utils.clip_grad_norm_(motion_parameters, 1.0)
            teleop_gradient_norm = nn.utils.clip_grad_norm_(teleop_parameters, 1.0)
            nn.utils.clip_grad_value_(motion_parameters, 0.1)
            nn.utils.clip_grad_value_(teleop_parameters, 0.1)
            gradient_norm = torch.maximum(motion_gradient_norm, teleop_gradient_norm)
            if torch.isfinite(motion_gradient_norm):
                self.policy_optimizer.step()
            if torch.isfinite(teleop_gradient_norm):
                self.teleop_optimizer.step()
            policy_loss = joint_policy_loss + 0.5 * float(
                self.cfg.get("velocity_alignment_weight", 0.1)
            ) * alignment_loss
        else:
            policy_loss = -objective.mean()
            motion_policy_loss = policy_loss
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
            if torch.isfinite(gradient_norm):
                self.policy_optimizer.step()
        self._requires_grad(self.world_model, True)
        self._requires_grad(self.critic, True)
        if self.velocity_critic is not None:
            self._requires_grad(self.velocity_critic, True)

        critic_loss = torch.zeros((), device=self.device)
        motion_critic_loss = torch.zeros((), device=self.device)
        velocity_critic_loss = torch.zeros((), device=self.device)
        if not self.cfg.get("disable_critic", False):
            with torch.no_grad():
                critic_states = tuple(
                    torch.cat(
                        (initial_state[index][None], torch.stack([item[index].detach() for item in states])), 0
                    ).flatten(0, 1)
                    for index in range(6)
                )
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
                    critic_inputs = self.critic_input(
                        critic_states, critic_frames, critic_context, critic_goals
                    ).view(self.horizon + 1, batch_size, -1)
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
                motion_critic_loss = self._fit_critic(
                    self.critic,
                    self.critic_optimizer,
                    motion_critic_inputs[:-1][motion_value_mask],
                    targets[motion_value_mask],
                )
                velocity_critic_loss = self._fit_critic(
                    self.velocity_critic,
                    self.velocity_critic_optimizer,
                    velocity_critic_inputs[:-1][teleop_value_mask],
                    targets[teleop_value_mask],
                )
                critic_loss = 0.5 * (motion_critic_loss + velocity_critic_loss)
            else:
                critic_loss = self._fit_critic(
                    self.critic, self.critic_optimizer, critic_inputs[:-1], targets
                )
                motion_critic_loss = critic_loss

        result = {
            "policy_loss": policy_loss.detach(),
            "motion_policy_loss": motion_policy_loss.detach(),
            "teleop_policy_loss": teleop_policy_loss.detach(),
            "alignment_loss": alignment_loss.detach(),
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
                weighted_rewards[motion].mean() / 0.1
                if motion.any()
                else torch.zeros((), device=self.device)
            )
            result["imagined_teleop_reward"] = (
                weighted_rewards[teleop].mean() / 0.1
                if teleop.any()
                else torch.zeros((), device=self.device)
            )
        if self.mdp.context_layout is not None:
            result["replay_recovery_ratio"] = start["xi"][
                ..., self.mdp.context_layout.recovery_active
            ].float().mean().detach()
            result["imagined_recovery_ratio"] = torch.stack(contexts[1:])[
                ..., self.mdp.context_layout.recovery_active
            ].float().mean().detach()
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
        state = {
            "robot_spec": self.robot_spec.to_dict(),
            "mixed_tasks": self.mixed_tasks,
            "world_model": self.world_model.state_dict(),
            "policy": self.policy.state_dict(),
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
        saved_spec = state.get("robot_spec")
        if saved_spec is not None and saved_spec != self.robot_spec.to_dict():
            raise ValueError(
                f"Motion checkpoint robot spec {saved_spec} does not match "
                f"environment {self.robot_spec.to_dict()}."
            )
        if bool(state.get("mixed_tasks", False)):
            raise ValueError("Expected a full-reference checkpoint, but received a mixed-task checkpoint.")
        missing = {"world_model", "policy", "critic"} - state.keys()
        if missing:
            raise ValueError(
                "Motion initialization requires a full training checkpoint; missing "
                f"{sorted(missing)}."
            )

        self.world_model.load_state_dict(state["world_model"])
        policy_state = state["policy"]
        for name in ("encoder", "decoder", "obs_norm", "goal_norm"):
            module = getattr(self.policy, name)
            module.load_state_dict(self._module_state(policy_state, f"{name}."))
        self.critic.load_state_dict(state["critic"])
        saved_context_norm = state.get("context_norm")
        if self.context_norm is not None:
            if saved_context_norm is None:
                raise ValueError("Motion checkpoint is missing the physics-context normalizer.")
            self.context_norm.load_state_dict(saved_context_norm)
        self.motion_initialized = True
        self.velocity_warmup_completed = 0

    def load_state_dict(self, state: dict, load_optimizers: bool = True) -> None:
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
        self.world_model.load_state_dict(state["world_model"])
        self.policy.load_state_dict(state["policy"])
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
