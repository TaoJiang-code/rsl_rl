"""SuperMimic Algorithm A.1 updates, adapted to an IsaacLab transition source."""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from .buffer import ReplayBuffer
from .mdp import (
    SuperMimicMDP,
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
        goal_dim = self.robot_spec.goal_dim(self.goal_frames)
        critic_dim = self.robot_spec.invariant_dim + self.robot_spec.critic_extra_dim + goal_dim

        networks.AMP_DTYPE = torch.bfloat16 if cfg.get("amp", "bf16") == "bf16" else None
        self.world_model = WorldModel(self.robot_spec).to(device)
        self.policy = Policy(
            self.robot_spec.observation_dim,
            goal_dim,
            self.robot_spec.num_joints,
            self.history_length,
        ).to(device)
        self.critic = Critic(critic_dim).to(device)
        self.context_norm = (
            RunningNorm(self.robot_spec.context_dim).to(device)
            if self.mdp.context_layout is not None
            else None
        )
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.base_learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.base_learning_rate, betas=(0.9, 0.95)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.base_learning_rate)
        self.replay = ReplayBuffer(int(cfg["replay_steps"]), num_envs, device, self.robot_spec)
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

    def learning_rate(self, iteration: int) -> float:
        decay_iterations = int(self.cfg["learning_rate_decay_iterations"])
        minimum_ratio = float(self.cfg["minimum_learning_rate_ratio"])
        if iteration >= decay_iterations:
            return self.base_learning_rate * minimum_ratio
        cosine = 0.5 * (1 + math.cos(math.pi * iteration / decay_iterations))
        return self.base_learning_rate * (minimum_ratio + (1 - minimum_ratio) * cosine)

    def set_learning_rate(self, iteration: int) -> float:
        learning_rate = self.learning_rate(iteration)
        for optimizer in (self.world_model_optimizer, self.policy_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        return learning_rate

    @torch.no_grad()
    def update_normalizers(self, transition: dict[str, torch.Tensor], observation, goal) -> None:
        self.policy.obs_norm.update(observation)
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

    def _sample_valid_starts(self, batch_size: int):
        starts, histories = [], []
        collected = 0
        valid_candidates = 0
        total_candidates = 0
        for _ in range(8):
            start, history = self.replay.sample_starts(2 * batch_size, self.history_length)
            frame = start["f"]
            clip_end = self.motion.clip_ends[self.motion.frame_clip_ids[frame]]
            valid = frame + self.horizon <= clip_end
            valid_candidates += int(valid.sum())
            total_candidates += valid.numel()
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

    def _reward_termination(self, state, frame, action, previous_action, context):
        termination, terms = self.mdp.soft_termination(
            state, frame, context, return_terms=True
        )
        return self.mdp.reward(state, frame, action, previous_action), termination, terms

    def policy_update(self) -> dict[str, torch.Tensor]:
        batch_size = int(self.cfg["policy_batch_size"])
        start, history = self._sample_valid_starts(batch_size)
        state = unflatten_state(start["s"])
        initial_state = state
        context = start["xi"]
        normalized_context = self._normalize_context(context)
        frame = start["f"]
        episode_step = start["ep_step"].long()
        previous_action = start["a_prev"]
        with torch.no_grad():
            observation_history = self.mdp.observation(
                unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"]
            )
        observation_history = self.mdp.add_noise(observation_history)

        self._requires_grad(self.world_model, False)
        self._requires_grad(self.critic, False)
        initial_goal = self.mdp.goal(frame, self.goal_frames)
        goal = initial_goal
        states, actions, previous_actions, frames, goals = [], [], [], [], []
        contexts = [context]
        for _ in range(self.horizon):
            previous_actions.append(previous_action)
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
            objective = objective + (
                float(self.cfg["gamma"]) ** self.horizon
            ) * final_survival * self.critic(self.critic_input(state, frame, normalized_context, goal))
        policy_loss = -objective.mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.policy.parameters(), 0.1)
        if torch.isfinite(gradient_norm):
            self.policy_optimizer.step()
        self._requires_grad(self.world_model, True)
        self._requires_grad(self.critic, True)

        critic_loss = torch.zeros((), device=self.device)
        if not self.cfg.get("disable_critic", False):
            with torch.no_grad():
                critic_states = tuple(
                    torch.cat(
                        (initial_state[index][None], torch.stack([item[index].detach() for item in states])), 0
                    ).flatten(0, 1)
                    for index in range(6)
                )
                critic_frames = torch.cat((start["f"][None], torch.stack(frames)), 0).flatten()
                critic_goals = torch.cat((initial_goal[None], torch.stack(goals)), 0).flatten(0, 1)
                critic_context = self._normalize_context(torch.stack(contexts)).flatten(0, 1)
                critic_inputs = self.critic_input(
                    critic_states, critic_frames, critic_context, critic_goals
                ).view(self.horizon + 1, batch_size, -1)
            self.critic.norm.update(critic_inputs)
            with torch.no_grad():
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
            flat_inputs = critic_inputs[:-1].reshape(-1, critic_inputs.shape[-1])
            flat_targets = targets.reshape(-1)
            num_minibatches = int(self.cfg["critic_minibatches"])
            minibatch_size = flat_inputs.shape[0] // num_minibatches
            for _ in range(int(self.cfg["critic_epochs"])):
                permutation = torch.randperm(flat_inputs.shape[0], device=self.device)
                for minibatch in range(num_minibatches):
                    ids = permutation[minibatch * minibatch_size : (minibatch + 1) * minibatch_size]
                    critic_loss = (self.critic(flat_inputs[ids]) - flat_targets[ids]).square().mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.critic_optimizer.step()

        result = {
            "policy_loss": policy_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "imagined_reward": (detached_reward_sum / self.horizon / 0.1).mean(),
            "policy_gradient_norm": gradient_norm.detach(),
            "imagined_survival": final_survival.mean().detach(),
            "valid_imagined_start_ratio": self.valid_imagined_start_ratio.detach(),
        }
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
        return {
            "robot_spec": self.robot_spec.to_dict(),
            "world_model": self.world_model.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "context_norm": None if self.context_norm is None else self.context_norm.state_dict(),
        }

    def load_state_dict(self, state: dict, load_optimizers: bool = True) -> None:
        saved_spec = state.get("robot_spec")
        if saved_spec is not None and saved_spec != self.robot_spec.to_dict():
            raise ValueError(
                f"Checkpoint robot spec {saved_spec} does not match environment {self.robot_spec.to_dict()}."
            )
        self.world_model.load_state_dict(state["world_model"])
        self.policy.load_state_dict(state["policy"])
        self.critic.load_state_dict(state["critic"])
        saved_context_norm = state.get("context_norm")
        if self.context_norm is not None:
            if saved_context_norm is None:
                raise ValueError("Checkpoint is missing the full SuperMimic context normalizer.")
            self.context_norm.load_state_dict(saved_context_norm)
        if load_optimizers:
            self.world_model_optimizer.load_state_dict(state["world_model_optimizer"])
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
