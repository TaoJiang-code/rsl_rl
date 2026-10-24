"""SuperMimic-style world-model training for an RGMT actor."""

from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn as nn

from rsl_rl.algorithms.rgmt import RGMTActorModel
from rsl_rl.algorithms.supermimic import networks as supermimic_networks
from rsl_rl.algorithms.supermimic.mdp import unflatten_state
from rsl_rl.algorithms.supermimic.networks import RunningNorm
from rsl_rl.utils import resolve_callable

from .buffer import ReplayBuffer
from .mdp import SuperRGMTMDP
from .networks import Critic, WorldModel


class SuperRGMTAlgorithm:
    """Fit dynamics from IsaacLab data and optimize RGMT through imagined rollouts."""

    def __init__(
        self,
        mdp: SuperRGMTMDP,
        num_envs: int,
        initial_observation,
        obs_groups: dict[str, list[str]],
        actor_cfg: dict,
        cfg: dict,
        device: str,
    ):
        self.mdp = mdp
        self.motion = mdp.motion
        self.spec = mdp.spec
        self.cfg = cfg
        self.device = device
        self.horizon = int(cfg["imagination_horizon"])
        self.world_model_window = int(cfg["world_model_window"])
        self.base_learning_rate = float(cfg["learning_rate"])
        self.clip_actions = float(cfg["clip_actions"])
        self.dropout_start_probability = float(cfg.get("dropout_start_probability", 0.002))
        self.dropout_duration_range = tuple(cfg.get("dropout_duration_range", (5, 50)))
        self.recovery_window_steps = int(cfg.get("recovery_window_steps", 150))
        self.max_episode_length = int(cfg["max_episode_length"])

        actor_cfg = copy.deepcopy(actor_cfg)
        actor_class: type[RGMTActorModel] = resolve_callable(actor_cfg.pop("class_name"))
        self.actor = actor_class(
            initial_observation,
            obs_groups,
            "actor",
            self.spec.num_joints,
            **actor_cfg,
        ).to(device)
        self.world_model = WorldModel(self.spec).to(device)
        self.context_norm = RunningNorm(self.spec.context_dim).to(device)
        critic_dim = self.spec.critic_feature_dim + self.spec.command_dim
        self.critic = Critic(critic_dim).to(device)

        supermimic_networks.AMP_DTYPE = torch.bfloat16 if cfg.get("amp", "bf16") == "bf16" else None
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.base_learning_rate)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.base_learning_rate, betas=(0.9, 0.95)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.base_learning_rate)
        self.replay = ReplayBuffer(int(cfg["replay_steps"]), num_envs, device, self.spec)
        self.error_floor = torch.cat(
            (
                torch.full((3,), 3.0e-3),
                torch.full((3,), 5.0e-3),
                torch.full((3,), 2.0e-2),
                torch.full((3,), 5.0e-2),
                torch.full((self.spec.num_joints,), 3.0e-3),
                torch.full((self.spec.num_joints,), 5.0e-2),
            )
        ).to(device)

        self.world_model_rollout = self._world_model_rollout
        self.world_model_error = self._world_model_error
        if cfg.get("compile", True):
            compile_threads = int(cfg.get("compile_threads", 0))
            if compile_threads > 0:
                os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
                from torch._inductor import config as inductor_config

                inductor_config.compile_threads = compile_threads
            self.world_model_rollout = torch.compile(self._world_model_rollout)
            self.world_model_error = torch.compile(self._world_model_error)

    def learning_rate(self, iteration: int) -> float:
        decay_iterations = int(self.cfg["learning_rate_decay_iterations"])
        minimum_ratio = float(self.cfg["minimum_learning_rate_ratio"])
        if iteration >= decay_iterations:
            return self.base_learning_rate * minimum_ratio
        cosine = 0.5 * (1.0 + math.cos(math.pi * iteration / decay_iterations))
        return self.base_learning_rate * (minimum_ratio + (1.0 - minimum_ratio) * cosine)

    def set_learning_rate(self, iteration: int) -> float:
        learning_rate = self.learning_rate(iteration)
        for optimizer in (self.world_model_optimizer, self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        return learning_rate

    @torch.no_grad()
    def update_normalizers(self, transition: dict[str, torch.Tensor], actor_observation) -> None:
        self.context_norm.update(transition["xi"])
        state = unflatten_state(transition["s"], self.spec.num_joints)
        next_state = unflatten_state(transition["s1"], self.spec.num_joints)
        normalized_context = self.context_norm(transition["xi"])
        self.world_model.in_norm.update(
            self.world_model.inputs(state, transition["joint_target"], normalized_context)
        )
        self.world_model.out_norm.update(self.world_model.targets(state, next_state))
        self.actor.update_normalization(actor_observation)

    def _world_model_rollout(self, state, joint_target, context):
        state = unflatten_state(state[:, 0], self.spec.num_joints)
        predictions = []
        for step in range(self.world_model_window):
            state = self.world_model.step(state, joint_target[:, step], self.context_norm(context[:, step]))
            predictions.append(state)
        return tuple(torch.stack([value[index] for value in predictions], dim=1) for index in range(6))

    def _world_model_error(self, predicted, next_state, mask):
        target = unflatten_state(next_state, self.spec.num_joints)
        std = torch.maximum(self.world_model.out_norm.std(), self.error_floor)
        error = (self.world_model.state_error(predicted, target) / std).square().mean(-1)
        return (error * mask).sum() / next_state.shape[0], error[:, -1]

    def world_model_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(self.cfg["world_model_batch_size"])
        sample = self.replay.sample_windows(batch_size, self.world_model_window)
        predicted = self.world_model_rollout(sample["s"], sample["joint_target"], sample["xi"])
        loss, last_errors = self.world_model_error(predicted, sample["s1"], sample["mask"])
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        last_mask = sample["mask"][:, -1]
        last_error = (last_errors.detach() * last_mask).sum() / last_mask.sum().clamp_min(1.0)
        return loss.detach() / self.world_model_window, last_error

    def _sample_valid_starts(self, batch_size: int):
        starts, histories = [], []
        collected = 0
        attempts = 0
        valid_candidates = 0
        total_candidates = 0
        while collected < batch_size:
            attempts += 1
            if attempts > 100:
                raise RuntimeError(
                    "Could not sample enough Super-RGMT starts with room for the imagination horizon. "
                    "Reduce imagination_horizon or add longer motion clips."
                )
            start, history = self.replay.sample_starts(2 * batch_size, self.spec.history_length)
            clip_end = self.motion.time_step_totals[start["motion_id"]] - 1
            motion_valid = start["f"] + self.horizon <= clip_end
            # time_out is an administrative reset boundary, not a failure for the actor to optimize.
            # Keep every imagined transition strictly before that hard environment reset.
            episode_valid = start["ep_step"].long() + self.horizon < self.max_episode_length
            administrative_valid = ~start["time_out"].bool()
            valid = motion_valid & episode_valid & administrative_valid
            valid_candidates += int(valid.sum().item())
            total_candidates += int(valid.numel())
            ids = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - collected]
            if ids.numel() == 0:
                continue
            starts.append({key: value[ids] for key, value in start.items()})
            histories.append({key: value[ids] for key, value in history.items()})
            collected += ids.numel()

        def concatenate(values):
            return {key: torch.cat([entry[key] for entry in values]) for key in values[0]}

        valid_ratio = torch.tensor(
            valid_candidates / max(total_candidates, 1), dtype=torch.float32, device=self.device
        )
        return concatenate(starts), concatenate(histories), valid_ratio

    @staticmethod
    def _requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _apply_command_dropout(
        self,
        command: torch.Tensor,
        cached: torch.Tensor,
        remaining: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active = remaining > 0
        start = (~active) & (
            torch.rand(remaining.shape, device=remaining.device) < self.dropout_start_probability
        )
        low, high = self.dropout_duration_range
        duration = torch.randint(low, high + 1, remaining.shape, device=remaining.device)
        remaining = torch.where(start, duration, remaining)
        hold = remaining > 0
        output = torch.where(hold[:, None, None], cached, command)
        cached = torch.where(hold[:, None, None], cached, command)
        return output, cached, torch.clamp_min(remaining - 1, 0), hold

    @torch.no_grad()
    def recent_world_model_error(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate an H-step rollout on the newest real trajectories in replay."""
        sample = self.replay.sample_recent_windows(batch_size, self.world_model_window)
        predicted = self.world_model_rollout(sample["s"], sample["joint_target"], sample["xi"])
        _, last_errors = self.world_model_error(predicted, sample["s1"], sample["mask"])
        last_mask = sample["mask"][:, -1]
        last_error = (last_errors * last_mask).sum() / last_mask.sum().clamp_min(1.0)
        return last_error, last_mask.mean()

    def _critic_input(self, state, motion_id, frame, normalized_context, command):
        features = self.mdp.critic_features(state, motion_id, frame, normalized_context)
        return torch.cat((features, command.flatten(-2)), dim=-1)

    def policy_update(self) -> dict[str, torch.Tensor]:
        batch_size = int(self.cfg["policy_batch_size"])
        start, history, valid_start_ratio = self._sample_valid_starts(batch_size)
        state = unflatten_state(start["s"], self.spec.num_joints)
        initial_state = state
        context = start["xi"]
        frame = start["f"]
        motion_id = start["motion_id"]
        episode_step = start["ep_step"].long()
        previous_action = start["previous_action"]

        with torch.no_grad():
            history_state = unflatten_state(history["s"], self.spec.num_joints)
            proprio_history = self.mdp.add_proprio_noise(self.mdp.proprioception(history_state))
            action_history = history["previous_action"]
        raw_command = self.mdp.command_window(state, motion_id, frame)
        cached_command = start["dropout_cached_command"].view_as(raw_command).to(raw_command.dtype)
        dropout_remaining = start["dropout_remaining"].long()
        command_dropout_active = start["dropout_active"].bool()
        command = torch.where(command_dropout_active[:, None, None], cached_command, raw_command)
        cached_command = command

        self._requires_grad(self.world_model, False)
        self._requires_grad(self.critic, False)
        states, alignment_states, actions, previous_actions = [], [], [], []
        reward_frames, critic_frames, commands, contexts, termination_contexts = [], [], [], [], []
        dropout_active_steps = []
        initial_command = command
        try:
            for _ in range(self.horizon):
                actor_input = self.mdp.actor_input(
                    state,
                    previous_action,
                    proprio_history,
                    action_history,
                    command,
                    add_noise=True,
                )
                action = self.actor(actor_input).clamp(-self.clip_actions, self.clip_actions)
                joint_target = self.mdp.joint_target(action, context)
                normalized_context = self.context_norm(context)
                next_state = self.world_model.step(state, joint_target, normalized_context)

                previous_actions.append(previous_action)
                alignment_states.append(state)
                states.append(next_state)
                actions.append(action)
                reward_frames.append(frame)
                commands.append(command)
                contexts.append(context)
                dropout_active_steps.append(command_dropout_active)

                frame = frame + 1
                episode_step = episode_step + 1
                context = self.mdp.advance_context(context, episode_step, self.recovery_window_steps)
                termination_contexts.append(context)
                next_proprio = self.mdp.add_proprio_noise(self.mdp.proprioception(next_state))
                proprio_history = torch.cat((proprio_history[:, 1:], next_proprio[:, None]), dim=1)
                action_history = torch.cat((action_history[:, 1:], action[:, None]), dim=1)
                next_command = self.mdp.command_window(next_state, motion_id, frame)
                command, cached_command, dropout_remaining, command_dropout_active = self._apply_command_dropout(
                    next_command, cached_command, dropout_remaining
                )
                state = next_state
                previous_action = action
                critic_frames.append(frame)

            flatten = lambda values: torch.stack(values).flatten(0, 1)  # noqa: E731
            all_states = tuple(
                flatten([state_at_time[index] for state_at_time in states]) for index in range(6)
            )
            all_alignment_states = tuple(
                flatten([state_at_time[index] for state_at_time in alignment_states]) for index in range(6)
            )
            rewards = self.mdp.reward(
                all_states,
                all_alignment_states,
                motion_id[None].expand(self.horizon, -1).flatten(),
                flatten(reward_frames),
                flatten(actions),
                flatten(previous_actions),
            ).view(self.horizon, batch_size)
            terminations, termination_terms = self.mdp.soft_termination(
                all_states,
                all_alignment_states,
                motion_id[None].expand(self.horizon, -1).flatten(),
                flatten(reward_frames),
                flatten(termination_contexts),
                return_terms=True,
            )
            terminations = terminations.view(self.horizon, batch_size)
            alive = torch.cumprod(1.0 - terminations, dim=0)
            survival = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), dim=0)
            discount = float(self.cfg["gamma"]) ** torch.arange(
                self.horizon, device=self.device, dtype=torch.float32
            )[:, None]
            objective = (discount * survival * rewards).sum(0)
            final_survival = alive[-1]
            if not self.cfg.get("disable_critic", False):
                final_input = self._critic_input(
                    state,
                    motion_id,
                    frame,
                    self.context_norm(context),
                    command,
                )
                objective = objective + (
                    float(self.cfg["gamma"]) ** self.horizon
                ) * final_survival * self.critic(final_input)
            actor_loss = -objective.mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            if torch.isfinite(gradient_norm):
                self.actor_optimizer.step()
        finally:
            self._requires_grad(self.world_model, True)
            self._requires_grad(self.critic, True)

        critic_loss = torch.zeros((), device=self.device)
        if not self.cfg.get("disable_critic", False):
            with torch.no_grad():
                critic_states = tuple(
                    torch.cat(
                        (initial_state[index][None], torch.stack([item[index].detach() for item in states])),
                        dim=0,
                    ).flatten(0, 1)
                    for index in range(6)
                )
                critic_frame_tensor = torch.cat((start["f"][None], torch.stack(critic_frames)), dim=0)
                critic_context = torch.cat(
                    (start["xi"][None], torch.stack(contexts[1:] + [context])), dim=0
                )
                critic_commands = torch.cat((initial_command[None], torch.stack(commands[1:] + [command])), dim=0)
                critic_motion_ids = motion_id[None].expand(self.horizon + 1, -1)
                critic_inputs = self._critic_input(
                    critic_states,
                    critic_motion_ids.flatten(),
                    critic_frame_tensor.flatten(),
                    self.context_norm(critic_context.flatten(0, 1)),
                    critic_commands.flatten(0, 1),
                ).view(self.horizon + 1, batch_size, -1)
            self.critic.norm.update(critic_inputs)
            with torch.no_grad():
                values = self.critic(critic_inputs)
                target = values[-1]
                targets = []
                gamma = float(self.cfg["gamma"])
                gae_lambda = float(self.cfg["gae_lambda"])
                for step in range(self.horizon - 1, -1, -1):
                    target = rewards[step].detach() + gamma * (1.0 - terminations[step].detach()) * (
                        (1.0 - gae_lambda) * values[step + 1] + gae_lambda * target
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

        return {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "imagined_reward": (survival * rewards).detach().sum(0).div(self.horizon * 0.1).mean(),
            "actor_gradient_norm": gradient_norm.detach(),
            "imagined_survival": final_survival.mean().detach(),
            "valid_start_ratio": valid_start_ratio.detach(),
            "replay_recovery_ratio": start["xi"][..., self.mdp.context_layout.recovery_active]
            .float()
            .mean()
            .detach(),
            "imagined_recovery_ratio": flatten(contexts)[
                ..., self.mdp.context_layout.recovery_active
            ]
            .float()
            .mean()
            .detach(),
            "replay_dropout_ratio": start["dropout_active"].float().mean().detach(),
            "imagined_dropout_ratio": torch.stack(dropout_active_steps).float().mean().detach(),
            "soft_termination_height": termination_terms["height"].mean().detach(),
            "soft_termination_tilt": termination_terms["tilt"].mean().detach(),
            "soft_termination_end_effector": termination_terms["end_effector"].mean().detach(),
        }

    def get_policy(self) -> nn.Module:
        return self.actor

    def eval_mode(self) -> None:
        self.actor.eval()

    def state_dict(self) -> dict:
        return {
            "super_rgmt_spec": self.spec.to_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "world_model_state_dict": self.world_model.state_dict(),
            "context_norm_state_dict": self.context_norm.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "world_model_optimizer_state_dict": self.world_model_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict, load_optimizers: bool = True) -> None:
        saved_spec = state.get("super_rgmt_spec")
        if saved_spec is None:
            self.actor.load_state_dict(state["actor_state_dict"], strict=True)
            print(
                "[WARN] Initialized only the actor weights from a PPO RGMT checkpoint. "
                "Super-RGMT uses default-centered actions instead of reference-residual actions; "
                "the world model and critic therefore start fresh."
            )
            return
        if saved_spec != self.spec.to_dict():
            raise ValueError(f"Checkpoint spec {saved_spec} does not match environment {self.spec.to_dict()}.")
        self.actor.load_state_dict(state["actor_state_dict"])
        self.world_model.load_state_dict(state["world_model_state_dict"])
        self.context_norm.load_state_dict(state["context_norm_state_dict"])
        self.critic.load_state_dict(state["critic_state_dict"])
        if load_optimizers:
            self.world_model_optimizer.load_state_dict(state["world_model_optimizer_state_dict"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
