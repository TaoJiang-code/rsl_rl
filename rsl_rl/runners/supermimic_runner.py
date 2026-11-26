"""IsaacLab runner for the full SuperMimic model-based training loop."""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

import torch

from rsl_rl.algorithms.supermimic.algorithm import SuperMimicAlgorithm
from rsl_rl.algorithms.supermimic.recovery_metrics import RecoveryMetrics
from rsl_rl.algorithms.supermimic.kinematics import UrdfKinematics
from rsl_rl.algorithms.supermimic.mdp import (
    RewardConfig,
    SuperMimicMDP,
    SuperMimicRobotSpec,
    unflatten_state,
)


# TensorBoard groups follow the shared task logger; CSV column names remain stable.
_TENSORBOARD_TAGS = {
    "world_model_loss": "Loss/world_model_loss",
    "policy_loss": "Loss/policy_loss",
    "critic_loss": "Loss/critic_loss",
    "self_imitation_loss": "Loss/self_imitation_loss",
    "motion_policy_loss": "Loss/motion_policy_loss",
    "teleop_policy_loss": "Loss/teleop_policy_loss",
    "alignment_loss": "Loss/alignment_loss",
    "switch_action_continuity_loss": "Loss/switch_action_continuity_loss",
    "motion_critic_loss": "Loss/motion_critic_loss",
    "velocity_critic_loss": "Loss/velocity_critic_loss",
    "learning_rate": "Loss/learning_rate",
    "sim_reward": "Train/sim_reward",
    "sim_motion_reward": "Train/sim_motion_reward",
    "sim_teleop_reward": "Train/sim_teleop_reward",
    "episode_length": "Train/episode_length",
    "normalized_return": "Train/normalized_return",
    "velocity_warmup_completed": "Train/velocity_warmup_completed",
    "policy_gradient_norm": "Policy/policy_gradient_norm",
    "self_imitation_positive_ratio": "Policy/self_imitation_positive_ratio",
    "gradient_consistency_error": "Policy/gradient_consistency_error",
    "gradient_consistency_cosine": "Policy/gradient_consistency_cosine",
    "world_model_error_8": "WorldModel/world_model_error_8",
    "imagined_reward": "Imagination/imagined_reward",
    "imagined_motion_reward": "Imagination/imagined_motion_reward",
    "imagined_teleop_reward": "Imagination/imagined_teleop_reward",
    "imagined_survival": "Imagination/imagined_survival",
    "valid_imagined_start_ratio": "Imagination/valid_imagined_start_ratio",
    "real_recovery_ratio": "Recovery/real_recovery_ratio",
    "replay_recovery_ratio": "Recovery/replay_recovery_ratio",
    "imagined_recovery_ratio": "Recovery/imagined_recovery_ratio",
    "recovery_assist_anneal": "Recovery/recovery_assist_anneal",
    "termination_fraction": "Termination/termination_fraction",
    "soft_termination_height": "Termination/soft_termination_height",
    "soft_termination_tilt": "Termination/soft_termination_tilt",
    "soft_termination_end_effector": "Termination/soft_termination_end_effector",
    "tracking_error": "Tracking/tracking_error",
    "sampling_mean_tracking_error": "Sampling/sampling_mean_tracking_error",
    "recovery_amp_loss": "AMP/recovery_amp_loss",
    "recovery_amp_reward": "AMP/recovery_amp_reward",
    "iterations_per_second": "Perf/iterations_per_second",
}


class _DeviceMetricAccumulator:
    """Accumulate scalar metrics on-device and synchronize only at log time."""

    def __init__(self, names: tuple[str, ...], device: str):
        self.names = names
        self.indices = {name: index for index, name in enumerate(names)}
        self.values = torch.zeros(len(names), 2, device=device, dtype=torch.float32)

    @torch.no_grad()
    def add(self, name: str, value: torch.Tensor | float) -> None:
        index = self.indices[name]
        self.values[index, 0].add_(torch.as_tensor(value, device=self.values.device).detach())
        self.values[index, 1].add_(1.0)

    @torch.no_grad()
    def add_ratio(
        self,
        name: str,
        numerator: torch.Tensor | float,
        denominator: torch.Tensor | float,
    ) -> None:
        index = self.indices[name]
        self.values[index, 0].add_(
            torch.as_tensor(numerator, device=self.values.device).detach()
        )
        self.values[index, 1].add_(
            torch.as_tensor(denominator, device=self.values.device).detach()
        )

    def snapshot(self) -> dict[str, float]:
        host = self.values.detach().cpu()
        means = host[:, 0] / host[:, 1].clamp_min(1.0)
        return {name: float(means[index]) for index, name in enumerate(self.names)}

    def reset(self) -> None:
        self.values.zero_()


class SuperMimicRunner:
    """Collect real transitions, fit dynamics, and optimize the policy through imagined rollouts."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        if log_dir is None:
            raise ValueError("SuperMimicRunner requires a log directory.")
        self.env = env
        self.unwrapped = env.unwrapped
        self.cfg = train_cfg
        self.algorithm_cfg = train_cfg["algorithm"]
        self.mixed_tasks = bool(self.algorithm_cfg.get("mixed_motion_teleop", False))
        self.log_dir = log_dir
        self.device = device
        self.current_learning_iteration = 0
        self.sim_steps = 0
        self.wall_time_offset = 0.0
        self.git_status_repos: list[str] = []
        os.makedirs(log_dir, exist_ok=True)

        self.motion = self.unwrapped.command_manager.get_term("motion")
        self.velocity_command_name = str(
            self.algorithm_cfg.get("velocity_command_name", "velocity")
        )
        self.initial_teleop_task = None
        if self.mixed_tasks:
            if self.unwrapped.cfg.motion_env_fraction is None:
                raise ValueError(
                    "Mixed SuperMimic algorithm requires env_cfg.motion_env_fraction."
                )
            motion_fraction = float(self.unwrapped.cfg.motion_env_fraction)
            if not 0.0 < motion_fraction < 1.0:
                raise ValueError("Mixed SuperMimic requires motion_env_fraction in (0, 1).")
            motion_task = self.unwrapped.supermimic_motion_task_mask
            if motion_task is None:
                raise RuntimeError("Environment did not create the mixed-task mask.")
            self.initial_teleop_task = ~motion_task
            self.velocity_term = self.unwrapped.command_manager.get_term(
                self.velocity_command_name
            )
            velocity_cfg = self.velocity_term.cfg
            self.algorithm_cfg["velocity_command_ranges"] = (
                tuple(velocity_cfg.ranges.lin_vel_x),
                tuple(velocity_cfg.ranges.lin_vel_y),
                tuple(velocity_cfg.ranges.ang_vel_z),
            )
            self.algorithm_cfg["velocity_resampling_time_range"] = tuple(
                velocity_cfg.resampling_time_range
            )
            self.algorithm_cfg["velocity_standing_probability"] = float(
                velocity_cfg.rel_standing_envs
            )
        self.action_term = self.unwrapped.action_manager.get_term("joint_pos")
        initial_context = self.unwrapped.refresh_supermimic_context()
        context_layout = self.unwrapped.supermimic_context_layout
        actuator_delay_groups = (
            () if context_layout is None else context_layout.actuator_delay_names
        )
        self.robot_spec = SuperMimicRobotSpec(
            num_joints=len(self.motion.cfg.joint_names),
            num_tracked_bodies=len(self.motion.cfg.tracked_body_names),
            context_dim_override=initial_context.shape[-1],
            actuator_delay_groups=actuator_delay_groups,
        )
        if self.action_term.action_dim != self.robot_spec.num_joints:
            raise ValueError(
                f"SuperMimic action dimension {self.action_term.action_dim} does not match "
                f"the {self.robot_spec.num_joints}-joint robot spec."
            )
        robot = self.unwrapped.scene["robot"]
        urdf_path = Path(self.algorithm_cfg["urdf_path"])
        if not urdf_path.is_absolute():
            urdf_path = Path.cwd() / urdf_path
        kinematics = UrdfKinematics(
            urdf_path=urdf_path,
            joint_names=list(self.motion.cfg.joint_names),
            tracked_body_names=list(self.motion.cfg.tracked_body_names),
            root_body_name=self.motion.cfg.root_body_name,
            device=device,
        )
        default_joint_position = robot.data.default_joint_pos[0].to(device)
        soft_limits = robot.data.soft_joint_pos_limits[0].to(device)
        tracking_failure_cfg = self.unwrapped.cfg.terminations.tracking_failure
        end_effector_cfg = getattr(self.unwrapped.cfg.terminations, "ee_body_pos", None)
        end_effector_body_names = (
            () if end_effector_cfg is None else tuple(end_effector_cfg.params["body_names"])
        )
        termination_penalty_cfg = getattr(
            self.unwrapped.cfg.rewards, "termination_penalty", None
        )

        def reward_term(*names):
            for name in names:
                term = getattr(self.unwrapped.cfg.rewards, name, None)
                if term is not None:
                    return term
            return None

        body_position_cfg = reward_term("motion_body_position", "body_position")
        body_orientation_cfg = reward_term("motion_body_orientation", "body_orientation")
        body_linear_velocity_cfg = reward_term(
            "motion_body_linear_velocity", "body_linear_velocity"
        )
        body_angular_velocity_cfg = reward_term(
            "motion_body_angular_velocity", "body_angular_velocity"
        )
        root_position_cfg = reward_term("motion_root_position", "root_position")
        root_orientation_cfg = reward_term("motion_root_orientation", "root_orientation")
        regularization_rewards_cfg = getattr(
            self.unwrapped.cfg, "regularization_rewards", self.unwrapped.cfg.rewards
        )
        action_smoothness_cfg = getattr(
            regularization_rewards_cfg, "action_rate", None
        ) or getattr(regularization_rewards_cfg, "action_smoothness", None)
        joint_limit_cfg = getattr(regularization_rewards_cfg, "joint_limit", None)
        self_collision_cfg = getattr(regularization_rewards_cfg, "self_collision", None)
        power_cfg = getattr(regularization_rewards_cfg, "power", None)
        teleop_linear_velocity_cfg = reward_term("track_lin_vel_xy")
        teleop_angular_velocity_cfg = reward_term("track_ang_vel_z")
        teleop_body_position_cfg = reward_term("upper_body_position")
        teleop_body_orientation_cfg = reward_term("upper_body_orientation")
        teleop_root_height_cfg = reward_term("pelvis_height")
        teleop_body_names = (
            ()
            if teleop_body_position_cfg is None
            else tuple(teleop_body_position_cfg.params["body_names"])
        )
        if teleop_body_orientation_cfg is not None and tuple(
            teleop_body_orientation_cfg.params["body_names"]
        ) != teleop_body_names:
            raise ValueError(
                "Teleop position and orientation rewards must track the same body names."
            )

        def weight(term, default):
            return default if term is None else float(term.weight)

        def std(term, default):
            return default if term is None else float(term.params["std"])

        reward_cfg = RewardConfig(
            body_position_weight=weight(body_position_cfg, 0.0),
            body_position_std=std(body_position_cfg, 0.3),
            body_orientation_weight=weight(body_orientation_cfg, 0.0),
            body_orientation_std=std(body_orientation_cfg, 0.4),
            body_linear_velocity_weight=weight(body_linear_velocity_cfg, 0.0),
            body_linear_velocity_std=std(body_linear_velocity_cfg, 1.0),
            body_angular_velocity_weight=weight(body_angular_velocity_cfg, 0.0),
            body_angular_velocity_std=std(body_angular_velocity_cfg, 3.14),
            root_position_weight=weight(root_position_cfg, 0.0),
            root_position_std=std(root_position_cfg, 0.3),
            root_orientation_weight=weight(root_orientation_cfg, 0.0),
            root_orientation_std=std(root_orientation_cfg, 0.4),
            action_rate_weight=weight(action_smoothness_cfg, 0.0),
            joint_limit_weight=weight(joint_limit_cfg, 0.0),
            termination_penalty_weight=(
                0.0
                if termination_penalty_cfg is None
                else float(termination_penalty_cfg.weight)
            ),
            termination_height=float(tracking_failure_cfg.params["height_threshold"]),
            termination_tilt=float(tracking_failure_cfg.params["tilt_threshold"]),
            termination_body_height=(
                0.25
                if end_effector_cfg is None
                else float(end_effector_cfg.params["threshold"])
            ),
            teleop_linear_velocity_weight=weight(teleop_linear_velocity_cfg, 0.0),
            teleop_linear_velocity_std=std(teleop_linear_velocity_cfg, 0.5),
            teleop_angular_velocity_weight=weight(teleop_angular_velocity_cfg, 0.0),
            teleop_angular_velocity_std=std(teleop_angular_velocity_cfg, 0.5),
            teleop_body_position_weight=weight(teleop_body_position_cfg, 0.0),
            teleop_body_position_std=std(teleop_body_position_cfg, 0.3),
            teleop_body_orientation_weight=weight(teleop_body_orientation_cfg, 0.0),
            teleop_body_orientation_std=std(teleop_body_orientation_cfg, 0.4),
            teleop_root_height_weight=weight(teleop_root_height_cfg, 0.0),
            teleop_root_height_std=std(teleop_root_height_cfg, 0.3),
        )
        regularization_reward_cfg = RewardConfig(
            action_rate_weight=weight(action_smoothness_cfg, 0.0),
            joint_limit_weight=weight(joint_limit_cfg, 0.0),
            self_collision_weight=weight(self_collision_cfg, 0.0),
            self_collision_minimum_distance=(
                0.12
                if self_collision_cfg is None
                else float(self_collision_cfg.params["minimum_distance"])
            ),
            power_weight=weight(power_cfg, 0.0),
            power_scale=(500.0 if power_cfg is None else float(power_cfg.params["scale"])),
        )
        self_collision_body_pairs = (
            () if self_collision_cfg is None else tuple(self_collision_cfg.params["body_pairs"])
        )
        action_scale = torch.as_tensor(self.action_term._scale).detach().float()
        if action_scale.ndim > 1:
            action_scale = action_scale[0]
        self.mdp = SuperMimicMDP(
            kinematics,
            self.motion,
            default_joint_position,
            soft_limits,
            self.robot_spec,
            context_layout=self.unwrapped.supermimic_context_layout,
            end_effector_body_names=end_effector_body_names,
            teleop_body_names=teleop_body_names,
            cfg=reward_cfg,
            regularization_cfg=regularization_reward_cfg,
            self_collision_body_pairs=self_collision_body_pairs,
            action_scale=action_scale.flatten(),
        )
        self.algorithm_cfg["recovery_window_steps"] = max(
            1, round(self.motion.cfg.recovery_window_s / self.unwrapped.step_dt)
        )
        self.algorithm_cfg["control_dt"] = float(self.unwrapped.step_dt)
        self.algorithm_cfg["action_scale"] = action_scale.flatten().cpu().tolist()
        task_switch_mean = self.unwrapped.cfg.task_switch_mean_time_s
        self.algorithm_cfg["task_switch_probability"] = (
            0.0
            if task_switch_mean is None
            else 1.0
            - math.exp(-float(self.unwrapped.step_dt) / float(task_switch_mean))
        )
        self.algorithm = SuperMimicAlgorithm(self.mdp, env.num_envs, self.algorithm_cfg, device)
        self.recovery_metrics = RecoveryMetrics(
            env.num_envs, device, float(self.unwrapped.step_dt),
            self.algorithm_cfg["recovery_window_steps"], self.algorithm_cfg,
        )
        self.observation_history = torch.zeros(
            env.num_envs,
            int(self.algorithm_cfg["history_length"]),
            self.robot_spec.observation_dim,
            device=device,
        )
        print(f"[INFO] SuperMimic robot spec: {self.robot_spec.to_dict()}")
        print(f"[INFO] SuperMimic physics context dimension: {self.robot_spec.context_dim}")
        if self.mixed_tasks:
            print(
                "[INFO] SuperMimic initial mixed allocation: "
                f"motion={(~self.initial_teleop_task).sum().item()}, "
                f"teleop={self.initial_teleop_task.sum().item()}, "
                f"switch_mean_s={self.unwrapped.cfg.task_switch_mean_time_s}"
            )
        if actuator_delay_groups:
            print(
                "[INFO] SuperMimic delayed actuator groups: "
                f"{', '.join(actuator_delay_groups)}; world model uses previous_action."
            )
        self.writer = None
        self._csv_file = None
        self._csv_writer = None

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    @torch.no_grad()
    def _collect(self, steps: int, statistics: _DeviceMetricAccumulator) -> None:
        for _ in range(steps):
            state_flat = self.unwrapped.supermimic_state().clone()
            state = unflatten_state(state_flat)
            context = self.unwrapped.refresh_supermimic_context().clone()
            frame = self.motion.time_steps.clone()
            previous_action = self.action_term.raw_actions.clone()
            episode_step = self.unwrapped.episode_length_buf.clone()
            reference_yaw_rotation = self.motion.reference_yaw_rotation.clone()
            reference_translation = self.motion.reference_translation.clone()
            observation = self.mdp.add_noise(
                self.mdp.observation(
                    state,
                    previous_action,
                    context,
                    frame,
                    reference_yaw_rotation=reference_yaw_rotation,
                )
            )
            fresh = episode_step == 0
            shifted = torch.cat((self.observation_history[:, 1:], observation[:, None]), 1)
            repeated = observation[:, None].expand_as(self.observation_history)
            self.observation_history = torch.where(fresh[:, None, None], repeated, shifted)
            goal = self.mdp.goal(frame, int(self.algorithm_cfg["goal_frames"]))
            velocity_command = None
            velocity_time_left = None
            teleop_goal = None
            teleop_task = None
            if self.mixed_tasks:
                teleop_task = (~self.unwrapped.supermimic_motion_task_mask).clone()
                velocity_command = self.unwrapped.command_manager.get_command(
                    self.velocity_command_name
                ).clone()
                velocity_time_left = self.velocity_term.time_left.clone()
                teleop_goal = self.mdp.teleop_goal(
                    frame, int(self.algorithm_cfg["goal_frames"]), velocity_command
                )
                action = self.algorithm.policy.forward_mixed(
                    self.observation_history,
                    goal,
                    teleop_goal,
                    teleop_task,
                )
            else:
                action = self.algorithm.policy(self.observation_history, goal)
            action = torch.clamp(
                action + float(self.algorithm_cfg["action_noise"]) * torch.randn_like(action),
                -float(self.cfg["clip_actions"]),
                float(self.cfg["clip_actions"]),
            )
            _, reward, done, extras = self.env.step(action)
            transition_info = extras.get("supermimic")
            if transition_info is None:
                raise RuntimeError("SuperMimic environment did not expose its pre-reset transition state.")
            layout = self.unwrapped.supermimic_context_layout
            if layout is not None:
                context[:, layout.push_velocity] = transition_info["push_velocity"]
            terminated = transition_info["terminated"].bool()
            time_outs = transition_info["time_outs"].bool()
            transition = {
                "s": state_flat,
                "s1": transition_info["next_state"],
                "a": action,
                "a_prev": previous_action,
                "xi": context,
                "r": reward,
                "f": frame,
                "reference_yaw_rotation": reference_yaw_rotation.to(torch.float16),
                "reference_translation": reference_translation.to(torch.float16),
                "ep_step": episode_step.to(torch.int32),
                "done": terminated | time_outs,
            }
            if layout is not None:
                self.recovery_metrics.update(
                    state, unflatten_state(transition["s1"]),
                    context[:, layout.recovery_active].squeeze(-1), episode_step,
                    terminated | time_outs, terminated,
                )
            if self.mixed_tasks:
                transition["velocity_command"] = velocity_command
                transition["velocity_time_left"] = velocity_time_left
                transition["teleop_task"] = teleop_task
            if self.algorithm.recovery_amp is not None:
                active = context[:, self.unwrapped.supermimic_context_layout.recovery_active].squeeze(-1)
                style = self.algorithm.recovery_amp.reward(state, unflatten_state(transition["s1"]), active)
                active_mask = active > 0
                statistics.add_ratio(
                    "recovery_amp_reward",
                    (style * active_mask).sum(),
                    active_mask.sum(),
                )
                reward = self.algorithm.recovery_amp.mix_reward(reward, style, active)
                transition["r"] = reward
            self.algorithm.replay.add(transition)
            self.algorithm.update_normalizers(
                transition,
                observation,
                goal,
                teleop_goal=teleop_goal,
                teleop_task=teleop_task,
            )
            self.sim_steps += self.env.num_envs
            statistics.add("sim_reward", reward.mean() / 0.1)
            if self.mixed_tasks:
                motion_mask = ~teleop_task
                statistics.add_ratio(
                    "sim_motion_reward",
                    (reward * motion_mask).sum() / 0.1,
                    motion_mask.sum(),
                )
                statistics.add_ratio(
                    "sim_teleop_reward",
                    (reward * teleop_task).sum() / 0.1,
                    teleop_task.sum(),
                )
            recovery_active = (
                context[:, self.unwrapped.supermimic_context_layout.recovery_active].squeeze(-1) > 0
                if self.unwrapped.supermimic_context_layout is not None
                else torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
            )
            statistics.add("recovery_ratio", recovery_active.float().mean())
            tracking_mask = ~recovery_active
            statistics.add_ratio(
                "tracking_error",
                (transition_info["tracking_error"] * tracking_mask).sum(),
                tracking_mask.sum(),
            )
            statistics.add(
                "sampling_mean_tracking_error",
                self.motion.metrics["sampling_mean_tracking_error"].mean(),
            )
            done_mask = done.bool()
            done_count = done_mask.sum()
            statistics.add_ratio(
                "episode_length",
                ((episode_step + 1).float() * done_mask).sum(),
                done_count,
            )
            statistics.add_ratio(
                "normalized_return",
                (transition_info["normalized_return"] * done_mask).sum(),
                done_count,
            )
            statistics.add_ratio(
                "termination_fraction",
                (terminated & done_mask).float().sum(),
                done_count,
            )

    def _open_logging(self, resume: bool) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=self.log_dir, purge_step=self.current_learning_iteration if resume else None)
        csv_path = os.path.join(self.log_dir, "train_log.csv")
        exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        columns = [
            "iteration",
            "wall_h",
            "sim_steps",
            "world_model_loss",
            "world_model_error_8",
            "policy_loss",
            "critic_loss",
            "imagined_reward",
            "policy_gradient_norm",
            "self_imitation_loss",
            "self_imitation_positive_ratio",
            "gradient_consistency_error",
            "gradient_consistency_cosine",
            "imagined_survival",
            "imagined_motion_reward",
            "imagined_teleop_reward",
            "sim_reward",
            "sim_motion_reward",
            "sim_teleop_reward",
            "episode_length",
            "normalized_return",
            "real_recovery_ratio",
            "replay_recovery_ratio",
            "imagined_recovery_ratio",
            "valid_imagined_start_ratio",
            "recovery_assist_anneal",
            "termination_fraction",
            "soft_termination_height",
            "soft_termination_tilt",
            "soft_termination_end_effector",
            "tracking_error",
            "sampling_mean_tracking_error",
            "iterations_per_second",
            "learning_rate",
        ]
        if self.mixed_tasks:
            columns[5:5] = [
                "training_stage", "velocity_warmup_completed",
                "motion_policy_loss", "teleop_policy_loss",
                "alignment_loss", "switch_action_continuity_loss",
                "motion_critic_loss", "velocity_critic_loss",
            ]
        if self.algorithm.recovery_amp is not None:
            columns += ["recovery_amp_loss", "recovery_amp_reward"]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=columns)
        if not exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def _log(self, iteration: int, elapsed: float, interval_elapsed: float, stats, aggregate, learning_rate):
        stats = stats.snapshot()
        aggregate = aggregate.snapshot()
        termination_fraction = stats["termination_fraction"]
        row = {
            "iteration": iteration,
            "wall_h": elapsed / 3600,
            "sim_steps": self.sim_steps,
            "world_model_loss": aggregate["world_model_loss"],
            "world_model_error_8": aggregate["world_model_error_8"],
            "policy_loss": aggregate["policy_loss"],
            "critic_loss": aggregate["critic_loss"],
            "imagined_reward": aggregate["imagined_reward"],
            "policy_gradient_norm": aggregate["policy_gradient_norm"],
            "self_imitation_loss": aggregate["self_imitation_loss"],
            "self_imitation_positive_ratio": aggregate["self_imitation_positive_ratio"],
            "gradient_consistency_error": aggregate["gradient_consistency_error"],
            "gradient_consistency_cosine": aggregate["gradient_consistency_cosine"],
            "imagined_survival": aggregate["imagined_survival"],
            "imagined_motion_reward": aggregate["imagined_motion_reward"],
            "imagined_teleop_reward": aggregate["imagined_teleop_reward"],
            "real_recovery_ratio": stats["recovery_ratio"],
            "replay_recovery_ratio": aggregate["replay_recovery_ratio"],
            "imagined_recovery_ratio": aggregate["imagined_recovery_ratio"],
            "valid_imagined_start_ratio": aggregate["valid_imagined_start_ratio"],
            "soft_termination_height": aggregate["soft_termination_height"],
            "soft_termination_tilt": aggregate["soft_termination_tilt"],
            "soft_termination_end_effector": aggregate["soft_termination_end_effector"],
            "recovery_assist_anneal": float(self.motion.recovery_assist_anneal),
            "tracking_error": stats["tracking_error"],
            "sampling_mean_tracking_error": stats["sampling_mean_tracking_error"],
            "sim_reward": stats["sim_reward"],
            "sim_motion_reward": stats["sim_motion_reward"],
            "sim_teleop_reward": stats["sim_teleop_reward"],
            "episode_length": stats["episode_length"],
            "normalized_return": stats["normalized_return"],
            "termination_fraction": termination_fraction,
            "iterations_per_second": int(self.algorithm_cfg["log_interval"]) / interval_elapsed,
            "learning_rate": learning_rate,
        }
        if self.mixed_tasks:
            row.update(
                {
                    "training_stage": self.algorithm.training_stage,
                    "velocity_warmup_completed": self.algorithm.velocity_warmup_completed,
                    "motion_policy_loss": aggregate["motion_policy_loss"],
                    "teleop_policy_loss": aggregate["teleop_policy_loss"],
                    "alignment_loss": aggregate["alignment_loss"],
                    "switch_action_continuity_loss": aggregate["switch_action_continuity_loss"],
                    "motion_critic_loss": aggregate["motion_critic_loss"],
                    "velocity_critic_loss": aggregate["velocity_critic_loss"],
                }
            )
        if self.algorithm.recovery_amp is not None:
            row["recovery_amp_loss"] = float(
                self.algorithm.recovery_amp.last_loss.detach().cpu()
            )
            row["recovery_amp_reward"] = stats["recovery_amp_reward"]
        row = {name: row[name] for name in self._csv_writer.fieldnames}
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        for name, value in row.items():
            if name not in ("iteration", "wall_h", "sim_steps", "training_stage"):
                self.writer.add_scalar(_TENSORBOARD_TAGS[name], value, iteration)
        for tag, value in self.recovery_metrics.flush().items():
            self.writer.add_scalar(tag, value, iteration)
        if self.algorithm.recovery_amp is not None and self.algorithm.recovery_amp.updates:
            amp_scores = torch.stack(
                (
                    self.algorithm.recovery_amp.expert_score,
                    self.algorithm.recovery_amp.policy_score,
                )
            ).detach().cpu()
            self.writer.add_scalar("AMP/expert_score", float(amp_scores[0]), iteration)
            self.writer.add_scalar("AMP/policy_score", float(amp_scores[1]), iteration)
        stage_text = (
            f"{row['training_stage']} "
            f"{row['velocity_warmup_completed']}/"
            f"{int(self.algorithm_cfg.get('velocity_warmup_updates', 0))} "
            f"align {row['alignment_loss']:.4f} "
            if self.mixed_tasks
            else ""
        )
        print(
            f"it {iteration} | {row['wall_h']:.3f} h | wm {row['world_model_loss']:.4f} "
            f"e8 {row['world_model_error_8']:.4f} | {stage_text}"
            f"pi {row['policy_loss']:.3f} "
            f"v {row['critic_loss']:.4f} imag_r {row['imagined_reward']:.3f} "
            f"surv {row['imagined_survival']:.3f} | sim_r {row['sim_reward']:.3f} "
            f"ep_len {row['episode_length']:.1f} ep_ret {row['normalized_return']:.3f} "
            f"term {termination_fraction:.2f} | {row['iterations_per_second']:.2f} it/s",
            flush=True,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        del init_at_random_ep_len
        self.algorithm.world_model.train()
        self.algorithm.policy.train()
        self.algorithm.critic.train()
        if self.algorithm.velocity_critic is not None:
            self.algorithm.velocity_critic.train()
        if self.mixed_tasks and not self.algorithm.motion_initialized:
            raise RuntimeError(
                "Mixed SuperMimic training requires a pretrained full-reference checkpoint. "
                "Pass --motion_checkpoint <model.pt>, or resume an existing mixed run."
            )
        warmup_steps = max(int(self.algorithm_cfg["warmup_steps"]), self.algorithm.world_model_window)
        stats = _DeviceMetricAccumulator(
            (
                "sim_reward",
                "sim_motion_reward",
                "sim_teleop_reward",
                "episode_length",
                "normalized_return",
                "termination_fraction",
                "recovery_ratio",
                "tracking_error",
                "recovery_amp_reward",
                "sampling_mean_tracking_error",
            ),
            self.device,
        )
        print(f"[INFO] Filling SuperMimic replay with {warmup_steps} simulator steps.")
        self._collect(warmup_steps, stats)
        self._open_logging(resume=self.current_learning_iteration > 0)

        aggregate = _DeviceMetricAccumulator(
            (
                "world_model_loss",
                "world_model_error_8",
                "policy_loss",
                "self_imitation_loss",
                "self_imitation_positive_ratio",
                "gradient_consistency_error",
                "gradient_consistency_cosine",
                "motion_policy_loss",
                "teleop_policy_loss",
                "alignment_loss",
                "switch_action_continuity_loss",
                "critic_loss",
                "motion_critic_loss",
                "velocity_critic_loss",
                "imagined_reward",
                "imagined_motion_reward",
                "imagined_teleop_reward",
                "policy_gradient_norm",
                "imagined_survival",
                "replay_recovery_ratio",
                "imagined_recovery_ratio",
                "valid_imagined_start_ratio",
                "soft_termination_height",
                "soft_termination_tilt",
                "soft_termination_end_effector",
            ),
            self.device,
        )
        start_time = time.time() - self.wall_time_offset
        interval_start = time.time()
        step_accumulator = 0.0
        start_iteration = self.current_learning_iteration
        # Match the reference implementation: max_iterations is the absolute final iteration after resume.
        final_iteration = num_learning_iterations
        try:
            for iteration in range(start_iteration, final_iteration):
                learning_rate = self.algorithm.set_learning_rate(iteration)
                step_accumulator += 1.0 / float(self.algorithm_cfg["update_to_data_ratio"])
                collect_steps = int(step_accumulator)
                step_accumulator -= collect_steps
                self._collect(collect_steps, stats)
                if self.algorithm.recovery_amp is not None:
                    self.algorithm.recovery_amp.update(
                        self.algorithm.replay,
                        self.unwrapped.supermimic_context_layout.recovery_active,
                    )
                for _ in range(int(self.algorithm_cfg["world_model_batches"])):
                    loss, error = self.algorithm.world_model_update()
                self.algorithm.seal_replay_prefetch()
                aggregate.add("world_model_loss", loss)
                aggregate.add("world_model_error_8", error)
                if iteration >= int(self.algorithm_cfg["world_model_only_iterations"]):
                    result = self.algorithm.policy_update()
                    for name, value in result.items():
                        aggregate.add(name, value)
                self.current_learning_iteration = iteration + 1
                if self.current_learning_iteration % int(self.algorithm_cfg["log_interval"]) == 0:
                    now = time.time()
                    self._log(
                        self.current_learning_iteration,
                        now - start_time,
                        now - interval_start,
                        stats,
                        aggregate,
                        learning_rate,
                    )
                    interval_start = now
                    stats.reset()
                    aggregate.reset()
                if self.current_learning_iteration % int(self.cfg["save_interval"]) == 0:
                    self.wall_time_offset = time.time() - start_time
                    self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        finally:
            self.wall_time_offset = time.time() - start_time
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
            if self._csv_file is not None:
                self._csv_file.close()

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint = {
            **self.algorithm.state_dict(),
            "iteration": self.current_learning_iteration,
            "wall_time": self.wall_time_offset,
            "sim_steps": self.sim_steps,
            "motion_sampler": self.motion.sampler_state_dict(),
            "config": self.cfg,
            "infos": infos,
        }
        temporary = f"{path}.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        last_path = os.path.join(self.log_dir, "last.pt")
        if os.path.abspath(path) != os.path.abspath(last_path):
            temporary_last = f"{last_path}.tmp"
            torch.save(checkpoint, temporary_last)
            os.replace(temporary_last, last_path)

    def load(self, path: str, load_cfg=None, strict=True, map_location=None) -> dict | None:
        del load_cfg, strict
        checkpoint = torch.load(path, weights_only=False, map_location=map_location or self.device)
        self.algorithm.load_state_dict(checkpoint, load_optimizers=True)
        self.current_learning_iteration = int(checkpoint.get("iteration", 0))
        self.wall_time_offset = float(checkpoint.get("wall_time", 0.0))
        self.sim_steps = int(checkpoint.get("sim_steps", 0))
        if "motion_sampler" in checkpoint:
            self.motion.load_sampler_state_dict(checkpoint["motion_sampler"])
        print(
            f"[INFO] Restored SuperMimic iteration {self.current_learning_iteration}, "
            f"sim_steps={self.sim_steps}, including the adaptive reset sampler."
        )
        return checkpoint.get("infos")

    def initialize_from_motion_checkpoint(self, path: str, map_location=None) -> None:
        checkpoint = torch.load(
            path, weights_only=False, map_location=map_location or "cpu"
        )
        self.algorithm.initialize_from_motion_checkpoint(checkpoint)
        print(
            "[INFO] Initialized mixed SuperMimic from full-reference checkpoint: "
            f"{path} (source iteration {int(checkpoint.get('iteration', 0))})."
        )

    def get_inference_policy(self, device: str | None = None):
        policy = self.algorithm.inference_policy().eval()
        return policy.to(device or self.device)
