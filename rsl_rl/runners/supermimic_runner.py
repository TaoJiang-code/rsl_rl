"""IsaacLab runner for the full SuperMimic model-based training loop."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch

from rsl_rl.algorithms.supermimic.algorithm import SuperMimicAlgorithm
from rsl_rl.algorithms.supermimic.kinematics import UrdfKinematics
from rsl_rl.algorithms.supermimic.mdp import (
    SuperMimicMDP,
    SuperMimicRobotSpec,
    unflatten_state,
)


class SuperMimicRunner:
    """Collect real transitions, fit dynamics, and optimize the policy through imagined rollouts."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        if log_dir is None:
            raise ValueError("SuperMimicRunner requires a log directory.")
        self.env = env
        self.unwrapped = env.unwrapped
        self.cfg = train_cfg
        self.algorithm_cfg = train_cfg["algorithm"]
        self.log_dir = log_dir
        self.device = device
        self.current_learning_iteration = 0
        self.sim_steps = 0
        self.wall_time_offset = 0.0
        self.git_status_repos: list[str] = []
        os.makedirs(log_dir, exist_ok=True)

        self.motion = self.unwrapped.command_manager.get_term("motion")
        self.action_term = self.unwrapped.action_manager.get_term("joint_pos")
        initial_context = self.unwrapped.refresh_supermimic_context()
        self.robot_spec = SuperMimicRobotSpec(
            num_joints=len(self.motion.cfg.joint_names),
            num_tracked_bodies=len(self.motion.cfg.tracked_body_names),
            context_dim_override=initial_context.shape[-1],
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
        self.mdp = SuperMimicMDP(
            kinematics,
            self.motion,
            default_joint_position,
            soft_limits,
            self.robot_spec,
            context_layout=self.unwrapped.supermimic_context_layout,
        )
        self.algorithm_cfg["recovery_window_steps"] = max(
            1, int(round(float(self.motion.cfg.recovery_window_s) / self.unwrapped.step_dt))
        )
        self.algorithm = SuperMimicAlgorithm(self.mdp, env.num_envs, self.algorithm_cfg, device)
        self.observation_history = torch.zeros(
            env.num_envs,
            int(self.algorithm_cfg["history_length"]),
            self.robot_spec.observation_dim,
            device=device,
        )
        print(f"[INFO] SuperMimic robot spec: {self.robot_spec.to_dict()}")
        print(f"[INFO] SuperMimic physics context dimension: {self.robot_spec.context_dim}")
        self.writer = None
        self._csv_file = None
        self._csv_writer = None

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    @torch.no_grad()
    def _collect(self, steps: int, statistics: dict[str, list[float]]) -> None:
        for _ in range(steps):
            state_flat = self.unwrapped.supermimic_state().clone()
            state = unflatten_state(state_flat)
            context = self.unwrapped.refresh_supermimic_context().clone()
            frame = self.motion.time_steps.clone()
            previous_action = self.action_term.raw_actions.clone()
            episode_step = self.unwrapped.episode_length_buf.clone()
            observation = self.mdp.add_noise(
                self.mdp.observation(state, previous_action, context, frame)
            )
            fresh = episode_step == 0
            shifted = torch.cat((self.observation_history[:, 1:], observation[:, None]), 1)
            repeated = observation[:, None].expand_as(self.observation_history)
            self.observation_history = torch.where(fresh[:, None, None], repeated, shifted)
            goal = self.mdp.goal(frame, int(self.algorithm_cfg["goal_frames"]))
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
                "ep_step": episode_step.to(torch.int32),
                "done": terminated | time_outs,
            }
            self.algorithm.replay.add(transition)
            self.algorithm.update_normalizers(transition, observation, goal)
            self.sim_steps += self.env.num_envs
            statistics["sim_reward"].append(float(reward.mean() / 0.1))
            layout = self.unwrapped.supermimic_context_layout
            statistics["recovery_ratio"].append(
                0.0
                if layout is None
                else float(context[..., layout.recovery_active].float().mean())
            )
            if done.any():
                ids = done.bool().nonzero(as_tuple=False).squeeze(-1)
                statistics["episode_length"].append(float((episode_step[ids] + 1).float().mean()))
                statistics["termination_count"].append(float(terminated[ids].float().sum()))
                statistics["done_count"].append(float(ids.numel()))
                normalized = transition_info["normalized_return"][ids]
                statistics["normalized_return"].append(float(normalized.mean()))

    @staticmethod
    def _mean(values: list) -> float:
        return sum(float(value) for value in values) / max(len(values), 1)

    def _open_logging(self, resume: bool) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=self.log_dir, purge_step=self.current_learning_iteration if resume else None)
        csv_path = os.path.join(self.log_dir, "train_log.csv")
        exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        columns = [
            "iteration", "wall_h", "sim_steps", "world_model_loss", "world_model_error_8",
            "policy_loss", "critic_loss", "imagined_reward", "policy_gradient_norm",
            "imagined_survival", "sim_reward", "episode_length", "normalized_return",
            "real_recovery_ratio", "replay_recovery_ratio", "imagined_recovery_ratio",
            "recovery_assist_anneal", "termination_fraction", "iterations_per_second", "learning_rate",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=columns)
        if not exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def _log(self, iteration: int, elapsed: float, interval_elapsed: float, stats, aggregate, learning_rate):
        termination_fraction = sum(stats["termination_count"]) / max(sum(stats["done_count"]), 1.0)
        row = {
            "iteration": iteration,
            "wall_h": elapsed / 3600,
            "sim_steps": self.sim_steps,
            "world_model_loss": self._mean(aggregate["world_model_loss"]),
            "world_model_error_8": self._mean(aggregate["world_model_error_8"]),
            "policy_loss": self._mean(aggregate["policy_loss"]),
            "critic_loss": self._mean(aggregate["critic_loss"]),
            "imagined_reward": self._mean(aggregate["imagined_reward"]),
            "policy_gradient_norm": self._mean(aggregate["policy_gradient_norm"]),
            "imagined_survival": self._mean(aggregate["imagined_survival"]),
            "real_recovery_ratio": self._mean(stats["recovery_ratio"]),
            "replay_recovery_ratio": self._mean(aggregate["replay_recovery_ratio"]),
            "imagined_recovery_ratio": self._mean(aggregate["imagined_recovery_ratio"]),
            "recovery_assist_anneal": float(self.motion.recovery_assist_anneal),
            "sim_reward": self._mean(stats["sim_reward"]),
            "episode_length": self._mean(stats["episode_length"]),
            "normalized_return": self._mean(stats["normalized_return"]),
            "termination_fraction": termination_fraction,
            "iterations_per_second": int(self.algorithm_cfg["log_interval"]) / interval_elapsed,
            "learning_rate": learning_rate,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        for name, value in row.items():
            if name not in ("iteration", "wall_h", "sim_steps"):
                self.writer.add_scalar(f"SuperMimic/{name}", value, iteration)
        print(
            f"it {iteration} | {row['wall_h']:.3f} h | wm {row['world_model_loss']:.4f} "
            f"e8 {row['world_model_error_8']:.4f} | pi {row['policy_loss']:.3f} "
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
        warmup_steps = max(int(self.algorithm_cfg["warmup_steps"]), self.algorithm.world_model_window)
        stats = {name: [] for name in (
            "sim_reward", "episode_length", "normalized_return", "termination_count", "done_count",
            "recovery_ratio",
        )}
        print(f"[INFO] Filling SuperMimic replay with {warmup_steps} simulator steps.")
        self._collect(warmup_steps, stats)
        self._open_logging(resume=self.current_learning_iteration > 0)

        aggregate = {name: [] for name in (
            "world_model_loss", "world_model_error_8", "policy_loss", "critic_loss",
            "imagined_reward", "policy_gradient_norm", "imagined_survival",
            "replay_recovery_ratio", "imagined_recovery_ratio",
        )}
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
                for _ in range(int(self.algorithm_cfg["world_model_batches"])):
                    loss, error = self.algorithm.world_model_update()
                aggregate["world_model_loss"].append(float(loss))
                aggregate["world_model_error_8"].append(float(error))
                if iteration >= int(self.algorithm_cfg["world_model_only_iterations"]):
                    result = self.algorithm.policy_update()
                    for name, value in result.items():
                        aggregate[name].append(float(value))
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
                    stats = {name: [] for name in stats}
                    aggregate = {name: [] for name in aggregate}
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

    def get_inference_policy(self, device: str | None = None):
        self.algorithm.policy.eval()
        return self.algorithm.policy.to(device or self.device)
