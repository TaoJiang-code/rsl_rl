"""IsaacLab runner for Super-RGMT world-model training."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch

from rsl_rl.algorithms.super_rgmt import RewardConfig, SuperRGMTAlgorithm, SuperRGMTMDP, SuperRGMTRobotSpec
from rsl_rl.algorithms.supermimic.kinematics import UrdfKinematics


class SuperRGMTRunner:
    """Collect RGMT transitions, fit dynamics, and train the RGMT actor through BPTT."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.env = env
        self.unwrapped = env.unwrapped
        self.cfg = train_cfg
        self.algorithm_cfg = dict(train_cfg["algorithm"])
        self.log_dir = log_dir
        self.device = device
        self.current_learning_iteration = 0
        self.sim_steps = 0
        self.wall_time_offset = 0.0
        self.git_status_repos: list[str] = []
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

        initial_observation = self._get_observations().to(device)
        self.observation = initial_observation
        self.motion = self.unwrapped.command_manager.get_term("motion")
        self.action_term = self.unwrapped.action_manager.get_term("joint_pos")
        robot = self.unwrapped.scene["robot"]
        context = self.unwrapped.refresh_super_rgmt_context()
        actor_cfg = train_cfg["actor"]
        history_length = int(actor_cfg["history_length"])
        command_window_size = int(actor_cfg["command_window_size"])
        command_step_dim = int(actor_cfg["command_step_dim"])
        self.spec = SuperRGMTRobotSpec(
            num_joints=self.action_term.action_dim,
            num_tracked_bodies=len(self.motion.cfg.body_names),
            context_dim=context.shape[-1],
            history_length=history_length,
            command_window_size=command_window_size,
            command_step_dim=command_step_dim,
        )

        urdf_path = Path(self.algorithm_cfg["urdf_path"])
        if not urdf_path.is_absolute():
            urdf_path = Path.cwd() / urdf_path
        kinematics = UrdfKinematics(
            urdf_path=urdf_path,
            joint_names=list(robot.joint_names),
            tracked_body_names=list(self.motion.cfg.body_names),
            root_body_name=self.motion.cfg.anchor_body_name,
            device=device,
        )
        action_scale = self.action_term._scale.detach().to(device)
        if action_scale.ndim > 1:
            action_scale = action_scale[0]
        end_effector_names = tuple(self.unwrapped.cfg.terminations.ee_body_pos.params["body_names"])
        body_name_to_id = {name: index for index, name in enumerate(self.motion.cfg.body_names)}
        missing_end_effectors = [name for name in end_effector_names if name not in body_name_to_id]
        if missing_end_effectors:
            raise ValueError(
                "Super-RGMT soft termination bodies are missing from the motion command: "
                f"{missing_end_effectors}."
            )
        end_effector_ids = tuple(body_name_to_id[name] for name in end_effector_names)
        reward_cfg = self._build_reward_cfg()
        self.mdp = SuperRGMTMDP(
            kinematics=kinematics,
            motion=self.motion,
            default_joint_position=robot.data.default_joint_pos[0].to(device),
            soft_limits=robot.data.soft_joint_pos_limits[0].to(device),
            action_scale=action_scale,
            context_layout=self.unwrapped.super_rgmt_context_layout,
            spec=self.spec,
            frame_offsets=tuple(self.algorithm_cfg["command_frame_offsets"]),
            end_effector_body_ids=end_effector_ids,
            cfg=reward_cfg,
        )
        algorithm_cfg = dict(self.algorithm_cfg)
        algorithm_cfg["clip_actions"] = float(train_cfg["clip_actions"])
        algorithm_cfg["recovery_window_steps"] = max(
            1, int(round(float(self.motion.cfg.recovery_window_s) / self.unwrapped.step_dt))
        )
        algorithm_cfg["max_episode_length"] = int(self.unwrapped.max_episode_length)
        self.algorithm = SuperRGMTAlgorithm(
            self.mdp,
            env.num_envs,
            initial_observation,
            train_cfg["obs_groups"],
            actor_cfg,
            algorithm_cfg,
            device,
        )
        self.alg = self.algorithm
        print(f"[INFO] Super-RGMT robot spec: {self.spec.to_dict()}")
        print(f"[INFO] Super-RGMT physics context dimension: {self.spec.context_dim}")
        print(f"[INFO] Super-RGMT replay allocation: {self.algorithm.replay.nbytes / 2**30:.2f} GiB")
        expected_control_steps = int(self.algorithm_cfg["warmup_steps"]) + int(
            round(float(train_cfg["max_iterations"]) / float(self.algorithm_cfg["update_to_data_ratio"]))
        )
        recovery_anneal_steps = int(self.motion.cfg.recovery_assist_anneal_steps)
        if self.motion.cfg.recovery_probability > 0.0 and expected_control_steps < recovery_anneal_steps:
            remaining_ratio = 1.0 - expected_control_steps / recovery_anneal_steps
            print(
                "[WARN] Super-RGMT training is scheduled for approximately "
                f"{expected_control_steps} control steps, but recovery assistance anneals over "
                f"{recovery_anneal_steps}. The final assistance multiplier will still be about "
                f"{remaining_ratio:.3f}."
            )
        self.writer = None
        self._csv_file = None
        self._csv_writer = None

    def _build_reward_cfg(self) -> RewardConfig:
        """Mirror real manager reward/termination parameters in the differentiable MDP."""
        rewards = self.unwrapped.cfg.rewards
        terminations = self.unwrapped.cfg.terminations

        def weight(name: str) -> float:
            return float(getattr(rewards, name).weight)

        def parameter(group, name: str, key: str) -> float:
            term = getattr(group, name)
            if key not in term.params:
                raise ValueError(f"Super-RGMT term '{name}' is missing required parameter '{key}'.")
            return float(term.params[key])

        return RewardConfig(
            control_dt=float(self.unwrapped.step_dt),
            body_position_weight=weight("body_position"),
            body_orientation_weight=weight("body_orientation"),
            body_linear_velocity_weight=weight("body_linear_velocity"),
            body_angular_velocity_weight=weight("body_angular_velocity"),
            root_position_weight=weight("root_position"),
            root_orientation_weight=weight("root_orientation"),
            body_position_std=parameter(rewards, "body_position", "std"),
            body_orientation_std=parameter(rewards, "body_orientation", "std"),
            body_linear_velocity_std=parameter(rewards, "body_linear_velocity", "std"),
            body_angular_velocity_std=parameter(rewards, "body_angular_velocity", "std"),
            root_position_std=parameter(rewards, "root_position", "std"),
            root_orientation_std=parameter(rewards, "root_orientation", "std"),
            action_rate_weight=weight("action_smoothness"),
            joint_limit_weight=weight("joint_limit"),
            termination_height=parameter(terminations, "anchor_pos", "threshold"),
            termination_tilt=parameter(terminations, "anchor_ori", "threshold"),
            termination_body_height=parameter(terminations, "ee_body_pos", "threshold"),
            termination_height_temperature=float(
                self.algorithm_cfg["termination_height_temperature"]
            ),
            termination_tilt_temperature=float(self.algorithm_cfg["termination_tilt_temperature"]),
        )

    def _get_observations(self):
        result = self.env.get_observations()
        return result[0] if isinstance(result, tuple) else result

    def _step(self, actions: torch.Tensor):
        return self.env.step(actions)

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    @torch.no_grad()
    def _collect(self, steps: int, statistics: dict[str, list[float]]) -> None:
        for _ in range(steps):
            state = self.unwrapped.super_rgmt_state().clone()
            context = self.unwrapped.refresh_super_rgmt_context().clone()
            frame = self.motion.time_steps.clone()
            motion_id = self.motion.motion_ids.clone()
            previous_action = self.action_term.raw_actions.clone()
            episode_step = self.unwrapped.episode_length_buf.clone()
            dropout_state = self.unwrapped.super_rgmt_command_dropout_state()

            action = self.algorithm.actor(self.observation)
            action = torch.clamp(
                action + float(self.algorithm_cfg["action_noise"]) * torch.randn_like(action),
                -float(self.cfg["clip_actions"]),
                float(self.cfg["clip_actions"]),
            )
            next_observation, reward, done, extras = self.env.step(action)
            transition_info = extras.get("super_rgmt")
            if transition_info is None:
                raise RuntimeError("Super-RGMT environment did not expose its pre-reset transition state.")
            terminated = transition_info["terminated"].bool()
            time_outs = transition_info["time_outs"].bool()
            transition = {
                "s": state,
                "s1": transition_info["next_state"],
                "joint_target": transition_info["joint_target"],
                "previous_action": previous_action,
                "dropout_cached_command": dropout_state["cached_command"].reshape(self.env.num_envs, -1),
                "dropout_remaining": dropout_state["remaining"].to(torch.int32),
                "dropout_active": dropout_state["active"],
                "xi": context,
                "r": reward,
                "f": frame,
                "motion_id": motion_id,
                "ep_step": episode_step.to(torch.int32),
                "done": terminated | time_outs,
                "time_out": time_outs,
            }
            self.algorithm.replay.add(transition)
            self.algorithm.update_normalizers(transition, self.observation)
            self.observation = next_observation.to(self.device)
            self.sim_steps += self.env.num_envs
            statistics["sim_reward"].append(float(reward.mean() / 0.1))
            statistics["recovery_ratio"].append(
                float(context[..., self.unwrapped.super_rgmt_context_layout.recovery_active].float().mean())
            )
            statistics["dropout_ratio"].append(float(dropout_state["active"].float().mean()))
            if done.any():
                ids = done.bool().nonzero(as_tuple=False).squeeze(-1)
                statistics["episode_length"].append(float((episode_step[ids] + 1).float().mean()))
                statistics["termination_count"].append(float(terminated[ids].float().sum()))
                statistics["done_count"].append(float(ids.numel()))

    @staticmethod
    def _mean(values: list) -> float:
        return sum(float(value) for value in values) / max(len(values), 1)

    def _open_logging(self, resume: bool) -> None:
        from torch.utils.tensorboard import SummaryWriter

        if self.log_dir is None:
            raise RuntimeError("Super-RGMT training requires a log directory.")
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            purge_step=self.current_learning_iteration if resume else None,
        )
        csv_path = os.path.join(self.log_dir, "train_log.csv")
        exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        columns = [
            "iteration",
            "wall_h",
            "sim_steps",
            "world_model_loss",
            "world_model_error_8",
            "world_model_recent_error_8",
            "world_model_recent_valid_ratio",
            "actor_loss",
            "critic_loss",
            "imagined_reward",
            "actor_gradient_norm",
            "imagined_survival",
            "valid_start_ratio",
            "real_recovery_ratio",
            "replay_recovery_ratio",
            "imagined_recovery_ratio",
            "real_dropout_ratio",
            "replay_dropout_ratio",
            "imagined_dropout_ratio",
            "soft_termination_height",
            "soft_termination_tilt",
            "soft_termination_end_effector",
            "sim_reward",
            "episode_length",
            "termination_fraction",
            "iterations_per_second",
            "learning_rate",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=columns)
        if not exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def _log(self, iteration: int, elapsed: float, interval_elapsed: float, stats, aggregate, learning_rate):
        termination_fraction = sum(stats["termination_count"]) / max(sum(stats["done_count"]), 1.0)
        row = {
            "iteration": iteration,
            "wall_h": elapsed / 3600.0,
            "sim_steps": self.sim_steps,
            "world_model_loss": self._mean(aggregate["world_model_loss"]),
            "world_model_error_8": self._mean(aggregate["world_model_error_8"]),
            "world_model_recent_error_8": self._mean(aggregate["world_model_recent_error_8"]),
            "world_model_recent_valid_ratio": self._mean(aggregate["world_model_recent_valid_ratio"]),
            "actor_loss": self._mean(aggregate["actor_loss"]),
            "critic_loss": self._mean(aggregate["critic_loss"]),
            "imagined_reward": self._mean(aggregate["imagined_reward"]),
            "actor_gradient_norm": self._mean(aggregate["actor_gradient_norm"]),
            "imagined_survival": self._mean(aggregate["imagined_survival"]),
            "valid_start_ratio": self._mean(aggregate["valid_start_ratio"]),
            "real_recovery_ratio": self._mean(stats["recovery_ratio"]),
            "replay_recovery_ratio": self._mean(aggregate["replay_recovery_ratio"]),
            "imagined_recovery_ratio": self._mean(aggregate["imagined_recovery_ratio"]),
            "real_dropout_ratio": self._mean(stats["dropout_ratio"]),
            "replay_dropout_ratio": self._mean(aggregate["replay_dropout_ratio"]),
            "imagined_dropout_ratio": self._mean(aggregate["imagined_dropout_ratio"]),
            "soft_termination_height": self._mean(aggregate["soft_termination_height"]),
            "soft_termination_tilt": self._mean(aggregate["soft_termination_tilt"]),
            "soft_termination_end_effector": self._mean(aggregate["soft_termination_end_effector"]),
            "sim_reward": self._mean(stats["sim_reward"]),
            "episode_length": self._mean(stats["episode_length"]),
            "termination_fraction": termination_fraction,
            "iterations_per_second": int(self.algorithm_cfg["log_interval"]) / interval_elapsed,
            "learning_rate": learning_rate,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        for name, value in row.items():
            if name not in ("iteration", "wall_h", "sim_steps"):
                self.writer.add_scalar(f"SuperRGMT/{name}", value, iteration)
        print(
            f"it {iteration} | {row['wall_h']:.3f} h | wm {row['world_model_loss']:.4f} "
            f"e8 {row['world_model_error_8']:.4f}/{row['world_model_recent_error_8']:.4f} "
            f"| actor {row['actor_loss']:.3f} "
            f"v {row['critic_loss']:.4f} imag_r {row['imagined_reward']:.3f} "
            f"surv {row['imagined_survival']:.3f} | sim_r {row['sim_reward']:.3f} "
            f"ep_len {row['episode_length']:.1f} term {termination_fraction:.2f} | "
            f"{row['iterations_per_second']:.2f} it/s",
            flush=True,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        del init_at_random_ep_len
        self.algorithm.world_model.train()
        self.algorithm.actor.train()
        self.algorithm.critic.train()
        warmup_steps = max(int(self.algorithm_cfg["warmup_steps"]), self.algorithm.world_model_window)
        stats = {
            name: []
            for name in (
                "sim_reward",
                "episode_length",
                "termination_count",
                "done_count",
                "recovery_ratio",
                "dropout_ratio",
            )
        }
        print(f"[INFO] Filling Super-RGMT replay with {warmup_steps} simulator steps.")
        self._collect(warmup_steps, stats)
        self._open_logging(resume=self.current_learning_iteration > 0)

        aggregate = {
            name: []
            for name in (
                "world_model_loss",
                "world_model_error_8",
                "world_model_recent_error_8",
                "world_model_recent_valid_ratio",
                "actor_loss",
                "critic_loss",
                "imagined_reward",
                "actor_gradient_norm",
                "imagined_survival",
                "valid_start_ratio",
                "replay_recovery_ratio",
                "imagined_recovery_ratio",
                "replay_dropout_ratio",
                "imagined_dropout_ratio",
                "soft_termination_height",
                "soft_termination_tilt",
                "soft_termination_end_effector",
            )
        }
        start_time = time.time() - self.wall_time_offset
        interval_start = time.time()
        step_accumulator = 0.0
        start_iteration = self.current_learning_iteration
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
                recent_error, recent_valid = self.algorithm.recent_world_model_error(
                    int(self.algorithm_cfg.get("diagnostic_batch_size", 2048))
                )
                aggregate["world_model_recent_error_8"].append(float(recent_error))
                aggregate["world_model_recent_valid_ratio"].append(float(recent_valid))
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
        last_path = None if self.log_dir is None else os.path.join(self.log_dir, "last.pt")
        if last_path is not None and os.path.abspath(path) != os.path.abspath(last_path):
            temporary_last = f"{last_path}.tmp"
            torch.save(checkpoint, temporary_last)
            os.replace(temporary_last, last_path)

    def load(self, path: str, load_cfg=None, strict=True, map_location=None) -> dict | None:
        del load_cfg, strict
        checkpoint = torch.load(path, weights_only=False, map_location=map_location or self.device)
        is_super_rgmt = "super_rgmt_spec" in checkpoint
        self.algorithm.load_state_dict(checkpoint, load_optimizers=is_super_rgmt)
        if is_super_rgmt:
            self.current_learning_iteration = int(checkpoint.get("iteration", 0))
            self.wall_time_offset = float(checkpoint.get("wall_time", 0.0))
            self.sim_steps = int(checkpoint.get("sim_steps", 0))
        sampler = checkpoint.get("motion_sampler")
        if sampler is None:
            sampler = checkpoint.get("rgmt_env_state_dict", {}).get("motion_sampler")
        if sampler is not None:
            self.motion.load_sampler_state_dict(sampler, strict=False)
            if "recovery_steps_elapsed" not in sampler and hasattr(self.motion, "recovery_steps_elapsed"):
                self.motion.recovery_steps_elapsed = max(0, self.sim_steps // self.env.num_envs)
                print(
                    "[INFO] Inferred legacy recovery anneal progress from sim_steps: "
                    f"{self.motion.recovery_steps_elapsed} control steps."
                )
        source = "Super-RGMT checkpoint" if is_super_rgmt else "PPO RGMT actor checkpoint"
        print(f"[INFO] Restored {source}; iteration={self.current_learning_iteration}, sim_steps={self.sim_steps}.")
        return checkpoint.get("infos")

    def get_inference_policy(self, device: str | None = None):
        self.algorithm.eval_mode()
        return self.algorithm.get_policy().to(device or self.device)

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export only the deterministic RGMT actor used by Super-RGMT."""
        model = self.algorithm.get_policy().as_jit().to("cpu")
        os.makedirs(path, exist_ok=True)
        torch.jit.script(model).save(os.path.join(path, filename))

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export only the deterministic RGMT actor used by Super-RGMT."""
        model = self.algorithm.get_policy().as_onnx(verbose=verbose).to("cpu")
        model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            model,
            model.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=model.input_names,
            output_names=model.output_names,
        )
