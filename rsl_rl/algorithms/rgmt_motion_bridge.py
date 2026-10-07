from __future__ import annotations

import copy
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.motion_bridge import MotionBridgeRetargeter
from rsl_rl.algorithms.rgmt import RGMT, RGMTActorModel
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer, unpad_trajectories


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    return int(value)


class RGMTMotionBridgeActorModel(RGMTActorModel):
    """RGMT actor with a trainable SMPL-X -> RGMT-command frontend.

    The environment provides both the ordinary RGMT reference command window and a
    paired SMPL/human-motion window. A per-env command-source flag chooses the
    active path:

    - reference path: use the reference command window directly;
    - SMPL path: run MotionBridge and use its RGMT command tokens directly.

    The actor outputs final joint-position targets. Internally it still predicts
    an RGMT residual, but adds it to the selected base motion before updating the
    action distribution.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        smpl_obs_set: str = "smpl_window",
        command_source_obs_set: str = "command_source",
        bridge_target_obs_set: str = "bridge_target_window",
        motion_bridge_checkpoint: str = "logs/motion_bridge/model_250.pt",
        motion_bridge_dt: float = 0.02,
        freeze_motion_bridge: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

        if smpl_obs_set not in obs_groups:
            raise KeyError(f"RGMTMotionBridgeActorModel requires obs_groups['{smpl_obs_set}'].")
        if command_source_obs_set not in obs_groups:
            raise KeyError(f"RGMTMotionBridgeActorModel requires obs_groups['{command_source_obs_set}'].")
        if bridge_target_obs_set and bridge_target_obs_set not in obs_groups:
            raise KeyError(f"RGMTMotionBridgeActorModel requires obs_groups['{bridge_target_obs_set}'].")
        self.smpl_obs_groups = obs_groups[smpl_obs_set]
        self.command_source_obs_groups = obs_groups[command_source_obs_set]
        self.bridge_target_obs_groups = obs_groups[bridge_target_obs_set] if bridge_target_obs_set else None
        self.smpl_step_dim = self._infer_step_dim(obs, self.smpl_obs_groups, self.command_window_size)
        self.motion_bridge_dt = float(motion_bridge_dt)
        self._output_dim = output_dim
        self._last_bridge_prediction = None
        self._last_mixed_command = None
        self._last_mixed_command_source = None
        self.freeze_motion_bridge = bool(freeze_motion_bridge)

        checkpoint_path = Path(motion_bridge_checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MotionBridge checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        input_dim = int(checkpoint.get("input_dim", self.smpl_step_dim))
        target_dim = int(checkpoint["target_dim"])
        self.motion_bridge_target_dim = target_dim
        if target_dim != self.command_step_dim:
            raise ValueError(
                f"MotionBridge checkpoint target_dim={target_dim}, but RGMT command step dim is "
                f"{self.command_step_dim}. The current MotionBridge must output [lin_vel_b, ang_vel_b, gravity_b, q_ref] "
                "directly."
            )
        if input_dim != self.smpl_step_dim:
            raise ValueError(
                f"MotionBridge checkpoint expects input_dim={input_dim}, but smpl_window step dim is {self.smpl_step_dim}."
            )
        args = checkpoint.get("args", {})
        self.motion_bridge = MotionBridgeRetargeter(
            input_dim=input_dim,
            output_dim=target_dim,
            hidden_dim=_as_int(args.get("hidden_dim"), 512),
            num_conv_blocks=_as_int(args.get("num_conv_blocks"), 3),
            num_transformer_layers=_as_int(args.get("num_transformer_layers"), 6),
            num_attention_heads=_as_int(args.get("num_attention_heads"), 8),
            feedforward_dim=_as_int(args.get("feedforward_dim"), 2048),
        )
        self.motion_bridge.load_state_dict(checkpoint["model_state_dict"], strict=True)

        normalization = checkpoint["normalization"]
        self.register_buffer(
            "motion_bridge_input_mean",
            torch.as_tensor(np.asarray(normalization["input_mean"], dtype=np.float32)),
            persistent=True,
        )
        self.register_buffer(
            "motion_bridge_input_std",
            torch.as_tensor(np.asarray(normalization["input_std"], dtype=np.float32)),
            persistent=True,
        )
        self.register_buffer(
            "motion_bridge_target_mean",
            torch.as_tensor(np.asarray(normalization["target_mean"], dtype=np.float32)),
            persistent=True,
        )
        self.register_buffer(
            "motion_bridge_target_std",
            torch.as_tensor(np.asarray(normalization["target_std"], dtype=np.float32)),
            persistent=True,
        )

        if self.freeze_motion_bridge:
            for param in self.motion_bridge.parameters():
                param.requires_grad_(False)
            self.motion_bridge.eval()

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent, base_joint_pos = self.get_latent_and_base_joint_pos(obs)
        residual_mean = self.mlp(latent)
        joint_target_mean = base_joint_pos + residual_mean
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(joint_target_mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(joint_target_mean)
        return joint_target_mean

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent, _ = self.get_latent_and_base_joint_pos(obs)
        return latent

    def get_latent_and_base_joint_pos(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        policy_obs = self.obs_normalizer(self._flatten_obs_groups(obs, self.obs_groups))
        state_history = self.state_normalizer(
            self._sequence_obs_groups(obs, self.state_history_obs_groups, self.history_length)
        )
        action_history = self.action_normalizer(
            self._sequence_obs_groups(obs, self.action_history_obs_groups, self.history_length)
        )
        command_raw, base_joint_pos = self._mixed_command_and_base_joint_pos(obs)
        command = self.command_normalizer(command_raw)

        state_tokens = self.state_encoder_norm(self.state_step_encoder(state_history))
        action_tokens = self.action_encoder_norm(self.action_step_encoder(action_history))
        history_tokens = torch.stack((action_tokens, state_tokens), dim=2).reshape(
            state_tokens.shape[0],
            self.history_length * 2,
            self.embedding_dim,
        )
        history_tokens = self.history_position_encoding(history_tokens)
        causal_mask = torch.triu(
            torch.ones(self.history_length * 2, self.history_length * 2, device=history_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.history_blocks:
            history_tokens = block(history_tokens, causal_mask)
        dynamics_latent = torch.max(history_tokens, dim=1).values

        command_query = self.dynamics_query_encoder(dynamics_latent)
        command_tokens = self.command_position_encoding(self.command_encoder_norm(self.command_step_encoder(command)))
        command_latent = self.command_block(command_query, command_tokens)
        command_latent = self.command_quantizer(command_latent)

        return torch.cat((policy_obs, command_latent), dim=-1), base_joint_pos

    def _mixed_command_and_base_joint_pos(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        reference_command = self._sequence_obs_groups(obs, self.command_obs_groups, self.command_window_size)
        command_source = self._flatten_obs_groups(obs, self.command_source_obs_groups)
        smpl_rows = command_source[:, 0] > 0.5

        command = reference_command.clone()
        bridge_prediction = reference_command.new_zeros(reference_command.shape)

        center_id = self.command_window_size // 2
        reference_base_joint_pos = reference_command[:, center_id, 9 : 9 + self._output_dim]
        base_joint_pos = reference_base_joint_pos.clone()

        if torch.any(smpl_rows):
            smpl_window = self._sequence_obs_groups(obs, self.smpl_obs_groups, self.command_window_size)
            bridge_target = self._motion_bridge_target(smpl_window[smpl_rows])
            command[smpl_rows] = bridge_target
            bridge_prediction[smpl_rows] = bridge_target
            base_joint_pos[smpl_rows] = self._paper_target_joint_pos(bridge_target)[:, center_id]

        self._last_bridge_prediction = bridge_prediction
        self._last_mixed_command = command
        self._last_mixed_command_source = torch.zeros_like(command_source)
        return command, base_joint_pos

    def _motion_bridge_target(self, smpl_window: torch.Tensor) -> torch.Tensor:
        if self.freeze_motion_bridge:
            with torch.no_grad():
                return self._motion_bridge_target_impl(smpl_window)
        return self._motion_bridge_target_impl(smpl_window)

    def _motion_bridge_target_impl(self, smpl_window: torch.Tensor) -> torch.Tensor:
        smpl_norm = (smpl_window - self.motion_bridge_input_mean) / self.motion_bridge_input_std
        target_norm = self.motion_bridge(smpl_norm)
        return target_norm * self.motion_bridge_target_std + self.motion_bridge_target_mean

    def motion_bridge_prediction_and_reference(self, obs: TensorDict, prediction=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.bridge_target_obs_groups is None:
            raise RuntimeError("MotionBridge target observations are not configured.")
        target_window = self._sequence_obs_groups(obs, self.bridge_target_obs_groups, self.command_window_size)
        command_source = self._flatten_obs_groups(obs, self.command_source_obs_groups)
        smpl_mask = (command_source[:, :1] > 0.5).to(target_window.dtype)
        if prediction is None or prediction.shape[:2] != target_window.shape[:2]:
            smpl_window = self._sequence_obs_groups(obs, self.smpl_obs_groups, self.command_window_size)
            prediction = self._motion_bridge_target(smpl_window)
        return prediction, target_window, smpl_mask

    def _paper_target_joint_pos(self, target: torch.Tensor) -> torch.Tensor:
        return target[..., -self._output_dim :]

    def as_jit(self) -> nn.Module:
        return _TorchRGMTMotionBridgeActorModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxRGMTMotionBridgeActorModel(self, verbose)


class _TorchRGMTMotionBridgeActorModel(nn.Module):
    """Exportable deterministic RGMT actor with MotionBridge frontend.

    Forward inputs:
        policy_obs: concatenated actor observations.
        state_history_obs: [B, history_length, state_step_dim].
        action_history_obs: [B, history_length, action_step_dim].
        command_obs: [B, command_window_size, command_step_dim].
        smpl_obs: [B, command_window_size, smpl_step_dim].
        command_source_obs: [B, 1], 1 for SMPL path and 0 for reference path.
    """

    def __init__(self, model: RGMTMotionBridgeActorModel) -> None:
        super().__init__()
        self.history_length = model.history_length
        self.command_window_size = model.command_window_size
        self.embedding_dim = model.embedding_dim
        self.output_dim = model._output_dim
        self.motion_bridge_dt = model.motion_bridge_dt
        self.motion_bridge_target_dim = model.motion_bridge_target_dim

        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.state_normalizer = copy.deepcopy(model.state_normalizer)
        self.action_normalizer = copy.deepcopy(model.action_normalizer)
        self.command_normalizer = copy.deepcopy(model.command_normalizer)
        self.state_step_encoder = copy.deepcopy(model.state_step_encoder)
        self.action_step_encoder = copy.deepcopy(model.action_step_encoder)
        self.state_encoder_norm = copy.deepcopy(model.state_encoder_norm)
        self.action_encoder_norm = copy.deepcopy(model.action_encoder_norm)
        self.history_position_encoding = copy.deepcopy(model.history_position_encoding)
        self.history_blocks = copy.deepcopy(model.history_blocks)
        self.dynamics_query_encoder = copy.deepcopy(model.dynamics_query_encoder)
        self.command_step_encoder = copy.deepcopy(model.command_step_encoder)
        self.command_encoder_norm = copy.deepcopy(model.command_encoder_norm)
        self.command_position_encoding = copy.deepcopy(model.command_position_encoding)
        self.command_block = copy.deepcopy(model.command_block)
        self.command_quantizer = copy.deepcopy(model.command_quantizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.motion_bridge = copy.deepcopy(model.motion_bridge)

        self.register_buffer("motion_bridge_input_mean", model.motion_bridge_input_mean.detach().clone(), persistent=True)
        self.register_buffer("motion_bridge_input_std", model.motion_bridge_input_std.detach().clone(), persistent=True)
        self.register_buffer("motion_bridge_target_mean", model.motion_bridge_target_mean.detach().clone(), persistent=True)
        self.register_buffer("motion_bridge_target_std", model.motion_bridge_target_std.detach().clone(), persistent=True)

        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(
        self,
        policy_obs: torch.Tensor,
        state_history_obs: torch.Tensor,
        action_history_obs: torch.Tensor,
        command_obs: torch.Tensor,
        smpl_obs: torch.Tensor,
        command_source_obs: torch.Tensor,
    ) -> torch.Tensor:
        policy_obs = self.obs_normalizer(policy_obs)
        state_history = self.state_normalizer(state_history_obs)
        action_history = self.action_normalizer(action_history_obs)
        command_raw, base_joint_pos = self._mixed_command_and_base_joint_pos(command_obs, smpl_obs, command_source_obs)
        command = self.command_normalizer(command_raw)

        state_tokens = self.state_encoder_norm(self.state_step_encoder(state_history))
        action_tokens = self.action_encoder_norm(self.action_step_encoder(action_history))
        history_tokens = torch.stack((action_tokens, state_tokens), dim=2).reshape(
            state_tokens.shape[0],
            self.history_length * 2,
            self.embedding_dim,
        )
        history_tokens = self.history_position_encoding(history_tokens)
        causal_mask = torch.triu(
            torch.ones(self.history_length * 2, self.history_length * 2, device=history_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.history_blocks:
            history_tokens = block(history_tokens, causal_mask)
        dynamics_latent = torch.max(history_tokens, dim=1).values

        command_query = self.dynamics_query_encoder(dynamics_latent)
        command_tokens = self.command_position_encoding(self.command_encoder_norm(self.command_step_encoder(command)))
        command_latent = self.command_block(command_query, command_tokens)
        command_latent = self.command_quantizer(command_latent)
        residual = self.deterministic_output(self.mlp(torch.cat((policy_obs, command_latent), dim=-1)))
        return base_joint_pos + residual

    def _mixed_command_and_base_joint_pos(
        self,
        reference_command: torch.Tensor,
        smpl_window: torch.Tensor,
        command_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bridge_target = self._motion_bridge_target(smpl_window)
        bridge_command = bridge_target

        smpl_mask = (command_source[:, :1] > 0.5).to(reference_command.dtype).reshape(-1, 1, 1)
        command = reference_command * (1.0 - smpl_mask) + bridge_command * smpl_mask

        center_id = self.command_window_size // 2
        reference_base_joint_pos = reference_command[:, center_id, 9 : 9 + self.output_dim]
        bridge_base_joint_pos = self._paper_target_joint_pos(bridge_target)[:, center_id]
        base_joint_pos = reference_base_joint_pos * (1.0 - smpl_mask[:, 0]) + bridge_base_joint_pos * smpl_mask[:, 0]
        return command, base_joint_pos

    def _motion_bridge_target(self, smpl_window: torch.Tensor) -> torch.Tensor:
        smpl_norm = (smpl_window - self.motion_bridge_input_mean) / self.motion_bridge_input_std
        target_norm = self.motion_bridge(smpl_norm)
        return target_norm * self.motion_bridge_target_std + self.motion_bridge_target_mean

    def _paper_target_joint_pos(self, target: torch.Tensor) -> torch.Tensor:
        return target[..., -self.output_dim :]

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxRGMTMotionBridgeActorModel(_TorchRGMTMotionBridgeActorModel):
    """Exportable RGMT-MotionBridge actor for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: RGMTMotionBridgeActorModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.policy_obs_dim = model.obs_dim
        self.state_step_dim = model.state_step_dim
        self.action_step_dim = model.action_step_dim
        self.command_step_dim = model.command_step_dim
        self.smpl_step_dim = model.smpl_step_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, self.policy_obs_dim),
            torch.zeros(1, self.history_length, self.state_step_dim),
            torch.zeros(1, self.history_length, self.action_step_dim),
            torch.zeros(1, self.command_window_size, self.command_step_dim),
            torch.zeros(1, self.command_window_size, self.smpl_step_dim),
            torch.ones(1, 1),
        )

    @property
    def input_names(self) -> list[str]:
        return [
            "policy_obs",
            "state_history_obs",
            "action_history_obs",
            "command_obs",
            "smpl_obs",
            "command_source_obs",
        ]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class RGMTMotionBridge(RGMT):
    """RGMT variant that can warm-start from ordinary RGMT checkpoints.

    The motion-bridge actor adds a pretrained retargeting frontend and a small
    command adapter. Old RGMT checkpoints do not contain those parameters, so a
    normal strict resume would fail. This loader keeps same-architecture resumes
    strict, but falls back to shape-matched partial actor loading for ordinary
    RGMT/Stage-II checkpoints and skips the incompatible optimizer state.
    """

    def __init__(
        self,
        *args,
        motion_bridge_lr_scale: float = 0.1,
        bridge_loss_coef: float = 0.1,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(*args, optimizer=optimizer, learning_rate=learning_rate, **kwargs)
        self.motion_bridge_lr_scale = float(motion_bridge_lr_scale)
        self.bridge_loss_coef = float(bridge_loss_coef)
        self._base_learning_rate = float(learning_rate)
        self._rebuild_motion_bridge_optimizer(optimizer, learning_rate)
        if hasattr(self.optimizer, "register_step_pre_hook"):
            self.optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: self._set_learning_rate(self.learning_rate))

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "RGMTMotionBridge":
        cfg.setdefault("obs_groups", {})
        cfg.setdefault("multi_gpu", None)

        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = [
            "actor",
            "critic",
            cfg["actor"].get("history_obs_set", "proprio_history"),
            cfg["actor"].get("action_history_obs_set", "action_history"),
            cfg["actor"].get("command_obs_set", "command_window"),
            cfg["actor"].get("smpl_obs_set", "smpl_window"),
            cfg["actor"].get("command_source_obs_set", "command_source"),
        ]
        bridge_target_obs_set = cfg["actor"].get("bridge_target_obs_set", "")
        if bridge_target_obs_set:
            default_sets.append(bridge_target_obs_set)
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"RGMT Actor Model: {actor}")
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage_obs = obs
        if getattr(actor, "freeze_motion_bridge", False):
            storage_obs = RGMTMotionBridge._storage_obs_without_frozen_bridge_inputs(obs, actor.smpl_obs_groups)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], storage_obs, [env.num_actions], device)
        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    @staticmethod
    def _storage_obs_without_frozen_bridge_inputs(obs: TensorDict, smpl_obs_groups: list[str]) -> TensorDict:
        drop_groups = set(smpl_obs_groups)
        return TensorDict(
            {key: value for key, value in obs.items() if key not in drop_groups},
            batch_size=obs.batch_size,
            device=obs.device,
        )

    def _rebuild_motion_bridge_optimizer(self, optimizer: str, learning_rate: float) -> None:
        all_motion_bridge_params = list(self._raw_actor.motion_bridge.parameters())
        motion_bridge_params = [param for param in all_motion_bridge_params if param.requires_grad]
        motion_bridge_param_ids = {id(param) for param in all_motion_bridge_params}
        actor_regular_params = [
            param for param in self.actor.parameters() if id(param) not in motion_bridge_param_ids and param.requires_grad
        ]
        critic_params = [param for param in self.critic.parameters() if param.requires_grad]
        param_groups = [
            {"params": chain(actor_regular_params, critic_params), "lr": learning_rate, "name": "rgmt"},
        ]
        if motion_bridge_params:
            param_groups.append(
                {
                    "params": motion_bridge_params,
                    "lr": learning_rate * self.motion_bridge_lr_scale,
                    "name": "motion_bridge",
                }
            )
        self.optimizer = resolve_optimizer(optimizer)(param_groups, lr=learning_rate)  # type: ignore

    def _set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "motion_bridge":
                param_group["lr"] = learning_rate * self.motion_bridge_lr_scale
            else:
                param_group["lr"] = learning_rate

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        if getattr(self._raw_actor, "freeze_motion_bridge", False):
            self.transition.observations = self._observations_with_cached_bridge_command(self.transition.observations)
        return actions

    def _observations_with_cached_bridge_command(self, obs: TensorDict) -> TensorDict:
        mixed_command = getattr(self._raw_actor, "_last_mixed_command", None)
        mixed_command_source = getattr(self._raw_actor, "_last_mixed_command_source", None)
        if mixed_command is None:
            return obs

        drop_groups = set(self._raw_actor.smpl_obs_groups)
        cached_obs = TensorDict(
            {key: value for key, value in obs.items() if key not in drop_groups},
            batch_size=obs.batch_size,
            device=obs.device,
        )
        flat_command = mixed_command.reshape(mixed_command.shape[0], -1).detach()
        for group in self._raw_actor.command_obs_groups:
            if group in cached_obs.keys():
                cached_obs[group] = flat_command.reshape_as(cached_obs[group])
                break

        if mixed_command_source is not None:
            flat_source = mixed_command_source.detach()
            for group in self._raw_actor.command_source_obs_groups:
                if group in cached_obs.keys():
                    cached_obs[group] = flat_source.reshape_as(cached_obs[group])
                    break
        return cached_obs

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_bridge_loss = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    self._set_learning_rate(self.learning_rate)

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            if self.bridge_loss_coef > 0.0:
                cached_bridge_prediction = getattr(self._raw_actor, "_last_bridge_prediction", None)
                if cached_bridge_prediction is not None:
                    cached_bridge_prediction = cached_bridge_prediction[:original_batch_size]
                bridge_loss = self._motion_bridge_supervised_loss(
                    batch.observations[:original_batch_size], cached_bridge_prediction
                )
            else:
                bridge_loss = torch.zeros((), device=surrogate_loss.device, dtype=surrogate_loss.dtype)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.bridge_loss_coef * bridge_loss
            )

            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self._set_learning_rate(self.learning_rate)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_bridge_loss += bridge_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_bridge_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self._set_learning_rate(self.learning_rate)
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "motion_bridge_supervised": mean_bridge_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict["motion_bridge_learning_rate"] = self.learning_rate * self.motion_bridge_lr_scale

        self.storage.clear()
        return loss_dict

    def _motion_bridge_supervised_loss(self, observations: TensorDict, prediction=None) -> torch.Tensor:
        prediction, reference, smpl_mask = self._raw_actor.motion_bridge_prediction_and_reference(observations, prediction)
        sample_loss = torch.abs(prediction - reference).mean(dim=tuple(range(1, prediction.ndim)))
        sample_mask = smpl_mask.reshape(-1).to(sample_loss.dtype)
        if torch.sum(sample_mask) <= 0.0:
            return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        return torch.sum(sample_loss * sample_mask) / torch.clamp_min(torch.sum(sample_mask), 1.0)

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        actor_fully_loaded = True
        if load_cfg.get("actor"):
            actor_fully_loaded = self._load_actor_compatible(loaded_dict["actor_state_dict"], strict=strict)

        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)

        if load_cfg.get("optimizer") and actor_fully_loaded:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        elif load_cfg.get("optimizer"):
            print("[RGMTMotionBridge] Skipped optimizer state: actor was warm-started from a partial checkpoint.")

        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def _load_actor_compatible(self, loaded_state: dict[str, torch.Tensor], strict: bool) -> bool:
        current_state = self._raw_actor.state_dict()
        same_keys = loaded_state.keys() == current_state.keys()
        same_shapes = same_keys and all(
            loaded_state[name].shape == current_state[name].shape for name in current_state.keys()
        )
        if same_shapes:
            self._raw_actor.load_state_dict(loaded_state, strict=strict)
            return True

        compatible_state = {}
        skipped = []
        for name, tensor in loaded_state.items():
            if name in current_state and current_state[name].shape == tensor.shape:
                compatible_state[name] = tensor
            else:
                skipped.append(name)

        merged_state = dict(current_state)
        merged_state.update(compatible_state)
        self._raw_actor.load_state_dict(merged_state, strict=False)
        missing_count = len(current_state) - len(compatible_state)
        print(
            "[RGMTMotionBridge] Warm-started actor with "
            f"{len(compatible_state)}/{len(current_state)} compatible tensors; "
            f"left {missing_count} new tensors initialized; skipped {len(skipped)} checkpoint tensors."
        )
        return False
