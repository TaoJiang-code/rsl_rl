"""JIT and ONNX export for the two-input SuperMimic policy."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from .mdp import SuperMimicRobotSpec


class _SuperMimicExportPolicy(nn.Module):
    """FP32 inference graph with the policy normalizers embedded."""

    def __init__(self, policy: nn.Module, interface: str = "motion"):
        super().__init__()
        self.policy = copy.deepcopy(policy).cpu().eval()
        self.interface = interface

    def forward(self, observation_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.interface == "motion":
            latent = torch.tanh(self.policy.encoder(self.policy.goal_norm(goal)))
        else:
            latent = torch.tanh(
                self.policy.teleop_encoder(self.policy.teleop_goal_norm(goal))
            )
        observation = self.policy.obs_norm(observation_history).flatten(-2)
        return self.policy.decoder(torch.cat((latent, observation), dim=-1))


def export_supermimic_policy(
    policy: nn.Module,
    path: str | Path,
    robot_spec: SuperMimicRobotSpec,
    history_length: int,
    goal_frames: int,
    jit_filename: str = "policy.pt",
    onnx_filename: str = "policy.onnx",
    metadata_filename: str = "policy_io.json",
    interface: str = "motion",
) -> tuple[Path, Path]:
    """Export a SuperMimic policy with dynamic batch size and explicit history/goal inputs."""
    export_dir = Path(path)
    export_dir.mkdir(parents=True, exist_ok=True)
    if interface not in ("motion", "teleop"):
        raise ValueError(f"Unknown SuperMimic export interface: {interface!r}.")
    goal_dim = (
        robot_spec.goal_dim(goal_frames)
        if interface == "motion"
        else robot_spec.teleop_goal_dim(goal_frames)
    )
    expected = (
        robot_spec.observation_dim,
        robot_spec.goal_dim(goal_frames),
        robot_spec.num_joints,
    )
    actual = (policy.obs_dim, policy.goal_dim, policy.action_dim)
    if actual != expected:
        raise ValueError(f"Policy dimensions {actual} do not match robot spec dimensions {expected}.")
    if interface == "teleop" and (
        policy.teleop_encoder is None
        or policy.teleop_goal_norm is None
        or policy.teleop_goal_dim != goal_dim
    ):
        raise ValueError("Policy does not contain the expected teleop encoder.")
    model = _SuperMimicExportPolicy(policy, interface=interface).eval()
    example_history = torch.zeros(1, history_length, robot_spec.observation_dim)
    example_goal = torch.zeros(1, goal_dim)

    jit_path = export_dir / jit_filename
    with torch.inference_mode():
        traced = torch.jit.trace(model, (example_history, example_goal), strict=True)
        traced.eval()
        traced = torch.jit.freeze(traced)
        torch.testing.assert_close(
            traced(example_history, example_goal),
            model(example_history, example_goal),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        traced.save(str(jit_path))

    onnx_path = export_dir / onnx_filename
    torch.onnx.export(
        model,
        (example_history, example_goal),
        str(onnx_path),
        input_names=["observation_history", "goal"],
        output_names=["action"],
        dynamic_axes={
            "observation_history": {0: "batch"},
            "goal": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    metadata = {
        "format_version": 2 if interface == "teleop" else 1,
        "interface": interface,
        "inputs": {
            "observation_history": ["batch", history_length, robot_spec.observation_dim],
            "goal": ["batch", goal_dim],
        },
        "outputs": {"action": ["batch", robot_spec.num_joints]},
        "observation_layout": [
            f"joint_position_relative:{robot_spec.num_joints}",
            f"joint_velocity:{robot_spec.num_joints}",
            "base_angular_velocity:3",
            "projected_gravity:3",
            f"previous_action:{robot_spec.num_joints}",
            "relative_reference_root_orientation_6d:6",
        ],
        "goal_layout": (
            [
                "velocity_command_base:[vx,vy,wz]",
                f"future_joint_position:{goal_frames}x{robot_spec.num_joints}",
                f"future_joint_velocity_scaled_0.1:{goal_frames}x{robot_spec.num_joints}",
            ]
            if interface == "teleop"
            else [
                "per_frame_root_position_heading_local:3",
                "per_frame_root_orientation_heading_local_6d:6",
                "per_frame_root_linear_velocity_local:3",
                "per_frame_root_angular_velocity_local:3",
                f"per_frame_joint_position:{robot_spec.num_joints}",
                f"per_frame_joint_velocity_scaled_0.1:{robot_spec.num_joints}",
            ]
        ),
        "goal_frames": goal_frames,
        "robot_spec": robot_spec.to_dict(),
    }
    if interface == "motion":
        # Preserve the original deployment metadata key for existing consumers.
        metadata["goal_frame_layout"] = [
            "root_position_heading_local:3",
            "root_orientation_heading_local_6d:6",
            "root_linear_velocity_local:3",
            "root_angular_velocity_local:3",
            f"joint_position:{robot_spec.num_joints}",
            f"joint_velocity_scaled_0.1:{robot_spec.num_joints}",
        ]
    (export_dir / metadata_filename).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return jit_path, onnx_path
