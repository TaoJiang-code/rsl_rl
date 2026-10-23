"""JIT and ONNX export for the two-input SuperMimic policy."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from .mdp import GOAL_STEP_DIM, OBS_DIM


class _SuperMimicExportPolicy(nn.Module):
    """FP32 inference graph with the policy normalizers embedded."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = copy.deepcopy(policy).cpu().eval()

    def forward(self, observation_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(self.policy.encoder(self.policy.goal_norm(goal)))
        observation = self.policy.obs_norm(observation_history).flatten(-2)
        return self.policy.decoder(torch.cat((latent, observation), dim=-1))


def export_supermimic_policy(
    policy: nn.Module,
    path: str | Path,
    history_length: int,
    goal_frames: int,
    jit_filename: str = "policy.pt",
    onnx_filename: str = "policy.onnx",
) -> tuple[Path, Path]:
    """Export a SuperMimic policy with dynamic batch size and explicit history/goal inputs."""
    export_dir = Path(path)
    export_dir.mkdir(parents=True, exist_ok=True)
    model = _SuperMimicExportPolicy(policy).eval()
    goal_dim = GOAL_STEP_DIM * goal_frames
    example_history = torch.zeros(1, history_length, OBS_DIM)
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
        "format_version": 1,
        "inputs": {
            "observation_history": ["batch", history_length, OBS_DIM],
            "goal": ["batch", goal_dim],
        },
        "outputs": {"action": ["batch", 29]},
        "observation_layout": [
            "joint_position_relative:29",
            "joint_velocity:29",
            "base_angular_velocity:3",
            "projected_gravity:3",
            "previous_action:29",
            "relative_reference_root_orientation_6d:6",
        ],
        "goal_frame_layout": [
            "root_position_heading_local:3",
            "root_orientation_heading_local_6d:6",
            "root_linear_velocity_local:3",
            "root_angular_velocity_local:3",
            "joint_position:29",
            "joint_velocity_scaled_0.1:29",
        ],
        "goal_frames": goal_frames,
    }
    (export_dir / "policy_io.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return jit_path, onnx_path
