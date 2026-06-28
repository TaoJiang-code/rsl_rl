# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
import numpy as np
from collections.abc import Generator, Sequence


class AMPLoader:
    """Load AMP expert motions and yield state-transition mini-batches.

    The loader supports the body-state ``.npz`` format used by the MuJoCo AMP
    code path and the closely related IsaacLab motion key names. It computes
    AMP states from selected body poses and velocities relative to an anchor
    body: local position, the first two rotation-matrix columns, local linear
    velocity, and local angular velocity.
    """

    def __init__(
        self,
        motion_file: str | Sequence[str],
        body_names: Sequence[str],
        anchor_name: str,
        all_body_names: Sequence[str] | None = None,
        device: str = "cpu",
    ) -> None:
        """Load one or more motion files."""
        self.device = device
        self.body_names = list(body_names)
        self.anchor_name = anchor_name

        motion_files = self._resolve_motion_files(motion_file)
        if not motion_files:
            raise ValueError(f"No AMP motion files found in: {motion_file}")

        self.motion_names: list[str] = []
        self._states: list[torch.Tensor] = []
        self._next_states: list[torch.Tensor] = []
        self.fps = None

        for path in motion_files:
            states = self._load_motion(path, all_body_names)
            self.motion_names.append(os.path.splitext(os.path.basename(path))[0])
            self._states.append(states[:-1])
            self._next_states.append(states[1:])

        self.observation_dim = self._states[0].shape[-1]
        self.num_motions = len(self._states)

    def feed_forward_generator(
        self, num_mini_batches: int, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yield expert AMP transition mini-batches."""
        for batch_idx in range(num_mini_batches):
            motion_idx = batch_idx % self.num_motions
            states = self._states[motion_idx]
            next_states = self._next_states[motion_idx]
            sample_idxs = torch.randint(states.shape[0], (mini_batch_size,), device=self.device)
            yield states[sample_idxs], next_states[sample_idxs]

    def _load_motion(self, path: str, all_body_names: Sequence[str] | None) -> torch.Tensor:
        """Load and convert a single motion file to AMP states."""
        data = np.load(path)
        if self.fps is None and "fps" in data:
            self.fps = float(data["fps"])

        body_pos_w = self._get_array(data, "body_pos_w", "body_positions")
        body_quat_w = self._get_array(data, "body_quat_w", "body_rotations")
        body_lin_vel_w = self._get_array(data, "body_lin_vel_w", "body_linear_velocities")
        body_ang_vel_w = self._get_array(data, "body_ang_vel_w", "body_angular_velocities")

        names = list(all_body_names) if all_body_names is not None else None
        if names is None and "body_names" in data:
            names = [_decode_name(name) for name in data["body_names"].tolist()]
        if names is None:
            raise ValueError(
                "AMP motion files without a 'body_names' array require 'all_body_names' in the runner config."
            )

        body_indexes = [names.index(name) for name in self.body_names]
        anchor_index = names.index(self.anchor_name)

        body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=self.device)
        body_quat_w = torch.tensor(body_quat_w, dtype=torch.float32, device=self.device)
        body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=self.device)
        body_ang_vel_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=self.device)

        anchor_pos_w = body_pos_w[:, anchor_index].unsqueeze(1).expand(-1, len(body_indexes), -1)
        anchor_quat_w = body_quat_w[:, anchor_index].unsqueeze(1).expand(-1, len(body_indexes), -1)
        target_pos_w = body_pos_w[:, body_indexes]
        target_quat_w = body_quat_w[:, body_indexes]
        target_lin_vel_w = body_lin_vel_w[:, body_indexes]
        target_ang_vel_w = body_ang_vel_w[:, body_indexes]

        body_pos_b, body_quat_b = _subtract_frame_transforms(
            anchor_pos_w,
            anchor_quat_w,
            target_pos_w,
            target_quat_w,
        )
        body_ori_b = _matrix_from_quat(body_quat_b)[..., :, :2].reshape(body_quat_b.shape[0], len(body_indexes), 6)
        body_lin_vel_b = _quat_apply_inverse(target_quat_w, target_lin_vel_w)
        body_ang_vel_b = _quat_apply_inverse(target_quat_w, target_ang_vel_w)

        return torch.cat(
            [
                body_pos_b.reshape(body_pos_b.shape[0], -1),
                body_ori_b.reshape(body_ori_b.shape[0], -1),
                body_lin_vel_b.reshape(body_lin_vel_b.shape[0], -1),
                body_ang_vel_b.reshape(body_ang_vel_b.shape[0], -1),
            ],
            dim=-1,
        )

    @staticmethod
    def _resolve_motion_files(motion_file: str | Sequence[str]) -> list[str]:
        """Resolve files from a path, directory, or sequence of paths."""
        if isinstance(motion_file, (list, tuple)):
            files: list[str] = []
            for item in motion_file:
                files.extend(AMPLoader._resolve_motion_files(item))
            return files
        if os.path.isfile(motion_file):
            return [motion_file]
        if os.path.isdir(motion_file):
            files = []
            for root, _dirs, filenames in os.walk(motion_file):
                files.extend(os.path.join(root, name) for name in sorted(filenames) if name.endswith(".npz"))
            return sorted(files)
        raise ValueError(f"AMP motion path is neither a file nor a directory: {motion_file}")

    @staticmethod
    def _get_array(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
        """Read the first available array name from an npz file."""
        for name in names:
            if name in data:
                return data[name]
        raise KeyError(f"None of the AMP motion arrays were found: {names}")


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a wxyz quaternion."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _decode_name(name: object) -> str:
    """Decode body or joint names stored in NumPy arrays."""
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a wxyz quaternion."""
    q_vec = q[..., 1:]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_vec, t, dim=-1)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a wxyz quaternion."""
    return _quat_apply(_quat_conjugate(q), v)


def _subtract_frame_transforms(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a child frame in a parent frame."""
    rel_pos = _quat_apply_inverse(parent_quat, child_pos - parent_pos)
    rel_quat = _quat_multiply(_quat_conjugate(parent_quat), child_quat)
    return rel_pos, rel_quat


def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
