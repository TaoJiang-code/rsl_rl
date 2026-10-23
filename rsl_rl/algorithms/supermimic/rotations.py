"""Batched SO(3) utilities using wxyz quaternions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    cands = torch.stack(
        (
            torch.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            torch.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        dim=-2,
    )
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def rpy_to_mat(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr,
        ),
        dim=-1,
    ).reshape(rpy.shape[:-1] + (3, 3))


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    x, y, z = axis.unbind(-1)
    c, s, one_c = torch.cos(angle), torch.sin(angle), 1 - torch.cos(angle)
    return torch.stack(
        (
            c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s,
            y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s,
            z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    k = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(k.shape)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    scale = torch.where(
        sin_half < 1e-6,
        2.0 / torch.clamp(q[..., :1], min=1e-6),
        2 * half / (sin_half + 1e-12),
    )
    return v * scale


def so3_log(matrix: torch.Tensor) -> torch.Tensor:
    return quat_to_rotvec(mat_to_quat(matrix))


def rot_angle_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    trace = (a * b).sum((-1, -2))
    cosine = torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine).square()


def yaw_mat(matrix: torch.Tensor) -> torch.Tensor:
    x, y = matrix[..., 0, 0], matrix[..., 1, 0]
    norm = torch.sqrt(x * x + y * y + 1e-9)
    c, s = x / norm, y / norm
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(matrix.shape[:-2] + (3, 3))


def mat_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))
