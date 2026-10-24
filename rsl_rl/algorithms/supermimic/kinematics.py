"""Differentiable G1 forward kinematics built directly from its URDF."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from .rotations import axis_angle_to_mat, rpy_to_mat


def _vec(text: str | None, default: tuple[float, float, float]) -> list[float]:
    return [float(value) for value in text.split()] if text else list(default)


class UrdfKinematics:
    """Torch FK with explicit joint-name mapping and support for fixed URDF links."""

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: list[str],
        tracked_body_names: list[str],
        root_body_name: str,
        device: str,
    ):
        root = ET.parse(urdf_path).getroot()
        joint_to_index = {name: index for index, name in enumerate(joint_names)}
        link_com_offsets = {}
        for link in root.findall("link"):
            inertial_origin = link.find("inertial/origin")
            xyz = None if inertial_origin is None else inertial_origin.attrib.get("xyz")
            link_com_offsets[link.attrib["name"]] = _vec(xyz, (0.0, 0.0, 0.0))
        pending = list(root.findall("joint"))
        body_names = [root_body_name]
        body_to_index = {root_body_name: 0}
        parent = [-1]
        joint_index = [-1]
        axes = [[0.0, 0.0, 1.0]]
        offsets = [[0.0, 0.0, 0.0]]
        offset_rpy = [[0.0, 0.0, 0.0]]
        com_offsets = [link_com_offsets.get(root_body_name, [0.0, 0.0, 0.0])]

        while pending:
            progressed = False
            for element in pending[:]:
                parent_name = element.find("parent").attrib["link"]
                child_name = element.find("child").attrib["link"]
                if parent_name not in body_to_index:
                    continue
                joint_type = element.attrib.get("type", "fixed")
                name = element.attrib["name"]
                if joint_type not in ("fixed", "revolute", "continuous"):
                    raise ValueError(f"Unsupported URDF joint type '{joint_type}' for {name}.")
                origin = element.find("origin")
                axis = element.find("axis")
                body_to_index[child_name] = len(body_names)
                body_names.append(child_name)
                parent.append(body_to_index[parent_name])
                joint_index.append(-1 if joint_type == "fixed" else joint_to_index[name])
                axes.append(_vec(None if axis is None else axis.attrib.get("xyz"), (0.0, 0.0, 1.0)))
                offsets.append(_vec(None if origin is None else origin.attrib.get("xyz"), (0.0, 0.0, 0.0)))
                offset_rpy.append(_vec(None if origin is None else origin.attrib.get("rpy"), (0.0, 0.0, 0.0)))
                com_offsets.append(link_com_offsets.get(child_name, [0.0, 0.0, 0.0]))
                pending.remove(element)
                progressed = True
            if not progressed:
                unresolved = [joint.attrib["name"] for joint in pending]
                raise ValueError(f"URDF tree cannot be resolved from root '{root_body_name}': {unresolved}")

        missing = set(tracked_body_names).difference(body_to_index)
        if missing:
            raise ValueError(f"Tracked bodies missing from URDF: {sorted(missing)}")
        self.body_names = body_names
        self.parent = parent
        self.joint_index = joint_index
        self.axes = torch.tensor(axes, dtype=torch.float32, device=device)
        self.offsets = torch.tensor(offsets, dtype=torch.float32, device=device)
        self.offset_rotations = rpy_to_mat(torch.tensor(offset_rpy, dtype=torch.float32, device=device))
        self.com_offsets = torch.tensor(com_offsets, dtype=torch.float32, device=device)
        self.tracked_indices = [body_to_index[name] for name in tracked_body_names]

    def forward(self, p, rotation, v, w, q, qd):
        """Return link-frame poses and PhysX-compatible center-of-mass velocities."""
        count = len(self.body_names)
        positions, rotations = [None] * count, [None] * count
        origin_linear_velocities = [None] * count
        linear_velocities, angular_velocities = [None] * count, [None] * count
        positions[0], rotations[0] = p, rotation
        root_com_displacement = (rotation @ self.com_offsets[0].unsqueeze(-1)).squeeze(-1)
        origin_linear_velocities[0] = v - torch.cross(w, root_com_displacement, dim=-1)
        linear_velocities[0], angular_velocities[0] = v, w
        for index in range(1, count):
            parent = self.parent[index]
            parent_rotation = rotations[parent]
            displacement = (parent_rotation @ self.offsets[index].unsqueeze(-1)).squeeze(-1)
            position = positions[parent] + displacement
            offset_rotation = parent_rotation @ self.offset_rotations[index]
            joint = self.joint_index[index]
            if joint >= 0:
                child_rotation = offset_rotation @ axis_angle_to_mat(self.axes[index], q[..., joint])
                axis_world = (offset_rotation @ self.axes[index].unsqueeze(-1)).squeeze(-1)
                angular_velocity = angular_velocities[parent] + axis_world * qd[..., joint : joint + 1]
            else:
                child_rotation = offset_rotation
                angular_velocity = angular_velocities[parent]
            origin_linear_velocity = origin_linear_velocities[parent] + torch.cross(
                angular_velocities[parent], displacement, dim=-1
            )
            com_displacement = (child_rotation @ self.com_offsets[index].unsqueeze(-1)).squeeze(-1)
            linear_velocity = origin_linear_velocity + torch.cross(
                angular_velocity, com_displacement, dim=-1
            )
            positions[index], rotations[index] = position, child_rotation
            origin_linear_velocities[index] = origin_linear_velocity
            linear_velocities[index], angular_velocities[index] = linear_velocity, angular_velocity
        ids = self.tracked_indices
        return (
            torch.stack([positions[index] for index in ids], -2),
            torch.stack([rotations[index] for index in ids], -3),
            torch.stack([linear_velocities[index] for index in ids], -2),
            torch.stack([angular_velocities[index] for index in ids], -2),
        )
