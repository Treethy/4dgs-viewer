from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RenderCamera:
    """The exact camera tensors consumed by gsplat."""

    view_matrix: Tensor
    K: Tensor
    width: int
    height: int
    near: float
    far: float


@dataclass(frozen=True, slots=True)
class ViserCameraSnapshot:
    """Immutable camera state copied out of a Viser CameraHandle."""

    wxyz: np.ndarray
    position: np.ndarray
    fov: float
    aspect: float
    image_width: int
    image_height: int
    near: float
    far: float


@dataclass(frozen=True, slots=True)
class ScreenProjection:
    """World points projected into Viser/OpenCV normalized screen coordinates."""

    xy: Tensor
    depth: Tensor
    in_front: Tensor
    inside_viewport: Tensor


def snapshot_viser_camera(camera: Any) -> ViserCameraSnapshot:
    """Copy all render-relevant Viser camera state.

    `aspect`, `image_width`, and `image_height` describe the browser canvas.
    They are essential because the gsplat result is used as a Viser background
    image and therefore must preserve the browser canvas aspect ratio.
    """

    near = max(float(camera.near), 1e-5)
    far = max(float(camera.far), near + 1e-3)

    image_width = max(int(camera.image_width), 0)
    image_height = max(int(camera.image_height), 0)

    aspect = float(camera.aspect)
    if not math.isfinite(aspect) or aspect <= 0.0:
        if image_width > 0 and image_height > 0:
            aspect = image_width / image_height
        else:
            aspect = 1.0

    return ViserCameraSnapshot(
        wxyz=np.asarray(camera.wxyz, dtype=np.float64).copy(),
        position=np.asarray(camera.position, dtype=np.float64).copy(),
        fov=float(camera.fov),
        aspect=aspect,
        image_width=image_width,
        image_height=image_height,
        near=near,
        far=far,
    )


def render_camera_from_viser(
    camera: ViserCameraSnapshot,
    *,
    width: int,
    height: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> RenderCamera:
    """Convert Viser's camera-to-world pose to a gsplat render camera.

    Viser reports a vertical FOV. With square pixels, fx == fy in pixel units.
    The horizontal FOV then follows naturally from `width / height`.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid render size: {width}x{height}")

    R_c2w = quaternion_wxyz_to_matrix(camera.wxyz)
    t_c2w = camera.position

    R_w2c = R_c2w.T
    t_w2c = -(R_w2c @ t_c2w)

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_w2c.astype(np.float32)
    w2c[:3, 3] = t_w2c.astype(np.float32)

    fy = 0.5 * height / math.tan(0.5 * camera.fov)
    fx = fy

    K = np.array(
        [
            [fx, 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return RenderCamera(
        view_matrix=torch.as_tensor(w2c, device=device, dtype=dtype),
        K=torch.as_tensor(K, device=device, dtype=dtype),
        width=width,
        height=height,
        near=camera.near,
        far=camera.far,
    )


def project_world_to_screen(
    points: Tensor,
    camera: ViserCameraSnapshot,
) -> ScreenProjection:
    """Project world-space points to normalized OpenCV screen coordinates.

    Returned ``xy`` uses the same convention as Viser scene pointer events:
    ``(0,0)`` is the upper-left corner and ``(1,1)`` is the lower-right.
    ``inside_viewport`` also requires the point to lie between the camera near
    and far planes.
    """

    if not isinstance(points, Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N,3], got {tuple(points.shape)}")
    if not points.is_floating_point():
        raise TypeError("points must use a floating dtype")

    dtype = points.dtype
    device = points.device

    R_c2w = torch.as_tensor(
        quaternion_wxyz_to_matrix(camera.wxyz),
        dtype=dtype,
        device=device,
    )
    position = torch.as_tensor(camera.position, dtype=dtype, device=device)

    # Viser camera convention: p_world = R_c2w @ p_camera + position.
    # For row-vector point storage this becomes p_camera = (p_world-t) @ R_c2w.
    camera_points = (points - position) @ R_c2w
    depth = camera_points[:, 2]

    safe_depth = torch.where(
        depth.abs() > torch.finfo(dtype).eps,
        depth,
        torch.ones_like(depth),
    )
    xy = camera_points[:, :2] / safe_depth[:, None]

    tan_half_fov = math.tan(0.5 * float(camera.fov))
    if not math.isfinite(tan_half_fov) or tan_half_fov <= 0.0:
        raise ValueError(f"invalid camera fov: {camera.fov}")

    xy = xy / tan_half_fov
    xy = xy.clone()
    xy[:, 0] = xy[:, 0] / float(camera.aspect)
    xy = (xy + 1.0) * 0.5

    in_front = (depth >= float(camera.near)) & (depth <= float(camera.far))
    inside = (
        in_front
        & torch.isfinite(xy).all(dim=-1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= 1.0)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= 1.0)
    )

    return ScreenProjection(
        xy=xy,
        depth=depth,
        in_front=in_front,
        inside_viewport=inside,
    )


def quaternion_wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(wxyz, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"wxyz must have shape (4,), got {q.shape}")

    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid camera quaternion")

    w, x, y, z = q / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
