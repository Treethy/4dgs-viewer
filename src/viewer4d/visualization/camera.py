from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from viewer4d.core.model import GaussianFrame


class ViserCameraLike(Protocol):
    wxyz: np.ndarray
    position: np.ndarray
    fov: float
    near: float
    far: float
    image_width: int
    image_height: int


@dataclass(frozen=True, slots=True)
class CameraState:
    """Immutable snapshot of one Viser/OpenCV camera."""

    wxyz: np.ndarray
    position: np.ndarray
    fov: float
    width: int
    height: int
    near: float
    far: float


@dataclass(frozen=True, slots=True)
class RenderCamera:
    """Pinhole camera tensors ready for gsplat."""

    viewmat: Tensor
    K: Tensor
    width: int
    height: int
    near: float
    far: float


@dataclass(frozen=True, slots=True)
class SceneBounds:
    center: np.ndarray
    radius: float


@dataclass(frozen=True, slots=True)
class InitialCameraPose:
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    near: float
    far: float


def capture_viser_camera(camera: ViserCameraLike) -> CameraState:
    """Copy a mutable Viser camera handle into an immutable snapshot."""

    width = int(camera.image_width)
    height = int(camera.image_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"camera canvas is not ready: {width}x{height}")
    return CameraState(
        wxyz=np.asarray(camera.wxyz, dtype=np.float64).copy(),
        position=np.asarray(camera.position, dtype=np.float64).copy(),
        fov=float(camera.fov),
        width=width,
        height=height,
        near=max(float(camera.near), 1e-6),
        far=max(float(camera.far), float(camera.near) + 1e-6),
    )


def camera_state_to_render_camera(
    state: CameraState,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    max_width: int | None = None,
    max_height: int | None = None,
) -> RenderCamera:
    """Convert Viser's OpenCV camera-to-world pose to gsplat world-to-camera."""

    width, height = fit_render_size(
        state.width,
        state.height,
        max_width=max_width,
        max_height=max_height,
    )
    rotation_c2w = quaternion_wxyz_to_matrix(state.wxyz)
    translation_c2w = state.position
    rotation_w2c = rotation_c2w.T
    translation_w2c = -(rotation_w2c @ translation_c2w)

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rotation_w2c.astype(np.float32)
    viewmat[:3, 3] = translation_w2c.astype(np.float32)
    K = pinhole_intrinsics(width=width, height=height, vertical_fov=state.fov)

    return RenderCamera(
        viewmat=torch.as_tensor(viewmat, device=device, dtype=dtype),
        K=torch.as_tensor(K, device=device, dtype=dtype),
        width=width,
        height=height,
        near=state.near,
        far=state.far,
    )


def pinhole_intrinsics(
    *,
    width: int,
    height: int,
    vertical_fov: float,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0.0 < vertical_fov < math.pi:
        raise ValueError("vertical_fov must be in (0, pi)")
    focal = 0.5 * height / math.tan(0.5 * vertical_fov)
    return np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def fit_render_size(
    width: int,
    height: int,
    *,
    max_width: int | None = None,
    max_height: int | None = None,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    scale = 1.0
    if max_width is not None:
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        scale = min(scale, max_width / width)
    if max_height is not None:
        if max_height <= 0:
            raise ValueError("max_height must be positive")
        scale = min(scale, max_height / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def quaternion_wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def estimate_scene_bounds(
    frame: GaussianFrame,
    *,
    quantile: float = 0.95,
    max_samples: int = 200_000,
) -> SceneBounds:
    """Estimate robust bounds without copying every Gaussian when scenes are large."""

    if not 0.5 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0.5, 1.0]")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    means = frame.means.detach()
    if means.shape[0] > max_samples:
        step = max(1, means.shape[0] // max_samples)
        means = means[::step][:max_samples]
    means_cpu = means.float().cpu()
    finite = torch.isfinite(means_cpu).all(dim=1)
    means_cpu = means_cpu[finite]
    if means_cpu.numel() == 0:
        raise ValueError("cannot estimate bounds: no finite Gaussian centers")

    center_tensor = means_cpu.median(dim=0).values
    distances = torch.linalg.vector_norm(means_cpu - center_tensor, dim=1)
    radius = float(torch.quantile(distances, quantile).item())
    if not math.isfinite(radius) or radius <= 1e-6:
        radius = float(distances.max().item())
    if not math.isfinite(radius) or radius <= 1e-6:
        radius = 1.0
    return SceneBounds(
        center=center_tensor.numpy().astype(np.float64),
        radius=radius,
    )


def initial_camera_from_bounds(bounds: SceneBounds) -> InitialCameraPose:
    radius = max(float(bounds.radius), 1e-3)
    center = np.asarray(bounds.center, dtype=np.float64)
    position = center + np.array([0.0, -2.5 * radius, 0.65 * radius])
    return InitialCameraPose(
        position=position,
        look_at=center.copy(),
        up=np.array([0.0, 0.0, 1.0]),
        near=max(radius * 0.005, 1e-4),
        far=max(radius * 20.0, 10.0),
    )
