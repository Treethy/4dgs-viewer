from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from viewer4d.core.model import GaussianFrame
from viewer4d.visualization.camera import RenderCamera

RasterizationFunction = Callable[..., tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]


class GsplatRenderer:
    """Server-side gsplat renderer for one already-evaluated static frame."""

    def __init__(
        self,
        *,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
        packed: bool = True,
        rasterize_mode: str = "classic",
        radius_clip: float = 0.0,
        rasterization_fn: RasterizationFunction | None = None,
    ) -> None:
        if len(background) != 3 or any(not 0.0 <= value <= 1.0 for value in background):
            raise ValueError("background must be three floats in [0, 1]")
        if rasterize_mode not in ("classic", "antialiased"):
            raise ValueError("rasterize_mode must be 'classic' or 'antialiased'")
        if radius_clip < 0:
            raise ValueError("radius_clip must be non-negative")
        self.background = tuple(float(value) for value in background)
        self.packed = bool(packed)
        self.rasterize_mode = rasterize_mode
        self.radius_clip = float(radius_clip)
        self._rasterization_fn = rasterization_fn
        self._render_lock = threading.Lock()

    @torch.inference_mode()
    def render(self, frame: GaussianFrame, camera: RenderCamera) -> np.ndarray:
        """Render an RGB uint8 image. Calls are serialized for CUDA safety."""

        frame.validate()
        device = frame.means.device
        if device.type != "cuda":
            raise RuntimeError(
                "gsplat mode requires GaussianFrame tensors on CUDA; "
                f"got {device}"
            )
        if camera.viewmat.device != device or camera.K.device != device:
            raise ValueError("camera tensors and GaussianFrame must share one device")

        sh_degree = infer_sh_degree(frame.sh_coeffs.shape[1])
        backgrounds = torch.tensor(
            [self.background],
            device=device,
            dtype=frame.means.dtype,
        )
        rasterization = self._rasterization_fn or _load_rasterization()

        with self._render_lock:
            rendered, _alphas, _info = rasterization(
                means=frame.means,
                quats=frame.quats,
                scales=frame.scales,
                opacities=frame.opacities,
                colors=frame.sh_coeffs,
                viewmats=camera.viewmat[None],
                Ks=camera.K[None],
                width=camera.width,
                height=camera.height,
                near_plane=camera.near,
                far_plane=camera.far,
                radius_clip=self.radius_clip,
                sh_degree=sh_degree,
                packed=self.packed,
                backgrounds=backgrounds,
                render_mode="RGB",
                rasterize_mode=self.rasterize_mode,
            )

        rgb = rendered[0, ..., :3].clamp(0.0, 1.0)
        return np.ascontiguousarray(
            (rgb * 255.0).round().to(torch.uint8).cpu().numpy()
        )


def infer_sh_degree(num_coefficients: int) -> int:
    if num_coefficients <= 0:
        raise ValueError("num_coefficients must be positive")
    root = math.isqrt(num_coefficients)
    if root * root != num_coefficients:
        raise ValueError(
            "SH coefficient count must be a perfect square, "
            f"got {num_coefficients}"
        )
    return root - 1


def _load_rasterization() -> RasterizationFunction:
    try:
        from gsplat.rendering import rasterization
    except ImportError as error:
        raise RuntimeError(
            "gsplat mode requires gsplat to be installed in the 4Dviewer environment"
        ) from error
    return rasterization
