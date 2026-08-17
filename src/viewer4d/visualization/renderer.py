from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
import torch

from viewer4d.core.model import GaussianFrame
from viewer4d.visualization.camera import RenderCamera


class GsplatRenderer:
    """gsplat renderer bound to one static GaussianFrame."""

    def __init__(
        self,
        frame: GaussianFrame,
        *,
        device: str | torch.device = "cuda",
        background: tuple[float, float, float] = (0.08, 0.08, 0.08),
        packed: bool = True,
        rasterize_mode: str = "classic",
        radius_clip: float = 0.0,
    ) -> None:
        frame.validate()

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        if len(background) != 3 or any(
            value < 0.0 or value > 1.0 for value in background
        ):
            raise ValueError("background must contain three values in [0,1]")

        self.frame = GaussianFrame(
            means=frame.means.detach().to(requested_device).contiguous(),
            scales=frame.scales.detach().to(requested_device).contiguous(),
            quats=frame.quats.detach().to(requested_device).contiguous(),
            opacities=frame.opacities.detach().to(requested_device).contiguous(),
            sh=frame.sh.detach().to(requested_device).contiguous(),
        )

        # Use the concrete tensor device (e.g. cuda:0), not torch.device("cuda").
        self.device = self.frame.means.device
        self.dtype = self.frame.means.dtype

        self.background = tuple(float(value) for value in background)
        self.packed = bool(packed)
        self.rasterize_mode = rasterize_mode
        self.radius_clip = float(radius_clip)
        self.sh_degree = infer_sh_degree(self.frame.sh.shape[1])

        self._render_lock = threading.Lock()

    @torch.inference_mode()
    def render(self, camera: RenderCamera) -> np.ndarray:
        try:
            from gsplat.rendering import rasterization
        except ImportError as error:
            raise RuntimeError("GsplatRenderer requires gsplat") from error

        # Camera matrices are tiny. Moving them defensively is cheaper and more
        # robust than rejecting equivalent devices such as "cuda" vs "cuda:0".
        view_matrix = camera.view_matrix.to(
            device=self.device,
            dtype=self.dtype,
        )
        K = camera.K.to(
            device=self.device,
            dtype=self.dtype,
        )

        with self._render_lock:
            rendered, alpha, _ = rasterization(
                means=self.frame.means,
                quats=self.frame.quats,
                scales=self.frame.scales,
                opacities=self.frame.opacities,
                colors=self.frame.sh,
                viewmats=view_matrix[None],
                Ks=K[None],
                width=camera.width,
                height=camera.height,
                near_plane=camera.near,
                far_plane=camera.far,
                radius_clip=self.radius_clip,
                sh_degree=self.sh_degree,
                packed=self.packed,
                backgrounds=None,
                render_mode="RGB",
                rasterize_mode=self.rasterize_mode,
            )

        rgb = rendered[0, ..., :3].clamp(0.0, 1.0)
        alpha = alpha[0, ..., :1].clamp(0.0, 1.0)

        background = torch.as_tensor(
            self.background,
            device=self.device,
            dtype=rgb.dtype,
        ).view(1, 1, 3)

        rgb = rgb + (1.0 - alpha) * background

        return np.ascontiguousarray(
            rgb.mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )


def infer_sh_degree(num_coefficients: int) -> int:
    if num_coefficients <= 0:
        raise ValueError("SH coefficient count must be positive")

    root = math.isqrt(num_coefficients)
    if root * root != num_coefficients:
        raise ValueError(
            f"SH coefficient count must be a perfect square, got {num_coefficients}"
        )

    return root - 1