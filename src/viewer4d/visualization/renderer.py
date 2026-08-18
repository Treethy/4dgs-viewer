from __future__ import annotations

import math
import threading

import numpy as np
import torch

from viewer4d.core.model import GaussianFrame
from viewer4d.visualization.camera import RenderCamera


class GsplatRenderer:
    """Stateless gsplat renderer for static or time-varying GaussianFrame data.

    A renderer owns CUDA/render configuration, but it does not own a current
    GaussianFrame. This is important for 4D playback: each render receives the
    frame corresponding to the requested time explicitly.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        background: tuple[float, float, float] = (0.08, 0.08, 0.08),
        packed: bool = True,
        rasterize_mode: str = "classic",
        radius_clip: float = 0.0,
    ) -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())

        if len(background) != 3 or any(
            value < 0.0 or value > 1.0 for value in background
        ):
            raise ValueError("background must contain three values in [0,1]")

        self.device = requested
        self.background = tuple(float(value) for value in background)
        self.packed = bool(packed)
        self.rasterize_mode = rasterize_mode
        self.radius_clip = float(radius_clip)
        self._render_lock = threading.Lock()

    def prepare_frame(self, frame: GaussianFrame) -> GaussianFrame:
        """Return a contiguous frame on the renderer device.

        If the input already lives on the correct device, no CPU↔GPU transfer
        is performed. Static viewers can call this once. AnytimeGS models used
        by the 4D viewer are moved to this device once at startup, so every
        frame produced by ``at_time()`` is already ready for rendering.
        """

        frame.validate()
        if frame.device == self.device:
            if all(
                tensor.is_contiguous()
                for tensor in (
                    frame.means,
                    frame.scales,
                    frame.quats,
                    frame.opacities,
                    frame.sh,
                )
            ):
                return frame

        return GaussianFrame(
            means=frame.means.detach().to(self.device).contiguous(),
            scales=frame.scales.detach().to(self.device).contiguous(),
            quats=frame.quats.detach().to(self.device).contiguous(),
            opacities=frame.opacities.detach().to(self.device).contiguous(),
            sh=frame.sh.detach().to(self.device).contiguous(),
        )

    @torch.inference_mode()
    def render(
        self,
        frame: GaussianFrame,
        camera: RenderCamera,
    ) -> np.ndarray:
        try:
            from gsplat.rendering import rasterization
        except ImportError as error:
            raise RuntimeError("GsplatRenderer requires gsplat") from error

        frame = self.prepare_frame(frame)
        dtype = frame.means.dtype

        view_matrix = camera.view_matrix.to(
            device=self.device,
            dtype=dtype,
        )
        K = camera.K.to(
            device=self.device,
            dtype=dtype,
        )

        sh_degree = infer_sh_degree(frame.sh.shape[1])

        with self._render_lock:
            rendered, alpha, _ = rasterization(
                means=frame.means,
                quats=frame.quats,
                scales=frame.scales,
                opacities=frame.opacities,
                colors=frame.sh,
                viewmats=view_matrix[None],
                Ks=K[None],
                width=camera.width,
                height=camera.height,
                near_plane=camera.near,
                far_plane=camera.far,
                radius_clip=self.radius_clip,
                sh_degree=sh_degree,
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
            "SH coefficient count must be a perfect square, "
            f"got {num_coefficients}"
        )

    return root - 1
