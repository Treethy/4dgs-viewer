from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from viewer4d.core.model import GaussianFrame
from viewer4d.visualization.camera import RenderCamera


@dataclass(frozen=True, slots=True)
class GaussianOverlayRequest:
    """Image-space overlay drawn from gsplat's own 2D projection metadata.

    This keeps selection markers synchronized with the exact camera used for the
    background Splat render. ``indices`` always contains original Gaussian IDs.
    """

    indices: tuple[int, ...]
    draw_centers: bool = True
    draw_ellipses: bool = False
    max_ellipses: int = 1
    opacity_cutoff: float = 0.01
    center_radius_px: float = 4.0
    center_radius_from_projected: bool = False
    center_radius_scale: float = 0.15
    center_radius_min_px: float = 0.35
    center_radius_max_px: float = 1.25
    center_min_projected_radius_px: float = 0.0
    ellipse_sigma: float = 2.0
    line_width_px: int = 2
    color: tuple[int, int, int] = (255, 190, 0)

    def __post_init__(self) -> None:
        if self.max_ellipses < 0:
            raise ValueError("max_ellipses must be non-negative")
        if not 0.0 <= self.opacity_cutoff <= 1.0:
            raise ValueError("opacity_cutoff must be in [0,1]")
        if self.center_radius_px <= 0:
            raise ValueError("center_radius_px must be positive")
        if self.center_radius_scale <= 0:
            raise ValueError("center_radius_scale must be positive")
        if self.center_radius_min_px < 0:
            raise ValueError("center_radius_min_px must be non-negative")
        if self.center_radius_max_px <= 0:
            raise ValueError("center_radius_max_px must be positive")
        if self.center_radius_min_px > self.center_radius_max_px:
            raise ValueError(
                "center_radius_min_px cannot exceed center_radius_max_px"
            )
        if self.center_min_projected_radius_px < 0:
            raise ValueError(
                "center_min_projected_radius_px must be non-negative"
            )
        if self.ellipse_sigma <= 0.0:
            raise ValueError("ellipse_sigma must be positive")
        if self.line_width_px <= 0:
            raise ValueError("line_width_px must be positive")


@dataclass(frozen=True, slots=True)
class RenderOutput:
    image: np.ndarray
    depth: np.ndarray | None = None


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
        """Return a contiguous frame on the renderer device."""

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
        *,
        overlay: GaussianOverlayRequest | None = None,
        radius_clip: float | None = None,
    ) -> np.ndarray:
        """Render RGB, optionally drawing synchronized Gaussian selection overlay."""

        return self.render_with_aux(
            frame,
            camera,
            overlay=overlay,
            return_depth=False,
            radius_clip=radius_clip,
        ).image

    @torch.inference_mode()
    def render_with_aux(
        self,
        frame: GaussianFrame,
        camera: RenderCamera,
        *,
        overlay: GaussianOverlayRequest | None = None,
        return_depth: bool = False,
        radius_clip: float | None = None,
    ) -> RenderOutput:
        try:
            from gsplat.rendering import rasterization
        except ImportError as error:
            raise RuntimeError("GsplatRenderer requires gsplat") from error

        frame = self.prepare_frame(frame)
        dtype = frame.means.dtype
        view_matrix = camera.view_matrix.to(device=self.device, dtype=dtype)
        K = camera.K.to(device=self.device, dtype=dtype)
        sh_degree = infer_sh_degree(frame.sh.shape[1])
        clip = self.radius_clip if radius_clip is None else float(radius_clip)
        render_mode = "RGB+ED" if return_depth else "RGB"

        with self._render_lock:
            rendered, alpha, meta = rasterization(
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
                radius_clip=clip,
                sh_degree=sh_degree,
                packed=self.packed,
                backgrounds=None,
                render_mode=render_mode,
                rasterize_mode=self.rasterize_mode,
            )

        rgb = rendered[0, ..., :3].clamp(0.0, 1.0)
        alpha0 = alpha[0, ..., :1].clamp(0.0, 1.0)
        background = torch.as_tensor(
            self.background,
            device=self.device,
            dtype=rgb.dtype,
        ).view(1, 1, 3)
        rgb = rgb + (1.0 - alpha0) * background

        image = np.ascontiguousarray(
            rgb.mul(255.0).round().to(torch.uint8).cpu().numpy()
        )

        if overlay is not None and overlay.indices:
            image = _draw_gaussian_overlay(
                image,
                frame=frame,
                meta=meta,
                request=overlay,
                packed=self.packed,
            )

        depth_np: np.ndarray | None = None
        if return_depth:
            depth = rendered[0, ..., -1]
            valid = alpha[0, ..., 0] > 1e-4
            depth = torch.where(valid, depth, torch.full_like(depth, float("nan")))
            depth_np = np.ascontiguousarray(
                depth.detach().float().cpu().numpy(), dtype=np.float32
            )

        return RenderOutput(image=image, depth=depth_np)

    @torch.inference_mode()
    def render_depth(
        self,
        frame: GaussianFrame,
        camera: RenderCamera,
        *,
        radius_clip: float | None = None,
    ) -> np.ndarray:
        """Render expected camera-z depth without evaluating spherical harmonics."""

        try:
            from gsplat.rendering import rasterization
        except ImportError as error:
            raise RuntimeError("GsplatRenderer requires gsplat") from error

        frame = self.prepare_frame(frame)
        dtype = frame.means.dtype
        view_matrix = camera.view_matrix.to(device=self.device, dtype=dtype)
        K = camera.K.to(device=self.device, dtype=dtype)
        clip = self.radius_clip if radius_clip is None else float(radius_clip)

        # For D/ED modes gsplat replaces `colors` with projected depths before
        # rasterization. Passing one cheap post-activation channel avoids SH work.
        dummy_colors = frame.opacities[:, None].contiguous()

        with self._render_lock:
            rendered, alpha, _ = rasterization(
                means=frame.means,
                quats=frame.quats,
                scales=frame.scales,
                opacities=frame.opacities,
                colors=dummy_colors,
                viewmats=view_matrix[None],
                Ks=K[None],
                width=camera.width,
                height=camera.height,
                near_plane=camera.near,
                far_plane=camera.far,
                radius_clip=clip,
                sh_degree=None,
                packed=self.packed,
                backgrounds=None,
                render_mode="ED",
                rasterize_mode=self.rasterize_mode,
            )

        depth = rendered[0, ..., 0]
        valid = alpha[0, ..., 0] > 1e-4
        depth = torch.where(valid, depth, torch.full_like(depth, float("nan")))
        return np.ascontiguousarray(
            depth.detach().float().cpu().numpy(), dtype=np.float32
        )


def _draw_gaussian_overlay(
    image: np.ndarray,
    *,
    frame: GaussianFrame,
    meta: dict[str, Any],
    request: GaussianOverlayRequest,
    packed: bool,
) -> np.ndarray:
    """Draw selected projected centers/ellipses on an RGB uint8 image."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Selection overlay requires Pillow") from error

    selected = torch.as_tensor(
        request.indices,
        device=frame.device,
        dtype=torch.long,
    )
    if selected.numel() == 0:
        return image

    selected = selected[(selected >= 0) & (selected < frame.num_gaussians)]
    if selected.numel() == 0:
        return image

    selected = selected[frame.opacities.index_select(0, selected) >= request.opacity_cutoff]
    if selected.numel() == 0:
        return image

    means2d, conics, radii, gaussian_ids = _projection_meta_for_single_camera(
        meta,
        packed=packed,
        num_gaussians=frame.num_gaussians,
        device=frame.device,
    )
    if gaussian_ids.numel() == 0:
        return image

    selected_sorted = torch.unique(selected, sorted=True)
    positions = torch.searchsorted(selected_sorted, gaussian_ids)
    valid_pos = positions < selected_sorted.numel()
    safe_positions = positions.clamp(max=max(selected_sorted.numel() - 1, 0))
    matched = valid_pos & (selected_sorted[safe_positions] == gaussian_ids)
    matched &= radii > 0
    visible_rows = torch.nonzero(matched, as_tuple=False).flatten()
    if visible_rows.numel() == 0:
        return image

    center_xy = means2d.index_select(0, visible_rows).detach().float().cpu().numpy()
    visible_radii = radii.index_select(0, visible_rows)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    color = tuple(int(v) for v in request.color)

    if request.draw_ellipses and request.max_ellipses > 0:
        k = min(int(request.max_ellipses), int(visible_rows.numel()))
        if k > 0:
            # For box selection, prioritize splats with larger visible footprint.
            top_local = torch.topk(visible_radii.float(), k=k, largest=True).indices
            ellipse_rows = visible_rows.index_select(0, top_local)
            ellipse_means = means2d.index_select(0, ellipse_rows).detach().float().cpu().numpy()
            ellipse_conics = conics.index_select(0, ellipse_rows).detach().float().cpu().numpy()
            ellipse_radii = radii.index_select(0, ellipse_rows).detach().float().cpu().numpy()

            for mean, conic, radius in zip(
                ellipse_means,
                ellipse_conics,
                ellipse_radii,
                strict=False,
            ):
                points = _ellipse_polyline_from_conic(
                    mean,
                    conic,
                    sigma=request.ellipse_sigma,
                    projected_radius=float(radius),
                )
                if points is None:
                    continue
                draw.line(
                    [tuple(map(float, point)) for point in points],
                    fill=color,
                    width=request.line_width_px,
                    joint="curve",
                )

    if request.draw_centers:
        # Single selection uses a fixed screen-space marker.  Box selection
        # can instead derive each marker radius from the Gaussian's own
        # projected footprint so markers naturally shrink as the camera moves
        # away.  Very small projected splats can be omitted from the overlay
        # while remaining part of the SelectionState / selected count.
        if request.center_radius_from_projected:
            center_radii = visible_radii.detach().float().cpu().numpy()
            for (x, y), projected_radius in zip(
                center_xy, center_radii, strict=False
            ):
                x = float(x)
                y = float(y)
                projected_radius = float(projected_radius)
                if not (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and math.isfinite(projected_radius)
                ):
                    continue
                if (
                    projected_radius
                    < request.center_min_projected_radius_px
                ):
                    continue

                r = float(
                    np.clip(
                        projected_radius * request.center_radius_scale,
                        request.center_radius_min_px,
                        request.center_radius_max_px,
                    )
                )
                draw.ellipse(
                    (x - r, y - r, x + r, y + r),
                    fill=color,
                )
        else:
            r = float(request.center_radius_px)
            for x, y in center_xy:
                x = float(x)
                y = float(y)
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                draw.ellipse(
                    (x - r, y - r, x + r, y + r),
                    fill=color,
                )

    return np.ascontiguousarray(np.asarray(pil_image, dtype=np.uint8))


def _projection_meta_for_single_camera(
    meta: dict[str, Any],
    *,
    packed: bool,
    num_gaussians: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means2d = meta["means2d"]
    conics = meta["conics"]
    radii = meta["radii"]

    # gsplat 1.5.x reports projected radii per axis: [..., 2].
    # For overlay visibility/ranking we only need one conservative scalar
    # support radius per projected Gaussian, so use max(rx, ry).
    if radii.shape[-1] == 2:
        scalar_radii = radii.amax(dim=-1)
    else:
        # Keep a defensive fallback for versions/configurations that expose
        # a scalar radius already. Do not flatten an arbitrary trailing axis.
        scalar_radii = radii.squeeze(-1) if radii.ndim > 1 and radii.shape[-1] == 1 else radii

    if packed:
        gaussian_ids = meta.get("gaussian_ids")
        if gaussian_ids is None:
            raise RuntimeError("gsplat packed metadata did not contain gaussian_ids")

        means2d = means2d.reshape(-1, 2)
        conics = conics.reshape(-1, 3)
        scalar_radii = scalar_radii.reshape(-1)
        gaussian_ids = gaussian_ids.reshape(-1).to(dtype=torch.long)

        nnz = int(gaussian_ids.numel())
        if (
            means2d.shape[0] != nnz
            or conics.shape[0] != nnz
            or scalar_radii.numel() != nnz
        ):
            raise RuntimeError(
                "inconsistent gsplat packed projection metadata: "
                f"gaussian_ids={nnz}, means2d={means2d.shape[0]}, "
                f"conics={conics.shape[0]}, radii={scalar_radii.numel()}"
            )
        return means2d, conics, scalar_radii, gaussian_ids

    # Single-camera unpacked projection metadata is [1, N, ...].
    means2d = means2d.reshape(-1, 2)
    conics = conics.reshape(-1, 3)
    scalar_radii = scalar_radii.reshape(-1)
    gaussian_ids = torch.arange(num_gaussians, device=device, dtype=torch.long)
    if (
        means2d.shape[0] != num_gaussians
        or conics.shape[0] != num_gaussians
        or scalar_radii.numel() != num_gaussians
    ):
        raise RuntimeError(
            "inconsistent gsplat unpacked projection metadata: "
            f"expected={num_gaussians}, means2d={means2d.shape[0]}, "
            f"conics={conics.shape[0]}, radii={scalar_radii.numel()}"
        )
    return means2d, conics, scalar_radii, gaussian_ids


def _ellipse_polyline_from_conic(
    mean: np.ndarray,
    conic: np.ndarray,
    *,
    sigma: float,
    projected_radius: float,
    samples: int = 48,
) -> np.ndarray | None:
    """Recover a 2D covariance ellipse from gsplat's inverse-covariance conic."""

    if mean.shape != (2,) or conic.shape != (3,):
        return None
    if not np.isfinite(mean).all() or not np.isfinite(conic).all():
        return None

    a, b, c = (float(v) for v in conic)
    determinant = a * c - b * b
    if not math.isfinite(determinant) or determinant <= 1e-12:
        return None

    covariance = np.array([[c, -b], [-b, a]], dtype=np.float64) / determinant
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
        return None

    axes = float(sigma) * np.sqrt(eigenvalues)
    # `radii` is a conservative projected support radius. Clamping keeps a
    # numerically noisy inverse-conic from drawing a huge screen-spanning line.
    if projected_radius > 0.0 and math.isfinite(projected_radius):
        axes = np.minimum(axes, max(projected_radius, 1.0))

    theta = np.linspace(0.0, 2.0 * math.pi, samples + 1, dtype=np.float64)
    circle = np.stack((np.cos(theta), np.sin(theta)), axis=0)
    ellipse = eigenvectors @ (axes[:, None] * circle)
    return (ellipse.T + mean[None, :]).astype(np.float32, copy=False)


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