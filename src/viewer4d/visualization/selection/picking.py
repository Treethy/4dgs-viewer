from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from viewer4d.core.model import AnytimeGS
from viewer4d.visualization.camera import (
    ViserCameraSnapshot,
    project_world_to_screen,
)
from viewer4d.visualization.selection.inspector import evaluate_gaussian_subset


def pick_gaussian(
    model: AnytimeGS,
    *,
    normalized_time: float,
    camera: ViserCameraSnapshot,
    screen_pos: Sequence[float],
    max_pixel_distance: float = 12.0,
    opacity_cutoff: float = 0.01,
) -> int | None:
    """Pick the nearest visible Gaussian center around one screen click."""

    if len(screen_pos) != 2:
        raise ValueError("screen_pos must contain two normalized coordinates")
    if max_pixel_distance <= 0.0:
        raise ValueError("max_pixel_distance must be positive")
    if not 0.0 <= opacity_cutoff <= 1.0:
        raise ValueError("opacity_cutoff must be in [0,1]")

    indices = torch.arange(model.num_gaussians, device=model.device)
    snapshot = evaluate_gaussian_subset(model, indices, normalized_time)
    projection = project_world_to_screen(snapshot.means, camera)

    width = max(int(camera.image_width), 1)
    height = max(int(camera.image_height), 1)
    click = torch.as_tensor(
        (float(screen_pos[0]), float(screen_pos[1])),
        device=model.device,
        dtype=snapshot.means.dtype,
    )
    pixel_scale = torch.as_tensor(
        (float(width), float(height)),
        device=model.device,
        dtype=snapshot.means.dtype,
    )

    delta_px = (projection.xy - click[None, :]) * pixel_scale[None, :]
    distance_sq = delta_px.square().sum(dim=-1)

    valid = (
        projection.inside_viewport
        & (snapshot.opacities >= float(opacity_cutoff))
        & torch.isfinite(distance_sq)
        & (distance_sq <= float(max_pixel_distance) ** 2)
    )
    candidate_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return None

    candidate_distance = distance_sq.index_select(0, candidate_indices)
    best_distance = torch.min(candidate_distance)

    # Centers that are effectively tied in screen space are resolved by depth.
    tied = candidate_indices[candidate_distance <= best_distance + 1.0]
    tied_depth = projection.depth.index_select(0, tied)
    best_local = torch.argmin(tied_depth)
    return int(tied[best_local].item())


def select_gaussians_in_rect(
    model: AnytimeGS,
    *,
    normalized_time: float,
    camera: ViserCameraSnapshot,
    screen_min: Sequence[float],
    screen_max: Sequence[float],
    opacity_cutoff: float = 0.01,
    surface_depth: np.ndarray | torch.Tensor | None = None,
    surface_depth_tolerance: float = 0.05,
    depth_neighborhood_radius: int = 1,
) -> tuple[int, ...]:
    """Select visible centers inside a box, optionally near the rendered surface.

    ``surface_depth`` is an expected camera-z depth map rendered from the same
    frame/camera used for the displayed Splat image. When provided, a Gaussian
    center is kept only when its camera-z depth lies within the requested
    relative tolerance of the local surface depth. A 3x3 neighborhood is used by
    default so isolated depth holes do not make selection flicker.
    """

    if len(screen_min) != 2 or len(screen_max) != 2:
        raise ValueError("screen_min and screen_max must each contain two values")
    if not 0.0 <= opacity_cutoff <= 1.0:
        raise ValueError("opacity_cutoff must be in [0,1]")
    if surface_depth_tolerance < 0.0:
        raise ValueError("surface_depth_tolerance must be non-negative")
    if depth_neighborhood_radius < 0:
        raise ValueError("depth_neighborhood_radius must be non-negative")

    x0 = min(float(screen_min[0]), float(screen_max[0]))
    y0 = min(float(screen_min[1]), float(screen_max[1]))
    x1 = max(float(screen_min[0]), float(screen_max[0]))
    y1 = max(float(screen_min[1]), float(screen_max[1]))

    indices = torch.arange(model.num_gaussians, device=model.device)
    snapshot = evaluate_gaussian_subset(model, indices, normalized_time)
    projection = project_world_to_screen(snapshot.means, camera)

    xy = projection.xy
    selected = (
        projection.inside_viewport
        & (snapshot.opacities >= float(opacity_cutoff))
        & (xy[:, 0] >= x0)
        & (xy[:, 0] <= x1)
        & (xy[:, 1] >= y0)
        & (xy[:, 1] <= y1)
    )

    result = torch.nonzero(selected, as_tuple=False).flatten()
    if result.numel() == 0 or surface_depth is None:
        return tuple(int(index) for index in result.detach().cpu().tolist())

    depth_map = torch.as_tensor(
        surface_depth,
        device=model.device,
        dtype=snapshot.means.dtype,
    )
    if depth_map.ndim != 2:
        raise ValueError(
            f"surface_depth must have shape [H,W], got {tuple(depth_map.shape)}"
        )

    height, width = int(depth_map.shape[0]), int(depth_map.shape[1])
    if height <= 0 or width <= 0:
        return ()

    candidate_xy = xy.index_select(0, result)
    candidate_depth = projection.depth.index_select(0, result)

    px = torch.round(candidate_xy[:, 0] * float(max(width - 1, 1))).long()
    py = torch.round(candidate_xy[:, 1] * float(max(height - 1, 1))).long()
    px = px.clamp(0, width - 1)
    py = py.clamp(0, height - 1)

    radius = int(depth_neighborhood_radius)
    offsets = torch.arange(-radius, radius + 1, device=model.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, -1)
    oy = oy.reshape(1, -1)

    sx = (px[:, None] + ox).clamp(0, width - 1)
    sy = (py[:, None] + oy).clamp(0, height - 1)
    samples = depth_map[sy, sx]

    valid_samples = torch.isfinite(samples) & (samples > 0.0)
    samples = torch.where(
        valid_samples,
        samples,
        torch.full_like(samples, float("nan")),
    )
    local_surface = torch.nanmedian(samples, dim=1).values

    valid_surface = torch.isfinite(local_surface) & (local_surface > 0.0)
    relative_error = torch.abs(candidate_depth - local_surface) / local_surface.clamp_min(
        1e-6
    )
    surface_consistent = valid_surface & (
        relative_error <= float(surface_depth_tolerance)
    )

    result = result[surface_consistent]
    return tuple(int(index) for index in result.detach().cpu().tolist())