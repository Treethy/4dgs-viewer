from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from viewer4d.core.model import GaussianFrame


class RenderMode(str, Enum):
    SPLAT = "Splat"
    ELLIPSOID = "Ellipsoid"
    CENTERS = "Centers"


@dataclass(frozen=True, slots=True)
class CentersUpdate:
    points: np.ndarray

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True, slots=True)
class EllipsoidUpdate:
    positions: np.ndarray
    quats: np.ndarray
    scales: np.ndarray

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


InspectionUpdate = CentersUpdate | EllipsoidUpdate


class InspectionScene:
    """Viser-native Gaussian inspection modes shared by 3D and 4D viewers.

    Sampling is deterministic over the original Gaussian indices. Therefore a
    point/ellipsoid keeps the same identity across time, while temporal opacity
    controls whether it is visible at the current frame.
    """

    def __init__(
        self,
        client: Any,
        *,
        num_gaussians: int,
        background: tuple[float, float, float],
        opacity_cutoff: float = 0.01,
        point_size: float = 0.002,
        point_sample_ratio: float = 0.10,
        ellipsoid_sample_ratio: float = 0.05,
        ellipsoid_sigma: float = 1.0,
    ) -> None:
        if num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive")
        if not 0.0 <= opacity_cutoff <= 1.0:
            raise ValueError("opacity_cutoff must be in [0,1]")
        if point_size <= 0.0:
            raise ValueError("point_size must be positive")
        if not 0.0 < point_sample_ratio <= 1.0:
            raise ValueError("point_sample_ratio must be in (0,1]")
        if not 0.0 < ellipsoid_sample_ratio <= 1.0:
            raise ValueError("ellipsoid_sample_ratio must be in (0,1]")
        if ellipsoid_sigma <= 0.0:
            raise ValueError("ellipsoid_sigma must be positive")

        self.client = client
        self.num_gaussians = int(num_gaussians)
        self.opacity_cutoff = float(opacity_cutoff)
        self.point_size = float(point_size)
        self.point_sample_ratio = float(point_sample_ratio)
        self.ellipsoid_sample_ratio = float(ellipsoid_sample_ratio)
        self.ellipsoid_sigma = float(ellipsoid_sigma)

        background_rgb = np.asarray(background, dtype=np.float32)
        if background_rgb.shape != (3,):
            raise ValueError("background must be an RGB tuple")
        self._background_rgb = np.clip(background_rgb, 0.0, 1.0)

        rng = np.random.default_rng(0)
        self._sample_order = np.ascontiguousarray(
            rng.permutation(self.num_gaussians),
            dtype=np.int64,
        )

        self._point_handle: Any | None = None
        self._ellipsoid_fill_handle: Any | None = None
        self._ellipsoid_outline_handle: Any | None = None
        self._mode = RenderMode.SPLAT
        self._lock = threading.RLock()

    @property
    def mode(self) -> RenderMode:
        return self._mode

    def set_mode(self, mode: RenderMode) -> None:
        if not isinstance(mode, RenderMode):
            mode = RenderMode(mode)

        with self._lock:
            if mode is RenderMode.SPLAT:
                self._hide_native_geometry()
                self._mode = mode
                self.client.flush()
                return

            self.client.scene.set_background_image(self._solid_background())

            if mode is RenderMode.CENTERS:
                if self._point_handle is not None:
                    self._point_handle.visible = True
                if self._ellipsoid_fill_handle is not None:
                    self._ellipsoid_fill_handle.visible = False
                if self._ellipsoid_outline_handle is not None:
                    self._ellipsoid_outline_handle.visible = False

            elif mode is RenderMode.ELLIPSOID:
                if self._point_handle is not None:
                    self._point_handle.visible = False
                if self._ellipsoid_fill_handle is not None:
                    self._ellipsoid_fill_handle.visible = True
                if self._ellipsoid_outline_handle is not None:
                    self._ellipsoid_outline_handle.visible = True

            self._mode = mode
            self.client.flush()

    def set_point_size(self, point_size: float) -> None:
        if point_size <= 0.0:
            raise ValueError("point_size must be positive")

        with self._lock:
            self.point_size = float(point_size)
            if self._point_handle is not None:
                self._point_handle.point_size = self.point_size
                self.client.flush()

    def set_point_sample_ratio(self, ratio: float) -> None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError("point sample ratio must be in (0,1]")
        with self._lock:
            self.point_sample_ratio = float(ratio)

    def set_ellipsoid_sample_ratio(self, ratio: float) -> None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ellipsoid sample ratio must be in (0,1]")
        with self._lock:
            self.ellipsoid_sample_ratio = float(ratio)

    def prepare_update(
        self,
        frame: GaussianFrame,
        mode: RenderMode,
    ) -> InspectionUpdate | None:
        """Prepare CPU geometry without mutating the Viser scene.

        Splitting preparation from application lets the 4D worker discard a
        stale frame before it touches browser-visible scene state.
        """

        frame.validate()
        if frame.num_gaussians != self.num_gaussians:
            raise ValueError(
                "inspection frame Gaussian count changed: "
                f"expected {self.num_gaussians}, got {frame.num_gaussians}"
            )

        if mode is RenderMode.SPLAT:
            return None

        with self._lock:
            if mode is RenderMode.CENTERS:
                indices = self._sample_indices(self.point_sample_ratio)
                selected = torch.as_tensor(indices, device=frame.device)
                opacity = frame.opacities.index_select(0, selected)
                active = opacity > self.opacity_cutoff
                selected = selected[active]
                points = frame.means.index_select(0, selected)

                return CentersUpdate(
                    points=np.ascontiguousarray(
                        points.detach().float().cpu().numpy(),
                        dtype=np.float32,
                    )
                )

            if mode is RenderMode.ELLIPSOID:
                indices = self._sample_indices(self.ellipsoid_sample_ratio)
                selected = torch.as_tensor(indices, device=frame.device)
                opacity = frame.opacities.index_select(0, selected)
                active = opacity > self.opacity_cutoff
                selected = selected[active]

                positions = frame.means.index_select(0, selected)
                scales = frame.scales.index_select(0, selected)
                quats = frame.quats.index_select(0, selected)
                quats = torch.nn.functional.normalize(quats, dim=-1)

                return EllipsoidUpdate(
                    positions=np.ascontiguousarray(
                        positions.detach().float().cpu().numpy(),
                        dtype=np.float32,
                    ),
                    quats=np.ascontiguousarray(
                        quats.detach().float().cpu().numpy(),
                        dtype=np.float32,
                    ),
                    scales=np.ascontiguousarray(
                        scales.detach().float().cpu().numpy()
                        * self.ellipsoid_sigma,
                        dtype=np.float32,
                    ),
                )

        raise ValueError(f"unsupported render mode: {mode}")

    def apply_update(self, update: InspectionUpdate | None) -> int:
        if update is None:
            return 0

        with self._lock:
            if isinstance(update, CentersUpdate):
                self._apply_centers(update)
                return update.count

            if isinstance(update, EllipsoidUpdate):
                self._apply_ellipsoids(update)
                return update.count

        raise TypeError(type(update).__name__)

    def _apply_centers(self, update: CentersUpdate) -> None:
        if self._point_handle is None:
            self._point_handle = self.client.scene.add_point_cloud(
                "/viewer4d/centers",
                points=update.points,
                colors=(225, 225, 225),
                point_size=self.point_size,
                point_shape="circle",
                point_shading="flat",
                precision="float32",
                visible=self._mode is RenderMode.CENTERS,
            )
        else:
            self._point_handle.points = update.points
            self._point_handle.visible = self._mode is RenderMode.CENTERS

        self.client.flush()

    def _apply_ellipsoids(self, update: EllipsoidUpdate) -> None:
        if update.count == 0:
            if self._ellipsoid_fill_handle is not None:
                self._ellipsoid_fill_handle.visible = False
            if self._ellipsoid_outline_handle is not None:
                self._ellipsoid_outline_handle.visible = False
            self.client.flush()
            return

        if self._ellipsoid_fill_handle is None:
            vertices, faces = _unit_icosphere(subdivisions=1)

            self._ellipsoid_fill_handle = (
                self.client.scene.add_batched_meshes_simple(
                    "/viewer4d/ellipsoids/fill",
                    vertices=vertices,
                    faces=faces,
                    batched_positions=update.positions,
                    batched_wxyzs=update.quats,
                    batched_scales=update.scales,
                    batched_colors=(205, 205, 205),
                    lod="auto",
                    wireframe=False,
                    material="toon5",
                    flat_shading=False,
                    side="front",
                    cast_shadow=False,
                    receive_shadow=False,
                    visible=self._mode is RenderMode.ELLIPSOID,
                )
            )

            self._ellipsoid_outline_handle = (
                self.client.scene.add_batched_meshes_simple(
                    "/viewer4d/ellipsoids/outline",
                    vertices=vertices,
                    faces=faces,
                    batched_positions=update.positions,
                    batched_wxyzs=update.quats,
                    batched_scales=np.ascontiguousarray(
                        update.scales * 1.035,
                        dtype=np.float32,
                    ),
                    batched_colors=(45, 45, 45),
                    lod="auto",
                    wireframe=False,
                    material="standard",
                    flat_shading=False,
                    side="back",
                    cast_shadow=False,
                    receive_shadow=False,
                    visible=self._mode is RenderMode.ELLIPSOID,
                )
            )
        else:
            self._ellipsoid_fill_handle.batched_positions = update.positions
            self._ellipsoid_fill_handle.batched_wxyzs = update.quats
            self._ellipsoid_fill_handle.batched_scales = update.scales
            self._ellipsoid_fill_handle.visible = (
                self._mode is RenderMode.ELLIPSOID
            )

            assert self._ellipsoid_outline_handle is not None
            self._ellipsoid_outline_handle.batched_positions = update.positions
            self._ellipsoid_outline_handle.batched_wxyzs = update.quats
            self._ellipsoid_outline_handle.batched_scales = np.ascontiguousarray(
                update.scales * 1.035,
                dtype=np.float32,
            )
            self._ellipsoid_outline_handle.visible = (
                self._mode is RenderMode.ELLIPSOID
            )

        self.client.flush()

    def _hide_native_geometry(self) -> None:
        if self._point_handle is not None:
            self._point_handle.visible = False
        if self._ellipsoid_fill_handle is not None:
            self._ellipsoid_fill_handle.visible = False
        if self._ellipsoid_outline_handle is not None:
            self._ellipsoid_outline_handle.visible = False

    def _sample_indices(self, ratio: float) -> np.ndarray:
        count = max(
            1,
            min(
                self.num_gaussians,
                int(round(self.num_gaussians * ratio)),
            ),
        )
        return self._sample_order[:count]

    def _solid_background(self) -> np.ndarray:
        rgb = np.round(self._background_rgb * 255.0).astype(np.uint8)
        return rgb.reshape(1, 1, 3)


def estimate_default_point_size(frame: GaussianFrame) -> float:
    """Estimate a small world-space point size for center visualization."""

    with torch.no_grad():
        means = frame.means.detach().float().cpu()
        scales = frame.scales.detach().float().cpu()
        opacities = frame.opacities.detach().float().cpu()

        finite = (
            torch.isfinite(means).all(dim=1)
            & torch.isfinite(scales).all(dim=1)
            & torch.isfinite(opacities)
        )
        active = finite & (opacities > 0.01)
        if int(active.sum().item()) < 10:
            active = finite

        means = means[active]
        scales = scales[active]
        if means.shape[0] == 0:
            return 1e-4

        if means.shape[0] > 200_000:
            idx = torch.linspace(
                0,
                means.shape[0] - 1,
                steps=200_000,
            ).long()
            means = means.index_select(0, idx)
            scales = scales.index_select(0, idx)

        center = means.median(dim=0).values
        radius = torch.linalg.vector_norm(means - center, dim=1)
        scene_radius = float(torch.quantile(radius, 0.90).item())
        typical_scale = float(scales.max(dim=1).values.median().item())

    return max(
        typical_scale * 0.08,
        scene_radius * 8e-5,
        1e-6,
    )


def _unit_icosphere(
    subdivisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    if subdivisions < 0:
        raise ValueError("subdivisions must be non-negative")

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices: list[np.ndarray] = [
        np.array(v, dtype=np.float64)
        for v in (
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        )
    ]
    vertices = [v / np.linalg.norm(v) for v in vertices]

    faces: list[tuple[int, int, int]] = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new_faces: list[tuple[int, int, int]] = []

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            cached = cache.get(key)
            if cached is not None:
                return cached

            v = vertices[i] + vertices[j]
            v /= np.linalg.norm(v)
            index = len(vertices)
            vertices.append(v)
            cache[key] = index
            return index

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend(
                [
                    (a, ab, ca),
                    (b, bc, ab),
                    (c, ca, bc),
                    (ab, bc, ca),
                ]
            )

        faces = new_faces

    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
    )
