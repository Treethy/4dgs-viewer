from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from viewer4d.core.model import AnytimeGS
from viewer4d.visualization.selection.inspector import evaluate_gaussian_subset
from viewer4d.visualization.selection.state import SelectionMode


class SelectionHighlight:
    """Client-local Viser overlay for the currently selected Gaussians."""

    def __init__(
        self,
        client: Any,
        *,
        base_point_size: float,
        color: tuple[int, int, int] = (255, 190, 0),
        opacity_cutoff: float = 0.01,
    ) -> None:
        if base_point_size <= 0.0:
            raise ValueError("base_point_size must be positive")
        if not 0.0 <= opacity_cutoff <= 1.0:
            raise ValueError("opacity_cutoff must be in [0,1]")

        self.client = client
        self.base_point_size = float(base_point_size)
        self.color = tuple(int(channel) for channel in color)
        self.opacity_cutoff = float(opacity_cutoff)

        self._handle: Any | None = None
        self._visible = True

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            self._handle.visible = self._visible
            self.client.flush()

    def clear(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            self.client.flush()

    def update(
        self,
        model: AnytimeGS,
        *,
        indices: Iterable[int],
        normalized_time: float,
        mode: SelectionMode,
    ) -> int:
        index_tuple = tuple(int(index) for index in indices)
        if not index_tuple:
            self.clear()
            return 0

        snapshot = evaluate_gaussian_subset(model, index_tuple, normalized_time)
        active = snapshot.opacities >= self.opacity_cutoff
        points = snapshot.means[active]

        if points.numel() == 0:
            if self._handle is not None:
                self._handle.visible = False
                self.client.flush()
            return 0

        points_np = np.ascontiguousarray(
            points.detach().float().cpu().numpy(),
            dtype=np.float32,
        )

        point_size = self.base_point_size * (
            4.0 if mode is SelectionMode.SINGLE and len(index_tuple) == 1 else 2.0
        )

        if self._handle is None:
            self._handle = self.client.scene.add_point_cloud(
                "/viewer4d/selection/highlight",
                points=points_np,
                colors=self.color,
                point_size=point_size,
                point_shape="circle",
                point_shading="flat",
                precision="float32",
                visible=self._visible,
            )
        else:
            self._handle.points = points_np
            self._handle.point_size = point_size
            self._handle.visible = self._visible

        self.client.flush()
        return int(points_np.shape[0])
