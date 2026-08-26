from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from viewer4d.visualization.trajectory.data import GaussianTrajectory


@dataclass(frozen=True, slots=True)
class TrajectoryRenderStats:
    tracking_gaussians: int
    visible_trajectories: int
    line_segments: int
    current_centers: int


class TrajectoryScene:
    """Batched Viser-native 3D visualization for Gaussian trajectories."""

    def __init__(
        self,
        client: Any,
        *,
        base_point_size: float,
        line_color: tuple[int, int, int] = (64, 180, 255),
        center_color: tuple[int, int, int] = (255, 190, 0),
        line_thickness_px: float = 1.5,
    ) -> None:
        self.client = client
        self.base_point_size = float(base_point_size)
        self.line_color = tuple(int(value) for value in line_color)
        self.center_color = tuple(int(value) for value in center_color)
        self.line_thickness_px = float(line_thickness_px)

        self._trajectories: dict[int, GaussianTrajectory] = {}
        self._line_handle: Any | None = None
        self._center_handle: Any | None = None

    @property
    def has_trajectories(self) -> bool:
        return bool(self._trajectories)

    def set_trajectories(
        self,
        trajectories: Mapping[int, GaussianTrajectory],
    ) -> None:
        self._trajectories = dict(trajectories)
        if not self._trajectories:
            if self._line_handle is not None:
                self._line_handle.visible = False
            if self._center_handle is not None:
                self._center_handle.visible = False

    def set_line_thickness(self, thickness_px: float) -> None:
        value = float(thickness_px)
        if value <= 0.0:
            raise ValueError("line thickness must be positive")
        self.line_thickness_px = value
        if self._line_handle is not None:
            self._line_handle.line_width = value
            self.client.flush()

    def clear(self) -> None:
        if self._line_handle is not None:
            self._line_handle.remove()
            self._line_handle = None
        if self._center_handle is not None:
            self._center_handle.remove()
            self._center_handle = None
        self.client.flush()

    def update(
        self,
        *,
        tracking_start_time: float,
        current_time: float,
        show_trajectories: bool,
        show_current_centers: bool,
    ) -> TrajectoryRenderStats:
        start = float(tracking_start_time)
        current = float(current_time)
        if not 0.0 <= start <= 1.0 or not 0.0 <= current <= 1.0:
            raise ValueError("trajectory display times must lie in [0,1]")

        # Empty tracking sets are common and should be essentially free during
        # timeline playback. In particular, do not flush the Viser connection
        # once per source frame when there is nothing to draw.
        if not self._trajectories:
            return TrajectoryRenderStats(
                tracking_gaussians=0,
                visible_trajectories=0,
                line_segments=0,
                current_centers=0,
            )

        all_segments: list[np.ndarray] = []
        current_centers: list[np.ndarray] = []
        visible_trajectory_count = 0

        if current >= start:
            for trajectory in self._trajectories.values():
                _, path = trajectory.clip(start, current)
                if path.shape[0] > 0:
                    visible_trajectory_count += 1

                if show_trajectories and path.shape[0] >= 2:
                    segments = np.stack((path[:-1], path[1:]), axis=1)
                    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1)
                    segments = segments[lengths > 1e-8]
                    if segments.size:
                        all_segments.append(segments)

                if (
                    show_current_centers
                    and current >= start
                    and trajectory.start_time <= current <= trajectory.end_time
                ):
                    current_centers.append(trajectory.position_at(current))

        if show_trajectories and all_segments:
            points = np.ascontiguousarray(
                np.concatenate(all_segments, axis=0),
                dtype=np.float32,
            )
            colors = np.empty((points.shape[0], 2, 3), dtype=np.uint8)
            colors[...] = np.asarray(self.line_color, dtype=np.uint8)

            if self._line_handle is None:
                self._line_handle = self.client.scene.add_line_segments(
                    "/viewer4d/trajectory/lines",
                    points=points,
                    colors=colors,
                    line_width=self.line_thickness_px,
                    visible=True,
                )
            else:
                self._line_handle.points = points
                self._line_handle.colors = colors
                self._line_handle.line_width = self.line_thickness_px
                self._line_handle.visible = True
            line_segment_count = int(points.shape[0])
        else:
            line_segment_count = 0
            if self._line_handle is not None:
                self._line_handle.visible = False

        if show_current_centers and current_centers:
            points = np.ascontiguousarray(
                np.stack(current_centers, axis=0),
                dtype=np.float32,
            )
            if self._center_handle is None:
                self._center_handle = self.client.scene.add_point_cloud(
                    "/viewer4d/trajectory/current-centers",
                    points=points,
                    colors=self.center_color,
                    point_size=self.base_point_size * 1.5,
                    point_shape="circle",
                    point_shading="flat",
                    precision="float32",
                    visible=True,
                )
            else:
                self._center_handle.points = points
                self._center_handle.visible = True
            current_center_count = int(points.shape[0])
        else:
            current_center_count = 0
            if self._center_handle is not None:
                self._center_handle.visible = False

        # Handle property assignments already enqueue scene updates in Viser.
        # Calling client.flush() here would make the playback thread synchronously
        # wait on browser transport for every source frame, which can stall the
        # timeline completely at high source FPS.
        return TrajectoryRenderStats(
            tracking_gaussians=len(self._trajectories),
            visible_trajectories=visible_trajectory_count,
            line_segments=line_segment_count,
            current_centers=current_center_count,
        )