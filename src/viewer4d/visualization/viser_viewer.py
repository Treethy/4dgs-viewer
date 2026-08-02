from __future__ import annotations

import time
from typing import Any

from viewer4d.core.model import GaussianFrame
from viewer4d.visualization.camera import (
    estimate_scene_bounds,
    initial_camera_from_bounds,
)
from viewer4d.visualization.viser_adapter import gaussian_frame_to_viser


class ViserStaticViewer:
    """Browser-side Gaussian viewer for a fixed static Gaussian frame."""

    def __init__(
        self,
        frame: GaussianFrame,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        label: str = "4Dviewer — Viser mode",
        scene_name: str = "/gaussians",
        opacity_threshold: float = 0.0,
        max_gaussians: int | None = None,
        server: Any | None = None,
    ) -> None:
        self.frame = frame
        self.server = server or _create_server(host=host, port=port, label=label)
        self.server.scene.set_up_direction("+z")
        self.server.scene.world_axes.visible = True

        bounds = estimate_scene_bounds(frame)
        initial = initial_camera_from_bounds(bounds)
        self.server.initial_camera.position = tuple(initial.position)
        self.server.initial_camera.look_at = tuple(initial.look_at)
        self.server.initial_camera.up = tuple(initial.up)
        self.server.initial_camera.near = initial.near
        self.server.initial_camera.far = initial.far

        data = gaussian_frame_to_viser(
            frame,
            opacity_threshold=opacity_threshold,
            max_gaussians=max_gaussians,
        )
        self.gaussian_handle = self.server.scene.add_gaussian_splats(
            scene_name,
            centers=data.centers,
            covariances=data.covariances,
            rgbs=data.rgbs,
            opacities=data.opacities,
        )

    def sleep_forever(self) -> None:
        if hasattr(self.server, "sleep_forever"):
            self.server.sleep_forever()
            return
        while True:
            time.sleep(3600.0)

    def stop(self) -> None:
        self.server.stop()


def _create_server(*, host: str, port: int, label: str) -> Any:
    try:
        import viser
    except ImportError as error:
        raise RuntimeError(
            "viser mode requires viser to be installed in the 4Dviewer environment"
        ) from error
    return viser.ViserServer(host=host, port=port, label=label)
