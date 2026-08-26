from __future__ import annotations

import math
import threading
import traceback
from typing import Any

import numpy as np
import torch

from viewer4d.core.camera import PinholeCamera
from viewer4d.core.model import GaussianFrame
from viewer4d.visualization.camera import (
    ViserCameraSnapshot,
    render_camera_from_viser,
    snapshot_viser_camera,
)
from viewer4d.visualization.modes import (
    InspectionScene,
    RenderMode,
    estimate_default_point_size,
)
from viewer4d.visualization.renderer import GsplatRenderer


class GaussianViewer:
    """Interactive viewer for one fixed GaussianFrame."""

    def __init__(
        self,
        frame: GaussianFrame,
        *,
        initial_camera: PinholeCamera | None = None,
        device: str | torch.device = "cuda",
        host: str = "127.0.0.1",
        port: int = 8080,
        render_width: int = 1000,
        jpeg_quality: int = 90,
        background: tuple[float, float, float] = (0.08, 0.08, 0.08),
        show_axes: bool = False,
    ) -> None:
        if render_width <= 0:
            raise ValueError("render_width must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1,100]")

        self.background = tuple(float(v) for v in background)
        self.renderer = GsplatRenderer(
            device=device,
            background=background,
        )
        self.frame = self.renderer.prepare_frame(frame)
        self.render_width = int(render_width)
        self.jpeg_quality = int(jpeg_quality)

        try:
            import viser
        except ImportError as error:
            raise RuntimeError("GaussianViewer requires viser") from error

        self.server = viser.ViserServer(
            host=host,
            port=port,
            label="AnytimeGS Gaussian Viewer",
        )
        self.server.scene.world_axes.visible = bool(show_axes)
        self.server.scene.configure_default_lights(cast_shadow=False)

        pose = initial_view(self.frame, initial_camera)
        apply_initial_camera(self.server, pose)

        fallback_aspect = (
            float(initial_camera.aspect)
            if initial_camera is not None
            else 1.0
        )
        base_point_size = estimate_default_point_size(self.frame)

        self._workers: dict[int, _StaticRenderWorker] = {}
        self._workers_lock = threading.Lock()

        @self.server.on_client_connect
        def _on_connect(client: Any) -> None:
            worker = _StaticRenderWorker(
                client=client,
                renderer=self.renderer,
                frame=self.frame,
                max_width=self.render_width,
                fallback_aspect=fallback_aspect,
                jpeg_quality=self.jpeg_quality,
            )
            inspection = InspectionScene(
                client,
                num_gaussians=self.frame.num_gaussians,
                background=self.background,
                point_size=base_point_size,
                point_sample_ratio=0.10,
                ellipsoid_sample_ratio=1.0,
            )

            with self._workers_lock:
                self._workers[int(client.client_id)] = worker

            mode_dropdown = client.gui.add_dropdown(
                "Render mode",
                options=tuple(mode.value for mode in RenderMode),
                initial_value=RenderMode.SPLAT.value,
            )
            point_size_slider = client.gui.add_slider(
                "Point size",
                min=0.10,
                max=5.00,
                step=0.05,
                initial_value=1.00,
                visible=False,
            )
            point_sample_slider = client.gui.add_slider(
                "Sampling ratio (%)",
                min=1,
                max=100,
                step=1,
                initial_value=10,
                visible=False,
            )
            point_count = client.gui.add_number(
                "Visible centers",
                initial_value=0,
                disabled=True,
                visible=False,
            )

            @client.camera.on_update
            def _on_camera_update(camera: Any) -> None:
                if inspection.mode is RenderMode.SPLAT:
                    worker.submit(snapshot_viser_camera(camera))

            @mode_dropdown.on_update
            async def _on_mode_update(_: Any) -> None:
                mode = RenderMode(mode_dropdown.value)
                centers = mode is RenderMode.CENTERS
                point_size_slider.visible = centers
                point_sample_slider.visible = centers
                point_count.visible = centers

                if mode is RenderMode.SPLAT:
                    inspection.set_mode(mode)
                    worker.set_enabled(True)
                    worker.submit(snapshot_viser_camera(client.camera))
                    return

                worker.set_enabled(False)
                inspection.set_mode(mode)
                update = inspection.prepare_update(self.frame, mode)
                count = inspection.apply_update(update)
                if centers:
                    point_count.value = count

            @point_size_slider.on_update
            async def _on_point_size(_: Any) -> None:
                inspection.set_point_size(
                    base_point_size * float(point_size_slider.value)
                )

            @point_sample_slider.on_update
            async def _on_point_sample(_: Any) -> None:
                inspection.set_point_sample_ratio(
                    float(point_sample_slider.value) / 100.0
                )
                if inspection.mode is RenderMode.CENTERS:
                    update = inspection.prepare_update(
                        self.frame,
                        RenderMode.CENTERS,
                    )
                    point_count.value = inspection.apply_update(update)

            set_client_camera(client, pose)
            worker.submit(snapshot_viser_camera(client.camera))

        @self.server.on_client_disconnect
        def _on_disconnect(client: Any) -> None:
            with self._workers_lock:
                worker = self._workers.pop(int(client.client_id), None)
            if worker is not None:
                worker.stop()

    def run(self) -> None:
        try:
            self.server.sleep_forever()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()
        self.server.stop()


class InitialView:
    def __init__(
        self,
        *,
        position: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray,
        fov: float,
        near: float,
        far: float,
    ) -> None:
        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov = fov
        self.near = near
        self.far = far


def initial_view(
    frame: GaussianFrame,
    camera: PinholeCamera | None,
) -> InitialView:
    center, radius = robust_scene_bounds(frame)

    if camera is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        position = center + np.array(
            [0.0, -3.0 * radius, 0.8 * radius],
            dtype=np.float64,
        )
        look_at = center
        fov = np.deg2rad(60.0)
    else:
        position = camera.position
        forward = camera.forward
        up = camera.up
        fov = camera.fov_y

        projected_distance = float(np.dot(center - position, forward))
        if projected_distance <= max(0.05 * radius, 1e-3):
            projected_distance = float(np.linalg.norm(center - position))
        if projected_distance <= 1e-3:
            projected_distance = max(radius, 1.0)
        look_at = position + forward * projected_distance

    focus_distance = float(np.linalg.norm(look_at - position))
    near = max(min(focus_distance * 0.01, radius * 0.02), 1e-3)
    far = max(focus_distance + 20.0 * radius, 50.0)

    return InitialView(
        position=np.asarray(position, dtype=np.float64),
        look_at=np.asarray(look_at, dtype=np.float64),
        up=_normalize(up),
        fov=float(fov),
        near=float(near),
        far=float(far),
    )


def apply_initial_camera(server: Any, pose: InitialView) -> None:
    server.scene.set_up_direction(tuple(pose.up))
    server.initial_camera.position = tuple(pose.position)
    server.initial_camera.look_at = tuple(pose.look_at)
    server.initial_camera.up = tuple(pose.up)
    server.initial_camera.fov = pose.fov
    server.initial_camera.near = pose.near
    server.initial_camera.far = pose.far


def set_client_camera(client: Any, pose: InitialView) -> None:
    with client.atomic():
        client.camera.position = pose.position
        client.camera.look_at = pose.look_at
        client.camera.up_direction = pose.up
        client.camera.fov = pose.fov
        client.camera.near = pose.near
        client.camera.far = pose.far
    client.flush()


def robust_scene_bounds(frame: GaussianFrame) -> tuple[np.ndarray, float]:
    means = frame.means.detach().float().cpu()
    opacities = frame.opacities.detach().float().cpu()

    finite = torch.isfinite(means).all(dim=1) & torch.isfinite(opacities)
    active = finite & (opacities > 0.01)
    if int(active.sum().item()) < 10:
        active = finite & (opacities > 1e-5)
    if int(active.sum().item()) < 10:
        active = finite

    points = means[active]
    if points.shape[0] == 0:
        raise ValueError("GaussianFrame has no finite Gaussian centers")

    if points.shape[0] > 200_000:
        indices = torch.linspace(
            0,
            points.shape[0] - 1,
            steps=200_000,
        ).long()
        points = points.index_select(0, indices)

    center = points.median(dim=0).values
    distances = torch.linalg.vector_norm(points - center, dim=1)
    if distances.numel() == 1:
        radius = float(frame.scales.max().item())
    else:
        radius = float(torch.quantile(distances, 0.90).item())

    radius = max(radius, float(frame.scales.max().item()), 1e-2)
    return center.numpy().astype(np.float64), radius


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid zero-length direction vector")
    return value / norm


class _StaticRenderWorker:
    def __init__(
        self,
        *,
        client: Any,
        renderer: GsplatRenderer,
        frame: GaussianFrame,
        max_width: int,
        fallback_aspect: float,
        jpeg_quality: int,
    ) -> None:
        self.client = client
        self.renderer = renderer
        self.frame = frame
        self.max_width = int(max_width)
        self.fallback_aspect = float(fallback_aspect)
        self.jpeg_quality = int(jpeg_quality)

        self._condition = threading.Condition()
        self._latest: tuple[int, ViserCameraSnapshot] | None = None
        self._enabled = True
        self._generation = 0
        self._stopped = False

        self._thread = threading.Thread(
            target=self._run,
            name=f"viewer4d-static-render-{client.client_id}",
            daemon=True,
        )
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        with self._condition:
            enabled = bool(enabled)
            if enabled == self._enabled:
                return
            self._enabled = enabled
            self._generation += 1
            self._latest = None
            self._condition.notify_all()

    def submit(self, camera: ViserCameraSnapshot) -> None:
        with self._condition:
            if not self._enabled or self._stopped:
                return
            self._latest = (self._generation, camera)
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._latest = None
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._latest is None and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                generation, snapshot = self._latest
                self._latest = None

            try:
                width, height = render_size(
                    snapshot,
                    max_width=self.max_width,
                    fallback_aspect=self.fallback_aspect,
                )
                camera = render_camera_from_viser(
                    snapshot,
                    width=width,
                    height=height,
                    device=self.renderer.device,
                    dtype=self.frame.means.dtype,
                )
                image = self.renderer.render(self.frame, camera)

                with self._condition:
                    valid = (
                        not self._stopped
                        and self._enabled
                        and generation == self._generation
                    )
                if not valid:
                    continue

                self.client.scene.set_background_image(
                    image,
                    format="jpeg",
                    jpeg_quality=self.jpeg_quality,
                )
                self.client.flush()
            except Exception:
                traceback.print_exc()


def render_size(
    snapshot: ViserCameraSnapshot,
    *,
    max_width: int,
    fallback_aspect: float,
) -> tuple[int, int]:
    if snapshot.image_width > 0 and snapshot.image_height > 0:
        scale = min(1.0, max_width / snapshot.image_width)
        width = max(1, int(round(snapshot.image_width * scale)))
        height = max(1, int(round(snapshot.image_height * scale)))
        return width, height

    aspect = snapshot.aspect
    if not math.isfinite(aspect) or aspect <= 0.0:
        aspect = fallback_aspect

    width = max_width
    height = max(1, int(round(width / aspect)))
    return width, height
