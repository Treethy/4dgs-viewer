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
from viewer4d.visualization.renderer import GsplatRenderer


class GaussianViewer:
    """Interactive viewer for one already-evaluated GaussianFrame.

    The optional calibrated camera controls the initial pose, up direction, and
    FOV. After connection, render resolution follows the actual browser canvas
    aspect ratio reported by Viser.
    """

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
        frame.validate()

        if render_width <= 0:
            raise ValueError("render_width must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1,100]")

        self.renderer = GsplatRenderer(
            frame,
            device=device,
            background=background,
        )

        # `render_width` is now a maximum render width, not a fixed image size.
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

        pose = _initial_view(frame, initial_camera)

        # Use the calibrated scene up, never a hard-coded world +Z for FTGS++.
        self.server.scene.set_up_direction(tuple(pose.up))

        self.server.initial_camera.position = tuple(pose.position)
        self.server.initial_camera.look_at = tuple(pose.look_at)
        self.server.initial_camera.up = tuple(pose.up)
        self.server.initial_camera.fov = pose.fov
        self.server.initial_camera.near = pose.near
        self.server.initial_camera.far = pose.far

        fallback_aspect = (
            float(initial_camera.aspect)
            if initial_camera is not None
            else 1.0
        )

        self._workers: dict[int, _RenderWorker] = {}
        self._workers_lock = threading.Lock()

        print("[viewer]")
        print(
            f"  initial camera: "
            f"{initial_camera.name if initial_camera else 'automatic'}"
        )
        print(f"  position:       {np.round(pose.position, 5).tolist()}")
        print(f"  look_at:        {np.round(pose.look_at, 5).tolist()}")
        print(f"  up:             {np.round(pose.up, 5).tolist()}")
        print(f"  fov_y:          {pose.fov:.6f} rad")
        print(f"  max render width: {self.render_width}px")
        print("  render aspect:    follows browser canvas")

        @self.server.on_client_connect
        def _on_connect(client: Any) -> None:
            worker = _RenderWorker(
                client=client,
                renderer=self.renderer,
                max_width=self.render_width,
                fallback_aspect=fallback_aspect,
                jpeg_quality=self.jpeg_quality,
            )

            with self._workers_lock:
                self._workers[int(client.client_id)] = worker

            @client.camera.on_update
            def _on_camera_update(camera: Any) -> None:
                worker.submit(snapshot_viser_camera(camera))

            # Put the connected client at the calibrated pose. Browser canvas
            # dimensions/aspect remain properties of the client and are read
            # from camera snapshots below.
            with client.atomic():
                client.camera.position = pose.position
                client.camera.look_at = pose.look_at
                client.camera.up_direction = pose.up
                client.camera.fov = pose.fov
                client.camera.near = pose.near
                client.camera.far = pose.far

            client.flush()

            # Render immediately. If browser dimensions are already available,
            # this first frame has the correct aspect. Otherwise the fallback
            # aspect is used until the next Viser camera update.
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


class _InitialView:
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


def _initial_view(
    frame: GaussianFrame,
    camera: PinholeCamera | None,
) -> _InitialView:
    center, radius = _robust_scene_bounds(frame)

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

        # Keep the exact calibrated forward direction, while choosing an orbit
        # center at approximately the scene depth.
        projected_distance = float(np.dot(center - position, forward))

        if projected_distance <= max(0.05 * radius, 1e-3):
            projected_distance = float(np.linalg.norm(center - position))
        if projected_distance <= 1e-3:
            projected_distance = max(radius, 1.0)

        look_at = position + forward * projected_distance

    focus_distance = float(np.linalg.norm(look_at - position))
    near = max(min(focus_distance * 0.01, radius * 0.02), 1e-3)
    far = max(focus_distance + 20.0 * radius, 50.0)

    return _InitialView(
        position=np.asarray(position, dtype=np.float64),
        look_at=np.asarray(look_at, dtype=np.float64),
        up=_normalize(up),
        fov=float(fov),
        near=float(near),
        far=float(far),
    )


def _robust_scene_bounds(frame: GaussianFrame) -> tuple[np.ndarray, float]:
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


class _RenderWorker:
    """Latest-camera-wins renderer for one Viser client."""

    def __init__(
        self,
        *,
        client: Any,
        renderer: GsplatRenderer,
        max_width: int,
        fallback_aspect: float,
        jpeg_quality: int,
    ) -> None:
        self.client = client
        self.renderer = renderer
        self.max_width = int(max_width)
        self.fallback_aspect = float(fallback_aspect)
        self.jpeg_quality = int(jpeg_quality)

        self._condition = threading.Condition()
        self._latest: ViserCameraSnapshot | None = None
        self._stopped = False
        self._last_render_size: tuple[int, int] | None = None

        self._thread = threading.Thread(
            target=self._run,
            name=f"viewer4d-render-{client.client_id}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, camera: ViserCameraSnapshot) -> None:
        with self._condition:
            self._latest = camera
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def _render_size(
        self,
        snapshot: ViserCameraSnapshot,
    ) -> tuple[int, int]:
        """Choose a render size with exactly the browser canvas aspect."""

        browser_width = snapshot.image_width
        browser_height = snapshot.image_height

        if browser_width > 0 and browser_height > 0:
            # Downscale the browser canvas if necessary, but never change its
            # aspect ratio.
            scale = min(1.0, self.max_width / browser_width)
            width = max(1, int(round(browser_width * scale)))
            height = max(1, int(round(browser_height * scale)))
            return width, height

        aspect = snapshot.aspect
        if not math.isfinite(aspect) or aspect <= 0.0:
            aspect = self.fallback_aspect

        width = self.max_width
        height = max(1, int(round(width / aspect)))
        return width, height

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._latest is None and not self._stopped:
                    self._condition.wait()

                if self._stopped:
                    return

                snapshot = self._latest
                self._latest = None

            assert snapshot is not None

            try:
                width, height = self._render_size(snapshot)

                if self._last_render_size != (width, height):
                    print(
                        f"[viewer] client {self.client.client_id}: "
                        f"render {width}x{height}, "
                        f"canvas {snapshot.image_width}x{snapshot.image_height}, "
                        f"aspect {snapshot.aspect:.4f}"
                    )
                    self._last_render_size = (width, height)

                camera = render_camera_from_viser(
                    snapshot,
                    width=width,
                    height=height,
                    device=self.renderer.device,
                    dtype=self.renderer.dtype,
                )

                image = self.renderer.render(camera)

                self.client.scene.set_background_image(
                    image,
                    format="jpeg",
                    jpeg_quality=self.jpeg_quality,
                )
                self.client.flush()
            except Exception:
                traceback.print_exc()