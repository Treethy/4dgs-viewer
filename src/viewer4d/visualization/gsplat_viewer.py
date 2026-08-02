from __future__ import annotations

import threading
import time
import traceback
from typing import Any

from viewer4d.core.model import GaussianFrame
from viewer4d.visualization.camera import (
    CameraState,
    camera_state_to_render_camera,
    capture_viser_camera,
    estimate_scene_bounds,
    initial_camera_from_bounds,
)
from viewer4d.visualization.gsplat_renderer import GsplatRenderer


class GsplatRemoteViewer:
    """Remote interactive viewer that renders each latest camera with gsplat.

    Camera events are coalesced: while one frame is being rendered, intermediate
    camera poses are discarded and only the newest pose is kept. This prevents a
    backlog after fast mouse movement.
    """

    def __init__(
        self,
        frame: GaussianFrame,
        *,
        renderer: GsplatRenderer | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        label: str = "4Dviewer — gsplat mode",
        max_width: int = 1280,
        max_height: int | None = None,
        jpeg_quality: int = 85,
        server: Any | None = None,
    ) -> None:
        frame.validate()
        if frame.means.device.type != "cuda":
            raise RuntimeError("GsplatRemoteViewer requires a CUDA GaussianFrame")
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        if max_height is not None and max_height <= 0:
            raise ValueError("max_height must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")

        self.frame = frame
        self.renderer = renderer or GsplatRenderer()
        self.max_width = max_width
        self.max_height = max_height
        self.jpeg_quality = jpeg_quality
        self.server = server or _create_server(host=host, port=port, label=label)
        self._workers: dict[int, _ClientRenderWorker] = {}
        self._workers_lock = threading.Lock()

        self.server.scene.set_up_direction("+z")
        self.server.scene.set_global_visibility(False)
        bounds = estimate_scene_bounds(frame)
        initial = initial_camera_from_bounds(bounds)
        self.server.initial_camera.position = tuple(initial.position)
        self.server.initial_camera.look_at = tuple(initial.look_at)
        self.server.initial_camera.up = tuple(initial.up)
        self.server.initial_camera.near = initial.near
        self.server.initial_camera.far = initial.far

        @self.server.on_client_connect
        def _on_connect(client: Any) -> None:
            client.scene.set_global_visibility(False)
            client.camera.near = initial.near
            client.camera.far = initial.far
            worker = _ClientRenderWorker(
                client=client,
                frame=self.frame,
                renderer=self.renderer,
                max_width=self.max_width,
                max_height=self.max_height,
                jpeg_quality=self.jpeg_quality,
            )
            with self._workers_lock:
                self._workers[int(client.client_id)] = worker

            @client.camera.on_update
            def _on_camera_update(camera: Any) -> None:
                worker.submit_camera(camera)

            worker.submit_camera(client.camera)

        @self.server.on_client_disconnect
        def _on_disconnect(client: Any) -> None:
            with self._workers_lock:
                worker = self._workers.pop(int(client.client_id), None)
            if worker is not None:
                worker.stop()

    def sleep_forever(self) -> None:
        if hasattr(self.server, "sleep_forever"):
            self.server.sleep_forever()
            return
        while True:
            time.sleep(3600.0)

    def stop(self) -> None:
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()
        self.server.stop()


class _ClientRenderWorker:
    def __init__(
        self,
        *,
        client: Any,
        frame: GaussianFrame,
        renderer: GsplatRenderer,
        max_width: int,
        max_height: int | None,
        jpeg_quality: int,
    ) -> None:
        self.client = client
        self.frame = frame
        self.renderer = renderer
        self.max_width = max_width
        self.max_height = max_height
        self.jpeg_quality = jpeg_quality
        self._condition = threading.Condition()
        self._latest_camera: CameraState | None = None
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"viewer4d-render-{client.client_id}",
            daemon=True,
        )
        self._thread.start()

    def submit_camera(self, camera: Any) -> None:
        try:
            snapshot = capture_viser_camera(camera)
        except ValueError:
            return
        with self._condition:
            self._latest_camera = snapshot
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._latest_camera is None and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                camera_state = self._latest_camera
                self._latest_camera = None

            assert camera_state is not None
            try:
                camera = camera_state_to_render_camera(
                    camera_state,
                    device=self.frame.means.device,
                    dtype=self.frame.means.dtype,
                    max_width=self.max_width,
                    max_height=self.max_height,
                )
                image = self.renderer.render(self.frame, camera)
                self.client.scene.set_background_image(
                    image,
                    format="jpeg",
                    jpeg_quality=self.jpeg_quality,
                )
            except Exception:
                traceback.print_exc()


def _create_server(*, host: str, port: int, label: str) -> Any:
    try:
        import viser
    except ImportError as error:
        raise RuntimeError(
            "gsplat mode uses Viser for camera control and requires viser to be installed"
        ) from error
    return viser.ViserServer(host=host, port=port, label=label)
