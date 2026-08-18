from __future__ import annotations

import threading
import time
import traceback
from typing import Any

import torch

from viewer4d.core.camera import PinholeCamera
from viewer4d.core.model import AnytimeGS, GaussianFrame
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
from viewer4d.visualization.viewer import (
    apply_initial_camera,
    initial_view,
    render_size,
    set_client_camera,
)


_PLAYBACK_SPEEDS: dict[str, float] = {
    "0.25×": 0.25,
    "0.50×": 0.50,
    "0.75×": 0.75,
    "1.00×": 1.00,
    "1.25×": 1.25,
    "1.50×": 1.50,
    "2.00×": 2.00,
}


class Gaussian4DViewer:
    """Interactive AnytimeGS viewer with a frame timeline and playback.

    The source sequence FPS comes from ``AnytimeGS.sequence.fps``. Playback
    speed is a multiplier of that FPS. For example, a 30 FPS sequence played at
    0.50× advances at 15 source frames per wall-clock second; 1.25× advances at
    37.5 source frames per second.

    Rendering is latest-state-wins. If rendering is slower than the requested
    playback rate, intermediate frames are skipped rather than queued.
    """

    def __init__(
        self,
        model: AnytimeGS,
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

        # Move the complete 4D representation once. Every at_frame()/at_time()
        # result then stays on GPU; timeline playback does not copy the model
        # from CPU for each frame.
        self.model = model.to(self.renderer.device)
        self.render_width = int(render_width)
        self.jpeg_quality = int(jpeg_quality)

        try:
            import viser
        except ImportError as error:
            raise RuntimeError("Gaussian4DViewer requires viser") from error

        self.server = viser.ViserServer(
            host=host,
            port=port,
            label="AnytimeGS 4D Gaussian Viewer",
        )
        self.server.scene.world_axes.visible = bool(show_axes)
        self.server.scene.configure_default_lights(cast_shadow=False)

        initial_frame = self.model.at_frame(0)
        pose = initial_view(initial_frame, initial_camera)
        apply_initial_camera(self.server, pose)

        self._fallback_aspect = (
            float(initial_camera.aspect)
            if initial_camera is not None
            else 1.0
        )
        self._base_point_size = estimate_default_point_size(initial_frame)
        self._pose = pose

        self._sessions: dict[int, _Client4DSession] = {}
        self._sessions_lock = threading.Lock()

        print("[4D viewer]")
        print(f"  Gaussians:         {self.model.num_gaussians:,}")
        print(f"  Frames:            {self.model.sequence.num_frames}")
        print(f"  Source FPS:        {self.model.sequence.fps:g}")
        print("  Playback speed:    browser GUI")
        print("  Render modes:      Splat / Ellipsoid / Centers")

        @self.server.on_client_connect
        def _on_connect(client: Any) -> None:
            session = _Client4DSession(
                client=client,
                model=self.model,
                renderer=self.renderer,
                pose=self._pose,
                background=self.background,
                base_point_size=self._base_point_size,
                max_width=self.render_width,
                fallback_aspect=self._fallback_aspect,
                jpeg_quality=self.jpeg_quality,
            )
            with self._sessions_lock:
                self._sessions[int(client.client_id)] = session

        @self.server.on_client_disconnect
        def _on_disconnect(client: Any) -> None:
            with self._sessions_lock:
                session = self._sessions.pop(int(client.client_id), None)
            if session is not None:
                session.stop()

    def run(self) -> None:
        try:
            self.server.sleep_forever()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.stop()
        self.server.stop()


class _Client4DSession:
    """Per-browser GUI, playback clock, and render state."""

    def __init__(
        self,
        *,
        client: Any,
        model: AnytimeGS,
        renderer: GsplatRenderer,
        pose: Any,
        background: tuple[float, float, float],
        base_point_size: float,
        max_width: int,
        fallback_aspect: float,
        jpeg_quality: int,
    ) -> None:
        self.client = client
        self.model = model
        self.renderer = renderer
        self.num_frames = model.sequence.num_frames
        self.source_fps = float(model.sequence.fps)

        self._state_lock = threading.RLock()
        self._play_condition = threading.Condition(self._state_lock)
        self._frame_index = 0
        self._mode = RenderMode.SPLAT
        self._playing = False
        self._loop = True
        self._speed = 1.0
        self._play_anchor_frame = 0
        self._play_anchor_wall = time.monotonic()
        self._stopped = False

        inspection = InspectionScene(
            client,
            num_gaussians=model.num_gaussians,
            background=background,
            point_size=base_point_size,
            point_sample_ratio=0.10,
            ellipsoid_sample_ratio=0.05,
            ellipsoid_sigma=1.0,
        )
        self.inspection = inspection

        self.worker = _DynamicRenderWorker(
            client=client,
            model=model,
            renderer=renderer,
            inspection=inspection,
            max_width=max_width,
            fallback_aspect=fallback_aspect,
            jpeg_quality=jpeg_quality,
            on_native_count=self._on_native_count,
            on_present=self._on_present,
        )

        self._build_gui(base_point_size)

        @client.camera.on_update
        def _on_camera_update(camera: Any) -> None:
            self.worker.update_camera(snapshot_viser_camera(camera))

        set_client_camera(client, pose)
        self.worker.update_camera(snapshot_viser_camera(client.camera))
        self.worker.update_frame(0)

        self._play_thread = threading.Thread(
            target=self._play_loop,
            name=f"viewer4d-playback-{client.client_id}",
            daemon=True,
        )
        self._play_thread.start()

    def _build_gui(self, base_point_size: float) -> None:
        self.mode_dropdown = self.client.gui.add_dropdown(
            "Render mode",
            options=tuple(mode.value for mode in RenderMode),
            initial_value=RenderMode.SPLAT.value,
        )

        self.frame_slider = self.client.gui.add_slider(
            "Frame",
            min=0,
            max=self.num_frames - 1,
            step=1,
            initial_value=0,
        )
        self.normalized_time = self.client.gui.add_number(
            "Normalized time",
            initial_value=0.0,
            disabled=True,
        )
        self.play_checkbox = self.client.gui.add_checkbox(
            "Play",
            initial_value=False,
        )
        self.speed_dropdown = self.client.gui.add_dropdown(
            "Playback speed",
            options=tuple(_PLAYBACK_SPEEDS.keys()),
            initial_value="1.00×",
        )
        self.loop_checkbox = self.client.gui.add_checkbox(
            "Loop",
            initial_value=True,
        )
        self.source_fps_number = self.client.gui.add_number(
            "Source FPS",
            initial_value=self.source_fps,
            disabled=True,
        )
        self.effective_fps_number = self.client.gui.add_number(
            "Effective FPS",
            initial_value=self.source_fps,
            disabled=True,
        )
        self.displayed_frame_number = self.client.gui.add_number(
            "Displayed frame",
            initial_value=0,
            disabled=True,
        )
        self.render_fps_number = self.client.gui.add_number(
            "Render FPS",
            initial_value=0.0,
            disabled=True,
        )

        self.point_size_slider = self.client.gui.add_slider(
            "Point size",
            min=0.10,
            max=5.00,
            step=0.05,
            initial_value=1.00,
            visible=False,
        )
        self.point_sample_slider = self.client.gui.add_slider(
            "Center sampling (%)",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            visible=False,
        )
        self.point_count_number = self.client.gui.add_number(
            "Visible centers",
            initial_value=0,
            disabled=True,
            visible=False,
        )

        self.ellipsoid_sample_slider = self.client.gui.add_slider(
            "Ellipsoid sampling (%)",
            min=1,
            max=100,
            step=1,
            initial_value=5,
            visible=False,
        )
        self.ellipsoid_count_number = self.client.gui.add_number(
            "Visible ellipsoids",
            initial_value=0,
            disabled=True,
            visible=False,
        )

        self._base_point_size = float(base_point_size)

        @self.mode_dropdown.on_update
        async def _mode_update(_: Any) -> None:
            mode = RenderMode(self.mode_dropdown.value)
            self.set_mode(mode)

        @self.frame_slider.on_update
        async def _frame_update(_: Any) -> None:
            self.set_frame(int(self.frame_slider.value), reanchor=True)

        @self.play_checkbox.on_update
        async def _play_update(_: Any) -> None:
            self.set_playing(bool(self.play_checkbox.value))

        @self.speed_dropdown.on_update
        async def _speed_update(_: Any) -> None:
            speed = _PLAYBACK_SPEEDS[self.speed_dropdown.value]
            self.set_speed(speed)

        @self.loop_checkbox.on_update
        async def _loop_update(_: Any) -> None:
            with self._play_condition:
                self._loop = bool(self.loop_checkbox.value)
                self._reanchor_locked()
                self._play_condition.notify_all()

        @self.point_size_slider.on_update
        async def _point_size_update(_: Any) -> None:
            self.inspection.set_point_size(
                self._base_point_size * float(self.point_size_slider.value)
            )

        @self.point_sample_slider.on_update
        async def _point_sample_update(_: Any) -> None:
            self.inspection.set_point_sample_ratio(
                float(self.point_sample_slider.value) / 100.0
            )
            if self.mode is RenderMode.CENTERS:
                self.worker.refresh()

        @self.ellipsoid_sample_slider.on_update
        async def _ellipsoid_sample_update(_: Any) -> None:
            self.inspection.set_ellipsoid_sample_ratio(
                float(self.ellipsoid_sample_slider.value) / 100.0
            )
            if self.mode is RenderMode.ELLIPSOID:
                self.worker.refresh()

    @property
    def mode(self) -> RenderMode:
        with self._state_lock:
            return self._mode

    def set_mode(self, mode: RenderMode) -> None:
        with self._state_lock:
            self._mode = mode

        centers = mode is RenderMode.CENTERS
        ellipsoids = mode is RenderMode.ELLIPSOID

        self.point_size_slider.visible = centers
        self.point_sample_slider.visible = centers
        self.point_count_number.visible = centers
        self.ellipsoid_sample_slider.visible = ellipsoids
        self.ellipsoid_count_number.visible = ellipsoids

        self.inspection.set_mode(mode)
        self.worker.update_mode(mode)

    def set_frame(self, frame_index: int, *, reanchor: bool) -> None:
        frame_index = max(0, min(self.num_frames - 1, int(frame_index)))

        with self._play_condition:
            if frame_index == self._frame_index:
                return

            self._frame_index = frame_index
            if reanchor and self._playing:
                self._reanchor_locked()

            self.frame_slider.value = frame_index
            self.normalized_time.value = self.model.sequence.frame_to_time(
                frame_index
            )
            self.worker.update_frame(frame_index)
            self._play_condition.notify_all()

    def set_playing(self, playing: bool) -> None:
        with self._play_condition:
            playing = bool(playing)
            if playing == self._playing:
                return

            self._playing = playing
            self.play_checkbox.value = playing
            if playing:
                self._reanchor_locked()
            self._play_condition.notify_all()

    def set_speed(self, speed: float) -> None:
        if speed <= 0.0:
            raise ValueError("playback speed must be positive")

        with self._play_condition:
            self._speed = float(speed)
            self.effective_fps_number.value = self.source_fps * self._speed
            self._reanchor_locked()
            self._play_condition.notify_all()

    def stop(self) -> None:
        with self._play_condition:
            self._stopped = True
            self._playing = False
            self._play_condition.notify_all()

        self.worker.stop()
        if threading.current_thread() is not self._play_thread:
            self._play_thread.join(timeout=2.0)

    def _reanchor_locked(self) -> None:
        self._play_anchor_frame = self._frame_index
        self._play_anchor_wall = time.monotonic()

    def _play_loop(self) -> None:
        while True:
            with self._play_condition:
                while not self._playing and not self._stopped:
                    self._play_condition.wait()

                if self._stopped:
                    return

                now = time.monotonic()
                elapsed = now - self._play_anchor_wall
                advance = int(elapsed * self.source_fps * self._speed)
                raw_target = self._play_anchor_frame + advance

                if self._loop:
                    target = raw_target % self.num_frames
                    stop_at_end = False
                else:
                    target = min(raw_target, self.num_frames - 1)
                    stop_at_end = raw_target >= self.num_frames - 1

                current = self._frame_index

            if target != current:
                # Playback-driven frame changes do not re-anchor the wall clock.
                # If rendering is behind, this can jump directly to the frame
                # implied by elapsed time instead of accumulating a queue.
                self.set_frame(target, reanchor=False)

            if stop_at_end:
                self.set_playing(False)
                continue

            # Wake often enough for high-FPS source sequences without busy-spin.
            effective_fps = max(self.source_fps * self._speed, 1.0)
            time.sleep(min(0.01, 0.25 / effective_fps))

    def _on_present(
        self,
        mode: RenderMode,
        frame_index: int,
        render_fps: float,
    ) -> None:
        """Update GUI with what actually reached the browser.

        The timeline may advance at 60 source FPS while rendering can be much
        slower. In that case intermediate source frames are intentionally
        skipped, but every completed render is still presented.
        """

        with self._state_lock:
            if mode != self._mode:
                return

        self.displayed_frame_number.value = int(frame_index)
        self.render_fps_number.value = float(render_fps)

    def _on_native_count(
        self,
        mode: RenderMode,
        count: int,
        frame_index: int,
    ) -> None:
        with self._state_lock:
            if frame_index != self._frame_index or mode != self._mode:
                return

        if mode is RenderMode.CENTERS:
            self.point_count_number.value = int(count)
        elif mode is RenderMode.ELLIPSOID:
            self.ellipsoid_count_number.value = int(count)


class _DynamicRenderWorker:
    """Latest-state-wins worker over (camera, frame index, render mode)."""

    def __init__(
        self,
        *,
        client: Any,
        model: AnytimeGS,
        renderer: GsplatRenderer,
        inspection: InspectionScene,
        max_width: int,
        fallback_aspect: float,
        jpeg_quality: int,
        on_native_count: Any,
        on_present: Any,
    ) -> None:
        self.client = client
        self.model = model
        self.renderer = renderer
        self.inspection = inspection
        self.max_width = int(max_width)
        self.fallback_aspect = float(fallback_aspect)
        self.jpeg_quality = int(jpeg_quality)
        self.on_native_count = on_native_count
        self.on_present = on_present

        self._condition = threading.Condition()
        self._camera: ViserCameraSnapshot | None = None
        self._frame_index = 0
        self._mode = RenderMode.SPLAT

        # `_serial` still coalesces pending requests: while one frame is being
        # rendered, any number of camera/time updates collapse into the latest
        # request. It is deliberately NOT used to discard a completed render.
        self._serial = 0

        # Results are discarded only across render-mode changes. This prevents
        # an old Splat image from overwriting Ellipsoid/Centers after switching
        # modes, without causing playback starvation when rendering < source FPS.
        self._mode_epoch = 0

        self._pending = False
        self._stopped = False
        self._render_fps_ema = 0.0

        # Cache the evaluated GaussianFrame for the current timeline frame.
        #
        # Camera motion must NOT call model.at_frame() again when time has not
        # changed. Without this cache, every mouse event recomputes dynamic
        # means + temporal opacity for all Gaussians before rasterization,
        # making the 4D viewer much less responsive than the static 3D viewer.
        self._cached_frame_index: int | None = None
        self._cached_frame: GaussianFrame | None = None

        self._thread = threading.Thread(
            target=self._run,
            name=f"viewer4d-dynamic-render-{client.client_id}",
            daemon=True,
        )
        self._thread.start()

    def update_camera(self, camera: ViserCameraSnapshot) -> None:
        with self._condition:
            self._camera = camera
            if self._mode is RenderMode.SPLAT:
                self._queue_locked()

    def update_frame(self, frame_index: int) -> None:
        with self._condition:
            self._frame_index = int(frame_index)
            self._queue_locked()

    def update_mode(self, mode: RenderMode) -> None:
        with self._condition:
            if mode != self._mode:
                self._mode = mode
                self._mode_epoch += 1
            self._queue_locked()

    def refresh(self) -> None:
        with self._condition:
            self._queue_locked()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._pending = False
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def _queue_locked(self) -> None:
        if self._stopped:
            return
        self._serial += 1
        self._pending = True
        self._condition.notify()

    def _can_present(
        self,
        mode: RenderMode,
        mode_epoch: int,
    ) -> bool:
        with self._condition:
            return (
                not self._stopped
                and self._mode == mode
                and self._mode_epoch == mode_epoch
            )

    def _update_render_fps(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0.0:
            return self._render_fps_ema

        instant = 1.0 / elapsed_seconds
        if self._render_fps_ema <= 0.0:
            self._render_fps_ema = instant
        else:
            self._render_fps_ema = (
                0.90 * self._render_fps_ema
                + 0.10 * instant
            )
        return self._render_fps_ema

    def _frame_for(self, frame_index: int) -> GaussianFrame:
        """Return the evaluated frame, recomputing only when time changes.

        This method is called only from the render worker thread, so the cache
        itself needs no additional lock.

        - camera-only update: reuse the existing GaussianFrame
        - timeline update: evaluate AnytimeGS exactly once for the new frame
        """

        if (
            self._cached_frame is not None
            and self._cached_frame_index == frame_index
        ):
            return self._cached_frame

        frame = self.model.at_frame(frame_index)

        # Prepare once here rather than repeatedly inside camera-only renders.
        # Since the AnytimeGS model already lives on renderer.device this
        # normally returns the frame directly, but it also guarantees the
        # cached tensors have the layout expected by gsplat.
        frame = self.renderer.prepare_frame(frame)

        self._cached_frame_index = frame_index
        self._cached_frame = frame
        return frame

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopped:
                    self._condition.wait()

                if self._stopped:
                    return

                serial = self._serial
                mode = self._mode
                mode_epoch = self._mode_epoch
                frame_index = self._frame_index
                camera_snapshot = self._camera
                self._pending = False

            try:
                render_started = time.monotonic()
                frame = self._frame_for(frame_index)

                if mode is RenderMode.SPLAT:
                    if camera_snapshot is None:
                        continue

                    width, height = render_size(
                        camera_snapshot,
                        max_width=self.max_width,
                        fallback_aspect=self.fallback_aspect,
                    )
                    camera = render_camera_from_viser(
                        camera_snapshot,
                        width=width,
                        height=height,
                        device=self.renderer.device,
                        dtype=frame.means.dtype,
                    )
                    image = self.renderer.render(frame, camera)

                    # A newer frame/camera request may have arrived while this
                    # image was rendering. Present this completed image anyway;
                    # the next loop iteration will immediately render only the
                    # latest pending state. This is what makes playback degrade
                    # gracefully from e.g. 60 source FPS to 15 rendered FPS.
                    if not self._can_present(mode, mode_epoch):
                        continue

                    self.client.scene.set_background_image(
                        image,
                        format="jpeg",
                        jpeg_quality=self.jpeg_quality,
                    )
                    self.client.flush()

                    render_fps = self._update_render_fps(
                        time.monotonic() - render_started
                    )
                    self.on_present(
                        mode,
                        frame_index,
                        render_fps,
                    )
                    continue

                update = self.inspection.prepare_update(frame, mode)

                if not self._can_present(mode, mode_epoch):
                    continue

                count = self.inspection.apply_update(update)

                if not self._can_present(mode, mode_epoch):
                    continue

                render_fps = self._update_render_fps(
                    time.monotonic() - render_started
                )
                self.on_native_count(mode, count, frame_index)
                self.on_present(
                    mode,
                    frame_index,
                    render_fps,
                )

            except Exception:
                traceback.print_exc()