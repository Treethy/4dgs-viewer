from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
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
from viewer4d.visualization.renderer import (
    GaussianOverlayRequest,
    GsplatRenderer,
)
from viewer4d.visualization.selection import (
    SelectionHighlight,
    SelectionMode,
    SelectionState,
    inspect_gaussian,
    pick_gaussian,
    select_gaussians_in_rect,
)
from viewer4d.visualization.trajectory import (
    TrajectorySamplingMode,
    TrajectorySamplingRange,
    TrajectoryScene,
    TrajectoryState,
    build_anytimegs_trajectories,
    sample_gaussians,
)
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

_PREVIEW_SCALE = 0.50
_PREVIEW_JPEG_QUALITY = 70
_PREVIEW_RADIUS_CLIP = 0.75
_CAMERA_SETTLE_SECONDS = 0.12
_BOX_DEPTH_MAX_WIDTH = 512
_TRAJECTORY_DEFAULT_OPACITY_CUTOFF = 0.05


@dataclass(frozen=True, slots=True)
class PresentedRenderState:
    """The Splat frame/camera that is actually visible in the browser."""

    frame_index: int
    camera: ViserCameraSnapshot
    width: int
    height: int
    camera_epoch: int
    preview: bool


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
        print("  Selection modes:   Camera / Single / Box")

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
    """Per-browser GUI, playback clock, render state, and Gaussian selection."""

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
        self._max_width = int(max_width)
        self._fallback_aspect = float(fallback_aspect)
        self._presented_splat_state: PresentedRenderState | None = None

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

        self.selection_state = SelectionState()
        self.selection_highlight = SelectionHighlight(
            client,
            base_point_size=base_point_size,
        )
        self._selection_click_callback: Any | None = None
        self._selection_rect_callback: Any | None = None
        self._selection_camera_lock: dict[str, Any] | None = None
        self._restoring_selection_camera = False

        self.trajectory_state = TrajectoryState()
        self.trajectory_scene = TrajectoryScene(
            client,
            base_point_size=base_point_size,
        )
        self._trajectory_random_generator = torch.Generator(device="cpu")
        self._trajectory_random_generator.manual_seed(0)
        self._trajectory_visible_count = 0
        self._trajectory_update_error_reported = False

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
            if self.selection_state.mode is SelectionMode.CAMERA:
                self.worker.update_camera(snapshot_viser_camera(camera))
                return
            self._restore_selection_camera(camera)

        set_client_camera(client, pose)
        self.worker.update_camera(
            snapshot_viser_camera(client.camera),
            interactive=False,
        )
        self.worker.update_frame(0)
        self._refresh_selection(layout=True)
        self._rebuild_trajectory_scene()

        self._play_thread = threading.Thread(
            target=self._play_loop,
            name=f"viewer4d-playback-{client.client_id}",
            daemon=True,
        )
        self._play_thread.start()

    def _build_gui(self, base_point_size: float) -> None:
        tab_group = self.client.gui.add_tab_group()

        with tab_group.add_tab("Render"):
            self.mode_dropdown = self.client.gui.add_dropdown(
                "Render mode",
                options=tuple(mode.value for mode in RenderMode),
                initial_value=RenderMode.SPLAT.value,
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

        with tab_group.add_tab("Trajectory"):
            with self.client.gui.add_folder(
                "Sampling",
                expand_by_default=True,
            ):
                self.trajectory_sampling_mode_dropdown = self.client.gui.add_dropdown(
                    "Mode",
                    options=tuple(mode.value for mode in TrajectorySamplingMode),
                    initial_value=TrajectorySamplingMode.HIGH_SPEED.value,
                )
                self.trajectory_sampling_range_dropdown = self.client.gui.add_dropdown(
                    "Range",
                    options=tuple(value.value for value in TrajectorySamplingRange),
                    initial_value=TrajectorySamplingRange.CURRENT_FRAME.value,
                )
                self.trajectory_sample_count_number = self.client.gui.add_number(
                    "Count",
                    initial_value=500,
                    min=0,
                    max=self.model.num_gaussians,
                    step=1,
                )
                self.trajectory_opacity_cutoff_slider = self.client.gui.add_slider(
                    "Min current opacity",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=_TRAJECTORY_DEFAULT_OPACITY_CUTOFF,
                    hint=(
                        "Current-frame sampling only: candidates must be inside their "
                        "3σ lifespan and have at least this temporal opacity."
                    ),
                )
                self.trajectory_sample_button = self.client.gui.add_button("Sample")
                self.trajectory_sampled_count_number = self.client.gui.add_number(
                    "Sampled Gaussians",
                    initial_value=0,
                    disabled=True,
                )
                self.trajectory_candidate_count_number = self.client.gui.add_number(
                    "Candidate Gaussians",
                    initial_value=0,
                    disabled=True,
                )
                self.trajectory_sample_source_text = self.client.gui.add_text(
                    "Sample source",
                    "Not sampled",
                    disabled=True,
                )

            with self.client.gui.add_folder(
                "Tracking",
                expand_by_default=True,
            ):
                self.trajectory_start_frame_number = self.client.gui.add_number(
                    "Start tracking frame",
                    initial_value=0,
                    min=0,
                    max=self.num_frames - 1,
                    step=1,
                    hint=(
                        "Trajectories are not drawn before this frame. Each Gaussian "
                        "is additionally clipped by its own 3σ lifespan."
                    ),
                )
                self.trajectory_include_sampled_checkbox = self.client.gui.add_checkbox(
                    "Include sampled Gaussians",
                    initial_value=True,
                )
                self.trajectory_include_manual_checkbox = self.client.gui.add_checkbox(
                    "Include manual Gaussians",
                    initial_value=False,
                )
                self.trajectory_current_selection_count = self.client.gui.add_number(
                    "Current Selection",
                    initial_value=0,
                    disabled=True,
                )
                self.trajectory_manual_add_action = self.client.gui.add_button_group(
                    "Manual selection",
                    options=("Add current selection",),
                    hint=(
                        "Add the current Selection-tab Gaussian IDs to the stored "
                        "manual trajectory set."
                    ),
                )
                self.trajectory_manual_clear_action = self.client.gui.add_button_group(
                    "",
                    options=("Clear manual selection",),
                    hint="Clear all Gaussian IDs stored in the manual trajectory set.",
                )
                self.trajectory_manual_count_number = self.client.gui.add_number(
                    "Manual Gaussians",
                    initial_value=0,
                    disabled=True,
                )
                self.trajectory_tracking_count_number = self.client.gui.add_number(
                    "Tracking Gaussians",
                    initial_value=0,
                    disabled=True,
                )
                self.trajectory_visible_count_number = self.client.gui.add_number(
                    "Visible Trajectories",
                    initial_value=0,
                    disabled=True,
                )

            with self.client.gui.add_folder(
                "Display",
                expand_by_default=True,
            ):
                self.trajectory_show_checkbox = self.client.gui.add_checkbox(
                    "Show trajectories",
                    initial_value=True,
                )
                self.trajectory_show_centers_checkbox = self.client.gui.add_checkbox(
                    "Show current centers",
                    initial_value=True,
                )
                self.trajectory_line_thickness_slider = self.client.gui.add_slider(
                    "Line thickness (px)",
                    min=0.5,
                    max=5.0,
                    step=0.5,
                    initial_value=1.5,
                )

        with tab_group.add_tab("Selection"):
            self.selection_mode_buttons = self.client.gui.add_button_group(
                "Selection mode",
                options=("Camera", "Single", "Box"),
                hint=(
                    "Camera: normal navigation. Single/Box: lock the current camera "
                    "and use the mouse for Gaussian selection."
                ),
            )

            self.selection_highlight_checkbox = self.client.gui.add_checkbox(
                "Highlight selection",
                initial_value=True,
            )
            self.box_depth_tolerance_slider = self.client.gui.add_slider(
                "Box depth tolerance (%)",
                min=1.0,
                max=20.0,
                step=0.5,
                initial_value=5.0,
                hint=(
                    "Keep box-selected Gaussian centers whose camera depth is close "
                    "to the locally rendered surface depth."
                ),
            )
            self.box_outline_checkbox = self.client.gui.add_checkbox(
                "Show box Gaussian outlines",
                initial_value=False,
            )
            self.box_outline_limit_slider = self.client.gui.add_slider(
                "Max box outlines",
                min=10,
                max=500,
                step=10,
                initial_value=200,
                visible=False,
            )
            self.selection_status = self.client.gui.add_text(
                "Status",
                "No Gaussian selected",
                disabled=True,
            )

            self.single_selection_folder = self.client.gui.add_folder(
                "Selected Gaussian",
                expand_by_default=True,
                visible=False,
            )
            with self.single_selection_folder:
                self.selected_index_number = self.client.gui.add_number(
                    "Gaussian index",
                    initial_value=0,
                    disabled=True,
                )
                self.selected_time_number = self.client.gui.add_number(
                    "Normalized time",
                    initial_value=0.0,
                    disabled=True,
                )

                motion_folder = self.client.gui.add_folder(
                    "Motion",
                    expand_by_default=True,
                )
                with motion_folder:
                    self.selected_velocity = self.client.gui.add_vector3(
                        "Velocity",
                        initial_value=(0.0, 0.0, 0.0),
                        disabled=True,
                    )
                    self.selected_speed = self.client.gui.add_number(
                        "Speed",
                        initial_value=0.0,
                        disabled=True,
                    )

                temporal_folder = self.client.gui.add_folder(
                    "Temporal",
                    expand_by_default=True,
                )
                with temporal_folder:
                    self.selected_time_center = self.client.gui.add_number(
                        "Time center",
                        initial_value=0.0,
                        disabled=True,
                    )
                    self.selected_duration = self.client.gui.add_number(
                        "Duration",
                        initial_value=0.0,
                        disabled=True,
                    )
                    self.selected_support_start = self.client.gui.add_number(
                        "3σ support start",
                        initial_value=0.0,
                        disabled=True,
                    )
                    self.selected_support_end = self.client.gui.add_number(
                        "3σ support end",
                        initial_value=0.0,
                        disabled=True,
                    )
                    self.selected_temporal_gate = self.client.gui.add_number(
                        "Temporal gate",
                        initial_value=0.0,
                        disabled=True,
                    )

                spatial_folder = self.client.gui.add_folder(
                    "Spatial",
                    expand_by_default=False,
                )
                with spatial_folder:
                    self.selected_base_center = self.client.gui.add_vector3(
                        "Center @ time center",
                        initial_value=(0.0, 0.0, 0.0),
                        disabled=True,
                    )
                    self.selected_current_center = self.client.gui.add_vector3(
                        "Current center",
                        initial_value=(0.0, 0.0, 0.0),
                        disabled=True,
                    )
                    self.selected_scale = self.client.gui.add_vector3(
                        "Scale",
                        initial_value=(0.0, 0.0, 0.0),
                        disabled=True,
                    )
                    self.selected_quaternion = self.client.gui.add_text(
                        "Quaternion (wxyz)",
                        "(1, 0, 0, 0)",
                        disabled=True,
                    )

                appearance_folder = self.client.gui.add_folder(
                    "Appearance",
                    expand_by_default=False,
                )
                with appearance_folder:
                    self.selected_base_opacity = self.client.gui.add_number(
                        "Base opacity",
                        initial_value=0.0,
                        disabled=True,
                    )
                    self.selected_current_opacity = self.client.gui.add_number(
                        "Current opacity",
                        initial_value=0.0,
                        disabled=True,
                    )

            self.box_selection_folder = self.client.gui.add_folder(
                "Selected Gaussians",
                expand_by_default=True,
                visible=False,
            )
            with self.box_selection_folder:
                self.selected_count_number = self.client.gui.add_number(
                    "Count",
                    initial_value=0,
                    disabled=True,
                )

            self.clear_selection_button = self.client.gui.add_button(
                "Clear selection",
                visible=False,
            )

        # Timeline is shared by all three functional tabs.
        timeline_folder = self.client.gui.add_folder(
            "Timeline",
            expand_by_default=True,
        )
        with timeline_folder:
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

        @self.trajectory_sampling_range_dropdown.on_update
        async def _trajectory_sampling_range_update(_: Any) -> None:
            self._refresh_trajectory_sampling_controls()

        @self.trajectory_sample_button.on_click
        async def _trajectory_sample(_: Any) -> None:
            self._sample_trajectory_gaussians()

        @self.trajectory_start_frame_number.on_update
        async def _trajectory_start_frame_update(_: Any) -> None:
            frame = max(
                0,
                min(
                    self.num_frames - 1,
                    int(round(float(self.trajectory_start_frame_number.value))),
                ),
            )
            self.trajectory_state.start_tracking_frame = frame
            if self.trajectory_start_frame_number.value != frame:
                self.trajectory_start_frame_number.value = frame
            self._refresh_trajectory_scene()

        @self.trajectory_include_sampled_checkbox.on_update
        async def _trajectory_include_sampled_update(_: Any) -> None:
            self.trajectory_state.include_sampled = bool(
                self.trajectory_include_sampled_checkbox.value
            )
            self._rebuild_trajectory_scene()

        @self.trajectory_include_manual_checkbox.on_update
        async def _trajectory_include_manual_update(_: Any) -> None:
            self.trajectory_state.include_manual = bool(
                self.trajectory_include_manual_checkbox.value
            )
            self._rebuild_trajectory_scene()

        @self.trajectory_manual_add_action.on_click
        async def _trajectory_manual_add(_: Any) -> None:
            self.trajectory_state.manual.update(
                self.selection_state.selected.indices
            )
            if self.trajectory_state.manual:
                self.trajectory_state.include_manual = True
                self.trajectory_include_manual_checkbox.value = True
            self._rebuild_trajectory_scene()

        @self.trajectory_manual_clear_action.on_click
        async def _trajectory_manual_clear(_: Any) -> None:
            self.trajectory_state.manual.clear()
            self._rebuild_trajectory_scene()

        @self.trajectory_show_checkbox.on_update
        async def _trajectory_show_update(_: Any) -> None:
            self.trajectory_state.show_trajectories = bool(
                self.trajectory_show_checkbox.value
            )
            self._refresh_trajectory_scene()

        @self.trajectory_show_centers_checkbox.on_update
        async def _trajectory_show_centers_update(_: Any) -> None:
            self.trajectory_state.show_current_centers = bool(
                self.trajectory_show_centers_checkbox.value
            )
            self._refresh_trajectory_scene()

        @self.trajectory_line_thickness_slider.on_update
        async def _trajectory_line_thickness_update(_: Any) -> None:
            self.trajectory_scene.set_line_thickness(
                float(self.trajectory_line_thickness_slider.value)
            )

        @self.selection_mode_buttons.on_click
        async def _selection_mode_click(_: Any) -> None:
            mode_by_label = {
                "Camera": SelectionMode.CAMERA,
                "Single": SelectionMode.SINGLE,
                "Box": SelectionMode.BOX,
            }
            self.set_selection_mode(
                mode_by_label[str(self.selection_mode_buttons.value)]
            )

        @self.selection_highlight_checkbox.on_update
        async def _selection_highlight_update(_: Any) -> None:
            self.selection_state.highlight_enabled = bool(
                self.selection_highlight_checkbox.value
            )
            self._sync_selection_render_state()
            self._refresh_selection(layout=False)

        @self.box_outline_checkbox.on_update
        async def _box_outline_update(_: Any) -> None:
            enabled = bool(self.box_outline_checkbox.value)
            self.box_outline_limit_slider.visible = enabled
            self._sync_selection_render_state()

        @self.box_outline_limit_slider.on_update
        async def _box_outline_limit_update(_: Any) -> None:
            self._sync_selection_render_state()

        @self.clear_selection_button.on_click
        async def _clear_selection(_: Any) -> None:
            self.selection_state.clear()
            self.selection_highlight.clear()
            self._sync_selection_render_state()
            self._refresh_selection(layout=True)

        self._refresh_selection_mode_buttons()
        self._refresh_selection_gui(layout=True)
        self._refresh_trajectory_sampling_controls()
        self._refresh_trajectory_gui()

    def _refresh_trajectory_sampling_controls(self) -> None:
        sampling_range = TrajectorySamplingRange(
            self.trajectory_sampling_range_dropdown.value
        )
        self.trajectory_opacity_cutoff_slider.visible = (
            sampling_range is TrajectorySamplingRange.CURRENT_FRAME
        )

    def _sample_trajectory_gaussians(self) -> None:
        with self._state_lock:
            frame_index = self._frame_index

        sampling_mode = TrajectorySamplingMode(
            self.trajectory_sampling_mode_dropdown.value
        )
        sampling_range = TrajectorySamplingRange(
            self.trajectory_sampling_range_dropdown.value
        )
        count = max(0, int(round(float(self.trajectory_sample_count_number.value))))
        if self.trajectory_sample_count_number.value != count:
            self.trajectory_sample_count_number.value = count

        result = sample_gaussians(
            self.model,
            mode=sampling_mode,
            sampling_range=sampling_range,
            count=count,
            frame_index=(
                frame_index
                if sampling_range is TrajectorySamplingRange.CURRENT_FRAME
                else None
            ),
            opacity_cutoff=float(self.trajectory_opacity_cutoff_slider.value),
            random_generator=self._trajectory_random_generator,
        )
        self.trajectory_state.sampled.replace(result.indices)
        self.trajectory_state.sampled_range = result.sampling_range
        self.trajectory_state.sampled_at_frame = result.sampled_at_frame
        self.trajectory_state.sampled_candidate_count = result.candidate_count
        self._rebuild_trajectory_scene()

    def _rebuild_trajectory_scene(self) -> None:
        indices = self.trajectory_state.tracking_indices()
        trajectories = build_anytimegs_trajectories(self.model, indices)
        self.trajectory_scene.set_trajectories(trajectories)
        self._refresh_trajectory_scene()

    def _refresh_trajectory_scene(self) -> None:
        with self._state_lock:
            frame_index = self._frame_index

        start_frame = max(
            0,
            min(self.num_frames - 1, self.trajectory_state.start_tracking_frame),
        )
        stats = self.trajectory_scene.update(
            tracking_start_time=self.model.sequence.frame_to_time(start_frame),
            current_time=self.model.sequence.frame_to_time(frame_index),
            show_trajectories=self.trajectory_state.show_trajectories,
            show_current_centers=self.trajectory_state.show_current_centers,
        )
        self._trajectory_visible_count = stats.visible_trajectories
        self._refresh_trajectory_gui()

    def _safe_refresh_trajectory_scene(self) -> None:
        """Refresh trajectory geometry without ever killing timeline playback."""

        try:
            self._refresh_trajectory_scene()
            self._trajectory_update_error_reported = False
        except Exception:
            # Trajectory visualization is auxiliary to the timeline. A bad scene
            # update must not terminate the playback thread. Report the first
            # failure and keep the underlying 4D render/timeline alive.
            if not self._trajectory_update_error_reported:
                print("[trajectory] scene update failed; playback will continue")
                traceback.print_exc()
                self._trajectory_update_error_reported = True

    def _refresh_trajectory_gui(self) -> None:
        self.trajectory_sampled_count_number.value = len(self.trajectory_state.sampled)
        self.trajectory_candidate_count_number.value = int(
            self.trajectory_state.sampled_candidate_count
        )
        if self.trajectory_state.sampled_range is None:
            sample_source = "Not sampled"
        elif self.trajectory_state.sampled_range is TrajectorySamplingRange.GLOBAL:
            sample_source = "Global"
        else:
            frame = self.trajectory_state.sampled_at_frame
            sample_source = f"Frame {frame}" if frame is not None else "Current frame"
        if self.trajectory_sample_source_text.value != sample_source:
            self.trajectory_sample_source_text.value = sample_source

        self.trajectory_current_selection_count.value = len(
            self.selection_state.selected
        )
        self.trajectory_manual_count_number.value = len(self.trajectory_state.manual)
        self.trajectory_tracking_count_number.value = len(
            self.trajectory_state.tracking_indices()
        )
        self.trajectory_visible_count_number.value = int(
            self._trajectory_visible_count
        )

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
        self._sync_selection_render_state()
        self._refresh_selection(layout=False)

    def set_selection_mode(self, mode: SelectionMode) -> None:
        if not isinstance(mode, SelectionMode):
            mode = SelectionMode(mode)

        old_mode = self.selection_state.mode
        if old_mode is mode:
            self._refresh_selection_mode_buttons()
            return

        self._remove_selection_pointer_callbacks()
        self.selection_state.set_mode(mode)

        if mode is SelectionMode.CAMERA:
            self._selection_camera_lock = None
            self.worker.update_camera(
                snapshot_viser_camera(self.client.camera),
                interactive=False,
            )
        else:
            self._capture_selection_camera()
            self._install_selection_pointer_callback(mode)

        self._refresh_selection_mode_buttons()
        self._refresh_selection_gui(layout=True)

    def _refresh_selection_mode_buttons(self) -> None:
        label_by_mode = {
            SelectionMode.CAMERA: "Camera",
            SelectionMode.SINGLE: "Single",
            SelectionMode.BOX: "Box",
        }
        desired = label_by_mode[self.selection_state.mode]
        if str(self.selection_mode_buttons.value) != desired:
            self.selection_mode_buttons.value = desired
        self._apply_selection_mode_button_style(desired)

    def _apply_selection_mode_button_style(self, active_label: str) -> None:
        """Restore a persistent dark active state for Viser 1.0.30 button groups.

        Viser 1.0.30 renders ``add_button_group()`` as a row of outline buttons
        and does not visually distinguish the current value.  We keep the
        compact horizontal layout, but explicitly style the active
        Camera/Single/Box button after each mode change.
        """
        try:
            from viser import _messages
        except Exception:
            return

        active = str(active_label).replace("\\", "\\\\").replace('"', '\\"')
        source = f"""
(() => {{
  const active = "{active}";
  const expected = new Set(["Camera", "Single", "Box"]);

  const apply = () => {{
    const buttons = Array.from(document.querySelectorAll("button"));
    let group = null;

    for (const button of buttons) {{
      const text = (button.textContent || "").trim();
      if (!expected.has(text) || button.parentElement === null) continue;

      const siblings = Array.from(button.parentElement.children).filter(
        (node) => node instanceof HTMLButtonElement
      );
      const labels = siblings.map(
        (node) => (node.textContent || "").trim()
      );

      if (
        siblings.length === 3 &&
        labels.every((label) => expected.has(label))
      ) {{
        group = siblings;
        break;
      }}
    }}

    if (group === null) return false;

    for (const button of group) {{
      const selected = (button.textContent || "").trim() === active;
      if (selected) {{
        button.style.setProperty("background-color", "#25262b", "important");
        button.style.setProperty("border-color", "#25262b", "important");
        button.style.setProperty("color", "#ffffff", "important");
      }} else {{
        button.style.removeProperty("background-color");
        button.style.removeProperty("border-color");
        button.style.removeProperty("color");
      }}
    }}
    return true;
  }};

  // React may mount/update the GUI component just after the Python message.
  // Re-apply a few times so the active style survives that render.
  apply();
  setTimeout(apply, 0);
  setTimeout(apply, 40);
  setTimeout(apply, 120);
}})();
"""
        try:
            self.client._websock_connection.queue_message(
                _messages.RunJavascriptMessage(source=source)
            )
        except Exception:
            # Styling is cosmetic; never let it affect viewer interaction.
            pass

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

        # Timeline playback updates values and native geometry only. It must not
        # touch GUI folder visibility; doing so makes the Timeline jump vertically.
        self._refresh_selection(layout=False)
        self._safe_refresh_trajectory_scene()

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
        self._remove_selection_pointer_callbacks()
        self.selection_highlight.clear()
        self.trajectory_scene.clear()

        with self._play_condition:
            self._stopped = True
            self._playing = False
            self._play_condition.notify_all()

        self.worker.stop()
        if threading.current_thread() is not self._play_thread:
            self._play_thread.join(timeout=2.0)

    def _current_normalized_time(self) -> float:
        with self._state_lock:
            frame_index = self._frame_index
        return self.model.sequence.frame_to_time(frame_index)

    def _capture_selection_camera(self) -> None:
        camera = self.client.camera
        self._selection_camera_lock = {
            "position": np.asarray(camera.position, dtype=np.float64).copy(),
            "look_at": np.asarray(camera.look_at, dtype=np.float64).copy(),
            "up_direction": np.asarray(camera.up_direction, dtype=np.float64).copy(),
            "fov": float(camera.fov),
        }

    def _restore_selection_camera(self, camera: Any) -> None:
        lock = self._selection_camera_lock
        if lock is None or self._restoring_selection_camera:
            return

        unchanged = (
            np.allclose(camera.position, lock["position"], rtol=0.0, atol=1e-9)
            and np.allclose(camera.look_at, lock["look_at"], rtol=0.0, atol=1e-9)
            and np.allclose(
                camera.up_direction,
                lock["up_direction"],
                rtol=0.0,
                atol=1e-9,
            )
            and abs(float(camera.fov) - float(lock["fov"])) <= 1e-9
        )
        if unchanged:
            return

        self._restoring_selection_camera = True
        try:
            with self.client.atomic():
                camera.position = lock["position"]
                camera.look_at = lock["look_at"]
                camera.up_direction = lock["up_direction"]
                camera.fov = float(lock["fov"])
            self.client.flush()
        finally:
            self._restoring_selection_camera = False

    def _selection_reference_state(
        self,
    ) -> tuple[int, float, ViserCameraSnapshot]:
        """Return the frame/camera that the user is actually looking at."""

        if self.mode is RenderMode.SPLAT and self._presented_splat_state is not None:
            state = self._presented_splat_state
            return (
                state.frame_index,
                self.model.sequence.frame_to_time(state.frame_index),
                state.camera,
            )

        with self._state_lock:
            frame_index = self._frame_index
        return (
            frame_index,
            self.model.sequence.frame_to_time(frame_index),
            snapshot_viser_camera(self.client.camera),
        )

    def _surface_depth_for_selection(
        self,
        *,
        frame_index: int,
        camera_snapshot: ViserCameraSnapshot,
    ) -> np.ndarray:
        """Render a small expected-depth map for surface-aware box selection."""

        frame = self.renderer.prepare_frame(self.model.at_frame(frame_index))
        depth_max_width = min(self._max_width, _BOX_DEPTH_MAX_WIDTH)
        width, height = render_size(
            camera_snapshot,
            max_width=depth_max_width,
            fallback_aspect=self._fallback_aspect,
        )
        camera = render_camera_from_viser(
            camera_snapshot,
            width=width,
            height=height,
            device=self.renderer.device,
            dtype=frame.means.dtype,
        )
        return self.renderer.render_depth(
            frame,
            camera,
            radius_clip=max(self.renderer.radius_clip, _PREVIEW_RADIUS_CLIP),
        )

    def _install_selection_pointer_callback(self, mode: SelectionMode) -> None:
        if mode is SelectionMode.SINGLE:

            @self.client.scene.on_click()
            async def _single_click(event: Any) -> None:
                if self.selection_state.mode is not SelectionMode.SINGLE:
                    return
                _, normalized_time, camera_snapshot = self._selection_reference_state()
                index = pick_gaussian(
                    self.model,
                    normalized_time=normalized_time,
                    camera=camera_snapshot,
                    screen_pos=event.screen_pos,
                )
                if index is None:
                    self.selection_state.clear()
                else:
                    self.selection_state.replace(
                        (index,),
                        source=SelectionMode.SINGLE,
                    )
                self._sync_selection_render_state()
                self._refresh_selection(layout=True)

            self._selection_click_callback = _single_click
            return

        if mode is SelectionMode.BOX:

            @self.client.scene.on_rect_select()
            async def _box_select(event: Any) -> None:
                if self.selection_state.mode is not SelectionMode.BOX:
                    return

                frame_index, normalized_time, camera_snapshot = (
                    self._selection_reference_state()
                )
                surface_depth = self._surface_depth_for_selection(
                    frame_index=frame_index,
                    camera_snapshot=camera_snapshot,
                )
                indices = select_gaussians_in_rect(
                    self.model,
                    normalized_time=normalized_time,
                    camera=camera_snapshot,
                    screen_min=event.screen_min,
                    screen_max=event.screen_max,
                    surface_depth=surface_depth,
                    surface_depth_tolerance=(
                        float(self.box_depth_tolerance_slider.value) / 100.0
                    ),
                )
                self.selection_state.replace(
                    indices,
                    source=SelectionMode.BOX,
                )
                self._sync_selection_render_state()
                self._refresh_selection(layout=True)

            self._selection_rect_callback = _box_select

    def _remove_selection_pointer_callbacks(self) -> None:
        if self._selection_click_callback is not None:
            self.client.scene.remove_click_callback(self._selection_click_callback)
            self._selection_click_callback = None
        if self._selection_rect_callback is not None:
            self.client.scene.remove_rect_select_callback(
                self._selection_rect_callback
            )
            self._selection_rect_callback = None

    def _sync_selection_render_state(self) -> None:
        """Send selection identity to the Splat worker without copying Gaussian data."""

        self.worker.update_selection(
            indices=self.selection_state.selected.indices,
            source=self.selection_state.selection_source,
            highlight_enabled=self.selection_state.highlight_enabled,
            box_outlines=bool(self.box_outline_checkbox.value),
            box_outline_limit=int(self.box_outline_limit_slider.value),
        )

    def _refresh_selection(self, *, layout: bool) -> None:
        normalized_time = self._current_normalized_time()

        # Splat highlighting is baked into the exact rendered image/camera so it
        # never races ahead of the delayed background during camera interaction.
        if self.mode is RenderMode.SPLAT:
            self.selection_highlight.clear()
        else:
            self.selection_highlight.set_visible(
                self.selection_state.highlight_enabled
            )
            if self.selection_state.selected and self.selection_state.highlight_enabled:
                self.selection_highlight.update(
                    self.model,
                    indices=self.selection_state.selected.indices,
                    normalized_time=normalized_time,
                    mode=(
                        self.selection_state.selection_source
                        or self.selection_state.mode
                    ),
                )
            else:
                self.selection_highlight.clear()

        self._refresh_selection_gui(
            normalized_time=normalized_time,
            layout=layout,
        )
        self._refresh_trajectory_gui()

    def _refresh_selection_gui(
        self,
        *,
        normalized_time: float | None = None,
        layout: bool,
    ) -> None:
        if normalized_time is None:
            normalized_time = self._current_normalized_time()

        count = len(self.selection_state.selected)
        source = self.selection_state.selection_source
        single_visible = source is SelectionMode.SINGLE and count == 1
        box_visible = source is SelectionMode.BOX and count > 0
        clear_visible = count > 0

        # Only explicit selection/mode changes may alter layout. Timeline frame
        # changes update values below but never folder visibility.
        if layout:
            if self.clear_selection_button.visible != clear_visible:
                self.clear_selection_button.visible = clear_visible
            if self.single_selection_folder.visible != single_visible:
                self.single_selection_folder.visible = single_visible
            if self.box_selection_folder.visible != box_visible:
                self.box_selection_folder.visible = box_visible

        if count == 0:
            if self.selection_status.value != "No Gaussian selected":
                self.selection_status.value = "No Gaussian selected"
            return

        if single_visible:
            index = self.selection_state.selected.indices[0]
            inspection = inspect_gaussian(
                self.model,
                index,
                normalized_time,
            )
            status = f"Gaussian #{index}"
            if self.selection_status.value != status:
                self.selection_status.value = status

            self.selected_index_number.value = inspection.index
            self.selected_time_number.value = inspection.normalized_time
            self.selected_velocity.value = inspection.velocity
            self.selected_speed.value = inspection.speed
            self.selected_time_center.value = inspection.time_center
            self.selected_duration.value = inspection.duration
            self.selected_support_start.value = inspection.support_start_3sigma
            self.selected_support_end.value = inspection.support_end_3sigma
            self.selected_temporal_gate.value = inspection.temporal_gate
            self.selected_base_center.value = inspection.base_center
            self.selected_current_center.value = inspection.current_center
            self.selected_scale.value = inspection.scale
            self.selected_quaternion.value = _format_quaternion(
                inspection.quaternion_wxyz
            )
            self.selected_base_opacity.value = inspection.base_opacity
            self.selected_current_opacity.value = inspection.current_opacity
            return

        status = f"{count:,} Gaussians selected"
        if self.selection_status.value != status:
            self.selection_status.value = status
        self.selected_count_number.value = count

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
        presented_state: PresentedRenderState | None = None,
    ) -> None:
        """Update GUI and remember the exact Splat image currently displayed."""

        with self._state_lock:
            if mode != self._mode:
                return
            if presented_state is not None:
                self._presented_splat_state = presented_state

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
    """Render worker with separate timeline and interactive-camera policies.

    Timeline updates may present completed older frames when rendering is slower
    than source FPS. Camera updates are stricter: any render produced from a
    stale camera epoch is discarded. While the camera is moving, a lower
    resolution preview is rendered; after it settles, a full-resolution image is
    generated automatically.
    """

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
        self._mode_epoch = 0
        self._camera_epoch = 0
        self._selection_epoch = 0
        self._last_camera_update_wall = 0.0
        self._full_render_due = False

        self._selection_indices: tuple[int, ...] = ()
        self._selection_source: SelectionMode | None = None
        self._selection_highlight_enabled = True
        self._box_outlines = False
        self._box_outline_limit = 200

        self._pending = False
        self._stopped = False
        self._render_fps_ema = 0.0

        self._cached_frame_index: int | None = None
        self._cached_frame: GaussianFrame | None = None

        self._thread = threading.Thread(
            target=self._run,
            name=f"viewer4d-dynamic-render-{client.client_id}",
            daemon=True,
        )
        self._thread.start()

    def update_camera(
        self,
        camera: ViserCameraSnapshot,
        *,
        interactive: bool = True,
    ) -> None:
        """Update the requested camera.

        ``interactive=True`` is reserved for real browser camera motion and
        enables the low-resolution preview -> settled full-resolution path.
        Programmatic camera synchronization (initialization or leaving a
        locked selection mode) uses ``interactive=False`` so simply pressing
        the Camera button never flashes a blurry preview.
        """

        with self._condition:
            self._camera = camera
            self._camera_epoch += 1
            if interactive:
                self._last_camera_update_wall = time.monotonic()
                self._full_render_due = True
            else:
                self._last_camera_update_wall = 0.0
                self._full_render_due = False
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

    def update_selection(
        self,
        *,
        indices: tuple[int, ...],
        source: SelectionMode | None,
        highlight_enabled: bool,
        box_outlines: bool,
        box_outline_limit: int,
    ) -> None:
        with self._condition:
            self._selection_indices = tuple(int(index) for index in indices)
            self._selection_source = source
            self._selection_highlight_enabled = bool(highlight_enabled)
            self._box_outlines = bool(box_outlines)
            self._box_outline_limit = max(0, int(box_outline_limit))
            self._selection_epoch += 1
            if self._mode is RenderMode.SPLAT:
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
        self._pending = True
        self._condition.notify()

    def _wait_for_work_locked(self) -> bool:
        while not self._pending and not self._stopped:
            timeout: float | None = None
            if (
                self._mode is RenderMode.SPLAT
                and self._full_render_due
                and self._camera is not None
            ):
                remaining = _CAMERA_SETTLE_SECONDS - (
                    time.monotonic() - self._last_camera_update_wall
                )
                if remaining <= 0.0:
                    self._pending = True
                    break
                timeout = remaining
            self._condition.wait(timeout=timeout)
        return not self._stopped

    def _can_present_splat(
        self,
        *,
        mode_epoch: int,
        camera_epoch: int,
        selection_epoch: int,
    ) -> bool:
        with self._condition:
            return (
                not self._stopped
                and self._mode is RenderMode.SPLAT
                and self._mode_epoch == mode_epoch
                and self._camera_epoch == camera_epoch
                and self._selection_epoch == selection_epoch
            )

    def _can_present_native(self, mode: RenderMode, mode_epoch: int) -> bool:
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
            self._render_fps_ema = 0.90 * self._render_fps_ema + 0.10 * instant
        return self._render_fps_ema

    def _frame_for(self, frame_index: int) -> GaussianFrame:
        if (
            self._cached_frame is not None
            and self._cached_frame_index == frame_index
        ):
            return self._cached_frame

        frame = self.renderer.prepare_frame(self.model.at_frame(frame_index))
        self._cached_frame_index = frame_index
        self._cached_frame = frame
        return frame

    def _overlay_request(
        self,
        *,
        indices: tuple[int, ...],
        source: SelectionMode | None,
        highlight_enabled: bool,
        box_outlines: bool,
        box_outline_limit: int,
        pixel_scale: float = 1.0,
    ) -> GaussianOverlayRequest | None:
        if not highlight_enabled or not indices:
            return None

        pixel_scale = max(1e-3, float(pixel_scale))
        if source is SelectionMode.SINGLE and len(indices) == 1:
            return GaussianOverlayRequest(
                indices=indices,
                draw_centers=True,
                draw_ellipses=True,
                max_ellipses=1,
                center_radius_px=3.0 * pixel_scale,
                ellipse_sigma=2.0,
                line_width_px=max(1, int(round(2.0 * pixel_scale))),
            )

        return GaussianOverlayRequest(
            indices=indices,
            draw_centers=True,
            draw_ellipses=box_outlines,
            max_ellipses=(box_outline_limit if box_outlines else 0),
            # Box selections can contain hundreds of Gaussians.  A fixed
            # screen-space marker radius makes distant selections collapse
            # into a solid blob, so size each center from gsplat's projected
            # Gaussian radius instead.  These pixel clamps are scaled for the
            # low-resolution interaction preview; after the browser stretches
            # the preview back to canvas size, the apparent marker size stays
            # consistent with the full-resolution render.
            center_radius_px=1.0 * pixel_scale,
            center_radius_from_projected=True,
            center_radius_scale=0.15,
            center_radius_min_px=0.35 * pixel_scale,
            center_radius_max_px=1.25 * pixel_scale,
            center_min_projected_radius_px=0.60 * pixel_scale,
            ellipse_sigma=2.0,
            line_width_px=max(1, int(round(1.0 * pixel_scale))),
        )

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._wait_for_work_locked():
                    return

                mode = self._mode
                mode_epoch = self._mode_epoch
                frame_index = self._frame_index
                camera_snapshot = self._camera
                camera_epoch = self._camera_epoch
                selection_epoch = self._selection_epoch
                selection_indices = self._selection_indices
                selection_source = self._selection_source
                selection_highlight_enabled = self._selection_highlight_enabled
                box_outlines = self._box_outlines
                box_outline_limit = self._box_outline_limit

                now = time.monotonic()
                preview = (
                    mode is RenderMode.SPLAT
                    and self._full_render_due
                    and now - self._last_camera_update_wall
                    < _CAMERA_SETTLE_SECONDS
                )
                self._pending = False

            try:
                render_started = time.monotonic()
                frame = self._frame_for(frame_index)

                if mode is RenderMode.SPLAT:
                    if camera_snapshot is None:
                        continue

                    max_width = self.max_width
                    jpeg_quality = self.jpeg_quality
                    radius_clip: float | None = None
                    if preview:
                        max_width = max(128, int(round(self.max_width * _PREVIEW_SCALE)))
                        jpeg_quality = min(self.jpeg_quality, _PREVIEW_JPEG_QUALITY)
                        radius_clip = max(
                            self.renderer.radius_clip,
                            _PREVIEW_RADIUS_CLIP,
                        )

                    width, height = render_size(
                        camera_snapshot,
                        max_width=max_width,
                        fallback_aspect=self.fallback_aspect,
                    )
                    camera = render_camera_from_viser(
                        camera_snapshot,
                        width=width,
                        height=height,
                        device=self.renderer.device,
                        dtype=frame.means.dtype,
                    )
                    # The preview background is later stretched to the same
                    # browser canvas size as a full-resolution render. Scale
                    # pixel-sized overlay primitives down in the preview image
                    # so their apparent on-screen size stays constant.
                    overlay_pixel_scale = (
                        _PREVIEW_SCALE if preview else 1.0
                    )
                    overlay = self._overlay_request(
                        indices=selection_indices,
                        source=selection_source,
                        highlight_enabled=selection_highlight_enabled,
                        box_outlines=box_outlines,
                        box_outline_limit=box_outline_limit,
                        pixel_scale=overlay_pixel_scale,
                    )
                    image = self.renderer.render(
                        frame,
                        camera,
                        overlay=overlay,
                        radius_clip=radius_clip,
                    )

                    # Camera/selection renders are never allowed to arrive late.
                    # Timeline frame staleness is intentionally tolerated so
                    # playback still degrades gracefully when render FPS is low.
                    if not self._can_present_splat(
                        mode_epoch=mode_epoch,
                        camera_epoch=camera_epoch,
                        selection_epoch=selection_epoch,
                    ):
                        continue

                    self.client.scene.set_background_image(
                        image,
                        format="jpeg",
                        jpeg_quality=jpeg_quality,
                    )
                    self.client.flush()

                    if not preview:
                        with self._condition:
                            if self._camera_epoch == camera_epoch:
                                self._full_render_due = False

                    render_fps = self._update_render_fps(
                        time.monotonic() - render_started
                    )
                    self.on_present(
                        mode,
                        frame_index,
                        render_fps,
                        PresentedRenderState(
                            frame_index=frame_index,
                            camera=camera_snapshot,
                            width=width,
                            height=height,
                            camera_epoch=camera_epoch,
                            preview=preview,
                        ),
                    )
                    continue

                update = self.inspection.prepare_update(frame, mode)
                if not self._can_present_native(mode, mode_epoch):
                    continue

                count = self.inspection.apply_update(update)
                if not self._can_present_native(mode, mode_epoch):
                    continue

                render_fps = self._update_render_fps(
                    time.monotonic() - render_started
                )
                self.on_native_count(mode, count, frame_index)
                self.on_present(mode, frame_index, render_fps, None)

            except Exception:
                traceback.print_exc()


def _format_quaternion(value: tuple[float, float, float, float]) -> str:
    return "(" + ", ".join(f"{component:.6g}" for component in value) + ")"