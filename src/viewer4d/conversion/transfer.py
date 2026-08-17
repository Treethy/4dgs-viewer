from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor

from viewer4d.core.model import AnytimeGS, SequenceInfo


def freetimegs_to_anytimegs(
    source: Any,
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> AnytimeGS:
    """Convert an in-memory FreeTimeGS Gaussian object to :class:`AnytimeGS`.

    The source object is expected to match the explicit-velocity checkpoint
    layout used by FreeTimeGS++ ``Gaussians`` checkpoints:

    ``means, scales, quats, opacities, sh_0, sh_n, times, durations,
    velocity_model, marginal_gates, max_duration``.

    FreeTimeGS stores time in source-video seconds. AnytimeGS always evaluates
    on normalized sequence time [0, 1], so the conversion applies the affine
    time change

        tau = (t - t_start) / (t_end - t_start)

    where ``t_start = frames.start / fps`` and
    ``t_end = (frames.stop - 1) / fps``.

    Under this coordinate change:

    * time centers are shifted and divided by the source time span;
    * temporal durations are divided by the source time span;
    * velocities are multiplied by the source time span.

    This preserves both Gaussian trajectories and temporal opacity exactly for
    the explicit linear-velocity FreeTimeGS representation.
    """

    fps, frame_start, frame_stop = _read_freetimegs_sequence(config)
    num_frames = frame_stop - frame_start
    if num_frames < 2:
        raise ValueError(
            "FreeTimeGS conversion requires at least two frames to define "
            "normalized time [0, 1]"
        )

    source_time_start = frame_start / fps
    source_time_end = (frame_stop - 1) / fps
    source_time_span = source_time_end - source_time_start
    if not math.isfinite(source_time_span) or source_time_span <= 0:
        raise ValueError(f"invalid source time span: {source_time_span}")

    means = _detach(source.means)
    scales = torch.exp(_detach(source.scales))
    quats = _detach(source.quats)
    opacities = torch.sigmoid(_detach(source.opacities))
    sh = torch.cat([_detach(source.sh_0), _detach(source.sh_n)], dim=1)

    source_times = _detach(source.times)
    time_center = (source_times - source_time_start) / source_time_span

    raw_duration = _detach(source.durations)
    max_duration = float(source.max_duration)
    if math.isinf(max_duration):
        duration_seconds = torch.exp(raw_duration)
        duration_encoding = "log"
    else:
        if not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError(
                "FreeTimeGS max_duration must be positive finite or +inf, "
                f"got {max_duration}"
            )
        duration_seconds = (max_duration / 6.0) * torch.sigmoid(raw_duration)
        duration_encoding = "bounded_logit"
    duration = duration_seconds / source_time_span

    velocity = _detach(source.velocity_model) * source_time_span
    temporal_gate = torch.sigmoid(20.0 * _detach(source.marginal_gates))

    model = AnytimeGS(
        sequence=SequenceInfo(num_frames=num_frames, fps=fps),
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh=sh,
        time_center=time_center,
        duration=duration,
        velocity=velocity,
        temporal_gate=temporal_gate,
        metadata={
            "source_format": "freetimegs",
            "source_sh_degree": int(source.sh_degree),
            "source_duration_encoding": duration_encoding,
            "source_max_duration_seconds": (
                "inf" if math.isinf(max_duration) else max_duration
            ),
            "source_sequence": {
                "fps": fps,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
                "time_start_seconds": source_time_start,
                "time_end_seconds": source_time_end,
                "time_span_seconds": source_time_span,
            },
            "time_transform": {
                "kind": "affine_normalization",
                "normalized_min": 0.0,
                "normalized_max": 1.0,
                "source_offset_seconds": source_time_start,
                "source_scale_seconds": source_time_span,
            },
        },
    )
    return model.to(device)


def _read_freetimegs_sequence(
    config: Mapping[str, Any],
) -> tuple[float, int, int]:
    try:
        data = config["data"]
        frames = data["frames"]
        fps = float(data["fps"])
        frame_start = int(frames["start"])
        frame_stop = int(frames["stop"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "FreeTimeGS config must contain data.fps and "
            "data.frames.{start, stop}"
        ) from error

    if fps <= 0 or not math.isfinite(fps):
        raise ValueError(f"data.fps must be positive and finite, got {fps}")
    if frame_start < 0:
        raise ValueError(f"data.frames.start must be non-negative, got {frame_start}")
    if frame_stop <= frame_start:
        raise ValueError(
            "data.frames.stop must be greater than data.frames.start, "
            f"got start={frame_start}, stop={frame_stop}"
        )

    return fps, frame_start, frame_stop


def _detach(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    return value.detach()
