from __future__ import annotations

from dataclasses import dataclass

import torch

from viewer4d.core.model import AnytimeGS
from viewer4d.visualization.trajectory.data import DEFAULT_LIFESPAN_SIGMA
from viewer4d.visualization.trajectory.state import (
    TrajectorySamplingMode,
    TrajectorySamplingRange,
)


@dataclass(frozen=True, slots=True)
class TrajectorySampleResult:
    indices: tuple[int, ...]
    candidate_count: int
    sampling_range: TrajectorySamplingRange
    sampled_at_frame: int | None


@torch.inference_mode()
def sample_gaussians(
    model: AnytimeGS,
    *,
    mode: TrajectorySamplingMode | str,
    sampling_range: TrajectorySamplingRange | str,
    count: int,
    frame_index: int | None = None,
    opacity_cutoff: float = 0.05,
    lifespan_sigma: float = DEFAULT_LIFESPAN_SIGMA,
    random_generator: torch.Generator | None = None,
) -> TrajectorySampleResult:
    """Sample Gaussian identities for trajectory visualization.

    ``Current frame`` candidates satisfy both conditions discussed for the
    viewer: the frame lies inside the Gaussian's ±3σ lifespan and its temporal
    opacity at that frame is at least ``opacity_cutoff``. No camera/frustum or
    occlusion visibility test is applied.
    """

    mode = mode if isinstance(mode, TrajectorySamplingMode) else TrajectorySamplingMode(mode)
    sampling_range = (
        sampling_range
        if isinstance(sampling_range, TrajectorySamplingRange)
        else TrajectorySamplingRange(sampling_range)
    )

    requested = max(int(count), 0)
    if not 0.0 <= opacity_cutoff <= 1.0:
        raise ValueError("opacity_cutoff must be in [0,1]")
    if lifespan_sigma <= 0.0:
        raise ValueError("lifespan_sigma must be positive")

    if sampling_range is TrajectorySamplingRange.GLOBAL:
        candidates = torch.arange(model.num_gaussians, device=model.device)
        sampled_at_frame = None
    else:
        if frame_index is None:
            raise ValueError("frame_index is required for Current frame sampling")
        frame = max(0, min(model.sequence.num_frames - 1, int(frame_index)))
        time = model.sequence.frame_to_time(frame)
        t = torch.as_tensor(time, device=model.device, dtype=model.means.dtype)

        support_start = model.time_center - lifespan_sigma * model.duration
        support_end = model.time_center + lifespan_sigma * model.duration
        in_lifespan = (t >= support_start) & (t <= support_end)

        dt = t - model.time_center
        temporal_weight = model.temporal_gate + (1.0 - model.temporal_gate) * torch.exp(
            -0.5 * (dt / model.duration).square()
        )
        current_opacity = model.opacities * temporal_weight
        candidates = torch.nonzero(
            in_lifespan & (current_opacity >= float(opacity_cutoff)),
            as_tuple=False,
        ).flatten()
        sampled_at_frame = frame

    candidate_count = int(candidates.numel())
    take = min(requested, candidate_count)
    if take == 0:
        return TrajectorySampleResult(
            indices=(),
            candidate_count=candidate_count,
            sampling_range=sampling_range,
            sampled_at_frame=sampled_at_frame,
        )

    if mode is TrajectorySamplingMode.RANDOM:
        candidates_cpu = candidates.detach().cpu()
        order = torch.randperm(
            candidate_count,
            generator=random_generator,
            device="cpu",
        )[:take]
        chosen = candidates_cpu.index_select(0, order)
    else:
        speed = torch.linalg.vector_norm(model.velocity.index_select(0, candidates), dim=-1)
        largest = mode is TrajectorySamplingMode.HIGH_SPEED
        local = torch.topk(speed, k=take, largest=largest, sorted=True).indices
        chosen = candidates.index_select(0, local).detach().cpu()

    return TrajectorySampleResult(
        indices=tuple(int(index) for index in chosen.tolist()),
        candidate_count=candidate_count,
        sampling_range=sampling_range,
        sampled_at_frame=sampled_at_frame,
    )