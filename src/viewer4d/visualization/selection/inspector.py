from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from viewer4d.core.model import AnytimeGS


@dataclass(frozen=True, slots=True)
class GaussianSubsetSnapshot:
    indices: Tensor
    means: Tensor
    opacities: Tensor


@dataclass(frozen=True, slots=True)
class GaussianInspection:
    index: int
    normalized_time: float
    base_center: tuple[float, float, float]
    current_center: tuple[float, float, float]
    velocity: tuple[float, float, float]
    speed: float
    time_center: float
    duration: float
    support_start_3sigma: float
    support_end_3sigma: float
    scale: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    base_opacity: float
    current_opacity: float
    temporal_gate: float


def evaluate_gaussian_subset(
    model: AnytimeGS,
    indices: Iterable[int] | Tensor,
    normalized_time: float,
) -> GaussianSubsetSnapshot:
    """Evaluate centers and temporal opacities for only the requested indices."""

    time = float(normalized_time)
    if not 0.0 <= time <= 1.0:
        raise ValueError(f"normalized_time must be in [0,1], got {time}")

    if isinstance(indices, Tensor):
        selected = indices.to(device=model.device, dtype=torch.long)
    else:
        selected = torch.as_tensor(
            tuple(int(index) for index in indices),
            device=model.device,
            dtype=torch.long,
        )

    if selected.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape {tuple(selected.shape)}")
    if selected.numel() == 0:
        empty_means = model.means.new_empty((0, 3))
        empty_opacity = model.opacities.new_empty((0,))
        return GaussianSubsetSnapshot(selected, empty_means, empty_opacity)

    if torch.any(selected < 0) or torch.any(selected >= model.num_gaussians):
        raise IndexError("Gaussian index is outside the model range")

    means0 = model.means.index_select(0, selected)
    centers = model.time_center.index_select(0, selected)
    velocity = model.velocity.index_select(0, selected)
    duration = model.duration.index_select(0, selected)
    base_opacity = model.opacities.index_select(0, selected)
    temporal_gate = model.temporal_gate.index_select(0, selected)

    t = torch.as_tensor(time, dtype=means0.dtype, device=model.device)
    dt = t - centers
    means = means0 + dt[:, None] * velocity
    temporal_weight = temporal_gate + (1.0 - temporal_gate) * torch.exp(
        -0.5 * (dt / duration).square()
    )
    opacities = base_opacity * temporal_weight

    return GaussianSubsetSnapshot(selected, means, opacities)


def inspect_gaussian(
    model: AnytimeGS,
    index: int,
    normalized_time: float,
) -> GaussianInspection:
    """Read one Gaussian's model parameters and current center/opacity."""

    index = int(index)
    if not 0 <= index < model.num_gaussians:
        raise IndexError(
            f"Gaussian index must be in [0,{model.num_gaussians - 1}], got {index}"
        )

    snapshot = evaluate_gaussian_subset(model, (index,), normalized_time)

    base_center = _tuple3(model.means[index])
    current_center = _tuple3(snapshot.means[0])
    velocity = _tuple3(model.velocity[index])
    speed = float(torch.linalg.vector_norm(model.velocity[index]).item())
    time_center = float(model.time_center[index].item())
    duration = float(model.duration[index].item())

    return GaussianInspection(
        index=index,
        normalized_time=float(normalized_time),
        base_center=base_center,
        current_center=current_center,
        velocity=velocity,
        speed=speed,
        time_center=time_center,
        duration=duration,
        support_start_3sigma=max(0.0, time_center - 3.0 * duration),
        support_end_3sigma=min(1.0, time_center + 3.0 * duration),
        scale=_tuple3(model.scales[index]),
        quaternion_wxyz=_tuple4(model.quats[index]),
        base_opacity=float(model.opacities[index].item()),
        current_opacity=float(snapshot.opacities[0].item()),
        temporal_gate=float(model.temporal_gate[index].item()),
    )


def _tuple3(value: Tensor) -> tuple[float, float, float]:
    array = value.detach().float().cpu().numpy().reshape(3)
    return tuple(float(x) for x in array)  # type: ignore[return-value]


def _tuple4(value: Tensor) -> tuple[float, float, float, float]:
    array = value.detach().float().cpu().numpy().reshape(4)
    return tuple(float(x) for x in array)  # type: ignore[return-value]
