from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from viewer4d.core.model import AnytimeGS


DEFAULT_LIFESPAN_SIGMA = 3.0


@dataclass(frozen=True, slots=True)
class GaussianTrajectory:
    """Source-independent trajectory of one Gaussian center.

    ``times`` uses AnytimeGS normalized time in [0, 1]. ``positions`` stores the
    matching 3D center for every time sample. FreeTimeGS only needs two samples
    because its center motion is linear, while future spline/deformation based
    importers can provide arbitrarily many samples through the same type.
    """

    gaussian_index: int
    times: np.ndarray
    positions: np.ndarray

    def __post_init__(self) -> None:
        index = int(self.gaussian_index)
        if index < 0:
            raise ValueError("gaussian_index must be non-negative")

        times = np.ascontiguousarray(self.times, dtype=np.float32).reshape(-1)
        positions = np.ascontiguousarray(self.positions, dtype=np.float32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape [T,3], got {positions.shape}"
            )
        if positions.shape[0] != times.shape[0]:
            raise ValueError("times and positions must have the same length")
        if times.size == 0:
            raise ValueError("trajectory must contain at least one time sample")
        if not np.isfinite(times).all() or not np.isfinite(positions).all():
            raise ValueError("trajectory contains NaN or infinity")
        if np.any(times < 0.0) or np.any(times > 1.0):
            raise ValueError("trajectory times must lie in [0,1]")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("trajectory times must be strictly increasing")

        object.__setattr__(self, "gaussian_index", index)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)

    @property
    def start_time(self) -> float:
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        return float(self.times[-1])

    def position_at(self, time: float) -> np.ndarray:
        """Piecewise-linearly evaluate the stored trajectory at one time."""

        value = float(time)
        if value < self.start_time - 1e-7 or value > self.end_time + 1e-7:
            raise ValueError(
                f"time {value} lies outside trajectory "
                f"[{self.start_time}, {self.end_time}]"
            )
        value = min(max(value, self.start_time), self.end_time)

        if self.times.size == 1:
            return self.positions[0].copy()

        right = int(np.searchsorted(self.times, value, side="left"))
        if right <= 0:
            return self.positions[0].copy()
        if right >= self.times.size:
            return self.positions[-1].copy()
        if abs(float(self.times[right]) - value) <= 1e-7:
            return self.positions[right].copy()

        left = right - 1
        t0 = float(self.times[left])
        t1 = float(self.times[right])
        weight = (value - t0) / max(t1 - t0, 1e-12)
        return np.ascontiguousarray(
            (1.0 - weight) * self.positions[left] + weight * self.positions[right],
            dtype=np.float32,
        )

    def clip(self, start_time: float, end_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Return samples inside the requested time window with exact boundaries.

        This method is what lets the viewer progressively reveal a trajectory as
        the timeline advances. Boundary positions are interpolated from the
        stored samples, so a two-sample linear FreeTimeGS trajectory and a dense
        nonlinear trajectory use exactly the same display code.
        """

        start = max(float(start_time), self.start_time)
        end = min(float(end_time), self.end_time)
        if end < start:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        if abs(end - start) <= 1e-8:
            position = self.position_at(start)
            return (
                np.asarray([start], dtype=np.float32),
                position.reshape(1, 3),
            )

        interior = (self.times > start + 1e-7) & (self.times < end - 1e-7)
        middle_times = self.times[interior]
        middle_positions = self.positions[interior]

        times = np.concatenate(
            (
                np.asarray([start], dtype=np.float32),
                middle_times,
                np.asarray([end], dtype=np.float32),
            )
        )
        positions = np.concatenate(
            (
                self.position_at(start).reshape(1, 3),
                middle_positions,
                self.position_at(end).reshape(1, 3),
            ),
            axis=0,
        )
        return (
            np.ascontiguousarray(times, dtype=np.float32),
            np.ascontiguousarray(positions, dtype=np.float32),
        )


@torch.inference_mode()
def build_anytimegs_trajectories(
    model: AnytimeGS,
    indices: Iterable[int],
    *,
    lifespan_sigma: float = DEFAULT_LIFESPAN_SIGMA,
) -> dict[int, GaussianTrajectory]:
    """Build explicit center trajectories for the current AnytimeGS schema.

    The present AnytimeGS schema stores explicit linear velocity, so each
    trajectory only needs its lifespan start/end samples. The returned data type
    deliberately does not encode that assumption and can later be produced by
    nonlinear model adapters as well.
    """

    if lifespan_sigma <= 0.0:
        raise ValueError("lifespan_sigma must be positive")

    index_tuple = tuple(dict.fromkeys(int(index) for index in indices))
    if not index_tuple:
        return {}
    if min(index_tuple) < 0 or max(index_tuple) >= model.num_gaussians:
        raise IndexError("Gaussian index is outside the model range")

    selected = torch.as_tensor(index_tuple, device=model.device, dtype=torch.long)
    means = model.means.index_select(0, selected).detach().float().cpu().numpy()
    centers = model.time_center.index_select(0, selected).detach().float().cpu().numpy()
    duration = model.duration.index_select(0, selected).detach().float().cpu().numpy()
    velocity = model.velocity.index_select(0, selected).detach().float().cpu().numpy()

    trajectories: dict[int, GaussianTrajectory] = {}
    for local, index in enumerate(index_tuple):
        raw_start = float(centers[local] - lifespan_sigma * duration[local])
        raw_end = float(centers[local] + lifespan_sigma * duration[local])
        if raw_end < 0.0 or raw_start > 1.0:
            continue

        start = max(0.0, raw_start)
        end = min(1.0, raw_end)
        if end < start:
            continue

        center = float(centers[local])
        base = means[local]
        vel = velocity[local]

        if abs(end - start) <= 1e-8:
            times = np.asarray([start], dtype=np.float32)
            positions = (base + (start - center) * vel).reshape(1, 3)
        else:
            times = np.asarray([start, end], dtype=np.float32)
            positions = np.stack(
                (
                    base + (start - center) * vel,
                    base + (end - center) * vel,
                ),
                axis=0,
            )

        trajectories[index] = GaussianTrajectory(
            gaussian_index=index,
            times=times,
            positions=positions,
        )

    return trajectories