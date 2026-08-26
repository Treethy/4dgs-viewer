from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from viewer4d.visualization.selection.state import GaussianSet


class TrajectorySamplingMode(str, Enum):
    HIGH_SPEED = "High speed"
    LOW_SPEED = "Low speed"
    RANDOM = "Random"


class TrajectorySamplingRange(str, Enum):
    CURRENT_FRAME = "Current frame"
    GLOBAL = "Global"


@dataclass(slots=True)
class TrajectoryState:
    """Per-client trajectory selection and display state."""

    sampled: GaussianSet = field(default_factory=GaussianSet)
    manual: GaussianSet = field(default_factory=GaussianSet)
    include_sampled: bool = True
    include_manual: bool = False
    start_tracking_frame: int = 0
    show_trajectories: bool = True
    show_current_centers: bool = True

    sampled_range: TrajectorySamplingRange | None = None
    sampled_at_frame: int | None = None
    sampled_candidate_count: int = 0

    def tracking_indices(self) -> tuple[int, ...]:
        """Return an ordered union of all currently enabled sources."""

        merged = GaussianSet()
        if self.include_sampled:
            merged.update(self.sampled)
        if self.include_manual:
            merged.update(self.manual)
        return merged.indices