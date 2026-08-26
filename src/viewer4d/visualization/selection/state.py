from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator


class SelectionMode(str, Enum):
    """Mouse interaction mode used by the Selection tab."""

    CAMERA = "Camera"
    SINGLE = "Single"
    BOX = "Box"


class GaussianSet:
    """Ordered set of Gaussian indices.

    The selection subsystem stores Gaussian identity only. Gaussian parameters
    always stay in :class:`AnytimeGS` and are looked up by index when needed.
    """

    def __init__(self, indices: Iterable[int] = ()) -> None:
        self._indices: dict[int, None] = {}
        self.replace(indices)

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(self._indices.keys())

    def replace(self, indices: Iterable[int]) -> None:
        self._indices.clear()
        self.update(indices)

    def update(self, indices: Iterable[int]) -> None:
        for index in indices:
            value = int(index)
            if value < 0:
                raise ValueError(f"Gaussian index must be non-negative, got {value}")
            self._indices[value] = None

    def add(self, index: int) -> None:
        self.update((index,))

    def remove(self, index: int) -> None:
        self._indices.pop(int(index), None)

    def clear(self) -> None:
        self._indices.clear()

    def __len__(self) -> int:
        return len(self._indices)

    def __contains__(self, index: object) -> bool:
        if not isinstance(index, int):
            return False
        return index in self._indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __bool__(self) -> bool:
        return bool(self._indices)


@dataclass(slots=True)
class SelectionState:
    """Mutable per-client Selection state."""

    mode: SelectionMode = SelectionMode.CAMERA
    selected: GaussianSet = field(default_factory=GaussianSet)
    highlight_enabled: bool = True
    selection_source: SelectionMode | None = None

    def set_mode(self, mode: SelectionMode | str) -> None:
        self.mode = mode if isinstance(mode, SelectionMode) else SelectionMode(mode)

    def replace(self, indices: Iterable[int], *, source: SelectionMode) -> None:
        self.selected.replace(indices)
        self.selection_source = source if self.selected else None

    def clear(self) -> None:
        self.selected.clear()
        self.selection_source = None
