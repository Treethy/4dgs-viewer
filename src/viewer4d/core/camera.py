from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PinholeCamera:
    """Source-independent calibrated pinhole camera.

    Convention:
        c2w maps OpenCV/COLMAP camera coordinates into world coordinates.

        camera +X = right
        camera +Y = down
        camera +Z = forward

    K is the 3x3 intrinsic matrix associated with `width` x `height`.
    """

    c2w: np.ndarray
    K: np.ndarray
    width: int
    height: int
    name: str | None = None

    def __post_init__(self) -> None:
        c2w = np.asarray(self.c2w, dtype=np.float64)
        K = np.asarray(self.K, dtype=np.float64)

        if c2w.shape != (4, 4):
            raise ValueError(f"c2w must have shape (4, 4), got {c2w.shape}")
        if K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {K.shape}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if not np.isfinite(c2w).all():
            raise ValueError("c2w contains NaN or infinity")
        if not np.isfinite(K).all():
            raise ValueError("K contains NaN or infinity")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")

        bottom = c2w[3]
        if not np.allclose(bottom, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
            raise ValueError(f"invalid homogeneous c2w bottom row: {bottom}")

        object.__setattr__(self, "c2w", c2w.copy())
        object.__setattr__(self, "K", K.copy())

    @property
    def w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w)

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3].copy()

    @property
    def right(self) -> np.ndarray:
        return _normalized(self.c2w[:3, 0])

    @property
    def up(self) -> np.ndarray:
        # OpenCV camera +Y points down.
        return _normalized(-self.c2w[:3, 1])

    @property
    def forward(self) -> np.ndarray:
        # OpenCV camera +Z points forward.
        return _normalized(self.c2w[:3, 2])

    @property
    def fov_y(self) -> float:
        """Total vertical field of view represented by K.

        This remains valid when the principal point is slightly off-center.
        Viser itself uses a symmetric vertical FOV, so the total angular span is
        the closest source-independent representation.
        """

        fy = float(self.K[1, 1])
        cy = float(self.K[1, 2])
        top = math.atan2(cy, fy)
        bottom = math.atan2(self.height - cy, fy)
        return top + bottom

    @property
    def aspect(self) -> float:
        return self.width / self.height


def _normalized(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("camera axis has zero or invalid norm")
    return value / norm