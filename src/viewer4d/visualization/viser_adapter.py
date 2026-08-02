from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from viewer4d.core.model import GaussianFrame

_SH_C0 = 0.28209479177387814


@dataclass(frozen=True, slots=True)
class ViserGaussianData:
    centers: np.ndarray
    covariances: np.ndarray
    rgbs: np.ndarray
    opacities: np.ndarray

    def validate(self) -> None:
        n = self.centers.shape[0]
        expected = {
            "centers": (n, 3),
            "covariances": (n, 3, 3),
            "rgbs": (n, 3),
            "opacities": (n, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")


def gaussian_frame_to_viser(
    frame: GaussianFrame,
    *,
    opacity_threshold: float = 0.0,
    max_gaussians: int | None = None,
) -> ViserGaussianData:
    """Convert one static Gaussian frame to Viser's browser-rendered format.

    Viser accepts fixed RGB rather than spherical harmonics, so RGB is derived
    from the degree-zero SH coefficient. Camera movement then stays entirely in
    the browser and does not trigger server-side rasterization.
    """

    frame.validate()
    if not 0.0 <= opacity_threshold <= 1.0:
        raise ValueError("opacity_threshold must be in [0, 1]")
    if max_gaussians is not None and max_gaussians <= 0:
        raise ValueError("max_gaussians must be positive")

    centers = frame.means.detach().float()
    scales = frame.scales.detach().float()
    quats = frame.quats.detach().float()
    opacities = frame.opacities.detach().float()
    dc = frame.sh_coeffs[:, 0, :].detach().float()

    valid = (
        torch.isfinite(centers).all(dim=1)
        & torch.isfinite(scales).all(dim=1)
        & torch.isfinite(quats).all(dim=1)
        & torch.isfinite(opacities)
        & torch.isfinite(dc).all(dim=1)
        & (scales > 0).all(dim=1)
        & (opacities >= opacity_threshold)
    )
    indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        raise ValueError("no valid Gaussians remain after filtering")

    if max_gaussians is not None and indices.numel() > max_gaussians:
        candidate_opacities = opacities[indices]
        top = torch.topk(candidate_opacities, k=max_gaussians, sorted=False).indices
        indices = indices[top]

    centers = centers[indices]
    scales = scales[indices]
    quats = quats[indices]
    opacities = opacities[indices].clamp(0.0, 1.0)
    dc = dc[indices]

    rotations = quaternion_wxyz_to_matrix_torch(quats)
    variances = scales.square()
    covariances = torch.matmul(
        rotations * variances.unsqueeze(-2),
        rotations.transpose(-1, -2),
    )
    rgbs = (0.5 + _SH_C0 * dc).clamp(0.0, 1.0)

    result = ViserGaussianData(
        centers=_to_numpy(centers),
        covariances=_to_numpy(covariances),
        rgbs=_to_numpy(rgbs),
        opacities=_to_numpy(opacities[:, None]),
    )
    result.validate()
    return result


def quaternion_wxyz_to_matrix_torch(quats: torch.Tensor) -> torch.Tensor:
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"quats must have shape [N, 4], got {tuple(quats.shape)}")
    quats = quats / torch.linalg.vector_norm(quats, dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quats.unbind(dim=1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=1,
    ).reshape(-1, 3, 3)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32, copy=False))
